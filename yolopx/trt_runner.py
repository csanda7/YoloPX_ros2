from __future__ import annotations
import threading
import atexit
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
import tensorrt as trt

from .cuda_utils import (
    DevicePtr,
    CudaStream,
    CudaEvent,
    cuda_stream_create,
    cuda_stream_destroy,
    cuda_event_create,
    cuda_event_destroy,
    cuda_event_record,
    cuda_event_elapsed_time,
    cuda_malloc,
    cuda_free,
    cuda_memcpy_htod_async,
    cuda_memcpy_dtoh_async,
    pinned_empty_1d,
    cuda_stream_sync,
)

# --- Helper a TRT dtype -> numpy dtype ---

def trt_nptype(dtype: trt.DataType) -> np.dtype:
    mapping = {
        trt.DataType.FLOAT: np.float32,
        trt.DataType.HALF: np.float16,
        trt.DataType.INT8: np.int8,
        trt.DataType.INT32: np.int32,
        trt.DataType.BOOL: np.bool_,
    }
    if hasattr(trt.DataType, "UINT8"):
        mapping[getattr(trt.DataType, "UINT8")] = np.uint8
    return np.dtype(mapping[dtype])


"""Globális (folyamat-szintű) TRT logger/runtime a konstrukciós overhead csökkentésére."""
_TRT_GLOBAL_LOGGER = trt.Logger(trt.Logger.WARNING)
_TRT_GLOBAL_RUNTIME = trt.Runtime(_TRT_GLOBAL_LOGGER)


class TRT10Runner:
    def __init__(self, engine_path: str, default_shape: Tuple[int, int, int, int]):
        self._infer_lock = threading.Lock()

        # Megosztott logger + runtime (nem hozzuk létre minden runnernél)
        self.logger = _TRT_GLOBAL_LOGGER
        self.runtime = _TRT_GLOBAL_RUNTIME

        with open(engine_path, "rb") as f:
            engine_bytes = f.read()
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError("Nem sikerült deszerializálni a TensorRT engine-t")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Nem sikerült létrehozni az execution contextet")

        self.input_names: List[str] = []
        self.output_names: List[str] = []
        for i in range(int(self.engine.num_io_tensors)):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)
        if len(self.input_names) != 1:
            raise RuntimeError(f"Pontosan 1 inputot várunk, most: {self.input_names}")

        in_name = self.input_names[0]
        in_shape = list(self.engine.get_tensor_shape(in_name))
        # Statikus shape elvárás: nincs -1 dim. (Ha -1 lenne, hibát dobunk a felesleges runtime shape beállítás elkerülésére.)
        if any(d < 0 for d in in_shape):
            raise RuntimeError(
                "A betöltött engine dinamikus input dimenziót tartalmaz (-1). "
                "Kérlek exportálj / buildelj statikus (fix shape) TensorRT engine-t a jobb teljesítmény érdekében."
            )

        self.stream: CudaStream = cuda_stream_create()
        self._ev_start: CudaEvent = cuda_event_create()
        self._ev_stop: CudaEvent = cuda_event_create()

        self.host: Dict[str, np.ndarray] = {}
        self._host_pinned_ptrs: Dict[str, int] = {}
        self.dev: Dict[str, DevicePtr] = {}
        self.shapes: Dict[str, Tuple[int, ...]] = {}

        for name in self.input_names + self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            if any(d < 0 for d in shape):
                raise RuntimeError(f"Dinamikus shape nincs feloldva a(z) {name} tensoron: {shape}")
            self.shapes[name] = shape
            dtype = trt_nptype(self.engine.get_tensor_dtype(name))
            nelems = int(np.prod(shape))
            arr, ptr = pinned_empty_1d(nelems, dtype)
            self.host[name] = arr
            self._host_pinned_ptrs[name] = ptr
            self.dev[name] = cuda_malloc(self.host[name].nbytes)

        for name in self.input_names + self.output_names:
            ok = self.context.set_tensor_address(name, int(self.dev[name]))
            if not ok:
                raise RuntimeError(f"set_tensor_address sikertelen: {name}")

        self.lane_name: str | None = None
        self.drive_name: str | None = None
        for n in self.output_names:
            nl = n.lower()
            if self.lane_name is None and ("lane" in nl or "lane_line" in nl):
                self.lane_name = n
            if self.drive_name is None and ("drive" in nl or "drivable" in nl or "drive_area" in nl):
                self.drive_name = n
        if self.lane_name is None or self.drive_name is None:
            if len(self.output_names) >= 2:
                self.lane_name = self.output_names[-2]
                self.drive_name = self.output_names[-1]
            else:
                raise RuntimeError("Nincs elég kimenet a lane/drivable maphez")

        # Gyors elérés meta adatok
        self._input_name = self.input_names[0]
        self._input_dtype = self.host[self._input_name].dtype
        self._input_shape = self.shapes[self._input_name]
        self._input_view = self.host[self._input_name].reshape(self._input_shape)
        self._input_is_fp16 = (self._input_dtype == np.float16)

    def close(self) -> None:
        try:
            for ptr in self.dev.values():
                cuda_free(ptr)
        except Exception:
            pass
        try:
            for p in self._host_pinned_ptrs.values():
                from .cuda_utils import cuda_host_free  # local import to avoid cycle
                cuda_host_free(p)
        except Exception:
            pass
        try:
            cuda_event_destroy(self._ev_start)
            cuda_event_destroy(self._ev_stop)
        except Exception:
            pass
        finally:
            try:
                cuda_stream_destroy(self.stream)
            except Exception:
                pass

    def get_input_view(self) -> np.ndarray:
        """Visszaadja a (N,C,H,W) alakú pinned host input nézetet közvetlen kitöltéshez."""
        return self._input_view

    def infer_timed(
        self,
        nchw_fp32: np.ndarray | None,
        copy_outputs: bool = True,
        measure_timing: bool = True,
    ) -> Tuple[Dict[str, np.ndarray], float]:
        """Futtat egy inference-et.

        nchw_fp32:
            None esetén feltételezzük, hogy a hívó fél már feltöltötte a get_input_view()-t.
            Egyébként bemásoljuk / konvertáljuk a pinned host input bufferbe.
        copy_outputs:
            True → kimenetek .copy() (izolált, biztonságos használat később is).
            False → közvetlenül a belső pinned bufferek nézete (következő infer felülírja!).
        measure_timing:
            True → CUDA event alapú gpu_ms mérés. False → kisebb overhead, gpu_ms=0.0.
        """
        with self._infer_lock:
            in_name = self._input_name
            host_in = self.host[in_name]
            dev_in = self.dev[in_name]

            if nchw_fp32 is not None:
                if tuple(nchw_fp32.shape) != self._input_shape:
                    raise ValueError(f"Input shape {nchw_fp32.shape} != várt {self._input_shape}")
                # FP16 engine esetén automatikus konverzió (pontosság a háló szintjén változatlan)
                if self._input_is_fp16:
                    if nchw_fp32.dtype != np.float16:
                        np.asarray(nchw_fp32, dtype=np.float16, order="C", out=host_in.reshape(self._input_shape))
                    else:
                        host_in[...] = nchw_fp32.ravel()
                else:
                    if nchw_fp32.dtype != np.float32:
                        raise TypeError("Input dtype legyen float32")
                    host_in[...] = nchw_fp32.ravel()

            # H2D másolás
            cuda_memcpy_htod_async(dev_in, host_in, self.stream)

            if measure_timing:
                cuda_event_record(self._ev_start, self.stream)

            ok = self.context.execute_async_v3(stream_handle=int(self.stream))
            if not ok:
                raise RuntimeError("execute_async_v3 sikertelen")

            if measure_timing:
                cuda_event_record(self._ev_stop, self.stream)

            # Kimenetek device→host
            for name in self.output_names:
                cuda_memcpy_dtoh_async(self.host[name], self.dev[name], self.stream)

            cuda_stream_sync(self.stream)

            if measure_timing:
                gpu_ms = cuda_event_elapsed_time(self._ev_start, self._ev_stop)
            else:
                gpu_ms = 0.0

            outs: Dict[str, np.ndarray] = {}
            if copy_outputs:
                for name in self.output_names:
                    outs[name] = self.host[name].reshape(self.shapes[name]).copy()
            else:
                for name in self.output_names:
                    outs[name] = self.host[name].reshape(self.shapes[name])
            return outs, float(gpu_ms)


_TRT_CACHE_LOCK = threading.Lock()
_TRT_CACHE: Dict[Tuple[str, Tuple[int,int,int,int]], TRT10Runner] = {}

def get_trt_runner(engine_path: str, default_shape: Tuple[int,int,int,int]) -> TRT10Runner:
    key = (str(Path(engine_path).resolve()), tuple(default_shape))
    with _TRT_CACHE_LOCK:
        r = _TRT_CACHE.get(key)
        if r is None:
            print(f"[TRT] Creating runtime/engine/context ONCE for {key[0]}")
            r = TRT10Runner(key[0], default_shape)
            _TRT_CACHE[key] = r
        return r


def _cleanup_trt_cache():
    for r in list(_TRT_CACHE.values()):
        try:
            r.close()
        except Exception:
            pass
    _TRT_CACHE.clear()

atexit.register(_cleanup_trt_cache)


# =========================== Pipeline (multi-slot) ===========================

class TRTPipelineSlot:
    """Egy pipeline slot: saját (pinned) input + device input + output bufferek + event.

    Kiegészítve frame és t_enqueue meta adatokkal az end-to-end késleltetés méréséhez."""
    __slots__ = ("idx", "in_host", "in_dev", "out_host", "out_dev", "event", "busy", "gpu_ms", "frame", "t_enqueue")

    def __init__(self, idx: int, in_host: np.ndarray, in_dev: DevicePtr,
                 out_host: Dict[str, np.ndarray], out_dev: Dict[str, DevicePtr], event: CudaEvent):
        self.idx = idx
        self.in_host = in_host
        self.in_dev = in_dev
        self.out_host = out_host
        self.out_dev = out_dev
        self.event = event
        self.busy = False
        self.gpu_ms = 0.0
        self.frame = None
        self.t_enqueue = 0.0


class TRTPipelineRunner:
    """Több párhuzamos slot (context + stream) a H2D / compute / D2H átfedéshez.

    Használat:
        pipe = TRTPipelineRunner(base_runner, num_slots=2)
        slot = pipe.acquire_slot(); if not slot: (drop frame)
        # töltsd be a slot.in_host reshaped view-ját vagy add át egy (N,C,H,W) tömböt enqueue-nak
        pipe.enqueue(slot, nchw, measure_timing=False)
        később poll/fetch: outs = pipe.fetch_ready(copy_outputs=False)
    """

    def __init__(self, base: TRT10Runner, num_slots: int = 3):
        from .cuda_utils import (
            pinned_empty_1d, cuda_malloc, cuda_stream_create, cuda_event_create,
            cuda_memcpy_htod_async, cuda_memcpy_dtoh_async, cuda_stream_sync,
            cuda_event_record, cuda_event_elapsed_time, cuda_event_destroy,
            cuda_stream_destroy
        )  # local import to avoid circular concerns
        self._cu = {
            'pinned_empty_1d': pinned_empty_1d,
            'cuda_malloc': cuda_malloc,
            'cuda_stream_create': cuda_stream_create,
            'cuda_event_create': cuda_event_create,
            'cuda_memcpy_htod_async': cuda_memcpy_htod_async,
            'cuda_memcpy_dtoh_async': cuda_memcpy_dtoh_async,
            'cuda_stream_sync': cuda_stream_sync,
            'cuda_event_record': cuda_event_record,
            'cuda_event_elapsed_time': cuda_event_elapsed_time,
            'cuda_event_destroy': cuda_event_destroy,
            'cuda_stream_destroy': cuda_stream_destroy,
        }
        self.base = base
        self.num_slots = max(1, int(num_slots))
        self._input_shape = base._input_shape
        self._input_is_fp16 = base._input_is_fp16
        self._input_dtype = base._input_dtype
        self._in_name = base._input_name
        self.output_names = list(base.output_names)
        # Csak a szükséges kimeneteket hozzuk vissza hostra (D2H): lane + drive (ha nincs mask mód)
        self.fetch_names = []  # type: List[str]
        if base.lane_name:
            self.fetch_names.append(base.lane_name)
        if base.drive_name and base.drive_name not in self.fetch_names:
            self.fetch_names.append(base.drive_name)
        if not self.fetch_names:
            # Fallback: ha nem találtuk meg a neveket, maradjon az összes
            self.fetch_names = list(self.output_names)

        # Opcionális eszköz oldali threshold (maszkok D2H, nem a teljes float térképek)
        self._dev_thr_enabled = False
        self._thr_funcs = None  # type: Tuple[object, object] | None
        self._drv = None
        self._mask_keys: Dict[str, str] = {}
        self._map_info: Dict[str, Tuple[int, int, bool]] = {}  # name -> (offset_elems, n_pix, is_fp16)
        if base.lane_name:
            self._mask_keys[base.lane_name] = base.lane_name + ":mask"
        if base.drive_name:
            self._mask_keys[base.drive_name] = base.drive_name + ":mask"

        # Előkészítjük a csatorna indexet és pixelek számát (NCHW feltételezve)
        def _map_offset_npix(name: str) -> Tuple[int, int, bool]:
            shp = self.base.shapes[name]
            is_fp16 = (self.base.host[name].dtype == np.float16)
            if len(shp) == 4:
                n, c, h, w = shp
                ch = 1 if c and c > 1 else 0
                return ch * int(h) * int(w), int(h) * int(w), is_fp16
            if len(shp) == 3:
                c, h, w = shp
                ch = 1 if c and c > 1 else 0
                return ch * int(h) * int(w), int(h) * int(w), is_fp16
            if len(shp) == 2:
                h, w = shp
                return 0, int(h) * int(w), is_fp16
            raise RuntimeError(f"Váratlan output shape a device thresholdhöz: {shp}")
        try:
            if base.lane_name and base.drive_name:
                self._map_info[base.lane_name] = _map_offset_npix(base.lane_name)
                self._map_info[base.drive_name] = _map_offset_npix(base.drive_name)
        except Exception:
            self._map_info = {}

        # NVRTC fordítás (ha elérhető)
        try:
            import importlib as _imp
            _nvrtc = _imp.import_module('cuda.nvrtc')
            _drv = _imp.import_module('cuda.cuda')
            code_f32 = (
                'extern "C" __global__ void thr2_f32(const float* lane, const float* drive, unsigned char* lm, unsigned char* dm, int n, float lt, float dt) {'
                ' int i = blockIdx.x * blockDim.x + threadIdx.x;'
                ' if (i < n) {'
                '   unsigned char l = lane[i] > lt;'
                '   unsigned char d = drive[i] > dt;'
                '   if (l) d = 0;'
                '   lm[i] = l; dm[i] = d;'
                ' } }'
            )
            code_f16 = (
                '#include <cuda_fp16.h>\n'
                'extern "C" __global__ void thr2_f16(const __half* lane, const __half* drive, unsigned char* lm, unsigned char* dm, int n, float lt, float dt) {'
                ' int i = blockIdx.x * blockDim.x + threadIdx.x;'
                ' if (i < n) {'
                '   float lf = __half2float(lane[i]);'
                '   float df = __half2float(drive[i]);'
                '   unsigned char l = lf > lt;'
                '   unsigned char d = df > dt;'
                '   if (l) d = 0;'
                '   lm[i] = l; dm[i] = d;'
                ' } }'
            )
            def _compile(name: str, src: str):
                prog = _nvrtc.nvrtcCreateProgram(src.encode('utf-8'), (name+".cu").encode('utf-8'), 0, [], [])[1]
                opts = [b"--std=c++11"]
                res = _nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
                if res[0] != 0:
                    log = _nvrtc.nvrtcGetProgramLog(prog)[1].decode('utf-8', 'ignore')
                    _nvrtc.nvrtcDestroyProgram(prog)
                    raise RuntimeError(f"NVRTC compile failed for {name}:\n{log}")
                ptx = _nvrtc.nvrtcGetPTX(prog)[1]
                _nvrtc.nvrtcDestroyProgram(prog)
                return ptx
            ptx32 = _compile("thr2_f32", code_f32)
            ptx16 = _compile("thr2_f16", code_f16)
            # Driver init
            _drv.cuInit(0)
            mod32 = _drv.cuModuleLoadData(ptx32)[1]
            mod16 = _drv.cuModuleLoadData(ptx16)[1]
            fn32 = _drv.cuModuleGetFunction(mod32, b"thr2_f32")[1]
            fn16 = _drv.cuModuleGetFunction(mod16, b"thr2_f16")[1]
            self._thr_funcs = (fn32, fn16)
            self._drv = _drv
            # Ha minden meta adott, engedélyezzük
            if self._map_info:
                self._dev_thr_enabled = True
        except Exception:
            self._dev_thr_enabled = False

        # Threshold értékek (Node-ból állítható)
        self._thr_lane = 0.5
        self._thr_drive = 0.5

        # Ha eszköz oldali threshold aktív, ne kérjünk vissza float térképeket
        if self._dev_thr_enabled:
            self.fetch_names = []
        self.slots: list[TRTPipelineSlot] = []
        self._streams: list[CudaStream] = []
        self._ctxs: list[trt.IExecutionContext] = []
        self._events_start: list[CudaEvent] = []
        self._events_stop: list[CudaEvent] = []

        for i in range(self.num_slots):
            ctx = base.engine.create_execution_context()
            if ctx is None:
                raise RuntimeError("Nem sikerült contextet létrehozni pipeline slothoz")

            # Input buffer
            nelems_in = int(np.prod(self._input_shape))
            in_host, _ = self._cu['pinned_empty_1d'](nelems_in, self._input_dtype)
            in_dev = self._cu['cuda_malloc'](in_host.nbytes)

            # Output bufferek slotonként
            out_host: Dict[str, np.ndarray] = {}
            out_dev: Dict[str, DevicePtr] = {}
            for name in self.output_names:
                shape = self.base.shapes[name]
                nelems = int(np.prod(shape))
                h_arr, _ = self._cu['pinned_empty_1d'](nelems, self.base.host[name].dtype)
                d_ptr = self._cu['cuda_malloc'](h_arr.nbytes)
                out_host[name] = h_arr
                out_dev[name] = d_ptr
                if not ctx.set_tensor_address(name, int(d_ptr)):
                    raise RuntimeError(f"set_tensor_address output fail: {name}")
            if not ctx.set_tensor_address(self._in_name, int(in_dev)):
                raise RuntimeError("set_tensor_address input fail")

            stream = self._cu['cuda_stream_create']()
            ev_start = self._cu['cuda_event_create']()
            ev_stop = self._cu['cuda_event_create']()
            slot_event_done = self._cu['cuda_event_create']()

            slot = TRTPipelineSlot(i, in_host, in_dev, out_host, out_dev, slot_event_done)
            # Opcionális mask bufferek
            if self._dev_thr_enabled and base.lane_name and base.drive_name:
                try:
                    _, n_pix, _ = self._map_info[base.lane_name]
                    ml_h, _ = self._cu['pinned_empty_1d'](n_pix, np.uint8)
                    md_h, _ = self._cu['pinned_empty_1d'](n_pix, np.uint8)
                    ml_d = self._cu['cuda_malloc'](n_pix)
                    md_d = self._cu['cuda_malloc'](n_pix)
                    slot.out_host[self._mask_keys[base.lane_name]] = ml_h
                    slot.out_dev[self._mask_keys[base.lane_name]] = ml_d
                    slot.out_host[self._mask_keys[base.drive_name]] = md_h
                    slot.out_dev[self._mask_keys[base.drive_name]] = md_d
                except Exception:
                    pass
            self.slots.append(slot)
            self._streams.append(stream)
            self._ctxs.append(ctx)
            self._events_start.append(ev_start)
            self._events_stop.append(ev_stop)

    # --- API ---
    def acquire_slot(self) -> TRTPipelineSlot | None:
        for s in self.slots:
            if not s.busy:
                return s
        return None

    def enqueue(self, slot: TRTPipelineSlot, nchw: np.ndarray, measure_timing: bool = True) -> None:
        if slot.busy:
            raise RuntimeError("Slot already busy")
        if tuple(nchw.shape) != self._input_shape:
            raise ValueError("Shape mismatch")
        # Host kitöltés (elkerüljük a felesleges másolatot, ha már a slot buffer nézete jön)
        slot_view = slot.in_host.reshape(self._input_shape)
        if np.shares_memory(nchw, slot_view):
            pass  # már a megfelelő helyen vannak az adatok
        else:
            if self._input_is_fp16 and nchw.dtype != np.float16:
                np.asarray(nchw, dtype=np.float16, order="C", out=slot_view)
            else:
                if nchw.dtype != self._input_dtype:
                    raise TypeError("Input dtype mismatch")
                slot_view[...] = nchw
        stream = self._streams[slot.idx]
        ctx = self._ctxs[slot.idx]
        # enqueue időbélyeg
        import time as _t
        slot.t_enqueue = _t.perf_counter()
        self._cu['cuda_memcpy_htod_async'](slot.in_dev, slot.in_host, stream)
        if measure_timing:
            self._cu['cuda_event_record'](self._events_start[slot.idx], stream)
        ok = ctx.execute_async_v3(stream_handle=int(stream))
        if not ok:
            raise RuntimeError("execute_async_v3 fail")
        if self._dev_thr_enabled and self.base.lane_name and self.base.drive_name and self._thr_funcs:
            try:
                lane = self.base.lane_name; drive = self.base.drive_name
                off_l, n_pix, is_f16 = self._map_info[lane]
                off_d, _, _ = self._map_info[drive]
                esz = 2 if is_f16 else 4
                lane_ptr = int(slot.out_dev[lane]) + off_l * esz
                drive_ptr = int(slot.out_dev[drive]) + off_d * esz
                ml_d = int(slot.out_dev[self._mask_keys[lane]])
                md_d = int(slot.out_dev[self._mask_keys[drive]])
                # thresholds Node-ból állítva
                lt = float(self._thr_lane)
                dt = float(self._thr_drive)
                fn = self._thr_funcs[1] if is_f16 else self._thr_funcs[0]
                import ctypes as _ct
                args = (
                    _ct.c_void_p(lane_ptr), _ct.c_void_p(drive_ptr),
                    _ct.c_void_p(ml_d), _ct.c_void_p(md_d),
                    _ct.c_int(n_pix), _ct.c_float(lt), _ct.c_float(dt)
                )
                arg_ptrs = (_ct.c_void_p * len(args))(*[ _ct.cast(_ct.pointer(a), _ct.c_void_p) for a in args ])
                block = 256
                grid = (n_pix + block - 1) // block
                self._drv.cuLaunchKernel(
                    fn, grid, 1, 1, block, 1, 1, 0,
                    int(stream), arg_ptrs, None
                )
                # D2H maskok
                self._cu['cuda_memcpy_dtoh_async'](slot.out_host[self._mask_keys[lane]], ml_d, stream)
                self._cu['cuda_memcpy_dtoh_async'](slot.out_host[self._mask_keys[drive]], md_d, stream)
            except Exception:
                # Fallback: eredeti térképek D2H
                for name in self.fetch_names:
                    self._cu['cuda_memcpy_dtoh_async'](slot.out_host[name], slot.out_dev[name], stream)
        else:
            for name in self.fetch_names:
                self._cu['cuda_memcpy_dtoh_async'](slot.out_host[name], slot.out_dev[name], stream)
        if measure_timing:
            self._cu['cuda_event_record'](self._events_stop[slot.idx], stream)
        # Done event
        self._cu['cuda_event_record'](slot.event, stream)
        slot.busy = True
        slot.gpu_ms = -1.0 if measure_timing else 0.0

    def fetch_ready(self, copy_outputs: bool = True) -> list[tuple[TRTPipelineSlot, Dict[str, np.ndarray], float]]:
        """Visszaadja az összes elkészült slot eredményét és felszabadítja azokat."""
        ready: list[tuple[TRTPipelineSlot, Dict[str, np.ndarray], float]] = []
        for slot in self.slots:
            if not slot.busy:
                continue
            # Non-blocking query
            from .cuda_utils import cuda_event_query
            if not cuda_event_query(slot.event):
                continue
            # A done event a D2H másolatok után lett beütemezve; ha kész, a host bufferek konzisztensen olvashatók, nincs szükség stream szinkronra.
            if slot.gpu_ms < 0:
                slot.gpu_ms = self._cu['cuda_event_elapsed_time'](self._events_start[slot.idx], self._events_stop[slot.idx])
            outs: Dict[str, np.ndarray] = {}
            if self._dev_thr_enabled and self.base.lane_name and self.base.drive_name:
                # Csak maszkokat adjunk vissza, az eredeti kimenetneveken (kompatibilis a jelenlegi postprocess-szel)
                h, w = self.base.shapes[self.base.lane_name][-2:]
                lk = self._mask_keys[self.base.lane_name]
                dk = self._mask_keys[self.base.drive_name]
                if copy_outputs:
                    if lk in slot.out_host:
                        outs[self.base.lane_name] = slot.out_host[lk].reshape(h, w).copy()
                    if dk in slot.out_host:
                        outs[self.base.drive_name] = slot.out_host[dk].reshape(h, w).copy()
                else:
                    if lk in slot.out_host:
                        outs[self.base.lane_name] = slot.out_host[lk].reshape(h, w)
                    if dk in slot.out_host:
                        outs[self.base.drive_name] = slot.out_host[dk].reshape(h, w)
            else:
                if copy_outputs:
                    for n in self.fetch_names:
                        shape = self.base.shapes[n]
                        outs[n] = slot.out_host[n].reshape(shape).copy()
                else:
                    for n in self.fetch_names:
                        outs[n] = slot.out_host[n].reshape(self.base.shapes[n])
            gpu_ms = slot.gpu_ms
            slot.busy = False
            ready.append((slot, outs, gpu_ms))
        return ready

    # --- Config API ---
    def set_thresholds(self, lane: float, drive: float) -> None:
        self._thr_lane = float(lane)
        self._thr_drive = float(drive)

    def close(self):
        # Felszabadítás
        for s in self.slots:
            try:
                self._cu['cuda_free'] = cuda_free  # ensure ref
            except Exception:
                pass
        for ev in self._events_start + self._events_stop:
            try:
                self._cu['cuda_event_destroy'](ev)
            except Exception:
                pass
        for st in self._streams:
            try:
                self._cu['cuda_stream_destroy'](st)
            except Exception:
                pass
