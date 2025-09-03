#!/usr/bin/env python3
from __future__ import annotations

# ============================== Standard libs ===============================
import ctypes
import time
import json
from pathlib import Path
from typing import Dict, Tuple, List

# ============================== Third-party libs =============================
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String, Header

import tensorrt as trt
from cuda import cudart  # CUDA-Python wrapper (JetPack 6.x / CUDA 12.x)

# ============================== Saját konfig ================================
try:
    from yolopx.config import (
        ENGINE,
        TOPIC,
        RELIABLE,
        LANE_THRESH,
        DRIVE_THRESH,
        INPUT_W,
        INPUT_H,
        DEFAULT_INPUT_SHAPE,
        WATCHDOG_PERIOD_SEC,
        NO_FRAME_TIMEOUT_SEC,
        OUT_TOPIC,
        OUT_SCALE,
        FRAME_ID,
    )
except Exception:
    # Új: egységes kimeneti topic + skálázás (0.5 → szélesség/magasság felezés = ~negyed terület)
    OUT_TOPIC = "/yolopx/mask"
    OUT_SCALE = 0.25
    FRAME_ID = "camera"

# ============================== Type aliases ================================
DevicePtr = int    # CUDA device pointer intként
CudaStream = int   # CUDA stream handle intként
CudaEvent = int    # CUDA event handle intként

# ====================== CUDA helpers (tuple-first, byref-fallback) ======================

def _cuda_err_code(e) -> int:
    if isinstance(e, tuple):
        e = e[0]
    try:
        return int(e)
    except Exception:
        v = getattr(e, "value", e)
        try:
            return int(v)
        except Exception:
            return 9999


def check_cuda(err) -> None:
    code = _cuda_err_code(err)
    ok = _cuda_err_code(cudart.cudaError_t.cudaSuccess)
    if code != ok:
        raise RuntimeError(f"CUDA error: {err}")


def cuda_stream_create() -> CudaStream:
    try:
        err, stream = cudart.cudaStreamCreate()
        check_cuda(err)
        return int(stream)
    except TypeError:
        s = ctypes.c_void_p()
        check_cuda(cudart.cudaStreamCreate(ctypes.byref(s)))
        return int(s.value)


def cuda_stream_destroy(stream: CudaStream) -> None:
    if stream is None:
        return
    try:
        err = cudart.cudaStreamDestroy(int(stream))
        if isinstance(err, tuple):
            check_cuda(err[0])
        else:
            check_cuda(err)
    except Exception:
        pass


def cuda_stream_sync(stream: CudaStream) -> None:
    check_cuda(cudart.cudaStreamSynchronize(int(stream)))


def cuda_malloc(nbytes: int) -> DevicePtr:
    try:
        err, dptr = cudart.cudaMalloc(nbytes)
        check_cuda(err)
        return int(dptr)
    except TypeError:
        p = ctypes.c_void_p()
        check_cuda(cudart.cudaMalloc(ctypes.byref(p), nbytes))
        return int(p.value)


def cuda_free(dptr: DevicePtr) -> None:
    check_cuda(cudart.cudaFree(int(dptr)))


def cuda_memcpy_htod_async(dst_dev_ptr: DevicePtr, src_np: np.ndarray, stream: CudaStream) -> None:
    nbytes = src_np.nbytes
    src_addr = int(src_np.ctypes.data)
    try:
        check_cuda(
            cudart.cudaMemcpyAsync(
                int(dst_dev_ptr),
                src_addr,
                nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                int(stream),
            )
        )
    except TypeError:
        check_cuda(
            cudart.cudaMemcpyAsync(
                ctypes.c_void_p(int(dst_dev_ptr)),
                ctypes.c_void_p(src_addr),
                nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                ctypes.c_void_p(int(stream)),
            )
        )


def cuda_memcpy_dtoh_async(dst_np: np.ndarray, src_dev_ptr: DevicePtr, stream: CudaStream) -> None:
    nbytes = dst_np.nbytes
    dst_addr = int(dst_np.ctypes.data)
    try:
        check_cuda(
            cudart.cudaMemcpyAsync(
                dst_addr,
                int(src_dev_ptr),
                nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                int(stream),
            )
        )
    except TypeError:
        check_cuda(
            cudart.cudaMemcpyAsync(
                ctypes.c_void_p(dst_addr),
                ctypes.c_void_p(int(src_dev_ptr)),
                nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                ctypes.c_void_p(int(stream)),
            )
        )

# ------------------------ CUDA Event helpers (időmérés) ------------------------

def cuda_event_create() -> CudaEvent:
    try:
        err, ev = cudart.cudaEventCreate()
        check_cuda(err)
        return int(ev)
    except TypeError:
        e = ctypes.c_void_p()
        check_cuda(cudart.cudaEventCreate(ctypes.byref(e)))
        return int(e.value)


def cuda_event_destroy(event: CudaEvent) -> None:
    try:
        err = cudart.cudaEventDestroy(int(event))
        if isinstance(err, tuple):
            check_cuda(err[0])
        else:
            check_cuda(err)
    except Exception:
        pass


def cuda_event_record(event: CudaEvent, stream: CudaStream) -> None:
    try:
        check_cuda(cudart.cudaEventRecord(int(event), int(stream)))
    except TypeError:
        check_cuda(cudart.cudaEventRecord(ctypes.c_void_p(int(event)), ctypes.c_void_p(int(stream))))


def cuda_event_elapsed_time(start_event: CudaEvent, stop_event: CudaEvent) -> float:
    try:
        err, ms = cudart.cudaEventElapsedTime(int(start_event), int(stop_event))
        check_cuda(err)
        return float(ms)
    except TypeError:
        ms_out = ctypes.c_float()
        check_cuda(
            cudart.cudaEventElapsedTime(
                ctypes.byref(ms_out),
                ctypes.c_void_p(int(start_event)),
                ctypes.c_void_p(int(stop_event)),
            )
        )
        return float(ms_out.value)


# ============================== TRT helpers ===============================

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


# ============================== TensorRT Runner ==============================

class TRT10Runner:

    def __init__(self, engine_path: str, default_shape: Tuple[int, int, int, int]):
        self.logger = trt.Logger(trt.Logger.INFO)
        self.runtime = trt.Runtime(self.logger)

        with open(engine_path, "rb") as f:
            engine_bytes = f.read()
        self.engine: trt.ICudaEngine | None = self.runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError("Nem sikerült deszerializálni a TensorRT engine-t")

        self.context: trt.IExecutionContext | None = self.engine.create_execution_context()
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
        if any(d < 0 for d in in_shape):
            ok = self.context.set_input_shape(in_name, tuple(default_shape))
            if not ok:
                raise RuntimeError(f"set_input_shape({in_name}, {default_shape}) sikertelen")

        self.stream: CudaStream = cuda_stream_create()
        self._ev_start: CudaEvent = cuda_event_create()
        self._ev_stop: CudaEvent = cuda_event_create()

        self.host: Dict[str, np.ndarray] = {}
        self.dev: Dict[str, DevicePtr] = {}
        self.shapes: Dict[str, Tuple[int, ...]] = {}

        for name in self.input_names + self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            if any(d < 0 for d in shape):
                raise RuntimeError(f"Dinamikus shape nincs feloldva a(z) {name} tensoron: {shape}")
            self.shapes[name] = shape
            dtype = trt_nptype(self.engine.get_tensor_dtype(name))
            self.host[name] = np.empty(int(np.prod(shape)), dtype=dtype)
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

    def close(self) -> None:
        try:
            for ptr in self.dev.values():
                cuda_free(ptr)
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

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def infer(self, nchw_fp32: np.ndarray) -> Dict[str, np.ndarray]:
        outs, _ = self.infer_timed(nchw_fp32)
        return outs

    def infer_timed(self, nchw_fp32: np.ndarray) -> Tuple[Dict[str, np.ndarray], float]:
        in_name = self.input_names[0]
        expected_shape = self.shapes[in_name]
        if tuple(nchw_fp32.shape) != expected_shape:
            raise ValueError(f"Input shape {nchw_fp32.shape} != várt {expected_shape}")
        if nchw_fp32.dtype != np.float32:
            raise TypeError("Input dtype legyen float32")

        self.host[in_name][...] = np.ravel(nchw_fp32, order="C")
        cuda_memcpy_htod_async(self.dev[in_name], self.host[in_name], self.stream)

        cuda_event_record(self._ev_start, self.stream)
        ok = self.context.execute_async_v3(stream_handle=int(self.stream))
        if not ok:
            raise RuntimeError("execute_async_v3 sikertelen")
        cuda_event_record(self._ev_stop, self.stream)

        for name in self.output_names:
            cuda_memcpy_dtoh_async(self.host[name], self.dev[name], self.stream)
        cuda_stream_sync(self.stream)

        gpu_ms = cuda_event_elapsed_time(self._ev_start, self._ev_stop)

        outs: Dict[str, np.ndarray] = {}
        for name in self.output_names:
            outs[name] = self.host[name].reshape(self.shapes[name]).copy()
        return outs, float(gpu_ms)


# ================================= ROS2 Node =================================

class InferenceNode(Node):

    def __init__(self) -> None:
        super().__init__("yolopx_trt")
        self.get_logger().info(f"TensorRT verzió: {trt.__version__}")
        self.get_logger().info(f"Engine: {ENGINE}")
        self.get_logger().info(f"Topic (base): {TOPIC}")

        if not ENGINE or not Path(ENGINE).exists():
            raise SystemExit(f"A configban megadott ENGINE nem található: {ENGINE}")

        self.trt_runner = TRT10Runner(ENGINE, default_shape=DEFAULT_INPUT_SHAPE)

        self.lane_thresh = float(LANE_THRESH)
        self.drive_thresh = float(DRIVE_THRESH)

        qos = QoSProfile(depth=1)
        qos.history = HistoryPolicy.KEEP_LAST
        qos.reliability = ReliabilityPolicy.RELIABLE if RELIABLE else ReliabilityPolicy.BEST_EFFORT

        # --- Dinamikus, preferált feliratkozás (compressed > raw) ---
        self._qos = qos
        if TOPIC.endswith('/compressed'):
            self._raw_topic = TOPIC[: -len('/compressed')] or '/'
            self._cmp_topic = TOPIC
        else:
            self._raw_topic = TOPIC
            self._cmp_topic = TOPIC.rstrip('/') + '/compressed'
        self._sub = None
        self._current_kind = None
        self._ensure_best_subscription()

        self._bootstrap_scans_left = 15
        self._bootstrap_timer = self.create_timer(0.5, self._bootstrap_rescan)

        self.last_frame_time: float | None = None
        self.frame_count: int = 0
        self.create_timer(WATCHDOG_PERIOD_SEC, self._watchdog)

        # ---- EGYETLEN KIMENETI PUBLISHER: SZÍNES OVERLAY (BGR8) ----
        self.pub_mask = self.create_publisher(Image, OUT_TOPIC, 10)
        self.get_logger().info(f"Publikálás: overlay (bgr8) -> {OUT_TOPIC}")

        # ---- METRICS ----
        self.metrics_pub = self.create_publisher(String, f"{self.get_name()}/metrics_json", 10)
        self.get_logger().info(f"Metrikák topic: /{self.get_name()}/metrics_json")

        self._last_in_stamp: float | None = None
        self._last_proc_t: float | None = None
        self._in_fps_ema: float = 0.0
        self._out_fps_ema: float = 0.0
        self._alpha: float = 0.1
        self.get_logger().info("Node készen áll. Várjuk a frame-eket…")

    # --------------------------- Dinamikus választó -----------------------------
    def _ensure_best_subscription(self) -> None:
        name_types = {name: types for name, types in self.get_topic_names_and_types()}
        IMG_T = 'sensor_msgs/msg/Image'
        CMP_T = 'sensor_msgs/msg/CompressedImage'

        want_kind = None
        want_topic = None

        if CMP_T in name_types.get(self._cmp_topic, []):
            want_kind = 'compressed'
            want_topic = self._cmp_topic
        elif IMG_T in name_types.get(self._raw_topic, []):
            want_kind = 'raw'
            want_topic = self._raw_topic
        else:
            if self._sub is None:
                self._sub = self.create_subscription(Image, self._raw_topic, self._image_cb_raw, qos_profile=self._qos)
                self._current_kind = 'raw'
                self.get_logger().warn(f"Nincs publisher; standby Image: {self._raw_topic}")
            return

        if self._current_kind == want_kind:
            return

        if self._sub is not None:
            try:
                self.destroy_subscription(self._sub)
            except Exception:
                pass
            finally:
                self._sub = None

        if want_kind == 'compressed':
            self._sub = self.create_subscription(CompressedImage, want_topic, self._image_cb_compressed, qos_profile=self._qos)
        else:
            self._sub = self.create_subscription(Image, want_topic, self._image_cb_raw, qos_profile=self._qos)
        self._current_kind = want_kind
        self.get_logger().info(f"Átváltott feliratkozás: {want_kind} → {want_topic}")

    def _bootstrap_rescan(self) -> None:
        try:
            self._ensure_best_subscription()
        except Exception as e:
            self.get_logger().warn(f"Bootstrap rescan hiba: {e}")
        self._bootstrap_scans_left -= 1
        if self._bootstrap_scans_left <= 0 or self._current_kind == 'compressed':
            try:
                self._bootstrap_timer.cancel()
            except Exception:
                pass

    # ------------------------------- Watchdog -------------------------------
    def _watchdog(self) -> None:
        try:
            self._ensure_best_subscription()
        except Exception as e:
            self.get_logger().warn(f"Rescan hiba: {e}")
        if self.frame_count == 0:
            self.get_logger().warn("Még nem jött frame. Ellenőrizd a topic/QoS beállításokat.")
        elif self.last_frame_time and (time.time() - self.last_frame_time > NO_FRAME_TIMEOUT_SEC):
            self.get_logger().warn("~2.5 másodperce nem jött frame.")

    # ----------------------------- Preprocess -------------------------------
    def _preprocess(self, bgr: np.ndarray) -> np.ndarray:
        resized = cv2.resize(bgr, (int(INPUT_W), int(INPUT_H)), interpolation=cv2.INTER_LINEAR)
        img = resized.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[None, ...]  # (1, 3, H, W)
        return img

    # ----------------------------- Decoderek --------------------------------
    def _decode_compressed(self, msg: CompressedImage) -> np.ndarray | None:
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return frame

    def _decode_image(self, msg: Image) -> np.ndarray | None:
        h, w = int(msg.height), int(msg.width)
        step = int(msg.step)
        enc = (msg.encoding or "").lower()
        buf = memoryview(msg.data)

        def rows_as(shape_cols: int, dtype) -> np.ndarray:
            arr = np.frombuffer(buf, dtype=dtype)
            elem_size = np.dtype(dtype).itemsize
            cols = step // elem_size
            arr = arr.reshape(h, cols)
            return arr[:, :shape_cols]

        try:
            if enc in ("bgr8",):
                arr = rows_as(w * 3, np.uint8).reshape(h, w, 3)
                return arr.copy()
            if enc in ("rgb8",):
                arr = rows_as(w * 3, np.uint8).reshape(h, w, 3)
                return arr[..., ::-1].copy()
            if enc in ("mono8", "8uc1", "mono8; compressed"):
                arr = rows_as(w, np.uint8).reshape(h, w)
                return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            if enc in ("mono16", "16uc1"):
                arr16 = np.frombuffer(buf, dtype=np.uint16)
                if msg.is_bigendian:
                    arr16 = arr16.byteswap()
                cols16 = step // 2
                arr16 = arr16.reshape(h, cols16)[:, :w].reshape(h, w)
                arr8 = (arr16 / 256).astype(np.uint8)
                return cv2.cvtColor(arr8, cv2.COLOR_GRAY2BGR)
        except Exception as e:
            self.get_logger().error(f"Image decode hiba ({enc}): {e}")
            return None

        try:
            self.get_logger().warn(f"Ismeretlen encoding: '{enc}'. Best-effort bgr8 reshape…")
            arr = rows_as(w * 3, np.uint8).reshape(h, w, 3)
            return arr.copy()
        except Exception:
            return None

    # ------------------------------ Common path -----------------------------
    def _process_frame(self, frame: np.ndarray) -> None:
        if frame is None:
            return

        now = time.perf_counter()
        if self._last_proc_t is not None:
            dt = now - self._last_proc_t
            if dt > 0:
                inst_out_fps = 1.0 / dt
                self._out_fps_ema = (1 - self._alpha) * self._out_fps_ema + self._alpha * inst_out_fps
        self._last_proc_t = now

        self.frame_count += 1
        self.last_frame_time = time.time()

        t0 = time.perf_counter()
        inp = self._preprocess(frame)
        outs, gpu_ms = self.trt_runner.infer_timed(inp)

        lane_raw = outs[self.trt_runner.lane_name].astype(np.float32)
        drive_raw = outs[self.trt_runner.drive_name].astype(np.float32)
        lane = self._select_map(lane_raw)
        drive = self._select_map(drive_raw)

        lane_m = (lane > self.lane_thresh).astype(np.uint8)
        drive_m = (drive > self.drive_thresh).astype(np.uint8)
        drive_m = drive_m * (1 - lane_m)  # kizáró osztályok

        h, w = frame.shape[:2]
        lane_m = cv2.resize(lane_m, (w, h), interpolation=cv2.INTER_NEAREST)
        drive_m = cv2.resize(drive_m, (w, h), interpolation=cv2.INTER_NEAREST)

        # === SZÍNES OVERLAY mint az imshow-ban: drive=piros, lane=zöld ===
        overlay = np.zeros_like(frame)
        overlay[drive_m == 1] = (0, 0, 255)
        overlay[lane_m == 1]  = (0, 255, 0)
        vis = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

        # === LEKICSINYÍTÉS a kimenetre (alap: OUT_SCALE=0.5 → fele szél/mag, negyed terület) ===
        scale = float(OUT_SCALE)
        if not (0.0 < scale <= 1.0):
            self.get_logger().warn(f"OUT_SCALE={scale} kívül esik (0,1] tartományon, 1.0-ra állítom")
            scale = 1.0
        out_w = max(1, int(round(w * scale)))
        out_h = max(1, int(round(h * scale)))
        if out_w != w or out_h != h:
            vis = cv2.resize(vis, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

        e2e_ms = (time.perf_counter() - t0) * 1000.0

        # Publikálás: bgr8 overlay
        msg = self._bgr8_to_image_msg(vis)
        self.pub_mask.publish(msg)

        self._publish_metrics(e2e_ms=e2e_ms, gpu_ms=gpu_ms)

    # --------------------------- Map kiválasztás ----------------------------
    @staticmethod
    def _select_map(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 4:
            _, c, _, _ = arr.shape
            return arr[0, 1 if c > 1 else 0]
        if arr.ndim == 3:
            c, _, _ = arr.shape
            return arr[1 if c > 1 else 0]
        if arr.ndim == 2:
            return arr
        raise ValueError(f"Váratlan output shape: {arr.shape}")

    # --------------------------- Numpy → Image msg --------------------------
    def _mono8_to_image_msg(self, mono: np.ndarray) -> Image:
        if mono.dtype != np.uint8:
            mono = mono.astype(np.uint8)
        msg = Image()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = FRAME_ID
        h, w = mono.shape[:2]
        msg.height = int(h)
        msg.width = int(w)
        msg.encoding = 'mono8'
        msg.is_bigendian = 0
        msg.step = int(w)
        msg.data = mono.tobytes()
        return msg

    def _bgr8_to_image_msg(self, bgr: np.ndarray) -> Image:
        if bgr.dtype != np.uint8:
            bgr = bgr.astype(np.uint8)
        if bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError("bgr8 message expects HxWx3 array")
        msg = Image()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = FRAME_ID
        h, w = bgr.shape[:2]
        msg.height = int(h)
        msg.width = int(w)
        msg.encoding = 'bgr8'
        msg.is_bigendian = 0
        msg.step = int(w) * 3
        msg.data = bgr.tobytes()
        return msg

    # --------------------------- ROS2 callbackok ----------------------------
    def _image_cb_compressed(self, msg: CompressedImage) -> None:
        try:
            now = time.perf_counter()
            if self._last_in_stamp is not None:
                dt = now - self._last_in_stamp
                if dt > 0:
                    inst_in_fps = 1.0 / dt
                    self._in_fps_ema = (1 - self._alpha) * self._in_fps_ema + self._alpha * inst_in_fps
            self._last_in_stamp = now

            frame = self._decode_compressed(msg)
            if frame is None:
                self.get_logger().warn("imdecode None-t adott vissza (CompressedImage)")
                return
            self._process_frame(frame)
        except Exception as e:
            self.get_logger().error(f"Compressed callback hiba: {e}")

    def _image_cb_raw(self, msg: Image) -> None:
        try:
            now = time.perf_counter()
            if self._last_in_stamp is not None:
                dt = now - self._last_in_stamp
                if dt > 0:
                    inst_in_fps = 1.0 / dt
                    self._in_fps_ema = (1 - self._alpha) * self._in_fps_ema + self._alpha * inst_in_fps
            self._last_in_stamp = now

            frame = self._decode_image(msg)
            if frame is None:
                self.get_logger().warn("Image decode sikertelen (raw)")
                return
            self._process_frame(frame)
        except Exception as e:
            self.get_logger().error(f"Raw Image callback hiba: {e}")

    # ------------------------------ Metrics pub ------------------------------
    def _publish_metrics(self, e2e_ms: float, gpu_ms: float) -> None:
        try:
            proc_ratio = (self._out_fps_ema / self._in_fps_ema) if self._in_fps_ema > 0 else 0.0
            m = {
                "in_fps": round(self._in_fps_ema, 2),
                "out_fps": round(self._out_fps_ema, 2),
                "latency_ms_e2e": round(float(e2e_ms), 2),
                "latency_ms_gpu": round(float(gpu_ms), 2),
                "proc_ratio": round(proc_ratio, 2),
                "frames": int(self.frame_count),
                "out_scale": float(OUT_SCALE),
            }
            self.metrics_pub.publish(String(data=json.dumps(m)))
        except Exception as e:
            self.get_logger().warn(f"Metrics publish hiba: {e}")


# ================================== main ===================================

def main() -> None:
    rclpy.init(args=None)
    node = InferenceNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.trt_runner.close()
        except Exception:
            pass
        node.destroy_node()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
