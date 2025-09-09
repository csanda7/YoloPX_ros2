from __future__ import annotations
import ctypes
from typing import Tuple
import numpy as np
from cuda import cudart

DevicePtr = int
CudaStream = int
CudaEvent = int

# --- Alap hibakezelés ---

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


# --- Stream ---

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


# --- Memória ---

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


# --- Pinned host memória ---

def cuda_host_alloc(nbytes: int) -> int:
    try:
        err, ptr = cudart.cudaHostAlloc(nbytes, 0)
        check_cuda(err)
        return int(ptr)
    except TypeError:
        p = ctypes.c_void_p()
        check_cuda(cudart.cudaHostAlloc(ctypes.byref(p), nbytes, 0))
        return int(p.value)


def cuda_host_free(ptr: int) -> None:
    try:
        err = cudart.cudaFreeHost(int(ptr))
        if isinstance(err, tuple):
            check_cuda(err[0])
        else:
            check_cuda(err)
    except Exception:
        pass


def pinned_empty_1d(nelems: int, dtype: np.dtype) -> Tuple[np.ndarray, int]:
    dtype = np.dtype(dtype)
    nbytes = int(nelems) * int(dtype.itemsize)
    ptr = cuda_host_alloc(nbytes)
    buf = (ctypes.c_uint8 * nbytes).from_address(int(ptr))
    arr = np.ctypeslib.as_array(buf)
    arr = arr.view(dtype=dtype)
    arr = arr[:nelems]
    return arr, ptr


# --- Event / timing ---

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


def cuda_event_query(event: CudaEvent) -> bool:
    """Non-blocking check: returns True if event completed."""
    try:
        err = cudart.cudaEventQuery(int(event))
        # tuple form first element is status
        if isinstance(err, tuple):
            err = err[0]
        code_ready = _cuda_err_code(cudart.cudaError_t.cudaSuccess)
        if _cuda_err_code(err) == code_ready:
            return True
        # cudaErrorNotReady → not finished yet
        return False
    except Exception:
        # Conservative: assume not ready
        return False
