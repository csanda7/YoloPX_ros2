#!/usr/bin/env python3
# YOLOPX TensorRT 10.9 (ROS2) – CUDA Python 12.9/13 robust adapter

import sys, time, argparse, ctypes
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage

import tensorrt as trt
from cuda import cudart   # 12.9: deprecated alias -> ok; 12.x classic -> ok; 13.x -> wrapper kezeli

# ====================== CUDA helpers (tuple-first, byref-fallback) ======================

def _cuda_err_code(e):
    # norm: int/enum/tuple -> int kód
    if isinstance(e, tuple):  # pl. (cudaSuccess,)
        e = e[0]
    try:
        return int(e)
    except Exception:
        # enum jellegű (pl. <cudaError_t.cudaSuccess: 0>)
        v = getattr(e, "value", e)
        try:
            return int(v)
        except Exception:
            return 9999  # ismeretlen, majd lekezeli alább

def checkCuda(err):
    code = _cuda_err_code(err)
    ok = _cuda_err_code(cudart.cudaError_t.cudaSuccess)
    if code != ok:
        raise RuntimeError(f"CUDA error: {err}")


def cuda_stream_create() -> int:
    """Return stream handle as int (works for 12.x/12.9/13.x)."""
    try:
        # New/tuple API
        err, stream = cudart.cudaStreamCreate()
        checkCuda(err)
        return int(stream)
    except TypeError:
        # Old/byref API
        s = ctypes.c_void_p()
        checkCuda(cudart.cudaStreamCreate(ctypes.byref(s)))
        return int(s.value)

def cuda_stream_sync(stream: int):
    checkCuda(cudart.cudaStreamSynchronize(int(stream)))

def cuda_malloc(nbytes: int) -> int:
    """Return device pointer as int."""
    try:
        err, dptr = cudart.cudaMalloc(nbytes)
        checkCuda(err)
        return int(dptr)
    except TypeError:
        p = ctypes.c_void_p()
        checkCuda(cudart.cudaMalloc(ctypes.byref(p), nbytes))
        return int(p.value)

def cuda_free(dptr: int):
    checkCuda(cudart.cudaFree(int(dptr)))

def cuda_memcpy_htod_async(dst_dev_ptr: int, src_np: np.ndarray, stream: int):
    nbytes = src_np.nbytes
    src_addr = int(src_np.ctypes.data)
    try:
        checkCuda(cudart.cudaMemcpyAsync(
            int(dst_dev_ptr), src_addr, nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, int(stream)))
    except TypeError:
        # Old API elfogadja a ctypes.c_void_p-t is, de int is jó
        checkCuda(cudart.cudaMemcpyAsync(
            ctypes.c_void_p(int(dst_dev_ptr)),
            ctypes.c_void_p(src_addr),
            nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            ctypes.c_void_p(int(stream))))

def cuda_memcpy_dtoh_async(dst_np: np.ndarray, src_dev_ptr: int, stream: int):
    nbytes = dst_np.nbytes
    dst_addr = int(dst_np.ctypes.data)
    try:
        checkCuda(cudart.cudaMemcpyAsync(
            dst_addr, int(src_dev_ptr), nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, int(stream)))
    except TypeError:
        checkCuda(cudart.cudaMemcpyAsync(
            ctypes.c_void_p(dst_addr),
            ctypes.c_void_p(int(src_dev_ptr)),
            nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            ctypes.c_void_p(int(stream))))

def trt_nptype(dtype):
    return np.dtype({
        trt.DataType.FLOAT:  np.float32,
        trt.DataType.HALF:   np.float16,
        trt.DataType.INT8:   np.int8,
        trt.DataType.INT32:  np.int32,
        trt.DataType.BOOL:   np.bool_,
    }[dtype])

# ============================== TensorRT Runner ==============================

class TRT10Runner:
    def __init__(self, engine_path: str, default_shape=(1,3,640,640)):
        self.logger = trt.Logger(trt.Logger.INFO)
        self.runtime = trt.Runtime(self.logger)
        with open(engine_path, "rb") as f:
            engine_bytes = f.read()
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create execution context")

        self.default_shape = tuple(default_shape)

        # IO discovery
        self.input_names, self.output_names = [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)
        if len(self.input_names) != 1:
            raise RuntimeError(f"Expected exactly 1 input, got {self.input_names}")

        # Input shape
        in_name = self.input_names[0]
        in_shape = list(self.engine.get_tensor_shape(in_name))
        if any(d < 0 for d in in_shape):
            self.context.set_input_shape(in_name, self.default_shape)
        else:
            self.default_shape = tuple(in_shape)

        # CUDA stream
        self.stream = cuda_stream_create()

        # Host/device buffers
        self.host = {}
        self.dev = {}
        self.shapes = {}

        for name in self.input_names + self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            if any(d < 0 for d in shape):
                raise RuntimeError(f"Dynamic shape unresolved for tensor {name}: {shape}")
            self.shapes[name] = shape
            dtype = trt_nptype(self.engine.get_tensor_dtype(name))
            self.host[name] = np.empty(int(np.prod(shape)), dtype=dtype)
            self.dev[name] = cuda_malloc(self.host[name].nbytes)

        # Bind device addresses (int ptr-eket adunk)
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            self.context.set_tensor_address(name, int(self.dev[name]))

        # Output name guess
        self.lane_name, self.drive_name = None, None
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
                raise RuntimeError("Not enough outputs to map lane/drive")

    def infer(self, nchw_fp32: np.ndarray):
        in_name = self.input_names[0]
        if tuple(nchw_fp32.shape) != self.shapes[in_name]:
            raise ValueError(f"Input shape {nchw_fp32.shape} != expected {self.shapes[in_name]}")

        # H2D
        self.host[in_name][...] = nchw_fp32.ravel()
        cuda_memcpy_htod_async(self.dev[in_name], self.host[in_name], self.stream)

        # Exec (TRT 10.9)
        self.context.execute_async_v3(stream_handle=int(self.stream))

        # D2H
        for name in self.output_names:
            cuda_memcpy_dtoh_async(self.host[name], self.dev[name], self.stream)
        cuda_stream_sync(self.stream)

        outs = {}
        for name in self.output_names:
            outs[name] = self.host[name].reshape(self.shapes[name]).copy()
        return outs

# ================================= ROS2 Node =================================

class InferenceNode(Node):
    def __init__(self, engine_path, topic, reliable=False, show=True):
        super().__init__('yolopx_trt')
        self.get_logger().info(f"TRT: {trt.__version__}")
        self.get_logger().info(f"Engine: {engine_path}")
        self.get_logger().info(f"Topic: {topic}")

        self.trt_runner = TRT10Runner(engine_path, default_shape=(1,3,640,640))
        self.lane_thresh = 0.7
        self.drive_thresh = 0.8
        self.show = show

        qos = QoSProfile(depth=1)
        qos.history = HistoryPolicy.KEEP_LAST
        qos.reliability = ReliabilityPolicy.RELIABLE if reliable else ReliabilityPolicy.BEST_EFFORT

        self.subscription = self.create_subscription(
            CompressedImage, topic, self.image_cb, qos_profile=qos
        )

        self.last_frame_time = None
        self.frame_count = 0
        self.create_timer(2.0, self._watchdog)
        self.get_logger().info("Node ready. Waiting for frames...")

    def _watchdog(self):
        if self.frame_count == 0:
            self.get_logger().warn("No frames yet. Check topic/QoS.")
        elif self.last_frame_time and time.time() - self.last_frame_time > 2.5:
            self.get_logger().warn("No frames in last ~2.5s.")

    def _preprocess(self, bgr):
        resized = cv2.resize(bgr, (640, 640), interpolation=cv2.INTER_LINEAR)
        img = resized.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[None, ...]  # (1,3,640,640)
        return img

    @staticmethod
    def _select_map(arr: np.ndarray) -> np.ndarray:
        # (1,2,H,W)->ch1, (2,H,W)->ch1, (1,H,W)->ch0, (H,W)->itself
        if arr.ndim == 4:
            _, c, _, _ = arr.shape
            return arr[0, 1 if c > 1 else 0]
        if arr.ndim == 3:
            c, _, _ = arr.shape
            return arr[1 if c > 1 else 0]
        if arr.ndim == 2:
            return arr
        raise ValueError(f"Unexpected output shape: {arr.shape}")

    def image_cb(self, msg: CompressedImage):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                self.get_logger().warn("imdecode returned None")
                return

            self.frame_count += 1
            self.last_frame_time = time.time()

            h, w = frame.shape[:2]
            inp = self._preprocess(frame)
            outs = self.trt_runner.infer(inp)

            lane_raw = outs[self.trt_runner.lane_name].astype(np.float32)
            drive_raw = outs[self.trt_runner.drive_name].astype(np.float32)
            lane = self._select_map(lane_raw)
            drive = self._select_map(drive_raw)

            lane_m = (lane > self.lane_thresh).astype(np.uint8)
            drive_m = (drive > self.drive_thresh).astype(np.uint8)
            drive_m = drive_m * (1 - lane_m)

            lane_m = cv2.resize(lane_m, (w, h), interpolation=cv2.INTER_NEAREST)
            drive_m = cv2.resize(drive_m, (w, h), interpolation=cv2.INTER_NEAREST)

            if self.show:
                overlay = np.zeros_like(frame)
                overlay[drive_m == 1] = (0, 0, 255)   # drive: piros
                overlay[lane_m == 1]  = (0, 255, 0)   # lane: zöld
                out = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
                cv2.imshow("YOLOPX TRT10.9", out)
                cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Callback error: {e}")

# ================================== main ===================================

def main():
    app_argv = remove_ros_args(sys.argv)

    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--topic", default="/zed2i/zed_node/left/image_rect_color/compressed")
    ap.add_argument("--reliable", action="store_true")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args(app_argv[1:])

    rclpy.init(args=None)
    node = InferenceNode(args.engine, args.topic, reliable=args.reliable, show=(not args.no_show))
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
if __name__ == "__main__":
    main()
