#!/usr/bin/env python3
from __future__ import annotations

# ============================== Standard libs ===============================
import time
import json
from pathlib import Path
from typing import Dict

# ============================== Third-party libs =============================
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String, Header

import tensorrt as trt

# ============================== Saját konfig ================================
from yolopx.config import (
    ENGINE,
    TOPIC,
    RELIABLE,
    LANE_THRESH,
    DRIVE_THRESH,
    DEFAULT_INPUT_SHAPE,
    WATCHDOG_PERIOD_SEC,
    NO_FRAME_TIMEOUT_SEC,
    OUT_TOPIC,
    OUT_SCALE,
    FRAME_ID,
    VIS_MODE,         # "overlay" vagy "palette"
    GPU_TIMING_EVERY,
)

from .trt_runner import get_trt_runner, TRTPipelineRunner

# ================================= ROS2 Node =================================

class InferenceNode(Node):

    def __init__(self) -> None:
        super().__init__("yolopx_trt")
        self.get_logger().info(f"TensorRT verzió: {trt.__version__}")
        self.get_logger().info(f"Engine: {ENGINE}")
        self.get_logger().info(f"Input topic: {TOPIC}")

        if not ENGINE or not Path(ENGINE).exists():
            raise SystemExit(f"A configban megadott ENGINE nem található: {ENGINE}")

        # !!! Single process-wide shared runner (context)!
        self.trt_runner = get_trt_runner(ENGINE, DEFAULT_INPUT_SHAPE)
        # Pipeline runner (double-buffer)
        try:
            self.pipeline = TRTPipelineRunner(self.trt_runner, num_slots=3)
            self.get_logger().info("Pipeline runner inicializálva (3 slot)")
        except Exception as e:
            self.get_logger().warn(f"Pipeline init hiba, fallback sync mód: {e}")
            self.pipeline = None

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

        # ---- Kimeneti publisher: bgr8 ----
        self.pub_mask = self.create_publisher(Image, OUT_TOPIC, 10)
        self.get_logger().info(f"Publikálás: {VIS_MODE} -> {OUT_TOPIC} (bgr8)")

        # ---- Viz bufferek (újrahasznosítás CPU kíméléshez) ----
        self._overlay: np.ndarray | None = None
        self._vis: np.ndarray | None = None
        self._lut = np.array([[0,0,0], [0,255,0], [0,0,255]], dtype=np.uint8)  # palette módhoz

        # ---- METRICS ----
        self.metrics_pub = self.create_publisher(String, f"{self.get_name()}/metrics_json", 10)
        self.get_logger().info(f"Metrikák topic: /{self.get_name()}/metrics_json")

        self._last_in_stamp: float | None = None
        self._last_proc_t: float | None = None
        self._in_fps_ema: float = 0.0
        self._out_fps_ema: float = 0.0
        self._alpha: float = 0.1
        self.get_logger().info("Node készen áll. Várjuk a frame-eket…")
        # Polling timer pipeline fetch-hez
        self._poll_timer = self.create_timer(0.0, self._poll_pipeline)
        # GPU timing mintavételezés
        try:
            self._timing_every = int(GPU_TIMING_EVERY)
        except Exception:
            self._timing_every = 1
        if self._timing_every <= 0:
            self._timing_every = 1
        self._last_gpu_ms: float = 0.0

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
        # EREDETI: új tömb létrehozása minden frame-nél.
        # OPT: közvetlenül a runner pinned host input bufferébe írunk, így nincs extra másolat.
        in_name = self.trt_runner.input_names[0]
        shp = self.trt_runner.shapes[in_name]  # (1,3,H,W)
        _, c, h, w = shp
        assert c == 3, "Csak 3 csatornás bgr támogatott"
        host_view = self.trt_runner.host[in_name].reshape(shp)

        resized = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_LINEAR)
        # float32 normalizálás (1/255)
        rf = resized.astype(np.float32)
        rf *= (1.0/255.0)
        # Csatornák átmásolása NCHW (N=1)
        host_view[0, 0, :, :] = rf[:, :, 0]
        host_view[0, 1, :, :] = rf[:, :, 1]
        host_view[0, 2, :, :] = rf[:, :, 2]
        return host_view  # visszaadjuk a shape ellenőrzés kedvéért (nem kötelező)

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

    # ----------------------------- FPS helper -------------------------------
    def _update_in_fps(self) -> None:
        now = time.perf_counter()
        if self._last_in_stamp is not None:
            dt = now - self._last_in_stamp
            if dt > 0:
                inst_in = 1.0 / dt
                self._in_fps_ema = (1 - self._alpha) * self._in_fps_ema + self._alpha * inst_in
        self._last_in_stamp = now

    # ------------------------------ Common path -----------------------------
    def _ensure_buffers(self, out_h: int, out_w: int) -> None:
        need_overlay = (self._overlay is None) or (self._overlay.shape[0] != out_h) or (self._overlay.shape[1] != out_w)
        need_vis = (self._vis is None) or (self._vis.shape[0] != out_h) or (self._vis.shape[1] != out_w)
        if need_overlay:
            self._overlay = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        if need_vis:
            self._vis = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    def _postprocess_and_publish(self, frame: np.ndarray, outs: Dict[str, np.ndarray], gpu_ms: float, t_enqueue: float) -> None:
        t0 = t_enqueue
        lane_raw = outs[self.trt_runner.lane_name]
        if lane_raw.dtype != np.float32:
            lane_raw = lane_raw.astype(np.float32)
        drive_raw = outs[self.trt_runner.drive_name]
        if drive_raw.dtype != np.float32:
            drive_raw = drive_raw.astype(np.float32)
        lane = self._select_map(lane_raw)
        drive = self._select_map(drive_raw)
        lane_m = (lane > self.lane_thresh).astype(np.uint8)
        drive_m = (drive > self.drive_thresh).astype(np.uint8)
        # In-place kizárás: ahol lane van, ott drive=0
        drive_m[lane_m == 1] = 0
        h, w = frame.shape[:2]
        scale = float(OUT_SCALE)
        if not (0.0 < scale <= 1.0):
            scale = 1.0
        out_w = max(1, int(round(w * scale)))
        out_h = max(1, int(round(h * scale)))
        self._ensure_buffers(out_h, out_w)
        lane_s  = cv2.resize(lane_m,  (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        drive_s = cv2.resize(drive_m, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        if VIS_MODE.lower() == "palette":
            combined = lane_s + (drive_s * 2)
            self._vis[:, :, :] = self._lut[combined]
        else:
            frame_s = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            ov = self._overlay
            ov.fill(0)
            ov[..., 1] = lane_s; np.multiply(ov[..., 1], 255, out=ov[..., 1], casting='unsafe')
            ov[..., 2] = drive_s; np.multiply(ov[..., 2], 255, out=ov[..., 2], casting='unsafe')
            cv2.addWeighted(frame_s, 0.6, ov, 0.4, 0.0, dst=self._vis)
        e2e_ms = (time.perf_counter() - t0) * 1000.0
        msg = self._bgr8_to_image_msg(self._vis)
        self.pub_mask.publish(msg)
        self._publish_metrics(e2e_ms=e2e_ms, gpu_ms=gpu_ms)

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
        if self.pipeline is None:
            t_enqueue = time.perf_counter()
            self._preprocess(frame)
            outs, gpu_ms = self.trt_runner.infer_timed(None, copy_outputs=False)
            self._postprocess_and_publish(frame, outs, gpu_ms, t_enqueue)
            return
        # Pipeline: slot -> preprocess -> enqueue (ha nincs szabad, eldobjuk a frame-et)
        slot = self.pipeline.acquire_slot()
        if slot is None:
            return  # drop frame
        in_name = self.trt_runner.input_names[0]
        host_view = slot.in_host.reshape(self.trt_runner.shapes[in_name])
        # Preprocess a slot host bufferébe
        resized = cv2.resize(frame, (host_view.shape[3], host_view.shape[2]), interpolation=cv2.INTER_LINEAR)
        rf = resized.astype(np.float32); rf *= (1.0/255.0)
        host_view[0,0,:,:] = rf[:,:,0]; host_view[0,1,:,:] = rf[:,:,1]; host_view[0,2,:,:] = rf[:,:,2]
        slot.frame = frame
        # GPU idő mérés csak minden N-edik frame-en a timing overhead csökkentésére
        measure_timing = (self.frame_count % self._timing_every) == 0
        self.pipeline.enqueue(slot, host_view, measure_timing=measure_timing)

    def _poll_pipeline(self) -> None:
        if self.pipeline is None:
            return
        ready = self.pipeline.fetch_ready(copy_outputs=False)
        if not ready:
            return
        for slot, outs, gpu_ms in ready:
            frame = slot.frame
            if frame is None:
                continue
            if gpu_ms <= 0.0:
                # Nincs új mérés → használjuk az utolsó ismert értéket (becslés)
                gpu_display = self._last_gpu_ms
            else:
                gpu_display = gpu_ms
                self._last_gpu_ms = gpu_ms
            self._postprocess_and_publish(frame, outs, gpu_display, slot.t_enqueue)
            slot.frame = None

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
            self._update_in_fps()

            frame = self._decode_compressed(msg)
            if frame is None:
                self.get_logger().warn("imdecode None-t adott vissza (CompressedImage)")
                return
            self._process_frame(frame)
        except Exception as e:
            self.get_logger().error(f"Compressed callback hiba: {e}")

    def _image_cb_raw(self, msg: Image) -> None:
        try:
            self._update_in_fps()

            frame = self._decode_image(msg)
            if frame is None:
                self.get_logger().warn("Image decode sikertelen (raw)")
                return
            self._process_frame(frame)
        except Exception as e:
            self.get_logger().error(f"Raw Image callback hiba: {e}")

    # ------------------------------ Metrics pub ------------------------------
    def _publish_metrics(self, e2e_ms: float, gpu_ms: float) -> None:
        # Ritkítás: max ~5 Hz (CPU kímélés)
        try:
            now = time.perf_counter()
            if not hasattr(self, "_metrics_last_pub"):
                self._metrics_last_pub = 0.0
            if (now - self._metrics_last_pub) < 0.2:  # 5 Hz
                return
            self._metrics_last_pub = now

            proc_ratio = (self._out_fps_ema / self._in_fps_ema) if self._in_fps_ema > 0 else 0.0
            m = {
                "in_fps": round(self._in_fps_ema, 2),
                "out_fps": round(self._out_fps_ema, 2),
                "latency_ms_e2e": round(float(e2e_ms), 2),
                "latency_ms_gpu": round(float(gpu_ms), 2),
                "proc_ratio": round(proc_ratio, 2),
                "frames": int(self.frame_count),
                "out_scale": float(OUT_SCALE),
                "vis_mode": str(VIS_MODE),
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
        node.destroy_node()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
