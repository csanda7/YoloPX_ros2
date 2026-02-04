# ------------------------------ Model & IO ---------------------------------
ENGINE: str = "/home/orin/ros2_ws/src/YoloPX_ros2/models/yolopx.engine"  
TOPIC: str = "/zed2i/zed_node/left/image_rect_color"
RELIABLE: bool = False  
SHOW: bool = False       

# ------------------------------ Thresholds ---------------------------------
LANE_THRESH: float = 0.70
DRIVE_THRESH: float = 0.80

# ------------------------------ Input size ---------------------------------
INPUT_W: int = 640
INPUT_H: int = 640
DEFAULT_INPUT_SHAPE = (1, 3, INPUT_H, INPUT_W)

# ------------------------------ UI / timing --------------------------------
WINDOW_TITLE: str = "YOLOPX Inference"
WATCHDOG_PERIOD_SEC: float = 1.0
NO_FRAME_TIMEOUT_SEC: float = 2.5

OUT_TOPIC = "/yolopx/mask"
OUT_SCALE = 0.25
FRAME_ID = "zed_camera_front"
VIS_MODE = "palette"   # overlay vagy "palette" (olcsóbb)
PREFER_COMPRESSED = True  

GPU_TIMING_EVERY: int = 3