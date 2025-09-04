# ------------------------------ Model & IO ---------------------------------
ENGINE: str = "//home/csanda/Work/JKK/ros2/yolopx/models/yolopx5.engine"  
TOPIC: str = "/zed2i/zed_node/left/image_rect_color"
RELIABLE: bool = False  
SHOW: bool = True       #opencv ablak

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
OUT_SCALE = 0.25        # 0<scale≤1; 0.5 → fele szél/mag, ~negyed terület
FRAME_ID = "camera"
VIS_MODE = "overlay"   # vagy "palette" (olcsóbb)
PREFER_COMPRESSED = True  # CPU kímélés: raw Image preferált, ha elérhető