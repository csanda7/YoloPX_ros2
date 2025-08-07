import argparse
import os
import sys

import cv2
import numpy as np
import onnx
import onnxruntime as ort

# fallback saver
try:
    import imageio
except ImportError:
    imageio = None


def to_probability(scores: np.ndarray) -> np.ndarray:
    if np.any(scores < 0) or np.any(scores > 1):
        return 1.0 / (1.0 + np.exp(-scores))
    return scores


def binarize_and_clean(scores: np.ndarray, threshold: float) -> np.ndarray:
    probs = to_probability(scores)
    mask = (probs >= threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def save_image(path: str, image: np.ndarray) -> bool:
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    # Handle duplicate filenames by appending a number
    base, ext = os.path.splitext(path)
    counter = 1
    while os.path.exists(path):
        path = f"{base}_{counter}{ext}"
        counter += 1

    success = False
    try:
        success = cv2.imwrite(path, image)
    except Exception as e:
        print(f"[WARN] cv2.imwrite failed: {e}")
    if not success and imageio is not None:
        try:
            imageio.imwrite(path, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            success = True
        except Exception as e:
            print(f"[WARN] imageio.imwrite fallback failed: {e}")
    if not success:
        print(f"[ERROR] Could not save image to {path}")
    return success


class RoadLaneSegmentation:
    def __init__(self, model_path, lane_thresh=0.5, drivable_thresh=0.5):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.lane_thresh = lane_thresh
        self.drivable_thresh = drivable_thresh

    def preprocess(self, image, input_size=(640, 640)):
        resized = cv2.resize(image, input_size)
        normalized = resized.astype(np.float32) / 255.0
        transposed = np.transpose(normalized, (2, 0, 1))  # HWC -> CHW
        return np.expand_dims(transposed, axis=0).astype(np.float32)

    def segment(self, image):
        inp = self.preprocess(image)
        outputs = self.session.run(None, {self.input_name: inp})

        names = [o.name for o in self.session.get_outputs()]
        print("Model outputs/shapes:")
        for n, out in zip(names, outputs):
            print(f"  - {n}: {out.shape}")


        # Grab the two segmentation heads
        if "lane_line_seg" not in names or "drive_area_seg" not in names:
            raise RuntimeError("Missing lane_line_seg or drive_area_seg in model outputs")

        lane_raw = np.squeeze(outputs[names.index("lane_line_seg")])      # shape (2,H,W)
        drive_raw = np.squeeze(outputs[names.index("drive_area_seg")])    # shape (2,H,W)

        # **Select the second channel only** (channel 1 = positive class)
        lane_scores = lane_raw[1] if lane_raw.ndim == 3 else lane_raw
        drive_scores = drive_raw[1] if drive_raw.ndim == 3 else drive_raw

        # Binarize + clean
        lane_mask_small = binarize_and_clean(lane_scores, self.lane_thresh)
        drive_mask_small = binarize_and_clean(drive_scores, self.drivable_thresh)

        # Lane priority
        drive_mask_small = drive_mask_small * (1 - lane_mask_small)

        # Upsample to original size
        h, w = image.shape[:2]
        lane_mask = cv2.resize(lane_mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
        drive_mask = cv2.resize(drive_mask_small, (w, h), interpolation=cv2.INTER_NEAREST)

        return lane_mask, drive_mask


def make_blackbg_overlay(base_image, lane_mask, drive_mask):
    overlay = np.zeros_like(base_image)
    overlay[lane_mask == 1] = [0, 0, 255]   # Red lanes
    overlay[drive_mask == 1] = [0, 255, 0]  # Green drivable
    return overlay


def make_on_image(base_image, lane_mask, drive_mask):
    img_copy = base_image.copy()
    img_copy[lane_mask == 1] = [0, 0, 255]
    img_copy[drive_mask == 1] = [0, 255, 0]
    return img_copy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  required=True)
    parser.add_argument("--image",  required=True)
    parser.add_argument("--out-dir", required=True,
                        help="Directory to save: blackbg.png and on_orig.png")
    parser.add_argument("--lane-thresh",    type=float, default=0.6)
    parser.add_argument("--drivable-thresh",type=float, default=0.85)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    # Validate files
    for path in (args.model, args.image):
        if not os.path.isfile(path):
            print(f"File not found: {path}")
            sys.exit(1)

    # Validate model
    try:
        m = onnx.load(args.model); onnx.checker.check_model(m)
        print("ONNX model is valid.")
    except Exception as e:
        print("ONNX validation failed:", e)
        sys.exit(1)

    img = cv2.imread(args.image)
    if img is None:
        print("Could not read image."); sys.exit(1)

    seg = RoadLaneSegmentation(
        args.model,
        lane_thresh=args.lane_thresh,
        drivable_thresh=args.drivable_thresh
    )

    try:
        lm, dm = seg.segment(img)
    except Exception as e:
        print("Segmentation error:", e); sys.exit(1)

    # Generate outputs
    blackbg = make_blackbg_overlay(img, lm, dm)
    on_orig = make_on_image    (img, lm, dm)

    # Save
    os.makedirs(args.out_dir, exist_ok=True)
    
    black_path = os.path.join(args.out_dir, "segmentation_blackbg.png")
    orig_path  = os.path.join(args.out_dir, "segmentation_on_orig.png")

    if save_image(black_path, blackbg):
        print("Saved black-bg overlay:", black_path)
    if save_image(orig_path, on_orig):
        print("Saved mask-on-original:", orig_path)

    # Display if requested
    if args.show:
        cv2.imshow("Black Background Overlay", blackbg)
        cv2.imshow("Mask on Original",      on_orig)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
