import sys
print("[DEBUG] PYTHON EXEC:", sys.executable)
print("[DEBUG] PYTHON PATH:", sys.path)

#!/home/csanda/Work/JKK/ros2/yolopx/venv/bin/python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import onnxruntime as ort
import cv2
import numpy as np


class InferenceNode(Node):
    def __init__(self):
        super().__init__('yolopx_inference_node')

        # Load model
        model_path = '/home/csanda/Work/JKK/ros2/yolopx/yolopx_fp32.onnx'
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

        # Parameters
        self.lane_thresh = 0.7
        self.drive_thresh = 0.8
        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            CompressedImage,
            '/zed2i/zed_node/left/image_rect_color/compressed',
            self.image_callback,
            10
        )

        self.get_logger().info("YoloPX inference node ready.")
        self.get_logger().info(self.subscription.topic_name)
    
    
    def compressed_to_image(self, compressed_msg):
        # Decode JPEG/PNG from compressed message
        np_arr = np.frombuffer(compressed_msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  

        return cv_image  

    def preprocess(self, image):
        resized = cv2.resize(image, (640, 640))
        img = resized.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return np.expand_dims(img, axis=0).astype(np.float32)

    def image_callback(self, msg):
        #print("I got the data 👀")  # Debugging log

        try:
            frame = self.compressed_to_image(msg)
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return

        input_tensor = self.preprocess(frame)
        outputs = self.session.run(None, {self.input_name: input_tensor})

        lane_seg = np.squeeze(outputs[4])[1]  # lane_line_seg
        drive_seg = np.squeeze(outputs[5])[1]  # drive_area_seg

        lane_mask = (lane_seg > self.lane_thresh).astype(np.uint8)
        drive_mask = ((drive_seg > self.drive_thresh).astype(np.uint8)) * (1 - lane_mask)

        lane_mask = cv2.resize(lane_mask, (frame.shape[1], frame.shape[0]))
        drive_mask = cv2.resize(drive_mask, (frame.shape[1], frame.shape[0]))

        overlay = np.zeros_like(frame)
        overlay[drive_mask == 1] = [0, 0, 255]     
        overlay[lane_mask == 1] = [0, 255, 0]    

        combined = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
        cv2.imshow("YoloPX Real-Time Inference", combined)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    print("Starting...")
    node = InferenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
