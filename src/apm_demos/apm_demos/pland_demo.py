import cv2
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

CAMERA_TOPIC = "/camera/image_raw"
HEIGHT = 640
WIDTH = 480
HFOV = 1.0236  # in rad


class MyNode(Node):
    def __init__(self):
        node_name = "pland_demo"
        super().__init__(node_name)
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            Image, CAMERA_TOPIC, self.image_handler, qos_profile=qos_profile_sensor_data
        )
        self.get_logger().info("Hello ROS2")

    def image_handler(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            cv2.imshow("image", frame)
            cv2.waitKey(1)
        except CvBridgeError:
            self.get_logger().error("load image failed")


def main():
    rclpy.init()
    node = MyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("User exit")
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
