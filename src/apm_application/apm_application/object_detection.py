import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import math

CAMERA_TOPIC = "/gimbal_camera/image_raw"

class ObjectTracker(Node):
    def __init__(self):
        super().__init__('object_tracker')
        self.get_logger().info("object tracker running")
        self.publisher_ = self.create_publisher(String, 'object_coordinates', 10)
        self.image_publisher_ = self.create_publisher(Image, 'object_tracking/image_raw', 10)
        self.subscription = self.create_subscription(Image, CAMERA_TOPIC, self.image_callback, 10)
        # self.back_sub = cv2.createBackgroundSubtractorMOG2(history=700, varThreshold=25, detectShadows=True)
        # self.kernel = np.ones((20,20),np.uint8)
        self.bridge = CvBridge()

    def image_callback(self, msg):
        mask = 0
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # fg_mask = self.back_sub.apply(frame)
        # fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel)
        # fg_mask = cv2.medianBlur(fg_mask, 5)
        # _, fg_mask = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)
        mask += cv2.inRange(hsv, (36, 25, 25), (70, 255,255))
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.imshow("debug", mask)
        # cv2.waitKey(1)
        areas = [cv2.contourArea(c) for c in contours]

        if len(areas) < 1:
            self.publish_image(frame)
            return

        max_index = np.argmax(areas)
        cnt = contours[max_index]
        x, y, w, h = cv2.boundingRect(cnt)

        obj_p = np.array([
                [x, y, 0],
                [x + w, y, 0],
                [x + w, y + h, 0],
                [x, y + h, 0]
            ])
        
        img_p = np.array([
            (x, y),
            (x + w, y),
            (x + w, y + h),
            (x, y + h)
        ])

        # 3d points
        img_p = np.zeros((6*9,3), np.float32)
        img_p[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

        print(x,y,w,h)
        size = (640, 480)
        camera_matrix = np.array(
            [[630, 0, size[0]/2],
             [0, 630, size[1]/2],
             [0,0,1]], dtype=np.double
        )
        dist_coeffs = np.zeros((4,1)) # no lens distortion
        _, rvec, tvec = cv2.solvePnP(obj_p, img_p, camera_matrix, dist_coeffs)

        # Convert rotation vector to Euler angles
        rmat, _ = cv2.Rodrigues(rvec)
        pitch = math.atan2(rmat[2, 0], rmat[2, 2])
        yaw = math.atan2(rmat[2, 1], rmat[2, 2])

        # Calculate position
        position = tvec[:2]  
        print(position)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)

        x2 = x + int(w/2)
        y2 = y + int(h/2)
        cv2.circle(frame, (x2, y2), 4, (255, 0, 0), -1)

        text = "x: " + str(x2) + ", y: " + str(y2)
        cv2.putText(frame, text, (x2 - 10, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        self.publisher_.publish(String(data=text))

        self.publish_image(frame)

    def publish_image(self, frame):
        image_message = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.image_publisher_.publish(image_message)

def main(args=None):
    rclpy.init(args=args)
    object_tracker = ObjectTracker()
    rclpy.spin(object_tracker)
    object_tracker.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()