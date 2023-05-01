"""
GLOBAL_VISION_POSITION_ESTIMATE ( #101 ) (no mavros implementation)
VISION_POSITION_ESTIMATE ( #102 )
VISION_SPEED_ESTIMATE ( #103 )
mavros: vision_pose (mavros_extras)
/pose: geometry_msgs::msg::PoseStamped
/pose_cov: geometry_msgs::msg::PoseWithCovarianceStamped

std_msgs/Header header
	builtin_interfaces/Time stamp
		int32 sec
		uint32 nanosec
	string frame_id
Pose pose
	Point position
		float64 x
		float64 y
		float64 z
	Quaternion orientation
		float64 x 0
		float64 y 0
		float64 z 0
		float64 w 1

[VISION_POSITION_ESTIMATE](http://mavlink.io/en/messages/common.html#VISION_POSITION_ESTIMATE)
usec	uint64_t	    us	Timestamp (UNIX time or time since system boot)
x	    float	m	    Local X position
y	    float	m	    Local Y position
z	    float	m	    Local Z position
roll	float	rad	    Roll angle
pitch	float	rad	    Pitch angle
yaw	    float	rad	    Yaw angle
"""
# from geometry_msgs.msg import Vector3Stamped  # gps_vel
import math
import time
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import rclpy
import rclpy.clock
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import PoseStamped
from mavros_msgs.srv import CommandBool, CommandTOL, MessageInterval, SetMode
from pymavlink.dialects.v20 import common
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
    qos_profile_system_default,
)
from sensor_msgs.msg import NavSatFix  # gps_pos
from sensor_msgs.msg import Imu, TimeReference
from std_msgs.msg import Header

EARTH_RADIUS = 6371000
TOPIC_GPS_POS = "/gps_pos"
TOPIC_IMU = "/imu/data"
TOPIC_TIME_REFERENCE = "/mavros/time_reference"
TOPIC_VISION_ESTIMATE = "/mavros/vision_pose/pose"
SERVICE_MESSAGE_INTERVAL = "/mavros/set_message_interval"

LAT = 31.0461
LON = 34.8516


@dataclass
class PoseAndOrientation:
    boot_time: float = 0
    lat: float = 0
    lon: float = 0
    alt: float = 0
    roll: float = 0
    pitch: float = 0
    yaw: float = 0

    def to_xy(self):
        cos_phi_0 = np.cos(np.radians(self.lon))
        x = EARTH_RADIUS * np.radians(self.lat) * cos_phi_0
        y = EARTH_RADIUS * np.radians(self.lon)

        return x, y

    def delta_xy(self, other) -> Tuple[float, float]:
        x, y = self.to_xy()
        x1, y1 = other.to_xy()

        return x - x1, y - y1

    def distance(self, other) -> float:
        dx, dy = self.delta_xy(other)
        dist = np.sqrt(dx**2 + dy**2)
        # Convert to km
        dist = dist / 1000
        return dist


def quaternion_from_euler(roll, pitch, yaw):
    """
    Converts euler roll, pitch, yaw to quaternion (w in last place)
    quat = [x, y, z, w]
    Bellow should be replaced when porting for ROS 2 Python tf_conversions is done.
    """
    # pylint: disable=invalid-name
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    q = [0] * 4
    q[0] = cy * cp * cr + sy * sp * sr
    q[1] = cy * cp * sr - sy * sp * cr
    q[2] = sy * cp * sr + cy * sp * cr
    q[3] = sy * cp * cr - cy * sp * sr

    return q


def euler_from_quaternion(quaternion):
    """
    Converts quaternion (w in last place) to euler roll, pitch, yaw
    quaternion = [x, y, z, w]
    Bellow should be replaced when porting for ROS 2 Python tf_conversions is done.
    """
    # pylint: disable=invalid-name

    x = quaternion.x
    y = quaternion.y
    z = quaternion.z
    w = quaternion.w

    sin_r_cos_p = 2 * (w * x + y * z)
    cos_r_cos_p = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sin_r_cos_p, cos_r_cos_p)

    sin_p = 2 * (w * y - z * x)
    pitch = np.arcsin(sin_p)

    sin_y_cos_p = 2 * (w * z + x * y)
    cos_y_cos_p = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(sin_y_cos_p, cos_y_cos_p)

    return roll, pitch, yaw


class MyNode(Node):
    def __init__(self):
        node_name = "vision_estimation"
        super().__init__(node_name)
        self.__pose_and_orientation = PoseAndOrientation()
        self.__ekf_origin = PoseAndOrientation(lat=LAT, lon=LON, alt=0)
        self.__time_sync_delta = time.time()
        self.__vision_estimate_pub = self.create_publisher(
            PoseStamped, TOPIC_VISION_ESTIMATE, qos_profile=qos_profile_system_default
        )
        self.create_subscription(NavSatFix, TOPIC_GPS_POS, self.__gps_pos_handler, qos_profile=qos_profile_sensor_data)
        self.create_subscription(Imu, TOPIC_IMU, self.__imu_handler, qos_profile=qos_profile_sensor_data)
        self.create_subscription(
            TimeReference, TOPIC_TIME_REFERENCE, self.__system_time_handler, qos_profile=qos_profile_sensor_data
        )
        self.__set_messages_intervals()
        self.get_logger().info("Hello ROS2")

    def __set_messages_intervals(self):
        self.__srv_set_msg_interval = self.create_client(MessageInterval, SERVICE_MESSAGE_INTERVAL)
        self.__srv_set_msg_interval.wait_for_service(timeout_sec=1.0)
        req = MessageInterval.Request()
        req.message_id = common.MAVLINK_MSG_ID_SYSTEM_TIME
        req.message_rate = 1.0
        future = self.__srv_set_msg_interval.call_async(req)
        rclpy.spin_until_future_complete(self, future)

    def __pub_viso_estimate(self, dx, dy, d_alt):
        msg = PoseStamped()
        stamp = TimeMsg()
        # stamp.sec, stamp.nanosec = rclpy.clock.Clock().now().seconds_nanoseconds()
        delta = time.time() - self.__time_sync_delta
        nanosec, sec = math.modf(delta)
        print(sec, nanosec, delta)
        stamp.sec = int(sec)
        stamp.nanosec = int(nanosec * 1e9)
        # stamp.nanosec = nanosec
        header = Header(stamp=stamp)
        msg.header = header
        msg.pose.position.x = dx
        msg.pose.position.y = dy
        msg.pose.position.z = d_alt
        msg.pose.orientation.x = float(self.__pose_and_orientation.roll)
        msg.pose.orientation.y = float(self.__pose_and_orientation.pitch)
        msg.pose.orientation.z = float(self.__pose_and_orientation.yaw)
        self.__vision_estimate_pub.publish(msg)

    def __system_time_handler(self, msg: TimeReference):
        self.__time_sync_delta = time.time() - (msg.time_ref.sec + msg.time_ref.nanosec / 1e9)

    def __imu_handler(self, msg: Imu):
        (
            self.__pose_and_orientation.roll,
            self.__pose_and_orientation.pitch,
            self.__pose_and_orientation.yaw,
        ) = euler_from_quaternion(msg.orientation)

    def __gps_pos_handler(self, msg: NavSatFix):
        current = PoseAndOrientation(lat=msg.latitude, lon=msg.longitude, alt=msg.altitude)

        dx, dy = current.delta_xy(self.__ekf_origin)
        d_alt = current.alt - self.__ekf_origin.alt
        self.__pub_viso_estimate(dx, dy, d_alt)


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
