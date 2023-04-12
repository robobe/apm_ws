import ctypes

# from .utils import Fifo
import os
import time

import rclpy
from builtin_interfaces.msg import Time as HTime
from mavros.mavlink import convert_to_bytes, convert_to_rosmsg
from mavros_msgs.msg import Mavlink
from pymavlink.dialects.v20 import ardupilotmega, common
from rclpy.clock import Clock
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

DRONE_NO = 1
TOPIC_MAVLINK = f"/uas{DRONE_NO}/mavlink_sink"
TOPIC_MAVLINK_SOURCE = f"/uas{DRONE_NO}/mavlink_source"

CLOCK_MONOTONIC_RAW = 4


class Fifo:
    def __init__(self):
        self.buf = []

    def write(self, data):
        self.buf += data
        return len(data)

    def read(self):
        return self.buf.pop(0)


class timespec(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]


librt = ctypes.CDLL("librt.so.1", use_errno=True)
clock_gettime = librt.clock_gettime
clock_gettime.argtypes = [ctypes.c_int, ctypes.POINTER(timespec)]


def monotonic_time():
    t = timespec()
    if clock_gettime(CLOCK_MONOTONIC_RAW, ctypes.pointer(t)) != 0:
        errno_ = ctypes.get_errno()
        raise OSError(errno_, os.strerror(errno_))
    return (t.tv_sec * 1e9) + t.tv_nsec


class MyNode(Node):
    def __init__(self):
        node_name = "time_sync_demo"
        super().__init__(node_name)
        self.__ref_time = time.time()
        buffer = Fifo()
        self.__mav = common.MAVLink(buffer, srcSystem=1, srcComponent=1)
        self.__pub_mavlink = self.create_publisher(
            Mavlink,
            TOPIC_MAVLINK,
            qos_profile=qos_profile_sensor_data,
        )
        self.create_subscription(
            Mavlink,
            TOPIC_MAVLINK_SOURCE,
            self.__mavlink_handler,
            qos_profile=qos_profile_sensor_data,
        )
        self.create_timer(1 / 2, self.send_timesync_long)
        # self.create_timer(1/10, self.send_sys_time)

    def __mavlink_handler(self, msg: Mavlink):
        if msg.msgid == common.MAVLINK_MSG_ID_TIMESYNC:
            data = convert_to_bytes(msg)
            mav_msg = self.__mav.decode(data)
            self.get_logger().info(str(mav_msg))

    def send_sys_time(self):
        current = time.time()
        boot = int((current - self.__ref_time) * 1e3)
        usec = int(current * 1e6)
        msg = common.MAVLink_system_time_message(usec, boot)
        msg.pack(self.__mav)
        ros_msg = convert_to_rosmsg(msg)
        self.__pub_mavlink.publish(ros_msg)
        self.get_logger().info("---")

    def send_timesync(self) -> None:
        """
        https://mavlink.io/en/services/timesync.html
        https://mavlink.io/en/messages/common.html#TIMESYNC
        tc1 int64 ns
        ts1 int64 ns
        """
        tc1 = 0
        # ts1 = int(Clock().now().nanoseconds/1e3)
        ts1 = int(time.time() * 1e6)  # int(monotonic_time())
        print(ts1)
        msg = common.MAVLink_timesync_message(tc1, ts1)
        msg.pack(self.__mav)
        ros_msg = convert_to_rosmsg(msg)
        self.__pub_mavlink.publish(ros_msg)

    def send_timesync_long(self):
        tc1 = 0
        # ts1 = int(Clock().now().nanoseconds/1e3)
        ts1 = int(time.time() * 1e6)  # int(monotonic_time())
        target_system = 0  # broadcast to everyone
        msg = ardupilotmega.MAVLink_command_long_message(
            target_system, 1, ardupilotmega.MAVLINK_MSG_ID_TIMESYNC, 0, tc1, ts1, 0, 0, 0, 0, 0
        )

        msg.pack(self.__mav)
        current = Clock().now()
        sec, nanosec = current.seconds_nanoseconds()
        stamp = HTime(sec=sec, nanosec=nanosec)
        ros_msg = convert_to_rosmsg(msg, stamp=stamp)
        self.__pub_mavlink.publish(ros_msg)
        self.get_logger().info("---")


def main(args=None):
    rclpy.init(args=args)
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
