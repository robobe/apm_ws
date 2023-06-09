import time
from typing import Any, List

import rclpy
from builtin_interfaces.msg import Time
from mavros.mavlink import convert_to_rosmsg
from mavros_msgs.msg import Mavlink
from pymavlink.dialects.v20 import ardupilotmega as MAV_APM
from rclpy.node import Node

# Global position of the origin
LAT = int(41.6996 * 1e7)
LON = int(-86.237177 * 1e7)
ALT = int(200 * 1e3)

TOPIC_MAVLINK_TO = "/uas1/mavlink_sink"


class fifo:
    def __init__(self) -> None:
        self.buf: List[Any] = []

    def write(self, data):
        self.buf += data
        return len(data)

    def read(self):
        return self.buf.pop(0)


class MyNode(Node):
    def __init__(self):
        node_name = "set_origin"
        super().__init__(node_name)
        self.mavlink_pub = self.create_publisher(Mavlink, TOPIC_MAVLINK_TO, 10)
        self.timer = self.create_timer(1.0, self.__timer_handler)
        self.counter = 0
        f = fifo()
        self.mav = MAV_APM.MAVLink(f, srcSystem=1, srcComponent=1)
        while self.mavlink_pub.get_subscription_count() <= 0:
            self.get_logger().warning("init mavlink")
            time.sleep(1)
            pass
        self.get_logger().info("init set origen")

    def send_global_origin(self):
        """
        Send a mavlink SET_GPS_GLOBAL_ORIGIN message, which allows us
        to use local position information without a GPS.
        """
        # target_system = mav.srcSystem
        target_system = 0  # 0 --> broadcast to everyone
        lattitude = LAT
        longitude = LON
        altitude = ALT

        msg = MAV_APM.MAVLink_set_gps_global_origin_message(target_system, lattitude, longitude, altitude)
        self.send_message(msg)

    def __timer_handler(self):
        self.counter += 1
        if self.counter > 5:
            self.timer.cancel()

        self.send_global_origin()
        # self.send_home_position()
        self.get_logger().info(f"Timer tick {self.counter}")

    def send_home_position(self):
        """
        Send a mavlink SET_HOME_POSITION message, which should allow
        us to use local position information without a GPS
        """
        target_system = 0  # broadcast to everyone

        lattitude = LAT
        longitude = LON
        altitude = ALT

        x = 0
        y = 0
        z = 0
        q = [1, 0, 0, 0]  # w x y z

        approach_x = 0
        approach_y = 0
        approach_z = 1

        msg = MAV_APM.MAVLink_set_home_position_message(
            target_system, lattitude, longitude, altitude, x, y, z, q, approach_x, approach_y, approach_z
        )
        self.send_message(msg)

    def send_message(self, msg):
        msg.pack(self.mav)
        # ros_time = ROSTime()
        t = Time()
        t.sec = 0
        t.nanosec = 1
        # t.sec, t.nanosec = ros_time.seconds_nanoseconds()
        rosmsg = convert_to_rosmsg(msg, stamp=t)

        self.mavlink_pub.publish(rosmsg)


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
