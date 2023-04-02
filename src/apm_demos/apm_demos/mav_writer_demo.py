import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time as ROSTime
from mavros_msgs.msg import Mavlink
from builtin_interfaces.msg import Time
from pymavlink.dialects.v10 import ardupilotmega
from mavros.mavlink import convert_to_rosmsg
from pymavlink import mavutil
import time as py_time

DRONE_NO = 1
TOPIC_MAVLINK = f"/uas{DRONE_NO}/mavlink_sink"

class MyNode(Node):
    def __init__(self) -> None:
        node_name = "mav_writer"
        super().__init__(node_name)
        self.__secs = 0
        self.__nanosec = 0
        self.conn = mavutil.mavlink_connection("udp:127.0.0.1:14551")
        self.create_publisher(
            Mavlink,
            TOPIC_MAVLINK,
            qos_profile=qos_profile_sensor_data,
        )
        self.get_logger().info("init mavlink write demo")
        self.set_home()

    def set_home(self) -> None:
        target_system = 0  # broadcast to everyone

        lattitude = int(41.6996 * 1e7)
        longitude = int(-86.237177 * 1e7)
        altitude = int(200 * 1e3)
        
        x = 0
        y = 0
        z = 0
        q = [1, 0, 0, 0]   # w x y z

        approach_x = 0
        approach_y = 0
        approach_z = 1

        ros_time = ROSTime()
        time = Time()
        time.sec, time.nanosec = ros_time.seconds_nanoseconds()

        msg = ardupilotmega.MAVLink_set_home_position_message(
                target_system,
                lattitude,
                longitude,
                altitude,
                x,
                y,
                z,
                q,
                approach_x,
                approach_y,
                approach_z)
        
        msg.pack(self.conn.mav)
        
        
        payload_bytes = msg.get_payload()
        
        payload_bytes = bytearray(payload_bytes)
        payload_len = len(payload_bytes)
        print(len(payload_bytes))
        # payload_octets = payload_len / 8
        # if payload_len % 8 > 0:
        #     payload_octets += 1
        #     payload_bytes += b"\0" * (8 - payload_len % 8)
        # import struct
        # print(int(payload_octets))
        # data = struct.unpack(f"<{7}Q", payload_bytes)
        raw = convert_to_rosmsg(msg, stamp=time)



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
