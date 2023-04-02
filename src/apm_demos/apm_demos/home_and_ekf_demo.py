"""
https://ardupilot.org/dev/docs/mavlink-get-set-home-and-origin.html
- COMMAND_LONG
- COMMAND_INT
- MAV_CMD_DO_SET_HOME (179)
- https://mavlink.io/en/messages/common.html#MAV_CMD_DO_SET_HOME
- SET_GPS_GLOBAL_ORIGIN (48)
- https://mavlink.io/en/messages/common.html#SET_GPS_GLOBAL_ORIGIN
- MAV_CMD_GET_HOME_POSITION (410)
- MAV_CMD_REQUEST_MESSAGE (512)
- HOME_POSITION (242)
- GPS_GLOBAL_ORIGIN (49)

/mavros/cmd/command
/mavros/cmd/command_int

ros2 service call /mavros/cmd/arming mavros_msgs/srv/CommandBool "{value: True}"
ros2 service call /mavros/cmd/command mavros_msgs/srv/CommandLong "{command: 410, param1: 0, param2: 0, param3: 0, param4: 0}"

mavlink command protocol (micro service)
"""
import time
import rclpy
from rclpy.clock import Clock
from rclpy.node import Node
from rclpy.task import Future
from rclpy.qos import qos_profile_system_default
from mavros_msgs.srv import CommandLong, CommandInt
from mavros_msgs.msg import HomePosition
from pymavlink.dialects.v20 import ardupilotmega
from pymavlink.dialects.v20 import common as mav_common
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from enum import IntEnum
from geographic_msgs.msg import GeoPointStamped
from builtin_interfaces.msg import Time
from std_msgs.msg import Header

LAT = 32.606
LON = 35.505
ALT = 230.0

class Location_use_type(IntEnum):
    use_specified_location = 0
    use_current_location = 1

SRV_LONG_COMMAND = "/mavros/cmd/command"
SRV_INT_COMMAND = "/mavros/cmd/command_int"
TOPIC_HOME_POSITION = "/mavros/home_position/home"
TOPIC_SET_EKF_ORIGIN = "/mavros/global_position/set_gp_origin"

class MyNode(Node):
    def __init__(self) -> None:
        node_name="home_and_ekf"
        super().__init__(node_name)
        call_group = ReentrantCallbackGroup()
        self.__long_cmd_srv = self.create_client(CommandLong, SRV_LONG_COMMAND, callback_group=call_group)
        self.create_subscription(HomePosition, TOPIC_HOME_POSITION, self.__home_position_handler, qos_profile=qos_profile_system_default)
        self.__pub_ekf_origin = self.create_publisher(GeoPointStamped, TOPIC_SET_EKF_ORIGIN, qos_profile=qos_profile_system_default)
        self.__int_cmd_srv = self.create_client(CommandInt, SRV_INT_COMMAND)
        if not self.__long_cmd_srv.wait_for_service(2.0):
            self.get_logger().error("Server not ready")
        if not self.__int_cmd_srv.wait_for_service(2.0):
            self.get_logger().error("Command int service not ready")
        self.__send_set_ekf_origin()
        self.get_logger().info("wait -------------------------")
        # self.__send_set_home_long()
        self.__send_set_home()
        # self.__send_get_home_request()
        
    def __send_set_ekf_origin(self):
        msg = GeoPointStamped()
        sec, nano_sec = Clock().now().seconds_nanoseconds()
        msg.header.stamp.sec = sec
        msg.header.stamp.nanosec = nano_sec
        print(sec, nano_sec)
        msg.header.frame_id = ""
        msg.position.altitude = ALT
        msg.position.latitude = LAT
        msg.position.longitude = LON
        self.__pub_ekf_origin.publish(msg)
        self.get_logger().info("Run set ekf")


    def __send_set_home_long(self) -> None:
        req = CommandLong.Request()
        req.command = mav_common.MAV_CMD_DO_SET_HOME
        req.param1 = float(Location_use_type.use_specified_location)
        req.param5 = LAT
        req.param6 = LON
        req.param7 = ALT
        future = self.__long_cmd_srv.call_async(req)
        future.add_done_callback(self.__long_command_handler)

    def __home_position_handler(self, msg: HomePosition):
        self.get_logger().info(str(msg))

    def __send_get_home_request(self) -> None:
        msg = CommandLong.Request()
        
        msg.command = mav_common.MAV_CMD_GET_HOME_POSITION
        msg.broadcast = False
        msg.param1 = 0.0
        msg.param2 = 0.0
        msg.param3 = 0.0
        msg.param4 = 0.0
        future = self.__long_cmd_srv.call_async(msg)
        future.add_done_callback(self.__long_command_handler)
        self.get_logger().info("run service")

    def __long_command_handler(self, future: Future) -> None:
        self.get_logger().info("service return")
        print(future.result())
        

    def __int_command_handler(self, future: Future) -> None:
        self.get_logger().info("int service return")
        print(future.result())
        self.__send_get_home_request()

    def __send_set_home(self) -> None:
        req = CommandInt.Request()
        req.frame = mav_common.MAV_FRAME_GLOBAL
        req.command = mav_common.MAV_CMD_DO_SET_HOME
        req.param1 = float(Location_use_type.use_specified_location)
        req.x = int(LAT * 1e7)
        req.y = int(LON * 1e7)
        req.z = ALT
        future = self.__int_cmd_srv.call_async(req)
        future.add_done_callback(self.__int_command_handler)


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

if __name__ == '__main__':
    main()