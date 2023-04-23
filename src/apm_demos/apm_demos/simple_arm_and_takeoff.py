"""mavros mode,arm,takeoff

RUN: sitl_gazebo.sh
RUN: ros2 run apm_demos arm_and_takeoff
"""
import rclpy
from mavros_msgs.msg import Altitude, State
from mavros_msgs.srv import CommandBool, CommandTOL, MessageInterval, SetMode
from pymavlink.dialects.v20 import ardupilotmega
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
    qos_profile_system_default,
)

TOPIC_STATE = "/mavros/state"
TOPIC_ALTITUDE = "/mavros/altitude"
SERVICE_MODE = "/mavros/set_mode"
SERVICE_ARMING = "/mavros/cmd/arming"
SERVICE_TAKEOFF = "/mavros/cmd/takeoff"
SERVICE_MESSAGE_INTERVAL = "/mavros/set_message_interval"

qos_mavros_policy = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    durability=DurabilityPolicy.VOLATILE,
    depth=1,
)

TAKEOFF_ALT = 5.0


class MyNode(Node):
    def __init__(self):
        node_name = "minimal"
        self.__state = State()
        self.__altitude = Altitude()
        self.__tmp_one_time = True
        super().__init__(node_name)
        self.create_subscription(State, TOPIC_STATE, self.__state_handler, qos_profile=qos_profile_system_default)
        self.create_subscription(Altitude, TOPIC_ALTITUDE, self.__altitude_handler, qos_profile=qos_profile_sensor_data)
        self.__srv_mode = self.create_client(SetMode, SERVICE_MODE)
        self.__srv_arming = self.create_client(CommandBool, SERVICE_ARMING)
        self.__srv_takeoff = self.create_client(CommandTOL, SERVICE_TAKEOFF)
        self.__srv_set_msg_interval = self.create_client(MessageInterval, SERVICE_MESSAGE_INTERVAL)
        self.__srv_set_msg_interval.wait_for_service(timeout_sec=1.0)
        msg = MessageInterval.Request()
        msg.message_id = ardupilotmega.MAVLINK_MSG_ID_SCALED_PRESSURE
        msg.message_rate = 1.0
        self.__srv_set_msg_interval.call_async(msg)
        # self.__srv_mode.wait_for_service()
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.__tick)
        self.get_logger().info("start seq")

    def __tick(self):
        if not self.__state.connected:
            self.get_logger().warning("Drone not connected")
            return

        if not self.__state.guided or self.__state.mode != "GUIDED":
            self.get_logger().info("Set mode to guided")
            msg = SetMode.Request()
            msg.custom_mode = "GUIDED"
            self.__srv_mode.call_async(msg)

        if not self.__state.armed and self.__state.mode == "GUIDED":
            self.get_logger().info("Try to ARM")
            msg = CommandBool.Request()
            msg.value = True
            self.__srv_arming.call_async(msg)

        if self.__state.armed and self.__tmp_one_time:
            self.__tmp_one_time = False
            self.get_logger().info("Try to Takeoff")
            msg = CommandTOL.Request()
            msg.altitude = TAKEOFF_ALT
            self.__srv_takeoff.call_async(msg)

    def __state_handler(self, msg: State):
        self.__state = msg

    def __altitude_handler(self, msg: Altitude):
        print(msg)
        self.__altitude = msg


def main(args=None):
    rclpy.init(args=args)
    node = MyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
