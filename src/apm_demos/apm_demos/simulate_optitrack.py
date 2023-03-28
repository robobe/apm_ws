import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from mavros_msgs.msg import Mavlink
from gazebo_msgs.srv import GetEntityState

import diagnostic_updater
import diagnostic_msgs

DRONE_NO = 1
TOPIC_MAVLINK_SOURCE = f"/uas{DRONE_NO}/mavlink_source"
SRV_GAZEBO_ENTITY_STATE = "/gazebo/get_entity_state"
TIMER_TICK = 1/10

class DiagHelper:
    def __init__(self, node: Node) -> None:
        self.__updater = diagnostic_updater.Updater(node)
        self.pub1_freq = diagnostic_updater.HeaderlessTopicDiagnostic(
            "topic1",
            self.__updater,
            diagnostic_updater.FrequencyStatusParam({'min':40, 'max':60}, 0.1, 10),
        )

    def update(self) -> None:
        self.__updater.update()

class MyNode(Node):
    def __init__(self) -> None:
        node_name = "optitrack"
        super().__init__(node_name)
        self.__diag = DiagHelper(self)
        self.__secs = 0
        self.__nanosec = 0
        self.create_subscription(
            Mavlink,
            TOPIC_MAVLINK_SOURCE,
            self.__mavlink_handler,
            qos_profile=qos_profile_sensor_data,
        )
        self.entity_state = self.create_client(GetEntityState, SRV_GAZEBO_ENTITY_STATE)
        self.create_timer(TIMER_TICK, self.__tick_handler)
        self.get_logger().info("Hello ROS2")

    def __tick_handler(self):
        msg = GetEntityState.Request()
        msg.name = "iris_demo::iris_demo::iris::base_link"
        msg.reference_frame = "world"
        future = self.entity_state.call_async(msg)
        future.add_done_callback(self.__entity_state_handler)

    def __entity_state_handler(self, future):
        result: GetEntityState.Response = future.result()
        print(result.state.pose.position)


    def __mavlink_handler(self, msg: Mavlink) -> None:
        self.__secs = msg.header.stamp.sec
        self.__nanosec = msg.header.stamp.nanosec
        self.__diag.pub1_freq.tick()
        self.__diag.update()


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
