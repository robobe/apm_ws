import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from mavros_msgs.msg import Mavlink

DRONE_NO = 1
TOPIC_MAVLINK_SOURCE = f"/uas{DRONE_NO}/mavlink_source"

class MyNode(Node):
    def __init__(self) -> None:
        node_name = "read_mav"
        super().__init__(node_name)
        self.__secs = 0
        self.__nanosec = 0
        self.create_subscription(
            Mavlink,
            TOPIC_MAVLINK_SOURCE,
            self.__mavlink_handler,
            qos_profile=qos_profile_sensor_data,
        )
        self.get_logger().info("init mavlink reader demo")

    def __mavlink_handler(self, msg: Mavlink) -> None:
        self.__secs = msg.header.stamp.sec
        self.__nanosec = msg.header.stamp.nanosec
        print(type(msg.header.stamp))


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
