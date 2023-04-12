import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Range

TOPIC_DISTANCE_READ = "/mavros/rangefinder_pub"
TOPIC_DISTANCE_WRITE = "/mavros/rangefinder_sub"
TOPIC_APM_RANGEFINDER = "/mavros/rangefinder/rangefinder"
DRONE_NO = 1
MIN_RANGE = 0.0
MAX_RANGE = 4.0
RANGE_SENSOR_TYPE = 1
SENSOR_ID = 1
COVARIANCE = 0

PUB_INTERVAL = 1 / 10


class RangeFinderNode(Node):
    def __init__(self):
        node_name = "range_finder"
        super().__init__(node_name)

        # mavros open subscriber, our node pub to it
        self.__range_pub = self.create_publisher(Range, TOPIC_DISTANCE_WRITE, qos_profile=qos_profile_sensor_data)
        self.create_subscription(
            Range, TOPIC_APM_RANGEFINDER, self.__apm_rangefinder_message_handler, qos_profile=qos_profile_sensor_data
        )

        self.create_timer(PUB_INTERVAL, self.__send_range_message)

    def __apm_rangefinder_message_handler(self, msg: Range):
        self.get_logger().info(f"apm rangefinder: {msg.range}")

    def __send_range_message(self, distance=2.0):
        sec, nanosec = self.get_clock().now().seconds_nanoseconds()
        range_msg = Range()
        range_msg.header.frame_id = "rangefinder"
        range_msg.header.stamp.sec = sec
        range_msg.header.stamp.nanosec = nanosec
        range_msg.range = float(distance)
        range_msg.radiation_type = RANGE_SENSOR_TYPE
        range_msg.min_range = MIN_RANGE
        range_msg.max_range = MAX_RANGE
        self.__range_pub.publish(range_msg)
        print("--")


def main(args=None):
    rclpy.init(args=args)
    node = RangeFinderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("User exit")
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
