# https://github.com/vincekurtz/ardupilot_gazebo/blob/master/src/pose_republisher.py
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import Mavlink
from mavros.mavlink import convert_to_bytes
from pymavlink.dialects.v20 import ardupilotmega as MAV_APM
from pymavlink.dialects.v20.ardupilotmega import MAVLink_message
import time

TOPIC_MAVLINK_TO = "/uas1/mavlink_sink"
TOPIC_MAVLINK_FROM = "/uas1/mavlink_source"


class fifo:
    def __init__(self) -> None:
        self.buf = []

    def write(self, data):
        self.buf += data
        return len(data)

    def read(self):
        return self.buf.pop(0)


class MyNode(Node):
    def __init__(self):
        node_name = "pose_republsher"
        super().__init__(node_name)
        self.secs = None
        self.nsecs = None
        self.param_in_topic = self.declare_parameter("in_topic", value="/Robot_1/pose")
        self.param_out_topic = self.declare_parameter(
            "out_topic", value="/mavros/mocap/pose"
        )
        self.in_topic = self.param_in_topic.value
        self.out_topic = self.param_out_topic.value

        self.mavlink_pub = self.create_publisher(Mavlink, TOPIC_MAVLINK_TO, 10)
        self.pose_pub = self.create_publisher(PoseStamped, self.out_topic, 10)
        self.mav_time_sub = self.create_subscription(
            Mavlink,
            TOPIC_MAVLINK_FROM,
            self.__mav_time_handler,
            qos_profile=qos_profile_sensor_data,
        )
        self.pose_sub = self.create_subscription(
            PoseStamped,
            self.in_topic,
            self.__in_pose_hanlder,
            qos_profile=qos_profile_sensor_data,
        )
        f = fifo()
        self.mav = MAV_APM.MAVLink(f, srcSystem=1, srcComponent=1)

        while self.mavlink_pub.get_subscription_count() <= 0:
            self.get_logger().warning("init mavlink")
            time.sleep(1)
            pass

        self.get_logger().info("init node")

    def __mav_time_handler(self, msg: Mavlink):
        b = convert_to_bytes(msg)
        m: MAVLink_message
        m = self.mav.decode(b)

        if m.get_msgId() == MAV_APM.MAVLINK_MSG_ID_TIMESYNC:
            # my code use ts1 and not tc1 ??
            t = Time(nanoseconds=m.ts1)
            self.secs, self.nsecs = t.seconds_nanoseconds()

    def __in_pose_hanlder(self, msg: PoseStamped):
        delay_ms = 0
        delay_ns = delay_ms * 1e6

        if all([self.secs, self.nsecs]):
            msg.header.stamp.sec = self.secs
            msg.header.stamp.nanosec = self.nsecs + delay_ns

            self.pose_pub.publish(msg)

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
