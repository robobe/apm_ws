import rclpy
from mavros.mavlink import convert_to_bytes
from mavros_msgs.msg import Mavlink
from pymavlink.dialects.v20 import ardupilotmega as apm
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

DRONE_NO = 1
TOPIC_MAVLINK_SOURCE = f"/uas{DRONE_NO}/mavlink_source"


class fifo(object):
    """A simple buffer"""

    def __init__(self):
        self.buf = []

    def write(self, data):
        self.buf += data
        return len(data)

    def read(self):
        return self.buf.pop(0)


class MavReaderNode(Node):
    def __init__(self) -> None:
        node_name = "read_mav"
        super().__init__(node_name)
        f = fifo()
        self.__mav = apm.MAVLink(f, srcSystem=1, srcComponent=1)

        self.create_subscription(
            Mavlink,
            TOPIC_MAVLINK_SOURCE,
            self.__mavlink_handler,
            qos_profile=qos_profile_sensor_data,
        )
        self.get_logger().info("init mavlink reader demo")

    def __mavlink_handler(self, msg: Mavlink) -> None:
        if msg.msgid in [apm.MAVLINK_MSG_ID_HOME_POSITION]:
            self.get_logger().info(str(msg))
            data = convert_to_bytes(msg)
            mav_msg = self.__mav.decode(data)
            self.get_logger().info(str(mav_msg))


def main(args=None):
    rclpy.init(args=args)
    node = MavReaderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("User exit")
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
