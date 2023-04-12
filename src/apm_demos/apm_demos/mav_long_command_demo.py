import rclpy
from mavros_msgs.srv import CommandLong
from pymavlink.dialects.v20 import common as mav_common
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node

SRV_LONG_COMMAND = "/mavros/cmd/command"
ARM = 1


class MyNode(Node):
    def __init__(self):
        node_name = "minimal"
        super().__init__(node_name)
        call_group = ReentrantCallbackGroup()
        self.__long_cmd_srv = self.create_client(CommandLong, SRV_LONG_COMMAND, callback_group=call_group)
        if not self.__long_cmd_srv.wait_for_service(2.0):
            self.get_logger().error("Server not ready")
        self.__send_set_home_long()

    def __send_set_home_long(self) -> None:
        req = CommandLong.Request()
        req.command = mav_common.MAV_CMD_COMPONENT_ARM_DISARM
        req.param1 = float(ARM)
        req.param2 = 0.0
        req.param3 = 0.0
        req.param4 = 0.0
        future = self.__long_cmd_srv.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        print(future.result())


def main(args=None):
    rclpy.init(args=args)
    node = MyNode()
    try:
        rclpy.spin_once(node)
    except KeyboardInterrupt:
        print("User exit")
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
