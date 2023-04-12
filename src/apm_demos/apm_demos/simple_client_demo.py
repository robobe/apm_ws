import rclpy
from example_interfaces.srv import AddTwoInts
from rclpy.node import Node


class MyNode(Node):
    def __init__(self):
        node_name = "simple_client"
        super().__init__(node_name)
        self.cli = self.create_client(AddTwoInts, "/add_two_ints")
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("service not available, waiting again...")
        self.get_logger().info("Start node call service")
        self.create_timer(1 / 2, self.__timer_handler)
        self.send_request(1, 2)

    def __timer_handler(self):
        self.get_logger().info("timer cb")

    def send_request(self, a, b):
        self.req = AddTwoInts.Request()
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future, timeout_sec=1)
        resp = self.future.result()
        self.get_logger().info(f"response / result {resp}")


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
