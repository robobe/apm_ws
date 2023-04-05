import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts
from example_interfaces.srv._add_two_ints import AddTwoInts_Response, AddTwoInts_Request

import time

class MyNode(Node):
    def __init__(self):
        node_name="service_demo"
        super().__init__(node_name)
        self.srv = self.create_service(AddTwoInts, '/add_two_ints', self.add_two_ints_callback)
        self.get_logger().info("start service")

    def add_two_ints_callback(self, request: AddTwoInts_Request, response: AddTwoInts_Response):
        self.get_logger().info("start service callback")
        # time.sleep(3)
        # response.sum = request.a + request.b
        # self.get_logger().info('Incoming request\na: %d b: %d' % (request.a, request.b))
        raise Exception("00000000000000000")
        return response


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