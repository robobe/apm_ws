import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_services_default
from mavros_msgs.srv import ParamPull, ParamSetV2, ParamGet
from rcl_interfaces.srv import GetParameters, ListParameters

SRV_PARAM_PULL = "/mavros/param/pull"
SRV_PARAM_SET = "/mavros/param/set"
SRV_PARAMS_GET = "/mavros/get_parameters"
SRV_PARAMS_LIST = "/mavros/param/list_parameters"

class MyNode(Node):
    def __init__(self):
        node_name="params_demo"
        super().__init__(node_name)
        # self.__param_pull_srv = self.create_client(ParamPull, SRV_PARAM_PULL)
        # if not self.__param_pull_srv.wait_for_service(1.0):
        #     self.get_logger().error("service not ready")
        
        # req = ParamPull.Request()
        # req.force_pull = False
        # future = self.__param_pull_srv.call_async(req)
        # rclpy.spin_until_future_complete(self, future)
        # print(future.result())


        self.__params_list_srv = self.create_client(ListParameters, SRV_PARAMS_LIST)
        if not self.__params_list_srv.wait_for_service(1.0):
            self.get_logger().error("list parameter")
        req = ListParameters.Request()
        future = self.__params_list_srv.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        print(future.result())
        self.get_logger().info("---------- 1 ------------")
        
        self.__param_get_srv = self.create_client(GetParameters, SRV_PARAMS_GET)
        if not self.__param_get_srv.wait_for_service(1.0):
            self.get_logger().error("get parameter")

        req = GetParameters.Request()
        req.names.append("RNGFND1_FUNCTION")
        future = self.__param_get_srv.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        print(future.result())
        self.get_logger().info("---------- 2 ------------")



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