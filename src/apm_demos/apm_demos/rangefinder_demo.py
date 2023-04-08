import time
import rclpy
from rclpy.time import Time
from rclpy.clock import Clock
from builtin_interfaces.msg import Time as HTime
from rclpy.node import Node
from rclpy.qos import (qos_profile_sensor_data,
    qos_profile_system_default,
    QoSProfile,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy
    )
from sensor_msgs.msg import (Range, TimeReference, LaserScan)
from mavros_msgs.msg import Mavlink
from mavros.mavlink import convert_to_rosmsg, convert_to_bytes
from pymavlink.dialects.v20 import ardupilotmega as apm

TOPIC_RANGEFINDER = "/ultrasonic_sensor/out"
TOPIC_MAVLINK_DISTANCE_READ = "/mavros/rangefinder_pub"
TOPIC_MAVLINK_DISTANCE_WRITE = "mavros/rangefinder_sub"
TOPIC_APM_RANGEFINDER = "/mavros/rangefinder/rangefinder"
DRONE_NO = 1
TOPIC_MAVLINK_SOURCE = f"/uas{DRONE_NO}/mavlink_source"
TOPIC_MAVLINK_SINK = f"/uas{DRONE_NO}/mavlink_sink"
TOPIC_TIME_REFERENCE = "/mavros/time_reference"
MIN_RANGE = 0.0
MAX_RANGE = 5.0
RANGE_SENSOR_TYPE = 1
SENSOR_ID = 1
COVARIANCE = 0

class fifo(object):
    """ A simple buffer """
    def __init__(self):
        self.buf = []
    def write(self, data):
        self.buf += data
        return len(data)
    def read(self):
        return self.buf.pop(0)
    

class RangeFinderNode(Node):
    def __init__(self):
        node_name="range_finder"
        super().__init__(node_name)
        # qos_profile = QoSProfile(depth=10)
        # qos_profile.reliability = QoSReliabilityPolicy.BEST_EFFORT
        # qos_profile.durability = QoSDurabilityPolicy.VOLATILE
        self.__last_msg = None
        self.__ref_time = time.time()
        self.__boot_time = 0
        f = fifo()
        self.__mav = apm.MAVLink(f, srcSystem=1, srcComponent=1)
        self.create_subscription(LaserScan, TOPIC_RANGEFINDER, self.__laser_reading_handler, qos_profile=qos_profile_system_default)
        self.__range_pub = self.create_publisher(Range, TOPIC_MAVLINK_DISTANCE_WRITE, qos_profile=qos_profile_sensor_data)
        self.create_subscription(Range, TOPIC_APM_RANGEFINDER, self.__apm_rangefinder_message_handler, qos_profile=qos_profile_sensor_data)
        self.create_subscription(
            Mavlink,
            TOPIC_MAVLINK_SOURCE,
            self.__mavlink_handler,
            qos_profile=qos_profile_sensor_data,
        )
        self.__pub_mavlink = self.create_publisher(Mavlink, TOPIC_MAVLINK_SINK, qos_profile=qos_profile_system_default)
        # self.create_subscription(TimeReference, TOPIC_TIME_REFERENCE, self.__time_reference_handler, qos_profile=qos_profile_sensor_data)

    def __time_reference_handler(self, msg: TimeReference):
        current = Time(seconds=msg.time_ref.sec, nanoseconds=msg.time_ref.nanosec)
        print(current)

    def __mavlink_handler(self, msg: Mavlink):
        if msg.msgid in [apm.MAVLINK_MSG_ID_RANGEFINDER, 
                         apm.MAVLINK_MSG_ID_DISTANCE_SENSOR]:
            data = convert_to_bytes(msg)
            mav_msg = self.__mav.decode(data)
            self.get_logger().info(str(mav_msg))
        if msg.msgid in [apm.MAVLINK_MSG_ID_SYSTEM_TIME]:
            data = convert_to_bytes(msg)
            mav_msg = self.__mav.decode(data)
            self.__boot_time = mav_msg.time_boot_ms

    def __apm_rangefinder_message_handler(self, msg: Range):
        """
        sensor_msgs.msg.Range(header=std_msgs.msg.Header(stamp=builtin_interfaces.msg.Time(sec=1680845505, nanosec=755051282), frame_id='/rangefinder'), radiation_type=1, field_of_view=0.0, min_range=0.0, max_range=1000.0, range=0.0)
        """
        self.__last_msg = msg
        self.get_logger().info(f"apm rangefinder: {msg.range}")

    
    def __laser_reading_handler(self, msg: LaserScan):
        middle_index = int(len(msg.ranges) / 2)
        distance_m = msg.ranges[middle_index]
        self.get_logger().info(f"laser reading: {distance_m}")
        self.__send_mavlink_distance_message(distance_m)
        
    def __send_range_message(self, distance):
        time = self.get_clock().now()
        sec, nanosec = time.seconds_nanoseconds()
        range_msg = Range()
        range_msg.header.frame_id = "rangefinder"
        range_msg.header.stamp.sec = sec
        range_msg.header.stamp.nanosec = nanosec
        range_msg.range = distance
        range_msg.radiation_type = RANGE_SENSOR_TYPE
        range_msg.min_range = MIN_RANGE
        range_msg.max_range = MAX_RANGE
        self.__range_pub.publish(range_msg)

    
    def __get_time_boot_ms(self, msg: HTime):
        t = Time.from_msg(msg)
        boot_time = t.nanoseconds / 1e6
        print(boot_time)
        return boot_time
    
    def __send_mavlink_distance_message(self, distance):
        if not self.__last_msg:
            return
        boot = self.__get_time_boot_ms(self.__last_msg.header.stamp)
        msg = apm.MAVLink_distance_sensor_message(
            time_boot_ms=int(boot),
            min_distance=int(MIN_RANGE*1e2),
            max_distance=int(MAX_RANGE*1e2),
            current_distance=int(distance*1e2),
            type=RANGE_SENSOR_TYPE,
            id=SENSOR_ID,
            orientation=apm.MAV_SENSOR_ROTATION_PITCH_270,
            covariance=COVARIANCE
        )
        msg.pack(self.__mav)
        current = Clock().now()
        sec, nanosec = current.seconds_nanoseconds()
        stamp = HTime(sec=sec, nanosec=nanosec)
        ros_mag = convert_to_rosmsg(msg, stamp=stamp)
        self.__pub_mavlink.publish(ros_mag)

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

if __name__ == '__main__':
    main()