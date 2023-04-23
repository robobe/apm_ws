"""
pymavlink send long command
"""
from pymavlink import mavutil

master = mavutil.mavlink_connection("udp:127.0.0.1:14550")
# Wait a heartbeat before sending commands
master.wait_heartbeat()

ARM = 1
master.mav.command_long_send(
    master.target_system,
    master.target_component,
    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
    0,
    ARM,
    0,
    0,
    0,
    0,
    0,
    0,
)
