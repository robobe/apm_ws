"""
module doctring
"""
from typing import List

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


def foo() -> None:
    """_summary_"""


def hello(a: List, b: int = 0) -> bool:
    """_summary_

    Args:
        a (List): _description_
        b (int, optional): _description_. Defaults to 0.

    Raises:
        Exception: _description_

    Returns:
        bool: _description_
    """
    print(a)
    print(b)
    if b:
        raise Exception("new demo exception")
    return True
