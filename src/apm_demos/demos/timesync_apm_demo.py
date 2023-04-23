"""TIMESYNC demo
[mavlink.io](https://mavlink.io/en/messages/common.html#TIMESYNC)


"""
import logging
import time

from pymavlink import mavutil
from pymavlink.dialects.v20 import ardupilotmega

FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(format=FMT, level=logging.INFO)
log = logging.getLogger(__name__)

# Create the connection
# master = mavutil.mavlink_connection("udp:127.0.0.1:14550")
master = mavutil.mavlink_connection("tcp:0.0.0.0:5760")

# Wait a heartbeat before sending commands
master.wait_heartbeat()


def send_timesync_response(ts1):
    tc1 = int(time.time() * 1e9)
    master.mav.timesync_send(tc1, ts1)
    print("--send--")


def send_timesync_request(ts1=0):
    tc1 = 0
    if not ts1:
        ts1 = int(time.time() * 1e9)
    master.mav.timesync_send(tc1, ts1)
    log.info("Send timesync request %d", ts1)


# def send_timesync_request_loop():
#     while True:
#         send_timesync_request()
#         time.sleep(1/1)

# t = Thread(target=send_timesync_request_loop, daemon=True)
# t.start()

ONE_SEC = 1e6


def set_message_interval(interval_us):
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        ardupilotmega.MAV_CMD_SET_MESSAGE_INTERVAL,
        0,
        ardupilotmega.MAVLINK_MSG_ID_SYSTEM_TIME,
        interval_us,
        0,
        0,
        0,
        0,
        0,
    )


set_message_interval(ONE_SEC)
last_ts1 = 0
while True:
    msg = master.recv_match()
    if not msg:
        continue
    if msg.get_type() == "TIMESYNC":
        # print("\nAs dictionary: %s" % msg.to_dict())
        if msg.tc1 and msg.ts1 == last_ts1:
            rtt = (msg.tc1 - msg.ts1) / 2
            log.info("rtt: %f", rtt / 1e6)
    if msg.get_type() == "SYSTEM_TIME":
        last_ts1 = int(msg.time_boot_ms * 1e6)
        send_timesync_request(last_ts1)
