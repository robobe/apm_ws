"""SYSTEM_TIME
mavlink.io common message #2
The system time is the time of the master clock

# Demo
- run sitl with debug (no gazebo)

```
./Tools/autotest/sim_vehicle.py -v ArduCopter -f quad -D
--add-param-file=/home/user/apm_ws/src/apm_demos/config/system_time.parm
```


## config
- With GPS
- RTC config to mavlink to allow GPS time source
```
BRD_RTC_TYPES 1
GPS_TYPE 1
```

[check my blog](https://robobe.github.io/blog/Ardupilot/system_time/)
"""
import logging
import time

# os.environ["MAVLINK20"] = "1"
from pymavlink import mavutil
from pymavlink.dialects.v20 import ardupilotmega

FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(format=FMT, level=logging.INFO)
log = logging.getLogger(__name__)

# Create the connection
# master = mavutil.mavlink_connection("/dev/ttyACM0")
# master = mavutil.mavlink_connection("tcp:0.0.0.0:5760")
master = mavutil.mavlink_connection("udp:127.0.0.1:14550")
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


def set_system_time():
    current = int(time.time() * 1e6)
    log.info("%s", current)
    master.mav.system_time_send(current, 0)


# Wait a heartbeat before sending commands
master.wait_heartbeat()

set_message_interval(ONE_SEC)
set_system_time()

USE_PARAM_ID = -1
master.mav.param_request_read_send(master.target_system, master.target_component, b"BRD_RTC_TYPES", USE_PARAM_ID)

while True:
    msg = master.recv_match()
    if not msg:
        continue
    if msg.get_type() == "SYSTEM_TIME":
        log.info("%s", msg.to_dict())
    if msg.get_type() == "TIMESYNC":
        log.info("%s", msg.to_dict())
    if msg.get_type() == "PARAM_VALUE":
        message = msg.to_dict()
        param_name = message["param_id"]
        if param_name == "BRD_RTC_TYPES":
            param_value = message["param_value"]
            log.info("param_name: %s value: %s", param_name, param_value)
