"""
DISTANCE_SENSOR ( #132 )
https://mavlink.io/en/messages/common.html#DISTANCE_SENSOR

./sim_vehicle.py -v ArduCopter \
-f quad -D \
--console \
--add-param-file /home/user/apm_ws/src/apm_bringup/config/range_finder.parm

RNGFND1_TYPE 10             # mavlink
RNGFND1_ORIENT 25           # down
RNGFND1_MAX_CM 1000         # cm
RNGFND1_MIN_CM 10           # cm
"""
import time

from pymavlink import mavutil
from pymavlink.dialects.v20 import ardupilotmega

TO_MS = 1e3
UPDATE_RATE = 0.5
RNGFND_TYPE_MAVLINK = 10
SENSOR_ID = 1
SENSOR_MAX_CM = 1000
SENSOR_MIN_CM = 10
SENSOR_COVARIANCE = 0

SIM_CURRENT_READING_CM = 200

# Create the connection
master = mavutil.mavlink_connection("udp:127.0.0.1:14550")
# Wait a heartbeat before sending commands
master.wait_heartbeat()

t_start = time.time()
while True:
    time.sleep(UPDATE_RATE)
    boot_time = int((time.time() - t_start) * TO_MS)
    master.mav.distance_sensor_send(
        boot_time,
        SENSOR_MIN_CM,
        SENSOR_MAX_CM,
        SIM_CURRENT_READING_CM,
        ardupilotmega.MAV_DISTANCE_SENSOR_UNKNOWN,
        SENSOR_ID,
        ardupilotmega.MAV_SENSOR_ROTATION_PITCH_270,
        SENSOR_COVARIANCE,
    )
    print("send --")
