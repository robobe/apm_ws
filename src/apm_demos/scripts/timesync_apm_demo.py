import time

from pymavlink import mavutil

# Create the connection
master = mavutil.mavlink_connection("udp:127.0.0.1:14550")
# Wait a heartbeat before sending commands
master.wait_heartbeat()


def send_timesync_response(ts1):
    tc1 = int(time.time() * 1e9)
    master.mav.timesync_send(tc1, ts1)
    print("--send--")


while True:
    msg = master.recv_match()
    if not msg:
        continue
    if msg.get_type() == "TIMESYNC":
        print("\nAs dictionary: %s" % msg.to_dict())
        if msg.tc1 == 0:
            send_timesync_response(msg.ts1)
