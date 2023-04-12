import time
from threading import Thread

# Import mavutil
from pymavlink import mavutil

# Create the connection
master = mavutil.mavlink_connection("udp:127.0.0.1:14550")
# Wait a heartbeat before sending commands
master.wait_heartbeat()

# def wait_conn():
#     """
#     Sends a ping to the autopilot to stabilish the UDP connection and waits for a reply
#     """
#     msg = None
#     while not msg:
#         master.mav.ping_send(
#             int(time.time() * 1e6), # Unix time in microseconds
#             0, # Ping number
#             0, # Request ping of all systems
#             0 # Request ping of all components
#         )
#         msg = master.recv_match()
#         time.sleep(0.5)

# wait_conn()


def send_loop():
    while True:
        tc1 = 0
        ts1 = int(time.time() * 1e9)
        print(ts1)
        master.mav.timesync_send(tc1, ts1)
        print("--send--")
        time.sleep(1)


t = Thread(target=send_loop, daemon=True)
t.start()

while True:
    msg = master.recv_match()
    if not msg:
        continue
    if msg.get_type() == "TIMESYNC":
        print("\nAs dictionary: %s" % msg.to_dict())
        print(msg.tc1 / 1e9, msg.ts1)
