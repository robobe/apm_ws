"""simple pymavlink script
Run against sitl without mavproxy or any other middle man


"""
from pymavlink import mavutil

# master = mavutil.mavlink_connection("tcp:127.0.0.1:5760")
master = mavutil.mavlink_connection("udp:127.0.0.1:14550")
master.wait_heartbeat()

while True:
    msg = master.recv_match()
    if not msg:
        continue
    print(msg.to_dict())
