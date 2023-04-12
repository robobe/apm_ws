from pymavlink import mavutil

# Create the connection
master = mavutil.mavlink_connection("udp:127.0.0.1:14550")
# Wait a heartbeat before sending commands
master.wait_heartbeat()

while True:
    msg = master.recv_match()
    if not msg:
        continue
    if msg.get_type() == "SYSTEM_TIME":
        print("\nAs dictionary: %s" % msg.to_dict())
