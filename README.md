
## wiki
- Add github wiki as submodule

```
git submodule add https://github.com/robobe/apm_ws.wiki.git wiki
```

## How to run
```bash
# run from apm_ws
# Terminal 1
ros2 launch apm_bringup sim_and_spawn.launch.py

# Terminal 2
sim_vehicle.py -v ArduCopter -f gazebo-iris

```

## Run SITL
```
sim_vehicle.py -v ArduCopter -f gazebo-iris
```

```
mavproxy.py --out 127.0.0.1:14550 --out 127.0.0.1:14551 --out 127.0.0.1:14552 --master tcp:127.0.0.1:5760 
```

# Resource
- [Ardupilot Gazebo plugin](https://github.com/khancyr/ardupilot_gazebo)
- [Ardupilot Gazebo ROS Package](https://github.com/vincekurtz/ardupilot_gazebo)