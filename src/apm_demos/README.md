# APM_DEMOS

## Scripts

| name               | desc                                                | comp                                   |
| ------------------ | --------------------------------------------------- | -------------------------------------- |
| apm_time_sync      | connect sitl and gcs over mavros                    | gazebo, sitl(launch), mavproxy, mavros |
| sim_vehicle_mavros | launch without gazebo, mavproxy, mavros listen 5760 | sim_vehicle, mavproxy, mavros          |
| sim_vehicle_debug  |                                                     | sim_vehicle debug                      |
| sitl_gazebo |  | gazebo iris, sitl, mavros |


## mavros connection
```
ros2 run mavros mavros_node --ros-args -p fcu_url:=tcp://127.0.0.1:5760@ -p gcs_url:=udp://@localhost --params-file apm_config.yaml
```

## Demos
### arm and takeoff
- run `sitl_gazebo.sh`
- run `ros2 run apm_demos arm_and_takeoff`


### mavlink reader
- Read mavlink message using mavros,
- mavros uas publish raw mavlink to `/uas1/mavlink_source` topic

- run `ros2 run apm_demos mav_reader_demo` 

### mavlink writer
- write/send mavlink message using mavros,
- mavros uas subscribe to raw mavlink using `/uas1/mavlink_sink` topic

- run `ros2 run apm_demos mav_writer_demo` 

### mavros long command
- send mavlink longcommand using mavros `/mavros/cmd/command` service
- Arm vehicle using long command

```bash
ros2 run apm_demos arm_long_cmd
```





---

!!! tip "Disabled colcon python package warnings"
    ```
    export PYTHONWARNINGS=ignore:::setuptools.command.install
    ```
     

# gps vs optitrack params

| param        | gps     | optitrack   | description                                                                           |
| ------------ | ------- | ----------- | ------------------------------------------------------------------------------------- |
| AHRS_GPS_USE | 1       | 0: disabled | AHRS use GPS for DCM navigation and position-down                                     |
| COMPASS_USE  | 1       | 0: disabled | Enable or disable the use of the compass (instead of the GPS) for determining heading |
| COMPASS_USE2 | 1       | 0           |                                                                                       |
| COMPASS_USE3 | 1       | 0           |                                                                                       |
| EK2_GPS_TYPE | 0       | 3: no gps   | GPS mode control                                                                      |
| GPS_TYPE     | 1: Auto | 0: None     | Note: Reboot required after change                                                    |



!!! tip "DCM"
    Data Communication Module (GPS Receiver)
     


# to-read
-[[ROS2] asyncio await with timeout a service call in a callback](https://answers.ros.org/question/413482/ros2-asyncio-await-with-timeout-a-service-call-in-a-callback/)

