# APM_DEMOS


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


```

```