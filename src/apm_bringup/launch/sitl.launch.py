"""
/home/user/git/ardupilot/build/sitl/bin/arducopter -S --model + --speedup 1 --slave 0 --uartB=uart:/dev/pts/3,9600 --defaults /home/user/git/ardupilot/Tools/autotest/default_params/copter.parm -I0
sim_vehicle.py -v ArduCopter -A "--uartB=uart:/dev/pts/3,9600"
sim_vehicle.py -v ArduCopter -A "--uartF=sim:rf_mavlink"
./arducopter -S --model + --speedup 1 --slave 0 --uartF=sim:rf_mavlink --defaults /home/user/wasp_ws/src/wasp_bringup/config/copter.parm,/home/user/wasp_ws/src/wasp_bringup/config/gazebo-iris.parm -I0
"""
#region imports
from ament_index_python.packages import get_package_share_directory
import os
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent, ExecuteProcess,
                            LogInfo, RegisterEventHandler, TimerAction)
from launch.conditions import IfCondition
from launch.event_handlers import (OnExecutionComplete, OnProcessExit,
                                OnProcessIO, OnProcessStart, OnShutdown)
from launch.events import Shutdown
from launch.substitutions import (EnvironmentVariable, FindExecutable,
                                LaunchConfiguration, LocalSubstitution,
                                PythonExpression)

#endregion

PACKAGE = "apm_bringup"

def generate_launch_description():
    ld = LaunchDescription()

    pkg = get_package_share_directory(PACKAGE)
    sitl_executable = os.path.join(pkg, "bin", "arducopter")
    copter_param = os.path.join(pkg, "config", "copter.parm")
    gazebo_param = os.path.join(pkg, "config", "gazebo-iris.parm")

    # arducopter -S --model gazebo-iris --defaults copter.param,gazebo-iris.param -I0
    spawn_sitl = ExecuteProcess(
        cmd=[[
            sitl_executable,
            ' -S ',
            "--model gazebo-iris ",
            f'--defaults {copter_param},{gazebo_param} ',
            "-I0"
        ]],
        shell=True
    )

    ld.add_action(spawn_sitl)
    # ld.add_action(spawn_mavproxy)
    # ld.add_action(on_sitl_start)
    
    return ld