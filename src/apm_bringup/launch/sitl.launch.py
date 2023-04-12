"""
launch SITL with iris gazebo 
append param file from config folder if additional_param is set

/home/user/git/ardupilot/build/sitl/bin/arducopter -S --model + --speedup 1 --slave 0 --uartB=uart:/dev/pts/3,9600 --defaults /home/user/git/ardupilot/Tools/autotest/default_params/copter.parm -I0
sim_vehicle.py -v ArduCopter -A "--uartB=uart:/dev/pts/3,9600"
sim_vehicle.py -v ArduCopter -A "--uartF=sim:rf_mavlink"
./arducopter -S --model + --speedup 1 --slave 0 --uartF=sim:rf_mavlink --defaults /home/user/wasp_ws/src/wasp_bringup/config/copter.parm,/home/user/wasp_ws/src/wasp_bringup/config/gazebo-iris.parm -I0
"""
# region imports
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration

# endregion

PACKAGE = "apm_bringup"
BINARY = "arducopter"
COMMON_PARAM = "copter.parm"
GAZEBO_PARAM = "gazebo-iris.parm"

CUSTOM_ARG = "custom_param"


def build_sitl_command(context: LaunchContext, arg1: LaunchConfiguration):
    pkg = get_package_share_directory(PACKAGE)
    sitl_executable = os.path.join(pkg, "bin", BINARY)
    copter_param = os.path.join(pkg, "config", COMMON_PARAM)
    gazebo_param = os.path.join(pkg, "config", GAZEBO_PARAM)
    custom_param = context.perform_substitution(arg1)

    params_files = [copter_param, gazebo_param]
    if custom_param:
        custom_param = os.path.join(pkg, "config", custom_param)
        params_files.append(custom_param)

    params = ",".join(params_files)
    print(params)
    sitl = ExecuteProcess(
        cmd=[[sitl_executable, " -S ", "--model gazebo-iris ", f"--defaults {params} ", " -I0"]], shell=True
    )

    return [sitl]


def generate_launch_description():
    ld = LaunchDescription()
    custom_param_arg = DeclareLaunchArgument(
        CUSTOM_ARG,
        default_value="",
        description="additional SITL parm name only, load from config folder, append to end of params list",
    )
    custom_param = LaunchConfiguration(CUSTOM_ARG)

    func_action = OpaqueFunction(function=build_sitl_command, args=[custom_param])

    ld.add_action(custom_param_arg)
    ld.add_action(func_action)

    return ld
