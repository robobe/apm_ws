"""
"mavproxy.py" "--master" "tcp:127.0.0.1:5760" "--sitl" "127.0.0.1:5501" "--out" "127.0.0.1:14550" "--out" "127.0.0.1:14551" "--map" "--console"
"""
import os

# region imports
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess

# endregion

PACKAGE = "apm_bringup"


def generate_launch_description():
    ld = LaunchDescription()

    pkg = get_package_share_directory(PACKAGE)
    gcs_port = 14550
    cc_port = 14551

    # arducopter -S --model gazebo-iris --defaults copter.param,gazebo-iris.param -I0
    spawn_sitl = ExecuteProcess(
        cmd=[
            [
                "mavproxy.py",
                " --master ",
                "tcp:127.0.0.1:5760 ",
                f"--out 127.0.0.1:{gcs_port} " f"--out 127.0.0.1:{cc_port} ",
            ]
        ],
        shell=True,
    )

    ld.add_action(spawn_sitl)

    return ld
