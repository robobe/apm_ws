"""
Launch gazebo and spawn drone model
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    AppendEnvironmentVariable,
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

PACKAGE = "apm_bringup"
WORLD = "empty.world"
MODEL = "iris_camera"


def generate_launch_description():
    ld = LaunchDescription()

    pkg = get_package_share_directory(PACKAGE)
    gazebo_pkg = get_package_share_directory("gazebo_ros")

    verbose = LaunchConfiguration("verbose")
    arg_gazebo_verbose = DeclareLaunchArgument("verbose", default_value="true")
    world = LaunchConfiguration("world")
    arg_gazebo_world = DeclareLaunchArgument("world", default_value=WORLD)

    resources = ["/usr/share/gazebo-11", os.path.join(pkg, "worlds")]

    resource_env = AppendEnvironmentVariable(name="GAZEBO_RESOURCE_PATH", value=":".join(resources))

    models = [os.path.join(pkg, "models"), "~/apm_ws/src/apm_bringup/models"]

    models_env = AppendEnvironmentVariable(name="GAZEBO_MODEL_PATH", value=":".join(models))

    plugins = [os.path.join(pkg, "bin")]

    plugins_env = AppendEnvironmentVariable(name="GAZEBO_PLUGIN_PATH", value=":".join(plugins))

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(gazebo_pkg, "launch", "gazebo.launch.py")]),
        launch_arguments={"verbose": verbose, "world": world}.items(),
    )

    spawn_entity = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=["-entity", "demo", "-database", MODEL],
        output="screen",
    )

    timer = TimerAction(period=3.0, actions=[spawn_entity])

    ld.add_action(models_env)
    ld.add_action(resource_env)
    ld.add_action(plugins_env)
    ld.add_action(arg_gazebo_verbose)
    ld.add_action(arg_gazebo_world)
    ld.add_action(gazebo)
    ld.add_action(timer)
    return ld
