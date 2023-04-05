from launch import LaunchDescription, LaunchContext
from launch.actions import ExecuteProcess, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.actions import OpaqueFunction

def func_demo(context: LaunchContext, arg1: LaunchConfiguration):
    value = context.perform_substitution(arg1)
    data = ""
    if value == "hello_world":
        data = "sssssssssssssssssss"
    run_script = ExecuteProcess(
        cmd=[["/home/user/apm_ws/src/apm_bringup/scripts/hello.zsh ", data]],
        shell=True
    )
    return [run_script]


def generate_launch_description():
    ld = LaunchDescription()
    arg1 = LaunchConfiguration('arg1')
    arg1_arg = DeclareLaunchArgument(
        'arg1', default_value="hello_world", description="arg1"
    )
    
    func = OpaqueFunction(function=func_demo, args=[arg1])
    
    ld.add_action(arg1_arg)
    ld.add_action(func)
    
    return ld
