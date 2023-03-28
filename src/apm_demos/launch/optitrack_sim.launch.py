from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import LogInfo

def generate_launch_description():
    ld = LaunchDescription()

    sim_node =  Node(
            package='apm_demos',
            namespace='',
            executable='simulate_optitrack',
            name='simulate_optitrack'
        )

    bridge_node =  Node(
            package='apm_demos',
            namespace='',
            executable='pose_republisher',
            name='pose_republisher'
        )

    set_origin_node =  Node(
            package='apm_demos',
            namespace='',
            executable='set_origin',
            name='set_origin'
        )
    

    ld.add_action(sim_node)
    ld.add_action(bridge_node)
    ld.add_action(set_origin_node)

    return ld