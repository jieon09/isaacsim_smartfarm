import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction, SetEnvironmentVariable
from launch_ros.actions import Node


def generate_launch_description():
    # launch/ 폴더의 상위 = D1_strawberry_src/
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    map_yaml    = os.path.join(pkg_dir, "maps", "carter_warehouse_navigation.yaml")
    nav2_params = os.path.join(pkg_dir, "robot2_nav2_params.yaml")
    rviz_config = os.path.join(pkg_dir, "robot2_spot.rviz")

    return LaunchDescription([
        SetEnvironmentVariable("ROS_DOMAIN_ID", "135"),
        SetEnvironmentVariable("ROS_LOCALHOST_ONLY", "0"),

        Node(
            package="nav2_map_server",
            executable="map_server",
            name="map_server",
            output="screen",
            parameters=[{"yaml_filename": map_yaml}],
        ),

        TimerAction(
            period=2.0,
            actions=[
                ExecuteProcess(
                    cmd=["ros2", "lifecycle", "set", "/map_server", "configure"],
                    output="screen",
                )
            ],
        ),

        TimerAction(
            period=4.0,
            actions=[
                ExecuteProcess(
                    cmd=["ros2", "lifecycle", "set", "/map_server", "activate"],
                    output="screen",
                )
            ],
        ),

        TimerAction(
            period=5.0,
            actions=[
                ExecuteProcess(
                    cmd=[
                        "ros2", "launch", "nav2_bringup", "navigation_launch.py",
                        "use_sim_time:=False",
                        f"params_file:={nav2_params}",
                        "autostart:=True",
                    ],
                    output="screen",
                )
            ],
        ),

        TimerAction(
            period=8.0,
            actions=[
                ExecuteProcess(
                    cmd=["rviz2", "-d", rviz_config],
                    output="screen",
                )
            ],
        ),
    ])
