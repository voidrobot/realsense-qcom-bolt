# Copyright 2026 voidrobot
# SPDX-License-Identifier: Apache-2.0

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("color_profile", default_value="848x480x60"),
        DeclareLaunchArgument("depth_profile", default_value="848x480x60"),
        DeclareLaunchArgument("align_depth", default_value="true"),
        DeclareLaunchArgument("pointcloud", default_value="true"),
        DeclareLaunchArgument("ordered_pc", default_value="false"),
        DeclareLaunchArgument("coord_frame", default_value="depth"),
        DeclareLaunchArgument("use_qirp_transport", default_value="true"),
        DeclareLaunchArgument("qirp_dmabuf_pool_size", default_value="8"),
        ComposableNodeContainer(
            name="bolt_container",
            namespace="camera",
            package="rclcpp_components",
            executable="component_container_mt",
            composable_node_descriptions=[
                ComposableNode(
                    package="realsense_qcom_bolt",
                    plugin="realsense_qcom_bolt::BoltNode",
                    name="camera",
                    namespace="camera",
                    parameters=[{
                        "rgb_camera.color_profile": LaunchConfiguration("color_profile"),
                        "depth_module.depth_profile": LaunchConfiguration("depth_profile"),
                        "align_depth.enable": LaunchConfiguration("align_depth"),
                        "pointcloud.enable": LaunchConfiguration("pointcloud"),
                        "pointcloud.ordered_pc": LaunchConfiguration("ordered_pc"),
                        "pointcloud.coord_frame": LaunchConfiguration("coord_frame"),
                        "use_qirp_transport": LaunchConfiguration("use_qirp_transport"),
                        "qirp_dmabuf_pool_size": LaunchConfiguration("qirp_dmabuf_pool_size"),
                    }],
                    extra_arguments=[{"use_intra_process_comms": True}],
                ),
            ],
            output="screen",
        ),
    ])
