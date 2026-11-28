// Copyright 2026 voidrobot
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "realsense_qcom_bolt/config.hpp"
#include "realsense_qcom_bolt/math_utils.hpp"
#include "realsense_qcom_bolt/output_backend.hpp"
#include "realsense_qcom_bolt/realsense_capture.hpp"
#include "realsense_qcom_bolt/vulkan_context.hpp"
#include "realsense_qcom_bolt/vulkan_pipeline.hpp"

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <tf2_ros/static_transform_broadcaster.h>

#include <memory>
#include <mutex>

namespace realsense_qcom_bolt
{

class BoltNode : public rclcpp::Node
{
public:
    explicit BoltNode(const rclcpp::NodeOptions& options);
    ~BoltNode() override;

private:
    void declare_parameters();
    void setup_publishers();
    void init_camera_info_templates();
    void publish_static_tf();
    void on_frameset(const rs2::frameset& fs);
    void on_motion(const rs2::motion_frame& mf);
    rclcpp::Time frame_to_ros_time(const rs2::frame& f) const;
    static sensor_msgs::msg::CameraInfo build_camera_info_template(const rs2_intrinsics& intrin,
                                                                   const std::string& frame_id);

    NodeConfig _cfg;
    PipelinePreset _preset{PipelinePreset::P0_BYPASS_ONLY};
    DynamicFilterParams _dyn_params;
    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr _param_cb_handle;

    // Destruction order: _capture -> _vk_pipeline -> _output -> _vk_ctx
    std::unique_ptr<VulkanContext> _vk_ctx;
    std::unique_ptr<OutputBackend> _output;
    std::unique_ptr<VulkanPipeline> _vk_pipeline;
    std::unique_ptr<RealSenseCapture> _capture;

    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr _color_info_pub;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr _depth_info_pub;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr _aligned_info_pub;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr _infra1_info_pub;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr _infra2_info_pub;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr _gyro_pub;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr _accel_pub;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr _imu_pub;
    std::shared_ptr<tf2_ros::StaticTransformBroadcaster> _tf_broadcaster;

    // Cached CameraInfo templates (zero per-frame std::vector heap allocations)
    sensor_msgs::msg::CameraInfo _color_info_tpl;
    sensor_msgs::msg::CameraInfo _depth_info_tpl;
    sensor_msgs::msg::CameraInfo _aligned_info_tpl;
    sensor_msgs::msg::CameraInfo _infra1_info_tpl;
    sensor_msgs::msg::CameraInfo _infra2_info_tpl;

    alignas(kCacheLineBytes) std::mutex _imu_mtx;
    std::unique_ptr<ImuSynchronizer> _imu_sync;
};

}  // namespace realsense_qcom_bolt
