#pragma once

#include "realsense_qcom_bolt/config.hpp"
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
    void publish_static_tf();
    void on_frameset(const rs2::frameset& fs);
    void on_motion(const rs2::motion_frame& mf);
    rclcpp::Time frame_to_ros_time(const rs2::frame& f) const;
    sensor_msgs::msg::CameraInfo make_camera_info(const rs2_intrinsics& intrin,
                                                  const std::string& frame_id,
                                                  const rclcpp::Time& stamp) const;

    NodeConfig _cfg;
    PipelinePreset _preset{PipelinePreset::P0_BYPASS_ONLY};
    DynamicFilterParams _dyn_params;
    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr _param_cb_handle;

    std::unique_ptr<RealSenseCapture> _capture;
    std::unique_ptr<OutputBackend> _output;
    std::unique_ptr<VulkanContext> _vk_ctx;
    std::unique_ptr<VulkanPipeline> _vk_pipeline;

    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr _color_info_pub;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr _depth_info_pub;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr _aligned_info_pub;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr _gyro_pub;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr _accel_pub;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr _imu_pub;
    std::shared_ptr<tf2_ros::StaticTransformBroadcaster> _tf_broadcaster;

    std::mutex _imu_mtx;
    bool _has_accel{false};
    rs2_vector _last_accel{};
};

}  // namespace realsense_qcom_bolt
