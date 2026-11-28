// Copyright 2026 voidrobot
// SPDX-License-Identifier: Apache-2.0

#include "realsense_qcom_bolt/bolt_node.hpp"

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/image_encodings.hpp>

#include <cmath>

namespace realsense_qcom_bolt
{

namespace
{
rcl_interfaces::msg::ParameterDescriptor make_ro_desc(const std::string& desc)
{
    rcl_interfaces::msg::ParameterDescriptor d;
    d.description = desc;
    d.read_only = true;
    return d;
}
}  // namespace

BoltNode::BoltNode(const rclcpp::NodeOptions& options)
    : rclcpp::Node("realsense_qcom_bolt", options)
{
    declare_parameters();
    _preset = resolve_pipeline_preset(_cfg);
    RCLCPP_INFO(this->get_logger(),
                "Selected fixed pipeline preset: %s (QIRP=%s, pool_size=%zu)",
                preset_to_string(_preset),
                _cfg.use_qirp_transport ? "ON" : "OFF",
                _cfg.qirp_dmabuf_pool_size);

    _imu_sync = std::make_unique<ImuSynchronizer>(_cfg.unite_imu_method);

    _capture = std::make_unique<RealSenseCapture>(_cfg, _preset);
    _capture->open();

    if (preset_uses_vulkan(_preset))
    {
        _vk_ctx = std::make_unique<VulkanContext>();
        RCLCPP_INFO(this->get_logger(),
                    "Vulkan 1.3 Compute initialized on '%s' (dmabuf_import=%s, atom=%luB)",
                    _vk_ctx->device_name().c_str(),
                    _vk_ctx->supports_dmabuf_import() ? "YES" : "NO",
                    static_cast<unsigned long>(_vk_ctx->non_coherent_atom_size()));
    }
    else
    {
        RCLCPP_INFO(this->get_logger(), "P0_bypass_only active: Vulkan context not initialized (0 VRAM).");
    }

    const rclcpp::QoS qos(rclcpp::KeepLast(10));
    _output = std::make_unique<OutputBackend>(
        *this, _cfg.use_qirp_transport, _cfg.qirp_dmabuf_pool_size, qos, _vk_ctx.get());
    setup_publishers();
    init_camera_info_templates();

    if (_vk_ctx)
    {
        _vk_pipeline = std::make_unique<VulkanPipeline>(
            *_vk_ctx, _preset, _cfg, _capture->calibration(), *_output);
    }

    if (_cfg.publish_tf)
    {
        publish_static_tf();
    }

    _param_cb_handle = this->add_on_set_parameters_callback(
        [this](const std::vector<rclcpp::Parameter>& params) {
            rcl_interfaces::msg::SetParametersResult res;
            res.successful = true;
            bool changed = false;
            for (const auto& p : params)
            {
                const auto& name = p.get_name();
                if (name == param_keys::kClippingDistance)
                {
                    _dyn_params.max_depth_m = static_cast<float>(p.as_double());
                    changed = true;
                }
                else if (name == param_keys::kMinDepthM)
                {
                    _dyn_params.min_depth_m = static_cast<float>(p.as_double());
                    changed = true;
                }
                else if (name == param_keys::kBilateralSigmaSpatial)
                {
                    _dyn_params.bilateral_sigma_spatial = static_cast<float>(p.as_double());
                    changed = true;
                }
                else if (name == param_keys::kBilateralSigmaRange)
                {
                    _dyn_params.bilateral_sigma_range = static_cast<float>(p.as_double());
                    changed = true;
                }
                else if (name == param_keys::kTemporalAlpha)
                {
                    _dyn_params.temporal_alpha = static_cast<float>(p.as_double());
                    changed = true;
                }
                else if (name == param_keys::kTemporalDelta)
                {
                    _dyn_params.temporal_delta_m = static_cast<float>(p.as_double());
                    changed = true;
                }
            }
            if (changed && _vk_pipeline)
            {
                _vk_pipeline->update_dynamic_params(_dyn_params);
            }
            return res;
        });

    _capture->start_loop(
        [this](const rs2::frameset& fs) { on_frameset(fs); },
        [this](const rs2::motion_frame& mf) { on_motion(mf); });
}

BoltNode::~BoltNode()
{
    if (_capture)
        _capture->stop();
}

void BoltNode::declare_parameters()
{
    _cfg.enable_color = this->declare_parameter<bool>("enable_color", true, make_ro_desc("Enable color stream"));
    _cfg.enable_depth = this->declare_parameter<bool>("enable_depth", true, make_ro_desc("Enable depth stream"));
    _cfg.enable_infra1 = this->declare_parameter<bool>("enable_infra1", false, make_ro_desc("Enable IR1 stream"));
    _cfg.enable_infra2 = this->declare_parameter<bool>("enable_infra2", false, make_ro_desc("Enable IR2 stream"));
    _cfg.enable_gyro = this->declare_parameter<bool>("enable_gyro", false, make_ro_desc("Enable gyro stream"));
    _cfg.enable_accel = this->declare_parameter<bool>("enable_accel", false, make_ro_desc("Enable accel stream"));

    _cfg.color_profile = parse_stream_profile(
        this->declare_parameter<std::string>("rgb_camera.color_profile", "848x480x60", make_ro_desc("Color WxHxFPS")));
    _cfg.depth_profile = parse_stream_profile(
        this->declare_parameter<std::string>("depth_module.depth_profile", "848x480x60", make_ro_desc("Depth WxHxFPS")));
    _cfg.infra_profile = parse_stream_profile(
        this->declare_parameter<std::string>("depth_module.infra_profile", "848x480x60", make_ro_desc("IR WxHxFPS")));

    _cfg.gyro_fps = this->declare_parameter<int>("gyro_fps", 200, make_ro_desc("Gyro FPS"));
    _cfg.accel_fps = this->declare_parameter<int>("accel_fps", 200, make_ro_desc("Accel FPS"));
    _cfg.unite_imu_method = parse_unite_imu_method(
        this->declare_parameter<std::string>("unite_imu_method", "none", make_ro_desc("IMU unite method")));

    _cfg.color_format = this->declare_parameter<std::string>("color_format", "rgb8", make_ro_desc("Color format"));
    _cfg.depth_format = this->declare_parameter<std::string>("depth_format", "z16", make_ro_desc("Depth format"));
    _cfg.infra_format = this->declare_parameter<std::string>("infra_format", "y8", make_ro_desc("IR format (y8/y16)"));

    _cfg.align_depth_enable = this->declare_parameter<bool>("align_depth.enable", true, make_ro_desc("Align depth to color"));
    _cfg.pointcloud_enable = this->declare_parameter<bool>("pointcloud.enable", true, make_ro_desc("Enable pointcloud"));
    _cfg.pointcloud_ordered = this->declare_parameter<bool>("pointcloud.ordered_pc", false, make_ro_desc("Ordered pointcloud"));
    _cfg.pointcloud_allow_no_texture = this->declare_parameter<bool>("pointcloud.allow_no_texture_points", false, make_ro_desc("Allow untextured points"));
    _cfg.pointcloud_coord_frame = parse_coord_frame(
        this->declare_parameter<std::string>("pointcloud.coord_frame", "depth", make_ro_desc("PointCloud frame: depth|color")));

    _cfg.gpu_bilateral_enable = this->declare_parameter<bool>("gpu_filter.bilateral_enable", false, make_ro_desc("Enable GPU bilateral filter"));
    _cfg.gpu_temporal_enable = this->declare_parameter<bool>("gpu_filter.temporal_enable", false, make_ro_desc("Enable GPU temporal filter"));

    _cfg.serial_no = this->declare_parameter<std::string>("serial_no", "", make_ro_desc("Camera serial number"));
    _cfg.rosbag_filename = this->declare_parameter<std::string>("rosbag_filename", "", make_ro_desc("Playback .db3 path"));
    _cfg.rosbag_loop = this->declare_parameter<bool>("rosbag_loop", false, make_ro_desc("Loop .db3 playback"));
    _cfg.global_time_enabled = this->declare_parameter<bool>("global_time_enabled", true, make_ro_desc("Global time domain"));
    _cfg.publish_tf = this->declare_parameter<bool>("publish_tf", true, make_ro_desc("Publish static TF"));
    _cfg.base_frame_id = this->declare_parameter<std::string>("base_frame_id", "camera_link", make_ro_desc("Base frame ID"));
    _cfg.tf_prefix = this->declare_parameter<std::string>("tf_prefix", "", make_ro_desc("TF prefix"));

    _cfg.use_qirp_transport = this->declare_parameter<bool>("use_qirp_transport", false, make_ro_desc("Use QIRP DMA-BUF Zero-Copy"));
#ifndef BUILD_WITH_QIRP
    if (_cfg.use_qirp_transport)
    {
        RCLCPP_WARN(this->get_logger(), "use_qirp_transport:=true requested with BUILD_WITH_QIRP=OFF; falling back to standard ROS.");
        _cfg.use_qirp_transport = false;
    }
#endif

    rcl_interfaces::msg::ParameterDescriptor pool_desc = make_ro_desc("QIRP DMA-BUF ring pool slots [0,32]");
    rcl_interfaces::msg::IntegerRange pool_range;
    pool_range.from_value = 0;
    pool_range.to_value = static_cast<int64_t>(defaults::kMaxPoolSize);
    pool_range.step = 1;
    pool_desc.integer_range.push_back(pool_range);
    _cfg.qirp_dmabuf_pool_size = static_cast<std::size_t>(
        this->declare_parameter<int>("qirp_dmabuf_pool_size", static_cast<int>(defaults::kDefaultPoolSize), pool_desc));

    // Dynamic filter constants (Push Constants / Calib UBO, runtime mutable)
    _dyn_params.max_depth_m = static_cast<float>(this->declare_parameter<double>(param_keys::kClippingDistance, -1.0));
    _dyn_params.min_depth_m = static_cast<float>(this->declare_parameter<double>(param_keys::kMinDepthM, 0.10));
    _dyn_params.bilateral_sigma_spatial = static_cast<float>(this->declare_parameter<double>(param_keys::kBilateralSigmaSpatial, 2.0));
    _dyn_params.bilateral_sigma_range = static_cast<float>(this->declare_parameter<double>(param_keys::kBilateralSigmaRange, 0.05));
    _dyn_params.temporal_alpha = static_cast<float>(this->declare_parameter<double>(param_keys::kTemporalAlpha, 0.4));
    _dyn_params.temporal_delta_m = static_cast<float>(this->declare_parameter<double>(param_keys::kTemporalDelta, 0.05));
    _dyn_params.rotation_deg = this->declare_parameter<int>("rotation_deg", 0, make_ro_desc("Image rotation (only 0 supported)"));
    _cfg.initial_filter_params = _dyn_params;
}

void BoltNode::setup_publishers()
{
    const rclcpp::QoS info_qos(rclcpp::KeepLast(10));
    if (_cfg.enable_color)
    {
        _output->create_color_publisher("~/color/image_raw");
        _color_info_pub = this->create_publisher<sensor_msgs::msg::CameraInfo>("~/color/camera_info", info_qos);
    }
    if (_cfg.enable_depth)
    {
        _output->create_depth_publisher("~/depth/image_rect_raw");
        _depth_info_pub = this->create_publisher<sensor_msgs::msg::CameraInfo>("~/depth/camera_info", info_qos);
    }
    if (_cfg.align_depth_enable)
    {
        _output->create_aligned_depth_publisher("~/aligned_depth_to_color/image_raw");
        _aligned_info_pub = this->create_publisher<sensor_msgs::msg::CameraInfo>("~/aligned_depth_to_color/camera_info", info_qos);
    }
    if (_cfg.pointcloud_enable)
    {
        const std::string pc_topic =
            (_preset == PipelinePreset::P5_RGB_ALIGNED_PC_COLOR_FRAME)
                ? "~/aligned_depth_to_color/points"
                : (_preset == PipelinePreset::P6_DEPTH_PC_XYZ ? "~/depth/points" : "~/depth/color/points");
        _output->create_pointcloud_publisher(pc_topic);
    }
    if (_cfg.enable_infra1)
    {
        _output->create_infra_publisher(1, "~/infra1/image_rect_raw");
        _infra1_info_pub = this->create_publisher<sensor_msgs::msg::CameraInfo>("~/infra1/camera_info", info_qos);
    }
    if (_cfg.enable_infra2)
    {
        _output->create_infra_publisher(2, "~/infra2/image_rect_raw");
        _infra2_info_pub = this->create_publisher<sensor_msgs::msg::CameraInfo>("~/infra2/camera_info", info_qos);
    }
    if (_cfg.enable_gyro)
        _gyro_pub = this->create_publisher<sensor_msgs::msg::Imu>("~/gyro/sample", info_qos);
    if (_cfg.enable_accel)
        _accel_pub = this->create_publisher<sensor_msgs::msg::Imu>("~/accel/sample", info_qos);
    if (_cfg.unite_imu_method != UniteImuMethod::NONE && _cfg.enable_gyro && _cfg.enable_accel)
        _imu_pub = this->create_publisher<sensor_msgs::msg::Imu>("~/imu", info_qos);
}

sensor_msgs::msg::CameraInfo BoltNode::build_camera_info_template(const rs2_intrinsics& intrin,
                                                                  const std::string& frame_id)
{
    sensor_msgs::msg::CameraInfo ci;
    ci.header.frame_id = frame_id;
    ci.width = static_cast<uint32_t>(intrin.width);
    ci.height = static_cast<uint32_t>(intrin.height);
    ci.distortion_model = "plumb_bob";
    ci.d.assign(intrin.coeffs, intrin.coeffs + 5);
    ci.k = {intrin.fx, 0.0, intrin.ppx, 0.0, intrin.fy, intrin.ppy, 0.0, 0.0, 1.0};
    ci.r = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    ci.p = {intrin.fx, 0.0, intrin.ppx, 0.0, 0.0, intrin.fy, intrin.ppy, 0.0, 0.0, 0.0, 1.0, 0.0};
    return ci;
}

void BoltNode::init_camera_info_templates()
{
    const auto& calib = _capture->calibration();
    if (calib.has_color)
    {
        _color_info_tpl = build_camera_info_template(calib.color_intrin, _cfg.tf_prefix + "camera_color_optical_frame");
        _aligned_info_tpl = _color_info_tpl;
    }
    if (calib.has_depth)
    {
        _depth_info_tpl = build_camera_info_template(calib.depth_intrin, _cfg.tf_prefix + "camera_depth_optical_frame");
    }
    if (calib.has_infra1)
    {
        _infra1_info_tpl = build_camera_info_template(calib.infra1_intrin, _cfg.tf_prefix + "camera_infra1_optical_frame");
    }
    if (calib.has_infra2)
    {
        _infra2_info_tpl = build_camera_info_template(calib.infra2_intrin, _cfg.tf_prefix + "camera_infra2_optical_frame");
    }
}

void BoltNode::publish_static_tf()
{
    _tf_broadcaster = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);
    const auto& calib = _capture->calibration();
    std::vector<geometry_msgs::msg::TransformStamped> tfs;

    auto make_identity_tf = [&](const std::string& parent, const std::string& child) {
        geometry_msgs::msg::TransformStamped tf;
        tf.header.stamp = this->now();
        tf.header.frame_id = _cfg.tf_prefix + parent;
        tf.child_frame_id = _cfg.tf_prefix + child;
        tf.transform.rotation.w = 1.0;
        return tf;
    };

    tfs.push_back(make_identity_tf(_cfg.base_frame_id, "camera_depth_optical_frame"));
    if (calib.has_depth && calib.has_color)
    {
        // ROS TF (parent=depth_optical, child=color_optical) requires T_{depth <- color} = inv(depth_to_color).
        const float* r = calib.depth_to_color.rotation;  // column-major: r[col*3 + row]
        const float* t = calib.depth_to_color.translation;
        const float r_inv[9] = {
            r[0], r[3], r[6],
            r[1], r[4], r[7],
            r[2], r[5], r[8],
        };
        geometry_msgs::msg::TransformStamped tf = make_identity_tf("camera_depth_optical_frame", "camera_color_optical_frame");
        tf.transform.translation.x = -(static_cast<double>(r[0]) * t[0] + static_cast<double>(r[1]) * t[1] + static_cast<double>(r[2]) * t[2]);
        tf.transform.translation.y = -(static_cast<double>(r[3]) * t[0] + static_cast<double>(r[4]) * t[1] + static_cast<double>(r[5]) * t[2]);
        tf.transform.translation.z = -(static_cast<double>(r[6]) * t[0] + static_cast<double>(r[7]) * t[1] + static_cast<double>(r[8]) * t[2]);
        const Quaternion q = rotation_matrix_to_quaternion(r_inv);
        tf.transform.rotation.x = q.x;
        tf.transform.rotation.y = q.y;
        tf.transform.rotation.z = q.z;
        tf.transform.rotation.w = q.w;
        tfs.push_back(tf);
    }
    _tf_broadcaster->sendTransform(tfs);
}

rclcpp::Time BoltNode::frame_to_ros_time(const rs2::frame& f) const
{
    const double ts_ms = f.get_timestamp();
    const int64_t ns = static_cast<int64_t>(std::llround(ts_ms * 1e6));
    return rclcpp::Time(ns, this->get_clock()->get_clock_type());
}

void BoltNode::on_frameset(const rs2::frameset& fs)
{
    rs2::video_frame color_frame = fs.get_color_frame();
    rs2::depth_frame depth_frame = fs.get_depth_frame();

    const rclcpp::Time fs_stamp = frame_to_ros_time(fs);

    std_msgs::msg::Header color_hdr;
    color_hdr.frame_id = _cfg.tf_prefix + "camera_color_optical_frame";
    color_hdr.stamp = fs_stamp;

    std_msgs::msg::Header depth_hdr;
    depth_hdr.frame_id = _cfg.tf_prefix + "camera_depth_optical_frame";
    depth_hdr.stamp = fs_stamp;

    if (_preset == PipelinePreset::P0_BYPASS_ONLY)
    {
        if (color_frame && _output->has_color_subscribers())
        {
            _output->publish_color_from_cpu(
                color_hdr, color_frame.get_width(), color_frame.get_height(),
                color_frame.get_stride_in_bytes(), color_frame.get_data());
        }
        if (depth_frame && _output->has_depth_subscribers())
        {
            _output->publish_depth_from_cpu(
                depth_hdr, depth_frame.get_width(), depth_frame.get_height(),
                depth_frame.get_data(), false);
        }
    }
    else if (_vk_pipeline)
    {
        const bool color_is_yuyv = (color_frame && color_frame.get_profile().format() == RS2_FORMAT_YUYV);
        _vk_pipeline->process_and_publish(
            color_frame ? color_frame.get_data() : nullptr,
            color_frame ? static_cast<std::size_t>(color_frame.get_data_size()) : 0u,
            color_is_yuyv,
            depth_frame ? depth_frame.get_data() : nullptr,
            depth_frame ? static_cast<std::size_t>(depth_frame.get_data_size()) : 0u,
            color_frame ? color_hdr : depth_hdr,
            depth_frame ? depth_hdr : color_hdr);
    }

    // Independent IR1/IR2 passthrough + cached CameraInfo
    for (int idx : {1, 2})
    {
        rs2::video_frame ir = fs.get_infrared_frame(idx);
        if (!ir)
            continue;
        if (_output->has_infra_subscribers(idx))
        {
            std_msgs::msg::Header ir_hdr;
            ir_hdr.frame_id = _cfg.tf_prefix + "camera_infra" + std::to_string(idx) + "_optical_frame";
            ir_hdr.stamp = fs_stamp;
            const bool is_y16 = (ir.get_profile().format() == RS2_FORMAT_Y16);
            _output->publish_infra_from_cpu(
                idx, ir_hdr, ir.get_width(), ir.get_height(),
                is_y16 ? sensor_msgs::image_encodings::MONO16 : sensor_msgs::image_encodings::MONO8,
                is_y16 ? 2 : 1, ir.get_data());
        }
        auto& info_pub = (idx == 1) ? _infra1_info_pub : _infra2_info_pub;
        auto& info_tpl = (idx == 1) ? _infra1_info_tpl : _infra2_info_tpl;
        if (info_pub && info_pub->get_subscription_count() > 0 && info_tpl.width > 0)
        {
            info_tpl.header.stamp = fs_stamp;
            info_pub->publish(info_tpl);
        }
    }

    // Cached CameraInfo publishing (zero per-frame heap allocations)
    if (_color_info_pub && color_frame && _color_info_pub->get_subscription_count() > 0)
    {
        _color_info_tpl.header.stamp = color_hdr.stamp;
        _color_info_pub->publish(_color_info_tpl);
    }
    if (_depth_info_pub && depth_frame && _depth_info_pub->get_subscription_count() > 0)
    {
        _depth_info_tpl.header.stamp = depth_hdr.stamp;
        _depth_info_pub->publish(_depth_info_tpl);
    }
    if (_aligned_info_pub && color_frame && _aligned_info_pub->get_subscription_count() > 0)
    {
        _aligned_info_tpl.header.stamp = depth_hdr.stamp;
        _aligned_info_pub->publish(_aligned_info_tpl);
    }
}

void BoltNode::on_motion(const rs2::motion_frame& mf)
{
    const auto st = mf.get_profile().stream_type();
    const rs2_vector v = mf.get_motion_data();
    const double ts_ms = mf.get_timestamp();
    const rclcpp::Time stamp = frame_to_ros_time(mf);

    if (st == RS2_STREAM_ACCEL)
    {
        {
            std::lock_guard<std::mutex> lk(_imu_mtx);
            _imu_sync->push_accel(ImuSample{ts_ms, v.x, v.y, v.z});
        }
        if (_accel_pub && _accel_pub->get_subscription_count() > 0)
        {
            sensor_msgs::msg::Imu msg;
            msg.header.stamp = stamp;
            msg.header.frame_id = _cfg.tf_prefix + "camera_accel_optical_frame";
            msg.linear_acceleration.x = v.x;
            msg.linear_acceleration.y = v.y;
            msg.linear_acceleration.z = v.z;
            _accel_pub->publish(msg);
        }
    }
    else if (st == RS2_STREAM_GYRO)
    {
        if (_gyro_pub && _gyro_pub->get_subscription_count() > 0)
        {
            sensor_msgs::msg::Imu msg;
            msg.header.stamp = stamp;
            msg.header.frame_id = _cfg.tf_prefix + "camera_gyro_optical_frame";
            msg.angular_velocity.x = v.x;
            msg.angular_velocity.y = v.y;
            msg.angular_velocity.z = v.z;
            _gyro_pub->publish(msg);
        }
        if (_imu_pub && _imu_pub->get_subscription_count() > 0)
        {
            std::optional<UnitedImuSample> united;
            {
                std::lock_guard<std::mutex> lk(_imu_mtx);
                united = _imu_sync->push_gyro(ImuSample{ts_ms, v.x, v.y, v.z});
            }
            if (united.has_value())
            {
                sensor_msgs::msg::Imu msg;
                msg.header.stamp = stamp;
                msg.header.frame_id = _cfg.tf_prefix + "camera_imu_optical_frame";
                msg.angular_velocity.x = united->gx;
                msg.angular_velocity.y = united->gy;
                msg.angular_velocity.z = united->gz;
                msg.linear_acceleration.x = united->ax;
                msg.linear_acceleration.y = united->ay;
                msg.linear_acceleration.z = united->az;
                _imu_pub->publish(msg);
            }
        }
    }
}

}  // namespace realsense_qcom_bolt

RCLCPP_COMPONENTS_REGISTER_NODE(realsense_qcom_bolt::BoltNode)
