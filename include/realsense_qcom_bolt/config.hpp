// Copyright 2026 voidrobot
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace realsense_qcom_bolt
{

namespace defaults
{
inline constexpr std::size_t kDefaultPoolSize = 8;
inline constexpr std::size_t kMaxPoolSize = 32;
inline constexpr const char* kDefaultDmaHeapPath = "/dev/dma_heap/system";
}  // namespace defaults

namespace param_keys
{
inline constexpr const char* kClippingDistance = "clipping_distance";
inline constexpr const char* kMinDepthM = "min_depth_m";
inline constexpr const char* kBilateralSigmaSpatial = "gpu_filter.bilateral_sigma_spatial";
inline constexpr const char* kBilateralSigmaRange = "gpu_filter.bilateral_sigma_range";
inline constexpr const char* kTemporalAlpha = "gpu_filter.temporal_alpha";
inline constexpr const char* kTemporalDelta = "gpu_filter.temporal_delta";
}  // namespace param_keys

enum class PipelinePreset : uint8_t
{
    P0_BYPASS_ONLY = 0,
    P1_RGB_DEPTH = 1,
    P2_RGB_ALIGNED_DEPTH = 2,
    P3_RGB_PC_DEPTH_FRAME = 3,
    P4_RGB_ALIGNED_PC_DEPTH_FRAME = 4,
    P5_RGB_ALIGNED_PC_COLOR_FRAME = 5,
    P6_DEPTH_PC_XYZ = 6
};

enum class PointCloudCoordFrame : uint8_t
{
    DEPTH = 0,
    COLOR = 1
};

enum class UniteImuMethod : uint8_t
{
    NONE = 0,
    COPY = 1,
    LINEAR_INTERPOLATION = 2
};

struct StreamProfileSpec
{
    int width{848};
    int height{480};
    int fps{60};
};

// Dynamic filter constants delivered to Vulkan shaders via Push Constants (zero reallocation).
struct DynamicFilterParams
{
    float min_depth_m{0.10f};
    float max_depth_m{0.0f};  // <= 0.0f disables clipping
    float bilateral_sigma_spatial{2.0f};
    float bilateral_sigma_range{0.05f};
    float temporal_alpha{0.4f};
    float temporal_delta_m{0.05f};
    int32_t rotation_deg{0};  // 0, 90, 180, 270
};

struct NodeConfig
{
    // Stream enables (startup-only)
    bool enable_color{true};
    bool enable_depth{true};
    bool enable_infra1{false};
    bool enable_infra2{false};
    bool enable_gyro{false};
    bool enable_accel{false};

    // Stream profiles (startup-only)
    StreamProfileSpec color_profile{848, 480, 60};
    StreamProfileSpec depth_profile{848, 480, 60};
    StreamProfileSpec infra_profile{848, 480, 60};
    int gyro_fps{200};
    int accel_fps{200};
    UniteImuMethod unite_imu_method{UniteImuMethod::NONE};

    // Pixel format constraints (Fail-Fast on unsupported formats)
    std::string color_format{"rgb8"};  // Only "rgb8" supported
    std::string depth_format{"z16"};   // Only "z16" supported
    std::string infra_format{"y8"};    // "y8" or "y16"

    // Processing toggles (startup-only)
    bool align_depth_enable{true};
    bool pointcloud_enable{true};
    bool pointcloud_ordered{false};
    bool pointcloud_allow_no_texture{false};
    PointCloudCoordFrame pointcloud_coord_frame{PointCloudCoordFrame::DEPTH};
    bool gpu_bilateral_enable{false};
    bool gpu_temporal_enable{false};

    // Playback / time / device
    std::string serial_no;
    std::string rosbag_filename;
    bool rosbag_loop{false};
    bool global_time_enabled{true};
    bool enable_sync{true};
    bool publish_tf{true};
    std::string base_frame_id{"camera_link"};
    std::string tf_prefix;

    // Transport (startup-only)
    bool use_qirp_transport{false};
    std::size_t qirp_dmabuf_pool_size{defaults::kDefaultPoolSize};

    // Initial dynamic filter values
    DynamicFilterParams initial_filter_params;
};

// Parse "WxHxFPS" (e.g. "848x480x60") or "W,H,FPS". Throws std::invalid_argument on malformed input.
StreamProfileSpec parse_stream_profile(const std::string& profile_str);

// Parse "depth" / "color" (case-insensitive). Throws std::invalid_argument otherwise.
PointCloudCoordFrame parse_coord_frame(const std::string& str);

// Parse 0/1/2 or "none"/"copy"/"linear_interpolation".
UniteImuMethod parse_unite_imu_method(const std::string& str);

// Deterministic Fail-Fast router from NodeConfig to one of P0..P6.
// Throws std::invalid_argument on contradictory or unsupported option combinations.
PipelinePreset resolve_pipeline_preset(const NodeConfig& cfg);

const char* preset_to_string(PipelinePreset preset);
bool preset_uses_vulkan(PipelinePreset preset);

}  // namespace realsense_qcom_bolt
