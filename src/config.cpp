#include "realsense_qcom_bolt/config.hpp"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <stdexcept>

namespace realsense_qcom_bolt
{

namespace
{
std::string to_lower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}
}  // namespace

StreamProfileSpec parse_stream_profile(const std::string& profile_str)
{
    std::string normalized = profile_str;
    for (char& ch : normalized)
    {
        if (ch == 'x' || ch == 'X' || ch == ',' || ch == '@')
            ch = ' ';
    }
    std::istringstream iss(normalized);
    StreamProfileSpec spec;
    if (!(iss >> spec.width >> spec.height >> spec.fps))
    {
        throw std::invalid_argument("Invalid stream profile string: '" + profile_str +
                                    "' (expected WxHxFPS, e.g. 848x480x60)");
    }
    std::string trailing;
    if (iss >> trailing)
    {
        throw std::invalid_argument("Unexpected trailing tokens in stream profile: '" + profile_str + "'");
    }
    if (spec.width <= 0 || spec.height <= 0 || spec.fps <= 0)
    {
        throw std::invalid_argument("Stream profile dimensions and FPS must be positive: '" + profile_str + "'");
    }
    return spec;
}

PointCloudCoordFrame parse_coord_frame(const std::string& str)
{
    const std::string low = to_lower(str);
    if (low == "depth" || low == "depth_optical")
        return PointCloudCoordFrame::DEPTH;
    if (low == "color" || low == "color_optical")
        return PointCloudCoordFrame::COLOR;
    throw std::invalid_argument("Unsupported pointcloud.coord_frame: '" + str +
                                "' (supported: 'depth', 'color')");
}

UniteImuMethod parse_unite_imu_method(const std::string& str)
{
    const std::string low = to_lower(str);
    if (low.empty() || low == "0" || low == "none")
        return UniteImuMethod::NONE;
    if (low == "1" || low == "copy")
        return UniteImuMethod::COPY;
    if (low == "2" || low == "linear_interpolation")
        return UniteImuMethod::LINEAR_INTERPOLATION;
    throw std::invalid_argument("Unsupported unite_imu_method: '" + str + "'");
}

PipelinePreset resolve_pipeline_preset(const NodeConfig& cfg)
{
    // Validate supported formats (Fail-Fast)
    const std::string color_fmt = to_lower(cfg.color_format);
    if (cfg.enable_color && color_fmt != "rgb8")
    {
        throw std::invalid_argument(
            "Unsupported color_format '" + cfg.color_format +
            "'. realsense_qcom_bolt supports 'rgb8' only; use realsense2_camera for other formats.");
    }
    const std::string depth_fmt = to_lower(cfg.depth_format);
    if (cfg.enable_depth && depth_fmt != "z16")
    {
        throw std::invalid_argument(
            "Unsupported depth_format '" + cfg.depth_format + "'. Only 'z16' is supported.");
    }
    const std::string infra_fmt = to_lower(cfg.infra_format);
    if ((cfg.enable_infra1 || cfg.enable_infra2) && infra_fmt != "y8" && infra_fmt != "y16")
    {
        throw std::invalid_argument(
            "Unsupported infra_format '" + cfg.infra_format +
            "'. Supported: 'y8', 'y16'. Use realsense2_camera for color-mapped IR.");
    }

    const int rot = cfg.initial_filter_params.rotation_deg;
    if (rot != 0)
    {
        throw std::invalid_argument("Unsupported rotation_deg: " + std::to_string(rot) +
                                    " (only 0 is supported in realsense_qcom_bolt; Fail-Fast)");
    }

    // Validate contradictory stream/processing combinations (Fail-Fast)
    const bool any_stream = cfg.enable_color || cfg.enable_depth ||
                            cfg.enable_infra1 || cfg.enable_infra2 ||
                            cfg.enable_gyro || cfg.enable_accel;
    if (!any_stream)
    {
        throw std::invalid_argument("At least one stream (color/depth/infra/imu) must be enabled.");
    }
    if (cfg.align_depth_enable && (!cfg.enable_depth || !cfg.enable_color))
    {
        throw std::invalid_argument(
            "align_depth.enable:=true requires both enable_depth:=true and enable_color:=true.");
    }
    if (cfg.pointcloud_enable && !cfg.enable_depth)
    {
        throw std::invalid_argument(
            "pointcloud.enable:=true requires enable_depth:=true.");
    }
    if (cfg.pointcloud_enable && cfg.pointcloud_coord_frame == PointCloudCoordFrame::COLOR &&
        (!cfg.enable_color || !cfg.align_depth_enable))
    {
        throw std::invalid_argument(
            "pointcloud.coord_frame:='color' (P5) requires enable_color:=true and align_depth.enable:=true.");
    }

    const bool any_gpu_depth_filter = cfg.gpu_bilateral_enable || cfg.gpu_temporal_enable ||
                                      (cfg.initial_filter_params.max_depth_m > 0.0f);
    if (any_gpu_depth_filter && !cfg.enable_depth)
    {
        throw std::invalid_argument("GPU depth filters require enable_depth:=true.");
    }

    // Route to deterministic fixed preset P0..P6
    if (!cfg.align_depth_enable && !cfg.pointcloud_enable && !any_gpu_depth_filter && rot == 0)
    {
        return PipelinePreset::P0_BYPASS_ONLY;
    }

    if (!cfg.enable_color && cfg.enable_depth && cfg.pointcloud_enable)
    {
        return PipelinePreset::P6_DEPTH_PC_XYZ;
    }

    if (cfg.enable_color && cfg.enable_depth)
    {
        if (!cfg.align_depth_enable && !cfg.pointcloud_enable)
        {
            return PipelinePreset::P1_RGB_DEPTH;
        }
        if (cfg.align_depth_enable && !cfg.pointcloud_enable)
        {
            return PipelinePreset::P2_RGB_ALIGNED_DEPTH;
        }
        if (!cfg.align_depth_enable && cfg.pointcloud_enable)
        {
            return PipelinePreset::P3_RGB_PC_DEPTH_FRAME;
        }
        if (cfg.align_depth_enable && cfg.pointcloud_enable)
        {
            return (cfg.pointcloud_coord_frame == PointCloudCoordFrame::COLOR)
                       ? PipelinePreset::P5_RGB_ALIGNED_PC_COLOR_FRAME
                       : PipelinePreset::P4_RGB_ALIGNED_PC_DEPTH_FRAME;
        }
    }

    throw std::invalid_argument(
        "Requested stream/filter combination is not covered by fixed presets P0..P6. "
        "Use realsense2_camera for arbitrary dynamic filter pipelines.");
}

const char* preset_to_string(PipelinePreset preset)
{
    switch (preset)
    {
        case PipelinePreset::P0_BYPASS_ONLY: return "P0_bypass_only";
        case PipelinePreset::P1_RGB_DEPTH: return "P1_rgb_depth";
        case PipelinePreset::P2_RGB_ALIGNED_DEPTH: return "P2_rgb_aligned_depth";
        case PipelinePreset::P3_RGB_PC_DEPTH_FRAME: return "P3_rgb_pc_depth_frame";
        case PipelinePreset::P4_RGB_ALIGNED_PC_DEPTH_FRAME: return "P4_rgb_aligned_pc_depth_frame";
        case PipelinePreset::P5_RGB_ALIGNED_PC_COLOR_FRAME: return "P5_rgb_aligned_pc_color_frame";
        case PipelinePreset::P6_DEPTH_PC_XYZ: return "P6_depth_pc_xyz";
    }
    return "UNKNOWN";
}

bool preset_uses_vulkan(PipelinePreset preset)
{
    return preset != PipelinePreset::P0_BYPASS_ONLY;
}

}  // namespace realsense_qcom_bolt
