#include "realsense_qcom_bolt/config.hpp"

#include <cassert>
#include <iostream>
#include <stdexcept>

using namespace realsense_qcom_bolt;

static void expect_throws(const NodeConfig& cfg, const char* label)
{
    bool threw = false;
    try
    {
        (void)resolve_pipeline_preset(cfg);
    }
    catch (const std::invalid_argument&)
    {
        threw = true;
    }
    if (!threw)
    {
        std::cerr << "FAIL: expected invalid_argument for " << label << std::endl;
        std::abort();
    }
}

int main()
{
    // Profile parser
    {
        auto p = parse_stream_profile("848x480x60");
        assert(p.width == 848 && p.height == 480 && p.fps == 60);
        auto p2 = parse_stream_profile("1280,720,30");
        assert(p2.width == 1280 && p2.height == 720 && p2.fps == 30);
    }

    // P0: bypass_only (RGB only, IR only, raw Depth only, RGB+Depth no GPU filters)
    {
        NodeConfig c;
        c.enable_color = true;
        c.enable_depth = false;
        c.align_depth_enable = false;
        c.pointcloud_enable = false;
        assert(resolve_pipeline_preset(c) == PipelinePreset::P0_BYPASS_ONLY);
        assert(!preset_uses_vulkan(PipelinePreset::P0_BYPASS_ONLY));

        c.enable_color = false;
        c.enable_infra1 = true;
        assert(resolve_pipeline_preset(c) == PipelinePreset::P0_BYPASS_ONLY);
    }

    // P1: rgb_depth (with GPU filter or clipping)
    {
        NodeConfig c;
        c.enable_color = true;
        c.enable_depth = true;
        c.align_depth_enable = false;
        c.pointcloud_enable = false;
        c.gpu_bilateral_enable = true;
        assert(resolve_pipeline_preset(c) == PipelinePreset::P1_RGB_DEPTH);
    }

    // P2: rgb_aligned_depth
    {
        NodeConfig c;
        c.enable_color = true;
        c.enable_depth = true;
        c.align_depth_enable = true;
        c.pointcloud_enable = false;
        assert(resolve_pipeline_preset(c) == PipelinePreset::P2_RGB_ALIGNED_DEPTH);
    }

    // P3: rgb_pc_depth_frame
    {
        NodeConfig c;
        c.enable_color = true;
        c.enable_depth = true;
        c.align_depth_enable = false;
        c.pointcloud_enable = true;
        assert(resolve_pipeline_preset(c) == PipelinePreset::P3_RGB_PC_DEPTH_FRAME);
    }

    // P4: rgb_aligned_pc_depth_frame
    {
        NodeConfig c;
        c.enable_color = true;
        c.enable_depth = true;
        c.align_depth_enable = true;
        c.pointcloud_enable = true;
        c.pointcloud_coord_frame = PointCloudCoordFrame::DEPTH;
        assert(resolve_pipeline_preset(c) == PipelinePreset::P4_RGB_ALIGNED_PC_DEPTH_FRAME);
    }

    // P5: rgb_aligned_pc_color_frame
    {
        NodeConfig c;
        c.enable_color = true;
        c.enable_depth = true;
        c.align_depth_enable = true;
        c.pointcloud_enable = true;
        c.pointcloud_coord_frame = PointCloudCoordFrame::COLOR;
        assert(resolve_pipeline_preset(c) == PipelinePreset::P5_RGB_ALIGNED_PC_COLOR_FRAME);
    }

    // P6: depth_pc_xyz
    {
        NodeConfig c;
        c.enable_color = false;
        c.enable_depth = true;
        c.align_depth_enable = false;
        c.pointcloud_enable = true;
        assert(resolve_pipeline_preset(c) == PipelinePreset::P6_DEPTH_PC_XYZ);
    }

    // Fail-Fast checks
    {
        NodeConfig c;
        c.enable_color = false;
        c.enable_depth = false;
        c.align_depth_enable = false;
        c.pointcloud_enable = false;
        expect_throws(c, "no streams enabled");

        c.enable_color = true;
        c.pointcloud_enable = true;
        expect_throws(c, "pointcloud without depth");

        c.pointcloud_enable = false;
        c.align_depth_enable = true;
        expect_throws(c, "align without depth");

        c.enable_depth = true;
        c.align_depth_enable = false;
        c.color_format = "bgr8";
        expect_throws(c, "unsupported color format bgr8");

        c.color_format = "rgb8";
        c.enable_infra1 = true;
        c.infra_format = "uyvy";
        expect_throws(c, "unsupported infra format uyvy");

        c.enable_infra1 = false;
        c.pointcloud_enable = true;
        c.pointcloud_coord_frame = PointCloudCoordFrame::COLOR;
        expect_throws(c, "color coord_frame without align_depth");

        c.pointcloud_enable = false;
        c.pointcloud_coord_frame = PointCloudCoordFrame::DEPTH;
        c.initial_filter_params.rotation_deg = 90;
        expect_throws(c, "unsupported rotation_deg=90 Fail-Fast");

        c.initial_filter_params.rotation_deg = 180;
        expect_throws(c, "unsupported rotation_deg=180 Fail-Fast");

        c.initial_filter_params.rotation_deg = 0;
        c.enable_depth = false;
        c.gpu_bilateral_enable = true;
        expect_throws(c, "gpu filter without depth");
    }

    assert(parse_coord_frame("depth") == PointCloudCoordFrame::DEPTH);
    assert(parse_coord_frame("COLOR") == PointCloudCoordFrame::COLOR);
    assert(parse_unite_imu_method("copy") == UniteImuMethod::COPY);
    assert(parse_unite_imu_method("linear_interpolation") == UniteImuMethod::LINEAR_INTERPOLATION);
    assert(parse_unite_imu_method("none") == UniteImuMethod::NONE);

    std::cout << "ALL_CONFIG_ROUTER_TESTS_PASSED" << std::endl;
    return 0;
}
