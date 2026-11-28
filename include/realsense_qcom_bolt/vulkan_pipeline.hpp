// Copyright 2026 voidrobot
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "realsense_qcom_bolt/config.hpp"
#include "realsense_qcom_bolt/math_utils.hpp"
#include "realsense_qcom_bolt/output_backend.hpp"
#include "realsense_qcom_bolt/vulkan_context.hpp"

#include <librealsense2/rs.hpp>
#include <std_msgs/msg/header.hpp>

#include <cstddef>
#include <cstdint>
#include <mutex>

namespace realsense_qcom_bolt
{

// std430 layout matching shaders/common_math.glsl
struct alignas(16) GpuIntrinsics
{
    int32_t width{0};
    int32_t height{0};
    float ppx{0.0f};
    float ppy{0.0f};
    float fx{0.0f};
    float fy{0.0f};
    int32_t model{0};
    int32_t pad0{0};
    float coeffs0[4]{0.0f, 0.0f, 0.0f, 0.0f};
    float coeffs1[4]{0.0f, 0.0f, 0.0f, 0.0f};
};

struct alignas(16) GpuExtrinsics
{
    float col0[4]{1.0f, 0.0f, 0.0f, 0.0f};
    float col1[4]{0.0f, 1.0f, 0.0f, 0.0f};
    float col2[4]{0.0f, 0.0f, 1.0f, 0.0f};
    float trans[4]{0.0f, 0.0f, 0.0f, 0.0f};
};

struct alignas(16) GpuCalibData
{
    GpuIntrinsics depth_intrin;
    GpuIntrinsics color_intrin;
    GpuExtrinsics depth_to_color;
    float depth_scale{0.001f};
    float min_depth_m{0.10f};
    float max_depth_m{0.0f};
    uint32_t color_stride_bytes{0};
    uint32_t align_fill_2x2{0};
    uint32_t pad1{0};
    uint32_t pad2{0};
    uint32_t pad3{0};
};

static_assert(sizeof(GpuIntrinsics) == 64, "GpuIntrinsics std430 ABI mismatch");
static_assert(offsetof(GpuIntrinsics, coeffs0) == 32, "GpuIntrinsics::coeffs0 offset mismatch");
static_assert(offsetof(GpuIntrinsics, coeffs1) == 48, "GpuIntrinsics::coeffs1 offset mismatch");
static_assert(sizeof(GpuExtrinsics) == 64, "GpuExtrinsics std430 ABI mismatch");
static_assert(sizeof(GpuCalibData) == 224, "GpuCalibData std430 ABI mismatch");
static_assert(offsetof(GpuCalibData, depth_scale) == 192, "GpuCalibData::depth_scale offset mismatch");
static_assert(offsetof(GpuCalibData, color_stride_bytes) == 204, "GpuCalibData::color_stride_bytes offset mismatch");
static_assert(offsetof(GpuCalibData, align_fill_2x2) == 208, "GpuCalibData::align_fill_2x2 offset mismatch");

struct StreamCalibration
{
    bool has_depth{false};
    bool has_color{false};
    bool has_infra1{false};
    bool has_infra2{false};
    rs2_intrinsics depth_intrin{};
    rs2_intrinsics color_intrin{};
    rs2_intrinsics infra1_intrin{};
    rs2_intrinsics infra2_intrin{};
    rs2_extrinsics depth_to_color{};
    float depth_scale{0.001f};
};

struct ComputeStage
{
    ScopedComputePipeline pipe;
    VkDescriptorSet ds{VK_NULL_HANDLE};

    bool active() const { return pipe.valid(); }
};

// Executes one of the fixed Vulkan 1.3 Compute pipelines (P1..P6) in a single VkCommandBuffer.
// Completely backend-agnostic (zero #ifdef BUILD_WITH_QIRP).
class VulkanPipeline
{
public:
    VulkanPipeline(VulkanContext& ctx,
                   PipelinePreset preset,
                   const NodeConfig& cfg,
                   const StreamCalibration& calib,
                   OutputBackend& output);
    ~VulkanPipeline();

    VulkanPipeline(const VulkanPipeline&) = delete;
    VulkanPipeline& operator=(const VulkanPipeline&) = delete;

    void update_dynamic_params(const DynamicFilterParams& params);

    void process_and_publish(const void* color_data,
                             std::size_t color_bytes,
                             bool color_is_yuyv,
                             const void* depth_z16_data,
                             std::size_t depth_bytes,
                             const std_msgs::msg::Header& color_header,
                             const std_msgs::msg::Header& depth_header);

private:
    struct FrameContext
    {
        DynamicFilterParams params{};
        bool has_color_input{false};
        bool run_yuyv_shader{false};
        bool run_depth_filter{false};
        uint32_t dw{0}, dh{0}, cw{0}, ch{0};
        GpuOutputSlot color_slot;
        GpuOutputSlot pc_slot;
        VkBuffer active_depth_buf{VK_NULL_HANDLE};
    };

    void init_pipelines();
    void init_buffers();
    void init_descriptors();
    void write_calib_ubo(const DynamicFilterParams& params);
    bool has_any_active_subscriber() const;

    bool upload_inputs(FrameContext& fc,
                       const void* color_data,
                       std::size_t color_bytes,
                       bool color_is_yuyv,
                       const void* depth_z16_data,
                       std::size_t depth_bytes);
    void bind_descriptors(const FrameContext& fc);
    void record_and_submit(const FrameContext& fc);
    void publish_outputs(FrameContext& fc,
                         const void* depth_z16_data,
                         const std_msgs::msg::Header& color_header,
                         const std_msgs::msg::Header& depth_header);

    VulkanContext& _ctx;
    PipelinePreset _preset;
    NodeConfig _cfg;
    StreamCalibration _calib;
    OutputBackend& _output;
    PointCloudFieldsMeta _pc_meta;

    // Cache-line isolated dynamic parameter state (written by ROS param callback thread)
    alignas(kCacheLineBytes) std::mutex _param_mtx;
    DynamicFilterParams _dyn_params;

    // Hot-path state (accessed exclusively by capture worker thread)
    alignas(kCacheLineBytes) bool _has_prev_depth{false};
    VkDescriptorPool _desc_pool{VK_NULL_HANDLE};
    VkCommandBuffer _cmd{VK_NULL_HANDLE};
    VkFence _fence{VK_NULL_HANDLE};

    ComputeStage _st_yuyv;
    ComputeStage _st_depth_filter;
    ComputeStage _st_align_init;
    ComputeStage _st_align_scatter;
    ComputeStage _st_align_finalize;
    ComputeStage _st_deproject_gather;
    ComputeStage _st_aligned_pc_fused;
    ComputeStage _st_deproject_xyz;

    // Owned RAII Vulkan buffers (automatically destroyed on scope exit / exception)
    ScopedVulkanBuffer _staging_yuyv;
    ScopedVulkanBuffer _staging_depth;
    ScopedVulkanBuffer _calib_ubo;
    ScopedVulkanBuffer _depth_filtered;
    ScopedVulkanBuffer _depth_prev;
    ScopedVulkanBuffer _zbuf;
    ScopedVulkanBuffer _aligned_out;
    ScopedVulkanBuffer _counter_buf;
};

}  // namespace realsense_qcom_bolt
