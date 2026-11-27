#pragma once

#include "realsense_qcom_bolt/config.hpp"
#include "realsense_qcom_bolt/output_backend.hpp"
#include "realsense_qcom_bolt/vulkan_context.hpp"

#include <librealsense2/rs.hpp>
#include <std_msgs/msg/header.hpp>

#include <cstdint>
#include <mutex>
#include <unordered_map>

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

struct StreamCalibration
{
    bool has_depth{false};
    bool has_color{false};
    rs2_intrinsics depth_intrin{};
    rs2_intrinsics color_intrin{};
    rs2_extrinsics depth_to_color{};
    float depth_scale{0.001f};
};

// Executes one of the fixed Vulkan 1.3 Compute pipelines (P1..P6) in a single VkCommandBuffer.
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

    // Update dynamic filter constants (delivered via Push Constants / Calib UBO, zero reallocation).
    void update_dynamic_params(const DynamicFilterParams& params);

    // Execute the fixed pipeline for one synchronized frameset and publish outputs.
    void process_and_publish(const void* color_data,
                             std::size_t color_bytes,
                             bool color_is_yuyv,
                             const void* depth_z16_data,
                             std::size_t depth_bytes,
                             const std_msgs::msg::Header& color_header,
                             const std_msgs::msg::Header& depth_header);

private:
    void init_pipelines();
    void init_buffers();
    void init_descriptors();
    void write_calib_ubo(const DynamicFilterParams& params);
    VkBuffer get_or_import_dmabuf(int fd, std::size_t bytes);
    void evict_dmabuf_fd(int fd);

    VulkanContext& _ctx;
    PipelinePreset _preset;
    NodeConfig _cfg;
    StreamCalibration _calib;
    OutputBackend& _output;
    PointCloudFieldsMeta _pc_meta;

    std::mutex _param_mtx;
    DynamicFilterParams _dyn_params;
    bool _has_prev_depth{false};

    // Pipelines (only those needed by _preset are created)
    ComputePipelineHandle _pipe_yuyv;
    ComputePipelineHandle _pipe_depth_filter;
    ComputePipelineHandle _pipe_align_init;
    ComputePipelineHandle _pipe_align_scatter;
    ComputePipelineHandle _pipe_align_finalize;
    ComputePipelineHandle _pipe_deproject_gather;
    ComputePipelineHandle _pipe_aligned_pc_fused;
    ComputePipelineHandle _pipe_deproject_xyz;

    // Static descriptor pool & sets (no VK_KHR_push_descriptor)
    VkDescriptorPool _desc_pool{VK_NULL_HANDLE};
    VkDescriptorSet _ds_yuyv{VK_NULL_HANDLE};
    VkDescriptorSet _ds_depth_filter{VK_NULL_HANDLE};
    VkDescriptorSet _ds_align_init{VK_NULL_HANDLE};
    VkDescriptorSet _ds_align_scatter{VK_NULL_HANDLE};
    VkDescriptorSet _ds_align_finalize{VK_NULL_HANDLE};
    VkDescriptorSet _ds_deproject_gather{VK_NULL_HANDLE};
    VkDescriptorSet _ds_aligned_pc_fused{VK_NULL_HANDLE};
    VkDescriptorSet _ds_deproject_xyz{VK_NULL_HANDLE};

    // Command buffer & fence
    VkCommandBuffer _cmd{VK_NULL_HANDLE};
    VkFence _fence{VK_NULL_HANDLE};

    // Fixed buffers
    VulkanBuffer _staging_yuyv;
    VulkanBuffer _staging_depth;
    VulkanBuffer _calib_ubo;
    VulkanBuffer _depth_filtered;
    VulkanBuffer _depth_prev;
    VulkanBuffer _zbuf;
    VulkanBuffer _aligned_out;
    VulkanBuffer _counter_buf;
    VulkanBuffer _std_rgb_out;
    VulkanBuffer _std_pc_out;

    // Cached DMA-BUF fd -> imported VkBuffer (QIRP Zero-Copy output)
    std::mutex _dmabuf_cache_mtx;
    std::unordered_map<int, VulkanBuffer> _dmabuf_vk_cache;
};

}  // namespace realsense_qcom_bolt
