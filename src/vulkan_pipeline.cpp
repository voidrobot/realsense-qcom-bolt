// Copyright 2026 voidrobot
// SPDX-License-Identifier: Apache-2.0

#include "realsense_qcom_bolt/vulkan_pipeline.hpp"

#include <librealsense2/rsutil.h>

#include "align_z_finalize.spv.h"
#include "align_z_init.spv.h"
#include "align_z_scatter.spv.h"
#include "aligned_to_xyzrgb_fused.spv.h"
#include "deproject_gather_xyzrgb.spv.h"
#include "deproject_xyz_only.spv.h"
#include "depth_filter.spv.h"
#include "yuyv_to_rgb8.spv.h"

#include <algorithm>
#include <cstring>

namespace realsense_qcom_bolt
{

namespace
{
struct YuyvPush
{
    uint32_t width;
    uint32_t height;
    uint32_t dst_stride_u32;
    uint32_t pad0;
};
static_assert(sizeof(YuyvPush) == 16, "YuyvPush ABI mismatch");

struct DepthFilterPush
{
    uint32_t width;
    uint32_t height;
    float depth_scale;
    float min_depth_m;
    float max_depth_m;
    uint32_t enable_bilateral;
    float sigma_spatial;
    float sigma_range;
    uint32_t enable_temporal;
    float temporal_alpha;
    float temporal_delta_m;
    uint32_t has_prev;
};
static_assert(sizeof(DepthFilterPush) == 48, "DepthFilterPush ABI mismatch");

uint32_t div_up(uint32_t a, uint32_t b) { return (a + b - 1u) / b; }

void cmd_compute_barrier(VkCommandBuffer cmd)
{
    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb, 0, nullptr, 0, nullptr);
}

void cmd_host_read_barrier(VkCommandBuffer cmd)
{
    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT,
                         0, 1, &mb, 0, nullptr, 0, nullptr);
}

GpuIntrinsics to_gpu_intrinsics(const rs2_intrinsics& in)
{
    GpuIntrinsics out{};
    out.width = in.width;
    out.height = in.height;
    out.ppx = in.ppx;
    out.ppy = in.ppy;
    out.fx = in.fx;
    out.fy = in.fy;
    out.model = static_cast<int32_t>(in.model);
    out.coeffs0[0] = in.coeffs[0];
    out.coeffs0[1] = in.coeffs[1];
    out.coeffs0[2] = in.coeffs[2];
    out.coeffs0[3] = in.coeffs[3];
    out.coeffs1[0] = in.coeffs[4];
    return out;
}

GpuExtrinsics to_gpu_extrinsics(const rs2_extrinsics& in)
{
    GpuExtrinsics out{};
    out.col0[0] = in.rotation[0]; out.col0[1] = in.rotation[1]; out.col0[2] = in.rotation[2];
    out.col1[0] = in.rotation[3]; out.col1[1] = in.rotation[4]; out.col1[2] = in.rotation[5];
    out.col2[0] = in.rotation[6]; out.col2[1] = in.rotation[7]; out.col2[2] = in.rotation[8];
    out.trans[0] = in.translation[0]; out.trans[1] = in.translation[1]; out.trans[2] = in.translation[2];
    return out;
}
}  // namespace

VulkanPipeline::VulkanPipeline(VulkanContext& ctx,
                               PipelinePreset preset,
                               const NodeConfig& cfg,
                               const StreamCalibration& calib,
                               OutputBackend& output)
    : _ctx(ctx),
      _preset(preset),
      _cfg(cfg),
      _calib(calib),
      _output(output),
      _pc_meta(make_pointcloud_fields_meta(preset != PipelinePreset::P6_DEPTH_PC_XYZ)),
      _dyn_params(cfg.initial_filter_params)
{
    init_pipelines();
    init_buffers();
    init_descriptors();
    write_calib_ubo(_dyn_params);
    _cmd = _ctx.allocate_command_buffer();
    _fence = _ctx.create_fence(false);
}

VulkanPipeline::~VulkanPipeline()
{
    vkDeviceWaitIdle(_ctx.device());
    if (_fence != VK_NULL_HANDLE)
        vkDestroyFence(_ctx.device(), _fence, nullptr);
    if (_desc_pool != VK_NULL_HANDLE)
        vkDestroyDescriptorPool(_ctx.device(), _desc_pool, nullptr);
}

void VulkanPipeline::init_pipelines()
{
    const bool need_color = (_preset != PipelinePreset::P6_DEPTH_PC_XYZ);
    const bool need_depth_filter = (_cfg.gpu_bilateral_enable || _cfg.gpu_temporal_enable);
    const bool need_align = (_preset == PipelinePreset::P2_RGB_ALIGNED_DEPTH ||
                             _preset == PipelinePreset::P4_RGB_ALIGNED_PC_DEPTH_FRAME ||
                             _preset == PipelinePreset::P5_RGB_ALIGNED_PC_COLOR_FRAME);
    const bool need_gather_pc = (_preset == PipelinePreset::P3_RGB_PC_DEPTH_FRAME ||
                                 _preset == PipelinePreset::P4_RGB_ALIGNED_PC_DEPTH_FRAME);
    const bool need_color_pc = (_preset == PipelinePreset::P5_RGB_ALIGNED_PC_COLOR_FRAME);
    const bool need_xyz_pc = (_preset == PipelinePreset::P6_DEPTH_PC_XYZ);

    if (need_color)
    {
        _st_yuyv.pipe = _ctx.create_compute_pipeline(
            yuyv_to_rgb8_spv, sizeof(yuyv_to_rgb8_spv) / sizeof(uint32_t), 2, sizeof(YuyvPush));
    }

    if (need_depth_filter)
    {
        _st_depth_filter.pipe = _ctx.create_compute_pipeline(
            depth_filter_spv, sizeof(depth_filter_spv) / sizeof(uint32_t), 3, sizeof(DepthFilterPush));
    }

    if (need_align)
    {
        _st_align_init.pipe = _ctx.create_compute_pipeline(
            align_z_init_spv, sizeof(align_z_init_spv) / sizeof(uint32_t), 1, sizeof(uint32_t));
        _st_align_scatter.pipe = _ctx.create_compute_pipeline(
            align_z_scatter_spv, sizeof(align_z_scatter_spv) / sizeof(uint32_t), 3, 0);
        _st_align_finalize.pipe = _ctx.create_compute_pipeline(
            align_z_finalize_spv, sizeof(align_z_finalize_spv) / sizeof(uint32_t), 2, sizeof(uint32_t));
    }

    const uint32_t spec_vals[2] = {
        _cfg.pointcloud_ordered ? 1u : 0u,
        _cfg.pointcloud_allow_no_texture ? 1u : 0u
    };
    const VkSpecializationMapEntry entries[2] = {
        {0, 0, sizeof(uint32_t)},
        {1, sizeof(uint32_t), sizeof(uint32_t)}
    };

    if (need_gather_pc)
    {
        VkSpecializationInfo si{2, entries, sizeof(spec_vals), spec_vals};
        _st_deproject_gather.pipe = _ctx.create_compute_pipeline(
            deproject_gather_xyzrgb_spv, sizeof(deproject_gather_xyzrgb_spv) / sizeof(uint32_t), 5, 0, &si);
    }
    if (need_color_pc)
    {
        VkSpecializationInfo si{1, entries, sizeof(spec_vals), spec_vals};
        _st_aligned_pc_fused.pipe = _ctx.create_compute_pipeline(
            aligned_to_xyzrgb_fused_spv, sizeof(aligned_to_xyzrgb_fused_spv) / sizeof(uint32_t), 5, 0, &si);
    }
    if (need_xyz_pc)
    {
        VkSpecializationInfo si{1, entries, sizeof(spec_vals), spec_vals};
        _st_deproject_xyz.pipe = _ctx.create_compute_pipeline(
            deproject_xyz_only_spv, sizeof(deproject_xyz_only_spv) / sizeof(uint32_t), 4, 0, &si);
    }
}

void VulkanPipeline::init_buffers()
{
    constexpr VkMemoryPropertyFlags kHostCoherent =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    constexpr VkMemoryPropertyFlags kHostCached =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    constexpr VkMemoryPropertyFlags kDeviceLocal = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    const std::size_t dw = static_cast<std::size_t>(_calib.depth_intrin.width);
    const std::size_t dh = static_cast<std::size_t>(_calib.depth_intrin.height);
    const std::size_t depth_bytes = dw * dh * 2u;

    _staging_depth = _ctx.create_buffer(depth_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostCoherent);
    _calib_ubo = _ctx.create_buffer(sizeof(GpuCalibData), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostCoherent);
    _counter_buf = _ctx.create_buffer(sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostCoherent);

    if (_st_depth_filter.active())
    {
        _depth_filtered = _ctx.create_buffer(depth_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostCached);
        _depth_prev = _ctx.create_buffer(depth_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kDeviceLocal);
    }

    if (_st_yuyv.active())
    {
        const std::size_t cw = static_cast<std::size_t>(_calib.color_intrin.width);
        const std::size_t ch = static_cast<std::size_t>(_calib.color_intrin.height);
        _staging_yuyv = _ctx.create_buffer(cw * ch * 2u, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostCoherent);

        if (_st_align_init.active())
        {
            _zbuf = _ctx.create_buffer(cw * ch * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kDeviceLocal);
            _aligned_out = _ctx.create_buffer(cw * ch * 2u, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostCached);
        }
    }
}

void VulkanPipeline::init_descriptors()
{
    _desc_pool = _ctx.create_descriptor_pool(16, 64);

    auto alloc_ds = [this](ComputeStage& st) {
        if (st.active())
            st.ds = _ctx.allocate_descriptor_set(_desc_pool, st.pipe.desc_layout());
    };

    alloc_ds(_st_yuyv);
    alloc_ds(_st_depth_filter);
    alloc_ds(_st_align_init);
    alloc_ds(_st_align_scatter);
    alloc_ds(_st_align_finalize);
    alloc_ds(_st_deproject_gather);
    alloc_ds(_st_aligned_pc_fused);
    alloc_ds(_st_deproject_xyz);

    if (_st_depth_filter.active())
    {
        _ctx.update_storage_buffers(
            _st_depth_filter.ds, {_staging_depth.buffer(), _depth_prev.buffer(), _depth_filtered.buffer()});
    }
    if (_st_align_init.active())
    {
        _ctx.update_storage_buffers(_st_align_init.ds, {_zbuf.buffer()});
        _ctx.update_storage_buffers(_st_align_finalize.ds, {_zbuf.buffer(), _aligned_out.buffer()});
    }
}

void VulkanPipeline::write_calib_ubo(const DynamicFilterParams& params)
{
    GpuCalibData ubo{};
    ubo.depth_intrin = to_gpu_intrinsics(_calib.depth_intrin);
    if (_calib.has_color)
    {
        ubo.color_intrin = to_gpu_intrinsics(_calib.color_intrin);
        ubo.depth_to_color = to_gpu_extrinsics(_calib.depth_to_color);
        ubo.color_stride_bytes = static_cast<uint32_t>(_output.color_output_stride(_calib.color_intrin.width));
        float fov_d[2]{0.0f, 0.0f}, fov_c[2]{0.0f, 0.0f};
        rs2_fov(&_calib.depth_intrin, fov_d);
        rs2_fov(&_calib.color_intrin, fov_c);
        const bool need_fill =
            (static_cast<float>(_calib.depth_intrin.width) / fov_d[0] <
             static_cast<float>(_calib.color_intrin.width) / fov_c[0]) ||
            (static_cast<float>(_calib.depth_intrin.height) / fov_d[1] <
             static_cast<float>(_calib.color_intrin.height) / fov_c[1]);
        ubo.align_fill_2x2 = need_fill ? 1u : 0u;
    }
    ubo.depth_scale = _calib.depth_scale;
    ubo.min_depth_m = params.min_depth_m;
    ubo.max_depth_m = params.max_depth_m;
    std::memcpy(_calib_ubo.mapped(), &ubo, sizeof(GpuCalibData));
}

void VulkanPipeline::update_dynamic_params(const DynamicFilterParams& params)
{
    std::lock_guard<std::mutex> lock(_param_mtx);
    _dyn_params = params;
}

bool VulkanPipeline::has_any_active_subscriber() const
{
    return _output.has_color_subscribers() ||
           _output.has_depth_subscribers() ||
           _output.has_aligned_depth_subscribers() ||
           _output.has_pointcloud_subscribers();
}

bool VulkanPipeline::upload_inputs(FrameContext& fc,
                                   const void* color_data,
                                   std::size_t color_bytes,
                                   bool color_is_yuyv,
                                   const void* depth_z16_data,
                                   std::size_t depth_bytes)
{
    if (depth_z16_data && depth_bytes > 0)
    {
        std::memcpy(_staging_depth.mapped(), depth_z16_data,
                    std::min(depth_bytes, static_cast<std::size_t>(_staging_depth.size())));
    }

    fc.has_color_input = (_st_yuyv.active() && color_data && color_bytes > 0);
    fc.run_yuyv_shader = (fc.has_color_input && color_is_yuyv);
    fc.run_depth_filter = _st_depth_filter.active();

    if (fc.run_yuyv_shader)
    {
        std::memcpy(_staging_yuyv.mapped(), color_data,
                    std::min(color_bytes, static_cast<std::size_t>(_staging_yuyv.size())));
    }

    *static_cast<uint32_t*>(_counter_buf.mapped()) = 0u;

    if (fc.has_color_input)
    {
        fc.color_slot = _output.acquire_gpu_color_slot(static_cast<int>(fc.cw), static_cast<int>(fc.ch));
        if (!fc.color_slot.valid())
            return false;
        if (!color_is_yuyv)
        {
            _output.upload_cpu_rgb_to_slot(fc.color_slot, static_cast<int>(fc.cw), static_cast<int>(fc.ch), color_data);
        }
    }

    if (_st_deproject_gather.active() || _st_aligned_pc_fused.active() || _st_deproject_xyz.active())
    {
        const std::size_t max_pts =
            (_preset == PipelinePreset::P5_RGB_ALIGNED_PC_COLOR_FRAME)
                ? (static_cast<std::size_t>(fc.cw) * static_cast<std::size_t>(fc.ch))
                : (static_cast<std::size_t>(fc.dw) * static_cast<std::size_t>(fc.dh));
        fc.pc_slot = _output.acquire_gpu_pointcloud_slot(max_pts * _pc_meta.point_step);
        if (!fc.pc_slot.valid())
            return false;
    }

    fc.active_depth_buf = fc.run_depth_filter ? _depth_filtered.buffer() : _staging_depth.buffer();
    return true;
}

void VulkanPipeline::bind_descriptors(const FrameContext& fc)
{
    if (fc.run_yuyv_shader)
    {
        _ctx.update_storage_buffers(_st_yuyv.ds, {_staging_yuyv.buffer(), fc.color_slot.vk_buffer});
    }
    if (_st_align_scatter.active())
    {
        _ctx.update_storage_buffers(_st_align_scatter.ds, {fc.active_depth_buf, _calib_ubo.buffer(), _zbuf.buffer()});
    }
    if (_st_deproject_gather.active())
    {
        _ctx.update_storage_buffers(
            _st_deproject_gather.ds,
            {fc.active_depth_buf, fc.color_slot.vk_buffer, _calib_ubo.buffer(), fc.pc_slot.vk_buffer, _counter_buf.buffer()});
    }
    if (_st_aligned_pc_fused.active())
    {
        _ctx.update_storage_buffers(
            _st_aligned_pc_fused.ds,
            {_zbuf.buffer(), fc.color_slot.vk_buffer, _calib_ubo.buffer(), fc.pc_slot.vk_buffer, _counter_buf.buffer()});
    }
    if (_st_deproject_xyz.active())
    {
        _ctx.update_storage_buffers(
            _st_deproject_xyz.ds, {fc.active_depth_buf, _calib_ubo.buffer(), fc.pc_slot.vk_buffer, _counter_buf.buffer()});
    }
}

void VulkanPipeline::record_and_submit(const FrameContext& fc)
{
    vkResetCommandBuffer(_cmd, 0);
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(_cmd, &bi);

    // Pass A: YUYV -> RGB8, optional Depth Filter, and Align Z-Init in parallel
    if (fc.run_yuyv_shader)
    {
        YuyvPush ypc{fc.cw, fc.ch, static_cast<uint32_t>(_output.color_output_stride(static_cast<int>(fc.cw)) / 4), 0u};
        vkCmdBindPipeline(_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _st_yuyv.pipe.pipeline());
        vkCmdBindDescriptorSets(_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _st_yuyv.pipe.layout(), 0, 1, &_st_yuyv.ds, 0, nullptr);
        vkCmdPushConstants(_cmd, _st_yuyv.pipe.layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(YuyvPush), &ypc);
        vkCmdDispatch(_cmd, div_up(fc.cw / 4u, 16u), div_up(fc.ch, 16u), 1u);
    }

    if (fc.run_depth_filter)
    {
        DepthFilterPush fpc{};
        fpc.width = fc.dw;
        fpc.height = fc.dh;
        fpc.depth_scale = _calib.depth_scale;
        fpc.min_depth_m = fc.params.min_depth_m;
        fpc.max_depth_m = fc.params.max_depth_m;
        fpc.enable_bilateral = _cfg.gpu_bilateral_enable ? 1u : 0u;
        fpc.sigma_spatial = fc.params.bilateral_sigma_spatial;
        fpc.sigma_range = fc.params.bilateral_sigma_range;
        fpc.enable_temporal = _cfg.gpu_temporal_enable ? 1u : 0u;
        fpc.temporal_alpha = fc.params.temporal_alpha;
        fpc.temporal_delta_m = fc.params.temporal_delta_m;
        fpc.has_prev = _has_prev_depth ? 1u : 0u;
        vkCmdBindPipeline(_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _st_depth_filter.pipe.pipeline());
        vkCmdBindDescriptorSets(_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _st_depth_filter.pipe.layout(), 0, 1, &_st_depth_filter.ds, 0, nullptr);
        vkCmdPushConstants(_cmd, _st_depth_filter.pipe.layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(DepthFilterPush), &fpc);
        vkCmdDispatch(_cmd, div_up(fc.dw / 2u, 16u), div_up(fc.dh, 16u), 1u);
        _has_prev_depth = true;
    }

    if (_st_align_init.active())
    {
        const uint32_t total_px = fc.cw * fc.ch;
        vkCmdBindPipeline(_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _st_align_init.pipe.pipeline());
        vkCmdBindDescriptorSets(_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _st_align_init.pipe.layout(), 0, 1, &_st_align_init.ds, 0, nullptr);
        vkCmdPushConstants(_cmd, _st_align_init.pipe.layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &total_px);
        vkCmdDispatch(_cmd, div_up(total_px, 256u), 1u, 1u);
    }

    if (_st_align_scatter.active() || _st_deproject_gather.active() || _st_deproject_xyz.active())
    {
        cmd_compute_barrier(_cmd);
    }

    // Pass B: Align Scatter & Depth-Frame PointCloud in parallel
    if (_st_align_scatter.active())
    {
        vkCmdBindPipeline(_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _st_align_scatter.pipe.pipeline());
        vkCmdBindDescriptorSets(_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _st_align_scatter.pipe.layout(), 0, 1, &_st_align_scatter.ds, 0, nullptr);
        vkCmdDispatch(_cmd, div_up(fc.dw, 16u), div_up(fc.dh, 16u), 1u);
    }
    if (_st_deproject_gather.active())
    {
        vkCmdBindPipeline(_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _st_deproject_gather.pipe.pipeline());
        vkCmdBindDescriptorSets(_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _st_deproject_gather.pipe.layout(), 0, 1, &_st_deproject_gather.ds, 0, nullptr);
        vkCmdDispatch(_cmd, div_up(fc.dw, 16u), div_up(fc.dh, 16u), 1u);
    }
    if (_st_deproject_xyz.active())
    {
        vkCmdBindPipeline(_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _st_deproject_xyz.pipe.pipeline());
        vkCmdBindDescriptorSets(_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _st_deproject_xyz.pipe.layout(), 0, 1, &_st_deproject_xyz.ds, 0, nullptr);
        vkCmdDispatch(_cmd, div_up(fc.dw, 16u), div_up(fc.dh, 16u), 1u);
    }

    // Pass C: Align Finalize (+ P5 Fused Color-Frame PointCloud in parallel)
    if (_st_align_finalize.active())
    {
        cmd_compute_barrier(_cmd);
        const uint32_t pair_cnt = (fc.cw * fc.ch) / 2u;
        vkCmdBindPipeline(_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _st_align_finalize.pipe.pipeline());
        vkCmdBindDescriptorSets(_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _st_align_finalize.pipe.layout(), 0, 1, &_st_align_finalize.ds, 0, nullptr);
        vkCmdPushConstants(_cmd, _st_align_finalize.pipe.layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &pair_cnt);
        vkCmdDispatch(_cmd, div_up(pair_cnt, 256u), 1u, 1u);

        if (_st_aligned_pc_fused.active())
        {
            vkCmdBindPipeline(_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _st_aligned_pc_fused.pipe.pipeline());
            vkCmdBindDescriptorSets(_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _st_aligned_pc_fused.pipe.layout(), 0, 1, &_st_aligned_pc_fused.ds, 0, nullptr);
            vkCmdDispatch(_cmd, div_up(fc.cw, 16u), div_up(fc.ch, 16u), 1u);
        }
    }

    // Make all compute writes visible to host reads (standard readback & CPU-mapped DMA-BUF subscribers)
    cmd_host_read_barrier(_cmd);

    vkEndCommandBuffer(_cmd);
    _ctx.submit_and_wait(_cmd, _fence);
}

void VulkanPipeline::publish_outputs(FrameContext& fc,
                                     const void* depth_z16_data,
                                     const std_msgs::msg::Header& color_header,
                                     const std_msgs::msg::Header& depth_header)
{
    if (fc.has_color_input && _output.has_color_subscribers())
    {
        _output.publish_gpu_color_slot(
            color_header, static_cast<int>(fc.cw), static_cast<int>(fc.ch),
            std::move(fc.color_slot), fc.run_yuyv_shader);
    }

    if ((_preset == PipelinePreset::P1_RGB_DEPTH || _preset == PipelinePreset::P6_DEPTH_PC_XYZ) &&
        _output.has_depth_subscribers())
    {
        if (fc.run_depth_filter)
        {
            _ctx.invalidate_buffer(_depth_filtered.get());
            _output.publish_depth_from_cpu(
                depth_header, static_cast<int>(fc.dw), static_cast<int>(fc.dh), _depth_filtered.mapped(), false);
        }
        else
        {
            _output.publish_depth_from_cpu(
                depth_header, static_cast<int>(fc.dw), static_cast<int>(fc.dh), depth_z16_data, false);
        }
    }

    if (_st_align_finalize.active() && _output.has_aligned_depth_subscribers())
    {
        _ctx.invalidate_buffer(_aligned_out.get());
        std_msgs::msg::Header aligned_hdr = color_header;
        aligned_hdr.stamp = depth_header.stamp;
        _output.publish_depth_from_cpu(
            aligned_hdr, static_cast<int>(fc.cw), static_cast<int>(fc.ch), _aligned_out.mapped(), true);
    }

    if (fc.pc_slot.valid() && _output.has_pointcloud_subscribers())
    {
        const bool is_color_frame = (_preset == PipelinePreset::P5_RGB_ALIGNED_PC_COLOR_FRAME);
        const auto& pc_header = is_color_frame ? color_header : depth_header;
        uint32_t pc_w = 0;
        uint32_t pc_h = 1;
        bool is_dense = true;
        if (_cfg.pointcloud_ordered)
        {
            pc_w = is_color_frame ? fc.cw : fc.dw;
            pc_h = is_color_frame ? fc.ch : fc.dh;
            is_dense = false;
        }
        else
        {
            pc_w = *static_cast<const uint32_t*>(_counter_buf.mapped());
            if (pc_w == 0)
                return;
            pc_h = 1;
            is_dense = true;
        }
        _output.publish_gpu_pointcloud_slot(pc_header, pc_w, pc_h, is_dense, _pc_meta, std::move(fc.pc_slot));
    }
}

void VulkanPipeline::process_and_publish(const void* color_data,
                                         std::size_t color_bytes,
                                         bool color_is_yuyv,
                                         const void* depth_z16_data,
                                         std::size_t depth_bytes,
                                         const std_msgs::msg::Header& color_header,
                                         const std_msgs::msg::Header& depth_header)
{
    if (!has_any_active_subscriber())
        return;

    FrameContext fc;
    {
        std::lock_guard<std::mutex> lock(_param_mtx);
        fc.params = _dyn_params;
    }
    fc.dw = static_cast<uint32_t>(_calib.depth_intrin.width);
    fc.dh = static_cast<uint32_t>(_calib.depth_intrin.height);
    fc.cw = static_cast<uint32_t>(_calib.color_intrin.width);
    fc.ch = static_cast<uint32_t>(_calib.color_intrin.height);

    write_calib_ubo(fc.params);
    if (!upload_inputs(fc, color_data, color_bytes, color_is_yuyv, depth_z16_data, depth_bytes))
        return;
    bind_descriptors(fc);
    record_and_submit(fc);
    publish_outputs(fc, depth_z16_data, color_header, depth_header);
}

}  // namespace realsense_qcom_bolt
