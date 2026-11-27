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

#include <cstring>
#include <stdexcept>

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
    _output.set_dmabuf_discard_callback([this](int fd) { evict_dmabuf_fd(fd); });
}

VulkanPipeline::~VulkanPipeline()
{
    _output.set_dmabuf_discard_callback(nullptr);
    vkDeviceWaitIdle(_ctx.device());
    if (_fence != VK_NULL_HANDLE)
        vkDestroyFence(_ctx.device(), _fence, nullptr);
    if (_desc_pool != VK_NULL_HANDLE)
        vkDestroyDescriptorPool(_ctx.device(), _desc_pool, nullptr);

    {
        std::lock_guard<std::mutex> lock(_dmabuf_cache_mtx);
        for (auto& kv : _dmabuf_vk_cache)
        {
            _ctx.destroy_buffer(kv.second);
        }
        _dmabuf_vk_cache.clear();
    }

    _ctx.destroy_buffer(_staging_yuyv);
    _ctx.destroy_buffer(_staging_depth);
    _ctx.destroy_buffer(_calib_ubo);
    _ctx.destroy_buffer(_depth_filtered);
    _ctx.destroy_buffer(_depth_prev);
    _ctx.destroy_buffer(_zbuf);
    _ctx.destroy_buffer(_aligned_out);
    _ctx.destroy_buffer(_counter_buf);
    _ctx.destroy_buffer(_std_rgb_out);
    _ctx.destroy_buffer(_std_pc_out);

    _ctx.destroy_compute_pipeline(_pipe_yuyv);
    _ctx.destroy_compute_pipeline(_pipe_depth_filter);
    _ctx.destroy_compute_pipeline(_pipe_align_init);
    _ctx.destroy_compute_pipeline(_pipe_align_scatter);
    _ctx.destroy_compute_pipeline(_pipe_align_finalize);
    _ctx.destroy_compute_pipeline(_pipe_deproject_gather);
    _ctx.destroy_compute_pipeline(_pipe_aligned_pc_fused);
    _ctx.destroy_compute_pipeline(_pipe_deproject_xyz);
}

void VulkanPipeline::init_pipelines()
{
    const bool need_color = (_preset != PipelinePreset::P6_DEPTH_PC_XYZ);
    const bool need_align = (_preset == PipelinePreset::P2_RGB_ALIGNED_DEPTH ||
                             _preset == PipelinePreset::P4_RGB_ALIGNED_PC_DEPTH_FRAME ||
                             _preset == PipelinePreset::P5_RGB_ALIGNED_PC_COLOR_FRAME);
    const bool need_gather_pc = (_preset == PipelinePreset::P3_RGB_PC_DEPTH_FRAME ||
                                 _preset == PipelinePreset::P4_RGB_ALIGNED_PC_DEPTH_FRAME);
    const bool need_color_pc = (_preset == PipelinePreset::P5_RGB_ALIGNED_PC_COLOR_FRAME);
    const bool need_xyz_pc = (_preset == PipelinePreset::P6_DEPTH_PC_XYZ);

    if (need_color)
    {
        _pipe_yuyv = _ctx.create_compute_pipeline(
            yuyv_to_rgb8_spv, sizeof(yuyv_to_rgb8_spv) / sizeof(uint32_t), 2, sizeof(YuyvPush));
    }

    _pipe_depth_filter = _ctx.create_compute_pipeline(
        depth_filter_spv, sizeof(depth_filter_spv) / sizeof(uint32_t), 3, sizeof(DepthFilterPush));

    if (need_align)
    {
        _pipe_align_init = _ctx.create_compute_pipeline(
            align_z_init_spv, sizeof(align_z_init_spv) / sizeof(uint32_t), 1, sizeof(uint32_t));
        _pipe_align_scatter = _ctx.create_compute_pipeline(
            align_z_scatter_spv, sizeof(align_z_scatter_spv) / sizeof(uint32_t), 3, 0);
        _pipe_align_finalize = _ctx.create_compute_pipeline(
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
        _pipe_deproject_gather = _ctx.create_compute_pipeline(
            deproject_gather_xyzrgb_spv, sizeof(deproject_gather_xyzrgb_spv) / sizeof(uint32_t), 5, 0, &si);
    }
    if (need_color_pc)
    {
        VkSpecializationInfo si{1, entries, sizeof(uint32_t), spec_vals};
        _pipe_aligned_pc_fused = _ctx.create_compute_pipeline(
            aligned_to_xyzrgb_fused_spv, sizeof(aligned_to_xyzrgb_fused_spv) / sizeof(uint32_t), 5, 0, &si);
    }
    if (need_xyz_pc)
    {
        VkSpecializationInfo si{1, entries, sizeof(uint32_t), spec_vals};
        _pipe_deproject_xyz = _ctx.create_compute_pipeline(
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
    _depth_filtered = _ctx.create_buffer(depth_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostCached);
    _depth_prev = _ctx.create_buffer(depth_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kDeviceLocal);
    _calib_ubo = _ctx.create_buffer(sizeof(GpuCalibData), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostCoherent);
    _counter_buf = _ctx.create_buffer(sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostCoherent);

    if (_preset != PipelinePreset::P6_DEPTH_PC_XYZ)
    {
        const std::size_t cw = static_cast<std::size_t>(_calib.color_intrin.width);
        const std::size_t ch = static_cast<std::size_t>(_calib.color_intrin.height);
        _staging_yuyv = _ctx.create_buffer(cw * ch * 2u, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostCoherent);

        if (!_output.use_qirp())
        {
            const std::size_t rgb_bytes = _output.color_output_bytes(static_cast<int>(cw), static_cast<int>(ch)) + 4u;
            _std_rgb_out = _ctx.create_buffer(rgb_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostCached);
        }

        if (_preset == PipelinePreset::P2_RGB_ALIGNED_DEPTH ||
            _preset == PipelinePreset::P4_RGB_ALIGNED_PC_DEPTH_FRAME ||
            _preset == PipelinePreset::P5_RGB_ALIGNED_PC_COLOR_FRAME)
        {
            _zbuf = _ctx.create_buffer(cw * ch * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kDeviceLocal);
            _aligned_out = _ctx.create_buffer(cw * ch * 2u, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostCached);
        }
    }

    if (!_output.use_qirp() &&
        (_preset == PipelinePreset::P3_RGB_PC_DEPTH_FRAME ||
         _preset == PipelinePreset::P4_RGB_ALIGNED_PC_DEPTH_FRAME ||
         _preset == PipelinePreset::P5_RGB_ALIGNED_PC_COLOR_FRAME ||
         _preset == PipelinePreset::P6_DEPTH_PC_XYZ))
    {
        const std::size_t max_pts =
            (_preset == PipelinePreset::P5_RGB_ALIGNED_PC_COLOR_FRAME)
                ? (static_cast<std::size_t>(_calib.color_intrin.width) *
                   static_cast<std::size_t>(_calib.color_intrin.height))
                : (dw * dh);
        _std_pc_out = _ctx.create_buffer(
            max_pts * _pc_meta.point_step, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostCached);
    }
}

void VulkanPipeline::init_descriptors()
{
    _desc_pool = _ctx.create_descriptor_pool(16, 64);

    if (_pipe_yuyv.pipeline != VK_NULL_HANDLE)
        _ds_yuyv = _ctx.allocate_descriptor_set(_desc_pool, _pipe_yuyv.desc_layout);
    if (_pipe_depth_filter.pipeline != VK_NULL_HANDLE)
    {
        _ds_depth_filter = _ctx.allocate_descriptor_set(_desc_pool, _pipe_depth_filter.desc_layout);
        _ctx.update_storage_buffers(
            _ds_depth_filter, {_staging_depth.buffer, _depth_prev.buffer, _depth_filtered.buffer});
    }
    if (_pipe_align_init.pipeline != VK_NULL_HANDLE)
    {
        _ds_align_init = _ctx.allocate_descriptor_set(_desc_pool, _pipe_align_init.desc_layout);
        _ctx.update_storage_buffers(_ds_align_init, {_zbuf.buffer});

        _ds_align_scatter = _ctx.allocate_descriptor_set(_desc_pool, _pipe_align_scatter.desc_layout);
        _ds_align_finalize = _ctx.allocate_descriptor_set(_desc_pool, _pipe_align_finalize.desc_layout);
        _ctx.update_storage_buffers(_ds_align_finalize, {_zbuf.buffer, _aligned_out.buffer});
    }
    if (_pipe_deproject_gather.pipeline != VK_NULL_HANDLE)
        _ds_deproject_gather = _ctx.allocate_descriptor_set(_desc_pool, _pipe_deproject_gather.desc_layout);
    if (_pipe_aligned_pc_fused.pipeline != VK_NULL_HANDLE)
        _ds_aligned_pc_fused = _ctx.allocate_descriptor_set(_desc_pool, _pipe_aligned_pc_fused.desc_layout);
    if (_pipe_deproject_xyz.pipeline != VK_NULL_HANDLE)
        _ds_deproject_xyz = _ctx.allocate_descriptor_set(_desc_pool, _pipe_deproject_xyz.desc_layout);
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
    std::memcpy(_calib_ubo.mapped, &ubo, sizeof(GpuCalibData));
}

void VulkanPipeline::update_dynamic_params(const DynamicFilterParams& params)
{
    std::lock_guard<std::mutex> lock(_param_mtx);
    _dyn_params = params;
}

VkBuffer VulkanPipeline::get_or_import_dmabuf(int fd, std::size_t bytes)
{
    std::lock_guard<std::mutex> lock(_dmabuf_cache_mtx);
    auto it = _dmabuf_vk_cache.find(fd);
    if (it != _dmabuf_vk_cache.end() && it->second.size >= bytes)
    {
        return it->second.buffer;
    }
    if (it != _dmabuf_vk_cache.end())
    {
        _ctx.destroy_buffer(it->second);
        _dmabuf_vk_cache.erase(it);
    }
    VulkanBuffer imported = _ctx.import_dmabuf_buffer(fd, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    VkBuffer handle = imported.buffer;
    _dmabuf_vk_cache.emplace(fd, imported);
    return handle;
}

void VulkanPipeline::evict_dmabuf_fd(int fd)
{
    std::lock_guard<std::mutex> lock(_dmabuf_cache_mtx);
    auto it = _dmabuf_vk_cache.find(fd);
    if (it != _dmabuf_vk_cache.end())
    {
        _ctx.destroy_buffer(it->second);
        _dmabuf_vk_cache.erase(it);
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
    DynamicFilterParams params;
    {
        std::lock_guard<std::mutex> lock(_param_mtx);
        params = _dyn_params;
    }
    write_calib_ubo(params);

    const uint32_t dw = static_cast<uint32_t>(_calib.depth_intrin.width);
    const uint32_t dh = static_cast<uint32_t>(_calib.depth_intrin.height);
    const uint32_t cw = static_cast<uint32_t>(_calib.color_intrin.width);
    const uint32_t ch = static_cast<uint32_t>(_calib.color_intrin.height);

    // 1. Upload raw inputs to HOST_COHERENT staging
    if (depth_z16_data && depth_bytes > 0)
    {
        std::memcpy(_staging_depth.mapped, depth_z16_data, std::min(depth_bytes, static_cast<std::size_t>(_staging_depth.size)));
    }
    const bool has_color_input = (_pipe_yuyv.pipeline != VK_NULL_HANDLE && color_data && color_bytes > 0);
    const bool run_yuyv_shader = (has_color_input && color_is_yuyv);
    if (run_yuyv_shader)
    {
        std::memcpy(_staging_yuyv.mapped, color_data, std::min(color_bytes, static_cast<std::size_t>(_staging_yuyv.size)));
    }

    // Reset atomic compaction counter
    *static_cast<uint32_t*>(_counter_buf.mapped) = 0u;

    // 2. Acquire output buffers (QIRP DMA-BUF Zero-Copy or Standard HOST_CACHED)
    VkBuffer rgb_vk_buf = _std_rgb_out.buffer;
    VkBuffer pc_vk_buf = _std_pc_out.buffer;
    void* rgb_cpu_dst = _std_rgb_out.mapped;
#ifdef BUILD_WITH_QIRP
    std::shared_ptr<lib_mem_dmabuf::DmaBuffer> color_dmabuf;
    std::shared_ptr<lib_mem_dmabuf::DmaBuffer> pc_dmabuf;
    if (_output.use_qirp())
    {
        if (has_color_input)
        {
            color_dmabuf = _output.acquire_color_dmabuf(static_cast<int>(cw), static_cast<int>(ch));
            if (!color_dmabuf || color_dmabuf->fd() < 0)
                return;
            rgb_vk_buf = get_or_import_dmabuf(color_dmabuf->fd(), static_cast<std::size_t>(color_dmabuf->size()));
            rgb_cpu_dst = color_dmabuf->addr();
        }
        if (_pipe_deproject_gather.pipeline != VK_NULL_HANDLE ||
            _pipe_aligned_pc_fused.pipeline != VK_NULL_HANDLE ||
            _pipe_deproject_xyz.pipeline != VK_NULL_HANDLE)
        {
            const std::size_t max_pts =
                (_preset == PipelinePreset::P5_RGB_ALIGNED_PC_COLOR_FRAME)
                    ? (static_cast<std::size_t>(cw) * static_cast<std::size_t>(ch))
                    : (static_cast<std::size_t>(dw) * static_cast<std::size_t>(dh));
            pc_dmabuf = _output.acquire_pointcloud_dmabuf(max_pts * _pc_meta.point_step);
            if (!pc_dmabuf || pc_dmabuf->fd() < 0)
                return;
            pc_vk_buf = get_or_import_dmabuf(pc_dmabuf->fd(), static_cast<std::size_t>(pc_dmabuf->size()));
        }
    }
#endif

    // If input is already RGB8 (e.g., .db3 playback), copy directly into the stride-aligned RGB output buffer.
    if (has_color_input && !color_is_yuyv && rgb_cpu_dst)
    {
        const std::size_t row_bytes = static_cast<std::size_t>(cw) * 3u;
        const std::size_t dst_stride = static_cast<std::size_t>(_output.color_output_stride(static_cast<int>(cw)));
#ifdef BUILD_WITH_QIRP
        if (color_dmabuf)
            color_dmabuf->sync_start();
#endif
        auto* dst = static_cast<uint8_t*>(rgb_cpu_dst);
        const auto* src = static_cast<const uint8_t*>(color_data);
        if (dst_stride == row_bytes)
        {
            std::memcpy(dst, src, static_cast<std::size_t>(ch) * row_bytes);
        }
        else
        {
            for (uint32_t y = 0; y < ch; ++y)
            {
                std::memcpy(dst + static_cast<std::size_t>(y) * dst_stride,
                            src + static_cast<std::size_t>(y) * row_bytes,
                            row_bytes);
            }
        }
#ifdef BUILD_WITH_QIRP
        if (color_dmabuf)
            color_dmabuf->sync_end();
        else
#endif
            _ctx.flush_buffer(_std_rgb_out);
    }

    // 3. Bind active input/output buffers to static descriptor sets
    const bool run_depth_filter = (_cfg.gpu_bilateral_enable || _cfg.gpu_temporal_enable ||
                                   params.max_depth_m > 0.0f);
    VkBuffer active_depth_buf = run_depth_filter ? _depth_filtered.buffer : _staging_depth.buffer;

    if (run_yuyv_shader)
    {
        _ctx.update_storage_buffers(_ds_yuyv, {_staging_yuyv.buffer, rgb_vk_buf});
    }
    if (_pipe_align_scatter.pipeline != VK_NULL_HANDLE)
    {
        _ctx.update_storage_buffers(_ds_align_scatter, {active_depth_buf, _calib_ubo.buffer, _zbuf.buffer});
    }
    if (_pipe_deproject_gather.pipeline != VK_NULL_HANDLE)
    {
        _ctx.update_storage_buffers(
            _ds_deproject_gather, {active_depth_buf, rgb_vk_buf, _calib_ubo.buffer, pc_vk_buf, _counter_buf.buffer});
    }
    if (_pipe_aligned_pc_fused.pipeline != VK_NULL_HANDLE)
    {
        _ctx.update_storage_buffers(
            _ds_aligned_pc_fused, {_zbuf.buffer, rgb_vk_buf, _calib_ubo.buffer, pc_vk_buf, _counter_buf.buffer});
    }
    if (_pipe_deproject_xyz.pipeline != VK_NULL_HANDLE)
    {
        _ctx.update_storage_buffers(
            _ds_deproject_xyz, {active_depth_buf, _calib_ubo.buffer, pc_vk_buf, _counter_buf.buffer});
    }

    // 4. Record single fused VkCommandBuffer
    vkResetCommandBuffer(_cmd, 0);
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(_cmd, &bi);

    // Pass A: YUYV -> RGB8 (and Depth Filter / Align Z-Init in parallel!)
    if (run_yuyv_shader)
    {
        YuyvPush ypc{cw, ch, static_cast<uint32_t>(_output.color_output_stride(static_cast<int>(cw)) / 4), 0u};
        vkCmdBindPipeline(_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _pipe_yuyv.pipeline);
        vkCmdBindDescriptorSets(_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _pipe_yuyv.pipeline_layout, 0, 1, &_ds_yuyv, 0, nullptr);
        vkCmdPushConstants(_cmd, _pipe_yuyv.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(YuyvPush), &ypc);
        vkCmdDispatch(_cmd, div_up(cw / 4u, 16u), div_up(ch, 16u), 1u);
    }

    if (run_depth_filter)
    {
        DepthFilterPush fpc{};
        fpc.width = dw;
        fpc.height = dh;
        fpc.depth_scale = _calib.depth_scale;
        fpc.min_depth_m = params.min_depth_m;
        fpc.max_depth_m = params.max_depth_m;
        fpc.enable_bilateral = _cfg.gpu_bilateral_enable ? 1u : 0u;
        fpc.sigma_spatial = params.bilateral_sigma_spatial;
        fpc.sigma_range = params.bilateral_sigma_range;
        fpc.enable_temporal = _cfg.gpu_temporal_enable ? 1u : 0u;
        fpc.temporal_alpha = params.temporal_alpha;
        fpc.temporal_delta_m = params.temporal_delta_m;
        fpc.has_prev = _has_prev_depth ? 1u : 0u;
        vkCmdBindPipeline(_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _pipe_depth_filter.pipeline);
        vkCmdBindDescriptorSets(_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _pipe_depth_filter.pipeline_layout, 0, 1, &_ds_depth_filter, 0, nullptr);
        vkCmdPushConstants(_cmd, _pipe_depth_filter.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(DepthFilterPush), &fpc);
        vkCmdDispatch(_cmd, div_up(dw / 2u, 16u), div_up(dh, 16u), 1u);
        _has_prev_depth = true;
    }

    if (_pipe_align_init.pipeline != VK_NULL_HANDLE)
    {
        const uint32_t total_px = cw * ch;
        vkCmdBindPipeline(_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _pipe_align_init.pipeline);
        vkCmdBindDescriptorSets(_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _pipe_align_init.pipeline_layout, 0, 1, &_ds_align_init, 0, nullptr);
        vkCmdPushConstants(_cmd, _pipe_align_init.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &total_px);
        vkCmdDispatch(_cmd, div_up(total_px, 256u), 1u, 1u);
    }

    // Barrier after Pass A
    if (_pipe_align_scatter.pipeline != VK_NULL_HANDLE ||
        _pipe_deproject_gather.pipeline != VK_NULL_HANDLE ||
        _pipe_deproject_xyz.pipeline != VK_NULL_HANDLE)
    {
        cmd_compute_barrier(_cmd);
    }

    // Pass B: Align Scatter & Depth-Frame PointCloud (can execute in parallel on P4!)
    if (_pipe_align_scatter.pipeline != VK_NULL_HANDLE)
    {
        vkCmdBindPipeline(_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _pipe_align_scatter.pipeline);
        vkCmdBindDescriptorSets(_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _pipe_align_scatter.pipeline_layout, 0, 1, &_ds_align_scatter, 0, nullptr);
        vkCmdDispatch(_cmd, div_up(dw, 16u), div_up(dh, 16u), 1u);
    }

    if (_pipe_deproject_gather.pipeline != VK_NULL_HANDLE)
    {
        vkCmdBindPipeline(_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _pipe_deproject_gather.pipeline);
        vkCmdBindDescriptorSets(_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _pipe_deproject_gather.pipeline_layout, 0, 1, &_ds_deproject_gather, 0, nullptr);
        vkCmdDispatch(_cmd, div_up(dw, 16u), div_up(dh, 16u), 1u);
    }

    if (_pipe_deproject_xyz.pipeline != VK_NULL_HANDLE)
    {
        vkCmdBindPipeline(_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _pipe_deproject_xyz.pipeline);
        vkCmdBindDescriptorSets(_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _pipe_deproject_xyz.pipeline_layout, 0, 1, &_ds_deproject_xyz, 0, nullptr);
        vkCmdDispatch(_cmd, div_up(dw, 16u), div_up(dh, 16u), 1u);
    }

    // Pass C: Align Finalize (+ P5 Fused Color-Frame PointCloud in parallel!)
    if (_pipe_align_finalize.pipeline != VK_NULL_HANDLE)
    {
        cmd_compute_barrier(_cmd);
        const uint32_t pair_cnt = (cw * ch) / 2u;
        vkCmdBindPipeline(_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _pipe_align_finalize.pipeline);
        vkCmdBindDescriptorSets(_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _pipe_align_finalize.pipeline_layout, 0, 1, &_ds_align_finalize, 0, nullptr);
        vkCmdPushConstants(_cmd, _pipe_align_finalize.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &pair_cnt);
        vkCmdDispatch(_cmd, div_up(pair_cnt, 256u), 1u, 1u);

        if (_pipe_aligned_pc_fused.pipeline != VK_NULL_HANDLE)
        {
            vkCmdBindPipeline(_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _pipe_aligned_pc_fused.pipeline);
            vkCmdBindDescriptorSets(_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _pipe_aligned_pc_fused.pipeline_layout, 0, 1, &_ds_aligned_pc_fused, 0, nullptr);
            vkCmdDispatch(_cmd, div_up(cw, 16u), div_up(ch, 16u), 1u);
        }
    }

    vkEndCommandBuffer(_cmd);
    _ctx.submit_and_wait(_cmd, _fence);

    // 5. Publish outputs
    if (has_color_input && _output.has_color_subscribers())
    {
#ifdef BUILD_WITH_QIRP
        if (_output.use_qirp())
        {
            _output.publish_color_dmabuf(color_header, static_cast<int>(cw), static_cast<int>(ch), std::move(color_dmabuf));
        }
        else
#endif
        {
            if (run_yuyv_shader)
                _ctx.invalidate_buffer(_std_rgb_out);
            _output.publish_color_from_cpu(
                color_header, static_cast<int>(cw), static_cast<int>(ch),
                _output.color_output_stride(static_cast<int>(cw)), _std_rgb_out.mapped);
        }
    }

    if ((_preset == PipelinePreset::P1_RGB_DEPTH || _preset == PipelinePreset::P6_DEPTH_PC_XYZ) &&
        _output.has_depth_subscribers())
    {
        if (run_depth_filter)
        {
            _ctx.invalidate_buffer(_depth_filtered);
            _output.publish_depth_from_cpu(depth_header, static_cast<int>(dw), static_cast<int>(dh), _depth_filtered.mapped, false);
        }
        else
        {
            _output.publish_depth_from_cpu(depth_header, static_cast<int>(dw), static_cast<int>(dh), depth_z16_data, false);
        }
    }

    if (_pipe_align_finalize.pipeline != VK_NULL_HANDLE && _output.has_aligned_depth_subscribers())
    {
        _ctx.invalidate_buffer(_aligned_out);
        std_msgs::msg::Header aligned_hdr = color_header;
        aligned_hdr.stamp = depth_header.stamp;
        _output.publish_depth_from_cpu(aligned_hdr, static_cast<int>(cw), static_cast<int>(ch), _aligned_out.mapped, true);
    }

    if ((_pipe_deproject_gather.pipeline != VK_NULL_HANDLE ||
         _pipe_aligned_pc_fused.pipeline != VK_NULL_HANDLE ||
         _pipe_deproject_xyz.pipeline != VK_NULL_HANDLE) &&
        _output.has_pointcloud_subscribers())
    {
        const bool is_color_frame = (_preset == PipelinePreset::P5_RGB_ALIGNED_PC_COLOR_FRAME);
        const auto& pc_header = is_color_frame ? color_header : depth_header;
        uint32_t pc_w = 0;
        uint32_t pc_h = 1;
        bool is_dense = true;
        if (_cfg.pointcloud_ordered)
        {
            pc_w = is_color_frame ? cw : dw;
            pc_h = is_color_frame ? ch : dh;
            is_dense = false;
        }
        else
        {
            pc_w = *static_cast<const uint32_t*>(_counter_buf.mapped);
            if (pc_w == 0)
                return;  // Guard against empty/nullptr PointCloud2 publication
            pc_h = 1;
            is_dense = true;
        }

#ifdef BUILD_WITH_QIRP
        if (_output.use_qirp())
        {
            _output.publish_pointcloud_dmabuf(pc_header, pc_w, pc_h, is_dense, _pc_meta, std::move(pc_dmabuf));
        }
        else
#endif
        {
            const std::size_t valid_bytes = static_cast<std::size_t>(pc_w) * static_cast<std::size_t>(pc_h) * _pc_meta.point_step;
            _ctx.invalidate_buffer(_std_pc_out, 0, valid_bytes);
            _output.publish_pointcloud_from_cpu(pc_header, pc_w, pc_h, is_dense, _pc_meta, _std_pc_out.mapped, valid_bytes);
        }
    }
}

}  // namespace realsense_qcom_bolt
