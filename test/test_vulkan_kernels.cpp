#include "realsense_qcom_bolt/math_utils.hpp"
#include "realsense_qcom_bolt/vulkan_context.hpp"

#include "align_z_finalize.spv.h"
#include "align_z_init.spv.h"
#include "align_z_scatter.spv.h"
#include "deproject_gather_xyzrgb.spv.h"
#include "depth_filter.spv.h"
#include "yuyv_to_rgb8.spv.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

using namespace realsense_qcom_bolt;

namespace
{
struct alignas(16) TestIntrinsics
{
    int32_t width, height;
    float ppx, ppy, fx, fy;
    int32_t model, pad0;
    float coeffs0[4];
    float coeffs1[4];
};
struct alignas(16) TestExtrinsics
{
    float col0[4], col1[4], col2[4], trans[4];
};
struct alignas(16) TestCalib
{
    TestIntrinsics depth_intrin;
    TestIntrinsics color_intrin;
    TestExtrinsics depth_to_color;
    float depth_scale, min_depth_m, max_depth_m;
    uint32_t color_stride_bytes, align_fill_2x2, pad1, pad2, pad3;
};

void cpu_bt601(uint8_t y, uint8_t u, uint8_t v, uint8_t& r, uint8_t& g, uint8_t& b)
{
    int c = static_cast<int>(y) - 16;
    int d = static_cast<int>(u) - 128;
    int e = static_cast<int>(v) - 128;
    r = static_cast<uint8_t>(std::clamp((298 * c + 409 * e + 128) >> 8, 0, 255));
    g = static_cast<uint8_t>(std::clamp((298 * c - 100 * d - 208 * e + 128) >> 8, 0, 255));
    b = static_cast<uint8_t>(std::clamp((298 * c + 516 * d + 128) >> 8, 0, 255));
}

uint32_t div_up(uint32_t a, uint32_t b) { return (a + b - 1u) / b; }
}  // namespace

int main()
{
    VulkanContext ctx;
    assert(ctx.non_coherent_atom_size() >= 1u);
    assert(ctx.buffer_align_bytes() >= 64u);

    constexpr VkMemoryPropertyFlags kCoherent =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    constexpr VkMemoryPropertyFlags kCached =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;

    VkDescriptorPool pool = ctx.create_descriptor_pool(16, 64);
    VkCommandBuffer cmd = ctx.allocate_command_buffer();
    VkFence fence = ctx.create_fence(false);

    // 1. Test yuyv_to_rgb8.comp with 64px padded stride (matches QIRP stride)
    {
        const uint32_t W = 64, H = 16;
        const uint32_t stride_bytes = 256;  // > W*3 (192), multiple of 64
        auto in_buf = ctx.create_buffer(W * H * 2u, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kCoherent);
        auto out_buf = ctx.create_buffer(stride_bytes * H, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kCached);
        auto pipe = ctx.create_compute_pipeline(
            yuyv_to_rgb8_spv, sizeof(yuyv_to_rgb8_spv) / sizeof(uint32_t), 2, 16);
        VkDescriptorSet ds = ctx.allocate_descriptor_set(pool, pipe.desc_layout());
        ctx.update_storage_buffers(ds, {in_buf.buffer(), out_buf.buffer()});

        auto* yuyv = static_cast<uint8_t*>(in_buf.mapped());
        for (uint32_t y = 0; y < H; ++y)
        {
            for (uint32_t x = 0; x < W; x += 2)
            {
                const std::size_t off = (y * W + x) * 2u;
                yuyv[off + 0] = static_cast<uint8_t>(40 + (x * 3) % 180);
                yuyv[off + 1] = static_cast<uint8_t>(80 + (y * 5) % 90);
                yuyv[off + 2] = static_cast<uint8_t>(50 + (x * 2) % 170);
                yuyv[off + 3] = static_cast<uint8_t>(90 + (x + y * 3) % 80);
            }
        }

        struct { uint32_t w, h, s, p; } pc{W, H, stride_bytes / 4u, 0u};
        vkResetCommandBuffer(cmd, 0);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
        vkBeginCommandBuffer(cmd, &bi);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipe.layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, div_up(W / 4u, 16u), div_up(H, 16u), 1u);
        vkEndCommandBuffer(cmd);
        ctx.submit_and_wait(cmd, fence);
        ctx.invalidate_buffer(out_buf.get());

        const auto* rgb = static_cast<const uint8_t*>(out_buf.mapped());
        for (uint32_t y = 0; y < H; ++y)
        {
            for (uint32_t x = 0; x < W; x += 2)
            {
                const std::size_t io = (y * W + x) * 2u;
                uint8_t r0, g0, b0, r1, g1, b1;
                cpu_bt601(yuyv[io + 0], yuyv[io + 1], yuyv[io + 3], r0, g0, b0);
                cpu_bt601(yuyv[io + 2], yuyv[io + 1], yuyv[io + 3], r1, g1, b1);
                const std::size_t oo = y * stride_bytes + x * 3u;
                assert(rgb[oo + 0] == r0 && rgb[oo + 1] == g0 && rgb[oo + 2] == b0);
                assert(rgb[oo + 3] == r1 && rgb[oo + 4] == g1 && rgb[oo + 5] == b1);
            }
        }
    }

    // 2. Test deproject_gather_xyzrgb.comp (dense compaction) + depth_filter.comp (clipping)
    {
        const uint32_t W = 32, H = 16;
        const uint32_t rgb_stride = W * 3u;  // 96B (multiple of 4, exact tight buffer -> tests OOB guard!)
        auto depth_buf = ctx.create_buffer(W * H * 2u, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kCoherent);
        auto rgb_buf = ctx.create_buffer(rgb_stride * H, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kCoherent);
        auto calib_buf = ctx.create_buffer(sizeof(TestCalib), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kCoherent);
        auto pc_buf = ctx.create_buffer(W * H * 16u, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kCached);
        auto cnt_buf = ctx.create_buffer(sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kCoherent);

        // Fill half pixels with z=1000mm (1.0m), half with z=0
        auto* z16 = static_cast<uint16_t*>(depth_buf.mapped());
        for (uint32_t i = 0; i < W * H; ++i)
            z16[i] = (i % 2u == 0u) ? 1000u : 0u;
        auto* rgb = static_cast<uint8_t*>(rgb_buf.mapped());
        for (uint32_t i = 0; i < W * H; ++i)
        {
            rgb[i * 3 + 0] = 10;
            rgb[i * 3 + 1] = 20;
            rgb[i * 3 + 2] = 30;
        }
        *static_cast<uint32_t*>(cnt_buf.mapped()) = 0u;

        TestCalib cal{};
        cal.depth_intrin = {static_cast<int>(W), static_cast<int>(H), 16.0f, 8.0f, 100.0f, 100.0f, 0, 0, {0}, {0}};
        cal.color_intrin = cal.depth_intrin;
        cal.depth_to_color = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 0}};
        cal.depth_scale = 0.001f;
        cal.min_depth_m = 0.1f;
        cal.max_depth_m = 5.0f;
        cal.color_stride_bytes = rgb_stride;
        std::memcpy(calib_buf.mapped(), &cal, sizeof(cal));

        const uint32_t spec_vals[2] = {0u, 0u};  // dense compaction
        const VkSpecializationMapEntry entries[2] = {{0, 0, 4}, {1, 4, 4}};
        VkSpecializationInfo si{2, entries, sizeof(spec_vals), spec_vals};
        auto pipe = ctx.create_compute_pipeline(
            deproject_gather_xyzrgb_spv, sizeof(deproject_gather_xyzrgb_spv) / sizeof(uint32_t), 5, 0, &si);
        VkDescriptorSet ds = ctx.allocate_descriptor_set(pool, pipe.desc_layout());
        ctx.update_storage_buffers(ds, {depth_buf.buffer(), rgb_buf.buffer(), calib_buf.buffer(), pc_buf.buffer(), cnt_buf.buffer()});

        vkResetCommandBuffer(cmd, 0);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
        vkBeginCommandBuffer(cmd, &bi);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.layout(), 0, 1, &ds, 0, nullptr);
        vkCmdDispatch(cmd, div_up(W, 16u), div_up(H, 16u), 1u);
        vkEndCommandBuffer(cmd);
        ctx.submit_and_wait(cmd, fence);

        const uint32_t valid_cnt = *static_cast<const uint32_t*>(cnt_buf.mapped());
        assert(valid_cnt == (W * H) / 2u);
        // Test non-atom-aligned partial invalidation (valid_cnt * 16 may not be multiple of nonCoherentAtomSize)
        ctx.invalidate_buffer(pc_buf.get(), 0, valid_cnt * 16u);
        const auto* pts = static_cast<const uint8_t*>(pc_buf.mapped());
        for (uint32_t i = 0; i < valid_cnt; ++i)
        {
            float z = 0.0f;
            std::memcpy(&z, pts + i * 16u + 8u, sizeof(float));
            assert(std::abs(z - 1.0f) < 1e-5f);
            // ROS BGR byte order: B=30, G=20, R=10
            assert(pts[i * 16u + 12u] == 30 && pts[i * 16u + 13u] == 20 && pts[i * 16u + 14u] == 10);
        }
    }

    // 3. Test depth_filter.comp (Clipping + 5x5 Bilateral + Temporal EMA 2-frame sequence)
    {
        const uint32_t W = 32, H = 16;
        const std::size_t n_pix = static_cast<std::size_t>(W) * H;
        const std::size_t bytes = n_pix * sizeof(uint16_t);
        auto in_buf = ctx.create_buffer(bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kCoherent);
        auto prev_buf = ctx.create_buffer(bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kCoherent);
        auto out_buf = ctx.create_buffer(bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kCached);

        struct DepthFilterPush
        {
            uint32_t width, height;
            float depth_scale, min_depth_m, max_depth_m;
            uint32_t enable_bilateral;
            float sigma_spatial, sigma_range;
            uint32_t enable_temporal;
            float temporal_alpha, temporal_delta_m;
            uint32_t has_prev;
        };
        static_assert(sizeof(DepthFilterPush) == 48u, "DepthFilterPush size mismatch");

        auto pipe = ctx.create_compute_pipeline(
            depth_filter_spv, sizeof(depth_filter_spv) / sizeof(uint32_t), 3, sizeof(DepthFilterPush));
        VkDescriptorSet ds = ctx.allocate_descriptor_set(pool, pipe.desc_layout());
        ctx.update_storage_buffers(ds, {in_buf.buffer(), prev_buf.buffer(), out_buf.buffer()});

        auto cpu_depth_filter = [&](const std::vector<uint16_t>& src,
                                    std::vector<uint16_t>& prev,
                                    const DepthFilterPush& p) -> std::vector<uint16_t> {
            std::vector<uint16_t> out(n_pix, 0u);
            auto read_z = [&](int x, int y) -> uint16_t {
                x = std::clamp(x, 0, static_cast<int>(p.width) - 1);
                y = std::clamp(y, 0, static_cast<int>(p.height) - 1);
                return src[static_cast<std::size_t>(y) * p.width + static_cast<std::size_t>(x)];
            };
            for (uint32_t y = 0; y < p.height; ++y)
            {
                for (uint32_t x = 0; x < p.width; ++x)
                {
                    const std::size_t idx = static_cast<std::size_t>(y) * p.width + x;
                    const uint16_t raw = src[idx];
                    if (raw == 0u)
                    {
                        out[idx] = 0u;
                        continue;
                    }
                    float d = static_cast<float>(raw) * p.depth_scale;
                    if (d < p.min_depth_m || (p.max_depth_m > 0.0f && d > p.max_depth_m))
                    {
                        out[idx] = 0u;
                        continue;
                    }
                    if (p.enable_bilateral != 0u)
                    {
                        const float inv_2ss = -0.5f / std::max(p.sigma_spatial * p.sigma_spatial, 1e-4f);
                        const float inv_2sr = -0.5f / std::max(p.sigma_range * p.sigma_range, 1e-6f);
                        float sum = 0.0f, wsum = 0.0f;
                        for (int dy = -2; dy <= 2; ++dy)
                        {
                            for (int dx = -2; dx <= 2; ++dx)
                            {
                                const uint16_t nraw = read_z(static_cast<int>(x) + dx, static_cast<int>(y) + dy);
                                if (nraw == 0u)
                                    continue;
                                const float nd = static_cast<float>(nraw) * p.depth_scale;
                                const float dr = nd - d;
                                const float w = std::exp(static_cast<float>(dx * dx + dy * dy) * inv_2ss + (dr * dr) * inv_2sr);
                                sum += nd * w;
                                wsum += w;
                            }
                        }
                        if (wsum > 0.0f)
                            d = sum / wsum;
                    }
                    if (p.enable_temporal != 0u && p.has_prev != 0u)
                    {
                        const uint16_t praw = prev[idx];
                        if (praw != 0u)
                        {
                            const float pd = static_cast<float>(praw) * p.depth_scale;
                            if (std::abs(d - pd) <= p.temporal_delta_m)
                                d = pd + (d - pd) * p.temporal_alpha;
                        }
                    }
                    const float clamped = std::clamp(std::round(d / p.depth_scale), 0.0f, 65535.0f);
                    out[idx] = static_cast<uint16_t>(clamped);
                }
            }
            if (p.enable_temporal != 0u)
                prev = out;
            return out;
        };

        auto run_gpu_filter = [&](const std::vector<uint16_t>& src, const DepthFilterPush& p) -> std::vector<uint16_t> {
            std::memcpy(in_buf.mapped(), src.data(), bytes);
            vkResetCommandBuffer(cmd, 0);
            VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
            vkBeginCommandBuffer(cmd, &bi);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline());
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.layout(), 0, 1, &ds, 0, nullptr);
            vkCmdPushConstants(cmd, pipe.layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p), &p);
            vkCmdDispatch(cmd, div_up(W / 2u, 16u), div_up(H, 16u), 1u);
            vkEndCommandBuffer(cmd);
            ctx.submit_and_wait(cmd, fence);
            ctx.invalidate_buffer(out_buf.get());
            std::vector<uint16_t> out(n_pix);
            std::memcpy(out.data(), out_buf.mapped(), bytes);
            return out;
        };

        // Synthetic frame 0: near clip (50mm), noisy plane (1000+-25mm), step edge (2000mm), far clip (4000mm)
        std::vector<uint16_t> f0(n_pix, 0u);
        for (uint32_t y = 0; y < H; ++y)
        {
            for (uint32_t x = 0; x < W; ++x)
            {
                const std::size_t idx = static_cast<std::size_t>(y) * W + x;
                if (x < 4)
                    f0[idx] = 50u;  // below min_depth_m
                else if (x >= 28)
                    f0[idx] = 4000u;  // above max_depth_m
                else if (x < 16)
                    f0[idx] = static_cast<uint16_t>(1000 + (((x + y) % 2 == 0) ? 20 : -20));
                else
                    f0[idx] = 2000u;
            }
        }
        // Synthetic frame 1: small delta (+20mm <= 50mm) on left plane, large jump (+200mm > 50mm) on right plane
        std::vector<uint16_t> f1 = f0;
        for (uint32_t y = 0; y < H; ++y)
        {
            for (uint32_t x = 4; x < 28; ++x)
            {
                const std::size_t idx = static_cast<std::size_t>(y) * W + x;
                f1[idx] = static_cast<uint16_t>(f0[idx] + ((x < 16) ? 20u : 200u));
            }
        }

        // 3a. Clipping only
        {
            std::vector<uint16_t> cpu_prev(n_pix, 0u);
            std::memset(prev_buf.mapped(), 0, bytes);
            DepthFilterPush p{W, H, 0.001f, 0.2f, 3.0f, 0u, 2.0f, 0.05f, 0u, 0.4f, 0.05f, 0u};
            auto gpu_out = run_gpu_filter(f0, p);
            auto cpu_out = cpu_depth_filter(f0, cpu_prev, p);
            for (std::size_t i = 0; i < n_pix; ++i)
                assert(gpu_out[i] == cpu_out[i]);
            assert(gpu_out[0] == 0u && gpu_out[31] == 0u);
        }

        // 3b. 5x5 Bilateral only + sigma_range sensitivity (anti-no-op)
        {
            std::vector<uint16_t> cpu_prev(n_pix, 0u);
            DepthFilterPush p_narrow{W, H, 0.001f, 0.2f, 3.0f, 1u, 2.0f, 0.05f, 0u, 0.4f, 0.05f, 0u};
            auto gpu_narrow = run_gpu_filter(f0, p_narrow);
            auto cpu_narrow = cpu_depth_filter(f0, cpu_prev, p_narrow);
            int changed_vs_raw = 0;
            for (std::size_t i = 0; i < n_pix; ++i)
            {
                assert(std::abs(static_cast<int>(gpu_narrow[i]) - static_cast<int>(cpu_narrow[i])) <= 1);
                if (gpu_narrow[i] != 0u && gpu_narrow[i] != f0[i])
                    ++changed_vs_raw;
            }
            assert(changed_vs_raw > 50);

            DepthFilterPush p_wide = p_narrow;
            p_wide.sigma_range = 1.5f;
            auto gpu_wide = run_gpu_filter(f0, p_wide);
            int changed_vs_narrow = 0;
            for (std::size_t i = 0; i < n_pix; ++i)
            {
                if (gpu_wide[i] != gpu_narrow[i])
                    ++changed_vs_narrow;
            }
            assert(changed_vs_narrow > 20);
        }

        // 3c. Temporal EMA 2-frame sequence (blending <= delta vs edge preservation > delta)
        {
            std::vector<uint16_t> cpu_prev(n_pix, 0u);
            std::memset(prev_buf.mapped(), 0, bytes);
            DepthFilterPush p0{W, H, 0.001f, 0.2f, 3.0f, 0u, 2.0f, 0.05f, 1u, 0.4f, 0.05f, 0u};
            auto g0 = run_gpu_filter(f0, p0);
            auto c0 = cpu_depth_filter(f0, cpu_prev, p0);
            for (std::size_t i = 0; i < n_pix; ++i)
                assert(g0[i] == c0[i]);

            DepthFilterPush p1 = p0;
            p1.has_prev = 1u;
            auto g1 = run_gpu_filter(f1, p1);
            auto c1 = cpu_depth_filter(f1, cpu_prev, p1);
            for (std::size_t i = 0; i < n_pix; ++i)
                assert(std::abs(static_cast<int>(g1[i]) - static_cast<int>(c1[i])) <= 1);
            // Left plane blended: f0+20 * 0.4 = f0+8; right plane (>50mm jump) unblended: 2200
            assert(std::abs(static_cast<int>(g1[8 * W + 8]) - static_cast<int>(f0[8 * W + 8] + 8)) <= 1);
            assert(g1[8 * W + 20] == 2200u);
        }

        // 3d. Bilateral + Temporal combined (2-frame sequence)
        {
            std::vector<uint16_t> cpu_prev(n_pix, 0u);
            std::memset(prev_buf.mapped(), 0, bytes);
            DepthFilterPush p0{W, H, 0.001f, 0.2f, 3.0f, 1u, 2.0f, 0.05f, 1u, 0.4f, 0.05f, 0u};
            auto g0 = run_gpu_filter(f0, p0);
            auto c0 = cpu_depth_filter(f0, cpu_prev, p0);
            for (std::size_t i = 0; i < n_pix; ++i)
                assert(std::abs(static_cast<int>(g0[i]) - static_cast<int>(c0[i])) <= 1);

            DepthFilterPush p1 = p0;
            p1.has_prev = 1u;
            auto g1 = run_gpu_filter(f1, p1);
            auto c1 = cpu_depth_filter(f1, cpu_prev, p1);
            for (std::size_t i = 0; i < n_pix; ++i)
                assert(std::abs(static_cast<int>(g1[i]) - static_cast<int>(c1[i])) <= 1);
        }
    }

    vkDestroyFence(ctx.device(), fence, nullptr);
    vkDestroyDescriptorPool(ctx.device(), pool, nullptr);

    std::cout << "ALL_VULKAN_KERNEL_TESTS_PASSED" << std::endl;
    return 0;
}
