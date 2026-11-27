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

    vkDestroyFence(ctx.device(), fence, nullptr);
    vkDestroyDescriptorPool(ctx.device(), pool, nullptr);

    std::cout << "ALL_VULKAN_KERNEL_TESTS_PASSED" << std::endl;
    return 0;
}
