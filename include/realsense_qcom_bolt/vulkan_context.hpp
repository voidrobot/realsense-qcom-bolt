#pragma once

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <utility>

namespace realsense_qcom_bolt
{

class VulkanContext;

struct VulkanBuffer
{
    VkBuffer buffer{VK_NULL_HANDLE};
    VkDeviceMemory memory{VK_NULL_HANDLE};
    VkDeviceSize size{0};
    VkDeviceSize alloc_size{0};
    void* mapped{nullptr};
    bool is_imported{false};
};

struct ComputePipelineHandle
{
    VkDescriptorSetLayout desc_layout{VK_NULL_HANDLE};
    VkPipelineLayout pipeline_layout{VK_NULL_HANDLE};
    VkPipeline pipeline{VK_NULL_HANDLE};
};

// Move-only RAII wrapper for VulkanBuffer.
class ScopedVulkanBuffer
{
public:
    ScopedVulkanBuffer() = default;
    ScopedVulkanBuffer(VulkanContext& ctx, VulkanBuffer buf) : _ctx(&ctx), _buf(buf) {}
    ~ScopedVulkanBuffer() { reset(); }

    ScopedVulkanBuffer(const ScopedVulkanBuffer&) = delete;
    ScopedVulkanBuffer& operator=(const ScopedVulkanBuffer&) = delete;

    ScopedVulkanBuffer(ScopedVulkanBuffer&& o) noexcept : _ctx(o._ctx), _buf(o._buf)
    {
        o._ctx = nullptr;
        o._buf = {};
    }

    ScopedVulkanBuffer& operator=(ScopedVulkanBuffer&& o) noexcept
    {
        if (this != &o)
        {
            reset();
            _ctx = o._ctx;
            _buf = o._buf;
            o._ctx = nullptr;
            o._buf = {};
        }
        return *this;
    }

    void reset();
    bool valid() const { return _buf.buffer != VK_NULL_HANDLE; }
    const VulkanBuffer& get() const { return _buf; }
    VkBuffer buffer() const { return _buf.buffer; }
    void* mapped() const { return _buf.mapped; }
    VkDeviceSize size() const { return _buf.size; }

private:
    VulkanContext* _ctx{nullptr};
    VulkanBuffer _buf{};
};

// Move-only RAII wrapper for ComputePipelineHandle.
class ScopedComputePipeline
{
public:
    ScopedComputePipeline() = default;
    ScopedComputePipeline(VulkanContext& ctx, ComputePipelineHandle h) : _ctx(&ctx), _h(h) {}
    ~ScopedComputePipeline() { reset(); }

    ScopedComputePipeline(const ScopedComputePipeline&) = delete;
    ScopedComputePipeline& operator=(const ScopedComputePipeline&) = delete;

    ScopedComputePipeline(ScopedComputePipeline&& o) noexcept : _ctx(o._ctx), _h(o._h)
    {
        o._ctx = nullptr;
        o._h = {};
    }

    ScopedComputePipeline& operator=(ScopedComputePipeline&& o) noexcept
    {
        if (this != &o)
        {
            reset();
            _ctx = o._ctx;
            _h = o._h;
            o._ctx = nullptr;
            o._h = {};
        }
        return *this;
    }

    void reset();
    bool valid() const { return _h.pipeline != VK_NULL_HANDLE; }
    VkPipeline pipeline() const { return _h.pipeline; }
    VkPipelineLayout layout() const { return _h.pipeline_layout; }
    VkDescriptorSetLayout desc_layout() const { return _h.desc_layout; }

private:
    VulkanContext* _ctx{nullptr};
    ComputePipelineHandle _h{};
};

// Shared Vulkan 1.3 Compute context.
// Adreno 623 constraints enforced by design:
//   - No VK_KHR_push_descriptor (static VkDescriptorPool / VkDescriptorSet only)
//   - DMA-BUF import via VK_EXT_external_memory_dma_buf using dup(fd) to preserve DmaBuffer ownership
class VulkanContext
{
public:
    static constexpr std::size_t kMaxStageBindings = 8u;

    VulkanContext();
    ~VulkanContext();

    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;

    VkDevice device() const { return _device; }
    VkPhysicalDevice physical_device() const { return _phys_dev; }
    VkQueue compute_queue() const { return _compute_queue; }
    uint32_t compute_queue_family() const { return _compute_queue_family; }
    VkCommandPool command_pool() const { return _cmd_pool; }
    VkDeviceSize non_coherent_atom_size() const { return _non_coherent_atom_size; }
    VkDeviceSize buffer_align_bytes() const { return (_non_coherent_atom_size > 64u) ? _non_coherent_atom_size : 64u; }
    bool supports_dmabuf_import() const { return _supports_dmabuf_import; }
    const std::string& device_name() const { return _device_name; }

    // Create owned device/host buffer (size automatically rounded up to buffer_align_bytes()).
    ScopedVulkanBuffer create_buffer(VkDeviceSize size,
                                     VkBufferUsageFlags usage,
                                     VkMemoryPropertyFlags mem_props);

    // Import an external Linux DMA-BUF fd into a Vulkan VkBuffer (duplicates fd internally).
    ScopedVulkanBuffer import_dmabuf_buffer(int dmabuf_fd,
                                            VkDeviceSize size,
                                            VkBufferUsageFlags usage);

    void destroy_buffer(VulkanBuffer& buf);

    // Flush/invalidate non-coherent host-cached memory ranges (automatically aligned to nonCoherentAtomSize).
    void invalidate_buffer(const VulkanBuffer& buf, VkDeviceSize offset = 0, VkDeviceSize size = VK_WHOLE_SIZE);
    void flush_buffer(const VulkanBuffer& buf, VkDeviceSize offset = 0, VkDeviceSize size = VK_WHOLE_SIZE);

    ScopedComputePipeline create_compute_pipeline(
        const uint32_t* spv_words,
        std::size_t spv_word_count,
        uint32_t storage_buffer_count,
        uint32_t push_constant_bytes,
        const VkSpecializationInfo* spec_info = nullptr);

    void destroy_compute_pipeline(ComputePipelineHandle& handle);

    VkDescriptorPool create_descriptor_pool(uint32_t max_sets, uint32_t total_storage_buffers);
    VkDescriptorSet allocate_descriptor_set(VkDescriptorPool pool, VkDescriptorSetLayout layout);
    void update_storage_buffers(VkDescriptorSet set, std::initializer_list<VkBuffer> buffers);

    VkCommandBuffer allocate_command_buffer();
    VkFence create_fence(bool signaled = false);
    void submit_and_wait(VkCommandBuffer cmd, VkFence fence);

private:
    void cleanup();
    uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags props) const;
    VkMappedMemoryRange make_aligned_range(const VulkanBuffer& buf, VkDeviceSize offset, VkDeviceSize size) const;

    VkInstance _instance{VK_NULL_HANDLE};
    VkPhysicalDevice _phys_dev{VK_NULL_HANDLE};
    VkDevice _device{VK_NULL_HANDLE};
    VkQueue _compute_queue{VK_NULL_HANDLE};
    uint32_t _compute_queue_family{UINT32_MAX};
    VkCommandPool _cmd_pool{VK_NULL_HANDLE};
    VkDeviceSize _non_coherent_atom_size{64};
    VkPhysicalDeviceMemoryProperties _mem_props{};
    PFN_vkGetMemoryFdPropertiesKHR _vkGetMemoryFdPropertiesKHR{nullptr};
    bool _supports_dmabuf_import{false};
    std::string _device_name;
};

}  // namespace realsense_qcom_bolt
