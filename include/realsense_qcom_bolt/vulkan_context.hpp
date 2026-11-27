#pragma once

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace realsense_qcom_bolt
{

struct VulkanBuffer
{
    VkBuffer buffer{VK_NULL_HANDLE};
    VkDeviceMemory memory{VK_NULL_HANDLE};
    VkDeviceSize size{0};
    void* mapped{nullptr};
    bool is_imported{false};
};

struct ComputePipelineHandle
{
    VkDescriptorSetLayout desc_layout{VK_NULL_HANDLE};
    VkPipelineLayout pipeline_layout{VK_NULL_HANDLE};
    VkPipeline pipeline{VK_NULL_HANDLE};
};

// Shared Vulkan 1.3 Compute context.
// Adreno 623 constraints enforced by design:
//   - No VK_KHR_push_descriptor (static VkDescriptorPool / VkDescriptorSet only)
//   - DMA-BUF import via VK_EXT_external_memory_dma_buf using dup(fd) to preserve DmaBuffer ownership
class VulkanContext
{
public:
    VulkanContext();
    ~VulkanContext();

    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;

    VkDevice device() const { return _device; }
    VkPhysicalDevice physical_device() const { return _phys_dev; }
    VkQueue compute_queue() const { return _compute_queue; }
    uint32_t compute_queue_family() const { return _compute_queue_family; }
    VkCommandPool command_pool() const { return _cmd_pool; }
    bool supports_dmabuf_import() const { return _supports_dmabuf_import; }
    const std::string& device_name() const { return _device_name; }

    // Create owned device/host buffer. Maps memory automatically if HOST_VISIBLE is set.
    VulkanBuffer create_buffer(VkDeviceSize size,
                               VkBufferUsageFlags usage,
                               VkMemoryPropertyFlags mem_props);

    // Import an external Linux DMA-BUF fd into a Vulkan VkBuffer (duplicates fd internally).
    VulkanBuffer import_dmabuf_buffer(int dmabuf_fd,
                                      VkDeviceSize size,
                                      VkBufferUsageFlags usage);

    void destroy_buffer(VulkanBuffer& buf);

    // Flush/invalidate non-coherent host-cached memory ranges.
    void invalidate_buffer(const VulkanBuffer& buf, VkDeviceSize offset = 0, VkDeviceSize size = VK_WHOLE_SIZE);
    void flush_buffer(const VulkanBuffer& buf, VkDeviceSize offset = 0, VkDeviceSize size = VK_WHOLE_SIZE);

    ComputePipelineHandle create_compute_pipeline(
        const uint32_t* spv_words,
        std::size_t spv_word_count,
        uint32_t storage_buffer_count,
        uint32_t push_constant_bytes,
        const VkSpecializationInfo* spec_info = nullptr);

    void destroy_compute_pipeline(ComputePipelineHandle& handle);

    VkDescriptorPool create_descriptor_pool(uint32_t max_sets, uint32_t total_storage_buffers);
    VkDescriptorSet allocate_descriptor_set(VkDescriptorPool pool, VkDescriptorSetLayout layout);
    void update_storage_buffers(VkDescriptorSet set, const std::vector<VkBuffer>& buffers);

    VkCommandBuffer allocate_command_buffer();
    VkFence create_fence(bool signaled = false);
    void submit_and_wait(VkCommandBuffer cmd, VkFence fence);

private:
    uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags props) const;

    VkInstance _instance{VK_NULL_HANDLE};
    VkPhysicalDevice _phys_dev{VK_NULL_HANDLE};
    VkDevice _device{VK_NULL_HANDLE};
    VkQueue _compute_queue{VK_NULL_HANDLE};
    uint32_t _compute_queue_family{UINT32_MAX};
    VkCommandPool _cmd_pool{VK_NULL_HANDLE};
    VkPhysicalDeviceMemoryProperties _mem_props{};
    PFN_vkGetMemoryFdPropertiesKHR _vkGetMemoryFdPropertiesKHR{nullptr};
    bool _supports_dmabuf_import{false};
    std::string _device_name;
};

}  // namespace realsense_qcom_bolt
