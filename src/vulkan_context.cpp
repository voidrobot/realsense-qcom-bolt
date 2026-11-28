// Copyright 2026 voidrobot
// SPDX-License-Identifier: Apache-2.0

#include "realsense_qcom_bolt/vulkan_context.hpp"
#include "realsense_qcom_bolt/math_utils.hpp"

#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace realsense_qcom_bolt
{

namespace
{
void vk_check(VkResult res, const char* what)
{
    if (res != VK_SUCCESS)
    {
        throw std::runtime_error(std::string("Vulkan call failed: ") + what +
                                 " (VkResult=" + std::to_string(static_cast<int>(res)) + ")");
    }
}

bool has_extension(const std::vector<VkExtensionProperties>& exts, const char* name)
{
    return std::any_of(exts.begin(), exts.end(), [name](const VkExtensionProperties& e) {
        return std::strcmp(e.extensionName, name) == 0;
    });
}
}  // namespace

void ScopedVulkanBuffer::reset()
{
    if (_ctx)
    {
        _ctx->destroy_buffer(_buf);
        _ctx = nullptr;
    }
    _buf = {};
}

void ScopedComputePipeline::reset()
{
    if (_ctx)
    {
        _ctx->destroy_compute_pipeline(_h);
        _ctx = nullptr;
    }
    _h = {};
}

VulkanContext::VulkanContext()
{
    try
    {
        VkApplicationInfo app_info{};
        app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app_info.pApplicationName = "realsense_qcom_bolt";
        app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        app_info.pEngineName = "realsense_qcom_bolt";
        app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        app_info.apiVersion = VK_API_VERSION_1_3;

        VkInstanceCreateInfo inst_ci{};
        inst_ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        inst_ci.pApplicationInfo = &app_info;
        vk_check(vkCreateInstance(&inst_ci, nullptr, &_instance), "vkCreateInstance");

        uint32_t dev_count = 0;
        vk_check(vkEnumeratePhysicalDevices(_instance, &dev_count, nullptr), "vkEnumeratePhysicalDevices(count)");
        if (dev_count == 0)
        {
            throw std::runtime_error("No Vulkan physical devices found.");
        }
        std::vector<VkPhysicalDevice> devices(dev_count);
        vk_check(vkEnumeratePhysicalDevices(_instance, &dev_count, devices.data()), "vkEnumeratePhysicalDevices");

        for (VkPhysicalDevice pd : devices)
        {
            uint32_t qf_count = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(pd, &qf_count, nullptr);
            std::vector<VkQueueFamilyProperties> qfs(qf_count);
            vkGetPhysicalDeviceQueueFamilyProperties(pd, &qf_count, qfs.data());
            for (uint32_t i = 0; i < qf_count; ++i)
            {
                if (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
                {
                    _phys_dev = pd;
                    _compute_queue_family = i;
                    break;
                }
            }
            if (_phys_dev != VK_NULL_HANDLE)
                break;
        }
        if (_phys_dev == VK_NULL_HANDLE)
        {
            throw std::runtime_error("No Vulkan physical device with a compute queue family found.");
        }

        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(_phys_dev, &props);
        _device_name = props.deviceName;
        _non_coherent_atom_size = std::max<VkDeviceSize>(1u, props.limits.nonCoherentAtomSize);
        vkGetPhysicalDeviceMemoryProperties(_phys_dev, &_mem_props);

        uint32_t ext_count = 0;
        vk_check(vkEnumerateDeviceExtensionProperties(_phys_dev, nullptr, &ext_count, nullptr),
                 "vkEnumerateDeviceExtensionProperties(count)");
        std::vector<VkExtensionProperties> avail_exts(ext_count);
        vk_check(vkEnumerateDeviceExtensionProperties(_phys_dev, nullptr, &ext_count, avail_exts.data()),
                 "vkEnumerateDeviceExtensionProperties");

        std::vector<const char*> enabled_exts;
        if (has_extension(avail_exts, VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME) &&
            has_extension(avail_exts, VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME) &&
            has_extension(avail_exts, VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME))
        {
            enabled_exts.push_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
            enabled_exts.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
            enabled_exts.push_back(VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME);
            _supports_dmabuf_import = true;
        }

        const float queue_prio = 1.0f;
        VkDeviceQueueCreateInfo q_ci{};
        q_ci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        q_ci.queueFamilyIndex = _compute_queue_family;
        q_ci.queueCount = 1;
        q_ci.pQueuePriorities = &queue_prio;

        VkDeviceCreateInfo dev_ci{};
        dev_ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        dev_ci.queueCreateInfoCount = 1;
        dev_ci.pQueueCreateInfos = &q_ci;
        dev_ci.enabledExtensionCount = static_cast<uint32_t>(enabled_exts.size());
        dev_ci.ppEnabledExtensionNames = enabled_exts.data();
        vk_check(vkCreateDevice(_phys_dev, &dev_ci, nullptr, &_device), "vkCreateDevice");

        vkGetDeviceQueue(_device, _compute_queue_family, 0, &_compute_queue);

        if (_supports_dmabuf_import)
        {
            _vkGetMemoryFdPropertiesKHR = reinterpret_cast<PFN_vkGetMemoryFdPropertiesKHR>(
                vkGetDeviceProcAddr(_device, "vkGetMemoryFdPropertiesKHR"));
            if (!_vkGetMemoryFdPropertiesKHR)
            {
                _supports_dmabuf_import = false;
            }
        }

        VkCommandPoolCreateInfo pool_ci{};
        pool_ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool_ci.queueFamilyIndex = _compute_queue_family;
        vk_check(vkCreateCommandPool(_device, &pool_ci, nullptr, &_cmd_pool), "vkCreateCommandPool");
    }
    catch (...)
    {
        cleanup();
        throw;
    }
}

VulkanContext::~VulkanContext()
{
    cleanup();
}

void VulkanContext::cleanup()
{
    if (_device != VK_NULL_HANDLE)
    {
        vkDeviceWaitIdle(_device);
        if (_cmd_pool != VK_NULL_HANDLE)
        {
            vkDestroyCommandPool(_device, _cmd_pool, nullptr);
            _cmd_pool = VK_NULL_HANDLE;
        }
        vkDestroyDevice(_device, nullptr);
        _device = VK_NULL_HANDLE;
    }
    if (_instance != VK_NULL_HANDLE)
    {
        vkDestroyInstance(_instance, nullptr);
        _instance = VK_NULL_HANDLE;
    }
}

uint32_t VulkanContext::find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags props) const
{
    for (uint32_t i = 0; i < _mem_props.memoryTypeCount; ++i)
    {
        if ((type_filter & (1u << i)) &&
            (_mem_props.memoryTypes[i].propertyFlags & props) == props)
        {
            return i;
        }
    }
    throw std::runtime_error("Failed to find suitable Vulkan memory type.");
}

ScopedVulkanBuffer VulkanContext::create_buffer(VkDeviceSize size,
                                                VkBufferUsageFlags usage,
                                                VkMemoryPropertyFlags mem_props)
{
    const VkDeviceSize aligned_size = align_up(static_cast<std::size_t>(size),
                                               static_cast<std::size_t>(buffer_align_bytes()));
    VulkanBuffer out;
    out.size = aligned_size;

    try
    {
        VkBufferCreateInfo buf_ci{};
        buf_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buf_ci.size = aligned_size;
        buf_ci.usage = usage;
        buf_ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        vk_check(vkCreateBuffer(_device, &buf_ci, nullptr, &out.buffer), "vkCreateBuffer");

        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(_device, out.buffer, &req);
        out.alloc_size = req.size;

        VkMemoryAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc_info.allocationSize = req.size;
        alloc_info.memoryTypeIndex = find_memory_type(req.memoryTypeBits, mem_props);
        vk_check(vkAllocateMemory(_device, &alloc_info, nullptr, &out.memory), "vkAllocateMemory");
        vk_check(vkBindBufferMemory(_device, out.buffer, out.memory, 0), "vkBindBufferMemory");

        if (mem_props & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
        {
            vk_check(vkMapMemory(_device, out.memory, 0, VK_WHOLE_SIZE, 0, &out.mapped), "vkMapMemory");
        }
    }
    catch (...)
    {
        destroy_buffer(out);
        throw;
    }
    return ScopedVulkanBuffer(*this, out);
}

ScopedVulkanBuffer VulkanContext::import_dmabuf_buffer(int dmabuf_fd,
                                                       VkDeviceSize size,
                                                       VkBufferUsageFlags usage)
{
    if (!_supports_dmabuf_import || dmabuf_fd < 0)
    {
        throw std::runtime_error("DMA-BUF Vulkan import unavailable or invalid fd.");
    }

    VulkanBuffer out;
    out.size = size;
    out.is_imported = true;
    int dup_fd = -1;

    try
    {
        VkExternalMemoryBufferCreateInfo ext_buf_ci{};
        ext_buf_ci.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
        ext_buf_ci.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;

        VkBufferCreateInfo buf_ci{};
        buf_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buf_ci.pNext = &ext_buf_ci;
        buf_ci.size = size;
        buf_ci.usage = usage;
        buf_ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        vk_check(vkCreateBuffer(_device, &buf_ci, nullptr, &out.buffer), "vkCreateBuffer(dmabuf)");

        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(_device, out.buffer, &req);
        out.alloc_size = std::max(req.size, size);

        VkMemoryFdPropertiesKHR fd_props{};
        fd_props.sType = VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR;
        vk_check(_vkGetMemoryFdPropertiesKHR(
                     _device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, dmabuf_fd, &fd_props),
                 "vkGetMemoryFdPropertiesKHR");

        const uint32_t valid_bits = req.memoryTypeBits & fd_props.memoryTypeBits;
        if (valid_bits == 0)
        {
            throw std::runtime_error("No compatible Vulkan memory type for imported DMA-BUF fd.");
        }

        uint32_t mem_type_idx = 0;
        while (((valid_bits >> mem_type_idx) & 1u) == 0u)
        {
            ++mem_type_idx;
        }

        // Vulkan takes ownership of dup_fd on vkAllocateMemory success.
        dup_fd = ::dup(dmabuf_fd);
        if (dup_fd < 0)
        {
            throw std::runtime_error("dup(dmabuf_fd) failed during Vulkan import.");
        }

        VkImportMemoryFdInfoKHR import_info{};
        import_info.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR;
        import_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
        import_info.fd = dup_fd;

        VkMemoryDedicatedAllocateInfo ded_info{};
        ded_info.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
        ded_info.pNext = &import_info;
        ded_info.buffer = out.buffer;

        VkMemoryAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc_info.pNext = &ded_info;
        alloc_info.allocationSize = out.alloc_size;
        alloc_info.memoryTypeIndex = mem_type_idx;

        vk_check(vkAllocateMemory(_device, &alloc_info, nullptr, &out.memory), "vkAllocateMemory(dmabuf)");
        dup_fd = -1;  // Ownership transferred to Vulkan
        vk_check(vkBindBufferMemory(_device, out.buffer, out.memory, 0), "vkBindBufferMemory(dmabuf)");
    }
    catch (...)
    {
        if (dup_fd >= 0)
            ::close(dup_fd);
        destroy_buffer(out);
        throw;
    }
    return ScopedVulkanBuffer(*this, out);
}

void VulkanContext::destroy_buffer(VulkanBuffer& buf)
{
    if (_device == VK_NULL_HANDLE)
        return;
    if (buf.mapped && buf.memory != VK_NULL_HANDLE)
    {
        vkUnmapMemory(_device, buf.memory);
        buf.mapped = nullptr;
    }
    if (buf.buffer != VK_NULL_HANDLE)
    {
        vkDestroyBuffer(_device, buf.buffer, nullptr);
        buf.buffer = VK_NULL_HANDLE;
    }
    if (buf.memory != VK_NULL_HANDLE)
    {
        vkFreeMemory(_device, buf.memory, nullptr);
        buf.memory = VK_NULL_HANDLE;
    }
    buf.size = 0;
    buf.alloc_size = 0;
}

VkMappedMemoryRange VulkanContext::make_aligned_range(const VulkanBuffer& buf,
                                                      VkDeviceSize offset,
                                                      VkDeviceSize size) const
{
    VkMappedMemoryRange range{};
    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.memory = buf.memory;

    const VkDeviceSize atom = _non_coherent_atom_size;
    const VkDeviceSize aligned_off = (offset / atom) * atom;
    range.offset = aligned_off;

    if (size == VK_WHOLE_SIZE)
    {
        range.size = VK_WHOLE_SIZE;
    }
    else
    {
        const VkDeviceSize end = offset + size;
        const VkDeviceSize aligned_end =
            align_up(static_cast<std::size_t>(end), static_cast<std::size_t>(atom));
        const VkDeviceSize limit = (buf.alloc_size > 0) ? buf.alloc_size : buf.size;
        range.size = (aligned_end >= limit) ? VK_WHOLE_SIZE : (aligned_end - aligned_off);
    }
    return range;
}

void VulkanContext::invalidate_buffer(const VulkanBuffer& buf, VkDeviceSize offset, VkDeviceSize size)
{
    VkMappedMemoryRange range = make_aligned_range(buf, offset, size);
    vk_check(vkInvalidateMappedMemoryRanges(_device, 1, &range), "vkInvalidateMappedMemoryRanges");
}

void VulkanContext::flush_buffer(const VulkanBuffer& buf, VkDeviceSize offset, VkDeviceSize size)
{
    VkMappedMemoryRange range = make_aligned_range(buf, offset, size);
    vk_check(vkFlushMappedMemoryRanges(_device, 1, &range), "vkFlushMappedMemoryRanges");
}

ScopedComputePipeline VulkanContext::create_compute_pipeline(
    const uint32_t* spv_words,
    std::size_t spv_word_count,
    uint32_t storage_buffer_count,
    uint32_t push_constant_bytes,
    const VkSpecializationInfo* spec_info)
{
    if (storage_buffer_count > kMaxStageBindings)
    {
        throw std::invalid_argument("storage_buffer_count exceeds kMaxStageBindings");
    }

    ComputePipelineHandle h;
    VkShaderModule shader_mod = VK_NULL_HANDLE;
    try
    {
        std::array<VkDescriptorSetLayoutBinding, kMaxStageBindings> bindings{};
        for (uint32_t i = 0; i < storage_buffer_count; ++i)
        {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }

        VkDescriptorSetLayoutCreateInfo dsl_ci{};
        dsl_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dsl_ci.bindingCount = storage_buffer_count;
        dsl_ci.pBindings = bindings.data();
        vk_check(vkCreateDescriptorSetLayout(_device, &dsl_ci, nullptr, &h.desc_layout),
                 "vkCreateDescriptorSetLayout");

        VkPushConstantRange pc_range{};
        pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pc_range.offset = 0;
        pc_range.size = push_constant_bytes;

        VkPipelineLayoutCreateInfo pl_ci{};
        pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pl_ci.setLayoutCount = 1;
        pl_ci.pSetLayouts = &h.desc_layout;
        pl_ci.pushConstantRangeCount = (push_constant_bytes > 0) ? 1 : 0;
        pl_ci.pPushConstantRanges = (push_constant_bytes > 0) ? &pc_range : nullptr;
        vk_check(vkCreatePipelineLayout(_device, &pl_ci, nullptr, &h.pipeline_layout),
                 "vkCreatePipelineLayout");

        VkShaderModuleCreateInfo sm_ci{};
        sm_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        sm_ci.codeSize = spv_word_count * sizeof(uint32_t);
        sm_ci.pCode = spv_words;
        vk_check(vkCreateShaderModule(_device, &sm_ci, nullptr, &shader_mod), "vkCreateShaderModule");

        VkComputePipelineCreateInfo cp_ci{};
        cp_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cp_ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cp_ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cp_ci.stage.module = shader_mod;
        cp_ci.stage.pName = "main";
        cp_ci.stage.pSpecializationInfo = spec_info;
        cp_ci.layout = h.pipeline_layout;

        vk_check(vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1, &cp_ci, nullptr, &h.pipeline),
                 "vkCreateComputePipelines");
        vkDestroyShaderModule(_device, shader_mod, nullptr);
    }
    catch (...)
    {
        if (shader_mod != VK_NULL_HANDLE)
            vkDestroyShaderModule(_device, shader_mod, nullptr);
        destroy_compute_pipeline(h);
        throw;
    }
    return ScopedComputePipeline(*this, h);
}

void VulkanContext::destroy_compute_pipeline(ComputePipelineHandle& h)
{
    if (_device == VK_NULL_HANDLE)
        return;
    if (h.pipeline != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(_device, h.pipeline, nullptr);
        h.pipeline = VK_NULL_HANDLE;
    }
    if (h.pipeline_layout != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(_device, h.pipeline_layout, nullptr);
        h.pipeline_layout = VK_NULL_HANDLE;
    }
    if (h.desc_layout != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(_device, h.desc_layout, nullptr);
        h.desc_layout = VK_NULL_HANDLE;
    }
}

VkDescriptorPool VulkanContext::create_descriptor_pool(uint32_t max_sets, uint32_t total_storage_buffers)
{
    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = total_storage_buffers;

    VkDescriptorPoolCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    ci.maxSets = max_sets;
    ci.poolSizeCount = 1;
    ci.pPoolSizes = &pool_size;

    VkDescriptorPool pool = VK_NULL_HANDLE;
    vk_check(vkCreateDescriptorPool(_device, &ci, nullptr, &pool), "vkCreateDescriptorPool");
    return pool;
}

VkDescriptorSet VulkanContext::allocate_descriptor_set(VkDescriptorPool pool, VkDescriptorSetLayout layout)
{
    VkDescriptorSetAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = pool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &layout;

    VkDescriptorSet set = VK_NULL_HANDLE;
    vk_check(vkAllocateDescriptorSets(_device, &ai, &set), "vkAllocateDescriptorSets");
    return set;
}

void VulkanContext::update_storage_buffers(VkDescriptorSet set, std::initializer_list<VkBuffer> buffers)
{
    const std::size_t n = buffers.size();
    if (n == 0 || n > kMaxStageBindings)
    {
        throw std::invalid_argument("update_storage_buffers count out of [1, kMaxStageBindings]");
    }
    std::array<VkDescriptorBufferInfo, kMaxStageBindings> infos{};
    std::array<VkWriteDescriptorSet, kMaxStageBindings> writes{};
    uint32_t i = 0;
    for (VkBuffer b : buffers)
    {
        infos[i].buffer = b;
        infos[i].offset = 0;
        infos[i].range = VK_WHOLE_SIZE;

        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
        ++i;
    }
    vkUpdateDescriptorSets(_device, i, writes.data(), 0, nullptr);
}

VkCommandBuffer VulkanContext::allocate_command_buffer()
{
    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = _cmd_pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vk_check(vkAllocateCommandBuffers(_device, &ai, &cmd), "vkAllocateCommandBuffers");
    return cmd;
}

VkFence VulkanContext::create_fence(bool signaled)
{
    VkFenceCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    ci.flags = signaled ? VK_FENCE_CREATE_SIGNALED_BIT : 0;
    VkFence fence = VK_NULL_HANDLE;
    vk_check(vkCreateFence(_device, &ci, nullptr, &fence), "vkCreateFence");
    return fence;
}

void VulkanContext::submit_and_wait(VkCommandBuffer cmd, VkFence fence)
{
    vk_check(vkResetFences(_device, 1, &fence), "vkResetFences");
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vk_check(vkQueueSubmit(_compute_queue, 1, &si, fence), "vkQueueSubmit");
    vk_check(vkWaitForFences(_device, 1, &fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
}

}  // namespace realsense_qcom_bolt
