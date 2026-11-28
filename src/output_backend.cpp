// Copyright 2026 voidrobot
// SPDX-License-Identifier: Apache-2.0

#include "realsense_qcom_bolt/output_backend.hpp"

#include <sensor_msgs/image_encodings.hpp>

#include <cstring>

namespace realsense_qcom_bolt
{

PointCloudFieldsMeta make_pointcloud_fields_meta(bool has_rgb)
{
    PointCloudFieldsMeta meta;
    meta.has_rgb = has_rgb;
    meta.point_step = has_rgb ? 16u : 12u;

    auto make_field = [](const char* name, uint32_t offset) {
        sensor_msgs::msg::PointField f;
        f.name = name;
        f.offset = offset;
        f.datatype = sensor_msgs::msg::PointField::FLOAT32;
        f.count = 1;
        return f;
    };
    meta.fields.push_back(make_field("x", 0));
    meta.fields.push_back(make_field("y", 4));
    meta.fields.push_back(make_field("z", 8));
    if (has_rgb)
    {
        meta.fields.push_back(make_field("rgb", 12));
    }
    return meta;
}

OutputBackend::OutputBackend(rclcpp::Node& node,
                             bool use_qirp_transport,
                             std::size_t qirp_pool_size,
                             const rclcpp::QoS& qos,
                             VulkanContext* vk_ctx)
    : _node(node),
      _use_qirp(use_qirp_transport),
      _qos(qos),
      _vk_ctx(vk_ctx)
#ifdef BUILD_WITH_QIRP
      ,
      _color_pool(qirp_pool_size, defaults::kDefaultDmaHeapPath,
                  [this](const lib_mem_dmabuf::DmaBuffer& b) { evict_dmabuf_fd(b.fd()); }),
      _pc_pool(qirp_pool_size, defaults::kDefaultDmaHeapPath,
               [this](const lib_mem_dmabuf::DmaBuffer& b) { evict_dmabuf_fd(b.fd()); })
#endif
{
#ifndef BUILD_WITH_QIRP
    (void)qirp_pool_size;
    _use_qirp = false;
#endif
}

OutputBackend::~OutputBackend()
{
#ifdef BUILD_WITH_QIRP
    if (_vk_ctx && _vk_ctx->device() != VK_NULL_HANDLE)
    {
        vkDeviceWaitIdle(_vk_ctx->device());
    }
    std::lock_guard<std::mutex> lock(_dmabuf_cache_mtx);
    _dmabuf_vk_cache.clear();
#endif
}

void OutputBackend::create_color_publisher(const std::string& topic)
{
#ifdef BUILD_WITH_QIRP
    if (_use_qirp)
    {
        _qirp_color_pub = _node.create_publisher<qrb_ros::transport::type::Image>(topic, _qos);
        return;
    }
#endif
    _std_color_pub = _node.create_publisher<sensor_msgs::msg::Image>(topic, _qos);
}

void OutputBackend::create_depth_publisher(const std::string& topic)
{
    _depth_pub = _node.create_publisher<sensor_msgs::msg::Image>(topic, _qos);
}

void OutputBackend::create_aligned_depth_publisher(const std::string& topic)
{
    _aligned_depth_pub = _node.create_publisher<sensor_msgs::msg::Image>(topic, _qos);
}

void OutputBackend::create_pointcloud_publisher(const std::string& topic)
{
#ifdef BUILD_WITH_QIRP
    if (_use_qirp)
    {
        _qirp_pc_pub = _node.create_publisher<qrb_ros::transport::type::PointCloud2>(topic, _qos);
        return;
    }
#endif
    _std_pc_pub = _node.create_publisher<sensor_msgs::msg::PointCloud2>(topic, _qos);
}

void OutputBackend::create_infra_publisher(int index, const std::string& topic)
{
    if (index == 1)
        _infra1_pub = _node.create_publisher<sensor_msgs::msg::Image>(topic, _qos);
    else if (index == 2)
        _infra2_pub = _node.create_publisher<sensor_msgs::msg::Image>(topic, _qos);
}

bool OutputBackend::has_color_subscribers() const
{
#ifdef BUILD_WITH_QIRP
    if (_qirp_color_pub)
        return _qirp_color_pub->get_subscription_count() > 0;
#endif
    return _std_color_pub && _std_color_pub->get_subscription_count() > 0;
}

bool OutputBackend::has_depth_subscribers() const
{
    return _depth_pub && _depth_pub->get_subscription_count() > 0;
}

bool OutputBackend::has_aligned_depth_subscribers() const
{
    return _aligned_depth_pub && _aligned_depth_pub->get_subscription_count() > 0;
}

bool OutputBackend::has_pointcloud_subscribers() const
{
#ifdef BUILD_WITH_QIRP
    if (_qirp_pc_pub)
        return _qirp_pc_pub->get_subscription_count() > 0;
#endif
    return _std_pc_pub && _std_pc_pub->get_subscription_count() > 0;
}

bool OutputBackend::has_infra_subscribers(int index) const
{
    if (index == 1)
        return _infra1_pub && _infra1_pub->get_subscription_count() > 0;
    if (index == 2)
        return _infra2_pub && _infra2_pub->get_subscription_count() > 0;
    return false;
}

int OutputBackend::color_output_stride(int width) const
{
#ifdef BUILD_WITH_QIRP
    if (_use_qirp)
    {
        return qrb_ros::transport::image_utils::get_image_stride(
            width, sensor_msgs::image_encodings::RGB8);
    }
#endif
    return width * 3;
}

std::size_t OutputBackend::color_output_bytes(int width, int height) const
{
#ifdef BUILD_WITH_QIRP
    if (_use_qirp)
    {
        const int sz = qrb_ros::transport::image_utils::get_image_align_size(
            width, height, sensor_msgs::image_encodings::RGB8);
        return (sz > 0) ? static_cast<std::size_t>(sz) : 0;
    }
#endif
    return static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3u;
}

void OutputBackend::publish_color_from_cpu(const std_msgs::msg::Header& header,
                                           int width,
                                           int height,
                                           int src_step,
                                           const void* rgb8_data)
{
    if (!rgb8_data || width <= 0 || height <= 0)
        return;
    const std::size_t row_bytes = static_cast<std::size_t>(width) * 3u;
    if (static_cast<std::size_t>(src_step) < row_bytes)
        return;

#ifdef BUILD_WITH_QIRP
    if (_qirp_color_pub)
    {
        const int dst_stride = color_output_stride(width);
        const std::size_t align_bytes = color_output_bytes(width, height);
        if (dst_stride < static_cast<int>(row_bytes) ||
            static_cast<std::size_t>(height) * static_cast<std::size_t>(dst_stride) > align_bytes)
        {
            return;
        }
        auto dmabuf = _color_pool.acquire(align_bytes);
        if (!dmabuf || !dmabuf->addr())
            return;

        dmabuf->sync_start();
        copy_image_2d(dmabuf->addr(), static_cast<std::size_t>(dst_stride),
                      rgb8_data, static_cast<std::size_t>(src_step), row_bytes, height);
        dmabuf->sync_end();

        auto msg = std::make_unique<qrb_ros::transport::type::Image>();
        msg->header = header;
        msg->width = width;
        msg->height = height;
        msg->encoding = sensor_msgs::image_encodings::RGB8;
        msg->dmabuf = std::move(dmabuf);
        _qirp_color_pub->publish(std::move(msg));
        return;
    }
#endif

    if (!_std_color_pub)
        return;
    auto msg = std::make_unique<sensor_msgs::msg::Image>();
    msg->header = header;
    msg->width = static_cast<uint32_t>(width);
    msg->height = static_cast<uint32_t>(height);
    msg->encoding = sensor_msgs::image_encodings::RGB8;
    msg->is_bigendian = 0;
    msg->step = static_cast<uint32_t>(row_bytes);
    msg->data.resize(static_cast<std::size_t>(height) * row_bytes);
    copy_image_2d(msg->data.data(), row_bytes, rgb8_data, static_cast<std::size_t>(src_step), row_bytes, height);
    _std_color_pub->publish(std::move(msg));
}

void OutputBackend::publish_depth_from_cpu(const std_msgs::msg::Header& header,
                                           int width,
                                           int height,
                                           const void* z16_data,
                                           bool is_aligned)
{
    auto& pub = is_aligned ? _aligned_depth_pub : _depth_pub;
    if (!pub || !z16_data || width <= 0 || height <= 0)
        return;

    auto msg = std::make_unique<sensor_msgs::msg::Image>();
    msg->header = header;
    msg->width = static_cast<uint32_t>(width);
    msg->height = static_cast<uint32_t>(height);
    msg->encoding = sensor_msgs::image_encodings::TYPE_16UC1;
    msg->is_bigendian = 0;
    msg->step = static_cast<uint32_t>(width * 2);
    const std::size_t bytes = static_cast<std::size_t>(height) * msg->step;
    msg->data.resize(bytes);
    std::memcpy(msg->data.data(), z16_data, bytes);
    pub->publish(std::move(msg));
}

void OutputBackend::publish_infra_from_cpu(int index,
                                           const std_msgs::msg::Header& header,
                                           int width,
                                           int height,
                                           const std::string& encoding,
                                           int bpp,
                                           const void* data)
{
    auto& pub = (index == 1) ? _infra1_pub : _infra2_pub;
    if (!pub || !data || width <= 0 || height <= 0 || bpp <= 0)
        return;

    auto msg = std::make_unique<sensor_msgs::msg::Image>();
    msg->header = header;
    msg->width = static_cast<uint32_t>(width);
    msg->height = static_cast<uint32_t>(height);
    msg->encoding = encoding;
    msg->is_bigendian = 0;
    msg->step = static_cast<uint32_t>(width * bpp);
    const std::size_t bytes = static_cast<std::size_t>(height) * msg->step;
    msg->data.resize(bytes);
    std::memcpy(msg->data.data(), data, bytes);
    pub->publish(std::move(msg));
}

void OutputBackend::publish_pointcloud_from_cpu(const std_msgs::msg::Header& header,
                                                uint32_t width,
                                                uint32_t height,
                                                bool is_dense,
                                                const PointCloudFieldsMeta& meta,
                                                const void* data,
                                                std::size_t bytes)
{
    if (!_std_pc_pub || !data || bytes == 0)
        return;

    auto msg = std::make_unique<sensor_msgs::msg::PointCloud2>();
    msg->header = header;
    msg->width = width;
    msg->height = height;
    msg->fields = meta.fields;
    msg->is_bigendian = false;
    msg->point_step = meta.point_step;
    msg->row_step = width * meta.point_step;
    msg->is_dense = is_dense;
    msg->data.resize(bytes);
    std::memcpy(msg->data.data(), data, bytes);
    _std_pc_pub->publish(std::move(msg));
}

#ifdef BUILD_WITH_QIRP
VkBuffer OutputBackend::get_or_import_dmabuf(int fd, std::size_t bytes)
{
    std::lock_guard<std::mutex> lock(_dmabuf_cache_mtx);
    auto it = _dmabuf_vk_cache.find(fd);
    if (it != _dmabuf_vk_cache.end() && it->second.size() >= bytes)
    {
        return it->second.buffer();
    }
    if (it != _dmabuf_vk_cache.end())
    {
        _dmabuf_vk_cache.erase(it);
    }
    ScopedVulkanBuffer imported = _vk_ctx->import_dmabuf_buffer(fd, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    VkBuffer handle = imported.buffer();
    _dmabuf_vk_cache.emplace(fd, std::move(imported));
    return handle;
}

void OutputBackend::evict_dmabuf_fd(int fd)
{
    if (fd < 0)
        return;
    std::lock_guard<std::mutex> lock(_dmabuf_cache_mtx);
    _dmabuf_vk_cache.erase(fd);
}
#endif

GpuOutputSlot OutputBackend::acquire_gpu_color_slot(int width, int height)
{
    GpuOutputSlot slot;
    if (!_vk_ctx || width <= 0 || height <= 0)
        return slot;

    const std::size_t bytes = color_output_bytes(width, height);
#ifdef BUILD_WITH_QIRP
    if (_use_qirp)
    {
        auto dmabuf = _color_pool.acquire(bytes);
        if (!dmabuf || dmabuf->fd() < 0)
            return slot;
        slot.vk_buffer = get_or_import_dmabuf(dmabuf->fd(), static_cast<std::size_t>(dmabuf->size()));
        slot.cpu_ptr = dmabuf->addr();
        slot.size = static_cast<std::size_t>(dmabuf->size());
        slot.is_dmabuf = true;
        slot.lease = std::move(dmabuf);
        return slot;
    }
#endif

    if (!_std_rgb_buf.valid() || _std_rgb_buf.size() < bytes)
    {
        _std_rgb_buf = _vk_ctx->create_buffer(
            bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
    }
    slot.vk_buffer = _std_rgb_buf.buffer();
    slot.cpu_ptr = _std_rgb_buf.mapped();
    slot.size = bytes;
    slot.is_dmabuf = false;
    return slot;
}

void OutputBackend::upload_cpu_rgb_to_slot(GpuOutputSlot& slot, int width, int height, const void* rgb8_data)
{
    if (!slot.valid() || !slot.cpu_ptr || !rgb8_data || width <= 0 || height <= 0)
        return;

    const std::size_t row_bytes = static_cast<std::size_t>(width) * 3u;
    const std::size_t dst_stride = static_cast<std::size_t>(color_output_stride(width));

#ifdef BUILD_WITH_QIRP
    if (slot.is_dmabuf && slot.lease)
    {
        auto dmabuf = std::static_pointer_cast<lib_mem_dmabuf::DmaBuffer>(slot.lease);
        dmabuf->sync_start();
        copy_image_2d(slot.cpu_ptr, dst_stride, rgb8_data, row_bytes, row_bytes, height);
        dmabuf->sync_end();
        return;
    }
#endif

    copy_image_2d(slot.cpu_ptr, dst_stride, rgb8_data, row_bytes, row_bytes, height);
    if (_vk_ctx && _std_rgb_buf.valid())
    {
        _vk_ctx->flush_buffer(_std_rgb_buf.get());
    }
}

void OutputBackend::publish_gpu_color_slot(const std_msgs::msg::Header& header,
                                           int width,
                                           int height,
                                           GpuOutputSlot&& slot,
                                           bool gpu_written)
{
    if (!slot.valid() || !has_color_subscribers())
        return;

#ifdef BUILD_WITH_QIRP
    if (slot.is_dmabuf && _qirp_color_pub && slot.lease)
    {
        auto dmabuf = std::static_pointer_cast<lib_mem_dmabuf::DmaBuffer>(std::move(slot.lease));
        // GPU L2 visibility is guaranteed by cmd_host_read_barrier + vkWaitForFences;
        // per Linux DMA-BUF contract, CPU sync_start/sync_end is owned by CPU readers/writers only.
        auto msg = std::make_unique<qrb_ros::transport::type::Image>();
        msg->header = header;
        msg->width = width;
        msg->height = height;
        msg->encoding = sensor_msgs::image_encodings::RGB8;
        msg->dmabuf = std::move(dmabuf);
        _qirp_color_pub->publish(std::move(msg));
        return;
    }
#endif

    if (gpu_written && _vk_ctx && _std_rgb_buf.valid())
    {
        _vk_ctx->invalidate_buffer(_std_rgb_buf.get());
    }
    publish_color_from_cpu(header, width, height, color_output_stride(width), slot.cpu_ptr);
}

GpuOutputSlot OutputBackend::acquire_gpu_pointcloud_slot(std::size_t max_bytes)
{
    GpuOutputSlot slot;
    if (!_vk_ctx || max_bytes == 0)
        return slot;

#ifdef BUILD_WITH_QIRP
    if (_use_qirp)
    {
        auto dmabuf = _pc_pool.acquire(max_bytes);
        if (!dmabuf || dmabuf->fd() < 0)
            return slot;
        slot.vk_buffer = get_or_import_dmabuf(dmabuf->fd(), static_cast<std::size_t>(dmabuf->size()));
        slot.cpu_ptr = dmabuf->addr();
        slot.size = static_cast<std::size_t>(dmabuf->size());
        slot.is_dmabuf = true;
        slot.lease = std::move(dmabuf);
        return slot;
    }
#endif

    if (!_std_pc_buf.valid() || _std_pc_buf.size() < max_bytes)
    {
        _std_pc_buf = _vk_ctx->create_buffer(
            max_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
    }
    slot.vk_buffer = _std_pc_buf.buffer();
    slot.cpu_ptr = _std_pc_buf.mapped();
    slot.size = max_bytes;
    slot.is_dmabuf = false;
    return slot;
}

void OutputBackend::publish_gpu_pointcloud_slot(const std_msgs::msg::Header& header,
                                                uint32_t width,
                                                uint32_t height,
                                                bool is_dense,
                                                const PointCloudFieldsMeta& meta,
                                                GpuOutputSlot&& slot)
{
    if (!slot.valid() || width == 0 || height == 0 || !has_pointcloud_subscribers())
        return;

#ifdef BUILD_WITH_QIRP
    if (slot.is_dmabuf && _qirp_pc_pub && slot.lease)
    {
        auto dmabuf = std::static_pointer_cast<lib_mem_dmabuf::DmaBuffer>(std::move(slot.lease));
        auto msg = std::make_unique<qrb_ros::transport::type::PointCloud2>();
        msg->header = header;
        msg->width = width;
        msg->height = height;
        msg->fields = meta.fields;
        msg->is_bigendian = false;
        msg->point_step = meta.point_step;
        msg->row_step = width * meta.point_step;
        msg->is_dense = is_dense;
        msg->data_fd = std::move(dmabuf);
        _qirp_pc_pub->publish(std::move(msg));
        return;
    }
#endif

    const std::size_t valid_bytes =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * meta.point_step;
    if (_vk_ctx && _std_pc_buf.valid())
    {
        _vk_ctx->invalidate_buffer(_std_pc_buf.get(), 0, valid_bytes);
    }
    publish_pointcloud_from_cpu(header, width, height, is_dense, meta, slot.cpu_ptr, valid_bytes);
}

}  // namespace realsense_qcom_bolt
