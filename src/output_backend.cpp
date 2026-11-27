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
                             const rclcpp::QoS& qos)
    : _node(node),
      _use_qirp(use_qirp_transport),
      _qos(qos)
#ifdef BUILD_WITH_QIRP
      ,
      _color_pool(qirp_pool_size, defaults::kDefaultDmaHeapPath,
                  [this](const lib_mem_dmabuf::DmaBuffer& b) { notify_dmabuf_discard(b.fd()); }),
      _pc_pool(qirp_pool_size, defaults::kDefaultDmaHeapPath,
               [this](const lib_mem_dmabuf::DmaBuffer& b) { notify_dmabuf_discard(b.fd()); })
#endif
{
#ifndef BUILD_WITH_QIRP
    (void)qirp_pool_size;
    _use_qirp = false;
#endif
}

void OutputBackend::set_dmabuf_discard_callback(std::function<void(int)> cb)
{
#ifdef BUILD_WITH_QIRP
    std::lock_guard<std::mutex> lock(_discard_mtx);
    _on_dmabuf_discard = std::move(cb);
#else
    (void)cb;
#endif
}

#ifdef BUILD_WITH_QIRP
void OutputBackend::notify_dmabuf_discard(int fd)
{
    if (fd < 0)
        return;
    std::function<void(int)> cb;
    {
        std::lock_guard<std::mutex> lock(_discard_mtx);
        cb = _on_dmabuf_discard;
    }
    if (cb)
        cb(fd);
}
#endif

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
    const int row_bytes = width * 3;
    if (src_step < row_bytes)
        return;

#ifdef BUILD_WITH_QIRP
    if (_qirp_color_pub)
    {
        const int dst_stride = color_output_stride(width);
        const std::size_t align_bytes = color_output_bytes(width, height);
        if (dst_stride < row_bytes ||
            static_cast<std::size_t>(height) * static_cast<std::size_t>(dst_stride) > align_bytes)
        {
            return;
        }
        auto dmabuf = _color_pool.acquire(align_bytes);
        if (!dmabuf || !dmabuf->addr())
            return;

        dmabuf->sync_start();
        auto* dst = static_cast<uint8_t*>(dmabuf->addr());
        const auto* src = static_cast<const uint8_t*>(rgb8_data);
        if (dst_stride == row_bytes && src_step == row_bytes)
        {
            std::memcpy(dst, src, static_cast<std::size_t>(height) * static_cast<std::size_t>(row_bytes));
        }
        else
        {
            for (int y = 0; y < height; ++y)
            {
                std::memcpy(dst + static_cast<std::size_t>(y) * static_cast<std::size_t>(dst_stride),
                            src + static_cast<std::size_t>(y) * static_cast<std::size_t>(src_step),
                            static_cast<std::size_t>(row_bytes));
            }
        }
        dmabuf->sync_end();
        publish_color_dmabuf(header, width, height, std::move(dmabuf));
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
    msg->data.resize(static_cast<std::size_t>(height) * static_cast<std::size_t>(row_bytes));
    if (src_step == row_bytes)
    {
        std::memcpy(msg->data.data(), rgb8_data, msg->data.size());
    }
    else
    {
        const auto* src = static_cast<const uint8_t*>(rgb8_data);
        for (int y = 0; y < height; ++y)
        {
            std::memcpy(msg->data.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(row_bytes),
                        src + static_cast<std::size_t>(y) * static_cast<std::size_t>(src_step),
                        static_cast<std::size_t>(row_bytes));
        }
    }
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
std::shared_ptr<lib_mem_dmabuf::DmaBuffer> OutputBackend::acquire_color_dmabuf(int width, int height)
{
    return _color_pool.acquire(color_output_bytes(width, height));
}

std::shared_ptr<lib_mem_dmabuf::DmaBuffer> OutputBackend::acquire_pointcloud_dmabuf(std::size_t bytes)
{
    return _pc_pool.acquire(bytes);
}

void OutputBackend::publish_color_dmabuf(const std_msgs::msg::Header& header,
                                         int width,
                                         int height,
                                         std::shared_ptr<lib_mem_dmabuf::DmaBuffer> dmabuf)
{
    if (!_qirp_color_pub || !dmabuf)
        return;
    auto msg = std::make_unique<qrb_ros::transport::type::Image>();
    msg->header = header;
    msg->width = width;
    msg->height = height;
    msg->encoding = sensor_msgs::image_encodings::RGB8;
    msg->dmabuf = std::move(dmabuf);
    _qirp_color_pub->publish(std::move(msg));
}

void OutputBackend::publish_pointcloud_dmabuf(const std_msgs::msg::Header& header,
                                              uint32_t width,
                                              uint32_t height,
                                              bool is_dense,
                                              const PointCloudFieldsMeta& meta,
                                              std::shared_ptr<lib_mem_dmabuf::DmaBuffer> dmabuf)
{
    if (!_qirp_pc_pub || !dmabuf)
        return;
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
}
#endif

}  // namespace realsense_qcom_bolt
