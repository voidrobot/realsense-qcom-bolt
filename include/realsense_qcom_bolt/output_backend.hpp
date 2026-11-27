#pragma once

#include "realsense_qcom_bolt/config.hpp"
#include "realsense_qcom_bolt/dmabuf_pool.hpp"
#include "realsense_qcom_bolt/math_utils.hpp"
#include "realsense_qcom_bolt/vulkan_context.hpp"

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/header.hpp>

#ifdef BUILD_WITH_QIRP
#include <qrb_ros_transport_image_type/image.hpp>
#include <qrb_ros_transport_image_type/image_utils.hpp>
#include <qrb_ros_transport_point_cloud2_type/point_cloud2.hpp>
#endif

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace realsense_qcom_bolt
{

struct PointCloudFieldsMeta
{
    bool has_rgb{true};
    uint32_t point_step{16};
    std::vector<sensor_msgs::msg::PointField> fields;
};

PointCloudFieldsMeta make_pointcloud_fields_meta(bool has_rgb);

// Backend-agnostic GPU output slot (wraps either an imported QIRP DMA-BUF or a HOST_CACHED VulkanBuffer).
struct GpuOutputSlot
{
    VkBuffer vk_buffer{VK_NULL_HANDLE};
    void* cpu_ptr{nullptr};
    std::size_t size{0};
    bool is_dmabuf{false};
    std::shared_ptr<void> lease;

    bool valid() const { return vk_buffer != VK_NULL_HANDLE; }
};

// Encapsulates QIRP DMA-BUF Zero-Copy vs standard ROS 2 publishers and Vulkan DMA-BUF import cache.
// All #ifdef BUILD_WITH_QIRP branches are strictly isolated inside this class.
class OutputBackend
{
public:
    OutputBackend(rclcpp::Node& node,
                  bool use_qirp_transport,
                  std::size_t qirp_pool_size,
                  const rclcpp::QoS& qos,
                  VulkanContext* vk_ctx = nullptr);
    ~OutputBackend();

    OutputBackend(const OutputBackend&) = delete;
    OutputBackend& operator=(const OutputBackend&) = delete;

    void create_color_publisher(const std::string& topic);
    void create_depth_publisher(const std::string& topic);
    void create_aligned_depth_publisher(const std::string& topic);
    void create_pointcloud_publisher(const std::string& topic);
    void create_infra_publisher(int index, const std::string& topic);

    bool has_color_subscribers() const;
    bool has_depth_subscribers() const;
    bool has_aligned_depth_subscribers() const;
    bool has_pointcloud_subscribers() const;
    bool has_infra_subscribers(int index) const;

    bool use_qirp() const { return _use_qirp; }

    int color_output_stride(int width) const;
    std::size_t color_output_bytes(int width, int height) const;

    // CPU-source publishing (used by P0 bypass_only and non-QIRP image outputs).
    void publish_color_from_cpu(const std_msgs::msg::Header& header,
                                int width,
                                int height,
                                int src_step,
                                const void* rgb8_data);

    void publish_depth_from_cpu(const std_msgs::msg::Header& header,
                                int width,
                                int height,
                                const void* z16_data,
                                bool is_aligned);

    void publish_infra_from_cpu(int index,
                                const std_msgs::msg::Header& header,
                                int width,
                                int height,
                                const std::string& encoding,
                                int bpp,
                                const void* data);

    void publish_pointcloud_from_cpu(const std_msgs::msg::Header& header,
                                     uint32_t width,
                                     uint32_t height,
                                     bool is_dense,
                                     const PointCloudFieldsMeta& meta,
                                     const void* data,
                                     std::size_t bytes);

    // Backend-agnostic GPU slot lifecycle (used by VulkanPipeline with zero #ifdef BUILD_WITH_QIRP).
    GpuOutputSlot acquire_gpu_color_slot(int width, int height);
    void upload_cpu_rgb_to_slot(GpuOutputSlot& slot, int width, int height, const void* rgb8_data);
    void publish_gpu_color_slot(const std_msgs::msg::Header& header,
                                int width,
                                int height,
                                GpuOutputSlot&& slot,
                                bool gpu_written);

    GpuOutputSlot acquire_gpu_pointcloud_slot(std::size_t max_bytes);
    void publish_gpu_pointcloud_slot(const std_msgs::msg::Header& header,
                                     uint32_t width,
                                     uint32_t height,
                                     bool is_dense,
                                     const PointCloudFieldsMeta& meta,
                                     GpuOutputSlot&& slot);

private:
    rclcpp::Node& _node;
    bool _use_qirp{false};
    rclcpp::QoS _qos;
    VulkanContext* _vk_ctx{nullptr};

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr _std_color_pub;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr _depth_pub;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr _aligned_depth_pub;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr _infra1_pub;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr _infra2_pub;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr _std_pc_pub;

    // Standard-mode HOST_CACHED GPU output buffers (allocated lazily when use_qirp == false)
    ScopedVulkanBuffer _std_rgb_buf;
    ScopedVulkanBuffer _std_pc_buf;

#ifdef BUILD_WITH_QIRP
    VkBuffer get_or_import_dmabuf(int fd, std::size_t bytes);
    void evict_dmabuf_fd(int fd);

    rclcpp::Publisher<qrb_ros::transport::type::Image>::SharedPtr _qirp_color_pub;
    rclcpp::Publisher<qrb_ros::transport::type::PointCloud2>::SharedPtr _qirp_pc_pub;
    DmaBufferPool _color_pool;
    DmaBufferPool _pc_pool;
    alignas(kCacheLineBytes) std::mutex _dmabuf_cache_mtx;
    std::unordered_map<int, ScopedVulkanBuffer> _dmabuf_vk_cache;
#endif
};

}  // namespace realsense_qcom_bolt
