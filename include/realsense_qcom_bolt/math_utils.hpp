// Copyright 2026 voidrobot
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "realsense_qcom_bolt/config.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <optional>

namespace realsense_qcom_bolt
{

inline constexpr std::size_t kCacheLineBytes = 64u;

constexpr std::size_t align_up(std::size_t value, std::size_t alignment)
{
    if (alignment <= 1u)
        return value;
    return ((value + alignment - 1u) / alignment) * alignment;
}

inline void copy_image_2d(void* dst,
                          std::size_t dst_stride,
                          const void* src,
                          std::size_t src_stride,
                          std::size_t row_bytes,
                          int height)
{
    if (!dst || !src || row_bytes == 0 || height <= 0 ||
        dst_stride < row_bytes || src_stride < row_bytes)
    {
        return;
    }
    auto* d = static_cast<uint8_t*>(dst);
    const auto* s = static_cast<const uint8_t*>(src);
    if (dst_stride == row_bytes && src_stride == row_bytes)
    {
        std::memcpy(d, s, static_cast<std::size_t>(height) * row_bytes);
        return;
    }
    for (int y = 0; y < height; ++y)
    {
        std::memcpy(d + static_cast<std::size_t>(y) * dst_stride,
                    s + static_cast<std::size_t>(y) * src_stride,
                    row_bytes);
    }
}

struct Quaternion
{
    double x{0.0};
    double y{0.0};
    double z{0.0};
    double w{1.0};
};

// Convert librealsense column-major 3x3 rotation matrix (r[col*3 + row]) to a unit quaternion.
inline Quaternion rotation_matrix_to_quaternion(const float r[9])
{
    const double m00 = r[0], m01 = r[3], m02 = r[6];
    const double m10 = r[1], m11 = r[4], m12 = r[7];
    const double m20 = r[2], m21 = r[5], m22 = r[8];

    Quaternion q;
    const double trace = m00 + m11 + m22;
    if (trace > 0.0)
    {
        const double s = std::sqrt(trace + 1.0) * 2.0;
        q.w = 0.25 * s;
        q.x = (m21 - m12) / s;
        q.y = (m02 - m20) / s;
        q.z = (m10 - m01) / s;
    }
    else if (m00 > m11 && m00 > m22)
    {
        const double s = std::sqrt(1.0 + m00 - m11 - m22) * 2.0;
        q.w = (m21 - m12) / s;
        q.x = 0.25 * s;
        q.y = (m01 + m10) / s;
        q.z = (m02 + m20) / s;
    }
    else if (m11 > m22)
    {
        const double s = std::sqrt(1.0 + m11 - m00 - m22) * 2.0;
        q.w = (m02 - m20) / s;
        q.x = (m01 + m10) / s;
        q.y = 0.25 * s;
        q.z = (m12 + m21) / s;
    }
    else
    {
        const double s = std::sqrt(1.0 + m22 - m00 - m11) * 2.0;
        q.w = (m10 - m01) / s;
        q.x = (m02 + m20) / s;
        q.y = (m12 + m21) / s;
        q.z = 0.25 * s;
    }

    const double norm = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    if (norm > 1e-12)
    {
        q.x /= norm;
        q.y /= norm;
        q.z /= norm;
        q.w /= norm;
    }
    else
    {
        q = Quaternion{};
    }
    return q;
}

struct ImuSample
{
    double timestamp_ms{0.0};
    float x{0.0f};
    float y{0.0f};
    float z{0.0f};
};

struct UnitedImuSample
{
    double timestamp_ms{0.0};
    float gx{0.0f}, gy{0.0f}, gz{0.0f};
    float ax{0.0f}, ay{0.0f}, az{0.0f};
};

// Combines asynchronous accelerometer & gyroscope samples using COPY or LINEAR_INTERPOLATION.
class ImuSynchronizer
{
public:
    explicit ImuSynchronizer(UniteImuMethod method, std::size_t max_queue = 64u)
        : _method(method), _max_queue(std::max<std::size_t>(2u, max_queue))
    {
    }

    void push_accel(const ImuSample& a)
    {
        if (_method == UniteImuMethod::COPY)
        {
            _accel_buf.clear();
            _accel_buf.push_back(a);
            return;
        }
        if (!_accel_buf.empty() && a.timestamp_ms < _accel_buf.back().timestamp_ms)
        {
            _accel_buf.clear();
        }
        _accel_buf.push_back(a);
        while (_accel_buf.size() > _max_queue)
        {
            _accel_buf.pop_front();
        }
    }

    std::optional<UnitedImuSample> push_gyro(const ImuSample& g)
    {
        if (_method == UniteImuMethod::NONE || _accel_buf.empty())
            return std::nullopt;

        if (_method == UniteImuMethod::COPY || _accel_buf.size() == 1u)
        {
            const auto& a = _accel_buf.back();
            return UnitedImuSample{g.timestamp_ms, g.x, g.y, g.z, a.x, a.y, a.z};
        }

        // LINEAR_INTERPOLATION: find bracketing accel samples [a0, a1] with t0 <= g.t <= t1
        if (g.timestamp_ms <= _accel_buf.front().timestamp_ms)
        {
            const auto& a = _accel_buf.front();
            return UnitedImuSample{g.timestamp_ms, g.x, g.y, g.z, a.x, a.y, a.z};
        }
        if (g.timestamp_ms >= _accel_buf.back().timestamp_ms)
        {
            const auto& a = _accel_buf.back();
            return UnitedImuSample{g.timestamp_ms, g.x, g.y, g.z, a.x, a.y, a.z};
        }

        for (std::size_t i = 1; i < _accel_buf.size(); ++i)
        {
            const auto& a0 = _accel_buf[i - 1u];
            const auto& a1 = _accel_buf[i];
            if (g.timestamp_ms >= a0.timestamp_ms && g.timestamp_ms <= a1.timestamp_ms)
            {
                const double dt = a1.timestamp_ms - a0.timestamp_ms;
                const double alpha = (dt > 1e-9) ? ((g.timestamp_ms - a0.timestamp_ms) / dt) : 0.0;
                const float ax = static_cast<float>(a0.x + (a1.x - a0.x) * alpha);
                const float ay = static_cast<float>(a0.y + (a1.y - a0.y) * alpha);
                const float az = static_cast<float>(a0.z + (a1.z - a0.z) * alpha);
                // Trim older samples no longer needed for monotonic timestamps
                if (i > 1u)
                {
                    _accel_buf.erase(_accel_buf.begin(), _accel_buf.begin() + static_cast<std::ptrdiff_t>(i - 1u));
                }
                return UnitedImuSample{g.timestamp_ms, g.x, g.y, g.z, ax, ay, az};
            }
        }
        const auto& a = _accel_buf.back();
        return UnitedImuSample{g.timestamp_ms, g.x, g.y, g.z, a.x, a.y, a.z};
    }

    std::size_t accel_queue_size() const { return _accel_buf.size(); }

private:
    UniteImuMethod _method;
    std::size_t _max_queue;
    std::deque<ImuSample> _accel_buf;
};

}  // namespace realsense_qcom_bolt
