// Copyright 2026 voidrobot
// SPDX-License-Identifier: Apache-2.0

#include "realsense_qcom_bolt/math_utils.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace realsense_qcom_bolt;

namespace
{
void rotate_vec_by_quat(const Quaternion& q, double vx, double vy, double vz,
                        double& ox, double& oy, double& oz)
{
    // v' = q * (0, v) * q^-1
    const double tx = 2.0 * (q.y * vz - q.z * vy);
    const double ty = 2.0 * (q.z * vx - q.x * vz);
    const double tz = 2.0 * (q.x * vy - q.y * vx);
    ox = vx + q.w * tx + (q.y * tz - q.z * ty);
    oy = vy + q.w * ty + (q.z * tx - q.x * tz);
    oz = vz + q.w * tz + (q.x * ty - q.y * tx);
}

void verify_rot_matrix(const float r[9], const char* label)
{
    const Quaternion q = rotation_matrix_to_quaternion(r);
    const double norm = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    if (std::abs(norm - 1.0) > 1e-6)
    {
        std::cerr << "FAIL norm for " << label << ": " << norm << "\n";
        std::abort();
    }

    // Check R * v == q * v * q^-1 for an arbitrary non-axis vector
    const double vx = 0.37, vy = -1.23, vz = 0.89;
    // librealsense column-major: col0=(r[0..2]), col1=(r[3..5]), col2=(r[6..8])
    const double rx = r[0] * vx + r[3] * vy + r[6] * vz;
    const double ry = r[1] * vx + r[4] * vy + r[7] * vz;
    const double rz = r[2] * vx + r[5] * vy + r[8] * vz;

    double qx = 0.0, qy = 0.0, qz = 0.0;
    rotate_vec_by_quat(q, vx, vy, vz, qx, qy, qz);

    if (std::abs(rx - qx) > 1e-5 || std::abs(ry - qy) > 1e-5 || std::abs(rz - qz) > 1e-5)
    {
        std::cerr << "FAIL rotation mismatch for " << label << "\n";
        std::abort();
    }
}
}  // namespace

int main()
{
    // 1. align_up
    assert(align_up(0, 64) == 0);
    assert(align_up(1, 64) == 64);
    assert(align_up(64, 64) == 64);
    assert(align_up(65, 64) == 128);
    assert(align_up(100, 1) == 100);
    assert(align_up(100, 0) == 100);

    // 2. copy_image_2d (tight & padded stride)
    {
        const int w = 5, h = 3;
        const std::size_t row = w * 3;
        const std::size_t dst_stride = 24;
        std::vector<uint8_t> src(row * h);
        for (std::size_t i = 0; i < src.size(); ++i)
            src[i] = static_cast<uint8_t>((i * 7 + 3) & 0xFF);

        std::vector<uint8_t> dst(dst_stride * h, 0xAA);
        copy_image_2d(dst.data(), dst_stride, src.data(), row, row, h);
        for (int y = 0; y < h; ++y)
        {
            for (std::size_t x = 0; x < row; ++x)
            {
                assert(dst[y * dst_stride + x] == src[y * row + x]);
            }
            for (std::size_t x = row; x < dst_stride; ++x)
            {
                assert(dst[y * dst_stride + x] == 0xAA);
            }
        }
    }

    // 3. rotation_matrix_to_quaternion across all 4 Shepperd branches
    {
        // Identity (trace > 0)
        const float id[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        verify_rot_matrix(id, "identity");

        // 180 deg around X (m00 dominant, trace = -1)
        const float rx180[9] = {1, 0, 0, 0, -1, 0, 0, 0, -1};
        verify_rot_matrix(rx180, "rot_x_180");

        // 180 deg around Y (m11 dominant, trace = -1)
        const float ry180[9] = {-1, 0, 0, 0, 1, 0, 0, 0, -1};
        verify_rot_matrix(ry180, "rot_y_180");

        // 180 deg around Z (m22 dominant, trace = -1)
        const float rz180[9] = {-1, 0, 0, 0, -1, 0, 0, 0, 1};
        verify_rot_matrix(rz180, "rot_z_180");

        // 90 deg around Z (column-major: col0=(0,1,0), col1=(-1,0,0), col2=(0,0,1))
        const float rz90[9] = {0, 1, 0, -1, 0, 0, 0, 0, 1};
        verify_rot_matrix(rz90, "rot_z_90");

        // Realistic small D455 depth->color calibration rotation
        const float a = 0.015f, b = -0.008f;
        const float ca = std::cos(a), sa = std::sin(a), cb = std::cos(b), sb = std::sin(b);
        const float d455[9] = {
            cb, 0, -sb,
            sa * sb, ca, sa * cb,
            ca * sb, -sa, ca * cb
        };
        verify_rot_matrix(d455, "d455_small_angle");
    }

    // 4. ImuSynchronizer (COPY & LINEAR_INTERPOLATION & queue cap)
    {
        ImuSynchronizer sync_copy(UniteImuMethod::COPY);
        assert(!sync_copy.push_gyro({10.0, 1, 2, 3}).has_value());
        sync_copy.push_accel({5.0, 0.0f, -9.8f, 0.0f});
        sync_copy.push_accel({15.0, 2.0f, -9.8f, 0.0f});
        auto u = sync_copy.push_gyro({10.0, 0.1f, 0.2f, 0.3f});
        assert(u.has_value());
        assert(std::abs(u->ax - 2.0f) < 1e-6f);  // COPY uses latest
        assert(sync_copy.accel_queue_size() == 1u);
    }
    {
        ImuSynchronizer sync_lerp(UniteImuMethod::LINEAR_INTERPOLATION, 8);
        sync_lerp.push_accel({10.0, 0.0f, 10.0f, 20.0f});
        sync_lerp.push_accel({20.0, 10.0f, 20.0f, 40.0f});
        auto u = sync_lerp.push_gyro({12.5, 1.0f, 2.0f, 3.0f});
        assert(u.has_value());
        assert(std::abs(u->ax - 2.5f) < 1e-5f);
        assert(std::abs(u->ay - 12.5f) < 1e-5f);
        assert(std::abs(u->az - 25.0f) < 1e-5f);

        // Queue cap stress
        for (int i = 0; i < 50; ++i)
        {
            sync_lerp.push_accel({30.0 + i, static_cast<float>(i), 0.0f, 0.0f});
        }
        assert(sync_lerp.accel_queue_size() <= 8u);
    }

    std::cout << "ALL_MATH_AND_IMU_TESTS_PASSED" << std::endl;
    return 0;
}
