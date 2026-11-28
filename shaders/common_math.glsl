// Copyright 2026 voidrobot
// SPDX-License-Identifier: Apache-2.0
// Projection and distortion math follows Intel librealsense2 (Apache-2.0).

// Shared GLSL 450 math, calibration structures, and compaction helpers for realsense_qcom_bolt.

struct Intrinsics
{
    int width;
    int height;
    float ppx;
    float ppy;
    float fx;
    float fy;
    int model;  // rs2_distortion enum value
    int pad0;
    vec4 coeffs0;  // coeffs[0..3]
    vec4 coeffs1;  // coeffs[4], 0, 0, 0
};

struct Extrinsics
{
    // Column-major 3x3 rotation stored as 3 vec4 columns + translation
    vec4 col0;  // r[0], r[1], r[2], 0
    vec4 col1;  // r[3], r[4], r[5], 0
    vec4 col2;  // r[6], r[7], r[8], 0
    vec4 trans; // t[0], t[1], t[2], 0
};

struct CalibData
{
    Intrinsics depth_intrin;
    Intrinsics color_intrin;
    Extrinsics depth_to_color;
    float depth_scale;
    float min_depth_m;
    float max_depth_m;
    uint color_stride_bytes;
    uint align_fill_2x2;
    uint pad1;
    uint pad2;
    uint pad3;
};

uvec3 yuyv_to_rgb8_bt601(uint y_val, uint u_val, uint v_val)
{
    int c = int(y_val) - 16;
    int d = int(u_val) - 128;
    int e = int(v_val) - 128;
    int r = clamp((298 * c + 409 * e + 128) >> 8, 0, 255);
    int g = clamp((298 * c - 100 * d - 208 * e + 128) >> 8, 0, 255);
    int b = clamp((298 * c + 516 * d + 128) >> 8, 0, 255);
    return uvec3(uint(r), uint(g), uint(b));
}

// Extract RGB8 pixel from two uint32 words; callers must only read w1 when (byte_offset & 3u) != 0u.
uvec3 unpack_rgb8_pixel(uint byte_offset, uint w0, uint w1)
{
    uint shift = (byte_offset & 3u) * 8u;
    uint packed = (shift == 0u) ? w0 : ((w0 >> shift) | (w1 << (32u - shift)));
    return uvec3(packed & 0xFFu, (packed >> 8u) & 0xFFu, (packed >> 16u) & 0xFFu);
}

// Safe OOB-free RGB8 load macro: avoids reading rgb_buf[widx + 1u] when 4-byte aligned.
#define BOLT_LOAD_RGB8(rgb_buf, byte_off)                                              \
    unpack_rgb8_pixel((byte_off), (rgb_buf)[(byte_off) >> 2u],                         \
                      (((byte_off) & 3u) != 0u) ? (rgb_buf)[((byte_off) >> 2u) + 1u] : 0u)

// Pack RGB8 into ROS PointCloud2 little-endian BGR float32 bit pattern.
uint pack_rgb_as_ros_bgr(uvec3 rgb)
{
    return (rgb.b) | (rgb.g << 8u) | (rgb.r << 16u);
}

const int DISTORTION_NONE = 0;
const int DISTORTION_MODIFIED_BROWN_CONRADY = 1;
const int DISTORTION_INVERSE_BROWN_CONRADY = 2;
const int DISTORTION_FTHETA = 3;
const int DISTORTION_BROWN_CONRADY = 4;
const int DISTORTION_KANNALA_BRANDT4 = 5;

vec3 deproject_pixel_to_point(Intrinsics intrin, vec2 pixel, float depth_m)
{
    float x = (pixel.x - intrin.ppx) / intrin.fx;
    float y = (pixel.y - intrin.ppy) / intrin.fy;

    if (intrin.model == DISTORTION_INVERSE_BROWN_CONRADY ||
        intrin.model == DISTORTION_BROWN_CONRADY)
    {
        float r2 = x * x + y * y;
        float f = 1.0 + intrin.coeffs0.x * r2 + intrin.coeffs0.y * r2 * r2 + intrin.coeffs1.x * r2 * r2 * r2;
        float ux = x * f + 2.0 * intrin.coeffs0.z * x * y + intrin.coeffs0.w * (r2 + 2.0 * x * x);
        float uy = y * f + 2.0 * intrin.coeffs0.w * x * y + intrin.coeffs0.z * (r2 + 2.0 * y * y);
        x = ux;
        y = uy;
    }
    else if (intrin.model == DISTORTION_FTHETA)
    {
        float rd = sqrt(x * x + y * y);
        if (rd > 1e-6)
        {
            float r = tan(rd) / rd;
            x *= r;
            y *= r;
        }
    }
    return vec3(depth_m * x, depth_m * y, depth_m);
}

// Unified projection matching librealsense NEON backends:
// - align (distort_inverse_bc == false): matches neon-align.cpp (skips INVERSE_BROWN_CONRADY)
// - pointcloud (distort_inverse_bc == true): matches neon-pointcloud.cpp (applies Modified BC for INVERSE_BC)
vec2 project_point_to_pixel(Intrinsics intrin, vec3 pt, bool distort_inverse_bc)
{
    float x = pt.x / pt.z;
    float y = pt.y / pt.z;

    if (intrin.model == DISTORTION_MODIFIED_BROWN_CONRADY ||
        (distort_inverse_bc && intrin.model == DISTORTION_INVERSE_BROWN_CONRADY))
    {
        float r2 = x * x + y * y;
        float f = 1.0 + intrin.coeffs0.x * r2 + intrin.coeffs0.y * r2 * r2 + intrin.coeffs1.x * r2 * r2 * r2;
        x *= f;
        y *= f;
        float dx = x + 2.0 * intrin.coeffs0.z * x * y + intrin.coeffs0.w * (r2 + 2.0 * x * x);
        float dy = y + 2.0 * intrin.coeffs0.w * x * y + intrin.coeffs0.z * (r2 + 2.0 * y * y);
        x = dx;
        y = dy;
    }
    else if (intrin.model == DISTORTION_BROWN_CONRADY)
    {
        float r2 = x * x + y * y;
        float f = 1.0 + intrin.coeffs0.x * r2 + intrin.coeffs0.y * r2 * r2 + intrin.coeffs1.x * r2 * r2 * r2;
        float xf = x * f;
        float yf = y * f;
        float dx = xf + 2.0 * intrin.coeffs0.z * x * y + intrin.coeffs0.w * (r2 + 2.0 * x * x);
        float dy = yf + 2.0 * intrin.coeffs0.w * x * y + intrin.coeffs0.z * (r2 + 2.0 * y * y);
        x = dx;
        y = dy;
    }
    else if (intrin.model == DISTORTION_FTHETA)
    {
        float r = sqrt(x * x + y * y);
        if (r > 1e-6)
        {
            float rd = atan(r) / r;
            x *= rd;
            y *= rd;
        }
    }
    return vec2(x * intrin.fx + intrin.ppx, y * intrin.fy + intrin.ppy);
}

vec3 transform_point_to_point(Extrinsics extr, vec3 pt)
{
    return extr.col0.xyz * pt.x + extr.col1.xyz * pt.y + extr.col2.xyz * pt.z + extr.trans.xyz;
}

// Shared 2-stage workgroup compaction helpers (reduces global atomicAdd by 1/256).
#define BOLT_DECLARE_COMPACTION_SHARED \
    shared uint s_wg_count;            \
    shared uint s_wg_base;

#define BOLT_COMPACTION_INIT(ordered_const)             \
    if ((ordered_const) == 0u && gl_LocalInvocationIndex == 0u) \
    {                                                   \
        s_wg_count = 0u;                                \
        s_wg_base = 0u;                                 \
    }                                                   \
    if ((ordered_const) == 0u)                          \
    {                                                   \
        barrier();                                      \
    }

#define BOLT_COMPACTION_RESERVE(is_valid, counter_buf, out_global_idx) \
    uint bolt_local_idx_ = 0u;                                         \
    if (is_valid)                                                      \
    {                                                                  \
        bolt_local_idx_ = atomicAdd(s_wg_count, 1u);                   \
    }                                                                  \
    barrier();                                                         \
    if (gl_LocalInvocationIndex == 0u && s_wg_count > 0u)              \
    {                                                                  \
        s_wg_base = atomicAdd((counter_buf)[0], s_wg_count);           \
    }                                                                  \
    barrier();                                                         \
    uint out_global_idx = s_wg_base + bolt_local_idx_;
