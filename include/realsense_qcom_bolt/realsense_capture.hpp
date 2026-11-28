// Copyright 2026 voidrobot
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "realsense_qcom_bolt/config.hpp"
#include "realsense_qcom_bolt/math_utils.hpp"
#include "realsense_qcom_bolt/vulkan_pipeline.hpp"

#include <librealsense2/rs.hpp>

#include <atomic>
#include <functional>
#include <thread>

namespace realsense_qcom_bolt
{

// Thin librealsense2 capture driver (NO host processing_blocks in P1..P6).
// Separates video pipeline from asynchronous high-rate motion (IMU) pipeline on live cameras.
class RealSenseCapture
{
public:
    using FramesetCallback = std::function<void(const rs2::frameset&)>;
    using MotionCallback = std::function<void(const rs2::motion_frame&)>;

    RealSenseCapture(const NodeConfig& cfg, PipelinePreset preset);
    ~RealSenseCapture();

    RealSenseCapture(const RealSenseCapture&) = delete;
    RealSenseCapture& operator=(const RealSenseCapture&) = delete;

    void open();
    void start_loop(FramesetCallback on_frameset, MotionCallback on_motion = nullptr);
    void stop();

    const StreamCalibration& calibration() const { return _calib; }

private:
    NodeConfig _cfg;
    PipelinePreset _preset;
    rs2::pipeline _pipe;
    rs2::pipeline_profile _profile;
    bool _video_started{false};

    rs2::pipeline _imu_pipe;
    bool _imu_separate{false};
    bool _imu_started{false};

    StreamCalibration _calib;

    alignas(kCacheLineBytes) std::atomic<bool> _running{false};
    std::thread _worker;
    MotionCallback _motion_cb;
};

}  // namespace realsense_qcom_bolt
