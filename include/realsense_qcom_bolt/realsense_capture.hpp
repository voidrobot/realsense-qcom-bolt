#pragma once

#include "realsense_qcom_bolt/config.hpp"
#include "realsense_qcom_bolt/vulkan_pipeline.hpp"

#include <librealsense2/rs.hpp>

#include <atomic>
#include <functional>
#include <memory>
#include <thread>

namespace realsense_qcom_bolt
{

// Thin librealsense2 capture driver (NO host processing_blocks in P1..P6).
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
    StreamCalibration _calib;

    std::atomic<bool> _running{false};
    std::thread _worker;
};

}  // namespace realsense_qcom_bolt
