// Copyright 2026 voidrobot
// SPDX-License-Identifier: Apache-2.0

#include "realsense_qcom_bolt/realsense_capture.hpp"

#include <stdexcept>

namespace realsense_qcom_bolt
{

RealSenseCapture::RealSenseCapture(const NodeConfig& cfg, PipelinePreset preset)
    : _cfg(cfg),
      _preset(preset)
{
}

RealSenseCapture::~RealSenseCapture()
{
    stop();
}

void RealSenseCapture::open()
{
    const bool has_video = _cfg.enable_color || _cfg.enable_depth || _cfg.enable_infra1 || _cfg.enable_infra2;
    const bool has_imu = _cfg.enable_gyro || _cfg.enable_accel;
    const bool is_playback = !_cfg.rosbag_filename.empty();

    // Separate live IMU pipeline so 200 Hz motion is never throttled by 30/60 Hz video framesets.
    _imu_separate = (has_imu && !is_playback);

    if (has_video || is_playback)
    {
        rs2::config rs_cfg;
        if (is_playback)
        {
            rs_cfg.enable_device_from_file(_cfg.rosbag_filename, _cfg.rosbag_loop);
        }
        else if (!_cfg.serial_no.empty())
        {
            rs_cfg.enable_device(_cfg.serial_no);
        }

        const bool use_yuyv_for_gpu = (_preset != PipelinePreset::P0_BYPASS_ONLY && !is_playback);
        if (_cfg.enable_color)
        {
            if (!is_playback)
            {
                rs_cfg.enable_stream(
                    RS2_STREAM_COLOR,
                    _cfg.color_profile.width,
                    _cfg.color_profile.height,
                    use_yuyv_for_gpu ? RS2_FORMAT_YUYV : RS2_FORMAT_RGB8,
                    _cfg.color_profile.fps);
            }
            else
            {
                rs_cfg.enable_stream(RS2_STREAM_COLOR);
            }
        }
        if (_cfg.enable_depth)
        {
            if (!is_playback)
            {
                rs_cfg.enable_stream(
                    RS2_STREAM_DEPTH,
                    _cfg.depth_profile.width,
                    _cfg.depth_profile.height,
                    RS2_FORMAT_Z16,
                    _cfg.depth_profile.fps);
            }
            else
            {
                rs_cfg.enable_stream(RS2_STREAM_DEPTH);
            }
        }
        if (_cfg.enable_infra1)
        {
            const rs2_format fmt = (_cfg.infra_format == "y16") ? RS2_FORMAT_Y16 : RS2_FORMAT_Y8;
            rs_cfg.enable_stream(
                RS2_STREAM_INFRARED, 1,
                _cfg.infra_profile.width, _cfg.infra_profile.height, fmt, _cfg.infra_profile.fps);
        }
        if (_cfg.enable_infra2)
        {
            const rs2_format fmt = (_cfg.infra_format == "y16") ? RS2_FORMAT_Y16 : RS2_FORMAT_Y8;
            rs_cfg.enable_stream(
                RS2_STREAM_INFRARED, 2,
                _cfg.infra_profile.width, _cfg.infra_profile.height, fmt, _cfg.infra_profile.fps);
        }
        if (is_playback)
        {
            if (_cfg.enable_gyro)
                rs_cfg.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F, _cfg.gyro_fps);
            if (_cfg.enable_accel)
                rs_cfg.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F, _cfg.accel_fps);
        }

        _profile = _pipe.start(rs_cfg);
        _video_started = true;

        auto dev = _profile.get_device();
        for (auto&& s : dev.query_sensors())
        {
            if (!is_playback)
            {
                if (s.supports(RS2_OPTION_GLOBAL_TIME_ENABLED) &&
                    !s.is_option_read_only(RS2_OPTION_GLOBAL_TIME_ENABLED))
                {
                    s.set_option(RS2_OPTION_GLOBAL_TIME_ENABLED, _cfg.global_time_enabled ? 1.0f : 0.0f);
                }
                if (s.supports(RS2_OPTION_AUTO_EXPOSURE_PRIORITY) &&
                    !s.is_option_read_only(RS2_OPTION_AUTO_EXPOSURE_PRIORITY))
                {
                    s.set_option(RS2_OPTION_AUTO_EXPOSURE_PRIORITY, 0.0f);
                }
            }
            if (auto ds = s.as<rs2::depth_sensor>())
            {
                _calib.depth_scale = ds.get_depth_scale();
            }
        }

        rs2::stream_profile depth_sp;
        rs2::stream_profile color_sp;
        for (const auto& sp : _profile.get_streams())
        {
            if (sp.stream_type() == RS2_STREAM_DEPTH)
            {
                depth_sp = sp;
                _calib.depth_intrin = sp.as<rs2::video_stream_profile>().get_intrinsics();
                _calib.has_depth = true;
            }
            else if (sp.stream_type() == RS2_STREAM_COLOR)
            {
                color_sp = sp;
                _calib.color_intrin = sp.as<rs2::video_stream_profile>().get_intrinsics();
                _calib.has_color = true;
            }
            else if (sp.stream_type() == RS2_STREAM_INFRARED)
            {
                if (sp.stream_index() == 1)
                {
                    _calib.infra1_intrin = sp.as<rs2::video_stream_profile>().get_intrinsics();
                    _calib.has_infra1 = true;
                }
                else if (sp.stream_index() == 2)
                {
                    _calib.infra2_intrin = sp.as<rs2::video_stream_profile>().get_intrinsics();
                    _calib.has_infra2 = true;
                }
            }
        }
        if (_calib.has_depth && _calib.has_color)
        {
            _calib.depth_to_color = depth_sp.get_extrinsics_to(color_sp);
        }
    }
}

void RealSenseCapture::start_loop(FramesetCallback on_frameset, MotionCallback on_motion)
{
    if (_running.exchange(true))
        return;

    _motion_cb = std::move(on_motion);

    if (_imu_separate)
    {
        rs2::config imu_cfg;
        if (!_cfg.serial_no.empty())
            imu_cfg.enable_device(_cfg.serial_no);
        if (_cfg.enable_gyro)
            imu_cfg.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F, _cfg.gyro_fps);
        if (_cfg.enable_accel)
            imu_cfg.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F, _cfg.accel_fps);

        _imu_pipe.start(imu_cfg, [this](const rs2::frame& f) {
            if (!_running.load(std::memory_order_acquire) || !_motion_cb)
                return;
            if (auto fs = f.as<rs2::frameset>())
            {
                for (auto sub : fs)
                {
                    if (auto mf = sub.as<rs2::motion_frame>())
                        _motion_cb(mf);
                }
            }
            else if (auto mf = f.as<rs2::motion_frame>())
            {
                _motion_cb(mf);
            }
        });
        _imu_started = true;
    }

    if (!_video_started)
        return;

    _worker = std::thread([this, on_frameset = std::move(on_frameset)]() {
        while (_running.load(std::memory_order_acquire))
        {
            rs2::frameset fs;
            if (!_pipe.try_wait_for_frames(&fs, 500))
            {
                if (!_cfg.rosbag_filename.empty() && !_cfg.rosbag_loop)
                {
                    auto pb = _profile.get_device().as<rs2::playback>();
                    if (pb && pb.current_status() == RS2_PLAYBACK_STATUS_STOPPED)
                    {
                        break;
                    }
                }
                continue;
            }

            if (!_imu_separate && _motion_cb)
            {
                for (auto f : fs)
                {
                    if (auto mf = f.as<rs2::motion_frame>())
                    {
                        _motion_cb(mf);
                    }
                }
            }
            if (on_frameset)
            {
                on_frameset(fs);
            }
        }
    });
}

void RealSenseCapture::stop()
{
    if (!_running.exchange(false))
        return;
    if (_worker.joinable())
        _worker.join();
    if (_imu_started)
    {
        try { _imu_pipe.stop(); } catch (...) {}
        _imu_started = false;
    }
    if (_video_started)
    {
        try { _pipe.stop(); } catch (...) {}
        _video_started = false;
    }
}

}  // namespace realsense_qcom_bolt
