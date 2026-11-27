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
    rs2::config rs_cfg;
    if (!_cfg.rosbag_filename.empty())
    {
        rs_cfg.enable_device_from_file(_cfg.rosbag_filename, _cfg.rosbag_loop);
    }
    else if (!_cfg.serial_no.empty())
    {
        rs_cfg.enable_device(_cfg.serial_no);
    }

    const bool use_yuyv_for_gpu = (_preset != PipelinePreset::P0_BYPASS_ONLY && _cfg.rosbag_filename.empty());
    if (_cfg.enable_color)
    {
        if (_cfg.rosbag_filename.empty())
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
        if (_cfg.rosbag_filename.empty())
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
    if (_cfg.enable_gyro)
    {
        rs_cfg.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F, _cfg.gyro_fps);
    }
    if (_cfg.enable_accel)
    {
        rs_cfg.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F, _cfg.accel_fps);
    }

    _profile = _pipe.start(rs_cfg);

    // Apply global time option and disable auto_exposure_priority for stable FPS on live camera
    auto dev = _profile.get_device();
    for (auto&& s : dev.query_sensors())
    {
        if (_cfg.rosbag_filename.empty())
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

    // Extract fixed stream intrinsics & extrinsics once at startup
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
    }
    if (_calib.has_depth && _calib.has_color)
    {
        _calib.depth_to_color = depth_sp.get_extrinsics_to(color_sp);
    }
}

void RealSenseCapture::start_loop(FramesetCallback on_frameset, MotionCallback on_motion)
{
    if (_running.exchange(true))
        return;

    _worker = std::thread([this, on_frameset = std::move(on_frameset), on_motion = std::move(on_motion)]() {
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

            if (on_motion)
            {
                for (auto f : fs)
                {
                    if (auto mf = f.as<rs2::motion_frame>())
                    {
                        on_motion(mf);
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
    try
    {
        _pipe.stop();
    }
    catch (...)
    {
    }
}

}  // namespace realsense_qcom_bolt
