use std::io::Read;
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use serde::Deserialize;

use crate::config::{ColorSpace, HwAccel};

use super::{FrameMailbox, FramePayload, WorkerControl};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MediaProbe {
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub colorspace: ColorSpace,
}

#[derive(Debug, Deserialize)]
struct FfprobeOutput {
    #[serde(default)]
    streams: Vec<FfprobeStream>,
}

#[derive(Debug, Deserialize)]
struct FfprobeStream {
    width: Option<u32>,
    height: Option<u32>,
    r_frame_rate: Option<String>,
    avg_frame_rate: Option<String>,
    color_space: Option<String>,
    color_range: Option<String>,
}

pub fn probe_media(path: &Path) -> Result<MediaProbe> {
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,avg_frame_rate,color_space,color_range",
            "-of",
            "json",
        ])
        .arg(path)
        .output()
        .with_context(|| format!("failed to invoke ffprobe for {}", path.display()))?;

    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        bail!("ffprobe failed for {}: {}", path.display(), stderr.trim());
    }

    parse_ffprobe_json(&out.stdout)
        .with_context(|| format!("invalid ffprobe output for {}", path.display()))
}

pub fn parse_ffprobe_json(json_bytes: &[u8]) -> Result<MediaProbe> {
    let parsed: FfprobeOutput = serde_json::from_slice(json_bytes)?;
    let stream = parsed
        .streams
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("no video stream found"))?;

    let width = stream
        .width
        .filter(|w| *w > 0)
        .ok_or_else(|| anyhow!("missing width"))?;
    let height = stream
        .height
        .filter(|h| *h > 0)
        .ok_or_else(|| anyhow!("missing height"))?;

    let fps = stream
        .avg_frame_rate
        .as_deref()
        .and_then(parse_rate_str)
        .or_else(|| stream.r_frame_rate.as_deref().and_then(parse_rate_str))
        .unwrap_or(30.0)
        .clamp(1.0, 240.0);

    let colorspace = resolve_colorspace(
        height,
        stream.color_space.as_deref(),
        stream.color_range.as_deref(),
    );

    Ok(MediaProbe {
        width,
        height,
        fps,
        colorspace,
    })
}

fn parse_rate_str(raw: &str) -> Option<f64> {
    let trimmed = raw.trim();
    if trimmed.is_empty() || trimmed == "0/0" {
        return None;
    }
    if let Some((num_s, den_s)) = trimmed.split_once('/') {
        let num: f64 = num_s.parse().ok()?;
        let den: f64 = den_s.parse().ok()?;
        if den > 0.0 && num > 0.0 {
            return Some(num / den);
        }
        return None;
    }
    trimmed.parse::<f64>().ok().filter(|v| *v > 0.0)
}

fn resolve_colorspace(height: u32, space: Option<&str>, range: Option<&str>) -> ColorSpace {
    let is_full = matches!(
        range.map(str::to_ascii_lowercase).as_deref(),
        Some("pc" | "jpeg" | "full")
    );
    let is_bt601 = match space.map(str::to_ascii_lowercase).as_deref() {
        Some("smpte170m" | "bt470bg" | "bt470m" | "fcc") => true,
        Some("bt709") => false,
        _ => height < 720,
    };

    match (is_bt601, is_full) {
        (false, false) => ColorSpace::Bt709Tv,
        (false, true) => ColorSpace::Bt709Pc,
        (true, false) => ColorSpace::Bt601Tv,
        (true, true) => ColorSpace::Bt601Pc,
    }
}

pub fn run_video(
    path: &Path,
    requested_hwaccel: HwAccel,
    fps_cap: Option<f32>,
    control: &WorkerControl,
    mailbox: &FrameMailbox,
) {
    let probe = match probe_media(path) {
        Ok(p) => p,
        Err(err) => {
            mailbox.send_error(err.to_string());
            return;
        }
    };

    // NV12 chroma subsampling requires even dimensions.
    let out_w = (probe.width & !1).max(2);
    let out_h = (probe.height & !1).max(2);
    let needs_even_crop = out_w != probe.width || out_h != probe.height;

    let effective_fps = match fps_cap.filter(|f| *f > 0.5) {
        Some(cap) => probe.fps.min(f64::from(cap)),
        None => probe.fps,
    };
    let frame_interval = Duration::from_secs_f64(1.0 / effective_fps);

    if matches!(requested_hwaccel, HwAccel::Auto | HwAccel::Cuda) {
        match stream_cuda_session(path, probe, frame_interval, control, mailbox) {
            Ok(()) => return,
            Err(err) if !control.should_stop() => {
                tracing::warn!(
                    "CUDA video interop unavailable for {} ({err:#}); falling back to FFmpeg",
                    path.display()
                );
            }
            Err(_) => return,
        }
    }

    let mut hw = requested_hwaccel;
    loop {
        if control.should_stop() {
            return;
        }

        let delivered_frames = stream_ffmpeg_session(
            path,
            out_w,
            out_h,
            needs_even_crop,
            probe.colorspace,
            hw,
            frame_interval,
            control,
            mailbox,
        );

        match delivered_frames {
            Ok(0) if hw != HwAccel::None && !control.should_stop() => {
                tracing::warn!(
                    "ffmpeg hwaccel={hw:?} delivered 0 frames for {}; retrying in software mode",
                    path.display()
                );
                hw = HwAccel::None;
                continue;
            }
            Ok(0) if !control.should_stop() => {
                mailbox.send_error(format!("ffmpeg decoded 0 frames from {}", path.display()));
                return;
            }
            Ok(_) => {
                if control.should_stop() {
                    return;
                }
                // Unexpected EOF (stream ended despite -stream_loop -1); pause briefly before retry.
                if !control.interruptible_sleep(Duration::from_millis(250)) {
                    return;
                }
            }
            Err(err) => {
                if hw != HwAccel::None && !control.should_stop() {
                    tracing::warn!(
                        "ffmpeg hwaccel={hw:?} failed ({err}); falling back to software decode"
                    );
                    hw = HwAccel::None;
                    continue;
                }
                mailbox.send_error(format!("video decode failed for {}: {err}", path.display()));
                return;
            }
        }
    }
}

fn stream_cuda_session(
    path: &Path,
    probe: MediaProbe,
    frame_interval: Duration,
    control: &WorkerControl,
    mailbox: &FrameMailbox,
) -> Result<()> {
    // The GPU bridge currently supports exactly two even-sized NV12 planes.
    if probe.width < 2
        || probe.height < 2
        || !probe.width.is_multiple_of(2)
        || !probe.height.is_multiple_of(2)
    {
        bail!("CUDA interop does not support odd video dimensions");
    }
    let mut decoder = super::gst_cuda::CudaDecoder::open(path, probe.colorspace)?;
    let mut last_frame = Instant::now();
    let mut next_deadline = Instant::now();
    let mut count = 0u64;
    while !control.should_stop() {
        if control.is_paused() {
            if !control.wait_if_paused() {
                break;
            }
            last_frame = Instant::now();
        }
        if control.cuda_failed() {
            bail!("CUDA texture upload failed");
        }
        let Some(frame) = decoder.pull(Duration::from_millis(100))? else {
            if last_frame.elapsed() > Duration::from_secs(8) {
                bail!("CUDA decoder did not produce a frame");
            }
            continue;
        };
        if frame.width != probe.width || frame.height != probe.height {
            bail!("CUDA decoder changed video dimensions");
        }
        mailbox.publish(FramePayload::Cuda(frame));
        count += 1;
        last_frame = Instant::now();
        let now = last_frame;
        if count == 1 || now.saturating_duration_since(next_deadline) > frame_interval * 3 {
            next_deadline = now + frame_interval;
        } else {
            next_deadline += frame_interval;
        }
        if let Some(wait) = next_deadline.checked_duration_since(now)
            && !control.interruptible_sleep(wait)
        {
            break;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn stream_ffmpeg_session(
    path: &Path,
    width: u32,
    height: u32,
    needs_even_crop: bool,
    colorspace: ColorSpace,
    hwaccel: HwAccel,
    frame_interval: Duration,
    control: &WorkerControl,
    mailbox: &FrameMailbox,
) -> Result<u64> {
    let mut child = spawn_ffmpeg_nv12(path, width, height, needs_even_crop, hwaccel)?;
    let pid = child.id() as i32;
    control.set_child_pid(pid);

    let mut stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow!("ffmpeg stdout handle missing"))?;
    let mut stderr = child.stderr.take();

    let y_len = (width as usize) * (height as usize);
    let uv_len = ((width / 2) as usize) * ((height / 2) as usize) * 2;
    let mut count = 0u64;
    let mut next_deadline = Instant::now();

    while !control.should_stop() {
        if !control.wait_if_paused() {
            break;
        }

        let (mut y_plane, mut uv_plane) = mailbox.take_spare_nv12(y_len, uv_len);
        if let Err(err) = stdout.read_exact(&mut y_plane) {
            if !control.should_stop() && count == 0 {
                let err_detail = read_stderr_snippet(&mut stderr);
                cleanup_child(&mut child, control);
                if !err_detail.is_empty() {
                    bail!("{err}: {err_detail}");
                }
            }
            break;
        }
        if let Err(err) = stdout.read_exact(&mut uv_plane) {
            if !control.should_stop() && count == 0 {
                let err_detail = read_stderr_snippet(&mut stderr);
                cleanup_child(&mut child, control);
                if !err_detail.is_empty() {
                    bail!("{err}: {err_detail}");
                }
            }
            break;
        }

        mailbox.publish(FramePayload::Nv12 {
            width,
            height,
            colorspace,
            y_plane,
            uv_plane,
        });
        count += 1;

        let now = Instant::now();
        if count == 1 || now.saturating_duration_since(next_deadline) > frame_interval * 3 {
            next_deadline = now + frame_interval;
        } else {
            next_deadline += frame_interval;
        }
        if let Some(wait) = next_deadline.checked_duration_since(now)
            && !control.interruptible_sleep(wait)
        {
            break;
        }
    }

    cleanup_child(&mut child, control);
    Ok(count)
}

fn spawn_ffmpeg_nv12(
    path: &Path,
    width: u32,
    height: u32,
    needs_even_crop: bool,
    hwaccel: HwAccel,
) -> Result<Child> {
    // Capture parent PID before pre_exec to close the fork->prctl race window on Linux.
    let expected_ppid = unsafe { libc::getpid() };

    let mut cmd = Command::new("ffmpeg");
    cmd.args([
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-threads",
        "2",
    ]);
    if let Some(hw) = hwaccel.as_ffmpeg_arg() {
        cmd.args(["-hwaccel", hw]);
    }
    cmd.args(["-stream_loop", "-1", "-i"])
        .arg(path)
        .args(["-an", "-sn", "-dn"]);
    if needs_even_crop {
        cmd.args(["-vf", &format!("crop={width}:{height}:0:0")]);
    }
    cmd.args(["-f", "rawvideo", "-pix_fmt", "nv12", "pipe:1"]);

    cmd.stdin(Stdio::null());
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    unsafe {
        cmd.pre_exec(move || {
            if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            if libc::getppid() != expected_ppid {
                libc::raise(libc::SIGTERM);
            }
            Ok(())
        });
    }

    cmd.spawn()
        .with_context(|| format!("failed to spawn ffmpeg for {}", path.display()))
}

fn read_stderr_snippet(stderr: &mut Option<std::process::ChildStderr>) -> String {
    let Some(mut err_pipe) = stderr.take() else {
        return String::new();
    };
    let mut buf = [0u8; 512];
    match err_pipe.read(&mut buf) {
        Ok(n) if n > 0 => String::from_utf8_lossy(&buf[..n]).trim().to_string(),
        _ => String::new(),
    }
}

fn cleanup_child(child: &mut Child, control: &WorkerControl) {
    let pid = child.id() as i32;
    control.clear_child_pid();
    unsafe {
        libc::kill(pid, libc::SIGCONT);
        libc::kill(pid, libc::SIGTERM);
    }
    for _ in 0..15 {
        if let Ok(Some(_)) = child.try_wait() {
            return;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    unsafe {
        libc::kill(pid, libc::SIGKILL);
    }
    let _ = child.wait();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_ffprobe_hd_bt709_tv() {
        let json = br#"{
            "streams": [{
                "width": 1920,
                "height": 1080,
                "r_frame_rate": "60000/1001",
                "avg_frame_rate": "60/1",
                "color_space": "bt709",
                "color_range": "tv"
            }]
        }"#;
        let probe = parse_ffprobe_json(json).unwrap();
        assert_eq!(probe.width, 1920);
        assert_eq!(probe.height, 1080);
        assert!((probe.fps - 60.0).abs() < 0.01);
        assert_eq!(probe.colorspace, ColorSpace::Bt709Tv);
    }

    #[test]
    fn parses_ffprobe_sd_bt601_full_fallback() {
        let json = br#"{
            "streams": [{
                "width": 640,
                "height": 480,
                "r_frame_rate": "24000/1001",
                "color_range": "pc"
            }]
        }"#;
        let probe = parse_ffprobe_json(json).unwrap();
        assert_eq!(probe.width, 640);
        assert_eq!(probe.height, 480);
        assert!((probe.fps - 23.976).abs() < 0.01);
        assert_eq!(probe.colorspace, ColorSpace::Bt601Pc);
    }
}
