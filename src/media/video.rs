use std::io::{BufReader, Read};
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use serde::de::{DeserializeSeed, IgnoredAny, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};

use crate::config::{ColorSpace, HwAccel};

use super::{FrameMailbox, FramePayload, WorkerControl};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MediaProbe {
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub colorspace: ColorSpace,
    coded_width: u32,
    coded_height: u32,
    display_matrix: Option<[i32; 9]>,
    multiple_video_streams: bool,
    has_display_transform: bool,
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
    #[serde(default)]
    side_data_list: Vec<FfprobeSideData>,
    #[serde(default)]
    tags: FfprobeTags,
}

#[derive(Debug, Deserialize)]
struct FfprobeSideData {
    side_data_type: Option<String>,
    displaymatrix: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct FfprobeFrame {
    width: Option<u32>,
    height: Option<u32>,
    #[serde(default)]
    side_data_list: Vec<FfprobeSideData>,
}

#[derive(Debug, Default, Deserialize)]
struct FfprobeTags {
    #[serde(alias = "ROTATE")]
    rotate: Option<String>,
}

pub fn probe_media(path: &Path) -> Result<MediaProbe> {
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v",
            "-show_entries",
            "stream=width,height,r_frame_rate,avg_frame_rate,color_space,color_range:stream_side_data=side_data_type,displaymatrix,rotation:stream_tags=rotate",
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

    let probe = parse_ffprobe_json(&out.stdout)
        .with_context(|| format!("invalid ffprobe output for {}", path.display()))?;
    let frame = probe_first_frame(path)?;
    validate_first_frame(&probe, &frame)
        .with_context(|| format!("unsupported first frame in {}", path.display()))?;
    Ok(probe)
}
// Read only the first decoded frame. Frame-level orientation can override the
// container matrix, and its dimensions need not match the stream header.
fn probe_first_frame(path: &Path) -> Result<FfprobeFrame> {
    let mut child = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_frames",
            "-show_entries",
            "frame=width,height:frame_side_data=side_data_type,displaymatrix",
            "-of",
            "json",
        ])
        .arg(path)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .with_context(|| format!("failed to invoke ffprobe for {}", path.display()))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow!("ffprobe stdout handle missing"))?;
    let mut frame = None;
    let result = {
        let mut deserializer = serde_json::Deserializer::from_reader(BufReader::new(stdout));
        FirstFrame(&mut frame).deserialize(&mut deserializer)
    };
    // The JSON reader intentionally stops as soon as the first frame is parsed.
    // Closing its pipe and terminating ffprobe avoids decoding the entire video.
    let _ = child.kill();
    let status = child.wait().context("failed to reap ffprobe")?;
    if let Some(frame) = frame {
        return Ok(frame);
    }
    result.context("failed to read first decoded frame from ffprobe")?;
    bail!("ffprobe produced no decoded video frames (status: {status})")
}

struct FirstFrame<'a>(&'a mut Option<FfprobeFrame>);

impl<'de> DeserializeSeed<'de> for FirstFrame<'_> {
    type Value = ();

    fn deserialize<D: Deserializer<'de>>(
        self,
        deserializer: D,
    ) -> std::result::Result<(), D::Error> {
        deserializer.deserialize_map(self)
    }
}

impl<'de> Visitor<'de> for FirstFrame<'_> {
    type Value = ();

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("ffprobe output with a frames array")
    }

    fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> std::result::Result<(), M::Error> {
        while let Some(key) = map.next_key::<String>()? {
            if key == "frames" {
                map.next_value_seed(FirstFrameSequence(self.0))?;
                return Ok(());
            }
            map.next_value::<IgnoredAny>()?;
        }
        Ok(())
    }
}

struct FirstFrameSequence<'a>(&'a mut Option<FfprobeFrame>);

impl<'de> DeserializeSeed<'de> for FirstFrameSequence<'_> {
    type Value = ();

    fn deserialize<D: Deserializer<'de>>(
        self,
        deserializer: D,
    ) -> std::result::Result<(), D::Error> {
        deserializer.deserialize_seq(self)
    }
}

impl<'de> Visitor<'de> for FirstFrameSequence<'_> {
    type Value = ();

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("an array of decoded frames")
    }

    fn visit_seq<S: SeqAccess<'de>>(self, mut seq: S) -> std::result::Result<(), S::Error> {
        if let Some(frame) = seq.next_element()? {
            *self.0 = Some(frame);
            // Abort parsing immediately instead of reading the remaining frames.
            return Err(serde::de::Error::custom("first frame collected"));
        }
        Ok(())
    }
}

fn validate_first_frame(probe: &MediaProbe, frame: &FfprobeFrame) -> Result<()> {
    if (frame.width, frame.height) != (Some(probe.coded_width), Some(probe.coded_height)) {
        bail!("first decoded frame dimensions differ from the stream header");
    }
    let mut found_matrix = false;
    for side in &frame.side_data_list {
        if side.side_data_type.as_deref() != Some("Display Matrix") && side.displaymatrix.is_none()
        {
            continue;
        }
        if found_matrix {
            bail!("multiple first-frame display matrices are not supported");
        }
        found_matrix = true;
        let matrix = parse_display_matrix(
            side.displaymatrix
                .as_ref()
                .ok_or_else(|| anyhow!("first-frame display matrix is missing"))?,
        )?;
        let (transformed, _) =
            classify_display_matrix(matrix, probe.coded_width, probe.coded_height)?;
        match probe.display_matrix {
            Some(stream_matrix) => {
                // Translation is validated above; FFmpeg normalizes frame origins.
                if [0, 1, 2, 3, 4, 5, 8]
                    .iter()
                    .any(|&i| matrix[i] != stream_matrix[i])
                {
                    bail!("first-frame display matrix differs from the stream matrix");
                }
            }
            None if transformed => {
                bail!("frame-level display transform without a stream matrix is not supported")
            }
            None => {}
        }
    }
    Ok(())
}

pub fn parse_ffprobe_json(json_bytes: &[u8]) -> Result<MediaProbe> {
    let parsed: FfprobeOutput = serde_json::from_slice(json_bytes)?;
    let multiple_video_streams = parsed.streams.len() > 1;
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

    let mut has_display_transform = false;
    let mut swap_dimensions = false;
    let mut found_matrix = false;
    let mut display_matrix = None;
    for side_data in &stream.side_data_list {
        if side_data.side_data_type.as_deref() == Some("Display Matrix")
            || side_data.displaymatrix.is_some()
        {
            if found_matrix {
                bail!("multiple display matrices are not supported");
            }
            found_matrix = true;
            let matrix = parse_display_matrix(
                side_data
                    .displaymatrix
                    .as_ref()
                    .ok_or_else(|| anyhow!("display matrix missing from ffprobe output"))?,
            )?;
            display_matrix = Some(matrix);
            (has_display_transform, swap_dimensions) =
                classify_display_matrix(matrix, width, height)?;
        }
    }
    if let Some(rotation) = stream.tags.rotate.as_deref()
        && rotation.parse::<f64>() != Ok(0.0)
        && !found_matrix
    {
        bail!("rotation tag without a display matrix is not supported");
    }

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
    let (coded_width, coded_height) = (width, height);
    let (width, height) = if swap_dimensions {
        (height, width)
    } else {
        (width, height)
    };

    Ok(MediaProbe {
        width,
        height,
        fps,
        colorspace,
        coded_width,
        coded_height,
        display_matrix,
        multiple_video_streams,
        has_display_transform,
    })
}

fn parse_display_matrix(value: &serde_json::Value) -> Result<[i32; 9]> {
    let mut matrix = [0i32; 9];
    let mut len = 0;
    match value {
        serde_json::Value::String(text) => {
            for line in text.lines().filter(|line| !line.trim().is_empty()) {
                let (_, row) = line
                    .split_once(':')
                    .ok_or_else(|| anyhow!("invalid display matrix row"))?;
                for number in row.split_whitespace() {
                    if len == matrix.len() {
                        bail!("invalid display matrix length");
                    }
                    matrix[len] = number.parse().context("invalid display matrix entry")?;
                    len += 1;
                }
            }
        }
        serde_json::Value::Array(values) => {
            for value in values {
                if len == matrix.len() {
                    bail!("invalid display matrix length");
                }
                matrix[len] = i32::try_from(
                    value
                        .as_i64()
                        .ok_or_else(|| anyhow!("invalid display matrix entry"))?,
                )
                .context("invalid display matrix entry")?;
                len += 1;
            }
        }
        _ => bail!("invalid display matrix"),
    }
    if len != matrix.len() {
        bail!("invalid display matrix length");
    }
    Ok(matrix)
}

fn classify_display_matrix(matrix: [i32; 9], width: u32, height: u32) -> Result<(bool, bool)> {
    const ONE: i64 = 1 << 16;
    const PERSPECTIVE_ONE: i64 = 1 << 30;
    let [a, b, u, c, d, v, x, y, w] = matrix.map(i64::from);
    let straight = b == 0 && c == 0 && a.abs() == ONE && d.abs() == ONE;
    let swapped = a == 0 && d == 0 && b.abs() == ONE && c.abs() == ONE;
    if (!straight && !swapped) || u != 0 || v != 0 || w != PERSPECTIVE_ONE {
        bail!("unsupported display matrix (only right-angle rotations and mirrors are supported)");
    }
    // FFmpeg normalizes frame origins when applying orientation. Accept either
    // an untranslated matrix or the usual translation that places it at (0, 0).
    let expected_x = -(a * i64::from(width)).min(0) - (c * i64::from(height)).min(0);
    let expected_y = -(b * i64::from(width)).min(0) - (d * i64::from(height)).min(0);
    if (x != 0 && x != expected_x) || (y != 0 && y != expected_y) {
        bail!("unsupported display matrix translation");
    }
    Ok((a != ONE || b != 0 || c != 0 || d != ONE, swapped))
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
    if probe.has_display_transform {
        bail!("CUDA interop cannot apply video display transforms");
    }
    if probe.multiple_video_streams {
        bail!("CUDA interop cannot select the probed video stream");
    }
    // The GPU bridge currently supports exactly two even-sized NV12 planes.
    if probe.width < 2
        || probe.height < 2
        || !probe.width.is_multiple_of(2)
        || !probe.height.is_multiple_of(2)
    {
        bail!("CUDA interop does not support odd video dimensions");
    }
    let mut decoder = super::gst_cuda::CudaDecoder::open(path, probe.colorspace)?;
    let mut timeout = CudaFrameTimeout::new(Instant::now(), control.pause_state());
    let mut next_deadline = Instant::now();
    let mut count = 0u64;
    while !control.should_stop() {
        if !control.wait_if_paused() {
            break;
        }
        if control.cuda_failed() {
            bail!("CUDA texture upload failed");
        }
        let Some(frame) = decoder.pull(Duration::from_millis(100))? else {
            if timeout.expired(Instant::now(), control.pause_state()) {
                bail!("CUDA decoder did not produce a frame");
            }
            continue;
        };
        if frame.width != probe.width || frame.height != probe.height {
            bail!("CUDA decoder changed video dimensions");
        }
        mailbox.publish(FramePayload::Cuda(frame));
        count += 1;
        let now = Instant::now();
        timeout.frame_received(now, control.pause_state());
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

struct CudaFrameTimeout {
    last_frame: Instant,
    pause_state: u64,
}

impl CudaFrameTimeout {
    fn new(now: Instant, pause_state: u64) -> Self {
        Self {
            last_frame: now,
            pause_state,
        }
    }

    fn frame_received(&mut self, now: Instant, pause_state: u64) {
        self.last_frame = now;
        self.pause_state = pause_state;
    }

    fn expired(&mut self, now: Instant, pause_state: u64) -> bool {
        if pause_state & 1 != 0 {
            return false;
        }
        if self.pause_state != pause_state {
            self.frame_received(now, pause_state);
        }
        now.saturating_duration_since(self.last_frame) > Duration::from_secs(8)
    }
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
        .args(["-map", "0:v:0", "-an", "-sn", "-dn"]);
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

    #[test]
    fn display_matrices_select_ffmpeg_and_report_display_dimensions() {
        // FFprobe's displaymatrix is a fixed-point 3x3 matrix, including for mirrors.
        for (matrix, dimensions, transformed) in [
            ("65536 0 0 0 65536 0 0 0 1073741824", (1920, 1080), false),
            ("-65536 0 0 0 -65536 0 0 0 1073741824", (1920, 1080), true),
            ("0 -65536 0 65536 0 0 0 0 1073741824", (1080, 1920), true),
            ("0 65536 0 -65536 0 0 0 0 1073741824", (1080, 1920), true),
            ("-65536 0 0 0 65536 0 0 0 0 1073741824", (1920, 1080), true),
            ("0 65536 0 65536 0 0 0 0 1073741824", (1080, 1920), true),
        ] {
            let values: Vec<i32> = matrix
                .split_whitespace()
                .map(|s| s.parse().unwrap())
                .collect();
            let json = serde_json::json!({"streams": [{
                "width": 1920, "height": 1080,
                "side_data_list": [{"side_data_type": "Display Matrix",
                    "displaymatrix": format!("\n00000000: {} {} {}\n00000001: {} {} {}\n00000002: {} {} {}\n",
                        values[0], values[1], values[2], values[3], values[4], values[5],
                        values[6], values[7], values[8])}]
            }]});
            let probe = parse_ffprobe_json(json.to_string().as_bytes()).unwrap();
            assert_eq!((probe.width, probe.height), dimensions, "{matrix}");
            assert_eq!(probe.has_display_transform, transformed, "{matrix}");
            if transformed {
                let (sender, _receiver) = calloop::channel::channel();
                let mailbox = FrameMailbox::new(sender, String::new(), 0);
                let control = WorkerControl::new(false);
                let error = stream_cuda_session(
                    Path::new("unused"),
                    probe,
                    Duration::from_secs(1),
                    &control,
                    &mailbox,
                )
                .unwrap_err();
                assert!(error.to_string().contains("display transforms"), "{matrix}");
            }
        }
    }

    #[test]
    fn rejects_display_matrices_ffmpeg_cannot_fully_apply() {
        for matrix in [
            "\n00000000: 65536 32768 0\n00000001: 0 65536 0\n00000002: 0 0 1073741824\n",
            "\n00000000: 65536 0 1\n00000001: 0 65536 0\n00000002: 0 0 1073741824\n",
            "not a matrix",
        ] {
            let json = serde_json::json!({"streams": [{
                "width": 1920, "height": 1080,
                "side_data_list": [{"side_data_type": "Display Matrix", "displaymatrix": matrix}]
            }]});
            assert!(
                parse_ffprobe_json(json.to_string().as_bytes()).is_err(),
                "{matrix}"
            );
        }
        let missing_matrix = br#"{"streams":[{"width":1920,"height":1080,
            "side_data_list":[{"side_data_type":"Display Matrix","rotation":0}]}]}"#;
        assert!(parse_ffprobe_json(missing_matrix).is_err());
    }

    #[test]
    fn accepts_numeric_display_matrix_without_rotation_field() {
        let json = br#"{"streams":[{"width":1921,"height":1081,
            "side_data_list":[{"side_data_type":"Display Matrix",
              "displaymatrix":[0,65536,0,-65536,0,0,0,0,1073741824]}]}]}"#;
        let probe = parse_ffprobe_json(json).unwrap();
        assert_eq!((probe.width, probe.height), (1081, 1921));
        assert!(probe.has_display_transform);
    }

    #[test]
    fn first_frame_probe_rejects_frame_only_or_conflicting_orientation() {
        let probe = parse_ffprobe_json(br#"{"streams":[{"width":32,"height":20}]}"#).unwrap();
        let frame: FfprobeFrame = serde_json::from_str(
            r#"{"width":32,"height":20,
            "side_data_list":[{"side_data_type":"Display Matrix",
                "displaymatrix":[-65536,0,0,0,-65536,0,0,0,1073741824]}]}"#,
        )
        .unwrap();
        assert!(validate_first_frame(&probe, &frame).is_err());

        let probe = parse_ffprobe_json(
            br#"{"streams":[{"width":32,"height":20,
            "side_data_list":[{"side_data_type":"Display Matrix",
                "displaymatrix":[-65536,0,0,0,-65536,0,0,0,1073741824]}]}]}"#,
        )
        .unwrap();
        assert!(validate_first_frame(&probe, &frame).is_ok());
        let identity: FfprobeFrame = serde_json::from_str(
            r#"{"width":32,"height":20,
            "side_data_list":[{"side_data_type":"Display Matrix",
                "displaymatrix":[65536,0,0,0,65536,0,0,0,1073741824]}]}"#,
        )
        .unwrap();
        assert!(validate_first_frame(&probe, &identity).is_err());
        let unknown: FfprobeFrame = serde_json::from_str(
            r#"{"width":32,"height":20,
            "side_data_list":[{"side_data_type":"Display Matrix"}]}"#,
        )
        .unwrap();
        assert!(validate_first_frame(&probe, &unknown).is_err());
        let dimensions: FfprobeFrame = serde_json::from_str(r#"{"width":20,"height":32}"#).unwrap();
        assert!(validate_first_frame(&probe, &dimensions).is_err());
    }

    #[test]
    fn first_frame_json_reader_stops_before_later_frames() {
        let mut frame = None;
        let json = br#"{"frames":[{"width":32,"height":20},{"invalid": "#;
        let mut deserializer = serde_json::Deserializer::from_slice(json);
        assert!(
            FirstFrame(&mut frame)
                .deserialize(&mut deserializer)
                .is_err()
        );
        let frame = frame.unwrap();
        assert_eq!((frame.width, frame.height), (Some(32), Some(20)));
    }

    #[test]
    fn multiple_video_streams_force_ffmpeg_selection() {
        let probe = parse_ffprobe_json(
            br#"{"streams":[
            {"width":32,"height":20},{"width":1920,"height":1080}]}"#,
        )
        .unwrap();
        assert_eq!((probe.width, probe.height), (32, 20));
        let (sender, _receiver) = calloop::channel::channel();
        let mailbox = FrameMailbox::new(sender, String::new(), 0);
        let control = WorkerControl::new(false);
        let error = stream_cuda_session(
            Path::new("unused"),
            probe,
            Duration::from_secs(1),
            &control,
            &mailbox,
        )
        .unwrap_err();
        assert!(error.to_string().contains("select the probed video stream"));
    }

    struct FixtureDir(std::path::PathBuf);

    impl FixtureDir {
        fn new() -> Self {
            static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let path =
                std::env::temp_dir().join(format!("glwall-video-test-{}-{id}", std::process::id()));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }

        fn source(&self, width: u32, height: u32, codec: &str) -> std::path::PathBuf {
            let image = self.0.join("source.pgm");
            let mut bytes = format!("P5\n{width} {height}\n255\n").into_bytes();
            for y in 0..height {
                for x in 0..width {
                    let level = match (x >= width / 2, y >= height / 2) {
                        (false, false) => 24,
                        (true, false) => 80,
                        (false, true) => 150,
                        (true, true) => 225,
                    };
                    bytes.push(level);
                }
            }
            std::fs::write(&image, bytes).unwrap();
            let video = self.0.join("source.mp4");
            let output = Command::new("ffmpeg")
                .args(["-y", "-v", "error", "-framerate", "1", "-i"])
                .arg(&image)
                .args(["-frames:v", "1", "-c:v", codec, "-pix_fmt", "yuv420p"])
                .arg(&video)
                .output()
                .expect("ffmpeg is required for video regression tests");
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            video
        }
    }

    impl Drop for FixtureDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn next_frame(mailbox: &FrameMailbox, timeout: Duration) -> Option<FramePayload> {
        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            if let Some(frame) = mailbox.pop_latest() {
                return Some(frame);
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        None
    }

    fn decoded_frame(path: &Path) -> FramePayload {
        let (sender, _receiver) = calloop::channel::channel();
        let mailbox = FrameMailbox::new(sender, String::new(), 0);
        let control = WorkerControl::new(false);
        let worker_mailbox = std::sync::Arc::clone(&mailbox);
        let worker_control = std::sync::Arc::clone(&control);
        let path = path.to_owned();
        let worker = std::thread::spawn(move || {
            run_video(&path, HwAccel::None, None, &worker_control, &worker_mailbox);
        });
        let frame = next_frame(&mailbox, Duration::from_secs(10));
        control.request_stop();
        worker.join().unwrap();
        frame.expect("video worker did not publish a frame")
    }

    #[test]
    fn ffmpeg_presents_rotated_and_mirrored_odd_sized_frames() {
        let fixture = FixtureDir::new();
        let source = fixture.source(31, 21, "mpeg4");
        for rotation in ["90", "180", "270", "mirror"] {
            let output_path = fixture.0.join(format!("{rotation}.mp4"));
            let mut command = Command::new("ffmpeg");
            command.args(["-y", "-v", "error"]);
            if rotation == "mirror" {
                command.arg("-display_hflip");
            }
            command.arg("-i").arg(&source);
            if rotation != "mirror" {
                command.args(["-metadata:s:v:0", &format!("rotate={rotation}")]);
            }
            let output = command
                .args(["-c", "copy"])
                .arg(&output_path)
                .output()
                .unwrap();
            if rotation == "mirror"
                && !output.status.success()
                && String::from_utf8_lossy(&output.stderr).contains("Unrecognized option")
            {
                continue;
            }
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            let probe = probe_media(&output_path).unwrap();
            let matrix = probe
                .display_matrix
                .expect("fixture must contain a display matrix");
            assert!(probe.has_display_transform);
            let FramePayload::Nv12 {
                width,
                height,
                y_plane,
                ..
            } = decoded_frame(&output_path)
            else {
                panic!("expected FFmpeg NV12 frame");
            };
            let expected_size = if rotation == "90" || rotation == "270" {
                (20, 30)
            } else {
                (30, 20)
            };
            assert_eq!((width, height), expected_size, "{rotation}");
            for (sx, sy, level) in [(0, 0, 24), (1, 0, 80), (0, 1, 150), (1, 1, 225)] {
                let x = 2 * sx - 1;
                let y = 2 * sy - 1;
                let dx = usize::from(matrix[0] * x + matrix[3] * y > 0);
                let dy = usize::from(matrix[1] * x + matrix[4] * y > 0);
                let px = (width as usize * (1 + 2 * dx)) / 4;
                let py = (height as usize * (1 + 2 * dy)) / 4;
                let actual = i32::from(y_plane[py * width as usize + px]);
                let expected = 16 + level * 219 / 255;
                assert!(
                    (actual - expected).abs() <= 18,
                    "{rotation}: {actual} vs {expected}"
                );
            }
        }
    }

    #[test]
    fn ffmpeg_decodes_the_probed_first_video_stream() {
        let fixture = FixtureDir::new();
        let source = fixture.source(31, 21, "mpeg4");
        let video = fixture.0.join("two-streams.mp4");
        let output = Command::new("ffmpeg")
            .args(["-y", "-v", "error", "-i"])
            .arg(&source)
            .args([
                "-f",
                "lavfi",
                "-i",
                "color=white:s=64x64:r=1",
                "-map",
                "0:v:0",
                "-map",
                "1:v:0",
                "-c:v:0",
                "copy",
                "-c:v:1",
                "mpeg4",
                "-frames:v:1",
                "1",
                "-disposition:v:0",
                "0",
                "-disposition:v:1",
                "default",
            ])
            .arg(&video)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let probe = probe_media(&video).unwrap();
        assert!(probe.multiple_video_streams);
        let FramePayload::Nv12 {
            width,
            height,
            y_plane,
            ..
        } = decoded_frame(&video)
        else {
            panic!("expected FFmpeg NV12 frame");
        };
        assert_eq!((width, height), (30, 20));
        assert!(y_plane[width as usize * 5 + 5] < 80);
    }

    #[test]
    #[ignore = "requires NVIDIA CUDA/NVDEC, GStreamer CUDA, and FFmpeg with libx264"]
    fn cuda_one_frame_pause_through_eos_does_not_fallback() {
        let fixture = FixtureDir::new();
        let source = fixture.source(32, 24, "libx264");
        let (sender, _receiver) = calloop::channel::channel();
        let mailbox = FrameMailbox::new(sender, String::new(), 0);
        let control = WorkerControl::new(false);
        let worker_mailbox = std::sync::Arc::clone(&mailbox);
        let worker_control = std::sync::Arc::clone(&control);
        let worker = std::thread::spawn(move || {
            run_video(
                &source,
                HwAccel::Cuda,
                None,
                &worker_control,
                &worker_mailbox,
            );
        });
        let first = next_frame(&mailbox, Duration::from_secs(15));
        control.set_paused(true);
        std::thread::sleep(Duration::from_secs(9));
        let _ = mailbox.pop_latest();
        control.set_paused(false);
        let after_resume = next_frame(&mailbox, Duration::from_secs(6));
        control.set_paused(true);
        control.request_stop();
        worker.join().unwrap();
        assert!(
            matches!(first, Some(FramePayload::Cuda(_))),
            "initial frame was not CUDA"
        );
        assert!(
            matches!(after_resume, Some(FramePayload::Cuda(_))),
            "CUDA did not loop after resume"
        );
    }

    #[test]
    fn pause_entirely_inside_sleep_is_seen_at_next_eos_poll() {
        let control = WorkerControl::new(false);
        let start = Instant::now();
        let mut timeout = CudaFrameTimeout::new(start, control.pause_state());
        control.set_paused(true);
        control.set_paused(false);
        // Neither pause transition was observed by the decoder while it slept.
        let resume = start + Duration::from_secs(9);
        assert!(!timeout.expired(resume, control.pause_state()));
        assert!(!timeout.expired(resume + Duration::from_secs(8), control.pause_state()));
        assert!(timeout.expired(resume + Duration::from_millis(8001), control.pause_state()));
    }

    #[test]
    fn pause_during_frame_sleep_resets_cuda_timeout_after_resume() {
        let control = WorkerControl::new(false);
        let start = Instant::now();
        let mut timeout = CudaFrameTimeout::new(start, control.pause_state());
        control.set_paused(true);
        assert!(!timeout.expired(start + Duration::from_secs(9), control.pause_state()));
        control.set_paused(false);
        let resume = start + Duration::from_secs(9);
        assert!(!timeout.expired(resume, control.pause_state()));
        assert!(!timeout.expired(resume + Duration::from_secs(8), control.pause_state()));
        assert!(timeout.expired(resume + Duration::from_millis(8001), control.pause_state()));
    }
}
