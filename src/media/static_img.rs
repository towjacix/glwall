use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::process::Command;
use std::time::Duration;

use image::AnimationDecoder;
use image::codecs::gif::GifDecoder;
use image::codecs::webp::WebPDecoder;

use super::{FrameMailbox, FramePayload, WorkerControl};

pub fn is_animated_webp(path: &Path) -> bool {
    let Ok(file) = File::open(path) else {
        return false;
    };
    let Ok(decoder) = WebPDecoder::new(BufReader::new(file)) else {
        return false;
    };
    decoder.has_animation()
}

pub fn run_static(path: &Path, control: &WorkerControl, mailbox: &FrameMailbox) {
    if control.should_stop() {
        return;
    }
    match image::open(path) {
        Ok(img) => {
            if control.should_stop() {
                return;
            }
            let rgba = img.to_rgba8();
            let (width, height) = rgba.dimensions();
            let pixels = rgba.into_raw();
            mailbox.publish(FramePayload::Rgba {
                width,
                height,
                pixels,
            });
        }
        Err(err) => {
            mailbox.send_error(format!("failed to decode image {}: {err}", path.display()));
        }
    }
}

pub fn run_ffmpeg_static(path: &Path, control: &WorkerControl, mailbox: &FrameMailbox) {
    if control.should_stop() {
        return;
    }
    // Probe dimensions first so raw RGBA output needs no header parsing.
    let probe = match super::video::probe_media(path) {
        Ok(info) => info,
        Err(err) => {
            mailbox.send_error(format!(
                "ffprobe failed for static image {}: {err}",
                path.display()
            ));
            return;
        }
    };

    if control.should_stop() {
        return;
    }

    let output = Command::new("ffmpeg")
        .args(["-nostdin", "-hide_banner", "-loglevel", "error", "-i"])
        .arg(path)
        .args([
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "pipe:1",
        ])
        .output();

    match output {
        Ok(out) if out.status.success() => {
            let expected = (probe.width as usize) * (probe.height as usize) * 4;
            if out.stdout.len() < expected {
                mailbox.send_error(format!(
                    "ffmpeg returned truncated RGBA frame ({} < {expected}) for {}",
                    out.stdout.len(),
                    path.display()
                ));
                return;
            }
            let mut pixels = out.stdout;
            pixels.truncate(expected);
            mailbox.publish(FramePayload::Rgba {
                width: probe.width,
                height: probe.height,
                pixels,
            });
        }
        Ok(out) => {
            let err_msg = String::from_utf8_lossy(&out.stderr);
            mailbox.send_error(format!(
                "ffmpeg decode failed for {}: {}",
                path.display(),
                err_msg.trim()
            ));
        }
        Err(err) => {
            mailbox.send_error(format!(
                "failed to spawn ffmpeg for {}: {err}",
                path.display()
            ));
        }
    }
}

pub fn run_animated_gif(
    path: &Path,
    fps_cap: Option<f32>,
    control: &WorkerControl,
    mailbox: &FrameMailbox,
) {
    stream_animated_loop(path, fps_cap, control, mailbox, |file| {
        let decoder = GifDecoder::new(BufReader::new(file))?;
        Ok(decoder.into_frames())
    });
}

pub fn run_animated_webp(
    path: &Path,
    fps_cap: Option<f32>,
    control: &WorkerControl,
    mailbox: &FrameMailbox,
) {
    stream_animated_loop(path, fps_cap, control, mailbox, |file| {
        let decoder = WebPDecoder::new(BufReader::new(file))?;
        Ok(decoder.into_frames())
    });
}

fn stream_animated_loop<'a, F>(
    path: &Path,
    fps_cap: Option<f32>,
    control: &WorkerControl,
    mailbox: &FrameMailbox,
    open_frames: F,
) where
    F: Fn(File) -> image::ImageResult<image::Frames<'a>>,
{
    let min_interval = fps_cap
        .filter(|fps| *fps > 0.5)
        .map(|fps| Duration::from_secs_f64(1.0 / f64::from(fps)));

    while !control.should_stop() {
        let file = match File::open(path) {
            Ok(f) => f,
            Err(err) => {
                mailbox.send_error(format!("failed to open {}: {err}", path.display()));
                return;
            }
        };
        let frames = match open_frames(file) {
            Ok(f) => f,
            Err(err) => {
                mailbox.send_error(format!("failed to decode {}: {err}", path.display()));
                return;
            }
        };

        let mut emitted_any = false;
        for frame_res in frames {
            if !control.wait_if_paused() {
                return;
            }
            let frame = match frame_res {
                Ok(f) => f,
                Err(err) => {
                    if !emitted_any {
                        mailbox.send_error(format!(
                            "animation decode error on {}: {err}",
                            path.display()
                        ));
                    }
                    break;
                }
            };
            let mut delay = Duration::from(frame.delay());
            if delay < Duration::from_millis(16) {
                // Standard GIF players treat missing/0ms frame delays as ~100ms.
                delay = Duration::from_millis(100);
            }
            if let Some(cap) = min_interval {
                delay = delay.max(cap);
            }

            let rgba = frame.into_buffer();
            let (width, height) = rgba.dimensions();
            let raw = rgba.into_raw();
            let mut buf = mailbox.take_spare_rgba(raw.len());
            buf.copy_from_slice(&raw);
            mailbox.publish(FramePayload::Rgba {
                width,
                height,
                pixels: buf,
            });
            emitted_any = true;

            if !control.interruptible_sleep(delay) {
                return;
            }
        }

        if !emitted_any {
            mailbox.send_error(format!(
                "animated image {} contained 0 frames",
                path.display()
            ));
            return;
        }
    }
}
