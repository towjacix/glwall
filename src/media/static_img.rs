use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::process::Command;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use image::AnimationDecoder;
use image::codecs::gif::GifDecoder;
use image::codecs::webp::WebPDecoder;

use super::{FrameMailbox, FramePayload, WorkerControl};

pub fn decode_rgba_image(path: &Path, flip_y: bool) -> Result<(u32, u32, Vec<u8>)> {
    let is_ffmpeg_ext = path.extension().and_then(|s| s.to_str()).is_some_and(|e| {
        matches!(
            e.to_ascii_lowercase().as_str(),
            "avif" | "heif" | "heic" | "jxl"
        )
    });

    let (w, h, mut pixels) = if is_ffmpeg_ext {
        decode_ffmpeg_static_rgba(path)?
    } else {
        let img = image::open(path)
            .with_context(|| format!("failed to decode image {}", path.display()))?;
        let rgba = img.to_rgba8();
        let (w, h) = rgba.dimensions();
        (w, h, rgba.into_raw())
    };

    if flip_y && h > 1 {
        flip_rgba_vertical(w, h, &mut pixels);
    }
    Ok((w, h, pixels))
}

fn flip_rgba_vertical(width: u32, height: u32, pixels: &mut [u8]) {
    let row_bytes = (width as usize) * 4;
    let h = height as usize;
    if row_bytes == 0 || pixels.len() < row_bytes * h {
        return;
    }
    for top_row in 0..(h / 2) {
        let bot_row = h - 1 - top_row;
        let top_start = top_row * row_bytes;
        let bot_start = bot_row * row_bytes;
        let (head, tail) = pixels.split_at_mut(bot_start);
        head[top_start..top_start + row_bytes].swap_with_slice(&mut tail[..row_bytes]);
    }
}

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
    match decode_rgba_image(path, false) {
        Ok((width, height, pixels)) => {
            if control.should_stop() {
                return;
            }
            mailbox.publish(FramePayload::Rgba {
                width,
                height,
                pixels,
            });
        }
        Err(err) => {
            mailbox.send_error(format!("{err}"));
        }
    }
}

pub fn run_ffmpeg_static(path: &Path, control: &WorkerControl, mailbox: &FrameMailbox) {
    if control.should_stop() {
        return;
    }
    match decode_ffmpeg_static_rgba(path) {
        Ok((width, height, pixels)) => {
            if control.should_stop() {
                return;
            }
            mailbox.publish(FramePayload::Rgba {
                width,
                height,
                pixels,
            });
        }
        Err(err) => mailbox.send_error(format!("{err}")),
    }
}

fn decode_ffmpeg_static_rgba(path: &Path) -> Result<(u32, u32, Vec<u8>)> {
    // Probe dimensions first so raw RGBA output needs no header parsing.
    let probe = super::video::probe_media(path)
        .with_context(|| format!("ffprobe failed for static image {}", path.display()))?;

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
        .output()
        .with_context(|| format!("failed to spawn ffmpeg for {}", path.display()))?;

    if !output.status.success() {
        let err_msg = String::from_utf8_lossy(&output.stderr);
        bail!(
            "ffmpeg decode failed for {}: {}",
            path.display(),
            err_msg.trim()
        );
    }

    let expected = (probe.width as usize) * (probe.height as usize) * 4;
    if output.stdout.len() < expected {
        bail!(
            "ffmpeg returned truncated RGBA frame ({} < {expected}) for {}",
            output.stdout.len(),
            path.display()
        );
    }
    let mut pixels = output.stdout;
    pixels.truncate(expected);
    Ok((probe.width, probe.height, pixels))
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
