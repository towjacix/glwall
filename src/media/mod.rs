pub mod static_img;
pub mod video;

use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use calloop::channel::Sender;

use crate::config::{ColorSpace, HwAccel};
use crate::gpu::renderer::TextureKey;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaKind {
    ProceduralShader,
    StaticImage,
    FfmpegStaticImage,
    AnimatedGif,
    AnimatedWebp,
    Video,
}

impl MediaKind {
    pub fn from_path(path: &Path) -> Option<Self> {
        let ext = path
            .extension()
            .and_then(|s| s.to_str())?
            .to_ascii_lowercase();
        match ext.as_str() {
            "frag" | "glsl" => Some(Self::ProceduralShader),
            "jpg" | "jpeg" | "png" | "bmp" => Some(Self::StaticImage),
            "gif" => Some(Self::AnimatedGif),
            "webp" => {
                if static_img::is_animated_webp(path) {
                    Some(Self::AnimatedWebp)
                } else {
                    Some(Self::StaticImage)
                }
            }
            "avif" | "heif" | "heic" | "jxl" => Some(Self::FfmpegStaticImage),
            "mp4" | "webm" | "mkv" | "mov" | "m4v" | "avi" => Some(Self::Video),
            _ => None,
        }
    }

    pub fn is_animated(self) -> bool {
        matches!(self, Self::AnimatedGif | Self::AnimatedWebp | Self::Video)
    }
}

#[derive(Debug)]
pub enum FramePayload {
    Rgba {
        width: u32,
        height: u32,
        pixels: Vec<u8>,
    },
    Nv12 {
        width: u32,
        height: u32,
        colorspace: ColorSpace,
        y_plane: Vec<u8>,
        uv_plane: Vec<u8>,
    },
}

#[derive(Debug, Default)]
struct MailboxInner {
    pending: Option<FramePayload>,
    spare_rgba: Vec<u8>,
    spare_y: Vec<u8>,
    spare_uv: Vec<u8>,
}

#[derive(Debug)]
pub struct FrameMailbox {
    inner: Mutex<MailboxInner>,
    notify_queued: AtomicBool,
    sender: Sender<MediaEvent>,
    output: String,
    generation: u64,
}

impl FrameMailbox {
    pub fn new(sender: Sender<MediaEvent>, output: String, generation: u64) -> Arc<Self> {
        Arc::new(Self {
            inner: Mutex::new(MailboxInner::default()),
            notify_queued: AtomicBool::new(false),
            sender,
            output,
            generation,
        })
    }

    pub fn take_spare_rgba(&self, len: usize) -> Vec<u8> {
        let mut guard = self.inner.lock().unwrap();
        let mut buf = std::mem::take(&mut guard.spare_rgba);
        buf.resize(len, 0);
        buf
    }

    pub fn take_spare_nv12(&self, y_len: usize, uv_len: usize) -> (Vec<u8>, Vec<u8>) {
        let mut guard = self.inner.lock().unwrap();
        let mut y = std::mem::take(&mut guard.spare_y);
        let mut uv = std::mem::take(&mut guard.spare_uv);
        y.resize(y_len, 0);
        uv.resize(uv_len, 0);
        (y, uv)
    }

    pub fn publish(&self, payload: FramePayload) {
        {
            let mut guard = self.inner.lock().unwrap();
            if let Some(old) = guard.pending.replace(payload) {
                recycle_into(&mut guard, old);
            }
        }
        if !self.notify_queued.swap(true, Ordering::AcqRel) {
            let _ = self.sender.send(MediaEvent::FrameAvailable {
                output: self.output.clone(),
                generation: self.generation,
            });
        }
    }

    pub fn pop_latest(&self) -> Option<FramePayload> {
        self.notify_queued.store(false, Ordering::Release);
        self.inner.lock().unwrap().pending.take()
    }

    pub fn recycle(&self, payload: FramePayload) {
        let mut guard = self.inner.lock().unwrap();
        recycle_into(&mut guard, payload);
    }

    pub fn send_error(&self, message: String) {
        let _ = self.sender.send(MediaEvent::Error {
            output: self.output.clone(),
            generation: self.generation,
            message,
        });
    }
}

fn recycle_into(guard: &mut MailboxInner, payload: FramePayload) {
    match payload {
        FramePayload::Rgba { pixels, .. } => guard.spare_rgba = pixels,
        FramePayload::Nv12 {
            y_plane, uv_plane, ..
        } => {
            guard.spare_y = y_plane;
            guard.spare_uv = uv_plane;
        }
    }
}

#[derive(Debug)]
pub enum MediaEvent {
    FrameAvailable {
        output: String,
        generation: u64,
    },
    Error {
        output: String,
        generation: u64,
        message: String,
    },
    AuxTextureReady {
        key: TextureKey,
        width: u32,
        height: u32,
        pixels: Vec<u8>,
    },
    AuxTextureError {
        key: TextureKey,
        message: String,
    },
}

pub fn decode_texture_async(key: TextureKey, sender: Sender<MediaEvent>) {
    let _ = std::thread::Builder::new()
        .name("glwall-tex".to_string())
        .spawn(
            move || match static_img::decode_rgba_image(&key.path, key.flip_y) {
                Ok((width, height, pixels)) => {
                    let _ = sender.send(MediaEvent::AuxTextureReady {
                        key,
                        width,
                        height,
                        pixels,
                    });
                }
                Err(err) => {
                    let _ = sender.send(MediaEvent::AuxTextureError {
                        key,
                        message: format!("{err}"),
                    });
                }
            },
        );
}

#[derive(Debug)]
pub struct WorkerControl {
    stop: AtomicBool,
    paused: AtomicBool,
    cv_lock: Mutex<()>,
    cv: Condvar,
    child_pid: AtomicI32,
}

impl WorkerControl {
    pub fn new(initially_paused: bool) -> Arc<Self> {
        Arc::new(Self {
            stop: AtomicBool::new(false),
            paused: AtomicBool::new(initially_paused),
            cv_lock: Mutex::new(()),
            cv: Condvar::new(),
            child_pid: AtomicI32::new(0),
        })
    }

    pub fn should_stop(&self) -> bool {
        self.stop.load(Ordering::Acquire)
    }

    pub fn is_paused(&self) -> bool {
        self.paused.load(Ordering::Acquire)
    }

    pub fn set_child_pid(&self, pid: i32) {
        self.child_pid.store(pid, Ordering::Release);
        if pid > 0 {
            if self.should_stop() {
                unsafe {
                    libc::kill(pid, libc::SIGCONT);
                    libc::kill(pid, libc::SIGTERM);
                }
            } else if self.is_paused() {
                unsafe {
                    libc::kill(pid, libc::SIGSTOP);
                }
            }
        }
    }

    pub fn clear_child_pid(&self) {
        self.child_pid.store(0, Ordering::Release);
    }

    pub fn set_paused(&self, paused: bool) {
        let prev = self.paused.swap(paused, Ordering::AcqRel);
        if prev != paused {
            self.cv.notify_all();
            let pid = self.child_pid.load(Ordering::Acquire);
            if pid > 0 {
                let sig = if paused { libc::SIGSTOP } else { libc::SIGCONT };
                unsafe {
                    libc::kill(pid, sig);
                }
            }
        }
    }

    pub fn request_stop(&self) {
        self.stop.store(true, Ordering::Release);
        self.cv.notify_all();
        let pid = self.child_pid.load(Ordering::Acquire);
        if pid > 0 {
            unsafe {
                libc::kill(pid, libc::SIGCONT);
                libc::kill(pid, libc::SIGTERM);
            }
        }
    }

    /// Waits while paused (returning `false` if stop requested) or sleeps for `dur`
    /// while remaining interruptible by stop/pause transitions.
    pub fn wait_if_paused(&self) -> bool {
        let mut guard = self.cv_lock.lock().unwrap();
        while !self.should_stop() && self.is_paused() {
            guard = self.cv.wait(guard).unwrap();
        }
        !self.should_stop()
    }

    /// Sleeps up to `dur` unless stop requested; pauses freeze remaining wait cleanly.
    pub fn interruptible_sleep(&self, mut remaining: Duration) -> bool {
        while remaining > Duration::ZERO {
            if !self.wait_if_paused() {
                return false;
            }
            let start = Instant::now();
            let guard = self.cv_lock.lock().unwrap();
            let (_g, _) = self.cv.wait_timeout(guard, remaining).unwrap();
            if self.should_stop() {
                return false;
            }
            if !self.is_paused() {
                remaining = remaining.saturating_sub(start.elapsed());
            }
        }
        !self.should_stop()
    }
}

pub struct MediaJob {
    pub generation: u64,
    pub kind: MediaKind,
    pub control: Arc<WorkerControl>,
    pub mailbox: Arc<FrameMailbox>,
    handle: Option<JoinHandle<()>>,
}

impl MediaJob {
    #[allow(clippy::too_many_arguments)]
    pub fn spawn(
        output: String,
        generation: u64,
        path: &Path,
        kind: MediaKind,
        hwaccel: HwAccel,
        fps_cap: Option<f32>,
        initially_paused: bool,
        sender: Sender<MediaEvent>,
    ) -> Self {
        let control = WorkerControl::new(initially_paused);
        let mailbox = FrameMailbox::new(sender, output, generation);

        let worker_control = Arc::clone(&control);
        let worker_mailbox = Arc::clone(&mailbox);
        let path_buf = path.to_path_buf();

        let handle = std::thread::Builder::new()
            .name("glwall-media".to_string())
            .spawn(move || match kind {
                MediaKind::ProceduralShader => {}
                MediaKind::StaticImage => {
                    static_img::run_static(&path_buf, &worker_control, &worker_mailbox);
                }
                MediaKind::FfmpegStaticImage => {
                    static_img::run_ffmpeg_static(&path_buf, &worker_control, &worker_mailbox);
                }
                MediaKind::AnimatedGif => {
                    static_img::run_animated_gif(
                        &path_buf,
                        fps_cap,
                        &worker_control,
                        &worker_mailbox,
                    );
                }
                MediaKind::AnimatedWebp => {
                    static_img::run_animated_webp(
                        &path_buf,
                        fps_cap,
                        &worker_control,
                        &worker_mailbox,
                    );
                }
                MediaKind::Video => {
                    video::run_video(
                        &path_buf,
                        hwaccel,
                        fps_cap,
                        &worker_control,
                        &worker_mailbox,
                    );
                }
            })
            .ok();

        Self {
            generation,
            kind,
            control,
            mailbox,
            handle,
        }
    }

    pub fn set_paused(&self, paused: bool) {
        self.control.set_paused(paused);
    }

    pub fn stop(mut self) {
        self.control.request_stop();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for MediaJob {
    fn drop(&mut self) {
        self.control.request_stop();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_media_kinds_by_extension() {
        assert_eq!(
            MediaKind::from_path(Path::new("/tmp/aurora.frag")),
            Some(MediaKind::ProceduralShader)
        );
        assert_eq!(
            MediaKind::from_path(Path::new("/tmp/wall.JPG")),
            Some(MediaKind::StaticImage)
        );
        assert_eq!(
            MediaKind::from_path(Path::new("/tmp/photo.avif")),
            Some(MediaKind::FfmpegStaticImage)
        );
        assert_eq!(
            MediaKind::from_path(Path::new("/tmp/loop.gif")),
            Some(MediaKind::AnimatedGif)
        );
        assert_eq!(
            MediaKind::from_path(Path::new("/tmp/loop.mp4")),
            Some(MediaKind::Video)
        );
        assert_eq!(MediaKind::from_path(Path::new("/tmp/readme.txt")), None);
    }
}
