pub mod gst_cuda;
pub mod static_img;
pub mod video;

use std::collections::VecDeque;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
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
    Cuda(gst_cuda::CudaFrame),
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

    pub fn clear_pending(&self) {
        self.notify_queued.store(false, Ordering::Release);
        let mut guard = self.inner.lock().unwrap();
        guard.pending = None;
        guard.spare_rgba = Vec::new();
        guard.spare_y = Vec::new();
        guard.spare_uv = Vec::new();
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
        FramePayload::Cuda(_) => {}
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

/// Spawns an aux-texture decode on a helper thread. A panic inside the
/// worker is converted into `AuxTextureError` so the daemon never leaves a
/// `pending_apply`/`inflight_tex_decodes` entry stuck forever.
pub fn decode_texture_async(key: TextureKey, sender: Sender<MediaEvent>) {
    let spawn = std::thread::Builder::new()
        .name("glwall-tex".to_string())
        .spawn({
            let key = key.clone();
            let sender = sender.clone();
            move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    static_img::decode_rgba_image(&key.path, key.flip_y)
                }));
                match result {
                    Ok(Ok((width, height, pixels))) => {
                        let _ = sender.send(MediaEvent::AuxTextureReady {
                            key,
                            width,
                            height,
                            pixels,
                        });
                    }
                    Ok(Err(err)) => {
                        let _ = sender.send(MediaEvent::AuxTextureError {
                            key,
                            message: format!("{err}"),
                        });
                    }
                    Err(_) => {
                        let _ = sender.send(MediaEvent::AuxTextureError {
                            key,
                            message: "aux texture decode panicked".to_string(),
                        });
                    }
                }
            }
        });
    if spawn.is_err() {
        let _ = sender.send(MediaEvent::AuxTextureError {
            key,
            message: "failed to spawn aux texture decode thread".to_string(),
        });
    }
}

#[derive(Debug)]
pub struct WorkerControl {
    stop: AtomicBool,
    pause_state: AtomicU64,
    cuda_failed: AtomicBool,
    cv_lock: Mutex<()>,
    cv: Condvar,
    child_pid: AtomicI32,
}

impl WorkerControl {
    pub fn new(initially_paused: bool) -> Arc<Self> {
        Arc::new(Self {
            stop: AtomicBool::new(false),
            pause_state: AtomicU64::new(u64::from(initially_paused)),
            cuda_failed: AtomicBool::new(false),
            cv_lock: Mutex::new(()),
            cv: Condvar::new(),
            child_pid: AtomicI32::new(0),
        })
    }

    pub fn should_stop(&self) -> bool {
        self.stop.load(Ordering::Acquire)
    }

    pub fn cuda_failed(&self) -> bool {
        self.cuda_failed.load(Ordering::Acquire)
    }

    pub fn request_cuda_fallback(&self) {
        self.cuda_failed.store(true, Ordering::Release);
        self.cv.notify_all();
    }

    pub fn is_paused(&self) -> bool {
        self.pause_state() & 1 != 0
    }

    fn pause_state(&self) -> u64 {
        self.pause_state.load(Ordering::Acquire)
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
        let guard = self.cv_lock.lock().unwrap();
        let state = self.pause_state();
        if (state & 1 != 0) == paused {
            return;
        }
        self.pause_state
            .store(state.wrapping_add(1), Ordering::Release);
        drop(guard);
        self.cv.notify_all();
        let pid = self.child_pid.load(Ordering::Acquire);
        if pid > 0 {
            let sig = if paused { libc::SIGSTOP } else { libc::SIGCONT };
            unsafe {
                libc::kill(pid, sig);
            }
        }
    }

    pub fn request_stop(&self) {
        {
            let _guard = self.cv_lock.lock().unwrap();
            self.stop.store(true, Ordering::Release);
        }
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
            if self.should_stop() {
                return false;
            }
            if self.is_paused() {
                continue;
            }
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

#[cfg(test)]
thread_local! {
    static FORCE_SPAWN_FAIL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
pub fn set_force_spawn_fail(fail: bool) {
    FORCE_SPAWN_FAIL.with(|f| f.set(fail));
}

struct ReaperState {
    queue: VecDeque<JoinHandle<()>>,
    in_flight: usize,
}

struct WorkerReaper {
    state: Mutex<ReaperState>,
    cv: Condvar,
    done_cv: Condvar,
}

static REAPER: OnceLock<WorkerReaper> = OnceLock::new();

impl WorkerReaper {
    fn global() -> &'static Self {
        REAPER.get_or_init(|| {
            let reaper = Self {
                state: Mutex::new(ReaperState {
                    queue: VecDeque::new(),
                    in_flight: 0,
                }),
                cv: Condvar::new(),
                done_cv: Condvar::new(),
            };
            let _ = std::thread::Builder::new()
                .name("glwall-reaper".to_string())
                .spawn(|| {
                    let me = REAPER.wait();
                    loop {
                        let handle = {
                            let mut guard = me.state.lock().unwrap_or_else(|e| e.into_inner());
                            while guard.queue.is_empty() {
                                guard = me.cv.wait(guard).unwrap_or_else(|e| e.into_inner());
                            }
                            guard.queue.pop_front()
                        };
                        if let Some(h) = handle {
                            let _ = h.join();
                            let idle_now = {
                                let mut guard = me.state.lock().unwrap_or_else(|e| e.into_inner());
                                guard.in_flight = guard.in_flight.saturating_sub(1);
                                let empty = guard.queue.is_empty() && guard.in_flight == 0;
                                if empty {
                                    me.done_cv.notify_all();
                                }
                                empty
                            };
                            if idle_now {
                                gst_cuda::reclaim_idle_resources();
                                #[cfg(all(target_os = "linux", target_env = "gnu"))]
                                unsafe {
                                    libc::malloc_trim(0);
                                }
                            }
                        }
                    }
                });
            reaper
        })
    }

    fn enqueue(&self, handle: JoinHandle<()>) {
        let mut guard = self.state.lock().unwrap_or_else(|e| e.into_inner());
        guard.in_flight += 1;
        guard.queue.push_back(handle);
        self.cv.notify_one();
    }
}

/// Blocks until every worker thread handed to the background reaper has
/// finished joining (primarily for clean daemon shutdown and tests).
pub fn wait_for_reaped_workers() {
    let Some(reaper) = REAPER.get() else {
        return;
    };
    let mut guard = reaper.state.lock().unwrap_or_else(|e| e.into_inner());
    while !guard.queue.is_empty() || guard.in_flight > 0 {
        guard = reaper
            .done_cv
            .wait(guard)
            .unwrap_or_else(|e| e.into_inner());
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
    ) -> Result<Self> {
        #[cfg(test)]
        if FORCE_SPAWN_FAIL.with(|f| f.get()) {
            anyhow::bail!("injected media worker spawn failure");
        }

        let control = WorkerControl::new(initially_paused);
        let panic_sender = sender.clone();
        let panic_output = output.clone();
        let mailbox = FrameMailbox::new(sender, output, generation);

        let worker_control = Arc::clone(&control);
        let worker_mailbox = Arc::clone(&mailbox);
        let path_buf = path.to_path_buf();

        let handle = std::thread::Builder::new()
            .name("glwall-media".to_string())
            .spawn({
                // If a decoder panics, synthesize a media Error so the daemon
                // reverts/cancels cleanly instead of leaving `waiting_target_frame`
                // and slot1/prev_job resources stuck forever.

                move || {
                    let run = || match kind {
                        MediaKind::ProceduralShader => {}
                        MediaKind::StaticImage => {
                            static_img::run_static(&path_buf, &worker_control, &worker_mailbox);
                        }
                        MediaKind::FfmpegStaticImage => {
                            static_img::run_ffmpeg_static(
                                &path_buf,
                                &worker_control,
                                &worker_mailbox,
                            );
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
                    };
                    if std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)).is_err() {
                        let _ = panic_sender.send(MediaEvent::Error {
                            output: panic_output,
                            generation,
                            message: "media worker panicked".to_string(),
                        });
                    }
                    drop(worker_mailbox);
                }
            })
            .context("failed to spawn media decode worker thread")?;

        Ok(Self {
            generation,
            kind,
            control,
            mailbox,
            handle: Some(handle),
        })
    }

    pub fn set_paused(&self, paused: bool) {
        self.control.set_paused(paused);
    }

    fn stop_and_reap(&mut self) {
        self.control.request_stop();
        self.mailbox.clear_pending();
        if let Some(handle) = self.handle.take() {
            if handle.is_finished() {
                let _ = handle.join();
            } else {
                WorkerReaper::global().enqueue(handle);
            }
        }
    }

    pub fn stop(mut self) {
        self.stop_and_reap();
    }
}

impl Drop for MediaJob {
    fn drop(&mut self) {
        self.stop_and_reap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;

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

    #[test]
    fn media_job_spawn_propagates_error_on_fault_injection() {
        struct ResetGuard;
        impl Drop for ResetGuard {
            fn drop(&mut self) {
                set_force_spawn_fail(false);
            }
        }

        let (tx, _rx) = calloop::channel::channel::<MediaEvent>();
        set_force_spawn_fail(true);
        let _guard = ResetGuard;

        let res = MediaJob::spawn(
            "DP-1".to_string(),
            1,
            Path::new("/tmp/missing.jpg"),
            MediaKind::StaticImage,
            HwAccel::Auto,
            None,
            false,
            tx,
        );
        assert!(res.is_err());
    }

    #[test]
    fn media_job_stop_is_nonblocking_and_reaps_in_background() {
        let (tx, _rx) = calloop::channel::channel::<MediaEvent>();
        let control = WorkerControl::new(false);
        let mailbox = FrameMailbox::new(tx, "DP-1".to_string(), 7);
        let exited = Arc::new(AtomicBool::new(false));

        let w_ctrl = Arc::clone(&control);
        let w_exited = Arc::clone(&exited);
        let handle = std::thread::Builder::new()
            .name("glwall-test-worker".to_string())
            .spawn(move || {
                while !w_ctrl.should_stop() {
                    if !w_ctrl.interruptible_sleep(Duration::from_millis(20)) {
                        break;
                    }
                }
                std::thread::sleep(Duration::from_millis(250));
                w_exited.store(true, Ordering::Release);
            })
            .unwrap();

        let job = MediaJob {
            generation: 7,
            kind: MediaKind::Video,
            control,
            mailbox,
            handle: Some(handle),
        };

        let start = Instant::now();
        job.stop();
        let stop_elapsed = start.elapsed();
        assert!(
            stop_elapsed < Duration::from_millis(50),
            "MediaJob::stop blocked event loop for {stop_elapsed:?}"
        );

        wait_for_reaped_workers();
        assert!(
            exited.load(Ordering::Acquire),
            "background reaper did not join worker thread"
        );
    }
}
