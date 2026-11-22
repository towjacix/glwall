pub mod gst_cuda;
pub mod static_img;
pub mod video;

use std::collections::VecDeque;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use calloop::channel::Sender;

use crate::config::{ColorSpace, HwAccel};
use crate::gpu::renderer::TextureKey;
use crate::ipc::{IpcErrorCode, coded_err, error_code_of};

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
        self.send_coded_error(IpcErrorCode::RenderFailed, message);
    }

    pub fn send_coded_error(&self, code: IpcErrorCode, message: String) {
        let _ = self.sender.send(MediaEvent::Error {
            output: self.output.clone(),
            generation: self.generation,
            code,
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
        code: IpcErrorCode,
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
        code: IpcErrorCode,
        message: String,
    },
}

pub const AUX_DECODE_WORKERS: usize = 2;
pub const MAX_AUX_DECODE_QUEUE: usize = 16;

struct AuxDecodeTask {
    key: TextureKey,
    sender: Sender<MediaEvent>,
}

type AuxDecodeFn = dyn Fn(&TextureKey) -> Result<(u32, u32, Vec<u8>)> + Send + Sync;

struct AuxPoolState {
    queue: VecDeque<AuxDecodeTask>,
    workers_spawned: usize,
    active_workers: usize,
    shutdown: bool,
}

struct AuxDecodePool {
    max_workers: usize,
    max_queue: usize,
    decoder: Arc<AuxDecodeFn>,
    state: Mutex<AuxPoolState>,
    cv: Condvar,
    active_count: AtomicUsize,
    queued_count: AtomicUsize,
}

impl AuxDecodePool {
    fn new(max_workers: usize, max_queue: usize, decoder: Arc<AuxDecodeFn>) -> Self {
        Self {
            max_workers: max_workers.max(1),
            max_queue: max_queue.max(1),
            decoder,
            state: Mutex::new(AuxPoolState {
                queue: VecDeque::new(),
                workers_spawned: 0,
                active_workers: 0,
                shutdown: false,
            }),
            cv: Condvar::new(),
            active_count: AtomicUsize::new(0),
            queued_count: AtomicUsize::new(0),
        }
    }

    fn stats(&self) -> (usize, usize) {
        (
            self.active_count.load(Ordering::Acquire),
            self.queued_count.load(Ordering::Acquire),
        )
    }

    fn has_capacity(&self, additional: usize) -> bool {
        let guard = self.state.lock().unwrap_or_else(|e| e.into_inner());
        guard.queue.len() + additional <= self.max_queue
    }

    fn enqueue(self: &Arc<Self>, key: TextureKey, sender: Sender<MediaEvent>) -> Result<()> {
        let mut guard = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if guard.queue.len() >= self.max_queue {
            return Err(coded_err(
                IpcErrorCode::ResourceLimit,
                format!(
                    "auxiliary texture decode queue full ({}/{})",
                    guard.queue.len(),
                    self.max_queue
                ),
            ));
        }
        guard.queue.push_back(AuxDecodeTask { key, sender });
        self.queued_count
            .store(guard.queue.len(), Ordering::Release);

        let should_spawn = guard.workers_spawned < self.max_workers
            && guard.workers_spawned < guard.active_workers + guard.queue.len();
        if should_spawn {
            let worker_idx = guard.workers_spawned;
            let pool = Arc::clone(self);
            match std::thread::Builder::new()
                .name(format!("glwall-tex-{worker_idx}"))
                .spawn(move || {
                    loop {
                        let task = {
                            let mut g = pool.state.lock().unwrap_or_else(|e| e.into_inner());
                            while g.queue.is_empty() && !g.shutdown {
                                g = pool.cv.wait(g).unwrap_or_else(|e| e.into_inner());
                            }
                            if g.shutdown && g.queue.is_empty() {
                                return;
                            }
                            let task = g.queue.pop_front().unwrap();
                            g.active_workers += 1;
                            pool.queued_count.store(g.queue.len(), Ordering::Release);
                            pool.active_count.store(g.active_workers, Ordering::Release);
                            task
                        };

                        let key_for_decode = task.key.clone();
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            (pool.decoder)(&key_for_decode)
                        }));
                        match result {
                            Ok(Ok((width, height, pixels))) => {
                                let _ = task.sender.send(MediaEvent::AuxTextureReady {
                                    key: task.key,
                                    width,
                                    height,
                                    pixels,
                                });
                            }
                            Ok(Err(err)) => {
                                let code = error_code_of(&err, IpcErrorCode::UnsupportedMedia);
                                let _ = task.sender.send(MediaEvent::AuxTextureError {
                                    key: task.key,
                                    code,
                                    message: format!("{err:#}"),
                                });
                            }
                            Err(_) => {
                                let _ = task.sender.send(MediaEvent::AuxTextureError {
                                    key: task.key,
                                    code: IpcErrorCode::RenderFailed,
                                    message: "auxiliary texture decoder panicked".to_string(),
                                });
                            }
                        }

                        {
                            let mut g = pool.state.lock().unwrap_or_else(|e| e.into_inner());
                            g.active_workers = g.active_workers.saturating_sub(1);
                            pool.active_count.store(g.active_workers, Ordering::Release);
                        }
                    }
                }) {
                Ok(_) => {
                    guard.workers_spawned += 1;
                }
                Err(err) => {
                    if guard.workers_spawned == 0 {
                        let removed = guard.queue.pop_back();
                        self.queued_count
                            .store(guard.queue.len(), Ordering::Release);
                        drop(removed);
                        return Err(coded_err(
                            IpcErrorCode::Unavailable,
                            format!("failed to spawn auxiliary texture worker: {err}"),
                        ));
                    }
                }
            }
        }
        self.cv.notify_one();
        Ok(())
    }

    #[cfg(test)]
    fn shutdown(&self) {
        let mut guard = self.state.lock().unwrap_or_else(|e| e.into_inner());
        guard.shutdown = true;
        self.cv.notify_all();
    }
}

static AUX_POOL: OnceLock<Arc<AuxDecodePool>> = OnceLock::new();

fn global_aux_pool() -> &'static Arc<AuxDecodePool> {
    AUX_POOL.get_or_init(|| {
        Arc::new(AuxDecodePool::new(
            AUX_DECODE_WORKERS,
            MAX_AUX_DECODE_QUEUE,
            Arc::new(|key: &TextureKey| static_img::decode_rgba_image(&key.path, key.flip_y)),
        ))
    })
}

/// Checks whether the global auxiliary decode queue can accept `additional` jobs.
pub fn aux_decode_has_capacity(additional: usize) -> bool {
    global_aux_pool().has_capacity(additional)
}

/// Returns `(active_aux_workers, queued_aux_decodes)` without blocking workers.
pub fn aux_decode_stats() -> (usize, usize) {
    match AUX_POOL.get() {
        Some(pool) => pool.stats(),
        None => (0, 0),
    }
}

/// Submits an aux-texture decode to the bounded 2-worker pool.
pub fn decode_texture_async(key: TextureKey, sender: Sender<MediaEvent>) -> Result<()> {
    global_aux_pool().enqueue(key, sender)
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

static ACTIVE_MEDIA_WORKERS: AtomicUsize = AtomicUsize::new(0);

/// Returns the number of live primary media decode worker threads.
pub fn active_media_workers() -> usize {
    ACTIVE_MEDIA_WORKERS.load(Ordering::Acquire)
}

struct ActiveMediaWorkerGuard;

impl ActiveMediaWorkerGuard {
    fn enter() -> Self {
        ACTIVE_MEDIA_WORKERS.fetch_add(1, Ordering::AcqRel);
        Self
    }
}

impl Drop for ActiveMediaWorkerGuard {
    fn drop(&mut self) {
        ACTIVE_MEDIA_WORKERS.fetch_sub(1, Ordering::AcqRel);
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
                    let _worker_guard = ActiveMediaWorkerGuard::enter();
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
                            code: IpcErrorCode::RenderFailed,
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

    #[test]
    fn aux_decode_pool_bounds_concurrency_and_queue_depth() {
        use crate::gpu::renderer::TextureFingerprint;
        use crate::gpu::uniforms::{TextureFilter, TextureWrap};

        let in_flight = Arc::new(AtomicUsize::new(0));
        let max_seen = Arc::new(AtomicUsize::new(0));
        let completed = Arc::new(AtomicUsize::new(0));
        let gate = Arc::new((Mutex::new(false), Condvar::new()));

        let d_in_flight = Arc::clone(&in_flight);
        let d_max = Arc::clone(&max_seen);
        let d_done = Arc::clone(&completed);
        let d_gate = Arc::clone(&gate);

        let pool = Arc::new(AuxDecodePool::new(
            2,
            4,
            Arc::new(move |_| {
                let now = d_in_flight.fetch_add(1, Ordering::AcqRel) + 1;
                d_max.fetch_max(now, Ordering::AcqRel);
                let (lock, cv) = &*d_gate;
                let mut open = lock.lock().unwrap();
                while !*open {
                    open = cv.wait(open).unwrap();
                }
                drop(open);
                d_in_flight.fetch_sub(1, Ordering::AcqRel);
                d_done.fetch_add(1, Ordering::AcqRel);
                Ok((1, 1, vec![255, 0, 0, 255]))
            }),
        ));

        let (tx, _rx) = calloop::channel::channel::<MediaEvent>();
        let make_key = |n: usize| TextureKey {
            path: std::path::PathBuf::from(format!("/tmp/tex-{n}.png")),
            filter: TextureFilter::Linear,
            wrap: TextureWrap::ClampToEdge,
            flip_y: false,
            fingerprint: TextureFingerprint {
                size: n as u64,
                mtime_sec: 0,
                mtime_nsec: 0,
                content_hash: n as u64,
            },
        };

        pool.enqueue(make_key(0), tx.clone()).unwrap();
        pool.enqueue(make_key(1), tx.clone()).unwrap();

        let deadline = Instant::now() + Duration::from_secs(2);
        while (in_flight.load(Ordering::Acquire) < 2 || pool.stats().1 > 0)
            && Instant::now() < deadline
        {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(in_flight.load(Ordering::Acquire), 2);
        assert_eq!(pool.stats(), (2, 0));

        for idx in 2..6 {
            pool.enqueue(make_key(idx), tx.clone()).unwrap();
        }
        assert_eq!(pool.stats(), (2, 4));
        assert!(!pool.has_capacity(1));

        let err = pool.enqueue(make_key(6), tx).unwrap_err();
        assert_eq!(
            error_code_of(&err, IpcErrorCode::InvalidInput),
            IpcErrorCode::ResourceLimit
        );

        {
            let (lock, cv) = &*gate;
            *lock.lock().unwrap() = true;
            cv.notify_all();
        }

        let finish_deadline = Instant::now() + Duration::from_secs(2);
        while (completed.load(Ordering::Acquire) < 6 || pool.stats() != (0, 0))
            && Instant::now() < finish_deadline
        {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(completed.load(Ordering::Acquire), 6);
        assert_eq!(pool.stats(), (0, 0));
        assert!(max_seen.load(Ordering::Acquire) <= 2);
        pool.shutdown();
    }
}
