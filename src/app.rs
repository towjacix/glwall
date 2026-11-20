use std::collections::HashSet;
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use calloop::channel::{self, Sender};
use calloop::generic::Generic;
use calloop::signals::{Signal, Signals};
use calloop::{EventLoop, Interest, Mode, PostAction};
use serde_json::json;
use smithay_client_toolkit::reexports::calloop_wayland_source::WaylandSource;
use smithay_client_toolkit::shell::WaylandSurface;
use wayland_client::Proxy;
use wayland_client::protocol::{wl_output, wl_surface};

use crate::config::{Config, FitMode, HwAccel, OutputSpec, PersistedState, default_state_path};
use crate::gpu::egl::EglState;
use crate::gpu::renderer::{DrawParams, GlProgram, MediaSlotGpu, Renderer, SlotMediaKind};
use crate::gpu::shader::resolve_transition_source;
use crate::ipc::{
    IpcRequest, IpcResponse, bind_ipc_listener, read_ipc_request, write_ipc_response,
};
use crate::media::{FramePayload, MediaEvent, MediaJob, MediaKind};
use crate::pause::PauseState;
use crate::watcher::{ShaderWatcher, absolutize_path};
use crate::wayland::{SurfaceHandle, WaylandAppSink, WaylandContext, WaylandHandles};

struct OutputEntry {
    index: i32,
    wl_output: wl_output::WlOutput,
    name: String,
    name_resolved: bool,
    surface: SurfaceHandle,
    spec: OutputSpec,
    generation: u64,
    slot0: MediaSlotGpu,
    slot1: Option<MediaSlotGpu>,
    slot0_has_first_frame: bool,
    job: Option<MediaJob>,
    prev_job: Option<MediaJob>,
    custom_program: Option<GlProgram>,
    custom_shader_path: Option<PathBuf>,
    last_shader_error: Option<String>,
    last_media_error: Option<String>,
    transition_program: Option<GlProgram>,
    transition_started: Option<Instant>,
    waiting_target_frame: bool,
    in_transition: bool,
    anim_time_secs: f32,
    last_tick: Instant,
    frame_index: i32,
}

impl OutputEntry {
    fn uses_continuous_frames(&self) -> bool {
        if self.in_transition {
            return true;
        }
        if self.job.as_ref().is_some_and(|j| j.kind.is_animated()) {
            return true;
        }
        self.custom_program
            .as_ref()
            .is_some_and(|p| p.assembled.uses_time)
    }
}

pub struct AppCore {
    config: Config,
    state_path: Option<PathBuf>,
    persisted: PersistedState,
    restore_on_connect: bool,
    egl: EglState,
    renderer: Option<Renderer>,
    outputs: Vec<OutputEntry>,
    next_output_index: i32,
    pause: PauseState,
    fullscreen_outputs: HashSet<String>,
    media_tx: Sender<MediaEvent>,
    watcher: ShaderWatcher,
    should_exit: bool,
}

fn connect_wayland_and_egl(
    config: Config,
    state_path: Option<PathBuf>,
    persisted: PersistedState,
    restore_on_connect: bool,
    media_tx: Sender<MediaEvent>,
    watcher: ShaderWatcher,
) -> Result<(
    wayland_client::Connection,
    wayland_client::EventQueue<WaylandContext<AppCore>>,
    WaylandContext<AppCore>,
)> {
    WaylandContext::connect(|conn| {
        let wl_display_ptr = conn.display().id().as_ptr() as *mut std::ffi::c_void;
        let egl = EglState::new_wayland(wl_display_ptr)?;
        let renderer = match egl.create_glow_context() {
            Ok(gl) => Renderer::new(gl).ok(),
            Err(_) => None,
        };
        Ok(AppCore {
            config,
            state_path,
            persisted,
            restore_on_connect,
            egl,
            renderer,
            outputs: Vec::new(),
            next_output_index: 0,
            pause: PauseState::new(),
            fullscreen_outputs: HashSet::new(),
            media_tx,
            watcher,
            should_exit: false,
        })
    })
}

impl AppCore {
    fn ensure_renderer_on_surface(&mut self, surf_idx: usize) -> Result<bool> {
        let entry = &mut self.outputs[surf_idx];
        if !entry.surface.configured {
            return Ok(false);
        }
        entry.surface.ensure_egl(&self.egl)?;
        let egl_surf = entry
            .surface
            .egl_surface
            .ok_or_else(|| anyhow::anyhow!("missing EGLSurface"))?;
        self.egl
            .make_current_surface(egl_surf, &mut entry.surface.swap_interval_set)?;
        if self.renderer.is_none() {
            let gl = self.egl.create_glow_context()?;
            self.renderer = Some(Renderer::new(gl)?);
        }
        Ok(true)
    }

    fn ensure_any_gl_context(&mut self) -> Result<()> {
        if self.renderer.is_some() {
            self.egl.make_current_surfaceless();
            return Ok(());
        }
        if let Ok(gl) = self.egl.create_glow_context()
            && let Ok(r) = Renderer::new(gl)
        {
            self.renderer = Some(r);
            return Ok(());
        }
        for i in 0..self.outputs.len() {
            if self.ensure_renderer_on_surface(i)? {
                return Ok(());
            }
        }
        bail!("no Wayland EGL surface or surfaceless context available yet to compile shader")
    }

    fn find_output_by_surface(&self, surface: &wl_surface::WlSurface) -> Option<usize> {
        self.outputs
            .iter()
            .position(|o| o.surface.layer.wl_surface() == surface)
    }

    fn find_outputs_matching(&self, selector: &str) -> Vec<usize> {
        let trimmed = selector.trim();
        if trimmed.is_empty() || trimmed == "*" {
            return (0..self.outputs.len()).collect();
        }
        self.outputs
            .iter()
            .enumerate()
            .filter_map(|(idx, o)| (o.name == trimmed).then_some(idx))
            .collect()
    }

    fn sync_shader_watcher(&mut self) {
        let paths: Vec<PathBuf> = self
            .outputs
            .iter()
            .filter_map(|o| o.custom_shader_path.clone())
            .collect();
        self.watcher.sync_paths(paths.iter().map(PathBuf::as_path));
    }

    fn sync_pause_to_workers(&self) {
        for entry in &self.outputs {
            let paused = self.pause.is_paused(&entry.name);
            if let Some(job) = &entry.job {
                job.set_paused(paused);
            }
            if let Some(prev) = &entry.prev_job {
                prev.set_paused(paused);
            }
        }
    }

    fn save_state(&self) {
        if let Some(ref path) = self.state_path
            && let Err(err) = self.persisted.save_atomic(path)
        {
            tracing::warn!("failed to persist state: {err}");
        }
    }

    fn apply_initial_spec_if_needed(
        &mut self,
        idx: usize,
        qh: &wayland_client::QueueHandle<WaylandContext<Self>>,
    ) {
        let out_name = self.outputs[idx].name.clone();
        let mut desired = if self.restore_on_connect {
            self.persisted
                .outputs
                .get(&out_name)
                .cloned()
                .unwrap_or_else(|| self.config.base_spec_for(&out_name))
        } else {
            self.config.base_spec_for(&out_name)
        };
        if !self.restore_on_connect && self.config.base_spec_for(&out_name).target.is_none() {
            desired.target = None;
        }
        let current_target = self.outputs[idx].spec.target.clone();
        let current_shader = self.outputs[idx].spec.shader.clone();
        if desired.target != current_target || desired.shader != current_shader {
            let target = desired.target.clone();
            let shader = desired.shader.clone();
            self.outputs[idx].spec = desired;
            if let Some(t) = target {
                let _ = self.apply_set_to_output(
                    idx,
                    &t,
                    Some(self.outputs[idx].spec.fit),
                    Some(self.outputs[idx].spec.transition.clone()),
                    Some(self.outputs[idx].spec.duration_ms),
                    shader,
                    self.outputs[idx].spec.fps,
                    Some(self.outputs[idx].spec.hwaccel),
                    false,
                    qh,
                );
            } else if let Some(s) = shader {
                let _ = self.apply_shader_to_output(idx, &s, false, qh);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_set_to_output(
        &mut self,
        idx: usize,
        raw_target: &str,
        fit: Option<FitMode>,
        transition: Option<String>,
        duration_ms: Option<u32>,
        shader_override: Option<String>,
        fps: Option<f32>,
        hwaccel: Option<HwAccel>,
        persist: bool,
        qh: &wayland_client::QueueHandle<WaylandContext<Self>>,
    ) -> Result<()> {
        let resolved_target = absolutize_path(Path::new(raw_target));
        let kind = MediaKind::from_path(&resolved_target).ok_or_else(|| {
            anyhow::anyhow!(
                "unsupported wallpaper file type: {}",
                resolved_target.display()
            )
        })?;
        if !resolved_target.is_file() {
            bail!(
                "wallpaper file does not exist: {}",
                resolved_target.display()
            );
        }

        // Handle pure procedural shader targets directly.
        if kind == MediaKind::ProceduralShader {
            let shader_src = fs::read_to_string(&resolved_target)
                .with_context(|| format!("failed to read shader {}", resolved_target.display()))?;
            self.ensure_any_gl_context()?;
            let gl = &self
                .renderer
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("renderer not initialized"))?
                .gl;
            let compiled = GlProgram::compile(gl, &shader_src)?;

            let entry = &mut self.outputs[idx];
            entry.generation += 1;
            if let Some(job) = entry.job.take() {
                job.stop();
            }
            if let Some(prev) = entry.prev_job.take() {
                prev.stop();
            }
            entry.slot0.clear(gl);
            if let Some(s1) = entry.slot1.take() {
                s1.destroy(gl);
            }
            if let Some(old_prog) = entry.custom_program.replace(compiled) {
                old_prog.destroy(gl);
            }
            if let Some(tr) = entry.transition_program.take() {
                tr.destroy(gl);
            }
            entry.slot0_has_first_frame = false;
            entry.waiting_target_frame = false;
            entry.in_transition = false;
            entry.transition_started = None;
            entry.last_media_error = None;
            entry.last_shader_error = None;
            let canonical = resolved_target.to_string_lossy().to_string();
            entry.custom_shader_path = Some(resolved_target);
            entry.spec.target = Some(canonical.clone());
            entry.spec.shader = Some(canonical);
            if let Some(f) = fit {
                entry.spec.fit = f;
            }
            if let Some(tr) = transition {
                entry.spec.transition = tr;
            }
            if let Some(ms) = duration_ms {
                entry.spec.duration_ms = ms;
            }
            if fps.is_some() {
                entry.spec.fps = fps;
            }
            if let Some(hw) = hwaccel {
                entry.spec.hwaccel = hw;
            }
            entry.surface.dirty = true;
            entry.last_tick = Instant::now();

            if persist {
                self.persisted
                    .outputs
                    .insert(entry.name.clone(), entry.spec.clone());
                self.save_state();
            }
            self.sync_shader_watcher();
            self.render_if_ready(idx, qh);
            return Ok(());
        }

        // Validate optional shader before mutating live slots.
        if let Some(ref sh) = shader_override {
            let trimmed = sh.trim();
            if trimmed == "none" {
                let entry = &mut self.outputs[idx];
                if let Some(old) = entry.custom_program.take()
                    && let Some(ref r) = self.renderer
                {
                    old.destroy(&r.gl);
                }
                entry.custom_shader_path = None;
                entry.spec.shader = None;
            } else {
                self.apply_shader_to_output(idx, trimmed, false, qh)?;
            }
        } else if self.outputs[idx]
            .spec
            .target
            .as_deref()
            .and_then(|t| MediaKind::from_path(Path::new(t)))
            == Some(MediaKind::ProceduralShader)
        {
            // Switching away from a standalone procedural shader to media: clear implicit procedural shader.
            if let Some(old) = self.outputs[idx].custom_program.take()
                && let Some(ref r) = self.renderer
            {
                old.destroy(&r.gl);
            }
            self.outputs[idx].custom_shader_path = None;
            self.outputs[idx].spec.shader = None;
        }

        let transition_name =
            transition.unwrap_or_else(|| self.outputs[idx].spec.transition.clone());
        let duration = duration_ms.unwrap_or(self.outputs[idx].spec.duration_ms);
        let can_transition = self.outputs[idx].slot0.kind != SlotMediaKind::None
            && duration > 0
            && transition_name != "none";

        let transition_prog = if can_transition {
            if let Some(tr_src) = resolve_transition_source(&transition_name)? {
                self.ensure_any_gl_context()?;
                let gl = &self
                    .renderer
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("renderer not initialized"))?
                    .gl;
                Some(GlProgram::compile(gl, &tr_src)?)
            } else {
                None
            }
        } else {
            None
        };

        let entry = &mut self.outputs[idx];
        entry.generation += 1;
        let generation = entry.generation;

        if let Some(prev) = entry.prev_job.take() {
            prev.stop();
        }
        if let Some(old_tr) = entry.transition_program.take()
            && let Some(ref r) = self.renderer
        {
            old_tr.destroy(&r.gl);
        }
        if let Some(s1) = entry.slot1.take()
            && let Some(ref r) = self.renderer
        {
            s1.destroy(&r.gl);
        }

        if let Some(f) = fit {
            entry.spec.fit = f;
        }
        entry.spec.transition = transition_name;
        entry.spec.duration_ms = duration;
        if fps.is_some() {
            entry.spec.fps = fps;
        }
        if let Some(hw) = hwaccel {
            entry.spec.hwaccel = hw;
        }
        entry.spec.target = Some(resolved_target.to_string_lossy().to_string());
        entry.last_media_error = None;

        if transition_prog.is_some() {
            let old_slot0 = std::mem::replace(&mut entry.slot0, MediaSlotGpu::empty());
            entry.slot1 = Some(old_slot0);
            entry.prev_job = entry.job.take();
            entry.transition_program = transition_prog;
            entry.waiting_target_frame = true;
            entry.in_transition = false;
            entry.transition_started = None;
        } else {
            if let Some(old_job) = entry.job.take() {
                old_job.stop();
            }
            if let Some(ref r) = self.renderer {
                entry.slot0.clear(&r.gl);
            } else {
                entry.slot0 = MediaSlotGpu::empty();
            }
            entry.waiting_target_frame = false;
            entry.in_transition = false;
            entry.transition_started = None;
        }

        entry.slot0_has_first_frame = false;
        let paused = self.pause.is_paused(&entry.name);
        let job = MediaJob::spawn(
            entry.name.clone(),
            generation,
            &resolved_target,
            kind,
            entry.spec.hwaccel,
            entry.spec.fps.or(self.config.fps_cap),
            paused,
            self.media_tx.clone(),
        );
        entry.job = Some(job);
        entry.last_tick = Instant::now();

        if persist {
            self.persisted
                .outputs
                .insert(entry.name.clone(), entry.spec.clone());
            self.save_state();
        }
        self.sync_shader_watcher();
        Ok(())
    }

    fn apply_shader_to_output(
        &mut self,
        idx: usize,
        raw_path: &str,
        persist: bool,
        qh: &wayland_client::QueueHandle<WaylandContext<Self>>,
    ) -> Result<()> {
        let trimmed = raw_path.trim();
        if trimmed == "none" || trimmed.is_empty() {
            let entry = &mut self.outputs[idx];
            if let Some(old) = entry.custom_program.take()
                && let Some(ref r) = self.renderer
            {
                old.destroy(&r.gl);
            }
            entry.custom_shader_path = None;
            entry.last_shader_error = None;
            entry.spec.shader = None;
            entry.surface.dirty = true;
            if persist {
                self.persisted
                    .outputs
                    .insert(entry.name.clone(), entry.spec.clone());
                self.save_state();
            }
            self.sync_shader_watcher();
            self.render_if_ready(idx, qh);
            return Ok(());
        }

        let resolved = absolutize_path(Path::new(trimmed));
        let src = fs::read_to_string(&resolved)
            .with_context(|| format!("failed to read shader {}", resolved.display()))?;
        self.ensure_any_gl_context()?;
        let gl = &self
            .renderer
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("renderer not initialized"))?
            .gl;
        let compiled = GlProgram::compile(gl, &src)?;

        let entry = &mut self.outputs[idx];
        if let Some(old) = entry.custom_program.replace(compiled) {
            old.destroy(gl);
        }
        let canonical = resolved.to_string_lossy().to_string();
        entry.custom_shader_path = Some(resolved);
        entry.last_shader_error = None;
        entry.spec.shader = Some(canonical);
        entry.surface.dirty = true;
        entry.last_tick = Instant::now();

        if persist {
            self.persisted
                .outputs
                .insert(entry.name.clone(), entry.spec.clone());
            self.save_state();
        }
        self.sync_shader_watcher();
        self.render_if_ready(idx, qh);
        Ok(())
    }

    fn handle_media_event(
        &mut self,
        ev: MediaEvent,
        qh: &wayland_client::QueueHandle<WaylandContext<Self>>,
    ) {
        match ev {
            MediaEvent::Error {
                output,
                generation,
                message,
            } => {
                let Some(idx) = self.outputs.iter().position(|o| o.name == output) else {
                    return;
                };
                let entry = &mut self.outputs[idx];
                if entry.generation != generation {
                    return;
                }
                tracing::error!("media error on {output}: {message}");
                entry.last_media_error = Some(message);
                if entry.waiting_target_frame {
                    // Revert pending transition cleanly back to previous slot1.
                    entry.waiting_target_frame = false;
                    if let Some(failed_job) = entry.job.take() {
                        failed_job.stop();
                    }
                    entry.job = entry.prev_job.take();
                    if let Some(prev_slot) = entry.slot1.take() {
                        if let Some(ref r) = self.renderer {
                            let old = std::mem::replace(&mut entry.slot0, prev_slot);
                            old.destroy(&r.gl);
                        } else {
                            entry.slot0 = prev_slot;
                        }
                    }
                    if let Some(tr) = entry.transition_program.take()
                        && let Some(ref r) = self.renderer
                    {
                        tr.destroy(&r.gl);
                    }
                    entry.surface.dirty = true;
                    self.render_if_ready(idx, qh);
                }
            }
            MediaEvent::FrameAvailable { output, generation } => {
                let Some(idx) = self.outputs.iter().position(|o| o.name == output) else {
                    return;
                };
                if self.outputs[idx].generation != generation {
                    return;
                }
                let (mailbox_opt, is_prev) = if self.outputs[idx]
                    .job
                    .as_ref()
                    .is_some_and(|j| j.generation == generation)
                {
                    (
                        self.outputs[idx]
                            .job
                            .as_ref()
                            .map(|j| std::sync::Arc::clone(&j.mailbox)),
                        false,
                    )
                } else {
                    (None, false)
                };
                if is_prev {
                    return;
                }
                let Some(mailbox) = mailbox_opt else {
                    return;
                };
                let Some(payload) = mailbox.pop_latest() else {
                    return;
                };

                if !matches!(self.ensure_renderer_on_surface(idx), Ok(true)) {
                    mailbox.recycle(payload);
                    return;
                }

                let fit = self.outputs[idx].spec.fit;
                let upload_res = {
                    let gl = &self.renderer.as_ref().unwrap().gl;
                    let slot0 = &mut self.outputs[idx].slot0;
                    match &payload {
                        FramePayload::Rgba {
                            width,
                            height,
                            pixels,
                            ..
                        } => slot0.set_rgba(gl, *width, *height, fit, pixels),
                        FramePayload::Nv12 {
                            width,
                            height,
                            colorspace,
                            y_plane,
                            uv_plane,
                        } => {
                            slot0.set_nv12(gl, *width, *height, fit, *colorspace, y_plane, uv_plane)
                        }
                    }
                };
                mailbox.recycle(payload);

                if let Err(err) = upload_res {
                    tracing::error!("texture upload failed on {}: {err}", self.outputs[idx].name);
                    return;
                }

                let entry = &mut self.outputs[idx];
                if !entry.slot0_has_first_frame {
                    entry.slot0_has_first_frame = true;
                    if entry.waiting_target_frame {
                        entry.waiting_target_frame = false;
                        entry.in_transition = true;
                        entry.transition_started = Some(Instant::now());
                    }
                }
                entry.surface.dirty = true;
                self.render_if_ready(idx, qh);
            }
        }
    }

    fn reload_shader_file(&mut self, changed_path: &Path) {
        let mut to_reload = Vec::new();
        for (idx, entry) in self.outputs.iter().enumerate() {
            if let Some(ref p) = entry.custom_shader_path
                && (p == changed_path || p.file_name() == changed_path.file_name())
            {
                to_reload.push((idx, p.clone()));
            }
        }
        if to_reload.is_empty() {
            return;
        }
        if let Err(err) = self.ensure_any_gl_context() {
            tracing::warn!("deferring shader reload (no GL context yet): {err}");
            return;
        }
        let Some(ref renderer) = self.renderer else {
            return;
        };
        let gl = &renderer.gl;
        for (idx, path) in to_reload {
            let Ok(src) = fs::read_to_string(&path) else {
                continue;
            };
            match GlProgram::compile(gl, &src) {
                Ok(new_prog) => {
                    let entry = &mut self.outputs[idx];
                    if let Some(old) = entry.custom_program.replace(new_prog) {
                        old.destroy(gl);
                    }
                    entry.last_shader_error = None;
                    entry.surface.dirty = true;
                    tracing::info!("hot-reloaded shader {} on {}", path.display(), entry.name);
                }
                Err(err) => {
                    let entry = &mut self.outputs[idx];
                    tracing::warn!(
                        "shader hot-reload compile failed on {} (keeping active shader): {err}",
                        entry.name
                    );
                    entry.last_shader_error = Some(err.to_string());
                }
            }
        }
    }

    fn render_if_ready(
        &mut self,
        idx: usize,
        qh: &wayland_client::QueueHandle<WaylandContext<Self>>,
    ) {
        let entry = &self.outputs[idx];
        if !entry.surface.configured || entry.surface.frame_pending {
            return;
        }
        let paused = self.pause.is_paused(&entry.name);
        let continuous = !paused && entry.uses_continuous_frames();
        if !entry.surface.dirty && !continuous {
            return;
        }

        if let Err(err) = self.ensure_renderer_on_surface(idx) {
            tracing::warn!("EGL make_current failed on render: {err}");
            return;
        }

        let now = Instant::now();
        let mut promote_done = false;
        let mut delta = 0.0f32;
        let mut progress = 1.0f32;

        {
            let entry = &mut self.outputs[idx];
            if !paused {
                delta = now
                    .saturating_duration_since(entry.last_tick)
                    .as_secs_f32()
                    .clamp(0.0, 0.1);
                entry.anim_time_secs += delta;
                entry.frame_index = entry.frame_index.wrapping_add(1);
            }
            entry.last_tick = now;

            if entry.in_transition {
                let dur = Duration::from_millis(u64::from(entry.spec.duration_ms.max(1)));
                let elapsed = entry
                    .transition_started
                    .map_or(dur, |start| now.saturating_duration_since(start));
                let linear = (elapsed.as_secs_f32() / dur.as_secs_f32()).clamp(0.0, 1.0);
                // Smoothstep/cubic ease
                progress = linear * linear * (3.0 - 2.0 * linear);
                if linear >= 1.0 {
                    progress = 1.0;
                    promote_done = true;
                }
            }
        }

        let draw_res = {
            let renderer = self.renderer.as_mut().unwrap();
            let entry = &self.outputs[idx];
            let draw_params = DrawParams {
                width: entry.surface.phys_w,
                height: entry.surface.phys_h,
                time_secs: entry.anim_time_secs,
                delta_secs: delta,
                progress,
                frame_index: entry.frame_index,
                params: entry.spec.params,
                output_index: entry.index,
                output_name: &entry.name,
                slot0: &entry.slot0,
                slot1: entry.slot1.as_ref(),
                custom_program: entry.custom_program.as_ref(),
                transition_program: entry.transition_program.as_ref(),
                in_transition: entry.in_transition,
            };
            renderer.draw(&draw_params)
        };

        if let Err(err) = draw_res {
            tracing::error!("GL draw failed on {}: {err}", self.outputs[idx].name);
            return;
        }

        if promote_done {
            let gl = &self.renderer.as_ref().unwrap().gl;
            let entry = &mut self.outputs[idx];
            entry.in_transition = false;
            entry.transition_started = None;
            if let Some(prev_job) = entry.prev_job.take() {
                prev_job.stop();
            }
            if let Some(s1) = entry.slot1.take() {
                s1.destroy(gl);
            }
            if let Some(tr) = entry.transition_program.take() {
                tr.destroy(gl);
            }
        }

        let entry = &mut self.outputs[idx];
        entry.surface.dirty = false;
        let keep_animating = !paused && entry.uses_continuous_frames();
        if keep_animating {
            // Request wl_surface.frame BEFORE eglSwapBuffers so the callback attaches atomically to this commit.
            entry.surface.request_frame_if_idle(qh);
        }

        if let Some(egl_surf) = entry.surface.egl_surface
            && let Err(err) = self.egl.swap_buffers(egl_surf)
        {
            tracing::warn!("eglSwapBuffers failed on {}: {err}", entry.name);
        }
    }

    fn handle_ipc(
        &mut self,
        req: IpcRequest,
        qh: &wayland_client::QueueHandle<WaylandContext<Self>>,
    ) -> IpcResponse {
        match req {
            IpcRequest::Set {
                output,
                target,
                fit,
                transition,
                duration_ms,
                shader,
                fps,
                hwaccel,
            } => {
                let targets = self.find_outputs_matching(&output);
                if targets.is_empty() {
                    return IpcResponse::err(format!(
                        "no connected Wayland output matches '{output}'"
                    ));
                }
                for idx in targets {
                    if let Err(err) = self.apply_set_to_output(
                        idx,
                        &target,
                        fit,
                        transition.clone(),
                        duration_ms,
                        shader.clone(),
                        fps,
                        hwaccel,
                        true,
                        qh,
                    ) {
                        return IpcResponse::err(format!("{err:#}"));
                    }
                }
                IpcResponse::ok_msg("updated wallpaper")
            }
            IpcRequest::Shader { output, path } => {
                let targets = self.find_outputs_matching(&output);
                if targets.is_empty() {
                    return IpcResponse::err(format!(
                        "no connected Wayland output matches '{output}'"
                    ));
                }
                for idx in targets {
                    if let Err(err) = self.apply_shader_to_output(idx, &path, true, qh) {
                        return IpcResponse::err(format!("{err:#}"));
                    }
                }
                IpcResponse::ok_msg("updated shader")
            }
            IpcRequest::Params { output, values } => {
                let targets = self.find_outputs_matching(&output);
                if targets.is_empty() {
                    return IpcResponse::err(format!(
                        "no connected Wayland output matches '{output}'"
                    ));
                }
                for idx in targets {
                    let entry = &mut self.outputs[idx];
                    entry.spec.params = values;
                    entry.surface.dirty = true;
                    self.persisted
                        .outputs
                        .insert(entry.name.clone(), entry.spec.clone());
                    self.render_if_ready(idx, qh);
                }
                self.save_state();
                IpcResponse::ok_msg("updated u_params")
            }
            IpcRequest::Pause { output, reason } => {
                let out_arg = (!output.is_empty() && output != "*").then_some(output.as_str());
                self.pause.add(out_arg, &reason);
                self.sync_pause_to_workers();
                IpcResponse::ok_data(json!({
                    "paused_global": self.pause.global_reasons(),
                }))
            }
            IpcRequest::Resume { output, reason } => {
                let out_arg = (!output.is_empty() && output != "*").then_some(output.as_str());
                self.pause.remove(out_arg, &reason);
                self.sync_pause_to_workers();
                let indices: Vec<usize> = (0..self.outputs.len()).collect();
                for idx in indices {
                    self.outputs[idx].last_tick = Instant::now();
                    self.outputs[idx].surface.dirty = true;
                    self.render_if_ready(idx, qh);
                }
                IpcResponse::ok_data(json!({
                    "paused_global": self.pause.global_reasons(),
                }))
            }
            IpcRequest::Toggle { output } => {
                let out_arg = (!output.is_empty() && output != "*").then_some(output.as_str());
                let paused_now = self.pause.toggle_manual(out_arg);
                self.sync_pause_to_workers();
                if !paused_now {
                    let indices: Vec<usize> = (0..self.outputs.len()).collect();
                    for idx in indices {
                        self.outputs[idx].last_tick = Instant::now();
                        self.outputs[idx].surface.dirty = true;
                        self.render_if_ready(idx, qh);
                    }
                }
                IpcResponse::ok_data(json!({
                    "manual_paused": paused_now,
                    "paused_global": self.pause.global_reasons(),
                }))
            }
            IpcRequest::Reload { output } => {
                let targets = self.find_outputs_matching(&output);
                if targets.is_empty() {
                    return IpcResponse::err(format!(
                        "no connected Wayland output matches '{output}'"
                    ));
                }
                for idx in targets {
                    let spec = self.outputs[idx].spec.clone();
                    if let Some(target) = spec.target.clone() {
                        if let Err(err) = self.apply_set_to_output(
                            idx,
                            &target,
                            Some(spec.fit),
                            Some("none".to_string()),
                            Some(0),
                            spec.shader.clone(),
                            spec.fps,
                            Some(spec.hwaccel),
                            false,
                            qh,
                        ) {
                            return IpcResponse::err(format!("{err:#}"));
                        }
                    } else if let Some(sh) = spec.shader.clone()
                        && let Err(err) = self.apply_shader_to_output(idx, &sh, false, qh)
                    {
                        return IpcResponse::err(format!("{err:#}"));
                    }
                }
                IpcResponse::ok_msg("reloaded")
            }
            IpcRequest::Outputs => {
                let list: Vec<_> = self
                    .outputs
                    .iter()
                    .map(|o| {
                        json!({
                            "name": o.name,
                            "index": o.index,
                            "logical_size": [o.surface.logical_w, o.surface.logical_h],
                            "physical_size": [o.surface.phys_w, o.surface.phys_h],
                            "scale": f64::from(o.surface.scale_120) / 120.0,
                            "configured": o.surface.configured,
                        })
                    })
                    .collect();
                IpcResponse::ok_data(json!({ "outputs": list }))
            }
            IpcRequest::Status => {
                let outputs: Vec<_> = self
                    .outputs
                    .iter()
                    .map(|o| {
                        json!({
                            "name": o.name,
                            "logical_size": [o.surface.logical_w, o.surface.logical_h],
                            "physical_size": [o.surface.phys_w, o.surface.phys_h],
                            "scale": f64::from(o.surface.scale_120) / 120.0,
                            "configured": o.surface.configured,
                            "target": o.spec.target,
                            "shader": o.spec.shader,
                            "fit": o.spec.fit,
                            "params": o.spec.params,
                            "paused": self.pause.is_paused(&o.name),
                            "pause_reasons": self.pause.reasons_for(&o.name),
                            "continuous": o.uses_continuous_frames(),
                            "shader_mode": o.custom_program.as_ref().map(|p| format!("{:?}", p.assembled.mode)),
                            "last_shader_error": o.last_shader_error,
                            "last_media_error": o.last_media_error,
                        })
                    })
                    .collect();
                IpcResponse::ok_data(json!({
                    "auto_pause_fullscreen": self.config.auto_pause_fullscreen,
                    "global_pause_reasons": self.pause.global_reasons(),
                    "outputs": outputs,
                }))
            }
            IpcRequest::Stop => {
                self.should_exit = true;
                IpcResponse::ok_msg("stopping glwall daemon")
            }
        }
    }

    fn teardown_output_entry(&mut self, mut entry: OutputEntry) {
        if let Some(job) = entry.job.take() {
            job.stop();
        }
        if let Some(prev) = entry.prev_job.take() {
            prev.stop();
        }
        self.egl.make_current_surfaceless();
        if let Some(ref mut r) = self.renderer {
            r.remove_output_cache(&entry.name);
            entry.slot0.destroy(&r.gl);
            if let Some(s1) = entry.slot1.take() {
                s1.destroy(&r.gl);
            }
            if let Some(p) = entry.custom_program.take() {
                p.destroy(&r.gl);
            }
            if let Some(tr) = entry.transition_program.take() {
                tr.destroy(&r.gl);
            }
        }
        entry.surface.teardown(&self.egl);
    }

    fn shutdown(&mut self) {
        let entries = std::mem::take(&mut self.outputs);
        for entry in entries {
            self.teardown_output_entry(entry);
        }
        if let Some(r) = self.renderer.take() {
            self.egl.make_current_surfaceless();
            r.destroy();
        }
        self.egl.unbind_context();
    }
}

impl WaylandAppSink for AppCore {
    fn on_output_added(&mut self, handles: &WaylandHandles<'_, Self>, output: wl_output::WlOutput) {
        let info = handles.output_state.info(&output);
        let name = info
            .as_ref()
            .and_then(|i| i.name.clone())
            .unwrap_or_else(|| format!("wl-output-{}", output.id().protocol_id()));
        let name_resolved = info.as_ref().and_then(|i| i.name.as_ref()).is_some();
        let initial_logical = info
            .as_ref()
            .and_then(|i| i.logical_size)
            .map(|(w, h)| (w.max(1) as u32, h.max(1) as u32))
            .unwrap_or((1920, 1080));
        let initial_scale = info.as_ref().map_or(1, |i| i.scale_factor);

        let surface = SurfaceHandle::create(handles, &output, initial_logical, initial_scale);
        let index = self.next_output_index;
        self.next_output_index += 1;

        let spec = self.config.base_spec_for(&name);
        self.outputs.push(OutputEntry {
            index,
            wl_output: output,
            name,
            name_resolved,
            surface,
            spec,
            generation: 0,
            slot0: MediaSlotGpu::empty(),
            slot1: None,
            slot0_has_first_frame: false,
            job: None,
            prev_job: None,
            custom_program: None,
            custom_shader_path: None,
            last_shader_error: None,
            last_media_error: None,
            transition_program: None,
            transition_started: None,
            waiting_target_frame: false,
            in_transition: false,
            anim_time_secs: 0.0,
            last_tick: Instant::now(),
            frame_index: 0,
        });

        let idx = self.outputs.len() - 1;
        if self.outputs[idx].name_resolved {
            self.apply_initial_spec_if_needed(idx, handles.qh);
        }
    }

    fn on_output_updated(
        &mut self,
        handles: &WaylandHandles<'_, Self>,
        output: &wl_output::WlOutput,
    ) {
        let Some(idx) = self.outputs.iter().position(|o| &o.wl_output == output) else {
            return;
        };
        let Some(info) = handles.output_state.info(output) else {
            return;
        };
        if let Some(real_name) = info.name {
            let changed_name =
                !self.outputs[idx].name_resolved || self.outputs[idx].name != real_name;
            self.outputs[idx].name = real_name;
            self.outputs[idx].name_resolved = true;
            if changed_name {
                self.apply_initial_spec_if_needed(idx, handles.qh);
            }
        }
    }

    fn on_output_removed(
        &mut self,
        _handles: &WaylandHandles<'_, Self>,
        output: &wl_output::WlOutput,
    ) {
        let Some(idx) = self.outputs.iter().position(|o| &o.wl_output == output) else {
            return;
        };
        let entry = self.outputs.remove(idx);
        self.pause.drop_output(&entry.name);
        self.fullscreen_outputs.remove(&entry.name);
        self.teardown_output_entry(entry);
        self.sync_shader_watcher();
    }

    fn on_surface_configure(
        &mut self,
        handles: &WaylandHandles<'_, Self>,
        surface: &wl_surface::WlSurface,
        logical_w: u32,
        logical_h: u32,
    ) {
        let Some(idx) = self.find_output_by_surface(surface) else {
            return;
        };
        let entry = &mut self.outputs[idx];
        let scale_120 = entry.surface.scale_120;
        let next_w = if logical_w > 0 {
            logical_w
        } else {
            entry.surface.logical_w
        };
        let next_h = if logical_h > 0 {
            logical_h
        } else {
            entry.surface.logical_h
        };
        entry.surface.configured = true;
        entry.surface.update_geometry(next_w, next_h, scale_120);
        if !entry.name_resolved
            && let Some(info) = handles.output_state.info(&entry.wl_output)
            && let Some(real_name) = info.name
        {
            entry.name = real_name;
            entry.name_resolved = true;
        }
        let need_apply =
            entry.name_resolved && entry.job.is_none() && entry.custom_program.is_none();
        if need_apply {
            self.apply_initial_spec_if_needed(idx, handles.qh);
        }
        self.render_if_ready(idx, handles.qh);
    }

    fn on_surface_scale_120(&mut self, surface: &wl_surface::WlSurface, scale_120: u32) {
        let Some(idx) = self.find_output_by_surface(surface) else {
            return;
        };
        let entry = &mut self.outputs[idx];
        let lw = entry.surface.logical_w;
        let lh = entry.surface.logical_h;
        entry.surface.update_geometry(lw, lh, scale_120);
    }

    fn on_surface_closed(&mut self, surface: &wl_surface::WlSurface) {
        let Some(idx) = self.find_output_by_surface(surface) else {
            return;
        };
        let entry = self.outputs.remove(idx);
        self.teardown_output_entry(entry);
        self.sync_shader_watcher();
    }

    fn on_surface_frame(&mut self, surface: &wl_surface::WlSurface, _time_ms: u32) {
        let Some(idx) = self.find_output_by_surface(surface) else {
            return;
        };
        self.outputs[idx].surface.frame_pending = false;
        self.outputs[idx].surface.dirty = true;
        // Request next frame immediately in the main loop iteration after dispatch completes;
        // render_if_ready will run right inside WaylandSource post-dispatch or via queue handle.
    }

    fn on_fullscreen_outputs_changed(&mut self, next_fullscreen: HashSet<String>) {
        if !self.config.auto_pause_fullscreen {
            return;
        }
        let mut resumed = Vec::new();
        for old_name in &self.fullscreen_outputs {
            if !next_fullscreen.contains(old_name) {
                self.pause.remove(Some(old_name), "fullscreen");
                resumed.push(old_name.clone());
            }
        }
        for new_name in &next_fullscreen {
            if !self.fullscreen_outputs.contains(new_name) {
                self.pause.add(Some(new_name), "fullscreen");
            }
        }
        self.fullscreen_outputs = next_fullscreen;
        self.sync_pause_to_workers();

        for name in resumed {
            if let Some(entry) = self.outputs.iter_mut().find(|o| o.name == name) {
                entry.last_tick = Instant::now();
                entry.surface.dirty = true;
            }
        }
    }
}

pub fn run_daemon(
    config_path: Option<PathBuf>,
    socket_path: PathBuf,
    no_restore: bool,
) -> Result<()> {
    let config = Config::load_or_default(config_path.as_deref())?;
    let state_path = default_state_path();
    let persisted = if no_restore {
        PersistedState::default()
    } else if let Some(ref p) = state_path {
        PersistedState::load_or_default(p)
    } else {
        PersistedState::default()
    };

    let listener = bind_ipc_listener(&socket_path)?;

    let mut event_loop: EventLoop<'static, WaylandContext<AppCore>> =
        EventLoop::try_new().context("failed to initialize calloop event loop")?;
    let loop_handle = event_loop.handle();

    // Register Unix signal source BEFORE spawning any background decode threads so they inherit the blocked mask.
    let signals = Signals::new(&[Signal::SIGINT, Signal::SIGTERM])
        .context("failed to register SIGINT/SIGTERM calloop source")?;
    loop_handle
        .insert_source(signals, |_event, _, ctx| {
            ctx.sink.should_exit = true;
        })
        .map_err(|e| anyhow::anyhow!("failed to insert signal source: {e:?}"))?;

    let (media_tx, media_rx) = channel::channel::<MediaEvent>();
    let watcher = ShaderWatcher::insert_into_loop(&loop_handle, |ctx, path| {
        ctx.sink.reload_shader_file(path);
    })?;

    let (wl_conn, mut wl_queue, mut ctx) = connect_wayland_and_egl(
        config,
        state_path,
        persisted,
        !no_restore,
        media_tx,
        watcher,
    )?;

    // Process initial globals, output info, and initial layer configures before entering loop.
    wl_queue
        .roundtrip(&mut ctx)
        .context("initial Wayland roundtrip failed")?;
    wl_queue
        .roundtrip(&mut ctx)
        .context("secondary Wayland roundtrip failed")?;

    WaylandSource::new(wl_conn.clone(), wl_queue)
        .insert(loop_handle.clone())
        .map_err(|e| anyhow::anyhow!("failed to insert WaylandSource into calloop: {e:?}"))?;

    loop_handle
        .insert_source(media_rx, |ev, _, ctx| {
            if let channel::Event::Msg(media_ev) = ev {
                let qh = ctx.qh.clone();
                ctx.sink.handle_media_event(media_ev, &qh);
            }
        })
        .map_err(|e| anyhow::anyhow!("failed to insert media channel into calloop: {e:?}"))?;

    loop_handle
        .insert_source(
            Generic::new(listener, Interest::READ, Mode::Level),
            |_readiness, listener_wrap, ctx| {
                loop {
                    match listener_wrap.accept() {
                        Ok((mut stream, _)) => {
                            let response = match read_ipc_request(&mut stream) {
                                Ok(req) => {
                                    let qh = ctx.qh.clone();
                                    ctx.sink.handle_ipc(req, &qh)
                                }
                                Err(err) => IpcResponse::err(format!("{err:#}")),
                            };
                            if let Err(err) = write_ipc_response(&mut stream, &response) {
                                tracing::warn!("failed writing IPC response: {err}");
                            }
                            let _ = stream.shutdown(std::net::Shutdown::Both);
                        }
                        Err(err) if err.kind() == ErrorKind::WouldBlock => break,
                        Err(err) => {
                            tracing::warn!("IPC accept error: {err}");
                            break;
                        }
                    }
                }
                Ok(PostAction::Continue)
            },
        )
        .map_err(|e| anyhow::anyhow!("failed to insert IPC listener into calloop: {e:?}"))?;

    // Render any outputs configured during initial roundtrip.
    for idx in 0..ctx.sink.outputs.len() {
        let qh = ctx.qh.clone();
        ctx.sink.render_if_ready(idx, &qh);
    }
    let _ = wl_conn.flush();

    tracing::info!(
        "glwall daemon ready on {} ({} initial output(s))",
        socket_path.display(),
        ctx.sink.outputs.len()
    );

    while !ctx.sink.should_exit {
        event_loop
            .dispatch(None, &mut ctx)
            .context("calloop dispatch error")?;

        let qh = ctx.qh.clone();
        for idx in 0..ctx.sink.outputs.len() {
            ctx.sink.render_if_ready(idx, &qh);
        }
        if let Err(err) = wl_conn.flush() {
            tracing::warn!("Wayland flush failed: {err}");
        }
    }

    tracing::info!("shutting down glwall daemon cleanly");
    ctx.sink.shutdown();
    drop(ctx);
    let _ = fs::remove_file(&socket_path);
    Ok(())
}
