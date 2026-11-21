use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use calloop::channel::{self, Sender};
use calloop::generic::Generic;
use calloop::signals::{Signal, Signals};
use calloop::timer::{TimeoutAction, Timer};
use calloop::{EventLoop, Interest, LoopHandle, Mode, PostAction};
use serde_json::json;
use smithay_client_toolkit::reexports::calloop_wayland_source::WaylandSource;
use smithay_client_toolkit::shell::WaylandSurface;
use wayland_client::Proxy;
use wayland_client::protocol::{wl_output, wl_surface};

use crate::config::{Config, OutputSpec, PersistedState, default_state_path};
use crate::gpu::egl::EglState;
use crate::gpu::renderer::{
    DrawParams, GlProgram, MediaSlotGpu, Renderer, SlotMediaKind, TextureId, TextureKey,
    TexturePool,
};
use crate::gpu::shader::resolve_transition_source;
use crate::gpu::uniforms::{TextureSpec, UniformValue};
use crate::ipc::{
    ApplyMedia, ApplyTransition, DispatchOutcome, IpcClientId, IpcEvent, IpcHub, IpcRequest,
    IpcResponse, bind_ipc_listener,
};
use crate::media::{FramePayload, MediaEvent, MediaJob, MediaKind, decode_texture_async};
use crate::pause::PauseState;
use crate::watcher::{ShaderWatcher, absolutize_path};
use crate::wayland::{SurfaceHandle, WaylandAppSink, WaylandContext, WaylandHandles};

const WAIT_COMMIT_TIMEOUT: Duration = Duration::from_secs(10);

struct OutputEntry {
    index: i32,
    wl_output: wl_output::WlOutput,
    name: String,
    name_resolved: bool,
    surface: SurfaceHandle,
    spec: OutputSpec,
    revision: u64,
    generation: u64,
    slot0: MediaSlotGpu,
    slot1: Option<MediaSlotGpu>,
    slot0_has_first_frame: bool,
    job: Option<MediaJob>,
    prev_job: Option<MediaJob>,
    custom_program: Option<GlProgram>,
    custom_shader_path: Option<PathBuf>,
    custom_textures: BTreeMap<String, TextureId>,
    custom_uniforms: BTreeMap<String, UniformValue>,
    last_shader_error: Option<String>,
    last_media_error: Option<String>,
    transition_program: Option<GlProgram>,
    transition_started: Option<Instant>,
    waiting_target_frame: bool,
    in_transition: bool,
    anim_time_secs: f32,
    last_tick: Instant,
    frame_index: i32,
    pending_apply: Option<PendingApply>,
}

impl OutputEntry {
    fn uses_continuous_frames(&self) -> bool {
        if self.in_transition {
            return true;
        }
        if self.job.as_ref().is_some_and(|j| j.kind.is_animated()) {
            return true;
        }
        self.custom_program.as_ref().is_some_and(|p| p.uses_time)
    }
}

enum StagedShader {
    Keep,
    Clear,
    Set {
        path: PathBuf,
        program: Box<GlProgram>,
    },
}

struct PendingApply {
    revision: u64,
    persist: bool,
    shader: StagedShader,
    texture_updates: BTreeMap<String, Option<TextureSpec>>,
    acquired_textures: HashMap<TextureKey, TextureId>,
    pending_keys: HashSet<TextureKey>,
    uniform_updates: BTreeMap<String, UniformValue>,
    params_update: Option<[f32; 4]>,
    target_override: Option<String>,
    clear_implicit_shader: bool,
    awaiting_media_gen: Option<u64>,
}

impl PendingApply {
    fn is_ready(&self, entry_gen: u64, slot0_has_first_frame: bool) -> bool {
        if !self.pending_keys.is_empty() {
            return false;
        }
        match self.awaiting_media_gen {
            Some(expected) => entry_gen != expected || slot0_has_first_frame,
            None => true,
        }
    }
}

fn frame_needs_render(
    configured: bool,
    frame_pending: bool,
    dirty: bool,
    continuous: bool,
    apply_pending: bool,
) -> bool {
    configured && !frame_pending && !apply_pending && (dirty || continuous)
}

struct CommitWaiter {
    client_id: IpcClientId,
    revision_targets: Vec<(String, u64)>,
    completed: HashSet<String>,
    max_revision: u64,
    deadline: Instant,
}

struct PreparedTransaction {
    next_revision: u64,
    shader: StagedShader,
    resolved_media: Option<(PathBuf, MediaKind, ApplyMedia)>,
    transition_spec: Option<(String, u32)>,
    transition_prog: Option<GlProgram>,
    texture_updates: BTreeMap<String, Option<TextureSpec>>,
    uniform_updates: BTreeMap<String, UniformValue>,
    params_update: Option<[f32; 4]>,
    clear_implicit_shader: bool,
}

#[derive(Default)]
struct TxParams {
    output: String,
    revision: Option<u64>,
    media: Option<ApplyMedia>,
    shader: Option<String>,
    textures: BTreeMap<String, Option<TextureSpec>>,
    uniforms: BTreeMap<String, UniformValue>,
    params: Option<[f32; 4]>,
    transition: Option<ApplyTransition>,
    persist_transition_defaults: bool,
    persist: bool,
    wait_commit: bool,
}

pub struct AppCore {
    config: Config,
    state_path: Option<PathBuf>,
    persisted: PersistedState,
    restore_on_connect: bool,
    persist_enabled: bool,
    egl: EglState,
    renderer: Option<Renderer>,
    texture_pool: TexturePool,
    inflight_tex_decodes: HashSet<TextureKey>,
    outputs: Vec<OutputEntry>,
    deferred_outputs: Vec<OutputEntry>,
    deferred_applies: Vec<PendingApply>,
    deferred_slots: Vec<MediaSlotGpu>,
    deferred_programs: Vec<GlProgram>,
    next_output_index: i32,
    pause: PauseState,
    fullscreen_outputs: HashSet<String>,
    media_tx: Sender<MediaEvent>,
    watcher: ShaderWatcher,
    ipc_hub: IpcHub,
    commit_waiters: Vec<CommitWaiter>,
    timeout_timer_armed: bool,
    loop_handle: LoopHandle<'static, WaylandContext<AppCore>>,
    should_exit: bool,
}

struct DaemonInit {
    config: Config,
    state_path: Option<PathBuf>,
    persisted: PersistedState,
    restore_on_connect: bool,
    persist_enabled: bool,
    media_tx: Sender<MediaEvent>,
    watcher: ShaderWatcher,
    loop_handle: LoopHandle<'static, WaylandContext<AppCore>>,
}

fn connect_wayland_and_egl(
    init: DaemonInit,
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
            config: init.config,
            state_path: init.state_path,
            persisted: init.persisted,
            restore_on_connect: init.restore_on_connect,
            persist_enabled: init.persist_enabled,
            egl,
            renderer,
            texture_pool: TexturePool::new(),
            inflight_tex_decodes: HashSet::new(),
            outputs: Vec::new(),
            deferred_outputs: Vec::new(),
            deferred_applies: Vec::new(),
            deferred_slots: Vec::new(),
            deferred_programs: Vec::new(),
            next_output_index: 0,
            pause: PauseState::new(),
            fullscreen_outputs: HashSet::new(),
            media_tx: init.media_tx,
            watcher: init.watcher,
            ipc_hub: IpcHub::new(),
            commit_waiters: Vec::new(),
            timeout_timer_armed: false,
            loop_handle: init.loop_handle,
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
        self.drain_deferred_gpu();
        Ok(true)
    }

    fn ensure_any_gl_context(&mut self) -> Result<()> {
        if self.renderer.is_some() && self.egl.make_current_surfaceless().is_ok() {
            self.drain_deferred_gpu();
            return Ok(());
        }
        for i in 0..self.outputs.len() {
            if matches!(self.ensure_renderer_on_surface(i), Ok(true)) {
                return Ok(());
            }
        }
        if self.egl.make_current_surfaceless().is_ok()
            && let Ok(gl) = self.egl.create_glow_context()
            && let Ok(r) = Renderer::new(gl)
        {
            self.renderer = Some(r);
            self.drain_deferred_gpu();
            return Ok(());
        }
        bail!("no Wayland EGL surface or surfaceless context available yet")
    }

    fn bind_cleanup_gl_context(&mut self) -> Result<()> {
        if self.egl.make_current_surfaceless().is_ok() {
            return Ok(());
        }
        for i in 0..self.outputs.len() {
            if matches!(self.ensure_renderer_on_surface(i), Ok(true)) {
                return Ok(());
            }
        }
        bail!("no EGL context available for GPU cleanup")
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
            // Keep superseded transition sources paused so they stop decoding offscreen frames.
            if let Some(prev) = &entry.prev_job {
                prev.set_paused(true);
            }
        }
    }

    fn save_state(&self) {
        if !self.persist_enabled {
            return;
        }
        if let Some(ref path) = self.state_path
            && let Err(err) = self.persisted.save_atomic(path)
        {
            tracing::warn!("failed to persist state: {err}");
        }
    }

    fn release_pending_apply(&mut self, pending: PendingApply, reason: Option<&str>) {
        if self.renderer.is_some()
            && let Err(err) = self.ensure_any_gl_context()
        {
            tracing::warn!("deferring staged GPU cleanup: {err:#}");
            self.deferred_applies.push(pending);
            return;
        }
        self.release_pending_bound(pending);
        let _ = reason;
    }

    fn release_pending_bound(&mut self, mut pending: PendingApply) {
        if let Some(r) = &self.renderer {
            if let StagedShader::Set { program, .. } = pending.shader {
                (*program).destroy(&r.gl);
            }
            for (_, id) in pending.acquired_textures.drain() {
                self.texture_pool.release(&r.gl, id);
            }
        }
    }

    fn release_output_gpu(&mut self, entry: &mut OutputEntry) {
        if let Some(pending) = entry.pending_apply.take() {
            self.release_pending_bound(pending);
        }
        if let Some(r) = &mut self.renderer {
            r.remove_output_cache(&entry.name);
            std::mem::replace(&mut entry.slot0, MediaSlotGpu::empty()).destroy(&r.gl);
            if let Some(slot) = entry.slot1.take() {
                slot.destroy(&r.gl);
            }
            if let Some(program) = entry.custom_program.take() {
                program.destroy(&r.gl);
            }
            if let Some(program) = entry.transition_program.take() {
                program.destroy(&r.gl);
            }
            for (_, id) in std::mem::take(&mut entry.custom_textures) {
                self.texture_pool.release(&r.gl, id);
            }
        }
    }

    fn drain_deferred_gpu(&mut self) {
        if self.renderer.is_none() {
            return;
        }
        for pending in std::mem::take(&mut self.deferred_applies) {
            self.release_pending_bound(pending);
        }
        for mut entry in std::mem::take(&mut self.deferred_outputs) {
            self.release_output_gpu(&mut entry);
        }
        let gl = &self.renderer.as_ref().unwrap().gl;
        for slot in self.deferred_slots.drain(..) {
            slot.destroy(gl);
        }
        for program in self.deferred_programs.drain(..) {
            program.destroy(gl);
        }
        crate::gpu::renderer::retry_retired_cuda(gl);
    }

    fn cancel_output_pending(&mut self, idx: usize, reason: &str) {
        let out_name = self.outputs[idx].name.clone();
        if let Some(pending) = self.outputs[idx].pending_apply.take() {
            self.release_pending_apply(pending, Some(reason));
        }
        self.fail_commit_waiters_for_output(&out_name, reason);
    }

    fn arm_commit_timeout_timer(&mut self) {
        if self.timeout_timer_armed {
            return;
        }
        self.timeout_timer_armed = true;
        let _ = self.loop_handle.insert_source(
            Timer::from_duration(Duration::from_secs(1)),
            |_, _, ctx| {
                ctx.sink.sweep_expired_waiters();
                if ctx.sink.commit_waiters.is_empty() {
                    ctx.sink.timeout_timer_armed = false;
                    TimeoutAction::Drop
                } else {
                    TimeoutAction::ToDuration(Duration::from_secs(1))
                }
            },
        );
    }

    fn sweep_expired_waiters(&mut self) {
        let now = Instant::now();
        let mut i = 0;
        while i < self.commit_waiters.len() {
            if now >= self.commit_waiters[i].deadline {
                let expired = self.commit_waiters.swap_remove(i);
                self.ipc_hub.complete_pending_ack(
                    expired.client_id,
                    &IpcResponse::err("timed out waiting for frame commit"),
                );
            } else {
                i += 1;
            }
        }
    }

    fn fail_commit_waiters_for_output(&mut self, output_name: &str, message: &str) {
        let mut i = 0;
        while i < self.commit_waiters.len() {
            let matches_output = self.commit_waiters[i]
                .revision_targets
                .iter()
                .any(|(o, _)| o == output_name)
                && !self.commit_waiters[i].completed.contains(output_name);
            if matches_output {
                let waiter = self.commit_waiters.swap_remove(i);
                self.ipc_hub.complete_pending_ack(
                    waiter.client_id,
                    &IpcResponse::err(format!("{output_name}: {message}")),
                );
            } else {
                i += 1;
            }
        }
    }

    fn resolve_commit_waiters_on_frame(&mut self, output_name: &str, revision: u64, frame: i32) {
        let mut i = 0;
        while i < self.commit_waiters.len() {
            let mut matched = false;
            if let Some((_, target_rev)) = self.commit_waiters[i]
                .revision_targets
                .iter()
                .find(|(o, _)| o == output_name)
                && revision >= *target_rev
            {
                self.commit_waiters[i]
                    .completed
                    .insert(output_name.to_string());
                matched = true;
            }
            if matched
                && self.commit_waiters[i].completed.len()
                    == self.commit_waiters[i].revision_targets.len()
            {
                let waiter = self.commit_waiters.swap_remove(i);
                let outputs: Vec<&str> = waiter
                    .revision_targets
                    .iter()
                    .map(|(n, _)| n.as_str())
                    .collect();
                let resp = IpcResponse::ok_data(json!({
                    "revision": waiter.max_revision,
                    "frame": frame,
                    "outputs": outputs,
                }));
                self.ipc_hub.complete_pending_ack(waiter.client_id, &resp);
            } else {
                i += 1;
            }
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
        let has_initial_bindings = !desired.textures.is_empty() || !desired.uniforms.is_empty();
        if desired.target == current_target
            && desired.shader == current_shader
            && !has_initial_bindings
        {
            return;
        }

        let target = desired.target.clone();
        let shader = desired.shader.clone();
        let textures: BTreeMap<String, Option<TextureSpec>> = desired
            .textures
            .iter()
            .map(|(k, v)| (k.clone(), Some(v.clone())))
            .collect();
        let uniforms = desired.uniforms.clone();
        let params = desired.params;
        let fit = desired.fit;
        let fps = desired.fps;
        let hwaccel = desired.hwaccel;
        let transition = desired.transition.clone();
        let duration_ms = desired.duration_ms;

        self.outputs[idx].spec = desired;

        let tx = TxParams {
            output: out_name,
            revision: None,
            media: target.map(|t| ApplyMedia {
                target: t,
                fit: Some(fit),
                fps,
                hwaccel: Some(hwaccel),
            }),
            shader,
            textures,
            uniforms,
            params: Some(params),
            transition: Some(ApplyTransition {
                name: Some(transition),
                duration_ms: Some(duration_ms),
            }),
            persist_transition_defaults: true,
            persist: false,
            wait_commit: false,
        };
        if let Err(err) = self.execute_transaction(tx, None, qh) {
            tracing::warn!(
                "failed restoring initial spec on {}: {err:#}",
                self.outputs[idx].name
            );
        }
    }

    fn execute_transaction(
        &mut self,
        tx: TxParams,
        waiter_client: Option<IpcClientId>,
        qh: &wayland_client::QueueHandle<WaylandContext<Self>>,
    ) -> Result<DispatchOutcome> {
        let targets = self.find_outputs_matching(&tx.output);
        if targets.is_empty() {
            bail!("no connected Wayland output matches '{}'", tx.output);
        }

        for (name, val) in &tx.uniforms {
            if !val.is_finite() {
                bail!("uniform '{name}' has non-finite value");
            }
        }
        if self.renderer.is_some() && tx.media.is_some() {
            self.ensure_any_gl_context()?;
        }

        // Pre-validate and compile everything per-output before mutating any live state.
        let mut prepared: Vec<(usize, PreparedTransaction)> = Vec::with_capacity(targets.len());
        let preflight_res: Result<()> = (|| {
            for &idx in &targets {
                let entry = &self.outputs[idx];
                let next_revision = match tx.revision {
                    Some(r) => {
                        if r < entry.revision {
                            bail!(
                                "non-monotonic revision {r} < {} on {}",
                                entry.revision,
                                entry.name
                            );
                        }
                        r
                    }
                    None => entry.revision + 1,
                };

                let mut resolved_media = None;
                let mut shader_mode_request = tx.shader.clone();
                let mut clear_implicit_shader = false;

                if let Some(ref media) = tx.media {
                    let path = absolutize_path(Path::new(&media.target));
                    let kind = MediaKind::from_path(&path).ok_or_else(|| {
                        anyhow::anyhow!("unsupported wallpaper file type: {}", path.display())
                    })?;
                    if !path.is_file() {
                        bail!("wallpaper file does not exist: {}", path.display());
                    }
                    if kind == MediaKind::ProceduralShader {
                        shader_mode_request = Some(path.to_string_lossy().to_string());
                    } else {
                        if shader_mode_request.is_none()
                            && entry
                                .spec
                                .target
                                .as_deref()
                                .and_then(|t| MediaKind::from_path(Path::new(t)))
                                == Some(MediaKind::ProceduralShader)
                        {
                            clear_implicit_shader = true;
                        }
                        resolved_media = Some((path, kind, media.clone()));
                    }
                }

                let shader_stage = if let Some(ref raw_sh) = shader_mode_request {
                    let trimmed = raw_sh.trim();
                    if trimmed.is_empty() || trimmed == "none" {
                        StagedShader::Clear
                    } else {
                        let resolved_sh = absolutize_path(Path::new(trimmed));
                        let src = fs::read_to_string(&resolved_sh).with_context(|| {
                            format!("failed to read shader {}", resolved_sh.display())
                        })?;
                        self.ensure_any_gl_context()?;
                        let gl = &self
                            .renderer
                            .as_ref()
                            .ok_or_else(|| anyhow::anyhow!("renderer not initialized"))?
                            .gl;
                        let prog = GlProgram::compile(gl, &src)?;
                        StagedShader::Set {
                            path: resolved_sh,
                            program: Box::new(prog),
                        }
                    }
                } else if clear_implicit_shader {
                    StagedShader::Clear
                } else {
                    StagedShader::Keep
                };

                // Validate uniforms against whichever shader will be active after this apply.
                let active_prog_ref = match &shader_stage {
                    StagedShader::Set { program, .. } => Some(program.as_ref()),
                    StagedShader::Clear => None,
                    StagedShader::Keep => self.outputs[idx].custom_program.as_ref(),
                };
                if let Some(prog) = active_prog_ref {
                    for (name, val) in &tx.uniforms {
                        prog.validate_uniform(name, val)?;
                    }
                }

                // Validate texture file paths before mutating any state.
                let mut normalized_tex = BTreeMap::new();
                for (name, opt_spec) in &tx.textures {
                    let clean_name = name.trim();
                    if clean_name.is_empty() {
                        bail!("empty texture uniform name");
                    }
                    if let Some(spec) = opt_spec {
                        let abs_path = absolutize_path(&spec.path);
                        if !abs_path.is_file() {
                            bail!(
                                "texture '{clean_name}' file does not exist: {}",
                                abs_path.display()
                            );
                        }
                        normalized_tex.insert(
                            clean_name.to_string(),
                            Some(TextureSpec {
                                path: abs_path,
                                filter: spec.filter,
                                wrap: spec.wrap,
                                flip_y: spec.flip_y,
                            }),
                        );
                    } else {
                        normalized_tex.insert(clean_name.to_string(), None);
                    }
                }

                // Pre-compile transition shader when switching to non-procedural media.
                let entry = &self.outputs[idx];
                let transition_spec = tx.transition.as_ref().map(|tr| {
                    (
                        tr.name
                            .clone()
                            .unwrap_or_else(|| entry.spec.transition.clone()),
                        tr.duration_ms.unwrap_or(entry.spec.duration_ms),
                    )
                });
                let (tr_name, tr_dur) = transition_spec
                    .clone()
                    .unwrap_or_else(|| (entry.spec.transition.clone(), entry.spec.duration_ms));

                let transition_prog = if resolved_media.is_some()
                    && entry.slot0.kind != SlotMediaKind::None
                    && tr_dur > 0
                    && tr_name != "none"
                {
                    if let Some(tr_src) = resolve_transition_source(&tr_name)? {
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

                prepared.push((
                    idx,
                    PreparedTransaction {
                        next_revision,
                        shader: shader_stage,
                        resolved_media,
                        transition_spec,
                        transition_prog,
                        texture_updates: normalized_tex,
                        uniform_updates: tx.uniforms.clone(),
                        params_update: tx.params,
                        clear_implicit_shader,
                    },
                ));
            }
            Ok(())
        })();

        if let Err(err) = preflight_res {
            if let Some(ref r) = self.renderer {
                for (_, prep) in prepared {
                    if let StagedShader::Set { program, .. } = prep.shader {
                        (*program).destroy(&r.gl);
                    }
                    if let Some(tr) = prep.transition_prog {
                        tr.destroy(&r.gl);
                    }
                }
            }
            return Err(err);
        }

        // Pre-flight succeeded! Now apply media jobs & stage atomic visual updates.
        let mut wait_targets: Vec<(String, u64)> = Vec::new();
        let mut max_rev = 0u64;
        let mut keys_to_spawn: Vec<TextureKey> = Vec::new();

        for (idx, prep) in prepared {
            self.cancel_output_pending(idx, "superseded by newer apply");

            let is_standalone_shader = tx.media.as_ref().is_some_and(|m| {
                let p = absolutize_path(Path::new(&m.target));
                MediaKind::from_path(&p) == Some(MediaKind::ProceduralShader)
            });

            if let Some((tr_name, tr_dur)) = prep.transition_spec
                && tx.persist_transition_defaults
            {
                let entry = &mut self.outputs[idx];
                entry.spec.transition = tr_name;
                entry.spec.duration_ms = tr_dur;
            }

            let mut target_override: Option<String> = None;
            let mut awaiting_media_gen: Option<u64> = None;

            if is_standalone_shader {
                self.ensure_any_gl_context()?;
                let gl = &self.renderer.as_ref().unwrap().gl;
                let entry = &mut self.outputs[idx];
                entry.slot0.try_clear(gl)?;
                entry.generation += 1;
                if let Some(job) = entry.job.take() {
                    job.stop();
                }
                if let Some(prev) = entry.prev_job.take() {
                    prev.stop();
                }
                if let Some(s1) = entry.slot1.take() {
                    s1.destroy(gl);
                }
                if let Some(tr) = entry.transition_program.take() {
                    tr.destroy(gl);
                }
                entry.slot0_has_first_frame = false;
                entry.waiting_target_frame = false;
                entry.in_transition = false;
                entry.transition_started = None;
                entry.last_media_error = None;
                if let Some(ref m) = tx.media {
                    if let Some(f) = m.fit {
                        entry.spec.fit = f;
                    }
                    if m.fps.is_some() {
                        entry.spec.fps = m.fps;
                    }
                    if let Some(hw) = m.hwaccel {
                        entry.spec.hwaccel = hw;
                    }
                    let abs = absolutize_path(Path::new(&m.target));
                    target_override = Some(abs.to_string_lossy().to_string());
                }
            } else if let Some((resolved_target, kind, media)) = prep.resolved_media {
                if prep.transition_prog.is_none() {
                    if let Some(r) = &self.renderer {
                        self.outputs[idx].slot0.try_clear(&r.gl)?;
                    } else {
                        self.outputs[idx].slot0 = MediaSlotGpu::empty();
                    }
                }
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

                if let Some(f) = media.fit {
                    entry.spec.fit = f;
                }
                if media.fps.is_some() {
                    entry.spec.fps = media.fps;
                }
                if let Some(hw) = media.hwaccel {
                    entry.spec.hwaccel = hw;
                }
                target_override = Some(resolved_target.to_string_lossy().to_string());
                entry.last_media_error = None;

                if prep.transition_prog.is_some() {
                    let old_slot0 = std::mem::replace(&mut entry.slot0, MediaSlotGpu::empty());
                    entry.slot1 = Some(old_slot0);
                    entry.prev_job = entry.job.take();
                    if let Some(ref prev) = entry.prev_job {
                        prev.set_paused(true);
                    }
                    entry.transition_program = prep.transition_prog;
                    entry.waiting_target_frame = true;
                    entry.in_transition = false;
                    entry.transition_started = None;
                } else {
                    if let Some(old_job) = entry.job.take() {
                        old_job.stop();
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
                awaiting_media_gen = Some(generation);
            } else if let Some(tr) = prep.transition_prog
                && let Some(ref r) = self.renderer
            {
                tr.destroy(&r.gl);
            }

            let mut acquired_textures = HashMap::new();
            let mut pending_keys = HashSet::new();
            for spec in prep.texture_updates.values().flatten() {
                let key = TextureKey::from_spec(spec);
                if acquired_textures.contains_key(&key) || pending_keys.contains(&key) {
                    continue;
                }
                if let Some(id) = self.texture_pool.acquire(&key) {
                    acquired_textures.insert(key, id);
                } else if pending_keys.insert(key.clone())
                    && self.inflight_tex_decodes.insert(key.clone())
                {
                    keys_to_spawn.push(key);
                }
            }

            let out_name = self.outputs[idx].name.clone();
            max_rev = max_rev.max(prep.next_revision);
            wait_targets.push((out_name, prep.next_revision));

            self.outputs[idx].pending_apply = Some(PendingApply {
                revision: prep.next_revision,
                persist: tx.persist,
                shader: prep.shader,
                texture_updates: prep.texture_updates,
                acquired_textures,
                pending_keys,
                uniform_updates: prep.uniform_updates,
                params_update: prep.params_update,
                target_override,
                clear_implicit_shader: prep.clear_implicit_shader,
                awaiting_media_gen,
            });

            self.try_commit_pending_apply(idx, qh);
        }

        for key in keys_to_spawn {
            decode_texture_async(key, self.media_tx.clone());
        }

        if tx.wait_commit
            && let Some(client_id) = waiter_client
        {
            // If every target output has already swapped synchronously inside try_commit_pending_apply,
            // wait_commit resolves immediately; otherwise track it asynchronously.
            let mut completed = HashSet::new();
            let mut last_frame = 0i32;
            for (name, req_rev) in &wait_targets {
                if let Some(entry) = self.outputs.iter().find(|o| &o.name == name)
                    && entry.pending_apply.is_none()
                    && entry.revision >= *req_rev
                    && !entry.surface.dirty
                {
                    completed.insert(name.clone());
                    last_frame = entry.frame_index;
                }
            }
            if completed.len() == wait_targets.len() {
                let outs: Vec<&str> = wait_targets.iter().map(|(n, _)| n.as_str()).collect();
                return Ok(DispatchOutcome::Reply(IpcResponse::ok_data(json!({
                    "revision": max_rev,
                    "frame": last_frame,
                    "outputs": outs,
                }))));
            }
            self.ipc_hub.retain_pending_ack(client_id);
            self.commit_waiters.push(CommitWaiter {
                client_id,
                revision_targets: wait_targets,
                completed,
                max_revision: max_rev,
                deadline: Instant::now() + WAIT_COMMIT_TIMEOUT,
            });
            self.arm_commit_timeout_timer();
            return Ok(DispatchOutcome::Deferred);
        }

        Ok(DispatchOutcome::Reply(IpcResponse::ok_data(json!({
            "revision": max_rev,
        }))))
    }

    fn try_commit_pending_apply(
        &mut self,
        idx: usize,
        qh: &wayland_client::QueueHandle<WaylandContext<Self>>,
    ) {
        let ready = {
            let entry = &self.outputs[idx];
            entry
                .pending_apply
                .as_ref()
                .is_some_and(|p| p.is_ready(entry.generation, entry.slot0_has_first_frame))
        };
        if !ready {
            return;
        }

        if self.renderer.is_some()
            && let Err(err) = self.ensure_any_gl_context()
        {
            tracing::warn!("deferring pending apply until EGL context is available: {err:#}");
            return;
        }

        let mut pending = self.outputs[idx].pending_apply.take().unwrap();

        let (shader_changed, persist_now, out_name, spec_snapshot) = {
            let gl_opt = self.renderer.as_ref().map(|r| &r.gl);
            let entry = &mut self.outputs[idx];
            let mut sh_changed = false;

            match pending.shader {
                StagedShader::Keep => {}
                StagedShader::Clear => {
                    if let Some(old) = entry.custom_program.take()
                        && let Some(gl) = gl_opt
                    {
                        old.destroy(gl);
                    }
                    entry.custom_shader_path = None;
                    entry.last_shader_error = None;
                    if pending.clear_implicit_shader || entry.spec.shader.is_some() {
                        entry.spec.shader = None;
                    }
                    // With no shader, auxiliary textures have no consumer — release
                    // every pool ref and drop the spec entries, otherwise a stale
                    // `u_depth`-style matte stays pinned in VRAM forever. Any
                    // texture explicitly re-added in this same apply is re-inserted
                    // by the texture_updates loop below.
                    if let Some(gl) = gl_opt {
                        for (_, old_id) in std::mem::take(&mut entry.custom_textures) {
                            self.texture_pool.release(gl, old_id);
                        }
                    } else {
                        entry.custom_textures.clear();
                    }
                    entry.spec.textures.clear();
                    sh_changed = true;
                }
                StagedShader::Set { path, program } => {
                    if let Some(old) = entry.custom_program.replace(*program)
                        && let Some(gl) = gl_opt
                    {
                        old.destroy(gl);
                    }
                    let canonical = path.to_string_lossy().to_string();
                    entry.custom_shader_path = Some(path);
                    entry.last_shader_error = None;
                    entry.spec.shader = Some(canonical.clone());
                    if entry.job.is_none() && pending.target_override.is_some() {
                        entry.spec.shader = Some(canonical);
                    }
                    sh_changed = true;
                }
            }

            if let Some(target_str) = pending.target_override.take() {
                entry.spec.target = Some(target_str);
            }

            for (name, opt_spec) in pending.texture_updates {
                if let Some(old_id) = entry.custom_textures.remove(&name)
                    && let Some(gl) = gl_opt
                {
                    self.texture_pool.release(gl, old_id);
                }
                match opt_spec {
                    Some(spec) => {
                        let key = TextureKey::from_spec(&spec);
                        if let Some(tex_id) = pending
                            .acquired_textures
                            .remove(&key)
                            .or_else(|| self.texture_pool.acquire(&key))
                        {
                            entry.custom_textures.insert(name.clone(), tex_id);
                            entry.spec.textures.insert(name, spec);
                        }
                    }
                    None => {
                        entry.spec.textures.remove(&name);
                    }
                }
            }
            // Release any duplicate staging refs left in acquired_textures.
            if let Some(gl) = gl_opt {
                for (_, id) in pending.acquired_textures.drain() {
                    self.texture_pool.release(gl, id);
                }
            }

            for (name, val) in pending.uniform_updates {
                entry.custom_uniforms.insert(name.clone(), val.clone());
                entry.spec.uniforms.insert(name, val);
            }

            if let Some(params) = pending.params_update {
                entry.spec.params = params;
            }

            entry.revision = entry.revision.max(pending.revision);

            if entry.waiting_target_frame && entry.slot0_has_first_frame {
                entry.waiting_target_frame = false;
                entry.in_transition = true;
                entry.transition_started = Some(Instant::now());
            }

            entry.surface.dirty = true;
            entry.last_tick = Instant::now();

            (
                sh_changed,
                pending.persist,
                entry.name.clone(),
                entry.spec.clone(),
            )
        };

        if persist_now {
            self.persisted.outputs.insert(out_name, spec_snapshot);
            self.save_state();
        }
        if shader_changed {
            self.sync_shader_watcher();
        }
        self.render_if_ready(idx, qh);
    }

    fn handle_media_event(
        &mut self,
        ev: MediaEvent,
        qh: &wayland_client::QueueHandle<WaylandContext<Self>>,
    ) {
        match ev {
            MediaEvent::AuxTextureReady {
                key,
                width,
                height,
                pixels,
            } => {
                self.inflight_tex_decodes.remove(&key);
                if self.ensure_any_gl_context().is_err() {
                    tracing::warn!(
                        "no GL context available to upload aux texture {}",
                        key.path.display()
                    );
                    for i in 0..self.outputs.len() {
                        if self.outputs[i]
                            .pending_apply
                            .as_ref()
                            .is_some_and(|p| p.pending_keys.contains(&key))
                        {
                            self.cancel_output_pending(i, "no GL context for texture upload");
                        }
                    }
                    return;
                }
                let gl = &self.renderer.as_ref().unwrap().gl;
                let base_id =
                    match self
                        .texture_pool
                        .insert_rgba(gl, key.clone(), width, height, &pixels)
                    {
                        Ok(id) => id,
                        Err(err) => {
                            tracing::error!("aux texture upload failed: {err}");
                            for i in 0..self.outputs.len() {
                                if self.outputs[i]
                                    .pending_apply
                                    .as_ref()
                                    .is_some_and(|p| p.pending_keys.contains(&key))
                                {
                                    self.cancel_output_pending(i, &err.to_string());
                                }
                            }
                            return;
                        }
                    };

                // First waiter adopts base_id (refcount=1 already); additional outputs acquire().
                let mut used_base = false;
                for idx in 0..self.outputs.len() {
                    let needs_key = self.outputs[idx]
                        .pending_apply
                        .as_ref()
                        .is_some_and(|p| p.pending_keys.contains(&key));
                    if !needs_key {
                        continue;
                    }
                    let assigned_id = if !used_base {
                        used_base = true;
                        base_id
                    } else {
                        self.texture_pool.acquire(&key).unwrap_or(base_id)
                    };
                    if let Some(ref mut pending) = self.outputs[idx].pending_apply {
                        pending.pending_keys.remove(&key);
                        pending.acquired_textures.insert(key.clone(), assigned_id);
                    }
                    self.try_commit_pending_apply(idx, qh);
                }
                if !used_base && let Some(ref r) = self.renderer {
                    self.texture_pool.release(&r.gl, base_id);
                }
            }
            MediaEvent::AuxTextureError { key, message } => {
                self.inflight_tex_decodes.remove(&key);
                tracing::error!(
                    "auxiliary texture decode failed ({}): {message}",
                    key.path.display()
                );
                for idx in 0..self.outputs.len() {
                    if self.outputs[idx]
                        .pending_apply
                        .as_ref()
                        .is_some_and(|p| p.pending_keys.contains(&key))
                    {
                        self.cancel_output_pending(idx, &message);
                    }
                }
            }
            MediaEvent::Error {
                output,
                generation,
                message,
            } => {
                let Some(idx) = self.outputs.iter().position(|o| o.name == output) else {
                    return;
                };
                if !self.outputs[idx]
                    .job
                    .as_ref()
                    .is_some_and(|job| job.generation == generation)
                {
                    return;
                }
                tracing::error!("media error on {output}: {message}");
                self.outputs[idx].last_media_error = Some(message.clone());
                self.ipc_hub.broadcast_event(&IpcEvent::MediaError {
                    output: output.clone(),
                    message: message.clone(),
                });

                if self.outputs[idx]
                    .pending_apply
                    .as_ref()
                    .and_then(|p| p.awaiting_media_gen)
                    == Some(generation)
                {
                    self.cancel_output_pending(idx, &message);
                }

                let entry = &mut self.outputs[idx];
                if entry.waiting_target_frame {
                    // Revert pending transition cleanly back to previous slot1.
                    entry.waiting_target_frame = false;
                    if let Some(failed_job) = entry.job.take() {
                        failed_job.stop();
                    }
                    entry.job = entry.prev_job.take();
                    if let Some(job) = &entry.job {
                        job.set_paused(self.pause.is_paused(&entry.name));
                    }
                    if let Some(prev_slot) = entry.slot1.take() {
                        let old = std::mem::replace(&mut entry.slot0, prev_slot);
                        self.deferred_slots.push(old);
                    }
                    if let Some(tr) = entry.transition_program.take() {
                        self.deferred_programs.push(tr);
                    }
                    entry.surface.dirty = true;
                    self.render_if_ready(idx, qh);
                }
            }
            MediaEvent::FrameAvailable { output, generation } => {
                let Some(idx) = self.outputs.iter().position(|o| o.name == output) else {
                    return;
                };
                if !self.outputs[idx]
                    .job
                    .as_ref()
                    .is_some_and(|job| job.generation == generation)
                {
                    return;
                }
                let mailbox_opt = if self.outputs[idx]
                    .job
                    .as_ref()
                    .is_some_and(|j| j.generation == generation)
                {
                    self.outputs[idx]
                        .job
                        .as_ref()
                        .map(|j| std::sync::Arc::clone(&j.mailbox))
                } else {
                    None
                };
                let Some(mailbox) = mailbox_opt else {
                    return;
                };
                let Some(payload) = mailbox.pop_latest() else {
                    return;
                };

                let is_cuda = matches!(&payload, FramePayload::Cuda(_));
                if is_cuda
                    && self.outputs[idx]
                        .job
                        .as_ref()
                        .is_some_and(|j| j.control.cuda_failed())
                {
                    mailbox.recycle(payload);
                    return;
                }

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
                        FramePayload::Cuda(frame) => slot0.set_cuda(gl, frame, fit),
                    }
                };
                mailbox.recycle(payload);

                if let Err(err) = upload_res {
                    if is_cuda {
                        tracing::warn!(
                            "CUDA texture upload failed on {}: {err:#}; switching to FFmpeg",
                            self.outputs[idx].name
                        );
                        if let Some(job) = &self.outputs[idx].job {
                            job.control.request_cuda_fallback();
                        }
                        return;
                    }
                    tracing::error!("texture upload failed on {}: {err}", self.outputs[idx].name);
                    return;
                }

                self.outputs[idx].slot0_has_first_frame = true;

                // A one-shot (non-animated) job publishes exactly one frame then
                // its worker exits; keeping the MediaJob alive would pin the
                // FrameMailbox's spare/pending pixel buffers (w*h*4 each) for the
                // whole session. Drop it now that the frame is uploaded — the
                // texture in slot0 is unaffected and generation guards already
                // reject any stale events. Dropping (not stop()) avoids blocking
                // the event loop on a worker that already finished.
                if self.outputs[idx]
                    .job
                    .as_ref()
                    .is_some_and(|j| j.generation == generation && !j.kind.is_animated())
                {
                    self.outputs[idx].job = None;
                    // The dropped mailbox just freed multi-MB decoded-frame
                    // buffers; glibc keeps that memory in its arena instead of
                    // returning it to the OS, which shows up as a permanent RSS
                    // growth per apply. malloc_trim releases the free top pages.
                    unsafe { libc::malloc_trim(0) };
                }

                // If an atomic `apply` is still waiting on auxiliary textures, hold rendering
                // until `try_commit_pending_apply` completes cleanly.
                let blocked_on_aux = self.outputs[idx]
                    .pending_apply
                    .as_ref()
                    .is_some_and(|p| !p.pending_keys.is_empty());

                if blocked_on_aux {
                    return;
                }

                if self.outputs[idx].pending_apply.is_some() {
                    self.try_commit_pending_apply(idx, qh);
                    return;
                }

                let entry = &mut self.outputs[idx];
                if entry.waiting_target_frame {
                    entry.waiting_target_frame = false;
                    entry.in_transition = true;
                    entry.transition_started = Some(Instant::now());
                }
                entry.surface.dirty = true;
                self.render_if_ready(idx, qh);
            }
        }
    }

    fn reload_shader_file(&mut self, changed_path: &Path) {
        let changed_abs = absolutize_path(changed_path);
        let changed_canon = fs::canonicalize(&changed_abs).ok();
        let mut to_reload = Vec::new();
        for (idx, entry) in self.outputs.iter().enumerate() {
            if let Some(ref p) = entry.custom_shader_path {
                let exact_match = p == &changed_abs || p == changed_path;
                let canon_match = changed_canon
                    .as_ref()
                    .is_some_and(|cc| fs::canonicalize(p).ok().as_ref() == Some(cc));
                if exact_match || canon_match {
                    to_reload.push((idx, p.clone()));
                }
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
                    entry.revision += 1;
                    entry.last_shader_error = None;
                    entry.surface.dirty = true;
                    let out_name = entry.name.clone();
                    let rev = entry.revision;
                    tracing::info!("hot-reloaded shader {} on {}", path.display(), out_name);
                    self.ipc_hub.broadcast_event(&IpcEvent::ShaderReloaded {
                        output: out_name,
                        revision: rev,
                        ok: true,
                        error: None,
                    });
                }
                Err(err) => {
                    let msg = err.to_string();
                    let entry = &mut self.outputs[idx];
                    tracing::warn!(
                        "shader hot-reload compile failed on {} (keeping active shader): {msg}",
                        entry.name
                    );
                    entry.last_shader_error = Some(msg.clone());
                    let out_name = entry.name.clone();
                    let rev = entry.revision;
                    self.ipc_hub.broadcast_event(&IpcEvent::ShaderReloaded {
                        output: out_name,
                        revision: rev,
                        ok: false,
                        error: Some(msg),
                    });
                }
            }
        }
    }

    fn render_if_ready(
        &mut self,
        idx: usize,
        qh: &wayland_client::QueueHandle<WaylandContext<Self>>,
    ) {
        if self.outputs[idx]
            .pending_apply
            .as_ref()
            .is_some_and(|pending| {
                pending.is_ready(
                    self.outputs[idx].generation,
                    self.outputs[idx].slot0_has_first_frame,
                )
            })
        {
            self.try_commit_pending_apply(idx, qh);
            return;
        }
        let entry = &self.outputs[idx];
        let paused = self.pause.is_paused(&entry.name);
        let continuous = !paused && entry.uses_continuous_frames();
        // Preserve the last committed surface while the replacement media or
        // its auxiliary textures are still being staged.
        if !frame_needs_render(
            entry.surface.configured,
            entry.surface.frame_pending,
            entry.surface.dirty,
            continuous,
            entry.pending_apply.is_some(),
        ) {
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
                custom_textures: &entry.custom_textures,
                custom_uniforms: &entry.custom_uniforms,
            };
            renderer.draw(&draw_params, &self.texture_pool)
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

        let (out_name, revision, frame_index, swapped_ok) = {
            let entry = &mut self.outputs[idx];
            entry.surface.dirty = false;
            let keep_animating = !paused && entry.uses_continuous_frames();
            if keep_animating {
                // Request wl_surface.frame BEFORE eglSwapBuffers so the callback attaches atomically to this commit.
                entry.surface.request_frame_if_idle(qh);
            }

            let swapped_ok = if let Some(egl_surf) = entry.surface.egl_surface {
                match self.egl.swap_buffers(egl_surf) {
                    Ok(()) => true,
                    Err(err) => {
                        tracing::warn!("eglSwapBuffers failed on {}: {err}", entry.name);
                        false
                    }
                }
            } else {
                false
            };
            (
                entry.name.clone(),
                entry.revision,
                entry.frame_index,
                swapped_ok,
            )
        };

        if swapped_ok {
            self.ipc_hub.broadcast_event(&IpcEvent::FrameCommitted {
                output: out_name.clone(),
                revision,
                frame: frame_index,
            });
            if promote_done {
                self.ipc_hub.broadcast_event(&IpcEvent::TransitionDone {
                    output: out_name.clone(),
                    revision,
                });
            }
            self.resolve_commit_waiters_on_frame(&out_name, revision, frame_index);
        }
    }

    fn handle_ipc_line(
        &mut self,
        client_id: IpcClientId,
        raw_line: &str,
        qh: &wayland_client::QueueHandle<WaylandContext<Self>>,
    ) -> bool {
        let mut pending_req: Option<IpcRequest> = None;
        let early_remove = self.ipc_hub.handle_raw_line(client_id, raw_line, |_, req| {
            pending_req = Some(req);
            DispatchOutcome::Deferred
        });

        let Some(req) = pending_req else {
            return early_remove;
        };

        let outcome = self.dispatch_ipc(client_id, req, qh);
        match outcome {
            DispatchOutcome::Reply(resp) => {
                self.ipc_hub.send_response(client_id, &resp);
            }
            DispatchOutcome::Deferred => {}
        }
        !self.ipc_hub.is_client_alive(client_id)
    }

    fn dispatch_ipc(
        &mut self,
        client_id: IpcClientId,
        req: IpcRequest,
        qh: &wayland_client::QueueHandle<WaylandContext<Self>>,
    ) -> DispatchOutcome {
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
                let tx = TxParams {
                    output,
                    media: Some(ApplyMedia {
                        target,
                        fit,
                        fps,
                        hwaccel,
                    }),
                    shader,
                    transition: Some(ApplyTransition {
                        name: transition,
                        duration_ms,
                    }),
                    persist_transition_defaults: true,
                    persist: true,
                    ..Default::default()
                };
                match self.execute_transaction(tx, Some(client_id), qh) {
                    Ok(out) => out,
                    Err(err) => DispatchOutcome::Reply(IpcResponse::err(format!("{err:#}"))),
                }
            }
            IpcRequest::Shader { output, path } => {
                let tx = TxParams {
                    output,
                    shader: Some(path),
                    persist: true,
                    ..Default::default()
                };
                match self.execute_transaction(tx, Some(client_id), qh) {
                    Ok(out) => out,
                    Err(err) => DispatchOutcome::Reply(IpcResponse::err(format!("{err:#}"))),
                }
            }
            IpcRequest::Params { output, values } => {
                let tx = TxParams {
                    output,
                    params: Some(values),
                    persist: true,
                    ..Default::default()
                };
                match self.execute_transaction(tx, Some(client_id), qh) {
                    Ok(out) => out,
                    Err(err) => DispatchOutcome::Reply(IpcResponse::err(format!("{err:#}"))),
                }
            }
            IpcRequest::Uniform {
                output,
                name,
                value,
            } => {
                let mut uniforms = BTreeMap::new();
                uniforms.insert(name, value);
                let tx = TxParams {
                    output,
                    uniforms,
                    persist: true,
                    ..Default::default()
                };
                match self.execute_transaction(tx, Some(client_id), qh) {
                    Ok(out) => out,
                    Err(err) => DispatchOutcome::Reply(IpcResponse::err(format!("{err:#}"))),
                }
            }
            IpcRequest::Uniforms { output, values } => {
                let tx = TxParams {
                    output,
                    uniforms: values,
                    persist: true,
                    ..Default::default()
                };
                match self.execute_transaction(tx, Some(client_id), qh) {
                    Ok(out) => out,
                    Err(err) => DispatchOutcome::Reply(IpcResponse::err(format!("{err:#}"))),
                }
            }
            IpcRequest::TextureSet {
                output,
                name,
                path,
                filter,
                wrap,
                flip_y,
            } => {
                let mut textures = BTreeMap::new();
                textures.insert(
                    name,
                    Some(TextureSpec {
                        path: PathBuf::from(path),
                        filter,
                        wrap,
                        flip_y,
                    }),
                );
                let tx = TxParams {
                    output,
                    textures,
                    persist: true,
                    ..Default::default()
                };
                match self.execute_transaction(tx, Some(client_id), qh) {
                    Ok(out) => out,
                    Err(err) => DispatchOutcome::Reply(IpcResponse::err(format!("{err:#}"))),
                }
            }
            IpcRequest::TextureUnset { output, name } => {
                let mut textures = BTreeMap::new();
                textures.insert(name, None);
                let tx = TxParams {
                    output,
                    textures,
                    persist: true,
                    ..Default::default()
                };
                match self.execute_transaction(tx, Some(client_id), qh) {
                    Ok(out) => out,
                    Err(err) => DispatchOutcome::Reply(IpcResponse::err(format!("{err:#}"))),
                }
            }
            IpcRequest::Apply {
                output,
                revision,
                media,
                shader,
                textures,
                uniforms,
                transition,
                wait_commit,
            } => {
                let tx = TxParams {
                    output,
                    revision,
                    media,
                    shader,
                    textures,
                    uniforms,
                    params: None,
                    transition,
                    persist_transition_defaults: false,
                    persist: true,
                    wait_commit,
                };
                match self.execute_transaction(tx, Some(client_id), qh) {
                    Ok(out) => out,
                    Err(err) => DispatchOutcome::Reply(IpcResponse::err(format!("{err:#}"))),
                }
            }
            IpcRequest::Subscribe { .. } => {
                DispatchOutcome::Reply(IpcResponse::ok_msg("subscribed"))
            }
            IpcRequest::Pause { output, reason } => {
                let out_arg = (!output.is_empty() && output != "*").then_some(output.as_str());
                self.pause.add(out_arg, &reason);
                self.sync_pause_to_workers();
                DispatchOutcome::Reply(IpcResponse::ok_data(json!({
                    "paused_global": self.pause.global_reasons(),
                })))
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
                DispatchOutcome::Reply(IpcResponse::ok_data(json!({
                    "paused_global": self.pause.global_reasons(),
                })))
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
                DispatchOutcome::Reply(IpcResponse::ok_data(json!({
                    "manual_paused": paused_now,
                    "paused_global": self.pause.global_reasons(),
                })))
            }
            IpcRequest::Reload { output } => {
                let targets = self.find_outputs_matching(&output);
                if targets.is_empty() {
                    return DispatchOutcome::Reply(IpcResponse::err(format!(
                        "no connected Wayland output matches '{output}'"
                    )));
                }
                for idx in targets {
                    let spec = self.outputs[idx].spec.clone();
                    let name = self.outputs[idx].name.clone();
                    let textures: BTreeMap<String, Option<TextureSpec>> = spec
                        .textures
                        .iter()
                        .map(|(k, v)| (k.clone(), Some(v.clone())))
                        .collect();
                    let tx = TxParams {
                        output: name,
                        media: spec.target.map(|t| ApplyMedia {
                            target: t,
                            fit: Some(spec.fit),
                            fps: spec.fps,
                            hwaccel: Some(spec.hwaccel),
                        }),
                        shader: spec.shader,
                        textures,
                        uniforms: spec.uniforms,
                        params: Some(spec.params),
                        transition: Some(ApplyTransition {
                            name: Some("none".to_string()),
                            duration_ms: Some(0),
                        }),
                        persist_transition_defaults: false,
                        persist: false,
                        wait_commit: false,
                        revision: None,
                    };
                    if let Err(err) = self.execute_transaction(tx, Some(client_id), qh) {
                        return DispatchOutcome::Reply(IpcResponse::err(format!("{err:#}")));
                    }
                }
                DispatchOutcome::Reply(IpcResponse::ok_msg("reloaded"))
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
                DispatchOutcome::Reply(IpcResponse::ok_data(json!({ "outputs": list })))
            }
            IpcRequest::Status => {
                let outputs: Vec<_> = self
                    .outputs
                    .iter()
                    .map(|o| {
                        let bound_textures: BTreeMap<String, _> = o
                            .spec
                            .textures
                            .iter()
                            .map(|(name, spec)| {
                                let size = o
                                    .custom_textures
                                    .get(name)
                                    .and_then(|id| self.texture_pool.texture_info(*id))
                                    .map(|(_, w, h)| [w, h]);
                                (
                                    name.clone(),
                                    json!({
                                        "path": spec.path,
                                        "filter": spec.filter,
                                        "wrap": spec.wrap,
                                        "flip_y": spec.flip_y,
                                        "size": size,
                                    }),
                                )
                            })
                            .collect();
                        json!({
                            "name": o.name,
                            "revision": o.revision,
                            "logical_size": [o.surface.logical_w, o.surface.logical_h],
                            "physical_size": [o.surface.phys_w, o.surface.phys_h],
                            "scale": f64::from(o.surface.scale_120) / 120.0,
                            "configured": o.surface.configured,
                            "target": o.spec.target,
                            "shader": o.spec.shader,
                            "shader_schema": o.custom_program.as_ref().map(GlProgram::schema),
                            "textures": bound_textures,
                            "uniforms": o.custom_uniforms,
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
                DispatchOutcome::Reply(IpcResponse::ok_data(json!({
                    "auto_pause_fullscreen": self.config.auto_pause_fullscreen,
                    "persist_state": self.persist_enabled,
                    "pool_textures": self.texture_pool.len(),
                    "global_pause_reasons": self.pause.global_reasons(),
                    "outputs": outputs,
                })))
            }
            IpcRequest::Stop => {
                self.should_exit = true;
                DispatchOutcome::Reply(IpcResponse::ok_msg("stopping glwall daemon"))
            }
        }
    }

    fn on_ipc_client_disconnected(&mut self, id: IpcClientId) {
        self.ipc_hub.remove_client(id);
        self.commit_waiters.retain(|w| w.client_id != id);
    }

    fn flush_ipc_reregisters(&mut self) {
        for (id, token, closed) in self.ipc_hub.drain_pending_reregisters() {
            if closed {
                self.loop_handle.remove(token);
                self.on_ipc_client_disconnected(id);
            } else {
                let _ = self.loop_handle.update(&token);
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
        let bound = self.bind_cleanup_gl_context().is_ok()
            || entry.surface.egl_surface.is_some_and(|surface| {
                self.egl
                    .make_current_surface(surface, &mut entry.surface.swap_interval_set)
                    .is_ok()
            });
        if self.renderer.is_some() && !bound {
            tracing::error!(
                "cannot bind EGL context to clean up output {}; deferring GPU cleanup",
                entry.name
            );
            entry.surface.teardown(&self.egl);
            self.deferred_outputs.push(entry);
            return;
        }
        self.drain_deferred_gpu();
        self.release_output_gpu(&mut entry);
        if let Some(r) = &self.renderer {
            crate::gpu::renderer::retry_retired_cuda(&r.gl);
        }
        if self.outputs.is_empty()
            && let Some(r) = self.renderer.take()
        {
            let pool = std::mem::take(&mut self.texture_pool);
            pool.destroy(&r.gl);
            r.destroy();
        }
        entry.surface.teardown(&self.egl);
    }

    fn shutdown(&mut self) {
        while let Some(entry) = self.outputs.pop() {
            self.teardown_output_entry(entry);
        }
        for waiter in std::mem::take(&mut self.commit_waiters) {
            self.ipc_hub
                .complete_pending_ack(waiter.client_id, &IpcResponse::err("daemon shutting down"));
        }
        if self.renderer.is_some() {
            if let Err(err) = self.egl.make_current_surfaceless() {
                tracing::error!(
                    "cannot bind EGL context for renderer shutdown; retaining GPU owners: {err:#}"
                );
                std::mem::forget(std::mem::take(&mut self.deferred_outputs));
                std::mem::forget(std::mem::take(&mut self.deferred_applies));
                std::mem::forget(std::mem::take(&mut self.deferred_slots));
                std::mem::forget(std::mem::take(&mut self.deferred_programs));
                std::mem::forget(std::mem::take(&mut self.texture_pool));
                std::mem::forget(self.renderer.take());
            } else {
                self.drain_deferred_gpu();
                let pool = std::mem::take(&mut self.texture_pool);
                let r = self.renderer.take().unwrap();
                pool.destroy(&r.gl);
                r.destroy();
            }
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
            name: name.clone(),
            name_resolved,
            surface,
            spec,
            revision: 0,
            generation: 0,
            slot0: MediaSlotGpu::empty(),
            slot1: None,
            slot0_has_first_frame: false,
            job: None,
            prev_job: None,
            custom_program: None,
            custom_shader_path: None,
            custom_textures: BTreeMap::new(),
            custom_uniforms: BTreeMap::new(),
            last_shader_error: None,
            last_media_error: None,
            transition_program: None,
            transition_started: None,
            waiting_target_frame: false,
            in_transition: false,
            anim_time_secs: 0.0,
            last_tick: Instant::now(),
            frame_index: 0,
            pending_apply: None,
        });

        let idx = self.outputs.len() - 1;
        if self.outputs[idx].name_resolved {
            self.ipc_hub
                .broadcast_event(&IpcEvent::OutputAdded { output: name });
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
            let was_resolved = self.outputs[idx].name_resolved;
            let changed_name = !was_resolved || self.outputs[idx].name != real_name;
            self.outputs[idx].name = real_name.clone();
            self.outputs[idx].name_resolved = true;
            if !was_resolved {
                self.ipc_hub
                    .broadcast_event(&IpcEvent::OutputAdded { output: real_name });
            }
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
        let name = entry.name.clone();
        self.pause.drop_output(&name);
        self.fullscreen_outputs.remove(&name);
        self.fail_commit_waiters_for_output(&name, "output removed before commit");
        self.teardown_output_entry(entry);
        self.sync_shader_watcher();
        self.ipc_hub
            .broadcast_event(&IpcEvent::OutputRemoved { output: name });
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
            entry.name = real_name.clone();
            entry.name_resolved = true;
            self.ipc_hub
                .broadcast_event(&IpcEvent::OutputAdded { output: real_name });
        }
        let entry = &self.outputs[idx];
        let need_apply = entry.name_resolved
            && entry.job.is_none()
            && entry.custom_program.is_none()
            && entry.pending_apply.is_none();
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
        let name = entry.name.clone();
        self.fail_commit_waiters_for_output(&name, "surface closed before commit");
        self.teardown_output_entry(entry);
        self.sync_shader_watcher();
    }

    fn on_surface_frame(&mut self, surface: &wl_surface::WlSurface, _time_ms: u32) {
        let Some(idx) = self.find_output_by_surface(surface) else {
            return;
        };
        self.outputs[idx].surface.frame_pending = false;
        self.outputs[idx].surface.dirty = true;
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
    no_persist: bool,
) -> Result<()> {
    let config = Config::load_or_default(config_path.as_deref())?;
    let persist_enabled = config.persist_state && !no_persist;
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

    let (wl_conn, mut wl_queue, mut ctx) = connect_wayland_and_egl(DaemonInit {
        config,
        state_path,
        persisted,
        restore_on_connect: !no_restore,
        persist_enabled,
        media_tx,
        watcher,
        loop_handle: loop_handle.clone(),
    })?;

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
                        Ok((stream, _)) => match ctx.sink.ipc_hub.accept_stream(stream) {
                            Ok((client_id, client_source)) => {
                                let lh = ctx.sink.loop_handle.clone();
                                match lh.insert_source(
                                    client_source,
                                    move |(id, readiness), _, ctx| {
                                        let (lines, mut should_remove) =
                                            ctx.sink.ipc_hub.poll_client_lines(id, readiness);
                                        let qh = ctx.qh.clone();
                                        for line in lines {
                                            let rm = ctx.sink.handle_ipc_line(id, &line, &qh);
                                            should_remove = should_remove || rm;
                                        }
                                        if should_remove {
                                            ctx.sink.on_ipc_client_disconnected(id);
                                            PostAction::Remove
                                        } else {
                                            PostAction::Continue
                                        }
                                    },
                                ) {
                                    Ok(token) => {
                                        ctx.sink.ipc_hub.set_client_token(client_id, token)
                                    }
                                    Err(err) => {
                                        tracing::warn!(
                                            "failed to register IPC client source: {err:?}"
                                        );
                                        ctx.sink.ipc_hub.remove_client(client_id);
                                    }
                                }
                            }
                            Err(err) => {
                                tracing::warn!("failed configuring accepted IPC client: {err}");
                            }
                        },
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
    ctx.sink.flush_ipc_reregisters();
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
        ctx.sink.flush_ipc_reregisters();
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

#[cfg(test)]
mod tests {
    use super::frame_needs_render;

    #[test]
    fn pending_wallpaper_keeps_previous_frame_until_commit() {
        // A newly started video requests continuous frames before decode is ready.
        assert!(!frame_needs_render(true, false, false, true, true));
        // Damage or a surface reconfiguration must not expose the empty slot either.
        assert!(!frame_needs_render(true, false, true, true, true));
        // Once the target frame is committed, the transition may render.
        assert!(frame_needs_render(true, false, true, true, false));
        // Normal frame throttling remains in force.
        assert!(!frame_needs_render(true, true, true, true, false));
        assert!(!frame_needs_render(false, false, true, true, false));
        assert!(!frame_needs_render(true, false, false, false, false));
    }
}
