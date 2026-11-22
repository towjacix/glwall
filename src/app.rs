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

use crate::config::{Config, FitMode, HwAccel, OutputSpec, PersistedState, default_state_path};
use crate::gpu::egl::EglState;
use crate::gpu::renderer::{
    DrawParams, GlProgram, MediaSlotGpu, Renderer, SlotMediaKind, TextureFingerprint, TextureId,
    TextureKey, TexturePool,
};
use crate::gpu::shader::resolve_transition_source;
use crate::gpu::uniforms::{TextureSpec, UniformValue};
use crate::ipc::{
    ApplyMedia, ApplyTransition, BatchOutputApply, DispatchOutcome, IpcClientId, IpcErrorCode,
    IpcEvent, IpcHub, IpcRequest, IpcResponse, bind_ipc_listener, coded_err, error_code_of,
};
use crate::media::gst_cuda::active_pipeline_count;
use crate::media::static_img::check_image_limits;
use crate::media::{
    FramePayload, MediaEvent, MediaJob, MediaKind, active_media_workers, aux_decode_has_capacity,
    aux_decode_stats, decode_texture_async,
};
use crate::pause::PauseState;
use crate::watcher::{ShaderWatcher, absolutize_path};
use crate::wayland::{SurfaceHandle, WaylandAppSink, WaylandContext, WaylandHandles};

const WAIT_COMMIT_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Default)]
struct TransientUniformState {
    session: String,
    last_seq: u64,
    pending_overlay: Option<BTreeMap<String, UniformValue>>,
    active_overlay: BTreeMap<String, UniformValue>,
}

impl TransientUniformState {
    fn clear(&mut self) -> bool {
        let had_any = !self.active_overlay.is_empty() || self.pending_overlay.is_some();
        self.session.clear();
        self.last_seq = 0;
        self.pending_overlay = None;
        self.active_overlay.clear();
        had_any
    }

    fn prune(&mut self, allowed_uniforms: &HashSet<String>) {
        if let Some(ref mut pending) = self.pending_overlay {
            pending.retain(|k, _| allowed_uniforms.contains(k.as_str()));
            if pending.is_empty() {
                self.pending_overlay = None;
            }
        }
        self.active_overlay
            .retain(|k, _| allowed_uniforms.contains(k.as_str()));
    }

    /// Promotes any staged 1-slot overlay into `active_overlay` right before drawing.
    fn consume_pending(&mut self) {
        if let Some(next) = self.pending_overlay.take() {
            self.active_overlay = next;
        }
    }

    fn current_overlay(&self) -> &BTreeMap<String, UniformValue> {
        self.pending_overlay
            .as_ref()
            .unwrap_or(&self.active_overlay)
    }
}

fn effective_uniforms_with_overlay(
    base: &BTreeMap<String, UniformValue>,
    overlay: &BTreeMap<String, UniformValue>,
) -> BTreeMap<String, UniformValue> {
    let mut merged = base.clone();
    for (k, v) in overlay {
        merged.insert(k.clone(), v.clone());
    }
    merged
}

struct OutputEntry {
    index: i32,
    wl_output: wl_output::WlOutput,
    name: String,
    name_resolved: bool,
    surface: SurfaceHandle,
    spec: OutputSpec,
    revision: u64,
    generation: u64,
    active_generation: u64,
    pending_generation: Option<u64>,
    last_committed_frame: Option<i32>,
    last_committed_revision: u64,
    target_sig: Option<(u64, i64, i64)>,
    shader_sig: Option<(u64, i64, i64)>,
    last_transition: (String, u32),
    slot0: MediaSlotGpu,
    slot1: Option<MediaSlotGpu>,
    slot0_has_first_frame: bool,
    job: Option<MediaJob>,
    prev_job: Option<MediaJob>,
    custom_program: Option<GlProgram>,
    custom_shader_path: Option<PathBuf>,
    custom_textures: BTreeMap<String, TextureId>,
    custom_texture_keys: BTreeMap<String, TextureKey>,
    custom_uniforms: BTreeMap<String, UniformValue>,
    transient: TransientUniformState,
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

    fn needs_resolve_target(&self) -> bool {
        if self.in_transition && self.custom_program.is_some() && self.transition_program.is_some()
        {
            return true;
        }
        self.custom_program.as_ref().is_some_and(|p| {
            p.assembled.mode == crate::gpu::shader::ShaderMode::Shadertoy && p.uses_ichannel0
        })
    }

    /// Extracts the GPU-owned resources, leaving `self` free to be torn down.
    fn gpu_parts(&mut self) -> DeferredOutputGpu {
        DeferredOutputGpu {
            name: self.name.clone(),
            pending_apply: self.pending_apply.take(),
            slot0: std::mem::replace(&mut self.slot0, MediaSlotGpu::empty()),
            slot1: self.slot1.take(),
            custom_program: self.custom_program.take(),
            transition_program: self.transition_program.take(),
            custom_textures: std::mem::take(&mut self.custom_textures),
        }
    }
}

/// GPU-owned resources extracted from an [`OutputEntry`] whose Wayland/EGL
/// surface was already torn down. Deferred until a GL context can be bound.
struct DeferredOutputGpu {
    name: String,
    pending_apply: Option<PendingApply>,
    slot0: MediaSlotGpu,
    slot1: Option<MediaSlotGpu>,
    custom_program: Option<GlProgram>,
    transition_program: Option<GlProgram>,
    custom_textures: BTreeMap<String, TextureId>,
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
    group_id: Option<String>,
    persist: bool,
    shader: StagedShader,
    target_sig: Option<(u64, i64, i64)>,
    shader_sig: Option<(u64, i64, i64)>,
    applied_transition: Option<(String, u32)>,
    clear_transient: bool,
    texture_updates: BTreeMap<String, Option<(TextureSpec, TextureKey)>>,
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

impl OutputEntry {
    fn media_kind_label(&self) -> &'static str {
        if let Some(ref job) = self.job {
            return match job.kind {
                MediaKind::ProceduralShader => "shader",
                MediaKind::StaticImage | MediaKind::FfmpegStaticImage => "image",
                MediaKind::AnimatedGif | MediaKind::AnimatedWebp => "animated_image",
                MediaKind::Video => "video",
            };
        }
        match self.slot0.kind {
            SlotMediaKind::None => {
                if self.custom_program.is_some()
                    || self
                        .spec
                        .target
                        .as_deref()
                        .and_then(|t| MediaKind::from_path(Path::new(t)))
                        == Some(MediaKind::ProceduralShader)
                {
                    "shader"
                } else {
                    "none"
                }
            }
            SlotMediaKind::Rgba => {
                match self
                    .spec
                    .target
                    .as_deref()
                    .and_then(|t| MediaKind::from_path(Path::new(t)))
                {
                    Some(MediaKind::AnimatedGif | MediaKind::AnimatedWebp) => "animated_image",
                    _ => "image",
                }
            }
            SlotMediaKind::Nv12 => "video",
        }
    }

    fn decoder_status(&self, paused: bool) -> &'static str {
        if self.pending_apply.is_some() {
            "staging"
        } else if self.last_media_error.is_some() && self.job.is_none() {
            "error"
        } else if self.waiting_target_frame || self.in_transition {
            "transitioning"
        } else if self.job.is_some() {
            if paused { "paused" } else { "decoding" }
        } else if self.slot0.kind != SlotMediaKind::None || self.custom_program.is_some() {
            "ready"
        } else {
            "idle"
        }
    }

    fn estimated_gpu_bytes(&self, renderer: Option<&Renderer>) -> u64 {
        let s0 = self.slot0.estimated_gpu_bytes();
        let s1 = self.slot1.as_ref().map_or(0, |s| s.estimated_gpu_bytes());
        let res = renderer.map_or(0, |r| r.resolve_target_gpu_bytes(&self.name));
        s0 + s1 + res
    }
}

fn read_self_memory_stats() -> (u64, u64, u64) {
    let Ok(text) = fs::read_to_string("/proc/self/smaps_rollup") else {
        return (0, 0, 0);
    };
    let mut rss = 0u64;
    let mut pss = 0u64;
    let mut private_dirty = 0u64;
    for line in text.lines() {
        let mut parts = line.split_whitespace();
        let Some(key) = parts.next() else {
            continue;
        };
        let Some(val_str) = parts.next() else {
            continue;
        };
        let Ok(val) = val_str.parse::<u64>() else {
            continue;
        };
        match key {
            "Rss:" => rss = val,
            "Pss:" => pss = val,
            "Private_Dirty:" => private_dirty = val,
            _ => {}
        }
    }
    (rss, pss, private_dirty)
}

fn file_stat_sig(path: &Path) -> Option<(u64, i64, i64)> {
    use std::os::unix::fs::MetadataExt;
    let meta = fs::metadata(path).ok()?;
    Some((meta.len(), meta.mtime(), meta.mtime_nsec()))
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum UniformFingerprint {
    Bool(bool),
    Int(i32),
    Float(u32),
    Vec2([u32; 2]),
    Vec3([u32; 3]),
    Vec4([u32; 4]),
}

fn norm_f32_bits(v: f32) -> u32 {
    if v == 0.0 {
        0.0f32.to_bits()
    } else {
        v.to_bits()
    }
}

impl From<&UniformValue> for UniformFingerprint {
    fn from(v: &UniformValue) -> Self {
        match v {
            UniformValue::Bool(b) => Self::Bool(*b),
            UniformValue::Int(i) => Self::Int(*i),
            UniformValue::Float(f) => Self::Float(norm_f32_bits(*f)),
            UniformValue::Vec2([a, b]) => Self::Vec2([norm_f32_bits(*a), norm_f32_bits(*b)]),
            UniformValue::Vec3([a, b, c]) => {
                Self::Vec3([norm_f32_bits(*a), norm_f32_bits(*b), norm_f32_bits(*c)])
            }
            UniformValue::Vec4([a, b, c, d]) => Self::Vec4([
                norm_f32_bits(*a),
                norm_f32_bits(*b),
                norm_f32_bits(*c),
                norm_f32_bits(*d),
            ]),
        }
    }
}

type TexFpEntry = (
    PathBuf,
    crate::gpu::uniforms::TextureFilter,
    crate::gpu::uniforms::TextureWrap,
    bool,
    TextureFingerprint,
);

#[derive(Debug, Clone, PartialEq, Eq)]
struct EffectiveSpecFingerprint {
    target: Option<String>,
    target_sig: Option<(u64, i64, i64)>,
    fit: FitMode,
    fps_bits: Option<u32>,
    hwaccel: HwAccel,
    shader: Option<String>,
    shader_sig: Option<(u64, i64, i64)>,
    transition: String,
    duration_ms: u32,
    params_bits: [u32; 4],
    textures: BTreeMap<String, TexFpEntry>,
    uniforms: BTreeMap<String, UniformFingerprint>,
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

#[allow(clippy::too_many_arguments)]
fn prune_shader_bindings(
    custom_textures: &mut BTreeMap<String, TextureId>,
    custom_texture_keys: &mut BTreeMap<String, TextureKey>,
    spec_textures: &mut BTreeMap<String, TextureSpec>,
    custom_uniforms: &mut BTreeMap<String, UniformValue>,
    spec_uniforms: &mut BTreeMap<String, UniformValue>,
    allowed_samplers: &HashSet<String>,
    allowed_uniforms: &HashSet<String>,
) -> Vec<TextureId> {
    let stale_tex_keys: Vec<String> = custom_textures
        .keys()
        .filter(|k| !allowed_samplers.contains(k.as_str()))
        .cloned()
        .collect();
    let mut removed_ids = Vec::with_capacity(stale_tex_keys.len());
    for key in stale_tex_keys {
        if let Some(id) = custom_textures.remove(&key) {
            removed_ids.push(id);
        }
        custom_texture_keys.remove(&key);
        spec_textures.remove(&key);
    }
    custom_texture_keys.retain(|k, _| allowed_samplers.contains(k.as_str()));
    spec_textures.retain(|k, _| allowed_samplers.contains(k.as_str()));
    custom_uniforms.retain(|k, _| allowed_uniforms.contains(k.as_str()));
    spec_uniforms.retain(|k, _| allowed_uniforms.contains(k.as_str()));
    removed_ids
}

#[allow(clippy::too_many_arguments)]
fn apply_program_and_prune(
    gl: &glow::Context,
    new_prog: GlProgram,
    custom_program: &mut Option<GlProgram>,
    custom_textures: &mut BTreeMap<String, TextureId>,
    custom_texture_keys: &mut BTreeMap<String, TextureKey>,
    spec_textures: &mut BTreeMap<String, TextureSpec>,
    custom_uniforms: &mut BTreeMap<String, UniformValue>,
    spec_uniforms: &mut BTreeMap<String, UniformValue>,
    texture_pool: &mut TexturePool,
) {
    let allowed_samplers: HashSet<String> = new_prog
        .user_samplers
        .iter()
        .map(|s| s.name.clone())
        .collect();
    let allowed_uniforms: HashSet<String> = new_prog
        .user_uniforms
        .iter()
        .map(|u| u.name.clone())
        .collect();
    if let Some(old) = custom_program.replace(new_prog) {
        old.destroy(gl);
    }
    let removed = prune_shader_bindings(
        custom_textures,
        custom_texture_keys,
        spec_textures,
        custom_uniforms,
        spec_uniforms,
        &allowed_samplers,
        &allowed_uniforms,
    );
    for old_id in removed {
        texture_pool.release(gl, old_id);
    }
}

#[allow(clippy::too_many_arguments)]
fn reload_shader_into_bindings(
    gl: &glow::Context,
    src: &str,
    custom_program: &mut Option<GlProgram>,
    custom_textures: &mut BTreeMap<String, TextureId>,
    custom_texture_keys: &mut BTreeMap<String, TextureKey>,
    spec_textures: &mut BTreeMap<String, TextureSpec>,
    custom_uniforms: &mut BTreeMap<String, UniformValue>,
    spec_uniforms: &mut BTreeMap<String, UniformValue>,
    texture_pool: &mut TexturePool,
) -> Result<()> {
    let new_prog = GlProgram::compile(gl, src)
        .map_err(|e| coded_err(IpcErrorCode::ShaderCompile, format!("{e:#}")))?;
    apply_program_and_prune(
        gl,
        new_prog,
        custom_program,
        custom_textures,
        custom_texture_keys,
        spec_textures,
        custom_uniforms,
        spec_uniforms,
        texture_pool,
    );
    Ok(())
}

struct CommitWaiter {
    client_id: IpcClientId,
    group_id: Option<String>,
    revision_targets: Vec<(String, u64)>,
    completed: HashSet<String>,
    max_revision: u64,
    deadline: Instant,
}

struct PreparedTransaction {
    next_revision: u64,
    shader: StagedShader,
    target_sig: Option<(u64, i64, i64)>,
    shader_sig: Option<(u64, i64, i64)>,
    resolved_media: Option<(PathBuf, MediaKind, ApplyMedia)>,
    spawned_job: Option<MediaJob>,
    transition_spec: Option<(String, u32)>,
    applied_transition: Option<(String, u32)>,
    clear_transient: bool,
    transition_prog: Option<GlProgram>,
    texture_updates: BTreeMap<String, Option<(TextureSpec, TextureKey)>>,
    uniform_updates: BTreeMap<String, UniformValue>,
    params_update: Option<[f32; 4]>,
    clear_implicit_shader: bool,
}

impl PreparedTransaction {
    fn destroy_staged(mut self, gl: Option<&glow::Context>) {
        if let Some(job) = self.spawned_job.take() {
            job.stop();
        }
        if let Some(gl) = gl {
            if let StagedShader::Set { program, .. } = self.shader {
                (*program).destroy(gl);
            }
            if let Some(tr) = self.transition_prog.take() {
                tr.destroy(gl);
            }
        }
    }
}

#[derive(Default, Clone)]
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
    deferred_outputs: Vec<DeferredOutputGpu>,
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

fn normalize_optional_path(raw: Option<&str>) -> Option<String> {
    let trimmed = raw?.trim();
    if trimmed.is_empty() || trimmed == "none" {
        return None;
    }
    Some(
        absolutize_path(Path::new(trimmed))
            .to_string_lossy()
            .to_string(),
    )
}

#[allow(clippy::too_many_arguments)]
fn should_materialize_initial_spec(
    desired: &OutputSpec,
    current_spec: &OutputSpec,
    has_slot0_media: bool,
    has_active_job: bool,
    has_custom_program: bool,
    custom_textures_len: usize,
    custom_uniforms: &BTreeMap<String, UniformValue>,
    has_pending_apply: bool,
) -> bool {
    let desired_target = desired
        .target
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty());
    let desired_shader = desired
        .shader
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty() && *s != "none");

    if desired_target.is_none()
        && desired_shader.is_none()
        && desired.textures.is_empty()
        && desired.uniforms.is_empty()
    {
        return false;
    }

    if has_pending_apply {
        return false;
    }

    let desired_target_norm = normalize_optional_path(desired_target);
    let current_target_norm = normalize_optional_path(current_spec.target.as_deref());
    let is_procedural = desired_target_norm
        .as_deref()
        .and_then(|t| MediaKind::from_path(Path::new(t)))
        == Some(MediaKind::ProceduralShader);

    if desired_target_norm.is_some() {
        if current_target_norm != desired_target_norm {
            return true;
        }
        if is_procedural {
            if !has_custom_program {
                return true;
            }
        } else if !has_slot0_media && !has_active_job {
            return true;
        }
    } else if has_slot0_media || has_active_job {
        return true;
    }

    let desired_shader_norm = normalize_optional_path(desired_shader);
    let current_shader_norm = normalize_optional_path(current_spec.shader.as_deref());
    if desired_shader_norm.is_some()
        && (!has_custom_program || current_shader_norm != desired_shader_norm)
    {
        return true;
    }

    custom_textures_len != desired.textures.len()
        || current_spec.textures != desired.textures
        || custom_uniforms != &desired.uniforms
}

enum ShaderRequest {
    Keep,
    Clear,
    Set { path: PathBuf, source: String },
}

struct ResolvedPreflightInputs {
    next_revision: u64,
    resolved_media: Option<(PathBuf, MediaKind, ApplyMedia)>,
    target_sig: Option<(u64, i64, i64)>,
    shader_sig: Option<(u64, i64, i64)>,
    shader_request: ShaderRequest,
    clear_implicit_shader: bool,
    transition_spec: Option<(String, u32)>,
    transition_source: Option<String>,
    normalized_textures: BTreeMap<String, Option<(TextureSpec, TextureKey)>>,
    estimated_media_bytes: u64,
    estimated_aux_bytes: HashMap<TextureKey, u64>,
}

impl ResolvedPreflightInputs {
    fn needs_gl(&self) -> bool {
        matches!(self.shader_request, ShaderRequest::Set { .. }) || self.transition_source.is_some()
    }
}

fn map_transition_source_err(err: anyhow::Error) -> anyhow::Error {
    if err
        .downcast_ref::<std::io::Error>()
        .is_some_and(|io| io.kind() == ErrorKind::NotFound)
    {
        coded_err(IpcErrorCode::MissingSource, format!("{err:#}"))
    } else {
        coded_err(IpcErrorCode::InvalidInput, format!("{err:#}"))
    }
}

fn texture_spec_fp_entry(spec: &TextureSpec, key: &TextureKey) -> TexFpEntry {
    let canonical_path = std::fs::canonicalize(&spec.path).unwrap_or_else(|_| spec.path.clone());
    (
        canonical_path,
        spec.filter,
        spec.wrap,
        spec.flip_y,
        key.fingerprint,
    )
}

fn fingerprint_from_output_entry(entry: &OutputEntry) -> (EffectiveSpecFingerprint, bool) {
    if let Some(ref pending) = entry.pending_apply {
        let target = normalize_optional_path(
            pending
                .target_override
                .as_deref()
                .or(entry.spec.target.as_deref()),
        );
        let target_sig = if pending.target_override.is_some() {
            pending.target_sig
        } else {
            entry.target_sig
        };
        let (shader, shader_sig) = match &pending.shader {
            StagedShader::Keep => (
                normalize_optional_path(entry.spec.shader.as_deref()),
                entry.shader_sig,
            ),
            StagedShader::Clear => (None, None),
            StagedShader::Set { path, .. } => (
                normalize_optional_path(Some(&path.to_string_lossy())),
                pending.shader_sig,
            ),
        };
        let (transition, duration_ms) = pending
            .applied_transition
            .clone()
            .unwrap_or_else(|| entry.last_transition.clone());
        let params_bits = pending
            .params_update
            .unwrap_or(entry.spec.params)
            .map(norm_f32_bits);

        let mut textures: BTreeMap<String, TexFpEntry> = entry
            .spec
            .textures
            .iter()
            .map(|(k, spec)| {
                let key = entry
                    .custom_texture_keys
                    .get(k)
                    .cloned()
                    .unwrap_or_else(|| TextureKey::from_spec(spec));
                (k.clone(), texture_spec_fp_entry(spec, &key))
            })
            .collect();
        let mut uniforms: BTreeMap<String, UniformFingerprint> = entry
            .custom_uniforms
            .iter()
            .map(|(k, v)| (k.clone(), UniformFingerprint::from(v)))
            .collect();

        match &pending.shader {
            StagedShader::Keep => {}
            StagedShader::Clear => {
                textures.clear();
                uniforms.clear();
            }
            StagedShader::Set { program, .. } => {
                let allowed_s: HashSet<&str> = program
                    .user_samplers
                    .iter()
                    .map(|s| s.name.as_str())
                    .collect();
                let allowed_u: HashSet<&str> = program
                    .user_uniforms
                    .iter()
                    .map(|u| u.name.as_str())
                    .collect();
                textures.retain(|k, _| allowed_s.contains(k.as_str()));
                uniforms.retain(|k, _| allowed_u.contains(k.as_str()));
            }
        }
        for (k, opt) in &pending.texture_updates {
            match opt {
                Some((spec, key)) => {
                    textures.insert(k.clone(), texture_spec_fp_entry(spec, key));
                }
                None => {
                    textures.remove(k);
                }
            }
        }
        for (k, v) in &pending.uniform_updates {
            uniforms.insert(k.clone(), UniformFingerprint::from(v));
        }

        let fp = EffectiveSpecFingerprint {
            target,
            target_sig,
            fit: entry.spec.fit,
            fps_bits: entry.spec.fps.map(norm_f32_bits),
            hwaccel: entry.spec.hwaccel,
            shader,
            shader_sig,
            transition,
            duration_ms,
            params_bits,
            textures,
            uniforms,
        };
        return (fp, true);
    }

    let target = normalize_optional_path(entry.spec.target.as_deref());
    let shader = normalize_optional_path(entry.spec.shader.as_deref());
    let is_procedural = target
        .as_deref()
        .and_then(|t| MediaKind::from_path(Path::new(t)))
        == Some(MediaKind::ProceduralShader);
    let media_ok = if target.is_some() {
        if is_procedural {
            entry.custom_program.is_some()
        } else {
            entry.slot0.kind != SlotMediaKind::None || entry.job.is_some()
        }
    } else {
        true
    };
    let shader_ok = shader.is_none() || entry.custom_program.is_some();
    let tex_ok = entry
        .spec
        .textures
        .iter()
        .all(|(k, _)| entry.custom_textures.contains_key(k));

    let textures = entry
        .spec
        .textures
        .iter()
        .map(|(k, spec)| {
            let key = entry
                .custom_texture_keys
                .get(k)
                .cloned()
                .unwrap_or_else(|| TextureKey::from_spec(spec));
            (k.clone(), texture_spec_fp_entry(spec, &key))
        })
        .collect();
    let uniforms = entry
        .custom_uniforms
        .iter()
        .map(|(k, v)| (k.clone(), UniformFingerprint::from(v)))
        .collect();

    let fp = EffectiveSpecFingerprint {
        target,
        target_sig: entry.target_sig,
        fit: entry.spec.fit,
        fps_bits: entry.spec.fps.map(norm_f32_bits),
        hwaccel: entry.spec.hwaccel,
        shader,
        shader_sig: entry.shader_sig,
        transition: entry.last_transition.0.clone(),
        duration_ms: entry.last_transition.1,
        params_bits: entry.spec.params.map(norm_f32_bits),
        textures,
        uniforms,
    };
    (fp, media_ok && shader_ok && tex_ok)
}

fn fingerprint_for_tx(
    tx: &TxParams,
    pre: &ResolvedPreflightInputs,
    current: &EffectiveSpecFingerprint,
) -> EffectiveSpecFingerprint {
    let (target, target_sig) = if let Some(ref media) = tx.media {
        (normalize_optional_path(Some(&media.target)), pre.target_sig)
    } else {
        (current.target.clone(), current.target_sig)
    };
    let fit = tx.media.as_ref().and_then(|m| m.fit).unwrap_or(current.fit);
    let fps_bits = tx
        .media
        .as_ref()
        .and_then(|m| m.fps)
        .map(norm_f32_bits)
        .or(current.fps_bits);
    let hwaccel = tx
        .media
        .as_ref()
        .and_then(|m| m.hwaccel)
        .unwrap_or(current.hwaccel);

    let (shader, shader_sig) = match &pre.shader_request {
        ShaderRequest::Keep => (current.shader.clone(), current.shader_sig),
        ShaderRequest::Clear => (None, None),
        ShaderRequest::Set { path, .. } => (
            normalize_optional_path(Some(&path.to_string_lossy())),
            pre.shader_sig,
        ),
    };

    let (transition, duration_ms) = pre
        .transition_spec
        .clone()
        .unwrap_or_else(|| (current.transition.clone(), current.duration_ms));
    let params_bits = tx
        .params
        .map(|p| p.map(norm_f32_bits))
        .unwrap_or(current.params_bits);

    let mut textures = if matches!(pre.shader_request, ShaderRequest::Clear) {
        BTreeMap::new()
    } else {
        current.textures.clone()
    };
    let mut uniforms = if matches!(pre.shader_request, ShaderRequest::Clear) {
        BTreeMap::new()
    } else {
        current.uniforms.clone()
    };
    for (k, opt) in &pre.normalized_textures {
        match opt {
            Some((spec, key)) => {
                textures.insert(k.clone(), texture_spec_fp_entry(spec, key));
            }
            None => {
                textures.remove(k);
            }
        }
    }
    for (k, v) in &tx.uniforms {
        uniforms.insert(k.clone(), UniformFingerprint::from(v));
    }

    EffectiveSpecFingerprint {
        target,
        target_sig,
        fit,
        fps_bits,
        hwaccel,
        shader,
        shader_sig,
        transition,
        duration_ms,
        params_bits,
        textures,
        uniforms,
    }
}

fn resolve_preflight_inputs(
    tx: &TxParams,
    output_name: &str,
    entry_revision: u64,
    entry_spec: &OutputSpec,
    slot0_present: bool,
    effective_max_dim: u32,
    max_image_pixels: u64,
) -> Result<ResolvedPreflightInputs> {
    let next_revision = match tx.revision {
        Some(r) => {
            if r < entry_revision {
                return Err(coded_err(
                    IpcErrorCode::RevisionConflict,
                    format!("non-monotonic revision {r} < {entry_revision} on {output_name}"),
                ));
            }
            r
        }
        None => entry_revision + 1,
    };

    if let Some(params) = tx.params
        && !params.iter().all(|v| v.is_finite())
    {
        return Err(coded_err(
            IpcErrorCode::InvalidInput,
            "params contain non-finite float value",
        ));
    }

    let mut resolved_media = None;
    let mut target_sig = None;
    let mut shader_sig = None;
    let mut shader_mode_request = tx.shader.clone();
    let mut clear_implicit_shader = false;
    let mut estimated_media_bytes = 0u64;

    if let Some(ref media) = tx.media {
        let trimmed = media.target.trim();
        if trimmed.is_empty() {
            return Err(coded_err(
                IpcErrorCode::InvalidInput,
                "media target path cannot be empty",
            ));
        }
        let path = absolutize_path(Path::new(trimmed));
        if !path.is_file() {
            return Err(coded_err(
                IpcErrorCode::MissingSource,
                format!("wallpaper file does not exist: {}", path.display()),
            ));
        }
        let kind = MediaKind::from_path(&path).ok_or_else(|| {
            coded_err(
                IpcErrorCode::UnsupportedMedia,
                format!("unsupported wallpaper file type: {}", path.display()),
            )
        })?;
        target_sig = file_stat_sig(&path);
        if kind == MediaKind::ProceduralShader {
            shader_mode_request = Some(path.to_string_lossy().to_string());
        } else {
            if matches!(
                kind,
                MediaKind::StaticImage
                    | MediaKind::FfmpegStaticImage
                    | MediaKind::AnimatedGif
                    | MediaKind::AnimatedWebp
            ) {
                let (mw, mh) = check_image_limits(&path, effective_max_dim, max_image_pixels)?;
                estimated_media_bytes = u64::from(mw)
                    .saturating_mul(u64::from(mh))
                    .saturating_mul(4);
            }
            if shader_mode_request.is_none()
                && entry_spec
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

    let shader_request = if let Some(ref raw_sh) = shader_mode_request {
        let trimmed = raw_sh.trim();
        if trimmed.is_empty() || trimmed == "none" {
            ShaderRequest::Clear
        } else {
            let resolved_sh = absolutize_path(Path::new(trimmed));
            if !resolved_sh.is_file() {
                return Err(coded_err(
                    IpcErrorCode::MissingSource,
                    format!("shader file does not exist: {}", resolved_sh.display()),
                ));
            }
            let source = fs::read_to_string(&resolved_sh).map_err(|e| {
                coded_err(
                    if e.kind() == ErrorKind::NotFound {
                        IpcErrorCode::MissingSource
                    } else {
                        IpcErrorCode::InvalidInput
                    },
                    format!("failed to read shader {}: {e}", resolved_sh.display()),
                )
            })?;
            shader_sig = file_stat_sig(&resolved_sh);
            ShaderRequest::Set {
                path: resolved_sh,
                source,
            }
        }
    } else if clear_implicit_shader {
        ShaderRequest::Clear
    } else {
        ShaderRequest::Keep
    };

    let mut normalized_textures = BTreeMap::new();
    let mut estimated_aux_bytes = HashMap::new();
    for (name, opt_spec) in &tx.textures {
        let clean_name = name.trim();
        if clean_name.is_empty() {
            return Err(coded_err(
                IpcErrorCode::InvalidInput,
                "empty texture uniform name",
            ));
        }
        if let Some(spec) = opt_spec {
            let abs_path = absolutize_path(&spec.path);
            if !abs_path.is_file() {
                return Err(coded_err(
                    IpcErrorCode::MissingSource,
                    format!(
                        "texture '{clean_name}' file does not exist: {}",
                        abs_path.display()
                    ),
                ));
            }
            let (tw, th) = check_image_limits(&abs_path, effective_max_dim, max_image_pixels)?;
            let abs_spec = TextureSpec {
                path: abs_path,
                filter: spec.filter,
                wrap: spec.wrap,
                flip_y: spec.flip_y,
            };
            let key = TextureKey::from_spec(&abs_spec);
            estimated_aux_bytes.insert(
                key.clone(),
                u64::from(tw)
                    .saturating_mul(u64::from(th))
                    .saturating_mul(4),
            );
            normalized_textures.insert(clean_name.to_string(), Some((abs_spec, key)));
        } else {
            normalized_textures.insert(clean_name.to_string(), None);
        }
    }

    let transition_spec = tx.transition.as_ref().map(|tr| {
        (
            tr.name
                .clone()
                .unwrap_or_else(|| entry_spec.transition.clone()),
            tr.duration_ms.unwrap_or(entry_spec.duration_ms),
        )
    });
    let (tr_name, tr_dur) = transition_spec
        .clone()
        .unwrap_or_else(|| (entry_spec.transition.clone(), entry_spec.duration_ms));

    let transition_source =
        if resolved_media.is_some() && slot0_present && tr_dur > 0 && tr_name != "none" {
            resolve_transition_source(&tr_name).map_err(map_transition_source_err)?
        } else {
            None
        };

    Ok(ResolvedPreflightInputs {
        next_revision,
        resolved_media,
        target_sig,
        shader_sig,
        shader_request,
        clear_implicit_shader,
        transition_spec,
        transition_source,
        normalized_textures,
        estimated_media_bytes,
        estimated_aux_bytes,
    })
}

#[allow(clippy::too_many_arguments)]
fn stage_output_transaction<F>(
    tx: &TxParams,
    pre: &ResolvedPreflightInputs,
    output_name: &str,
    entry_generation: u64,
    entry_spec: &OutputSpec,
    existing_program: Option<&GlProgram>,
    paused: bool,
    default_fps_cap: Option<f32>,
    media_tx: &Sender<MediaEvent>,
    prepared_slot: &mut Option<PreparedTransaction>,
    mut compile_prog: F,
) -> Result<()>
where
    F: FnMut(&str) -> Result<GlProgram>,
{
    let initial_shader = match pre.shader_request {
        ShaderRequest::Keep => StagedShader::Keep,
        ShaderRequest::Clear | ShaderRequest::Set { .. } => StagedShader::Clear,
    };
    let clear_transient = tx.media.is_some() || !matches!(pre.shader_request, ShaderRequest::Keep);

    *prepared_slot = Some(PreparedTransaction {
        next_revision: pre.next_revision,
        shader: initial_shader,
        target_sig: pre.target_sig,
        shader_sig: pre.shader_sig,
        resolved_media: pre.resolved_media.clone(),
        spawned_job: None,
        transition_spec: pre.transition_spec.clone(),
        applied_transition: pre.transition_spec.clone(),
        clear_transient,
        transition_prog: None,
        texture_updates: pre.normalized_textures.clone(),
        uniform_updates: tx.uniforms.clone(),
        params_update: tx.params,
        clear_implicit_shader: pre.clear_implicit_shader,
    });
    let prep = prepared_slot.as_mut().unwrap();

    if let ShaderRequest::Set { path, source } = &pre.shader_request {
        let prog = compile_prog(source)
            .map_err(|e| coded_err(IpcErrorCode::ShaderCompile, format!("{e:#}")))?;
        prep.shader = StagedShader::Set {
            path: path.clone(),
            program: Box::new(prog),
        };
    }

    let active_prog_ref = match &prep.shader {
        StagedShader::Set { program, .. } => Some(program.as_ref()),
        StagedShader::Clear => None,
        StagedShader::Keep => existing_program,
    };
    if let Some(prog) = active_prog_ref {
        for (name, val) in &tx.uniforms {
            prog.validate_uniform(name, val)
                .map_err(|e| coded_err(IpcErrorCode::TypeMismatch, format!("{e:#}")))?;
        }
    }

    if let Some(tr_src) = &pre.transition_source {
        let tr_prog = compile_prog(tr_src)
            .map_err(|e| coded_err(IpcErrorCode::ShaderCompile, format!("{e:#}")))?;
        prep.transition_prog = Some(tr_prog);
    }

    if let Some((resolved_target, kind, media)) = &prep.resolved_media {
        let job = MediaJob::spawn(
            output_name.to_string(),
            entry_generation + 1,
            resolved_target,
            *kind,
            media.hwaccel.unwrap_or(entry_spec.hwaccel),
            media.fps.or(entry_spec.fps).or(default_fps_cap),
            paused,
            media_tx.clone(),
        )
        .map_err(|e| coded_err(IpcErrorCode::Unavailable, format!("{e:#}")))?;
        prep.spawned_job = Some(job);
    }

    Ok(())
}

#[cfg(test)]
fn abort_prepared_transactions(
    prepared: Vec<(usize, PreparedTransaction)>,
    gl_opt: Option<&glow::Context>,
) {
    for (_, prep) in prepared {
        prep.destroy_staged(gl_opt);
    }
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

    fn release_gpu_parts(&mut self, gpu: &mut DeferredOutputGpu) {
        if let Some(pending) = gpu.pending_apply.take() {
            self.release_pending_bound(pending);
        }
        if let Some(r) = &mut self.renderer {
            r.remove_output_cache(&gpu.name);
            std::mem::replace(&mut gpu.slot0, MediaSlotGpu::empty()).destroy(&r.gl);
            if let Some(slot) = gpu.slot1.take() {
                slot.destroy(&r.gl);
            }
            if let Some(program) = gpu.custom_program.take() {
                program.destroy(&r.gl);
            }
            if let Some(program) = gpu.transition_program.take() {
                program.destroy(&r.gl);
            }
            for (_, id) in std::mem::take(&mut gpu.custom_textures) {
                self.texture_pool.release(&r.gl, id);
            }
        }
    }

    fn release_output_gpu(&mut self, entry: &mut OutputEntry) {
        self.release_gpu_parts(&mut entry.gpu_parts());
    }

    fn drain_deferred_gpu(&mut self) {
        if self.renderer.is_none() {
            return;
        }
        for pending in std::mem::take(&mut self.deferred_applies) {
            self.release_pending_bound(pending);
        }
        for mut gpu in std::mem::take(&mut self.deferred_outputs) {
            self.release_gpu_parts(&mut gpu);
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

    fn cancel_pending_single(
        &mut self,
        idx: usize,
        reason: &str,
        revert_media: bool,
    ) -> Option<String> {
        let out_name = self.outputs[idx].name.clone();
        let pending = self.outputs[idx].pending_apply.take();
        let mut group_id = None;
        let mut awaiting_gen = None;
        if let Some(p) = pending {
            group_id = p.group_id.clone();
            awaiting_gen = p.awaiting_media_gen;
            self.release_pending_apply(p, Some(reason));
        }
        self.outputs[idx].pending_generation = None;

        if revert_media {
            let entry = &mut self.outputs[idx];
            if entry.waiting_target_frame && awaiting_gen == Some(entry.generation) {
                entry.waiting_target_frame = false;
                if let Some(failed_job) = entry.job.take() {
                    failed_job.stop();
                }
                entry.job = entry.prev_job.take();
                if let Some(ref job) = entry.job {
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
            } else if awaiting_gen == Some(entry.generation)
                && let Some(job) = entry.job.take()
            {
                job.stop();
            }
        }

        let code = if reason.contains("superseded")
            || reason.contains("aborted")
            || reason.contains("removed")
        {
            IpcErrorCode::Canceled
        } else {
            IpcErrorCode::RenderFailed
        };
        self.fail_commit_waiters_for_output(&out_name, code, reason);
        group_id
    }

    fn cancel_batch_siblings(&mut self, group_id: &str, failed_idx: usize, reason: &str) {
        let siblings: Vec<usize> = self
            .outputs
            .iter()
            .enumerate()
            .filter_map(|(i, o)| {
                (i != failed_idx
                    && o.pending_apply.as_ref().and_then(|p| p.group_id.as_deref())
                        == Some(group_id))
                .then_some(i)
            })
            .collect();
        let abort_msg = format!("batch '{group_id}' aborted: {reason}");
        for j in siblings {
            self.cancel_pending_single(j, &abort_msg, true);
        }
    }

    fn cancel_output_pending(&mut self, idx: usize, reason: &str) {
        if let Some(gid) = self.cancel_pending_single(idx, reason, false) {
            self.cancel_batch_siblings(&gid, idx, reason);
        }
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
                    &IpcResponse::err_coded(
                        IpcErrorCode::Timeout,
                        "timed out waiting for frame commit",
                    ),
                );
            } else {
                i += 1;
            }
        }
    }

    fn fail_commit_waiters_for_output(
        &mut self,
        output_name: &str,
        code: IpcErrorCode,
        message: &str,
    ) {
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
                    &IpcResponse::err_coded(code, format!("{output_name}: {message}")),
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
                let mut payload = json!({
                    "revision": waiter.max_revision,
                    "frame": frame,
                    "outputs": outputs,
                });
                if let Some(ref gid) = waiter.group_id {
                    payload["group_id"] = json!(gid);
                }
                let resp = IpcResponse::ok_data(payload);
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
        if self.outputs[idx].pending_apply.is_some() {
            return;
        }
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

        let needs_materialize = {
            let entry = &self.outputs[idx];
            should_materialize_initial_spec(
                &desired,
                &entry.spec,
                entry.slot0.kind != SlotMediaKind::None,
                entry.job.is_some(),
                entry.custom_program.is_some(),
                entry.custom_textures.len(),
                &entry.custom_uniforms,
                entry.pending_apply.is_some(),
            )
        };
        if !needs_materialize {
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

        // Only update non-resource defaults now so target/shader/bindings remain
        // unmaterialized until execute_transaction actually stages them.
        {
            let entry = &mut self.outputs[idx];
            let spec = &mut entry.spec;
            spec.fit = fit;
            spec.transition = transition.clone();
            spec.duration_ms = duration_ms;
            spec.params = params;
            spec.fps = fps;
            spec.hwaccel = hwaccel;
            entry.last_transition = (transition.clone(), duration_ms);
        }

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
        let wait_commit = tx.wait_commit;
        self.execute_transaction_group(vec![tx], None, wait_commit, waiter_client, qh)
    }

    fn execute_batch_transaction(
        &mut self,
        group_id: String,
        wait_commit: bool,
        outputs: Vec<BatchOutputApply>,
        waiter_client: Option<IpcClientId>,
        qh: &wayland_client::QueueHandle<WaylandContext<Self>>,
    ) -> Result<DispatchOutcome> {
        if group_id.trim().is_empty() {
            return Err(coded_err(
                IpcErrorCode::InvalidInput,
                "apply_batch group_id cannot be empty",
            ));
        }
        if outputs.is_empty() {
            return Err(coded_err(
                IpcErrorCode::InvalidInput,
                "apply_batch requires at least one output entry",
            ));
        }
        if self.outputs.iter().any(|o| {
            o.pending_apply.as_ref().and_then(|p| p.group_id.as_deref()) == Some(group_id.as_str())
        }) {
            return Err(coded_err(
                IpcErrorCode::RevisionConflict,
                format!("batch group_id '{group_id}' is already in flight"),
            ));
        }

        let txs: Vec<TxParams> = outputs
            .into_iter()
            .map(|o| TxParams {
                output: o.output,
                revision: o.revision,
                media: o.media,
                shader: o.shader,
                textures: o.textures,
                uniforms: o.uniforms,
                params: None,
                transition: o.transition,
                persist_transition_defaults: false,
                persist: true,
                wait_commit,
            })
            .collect();

        self.execute_transaction_group(txs, Some(group_id), wait_commit, waiter_client, qh)
    }

    fn execute_transaction_group(
        &mut self,
        txs: Vec<TxParams>,
        group_id: Option<String>,
        wait_commit: bool,
        waiter_client: Option<IpcClientId>,
        qh: &wayland_client::QueueHandle<WaylandContext<Self>>,
    ) -> Result<DispatchOutcome> {
        enum PreflightSlot {
            Idempotent {
                idx: usize,
                revision: u64,
                out_name: String,
            },
            Apply {
                idx: usize,
                tx: Box<TxParams>,
                pre: Box<ResolvedPreflightInputs>,
            },
        }

        let mut expanded: Vec<(usize, TxParams)> = Vec::new();
        let mut seen_outputs: HashSet<usize> = HashSet::new();
        for tx in txs {
            let targets = self.find_outputs_matching(&tx.output);
            if targets.is_empty() {
                return Err(coded_err(
                    IpcErrorCode::InvalidInput,
                    format!("no connected Wayland output matches '{}'", tx.output),
                ));
            }
            for &idx in &targets {
                if group_id.is_some() && !seen_outputs.insert(idx) {
                    return Err(coded_err(
                        IpcErrorCode::InvalidInput,
                        format!(
                            "duplicate output '{}' in apply_batch",
                            self.outputs[idx].name
                        ),
                    ));
                }
            }
            for &idx in &targets {
                expanded.push((idx, tx.clone()));
            }
        }

        let mut any_media = false;
        for (_, tx) in &expanded {
            if tx.media.is_some() {
                any_media = true;
            }
            for (name, val) in &tx.uniforms {
                if !val.is_finite() {
                    return Err(coded_err(
                        IpcErrorCode::InvalidInput,
                        format!("uniform '{name}' has non-finite value"),
                    ));
                }
            }
        }
        if self.renderer.is_some() && any_media {
            self.ensure_any_gl_context()
                .map_err(|e| coded_err(IpcErrorCode::Unavailable, format!("{e:#}")))?;
        }

        let gl_max = self
            .renderer
            .as_ref()
            .map_or(u32::MAX, |r| r.max_texture_size());
        let effective_max_dim = self.config.max_texture_dimension.max(1).min(gl_max);
        let max_image_pixels = self.config.max_image_pixels;

        let mut slots: Vec<PreflightSlot> = Vec::with_capacity(expanded.len());
        let mut new_aux_bytes: HashMap<TextureKey, u64> = HashMap::new();
        let mut replacement_media_bytes: HashMap<usize, u64> = HashMap::new();
        let mut has_new_media_or_aux = false;

        for (idx, tx) in expanded {
            let (entry_name, entry_revision, entry_spec, slot0_present, cur_fp, materialized) = {
                let entry = &self.outputs[idx];
                let (fp, mat) = fingerprint_from_output_entry(entry);
                (
                    entry.name.clone(),
                    entry.revision,
                    entry.spec.clone(),
                    entry.slot0.kind != SlotMediaKind::None,
                    fp,
                    mat,
                )
            };

            let pre = resolve_preflight_inputs(
                &tx,
                &entry_name,
                entry_revision,
                &entry_spec,
                slot0_present,
                effective_max_dim,
                max_image_pixels,
            )?;

            if tx.revision == Some(entry_revision) {
                let proj_fp = fingerprint_for_tx(&tx, &pre, &cur_fp);
                if materialized && proj_fp == cur_fp {
                    slots.push(PreflightSlot::Idempotent {
                        idx,
                        revision: entry_revision,
                        out_name: entry_name,
                    });
                    continue;
                }
                return Err(coded_err(
                    IpcErrorCode::RevisionConflict,
                    format!(
                        "revision {entry_revision} on {entry_name} conflicts with active state"
                    ),
                ));
            }

            if pre.estimated_media_bytes > 0 {
                has_new_media_or_aux = true;
                replacement_media_bytes.insert(idx, pre.estimated_media_bytes);
            }
            for (k, bytes) in &pre.estimated_aux_bytes {
                if !self.texture_pool.contains_key(k) && !self.inflight_tex_decodes.contains(k) {
                    has_new_media_or_aux = true;
                    new_aux_bytes.insert(k.clone(), *bytes);
                }
            }
            slots.push(PreflightSlot::Apply {
                idx,
                tx: Box::new(tx),
                pre: Box::new(pre),
            });
        }

        // If every target output matched current state idempotently, return without mutating state.
        if slots
            .iter()
            .all(|s| matches!(s, PreflightSlot::Idempotent { .. }))
        {
            let mut max_rev = 0u64;
            let mut out_names = Vec::with_capacity(slots.len());
            let mut wait_targets = Vec::new();
            let mut completed = HashSet::new();
            let mut last_frame = 0i32;

            for s in &slots {
                if let PreflightSlot::Idempotent {
                    idx,
                    revision,
                    out_name,
                } = s
                {
                    max_rev = max_rev.max(*revision);
                    out_names.push(out_name.clone());
                    let entry = &self.outputs[*idx];
                    if entry.pending_apply.is_none() && !entry.surface.dirty {
                        completed.insert(out_name.clone());
                        last_frame = entry.frame_index;
                    }
                    wait_targets.push((out_name.clone(), *revision));
                }
            }

            if wait_commit
                && completed.len() < wait_targets.len()
                && let Some(client_id) = waiter_client
            {
                self.ipc_hub.retain_pending_ack(client_id);
                self.commit_waiters.push(CommitWaiter {
                    client_id,
                    group_id,
                    revision_targets: wait_targets,
                    completed,
                    max_revision: max_rev,
                    deadline: Instant::now() + WAIT_COMMIT_TIMEOUT,
                });
                self.arm_commit_timeout_timer();
                return Ok(DispatchOutcome::Deferred);
            }

            let mut payload = json!({
                "revision": max_rev,
                "idempotent": true,
                "outputs": out_names,
            });
            if completed.len() == out_names.len() {
                payload["frame"] = json!(last_frame);
            }
            if let Some(ref gid) = group_id {
                payload["group_id"] = json!(gid);
            }
            return Ok(DispatchOutcome::Reply(IpcResponse::ok_data(payload)));
        }

        // Check auxiliary decode queue capacity and projected GPU texture memory budget across the batch.
        if !new_aux_bytes.is_empty() && !aux_decode_has_capacity(new_aux_bytes.len()) {
            return Err(coded_err(
                IpcErrorCode::ResourceLimit,
                format!(
                    "auxiliary texture decode queue capacity exceeded by {} new textures",
                    new_aux_bytes.len()
                ),
            ));
        }
        if self.config.max_gpu_texture_bytes > 0 && has_new_media_or_aux {
            let r_opt = self.renderer.as_ref();
            let mut projected_gpu = self.texture_pool.estimated_gpu_bytes();
            projected_gpu =
                projected_gpu.saturating_add(new_aux_bytes.values().copied().sum::<u64>());
            for (o_idx, entry) in self.outputs.iter().enumerate() {
                let resolve = r_opt.map_or(0, |r| r.resolve_target_gpu_bytes(&entry.name));
                let media_bytes = match replacement_media_bytes.get(&o_idx) {
                    Some(new_b) => *new_b,
                    None => entry.slot0.estimated_gpu_bytes(),
                };
                projected_gpu = projected_gpu
                    .saturating_add(media_bytes)
                    .saturating_add(entry.slot1.as_ref().map_or(0, |s| s.estimated_gpu_bytes()))
                    .saturating_add(resolve);
            }
            if projected_gpu > self.config.max_gpu_texture_bytes {
                return Err(coded_err(
                    IpcErrorCode::ResourceLimit,
                    format!(
                        "projected GPU texture memory ({projected_gpu} bytes) exceeds max_gpu_texture_bytes ({})",
                        self.config.max_gpu_texture_bytes
                    ),
                ));
            }
        }

        // Pre-validate and compile everything per-output before mutating any live state.
        let mut prepared: Vec<(usize, TxParams, PreparedTransaction)> = Vec::new();
        let mut idempotent_wait_targets: Vec<(String, u64)> = Vec::new();
        let mut max_rev = 0u64;

        let preflight_res: Result<()> = (|| {
            for slot in slots {
                match slot {
                    PreflightSlot::Idempotent {
                        revision, out_name, ..
                    } => {
                        max_rev = max_rev.max(revision);
                        idempotent_wait_targets.push((out_name, revision));
                    }
                    PreflightSlot::Apply { idx, tx, pre } => {
                        if pre.needs_gl() {
                            self.ensure_any_gl_context().map_err(|e| {
                                coded_err(IpcErrorCode::Unavailable, format!("{e:#}"))
                            })?;
                        }
                        let (entry_name, entry_generation, entry_spec, paused) = {
                            let entry = &self.outputs[idx];
                            (
                                entry.name.clone(),
                                entry.generation,
                                entry.spec.clone(),
                                self.pause.is_paused(&entry.name),
                            )
                        };
                        let gl_opt = self.renderer.as_ref().map(|r| &r.gl);
                        let existing_program = self.outputs[idx].custom_program.as_ref();

                        let mut stage_slot = None;
                        let res = stage_output_transaction(
                            &tx,
                            &pre,
                            &entry_name,
                            entry_generation,
                            &entry_spec,
                            existing_program,
                            paused,
                            self.config.fps_cap,
                            &self.media_tx,
                            &mut stage_slot,
                            |src| {
                                let gl = gl_opt.ok_or_else(|| {
                                    coded_err(IpcErrorCode::Unavailable, "renderer not initialized")
                                })?;
                                GlProgram::compile(gl, src)
                            },
                        );
                        if let Some(prep) = stage_slot {
                            prepared.push((idx, *tx, prep));
                        }
                        res?;
                    }
                }
            }
            Ok(())
        })();

        if let Err(err) = preflight_res {
            let _ = self.ensure_any_gl_context();
            let gl_opt = self.renderer.as_ref().map(|r| &r.gl);
            for (_, _, prep) in prepared {
                prep.destroy_staged(gl_opt);
            }
            return Err(err);
        }

        // Pre-flight succeeded! Stage PendingApply on ALL affected outputs BEFORE trying any commits
        // so batch barriers see every sibling PendingApply atomically.
        let mut wait_targets = idempotent_wait_targets;
        let mut keys_to_spawn: Vec<(usize, TextureKey)> = Vec::new();
        let mut staged_indices: Vec<usize> = Vec::with_capacity(prepared.len());
        let mut stopped_animated = false;

        for (idx, tx, mut prep) in prepared {
            self.cancel_output_pending(idx, "superseded by newer apply");

            let is_standalone_shader = tx.media.as_ref().is_some_and(|m| {
                let p = absolutize_path(Path::new(&m.target));
                MediaKind::from_path(&p) == Some(MediaKind::ProceduralShader)
            });

            if let Some((tr_name, tr_dur)) = prep.transition_spec.clone()
                && tx.persist_transition_defaults
            {
                let entry = &mut self.outputs[idx];
                entry.spec.transition = tr_name;
                entry.spec.duration_ms = tr_dur;
            }

            let mut target_override: Option<String> = None;
            let mut awaiting_media_gen: Option<u64> = None;

            if is_standalone_shader {
                if let Some(job) = prep.spawned_job.take() {
                    job.stop();
                }
                if let Some(ref r) = self.renderer {
                    self.outputs[idx].slot0.clear(&r.gl);
                } else {
                    self.outputs[idx].slot0 = MediaSlotGpu::empty();
                }
                let entry = &mut self.outputs[idx];
                entry.generation += 1;
                if let Some(job) = entry.job.take() {
                    stopped_animated |= job.kind.is_animated();
                    job.stop();
                }
                if let Some(prev) = entry.prev_job.take() {
                    stopped_animated |= prev.kind.is_animated();
                    prev.stop();
                }
                if let Some(ref r) = self.renderer {
                    if let Some(s1) = entry.slot1.take() {
                        s1.destroy(&r.gl);
                    }
                    if let Some(tr) = entry.transition_program.take() {
                        tr.destroy(&r.gl);
                    }
                } else {
                    if let Some(s1) = entry.slot1.take() {
                        self.deferred_slots.push(s1);
                    }
                    if let Some(tr) = entry.transition_program.take() {
                        self.deferred_programs.push(tr);
                    }
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
                let need_resolve = entry.needs_resolve_target();
                let name = entry.name.clone();
                if !need_resolve && let Some(ref mut r) = self.renderer {
                    r.remove_output_cache(&name);
                }
            } else if let Some((resolved_target, _kind, media)) = prep.resolved_media {
                if prep.transition_prog.is_none() {
                    if let Some(r) = &self.renderer {
                        self.outputs[idx].slot0.clear(&r.gl);
                    } else {
                        self.outputs[idx].slot0 = MediaSlotGpu::empty();
                    }
                }
                let entry = &mut self.outputs[idx];
                entry.generation += 1;
                let generation = entry.generation;

                if let Some(prev) = entry.prev_job.take() {
                    stopped_animated |= prev.kind.is_animated();
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
                        stopped_animated |= old_job.kind.is_animated();
                        old_job.stop();
                    }
                    entry.waiting_target_frame = false;
                    entry.in_transition = false;
                    entry.transition_started = None;
                    let need_resolve = entry.needs_resolve_target();
                    let name = entry.name.clone();
                    if !need_resolve && let Some(ref mut r) = self.renderer {
                        r.remove_output_cache(&name);
                    }
                }

                entry.slot0_has_first_frame = false;
                entry.job = prep.spawned_job.take();
                entry.last_tick = Instant::now();
                awaiting_media_gen = Some(generation);
            } else if let Some(tr) = prep.transition_prog
                && let Some(ref r) = self.renderer
            {
                tr.destroy(&r.gl);
            }

            let mut acquired_textures = HashMap::new();
            let mut pending_keys = HashSet::new();
            for (_, key) in prep.texture_updates.values().flatten() {
                if acquired_textures.contains_key(key) || pending_keys.contains(key) {
                    continue;
                }
                if let Some(id) = self.texture_pool.acquire(key) {
                    acquired_textures.insert(key.clone(), id);
                } else if pending_keys.insert(key.clone())
                    && self.inflight_tex_decodes.insert(key.clone())
                {
                    keys_to_spawn.push((idx, key.clone()));
                }
            }

            let out_name = self.outputs[idx].name.clone();
            max_rev = max_rev.max(prep.next_revision);
            wait_targets.push((out_name, prep.next_revision));
            self.outputs[idx].pending_generation = Some(self.outputs[idx].generation);
            self.outputs[idx].pending_apply = Some(PendingApply {
                revision: prep.next_revision,
                group_id: group_id.clone(),
                persist: tx.persist,
                shader: prep.shader,
                target_sig: prep.target_sig,
                shader_sig: prep.shader_sig,
                applied_transition: prep.applied_transition,
                clear_transient: prep.clear_transient,
                texture_updates: prep.texture_updates,
                acquired_textures,
                pending_keys,
                uniform_updates: prep.uniform_updates,
                params_update: prep.params_update,
                target_override,
                clear_implicit_shader: prep.clear_implicit_shader,
                awaiting_media_gen,
            });
            staged_indices.push(idx);
        }

        for (owner_idx, key) in keys_to_spawn {
            if let Err(err) = decode_texture_async(key.clone(), self.media_tx.clone()) {
                self.inflight_tex_decodes.remove(&key);
                let msg = format!("{err:#}");
                self.cancel_output_pending(owner_idx, &msg);
                return Err(err);
            }
        }

        for idx in staged_indices {
            if self.outputs[idx].pending_apply.is_some() {
                self.try_commit_pending_apply(idx, qh);
            }
        }

        if stopped_animated {
            crate::media::gst_cuda::reclaim_idle_resources();
            unsafe { libc::malloc_trim(0) };
        }

        if wait_commit && let Some(client_id) = waiter_client {
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
                let mut payload = json!({
                    "revision": max_rev,
                    "frame": last_frame,
                    "outputs": outs,
                });
                if let Some(ref gid) = group_id {
                    payload["group_id"] = json!(gid);
                }
                return Ok(DispatchOutcome::Reply(IpcResponse::ok_data(payload)));
            }
            self.ipc_hub.retain_pending_ack(client_id);
            self.commit_waiters.push(CommitWaiter {
                client_id,
                group_id,
                revision_targets: wait_targets,
                completed,
                max_revision: max_rev,
                deadline: Instant::now() + WAIT_COMMIT_TIMEOUT,
            });
            self.arm_commit_timeout_timer();
            return Ok(DispatchOutcome::Deferred);
        }

        let mut payload = json!({
            "revision": max_rev,
        });
        if let Some(ref gid) = group_id {
            payload["group_id"] = json!(gid);
        }
        Ok(DispatchOutcome::Reply(IpcResponse::ok_data(payload)))
    }

    fn try_commit_pending_apply(
        &mut self,
        idx: usize,
        qh: &wayland_client::QueueHandle<WaylandContext<Self>>,
    ) {
        let Some(pending_ref) = self.outputs[idx].pending_apply.as_ref() else {
            return;
        };
        let entry = &self.outputs[idx];
        if !pending_ref.is_ready(entry.generation, entry.slot0_has_first_frame) {
            return;
        }

        let group_indices: Vec<usize> = match pending_ref.group_id.as_deref() {
            Some(gid) => {
                let members: Vec<usize> = self
                    .outputs
                    .iter()
                    .enumerate()
                    .filter_map(|(i, o)| {
                        (o.pending_apply.as_ref().and_then(|p| p.group_id.as_deref()) == Some(gid))
                            .then_some(i)
                    })
                    .collect();
                for &m_idx in &members {
                    let m_entry = &self.outputs[m_idx];
                    let ready = m_entry.pending_apply.as_ref().is_some_and(|p| {
                        p.is_ready(m_entry.generation, m_entry.slot0_has_first_frame)
                    });
                    if !ready {
                        return;
                    }
                }
                members
            }
            None => vec![idx],
        };

        if self.renderer.is_some()
            && let Err(err) = self.ensure_any_gl_context()
        {
            tracing::warn!("deferring pending apply until EGL context is available: {err:#}");
            return;
        }

        let mut shader_changed_any = false;
        let mut caches_to_clear = Vec::new();
        let mut persist_any = false;

        for &m_idx in &group_indices {
            let Some(mut pending) = self.outputs[m_idx].pending_apply.take() else {
                continue;
            };

            let (shader_changed, needs_resolve, persist_now, out_name, spec_snapshot) = {
                let gl_opt = self.renderer.as_ref().map(|r| &r.gl);
                let entry = &mut self.outputs[m_idx];
                let mut sh_changed = false;

                if pending.clear_transient {
                    entry.transient.clear();
                }

                match pending.shader {
                    StagedShader::Keep => {}
                    StagedShader::Clear => {
                        if let Some(old) = entry.custom_program.take()
                            && let Some(gl) = gl_opt
                        {
                            old.destroy(gl);
                        }
                        entry.custom_shader_path = None;
                        entry.shader_sig = None;
                        entry.last_shader_error = None;
                        if pending.clear_implicit_shader || entry.spec.shader.is_some() {
                            entry.spec.shader = None;
                        }
                        let empty = HashSet::new();
                        entry.transient.prune(&empty);
                        let removed = prune_shader_bindings(
                            &mut entry.custom_textures,
                            &mut entry.custom_texture_keys,
                            &mut entry.spec.textures,
                            &mut entry.custom_uniforms,
                            &mut entry.spec.uniforms,
                            &empty,
                            &empty,
                        );
                        if let Some(gl) = gl_opt {
                            for old_id in removed {
                                self.texture_pool.release(gl, old_id);
                            }
                        }
                        sh_changed = true;
                    }
                    StagedShader::Set { path, program } => {
                        let allowed_uniforms: HashSet<String> = program
                            .user_uniforms
                            .iter()
                            .map(|u| u.name.clone())
                            .collect();
                        entry.transient.prune(&allowed_uniforms);
                        if let Some(gl) = gl_opt {
                            apply_program_and_prune(
                                gl,
                                *program,
                                &mut entry.custom_program,
                                &mut entry.custom_textures,
                                &mut entry.custom_texture_keys,
                                &mut entry.spec.textures,
                                &mut entry.custom_uniforms,
                                &mut entry.spec.uniforms,
                                &mut self.texture_pool,
                            );
                        } else {
                            let allowed_samplers: HashSet<String> = program
                                .user_samplers
                                .iter()
                                .map(|s| s.name.clone())
                                .collect();
                            if let Some(old) = entry.custom_program.replace(*program) {
                                self.deferred_programs.push(old);
                            }
                            let _ = prune_shader_bindings(
                                &mut entry.custom_textures,
                                &mut entry.custom_texture_keys,
                                &mut entry.spec.textures,
                                &mut entry.custom_uniforms,
                                &mut entry.spec.uniforms,
                                &allowed_samplers,
                                &allowed_uniforms,
                            );
                        }
                        let canonical = path.to_string_lossy().to_string();
                        entry.custom_shader_path = Some(path);
                        entry.shader_sig = pending.shader_sig;
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
                    entry.target_sig = pending.target_sig;
                }
                if let Some(tr_pair) = pending.applied_transition.take() {
                    entry.last_transition = tr_pair;
                }

                for (name, opt_entry) in pending.texture_updates {
                    if let Some(old_id) = entry.custom_textures.remove(&name)
                        && let Some(gl) = gl_opt
                    {
                        self.texture_pool.release(gl, old_id);
                    }
                    entry.custom_texture_keys.remove(&name);
                    match opt_entry {
                        Some((spec, key)) => {
                            if let Some(tex_id) = pending
                                .acquired_textures
                                .remove(&key)
                                .or_else(|| self.texture_pool.acquire(&key))
                            {
                                entry.custom_textures.insert(name.clone(), tex_id);
                                entry.custom_texture_keys.insert(name.clone(), key);
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
                entry.pending_generation = None;
                entry.active_generation = entry.generation;

                if entry.waiting_target_frame && entry.slot0_has_first_frame {
                    entry.waiting_target_frame = false;
                    entry.in_transition = true;
                    entry.transition_started = Some(Instant::now());
                }

                entry.surface.dirty = true;
                entry.last_tick = Instant::now();

                (
                    sh_changed,
                    entry.needs_resolve_target(),
                    pending.persist,
                    entry.name.clone(),
                    entry.spec.clone(),
                )
            };

            if shader_changed {
                shader_changed_any = true;
                if !needs_resolve {
                    caches_to_clear.push(out_name.clone());
                }
            }
            if persist_now {
                self.persisted.outputs.insert(out_name, spec_snapshot);
                persist_any = true;
            }
        }

        if let Some(ref mut r) = self.renderer {
            for out_name in caches_to_clear {
                r.remove_output_cache(&out_name);
            }
        }
        if persist_any {
            self.save_state();
        }
        if shader_changed_any {
            self.sync_shader_watcher();
        }
        for m_idx in group_indices {
            self.render_if_ready(m_idx, qh);
        }
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
                    drop(pixels);
                    unsafe { libc::malloc_trim(0) };
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
                let insert_res =
                    self.texture_pool
                        .insert_rgba(gl, key.clone(), width, height, &pixels);
                drop(pixels);
                unsafe { libc::malloc_trim(0) };
                let base_id = match insert_res {
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
            MediaEvent::AuxTextureError { key, code, message } => {
                self.inflight_tex_decodes.remove(&key);
                tracing::error!(
                    "auxiliary texture decode failed [{}] ({}): {message}",
                    code.as_str(),
                    key.path.display()
                );
                let _ = code;
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
                code,
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
                tracing::error!("media error on {output} [{}]: {message}", code.as_str());
                self.outputs[idx].last_media_error = Some(message.clone());
                self.ipc_hub.broadcast_event(&IpcEvent::MediaError {
                    output: output.clone(),
                    code: Some(code),
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
                entry.pending_generation = None;
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

                let is_one_shot = self.outputs[idx]
                    .job
                    .as_ref()
                    .is_some_and(|j| j.generation == generation && !j.kind.is_animated());
                let is_cuda = matches!(&payload, FramePayload::Cuda(_));
                if is_cuda
                    && self.outputs[idx]
                        .job
                        .as_ref()
                        .is_some_and(|j| j.control.cuda_failed())
                {
                    if is_one_shot {
                        drop(payload);
                        drop(mailbox);
                    } else {
                        mailbox.recycle(payload);
                    }
                    return;
                }

                if !matches!(self.ensure_renderer_on_surface(idx), Ok(true)) {
                    if is_one_shot {
                        drop(payload);
                        drop(mailbox);
                    } else {
                        mailbox.recycle(payload);
                    }
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
                if is_one_shot {
                    drop(payload);
                    mailbox.clear_pending();
                    drop(mailbox);
                } else {
                    mailbox.recycle(payload);
                    drop(mailbox);
                }

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
                // whole session. Drop all mailbox/job references BEFORE trimming
                // so freed heap pages actually return to the OS.
                if is_one_shot {
                    if let Some(job) = self.outputs[idx].job.take() {
                        job.mailbox.clear_pending();
                        drop(job);
                    }
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
        let Some(ref mut renderer) = self.renderer else {
            return;
        };
        let mut caches_to_clear = Vec::new();
        for (idx, path) in to_reload {
            let Ok(src) = fs::read_to_string(&path) else {
                continue;
            };
            let reload_res = {
                let entry = &mut self.outputs[idx];
                reload_shader_into_bindings(
                    &renderer.gl,
                    &src,
                    &mut entry.custom_program,
                    &mut entry.custom_textures,
                    &mut entry.custom_texture_keys,
                    &mut entry.spec.textures,
                    &mut entry.custom_uniforms,
                    &mut entry.spec.uniforms,
                    &mut self.texture_pool,
                )
            };
            match reload_res {
                Ok(()) => {
                    let entry = &mut self.outputs[idx];
                    entry.revision += 1;
                    entry.shader_sig = file_stat_sig(&path);
                    entry.last_shader_error = None;
                    entry.surface.dirty = true;
                    if let Some(ref prog) = entry.custom_program {
                        let allowed_uniforms: HashSet<String> =
                            prog.user_uniforms.iter().map(|u| u.name.clone()).collect();
                        entry.transient.prune(&allowed_uniforms);
                    }
                    let out_name = entry.name.clone();
                    let rev = entry.revision;
                    if !entry.needs_resolve_target() {
                        caches_to_clear.push(out_name.clone());
                    }
                    tracing::info!("hot-reloaded shader {} on {}", path.display(), out_name);
                    self.ipc_hub.broadcast_event(&IpcEvent::ShaderReloaded {
                        output: out_name,
                        revision: rev,
                        ok: true,
                        code: None,
                        error: None,
                    });
                }
                Err(err) => {
                    let code = error_code_of(&err, IpcErrorCode::ShaderCompile);
                    let msg = format!("{err:#}");
                    let entry = &mut self.outputs[idx];
                    tracing::warn!(
                        "shader hot-reload compile failed on {} [{}] (keeping active shader): {msg}",
                        entry.name,
                        code.as_str()
                    );
                    entry.last_shader_error = Some(msg.clone());
                    let out_name = entry.name.clone();
                    let rev = entry.revision;
                    self.ipc_hub.broadcast_event(&IpcEvent::ShaderReloaded {
                        output: out_name,
                        revision: rev,
                        ok: false,
                        code: Some(code),
                        error: Some(msg),
                    });
                }
            }
        }
        for out_name in caches_to_clear {
            renderer.remove_output_cache(&out_name);
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
            entry.transient.consume_pending();

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
                transient_uniforms: &entry.transient.active_overlay,
            };
            renderer.draw(&draw_params, &self.texture_pool)
        };

        if let Err(err) = draw_res {
            tracing::error!("GL draw failed on {}: {err}", self.outputs[idx].name);
            return;
        }

        if promote_done {
            let mut stopped_animated = false;
            let (out_name, need_resolve) = {
                let gl = &self.renderer.as_ref().unwrap().gl;
                let entry = &mut self.outputs[idx];
                entry.in_transition = false;
                entry.transition_started = None;
                if let Some(prev_job) = entry.prev_job.take() {
                    stopped_animated = prev_job.kind.is_animated();
                    prev_job.stop();
                }
                if let Some(s1) = entry.slot1.take() {
                    s1.destroy(gl);
                }
                if let Some(tr) = entry.transition_program.take() {
                    tr.destroy(gl);
                }
                (entry.name.clone(), entry.needs_resolve_target())
            };
            if !need_resolve && let Some(ref mut r) = self.renderer {
                r.remove_output_cache(&out_name);
            }
            if stopped_animated {
                crate::media::gst_cuda::reclaim_idle_resources();
            }
            unsafe { libc::malloc_trim(0) };
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
            if swapped_ok {
                entry.last_committed_frame = Some(entry.frame_index);
                entry.last_committed_revision = entry.revision;
                entry.active_generation = entry.generation;
            }
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

    fn handle_uniforms_transient(
        &mut self,
        output: String,
        session: String,
        seq: u64,
        values: BTreeMap<String, UniformValue>,
        clear: bool,
        qh: &wayland_client::QueueHandle<WaylandContext<Self>>,
    ) -> Result<DispatchOutcome> {
        let trimmed_session = session.trim();
        if trimmed_session.is_empty() {
            return Err(coded_err(
                IpcErrorCode::InvalidInput,
                "uniforms_transient session cannot be empty",
            ));
        }
        let targets = self.find_outputs_matching(&output);
        if targets.is_empty() {
            return Err(coded_err(
                IpcErrorCode::InvalidInput,
                format!("no connected Wayland output matches '{output}'"),
            ));
        }
        let values = if clear { BTreeMap::new() } else { values };
        for (name, val) in &values {
            if !val.is_finite() {
                return Err(coded_err(
                    IpcErrorCode::InvalidInput,
                    format!("uniform '{name}' has non-finite value"),
                ));
            }
        }

        // Preflight monotonic seq + schema/type validation across all matched outputs before mutating.
        for &idx in &targets {
            let entry = &self.outputs[idx];
            if entry.transient.session == trimmed_session && seq <= entry.transient.last_seq {
                return Err(coded_err(
                    IpcErrorCode::RevisionConflict,
                    format!(
                        "non-monotonic transient uniform seq {seq} <= {} for session '{trimmed_session}' on {}",
                        entry.transient.last_seq, entry.name
                    ),
                ));
            }
            if !values.is_empty() {
                let prog = entry.custom_program.as_ref().ok_or_else(|| {
                    coded_err(
                        IpcErrorCode::TypeMismatch,
                        format!("output '{}' has no active shader for uniforms", entry.name),
                    )
                })?;
                for (name, val) in &values {
                    if !prog.user_uniforms.iter().any(|u| u.name == *name) {
                        return Err(coded_err(
                            IpcErrorCode::TypeMismatch,
                            format!(
                                "unknown uniform '{name}' for active shader on {}",
                                entry.name
                            ),
                        ));
                    }
                    prog.validate_uniform(name, val)
                        .map_err(|e| coded_err(IpcErrorCode::TypeMismatch, format!("{e:#}")))?;
                }
            }
        }

        let mut any_dirty = false;
        let mut out_names = Vec::with_capacity(targets.len());
        for idx in targets {
            let (changed, name) = {
                let entry = &mut self.outputs[idx];
                let prev_effective = effective_uniforms_with_overlay(
                    &entry.custom_uniforms,
                    entry.transient.current_overlay(),
                );
                let next_effective =
                    effective_uniforms_with_overlay(&entry.custom_uniforms, &values);
                let changed = prev_effective != next_effective;

                entry.transient.session = trimmed_session.to_string();
                entry.transient.last_seq = seq;
                if values.is_empty() {
                    entry.transient.active_overlay.clear();
                    entry.transient.pending_overlay = None;
                } else {
                    entry.transient.pending_overlay = Some(values.clone());
                }
                if changed {
                    entry.surface.dirty = true;
                }
                (changed, entry.name.clone())
            };
            out_names.push(name);
            if changed {
                any_dirty = true;
                self.render_if_ready(idx, qh);
            }
        }

        Ok(DispatchOutcome::Reply(IpcResponse::ok_data(json!({
            "session": trimmed_session,
            "seq": seq,
            "noop": !any_dirty,
            "dirty": any_dirty,
            "outputs": out_names,
        }))))
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
                    Err(err) => DispatchOutcome::Reply(IpcResponse::from_anyhow(
                        &err,
                        IpcErrorCode::InvalidInput,
                    )),
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
                    Err(err) => DispatchOutcome::Reply(IpcResponse::from_anyhow(
                        &err,
                        IpcErrorCode::InvalidInput,
                    )),
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
                    Err(err) => DispatchOutcome::Reply(IpcResponse::from_anyhow(
                        &err,
                        IpcErrorCode::InvalidInput,
                    )),
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
                    Err(err) => DispatchOutcome::Reply(IpcResponse::from_anyhow(
                        &err,
                        IpcErrorCode::InvalidInput,
                    )),
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
                    Err(err) => DispatchOutcome::Reply(IpcResponse::from_anyhow(
                        &err,
                        IpcErrorCode::InvalidInput,
                    )),
                }
            }
            IpcRequest::UniformsTransient {
                output,
                session,
                seq,
                values,
                clear,
            } => match self.handle_uniforms_transient(output, session, seq, values, clear, qh) {
                Ok(out) => out,
                Err(err) => DispatchOutcome::Reply(IpcResponse::from_anyhow(
                    &err,
                    IpcErrorCode::InvalidInput,
                )),
            },
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
                    Err(err) => DispatchOutcome::Reply(IpcResponse::from_anyhow(
                        &err,
                        IpcErrorCode::InvalidInput,
                    )),
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
                    Err(err) => DispatchOutcome::Reply(IpcResponse::from_anyhow(
                        &err,
                        IpcErrorCode::InvalidInput,
                    )),
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
                    Err(err) => DispatchOutcome::Reply(IpcResponse::from_anyhow(
                        &err,
                        IpcErrorCode::InvalidInput,
                    )),
                }
            }
            IpcRequest::ApplyBatch {
                version,
                group_id,
                targets,
                wait_commit,
            } => {
                if version != 1 {
                    return DispatchOutcome::Reply(IpcResponse::err_coded(
                        IpcErrorCode::InvalidInput,
                        format!("unsupported apply_batch version {version} (expected 1)"),
                    ));
                }
                match self.execute_batch_transaction(
                    group_id,
                    wait_commit,
                    targets,
                    Some(client_id),
                    qh,
                ) {
                    Ok(out) => out,
                    Err(err) => DispatchOutcome::Reply(IpcResponse::from_anyhow(
                        &err,
                        IpcErrorCode::InvalidInput,
                    )),
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
                    return DispatchOutcome::Reply(IpcResponse::err_coded(
                        IpcErrorCode::InvalidInput,
                        format!("no connected Wayland output matches '{output}'"),
                    ));
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
                        return DispatchOutcome::Reply(IpcResponse::from_anyhow(
                            &err,
                            IpcErrorCode::InvalidInput,
                        ));
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
                let r_opt = self.renderer.as_ref();
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
                        let paused = self.pause.is_paused(&o.name);
                        let effective_uniforms = effective_uniforms_with_overlay(
                            &o.custom_uniforms,
                            o.transient.current_overlay(),
                        );
                        let output_gpu_bytes = o.estimated_gpu_bytes(r_opt);
                        json!({
                            "name": o.name,
                            "revision": o.revision,
                            "logical_size": [o.surface.logical_w, o.surface.logical_h],
                            "physical_size": [o.surface.phys_w, o.surface.phys_h],
                            "scale": f64::from(o.surface.scale_120) / 120.0,
                            "configured": o.surface.configured,
                            "target": o.spec.target,
                            "media_kind": o.media_kind_label(),
                            "active_generation": o.active_generation,
                            "pending_generation": o.pending_generation,
                            "decoder_status": o.decoder_status(paused),
                            "last_committed_frame": o.last_committed_frame,
                            "last_committed_revision": o.last_committed_revision,
                            "estimated_gpu_bytes": output_gpu_bytes,
                            "shader": o.spec.shader,
                            "shader_schema": o.custom_program.as_ref().map(GlProgram::schema),
                            "textures": bound_textures,
                            "uniforms": effective_uniforms,
                            "uniforms_base": o.custom_uniforms,
                            "uniforms_transient": o.transient.current_overlay(),
                            "fit": o.spec.fit,
                            "params": o.spec.params,
                            "paused": paused,
                            "pause_reasons": self.pause.reasons_for(&o.name),
                            "continuous": o.uses_continuous_frames(),
                            "shader_mode": o.custom_program.as_ref().map(|p| format!("{:?}", p.assembled.mode)),
                            "last_shader_error": o.last_shader_error,
                            "last_media_error": o.last_media_error,
                        })
                    })
                    .collect();
                let (aux_workers_active, queued_aux) = aux_decode_stats();
                let pool_bytes = self.texture_pool.estimated_gpu_bytes();
                let output_bytes: u64 = self
                    .outputs
                    .iter()
                    .map(|o| o.estimated_gpu_bytes(r_opt))
                    .sum();
                let estimated_vram_bytes = pool_bytes.saturating_add(output_bytes);
                let (rss_kb, pss_kb, private_dirty_kb) = read_self_memory_stats();
                DispatchOutcome::Reply(IpcResponse::ok_data(json!({
                    "auto_pause_fullscreen": self.config.auto_pause_fullscreen,
                    "persist_state": self.persist_enabled,
                    "pool_textures": self.texture_pool.len(),
                    "cached_textures": self.texture_pool.len(),
                    "inflight_aux_decodes": self.inflight_tex_decodes.len(),
                    "aux_decode_workers_active": aux_workers_active,
                    "queued_aux_decodes": queued_aux,
                    "active_media_workers": active_media_workers(),
                    "active_gstreamer_pipelines": active_pipeline_count(),
                    "estimated_vram_bytes": estimated_vram_bytes,
                    "memory": {
                        "rss_kb": rss_kb,
                        "pss_kb": pss_kb,
                        "private_dirty_kb": private_dirty_kb,
                        "estimated_vram_bytes": estimated_vram_bytes,
                    },
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
        let mut stopped_animated = false;
        if let Some(job) = entry.job.take() {
            stopped_animated |= job.kind.is_animated();
            job.stop();
        }
        if let Some(prev) = entry.prev_job.take() {
            stopped_animated |= prev.kind.is_animated();
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
            let gpu = entry.gpu_parts();
            entry.surface.teardown(&self.egl);
            self.deferred_outputs.push(gpu);
            if stopped_animated {
                crate::media::gst_cuda::reclaim_idle_resources();
            }
            unsafe { libc::malloc_trim(0) };
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
        if stopped_animated {
            crate::media::gst_cuda::reclaim_idle_resources();
            unsafe { libc::malloc_trim(0) };
        }
    }

    fn shutdown(&mut self) {
        while let Some(entry) = self.outputs.pop() {
            self.teardown_output_entry(entry);
        }
        crate::media::wait_for_reaped_workers();
        for waiter in std::mem::take(&mut self.commit_waiters) {
            self.ipc_hub.complete_pending_ack(
                waiter.client_id,
                &IpcResponse::err_coded(IpcErrorCode::Canceled, "daemon shutting down"),
            );
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
    fn remove_output_at(&mut self, idx: usize, waiter_reason: &str) {
        let entry_gid = self.outputs[idx]
            .pending_apply
            .as_ref()
            .and_then(|p| p.group_id.clone());
        let entry = self.outputs.remove(idx);
        let name = entry.name.clone();
        let was_announced = entry.name_resolved;
        self.pause.drop_output(&name);
        self.fullscreen_outputs.remove(&name);
        self.fail_commit_waiters_for_output(&name, IpcErrorCode::Canceled, waiter_reason);
        if let Some(ref gid) = entry_gid {
            self.cancel_batch_siblings(gid, usize::MAX, waiter_reason);
        }
        self.teardown_output_entry(entry);
        self.sync_shader_watcher();
        if was_announced {
            self.ipc_hub
                .broadcast_event(&IpcEvent::OutputRemoved { output: name });
        }
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
        let last_transition = (spec.transition.clone(), spec.duration_ms);
        self.outputs.push(OutputEntry {
            index,
            wl_output: output,
            name: name.clone(),
            name_resolved,
            surface,
            spec,
            revision: 0,
            generation: 0,
            active_generation: 0,
            pending_generation: None,
            last_committed_frame: None,
            last_committed_revision: 0,
            target_sig: None,
            shader_sig: None,
            last_transition,
            slot0: MediaSlotGpu::empty(),
            slot1: None,
            slot0_has_first_frame: false,
            job: None,
            prev_job: None,
            custom_program: None,
            custom_shader_path: None,
            custom_textures: BTreeMap::new(),
            custom_texture_keys: BTreeMap::new(),
            custom_uniforms: BTreeMap::new(),
            transient: TransientUniformState::default(),
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
        self.remove_output_at(idx, "output removed before commit");
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
        if self.outputs[idx].name_resolved {
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
        self.remove_output_at(idx, "surface closed before commit");
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

#[cfg(all(target_os = "linux", target_env = "gnu"))]
fn configure_glibc_allocator() {
    const M_MMAP_THRESHOLD: libc::c_int = -3;
    const M_ARENA_MAX: libc::c_int = -8;
    // Keep worker/GStreamer allocations inside the main arena and mmap
    // large frame buffers directly so free() + malloc_trim(0) return RSS
    // to the OS even when `glwall daemon` starts without a systemd drop-in.
    if std::env::var_os("MALLOC_ARENA_MAX").is_none() {
        unsafe {
            libc::mallopt(M_ARENA_MAX, 1);
        }
    }
    if std::env::var_os("MALLOC_MMAP_THRESHOLD_").is_none() {
        unsafe {
            libc::mallopt(M_MMAP_THRESHOLD, 256 * 1024);
        }
    }
}

#[cfg(not(all(target_os = "linux", target_env = "gnu")))]
fn configure_glibc_allocator() {}

pub fn run_daemon(
    config_path: Option<PathBuf>,
    socket_path: PathBuf,
    no_restore: bool,
    no_persist: bool,
) -> Result<()> {
    configure_glibc_allocator();
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
    use super::*;
    use crate::gpu::uniforms::{TextureFilter, TextureWrap};

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

    #[test]
    fn shader_switch_prunes_stale_textures_and_uniforms() {
        let mut custom_textures = BTreeMap::new();
        custom_textures.insert("u_depth".to_string(), TextureId(10));
        custom_textures.insert("u_shared".to_string(), TextureId(20));

        let mk_spec = |name: &str| TextureSpec {
            path: PathBuf::from(format!("/tmp/{name}.png")),
            filter: TextureFilter::Linear,
            wrap: TextureWrap::ClampToEdge,
            flip_y: false,
        };
        let mut spec_textures = BTreeMap::new();
        spec_textures.insert("u_depth".to_string(), mk_spec("depth"));
        spec_textures.insert("u_shared".to_string(), mk_spec("shared"));
        let mut custom_texture_keys = BTreeMap::new();

        let mut custom_uniforms = BTreeMap::new();
        custom_uniforms.insert("u_parallaxStrength".to_string(), UniformValue::Float(0.05));
        custom_uniforms.insert("u_keep".to_string(), UniformValue::Float(1.0));
        let mut spec_uniforms = custom_uniforms.clone();

        let allowed_samplers: HashSet<String> = ["u_shared".to_string()].into_iter().collect();
        let allowed_uniforms: HashSet<String> = ["u_keep".to_string()].into_iter().collect();

        let removed = prune_shader_bindings(
            &mut custom_textures,
            &mut custom_texture_keys,
            &mut spec_textures,
            &mut custom_uniforms,
            &mut spec_uniforms,
            &allowed_samplers,
            &allowed_uniforms,
        );
        assert_eq!(removed, vec![TextureId(10)]);
        assert!(!custom_textures.contains_key("u_depth"));
        assert!(!spec_textures.contains_key("u_depth"));
        assert_eq!(custom_textures.get("u_shared"), Some(&TextureId(20)));
        assert!(spec_textures.contains_key("u_shared"));
        assert!(!custom_uniforms.contains_key("u_parallaxStrength"));
        assert!(!spec_uniforms.contains_key("u_parallaxStrength"));
        assert!(custom_uniforms.contains_key("u_keep"));
        assert!(spec_uniforms.contains_key("u_keep"));

        let empty = HashSet::new();
        let cleared = prune_shader_bindings(
            &mut custom_textures,
            &mut custom_texture_keys,
            &mut spec_textures,
            &mut custom_uniforms,
            &mut spec_uniforms,
            &empty,
            &empty,
        );
        assert_eq!(cleared, vec![TextureId(20)]);
        assert!(custom_textures.is_empty());
        assert!(spec_textures.is_empty());
        assert!(custom_uniforms.is_empty());
        assert!(spec_uniforms.is_empty());
    }

    #[test]
    fn cold_start_materializes_configured_and_persisted_specs() {
        let mut base = OutputSpec {
            target: Some("/tmp/configured.jpg".to_string()),
            shader: Some("/tmp/grain.frag".to_string()),
            ..OutputSpec::default()
        };

        // Even when entry.spec starts equal to base_spec_for(), nothing is materialized yet.
        assert!(should_materialize_initial_spec(
            &base,
            &base,
            false,
            false,
            false,
            0,
            &BTreeMap::new(),
            false,
        ));

        // While initial apply is in flight, configure callbacks must not supersede it.
        assert!(!should_materialize_initial_spec(
            &base,
            &OutputSpec::default(),
            false,
            true,
            false,
            0,
            &BTreeMap::new(),
            true,
        ));

        // Static image uploaded into slot0 (one-shot job already dropped) + shader compiled -> settled.
        assert!(!should_materialize_initial_spec(
            &base,
            &base,
            true,
            false,
            true,
            0,
            &BTreeMap::new(),
            false,
        ));

        // If configured with a shader only (no target), custom_program must materialize.
        base.target = None;
        assert!(should_materialize_initial_spec(
            &base,
            &base,
            false,
            false,
            false,
            0,
            &BTreeMap::new(),
            false,
        ));
        assert!(!should_materialize_initial_spec(
            &base,
            &base,
            false,
            false,
            true,
            0,
            &BTreeMap::new(),
            false,
        ));

        // Standalone procedural shader target (.frag).
        let proc_spec = OutputSpec {
            target: Some("/tmp/aurora.frag".to_string()),
            ..OutputSpec::default()
        };
        assert!(should_materialize_initial_spec(
            &proc_spec,
            &proc_spec,
            false,
            false,
            false,
            0,
            &BTreeMap::new(),
            false,
        ));
        assert!(!should_materialize_initial_spec(
            &proc_spec,
            &proc_spec,
            false,
            false,
            true,
            0,
            &BTreeMap::new(),
            false,
        ));

        // Empty desired spec needs no materialization.
        assert!(!should_materialize_initial_spec(
            &OutputSpec::default(),
            &OutputSpec::default(),
            false,
            false,
            false,
            0,
            &BTreeMap::new(),
            false,
        ));
    }

    fn write_test_bmp(path: &Path, width: i32, height: i32, pixel_row: &[u8]) {
        let mut bytes = Vec::with_capacity(54 + pixel_row.len());
        let file_size = (54 + pixel_row.len()) as u32;
        bytes.extend_from_slice(b"BM");
        bytes.extend_from_slice(&file_size.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&54u32.to_le_bytes());
        bytes.extend_from_slice(&40u32.to_le_bytes());
        bytes.extend_from_slice(&width.to_le_bytes());
        bytes.extend_from_slice(&height.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&24u16.to_le_bytes());
        bytes.extend_from_slice(&[0u8; 24]);
        bytes.extend_from_slice(pixel_row);
        fs::write(path, bytes).unwrap();
    }

    #[test]
    fn preflight_failure_at_each_stage_destroys_compiled_gl_programs() {
        use glow::HasContext;

        struct SpawnFailReset;
        impl Drop for SpawnFailReset {
            fn drop(&mut self) {
                crate::media::set_force_spawn_fail(false);
            }
        }

        let dir = std::env::temp_dir().join(format!("glwall-preflight-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        let media_path = dir.join("wall.bmp");
        write_test_bmp(&media_path, 1, 1, &[255, 0, 0, 0]);
        let shader_path = dir.join("valid.frag");
        fs::write(
            &shader_path,
            "uniform float u_strength;\nvec4 wallpaper(vec2 uv) { return sampleWallpaper(uv) * (1.0 + u_strength * 0.01); }\n",
        )
        .unwrap();
        let bad_transition_path = dir.join("broken_transition.frag");
        fs::write(&bad_transition_path, "not valid glsl {").unwrap();

        let egl = crate::gpu::egl::EglState::new_test_pbuffer(8, 8).unwrap();
        let gl = egl.create_glow_context().unwrap();
        let (media_tx, _media_rx) = calloop::channel::channel::<MediaEvent>();
        let spec = OutputSpec::default();

        // Track every compiled program handle and assert every failed stage destroys all of them.
        let run_case = |tx: &TxParams| -> Vec<glow::Program> {
            let mut prepared = Vec::new();
            let mut compiled_handles = Vec::new();
            let pre =
                resolve_preflight_inputs(tx, "DP-1", 1, &spec, true, 8192, 33_554_432).unwrap();
            let mut slot = None;
            let res = stage_output_transaction(
                tx,
                &pre,
                "DP-1",
                0,
                &spec,
                None,
                false,
                None,
                &media_tx,
                &mut slot,
                |src| {
                    let prog = GlProgram::compile(&gl, src)?;
                    compiled_handles.push(prog.handle);
                    Ok(prog)
                },
            );
            if let Some(prep) = slot {
                prepared.push((0, prep));
            }
            assert!(res.is_err(), "expected preflight stage failure");
            abort_prepared_transactions(prepared, Some(&gl));
            compiled_handles
        };

        // Stage A: valid shader compiles, then uniform type validation fails.
        let mut bad_uniforms = BTreeMap::new();
        bad_uniforms.insert("u_strength".to_string(), UniformValue::Vec2([1.0, 2.0]));
        let handles_a = run_case(&TxParams {
            output: "DP-1".to_string(),
            shader: Some(shader_path.to_string_lossy().to_string()),
            uniforms: bad_uniforms,
            ..TxParams::default()
        });
        assert_eq!(handles_a.len(), 1);
        assert!(!unsafe { gl.is_program(handles_a[0]) });

        // Stage B: valid shader compiles, then transition shader compile fails.
        let handles_b = run_case(&TxParams {
            output: "DP-1".to_string(),
            media: Some(ApplyMedia {
                target: media_path.to_string_lossy().to_string(),
                fit: None,
                fps: None,
                hwaccel: None,
            }),
            shader: Some(shader_path.to_string_lossy().to_string()),
            transition: Some(ApplyTransition {
                name: Some(bad_transition_path.to_string_lossy().to_string()),
                duration_ms: Some(300),
            }),
            ..TxParams::default()
        });
        assert_eq!(handles_b.len(), 1);
        assert!(!unsafe { gl.is_program(handles_b[0]) });

        // Stage C: both custom shader and fade transition compile, then MediaJob::spawn fails.
        crate::media::set_force_spawn_fail(true);
        let _reset = SpawnFailReset;
        let handles_c = run_case(&TxParams {
            output: "DP-1".to_string(),
            media: Some(ApplyMedia {
                target: media_path.to_string_lossy().to_string(),
                fit: None,
                fps: None,
                hwaccel: None,
            }),
            shader: Some(shader_path.to_string_lossy().to_string()),
            transition: Some(ApplyTransition {
                name: Some("fade".to_string()),
                duration_ms: Some(300),
            }),
            ..TxParams::default()
        });
        assert_eq!(handles_c.len(), 2);
        for h in handles_c {
            assert!(!unsafe { gl.is_program(h) });
        }

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn hot_reload_prunes_removed_samplers_and_preserves_state_on_compile_error() {
        use glow::HasContext;

        let egl = crate::gpu::egl::EglState::new_test_pbuffer(8, 8).unwrap();
        let gl = egl.create_glow_context().unwrap();
        let mut pool = TexturePool::new();
        let mut custom_program = None;
        let mut custom_textures = BTreeMap::new();
        let mut custom_texture_keys = BTreeMap::new();
        let mut spec_textures = BTreeMap::new();
        let mut custom_uniforms = BTreeMap::new();
        let mut spec_uniforms = BTreeMap::new();

        let src_v1 = r#"
uniform sampler2D u_depth;
uniform float u_strength;
vec4 wallpaper(vec2 uv) {
    return sampleWallpaper(uv) + texture(u_depth, uv) * u_strength * 0.01;
}
"#;
        reload_shader_into_bindings(
            &gl,
            src_v1,
            &mut custom_program,
            &mut custom_textures,
            &mut custom_texture_keys,
            &mut spec_textures,
            &mut custom_uniforms,
            &mut spec_uniforms,
            &mut pool,
        )
        .unwrap();

        let key = TextureKey {
            path: PathBuf::from("/virtual/depth.png"),
            filter: TextureFilter::Linear,
            wrap: TextureWrap::ClampToEdge,
            flip_y: false,
            fingerprint: Default::default(),
        };
        let depth_id = pool
            .insert_rgba(&gl, key.clone(), 2, 2, &[100; 16])
            .unwrap();
        let depth_gl_handle = pool.texture_info(depth_id).unwrap().0;
        custom_textures.insert("u_depth".to_string(), depth_id);
        custom_texture_keys.insert("u_depth".to_string(), key);
        spec_textures.insert(
            "u_depth".to_string(),
            TextureSpec {
                path: PathBuf::from("/virtual/depth.png"),
                filter: TextureFilter::Linear,
                wrap: TextureWrap::ClampToEdge,
                flip_y: false,
            },
        );
        custom_uniforms.insert("u_strength".to_string(), UniformValue::Float(0.25));
        spec_uniforms.insert("u_strength".to_string(), UniformValue::Float(0.25));
        assert_eq!(pool.len(), 1);

        // Failed hot reload must leave active program and bindings untouched.
        let broken_src = "vec4 wallpaper(vec2 uv) { return missing_fn(uv); }";
        let err = reload_shader_into_bindings(
            &gl,
            broken_src,
            &mut custom_program,
            &mut custom_textures,
            &mut custom_texture_keys,
            &mut spec_textures,
            &mut custom_uniforms,
            &mut spec_uniforms,
            &mut pool,
        )
        .unwrap_err();
        assert_eq!(
            error_code_of(&err, IpcErrorCode::InvalidInput),
            IpcErrorCode::ShaderCompile
        );
        assert_eq!(pool.len(), 1);
        assert!(custom_textures.contains_key("u_depth"));
        assert!(custom_uniforms.contains_key("u_strength"));
        assert!(unsafe { gl.is_texture(depth_gl_handle) });

        // Successful hot reload dropping u_depth must release pool ref and GL texture.
        let src_v2 = r#"
uniform float u_strength;
vec4 wallpaper(vec2 uv) {
    return sampleWallpaper(uv) * (1.0 + u_strength * 0.01);
}
"#;
        reload_shader_into_bindings(
            &gl,
            src_v2,
            &mut custom_program,
            &mut custom_textures,
            &mut custom_texture_keys,
            &mut spec_textures,
            &mut custom_uniforms,
            &mut spec_uniforms,
            &mut pool,
        )
        .unwrap();

        assert_eq!(pool.len(), 0);
        assert!(!custom_textures.contains_key("u_depth"));
        assert!(!custom_texture_keys.contains_key("u_depth"));
        assert!(!spec_textures.contains_key("u_depth"));
        assert!(custom_uniforms.contains_key("u_strength"));
        assert!(!unsafe { gl.is_texture(depth_gl_handle) });

        if let Some(prog) = custom_program.take() {
            prog.destroy(&gl);
        }
        pool.destroy(&gl);
    }

    #[test]
    fn preflight_error_codes_cover_contract_taxonomy() {
        let dir = std::env::temp_dir().join(format!("glwall-codes-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        let txt = dir.join("readme.txt");
        fs::write(&txt, b"hello").unwrap();
        let huge = dir.join("huge.bmp");
        write_test_bmp(&huge, 16384, 4, &[0u8; 4]);
        let bad_shader = dir.join("broken.frag");
        fs::write(&bad_shader, "broken { glsl }").unwrap();
        let good_shader = dir.join("ok.frag");
        fs::write(
            &good_shader,
            "uniform float u_mix;\nvec4 wallpaper(vec2 uv) { return sampleWallpaper(uv) * u_mix; }\n",
        )
        .unwrap();

        let spec = OutputSpec::default();
        let (media_tx, _) = calloop::channel::channel::<MediaEvent>();
        let egl = crate::gpu::egl::EglState::new_test_pbuffer(8, 8).unwrap();
        let gl = egl.create_glow_context().unwrap();

        // MissingSource
        let err_missing = resolve_preflight_inputs(
            &TxParams {
                output: "DP-1".to_string(),
                media: Some(ApplyMedia {
                    target: dir.join("missing.png").to_string_lossy().to_string(),
                    fit: None,
                    fps: None,
                    hwaccel: None,
                }),
                ..Default::default()
            },
            "DP-1",
            1,
            &spec,
            false,
            8192,
            33_554_432,
        )
        .err()
        .unwrap();
        assert_eq!(
            error_code_of(&err_missing, IpcErrorCode::InvalidInput),
            IpcErrorCode::MissingSource
        );

        // UnsupportedMedia
        let err_unsup = resolve_preflight_inputs(
            &TxParams {
                output: "DP-1".to_string(),
                media: Some(ApplyMedia {
                    target: txt.to_string_lossy().to_string(),
                    fit: None,
                    fps: None,
                    hwaccel: None,
                }),
                ..Default::default()
            },
            "DP-1",
            1,
            &spec,
            false,
            8192,
            33_554_432,
        )
        .err()
        .unwrap();
        assert_eq!(
            error_code_of(&err_unsup, IpcErrorCode::InvalidInput),
            IpcErrorCode::UnsupportedMedia
        );

        // ResourceLimit
        let err_limit = resolve_preflight_inputs(
            &TxParams {
                output: "DP-1".to_string(),
                media: Some(ApplyMedia {
                    target: huge.to_string_lossy().to_string(),
                    fit: None,
                    fps: None,
                    hwaccel: None,
                }),
                ..Default::default()
            },
            "DP-1",
            1,
            &spec,
            false,
            8192,
            33_554_432,
        )
        .err()
        .unwrap();
        assert_eq!(
            error_code_of(&err_limit, IpcErrorCode::InvalidInput),
            IpcErrorCode::ResourceLimit
        );

        // RevisionConflict
        let err_rev = resolve_preflight_inputs(
            &TxParams {
                output: "DP-1".to_string(),
                revision: Some(1),
                ..Default::default()
            },
            "DP-1",
            3,
            &spec,
            false,
            8192,
            33_554_432,
        )
        .err()
        .unwrap();
        assert_eq!(
            error_code_of(&err_rev, IpcErrorCode::InvalidInput),
            IpcErrorCode::RevisionConflict
        );

        // InvalidInput (non-finite params)
        let err_nan = resolve_preflight_inputs(
            &TxParams {
                output: "DP-1".to_string(),
                params: Some([f32::NAN, 0.0, 0.0, 0.0]),
                ..Default::default()
            },
            "DP-1",
            0,
            &spec,
            false,
            8192,
            33_554_432,
        )
        .err()
        .unwrap();
        assert_eq!(
            error_code_of(&err_nan, IpcErrorCode::RenderFailed),
            IpcErrorCode::InvalidInput
        );

        // ShaderCompile
        let pre_sh = resolve_preflight_inputs(
            &TxParams {
                output: "DP-1".to_string(),
                shader: Some(bad_shader.to_string_lossy().to_string()),
                ..Default::default()
            },
            "DP-1",
            0,
            &spec,
            false,
            8192,
            33_554_432,
        )
        .unwrap();
        let mut slot = None;
        let err_compile = stage_output_transaction(
            &TxParams::default(),
            &pre_sh,
            "DP-1",
            0,
            &spec,
            None,
            false,
            None,
            &media_tx,
            &mut slot,
            |src| GlProgram::compile(&gl, src),
        )
        .unwrap_err();
        assert_eq!(
            error_code_of(&err_compile, IpcErrorCode::InvalidInput),
            IpcErrorCode::ShaderCompile
        );

        // TypeMismatch
        let mut bad_u = BTreeMap::new();
        bad_u.insert("u_mix".to_string(), UniformValue::Bool(true));
        let tx_u = TxParams {
            output: "DP-1".to_string(),
            shader: Some(good_shader.to_string_lossy().to_string()),
            uniforms: bad_u,
            ..Default::default()
        };
        let pre_u =
            resolve_preflight_inputs(&tx_u, "DP-1", 0, &spec, false, 8192, 33_554_432).unwrap();
        let mut slot2 = None;
        let err_type = stage_output_transaction(
            &tx_u,
            &pre_u,
            "DP-1",
            0,
            &spec,
            None,
            false,
            None,
            &media_tx,
            &mut slot2,
            |src| GlProgram::compile(&gl, src),
        )
        .unwrap_err();
        assert_eq!(
            error_code_of(&err_type, IpcErrorCode::InvalidInput),
            IpcErrorCode::TypeMismatch
        );
        if let Some(prep) = slot2 {
            prep.destroy_staged(Some(&gl));
        }

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn transient_uniform_state_overlay_merge_prune_and_clear() {
        let mut base = BTreeMap::new();
        base.insert("u_a".to_string(), UniformValue::Float(1.0));
        base.insert("u_b".to_string(), UniformValue::Int(2));

        let mut st = TransientUniformState {
            session: "cursor".to_string(),
            last_seq: 1,
            pending_overlay: Some({
                let mut m = BTreeMap::new();
                m.insert("u_a".to_string(), UniformValue::Float(0.25));
                m.insert("u_removed".to_string(), UniformValue::Bool(true));
                m
            }),
            active_overlay: BTreeMap::new(),
        };

        let eff = effective_uniforms_with_overlay(&base, st.current_overlay());
        assert_eq!(eff.get("u_a"), Some(&UniformValue::Float(0.25)));
        assert_eq!(eff.get("u_b"), Some(&UniformValue::Int(2)));

        let mut allowed = HashSet::new();
        allowed.insert("u_a".to_string());
        st.prune(&allowed);
        assert!(!st.current_overlay().contains_key("u_removed"));
        st.consume_pending();
        assert!(st.pending_overlay.is_none());
        assert_eq!(
            st.active_overlay.get("u_a"),
            Some(&UniformValue::Float(0.25))
        );

        assert!(st.clear());
        assert!(st.session.is_empty());
        assert_eq!(st.last_seq, 0);
        assert!(st.active_overlay.is_empty());
    }

    #[test]
    fn idempotent_fingerprint_matches_identical_retry_and_rejects_drift() {
        let dir = std::env::temp_dir().join(format!("glwall-idemp-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        let bmp = dir.join("wall.bmp");
        write_test_bmp(&bmp, 1, 1, &[10, 20, 30, 0]);
        let sh = dir.join("fx.frag");
        fs::write(&sh, "uniform float u_gain;\nvec4 wallpaper(vec2 uv) { return sampleWallpaper(uv) * u_gain; }\n").unwrap();

        let mut base_spec = OutputSpec::default();
        let bmp_str = bmp.to_string_lossy().to_string();
        let sh_str = sh.to_string_lossy().to_string();
        base_spec.target = Some(bmp_str.clone());
        base_spec.shader = Some(sh_str.clone());
        base_spec
            .uniforms
            .insert("u_gain".to_string(), UniformValue::Float(1.5));

        let current_fp = EffectiveSpecFingerprint {
            target: normalize_optional_path(Some(&bmp_str)),
            target_sig: file_stat_sig(&bmp),
            fit: base_spec.fit,
            fps_bits: base_spec.fps.map(norm_f32_bits),
            hwaccel: base_spec.hwaccel,
            shader: normalize_optional_path(Some(&sh_str)),
            shader_sig: file_stat_sig(&sh),
            transition: base_spec.transition.clone(),
            duration_ms: base_spec.duration_ms,
            params_bits: base_spec.params.map(norm_f32_bits),
            textures: BTreeMap::new(),
            uniforms: base_spec
                .uniforms
                .iter()
                .map(|(k, v)| (k.clone(), UniformFingerprint::from(v)))
                .collect(),
        };

        let retry_tx = TxParams {
            output: "DP-1".to_string(),
            revision: Some(4),
            media: Some(ApplyMedia {
                target: bmp_str.clone(),
                fit: None,
                fps: None,
                hwaccel: None,
            }),
            shader: Some(sh_str.clone()),
            uniforms: base_spec.uniforms.clone(),
            ..Default::default()
        };
        let pre_same =
            resolve_preflight_inputs(&retry_tx, "DP-1", 4, &base_spec, true, 8192, 33_554_432)
                .unwrap();
        assert_eq!(
            fingerprint_for_tx(&retry_tx, &pre_same, &current_fp),
            current_fp
        );

        let mut diff_uniforms = base_spec.uniforms.clone();
        diff_uniforms.insert("u_gain".to_string(), UniformValue::Float(2.0));
        let conflict_tx = TxParams {
            output: "DP-1".to_string(),
            revision: Some(4),
            uniforms: diff_uniforms,
            ..Default::default()
        };
        let pre_diff =
            resolve_preflight_inputs(&conflict_tx, "DP-1", 4, &base_spec, true, 8192, 33_554_432)
                .unwrap();
        assert_ne!(
            fingerprint_for_tx(&conflict_tx, &pre_diff, &current_fp),
            current_fp
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn one_shot_mailbox_clear_drops_buffers_before_trim() {
        use std::sync::Arc;

        let (tx, _rx) = calloop::channel::channel::<MediaEvent>();
        let mailbox = crate::media::FrameMailbox::new(tx, "DP-1".to_string(), 1);
        mailbox.publish(FramePayload::Rgba {
            width: 4,
            height: 4,
            pixels: vec![42u8; 64],
        });

        let payload = mailbox.pop_latest().unwrap();
        drop(payload);
        mailbox.clear_pending();
        let weak = Arc::downgrade(&mailbox);
        drop(mailbox);
        assert_eq!(
            weak.strong_count(),
            0,
            "FrameMailbox Arc must reach 0 strong refs before malloc_trim"
        );
    }
}
