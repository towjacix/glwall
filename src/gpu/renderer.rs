use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::Read;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::slice;

use anyhow::{Result, bail};
use glow::HasContext;

use crate::config::{ColorSpace, FitMode};
use crate::gpu::shader::{
    AssembledShader, BUILTIN_PASSTHROUGH, ShaderMode, VERTEX_SHADER_SRC, assemble_fragment_shader,
    is_builtin_sampler, is_builtin_uniform,
};
use crate::gpu::uniforms::{
    ReflectedSampler, ReflectedUniform, ShaderSchema, TextureFilter, TextureSpec, TextureWrap,
    UniformType, UniformValue, upload_uniform,
};

#[derive(Debug)]
pub struct UserUniformSlot {
    pub name: String,
    pub utype: UniformType,
    pub location: glow::UniformLocation,
}

#[derive(Debug)]
pub struct UserSamplerSlot {
    pub name: String,
    pub location: glow::UniformLocation,
    pub size_location: Option<glow::UniformLocation>,
}

pub struct GlProgram {
    pub handle: glow::Program,
    pub assembled: AssembledShader,
    pub uses_time: bool,
    pub uses_progress: bool,
    pub uses_ichannel0: bool,
    pub user_uniforms: Vec<UserUniformSlot>,
    pub user_samplers: Vec<UserSamplerSlot>,
    locs: ProgramLocations,
}

#[derive(Debug, Default)]
struct ProgramLocations {
    u_resolution: Option<glow::UniformLocation>,
    u_time: Option<glow::UniformLocation>,
    u_delta: Option<glow::UniformLocation>,
    u_progress: Option<glow::UniformLocation>,
    u_frame: Option<glow::UniformLocation>,
    u_params: Option<glow::UniformLocation>,
    u_output_index: Option<glow::UniformLocation>,
    u_tex0_rgba: Option<glow::UniformLocation>,
    u_tex0_y: Option<glow::UniformLocation>,
    u_tex0_uv: Option<glow::UniformLocation>,
    u_slot0_kind: Option<glow::UniformLocation>,
    u_slot0_framebuffer: Option<glow::UniformLocation>,
    u_slot0_size: Option<glow::UniformLocation>,
    u_slot0_fit: Option<glow::UniformLocation>,
    u_slot0_cs: Option<glow::UniformLocation>,
    u_tex1_rgba: Option<glow::UniformLocation>,
    u_tex1_y: Option<glow::UniformLocation>,
    u_tex1_uv: Option<glow::UniformLocation>,
    u_slot1_kind: Option<glow::UniformLocation>,
    u_slot1_size: Option<glow::UniformLocation>,
    u_slot1_fit: Option<glow::UniformLocation>,
    u_slot1_cs: Option<glow::UniformLocation>,
    i_channel0: Option<glow::UniformLocation>,
}

impl GlProgram {
    pub fn compile(gl: &glow::Context, user_fragment_src: &str) -> Result<Self> {
        let assembled = assemble_fragment_shader(user_fragment_src);
        let vs = compile_stage(gl, glow::VERTEX_SHADER, VERTEX_SHADER_SRC)?;
        let fs = match compile_stage(gl, glow::FRAGMENT_SHADER, &assembled.fragment_source) {
            Ok(s) => s,
            Err(err) => {
                unsafe { gl.delete_shader(vs) };
                return Err(err);
            }
        };

        let program = unsafe {
            gl.create_program()
                .map_err(|e| anyhow::anyhow!("glCreateProgram failed: {e}"))?
        };
        unsafe {
            gl.attach_shader(program, vs);
            gl.attach_shader(program, fs);
            gl.link_program(program);
            gl.detach_shader(program, vs);
            gl.detach_shader(program, fs);
            gl.delete_shader(vs);
            gl.delete_shader(fs);
        }

        let linked = unsafe { gl.get_program_link_status(program) };
        if !linked {
            let log = unsafe { gl.get_program_info_log(program) };
            unsafe { gl.delete_program(program) };
            bail!("shader link failed: {log}");
        }

        let locs = unsafe { ProgramLocations::query(gl, program) };
        let reflected = unsafe { reflect_program(gl, program) };
        let uses_time = assembled.uses_time
            && (reflected.active_builtins.contains("u_time")
                || reflected.active_builtins.contains("u_delta")
                || reflected.active_builtins.contains("u_frame"));
        let uses_progress = reflected.active_builtins.contains("u_progress");
        let uses_ichannel0 = assembled.uses_ichannel0
            && (reflected.active_builtins.contains("iChannel0") || locs.i_channel0.is_some());

        Ok(Self {
            handle: program,
            assembled,
            uses_time,
            uses_progress,
            uses_ichannel0,
            user_uniforms: reflected.user_uniforms,
            user_samplers: reflected.user_samplers,
            locs,
        })
    }

    pub fn schema(&self) -> ShaderSchema {
        let mode = match self.assembled.mode {
            ShaderMode::Glwall => "glwall",
            ShaderMode::Shadertoy => "shadertoy",
        };
        ShaderSchema {
            mode: mode.to_string(),
            uses_time: self.uses_time,
            uses_progress: self.uses_progress,
            uses_ichannel0: self.uses_ichannel0,
            uniforms: self
                .user_uniforms
                .iter()
                .map(|u| ReflectedUniform {
                    name: u.name.clone(),
                    utype: u.utype,
                })
                .collect(),
            samplers: self
                .user_samplers
                .iter()
                .map(|s| ReflectedSampler {
                    name: s.name.clone(),
                    has_size_uniform: s.size_location.is_some(),
                })
                .collect(),
        }
    }

    pub fn validate_uniform(&self, name: &str, value: &UniformValue) -> Result<()> {
        let Some(slot) = self.user_uniforms.iter().find(|u| u.name == name) else {
            return Ok(());
        };
        let compatible = matches!(
            (slot.utype, value),
            (
                UniformType::Float,
                UniformValue::Float(_) | UniformValue::Int(_)
            ) | (
                UniformType::Int,
                UniformValue::Int(_) | UniformValue::Bool(_)
            ) | (
                UniformType::Bool,
                UniformValue::Bool(_) | UniformValue::Int(_)
            ) | (UniformType::Vec2, UniformValue::Vec2(_))
                | (UniformType::Vec3, UniformValue::Vec3(_))
                | (UniformType::Vec4, UniformValue::Vec4(_))
        );
        if !compatible {
            bail!(
                "uniform '{name}' expects {}, got {}",
                slot.utype,
                value.value_type()
            );
        }
        Ok(())
    }

    pub fn destroy(self, gl: &glow::Context) {
        unsafe { gl.delete_program(self.handle) };
    }
}

struct ReflectedBindings {
    active_builtins: HashSet<String>,
    user_uniforms: Vec<UserUniformSlot>,
    user_samplers: Vec<UserSamplerSlot>,
}

unsafe fn reflect_program(gl: &glow::Context, program: glow::Program) -> ReflectedBindings {
    let count = unsafe { gl.get_active_uniforms(program) };
    let mut active_builtins = HashSet::new();
    let mut raw_samplers: Vec<(String, glow::UniformLocation)> = Vec::new();
    let mut raw_uniforms: Vec<(String, UniformType, glow::UniformLocation)> = Vec::new();

    for idx in 0..count {
        let Some(active) = (unsafe { gl.get_active_uniform(program, idx) }) else {
            continue;
        };
        let clean_name = active
            .name
            .strip_suffix("[0]")
            .unwrap_or(&active.name)
            .to_string();
        if is_builtin_sampler(&clean_name) || is_builtin_uniform(&clean_name) {
            active_builtins.insert(clean_name);
            continue;
        }
        if active.size > 1 {
            continue;
        }

        let Some(location) = (unsafe { gl.get_uniform_location(program, &active.name) }) else {
            continue;
        };

        if active.utype == glow::SAMPLER_2D {
            raw_samplers.push((clean_name, location));
        } else if let Some(utype) = UniformType::from_gl_type(active.utype) {
            raw_uniforms.push((clean_name, utype, location));
        }
    }

    let sampler_names: HashSet<String> = raw_samplers.iter().map(|(n, _)| n.clone()).collect();
    let mut size_locations: HashMap<String, glow::UniformLocation> = HashMap::new();
    let mut user_uniforms = Vec::new();

    for (name, utype, location) in raw_uniforms {
        if utype == UniformType::Vec2
            && let Some(base) = name.strip_suffix("_size")
            && sampler_names.contains(base)
        {
            size_locations.insert(base.to_string(), location);
            continue;
        }
        user_uniforms.push(UserUniformSlot {
            name,
            utype,
            location,
        });
    }

    let mut user_samplers: Vec<UserSamplerSlot> = raw_samplers
        .into_iter()
        .map(|(name, location)| {
            let size_location = size_locations.remove(&name);
            UserSamplerSlot {
                name,
                location,
                size_location,
            }
        })
        .collect();

    user_samplers.sort_by(|a, b| a.name.cmp(&b.name));
    user_uniforms.sort_by(|a, b| a.name.cmp(&b.name));

    ReflectedBindings {
        active_builtins,
        user_uniforms,
        user_samplers,
    }
}

impl ProgramLocations {
    unsafe fn query(gl: &glow::Context, p: glow::Program) -> Self {
        unsafe {
            Self {
                u_resolution: gl.get_uniform_location(p, "u_resolution"),
                u_time: gl.get_uniform_location(p, "u_time"),
                u_delta: gl.get_uniform_location(p, "u_delta"),
                u_progress: gl.get_uniform_location(p, "u_progress"),
                u_frame: gl.get_uniform_location(p, "u_frame"),
                u_params: gl.get_uniform_location(p, "u_params"),
                u_output_index: gl.get_uniform_location(p, "u_output_index"),
                u_tex0_rgba: gl.get_uniform_location(p, "u_tex0_rgba"),
                u_tex0_y: gl.get_uniform_location(p, "u_tex0_y"),
                u_tex0_uv: gl.get_uniform_location(p, "u_tex0_uv"),
                u_slot0_kind: gl.get_uniform_location(p, "u_slot0_kind"),
                u_slot0_size: gl.get_uniform_location(p, "u_slot0_size"),
                u_slot0_fit: gl.get_uniform_location(p, "u_slot0_fit"),
                u_slot0_cs: gl.get_uniform_location(p, "u_slot0_cs"),
                u_slot0_framebuffer: gl.get_uniform_location(p, "u_slot0_framebuffer"),
                u_tex1_rgba: gl.get_uniform_location(p, "u_tex1_rgba"),
                u_tex1_y: gl.get_uniform_location(p, "u_tex1_y"),
                u_tex1_uv: gl.get_uniform_location(p, "u_tex1_uv"),
                u_slot1_kind: gl.get_uniform_location(p, "u_slot1_kind"),
                u_slot1_size: gl.get_uniform_location(p, "u_slot1_size"),
                u_slot1_fit: gl.get_uniform_location(p, "u_slot1_fit"),
                u_slot1_cs: gl.get_uniform_location(p, "u_slot1_cs"),
                i_channel0: gl.get_uniform_location(p, "iChannel0"),
            }
        }
    }
}

fn compile_stage(gl: &glow::Context, stage: u32, src: &str) -> Result<glow::Shader> {
    let shader = unsafe {
        gl.create_shader(stage)
            .map_err(|e| anyhow::anyhow!("glCreateShader failed: {e}"))?
    };
    unsafe {
        gl.shader_source(shader, src);
        gl.compile_shader(shader);
    }
    if !unsafe { gl.get_shader_compile_status(shader) } {
        let log = unsafe { gl.get_shader_info_log(shader) };
        unsafe { gl.delete_shader(shader) };
        let stage_name = match stage {
            glow::VERTEX_SHADER => "vertex",
            glow::FRAGMENT_SHADER => "fragment",
            _ => "shader",
        };
        bail!("{stage_name} compile failed:\n{log}");
    }
    Ok(shader)
}

#[derive(Debug)]
struct OwnedTexture {
    handle: glow::Texture,
    width: u32,
    height: u32,
}

impl OwnedTexture {
    fn new(gl: &glow::Context) -> Result<Self> {
        let handle = unsafe {
            gl.create_texture()
                .map_err(|e| anyhow::anyhow!("glCreateTexture failed: {e}"))?
        };
        let tex = Self {
            handle,
            width: 0,
            height: 0,
        };
        tex.set_sampler_params(gl, TextureFilter::Linear, TextureWrap::ClampToEdge);
        Ok(tex)
    }

    fn set_sampler_params(&self, gl: &glow::Context, filter: TextureFilter, wrap: TextureWrap) {
        let gl_filter = filter.to_gl() as i32;
        let gl_wrap = wrap.to_gl() as i32;
        unsafe {
            gl.bind_texture(glow::TEXTURE_2D, Some(self.handle));
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, gl_filter);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, gl_filter);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, gl_wrap);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, gl_wrap);
            gl.bind_texture(glow::TEXTURE_2D, None);
        }
    }

    fn upload_rgba(&mut self, gl: &glow::Context, width: u32, height: u32, data: &[u8]) {
        unsafe {
            gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
            gl.bind_texture(glow::TEXTURE_2D, Some(self.handle));
            if self.width == width && self.height == height {
                gl.tex_sub_image_2d(
                    glow::TEXTURE_2D,
                    0,
                    0,
                    0,
                    width as i32,
                    height as i32,
                    glow::RGBA,
                    glow::UNSIGNED_BYTE,
                    glow::PixelUnpackData::Slice(Some(data)),
                );
            } else {
                gl.tex_image_2d(
                    glow::TEXTURE_2D,
                    0,
                    glow::RGBA8 as i32,
                    width as i32,
                    height as i32,
                    0,
                    glow::RGBA,
                    glow::UNSIGNED_BYTE,
                    glow::PixelUnpackData::Slice(Some(data)),
                );
                self.width = width;
                self.height = height;
            }
            gl.bind_texture(glow::TEXTURE_2D, None);
        }
    }

    fn alloc_storage(
        &mut self,
        gl: &glow::Context,
        internal_format: i32,
        format: u32,
        width: u32,
        height: u32,
    ) {
        if self.width == width && self.height == height {
            return;
        }
        unsafe {
            gl.bind_texture(glow::TEXTURE_2D, Some(self.handle));
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                internal_format,
                width as i32,
                height as i32,
                0,
                format,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(None),
            );
            gl.bind_texture(glow::TEXTURE_2D, None);
        }
        self.width = width;
        self.height = height;
    }

    fn destroy(self, gl: &glow::Context) {
        unsafe {
            gl.delete_texture(self.handle);
            // glDeleteTexture only *marks* the object for deletion; the NVIDIA
            // driver defers the actual free until the next command submission.
            // Since destroys usually happen on the apply path (no draw follows
            // immediately), flush so the deferred free is processed now and
            // the backing memory actually leaves the process.
            gl.flush();
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct TextureFingerprint {
    pub size: u64,
    pub mtime_sec: i64,
    pub mtime_nsec: i64,
    pub content_hash: u64,
}

impl TextureFingerprint {
    pub fn from_path(path: &Path) -> Self {
        let Ok(meta) = std::fs::metadata(path) else {
            return Self::default();
        };
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        if let Ok(mut file) = std::fs::File::open(path) {
            let mut buf = [0u8; 64 * 1024];
            while let Ok(n) = file.read(&mut buf) {
                if n == 0 {
                    break;
                }
                for &b in &buf[..n] {
                    hash ^= u64::from(b);
                    hash = hash.wrapping_mul(0x0100_0000_01b3);
                }
            }
        }
        Self {
            size: meta.len(),
            mtime_sec: meta.mtime(),
            mtime_nsec: meta.mtime_nsec(),
            content_hash: hash,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TextureKey {
    pub path: PathBuf,
    pub filter: TextureFilter,
    pub wrap: TextureWrap,
    pub flip_y: bool,
    pub fingerprint: TextureFingerprint,
}

impl TextureKey {
    pub fn from_spec(spec: &TextureSpec) -> Self {
        let canonical = std::fs::canonicalize(&spec.path).unwrap_or_else(|_| spec.path.clone());
        let fingerprint = TextureFingerprint::from_path(&canonical);
        Self {
            path: canonical,
            filter: spec.filter,
            wrap: spec.wrap,
            flip_y: spec.flip_y,
            fingerprint,
        }
    }
}

struct PoolEntry<T> {
    key: TextureKey,
    tex: T,
    width: u32,
    height: u32,
    refs: usize,
}

pub struct TexturePoolCore<T> {
    next_id: u64,
    entries: HashMap<TextureId, PoolEntry<T>>,
    by_key: HashMap<TextureKey, TextureId>,
}

impl<T> Default for TexturePoolCore<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> TexturePoolCore<T> {
    pub fn new() -> Self {
        Self {
            next_id: 1,
            entries: HashMap::new(),
            by_key: HashMap::new(),
        }
    }

    pub fn contains_key(&self, key: &TextureKey) -> bool {
        self.by_key.contains_key(key)
    }

    pub fn acquire(&mut self, key: &TextureKey) -> Option<TextureId> {
        let id = *self.by_key.get(key)?;
        if let Some(entry) = self.entries.get_mut(&id) {
            entry.refs += 1;
            return Some(id);
        }
        None
    }

    pub fn insert_or_acquire(
        &mut self,
        key: TextureKey,
        width: u32,
        height: u32,
        tex: T,
    ) -> (TextureId, Option<T>) {
        if let Some(id) = self.acquire(&key) {
            return (id, Some(tex));
        }
        let id = TextureId(self.next_id);
        self.next_id += 1;
        self.by_key.insert(key.clone(), id);
        self.entries.insert(
            id,
            PoolEntry {
                key,
                tex,
                width,
                height,
                refs: 1,
            },
        );
        (id, None)
    }

    pub fn release(&mut self, id: TextureId) -> Option<T> {
        let entry = self.entries.get_mut(&id)?;
        if entry.refs > 1 {
            entry.refs -= 1;
            return None;
        }
        let removed = self.entries.remove(&id)?;
        self.by_key.remove(&removed.key);
        Some(removed.tex)
    }

    pub fn get(&self, id: TextureId) -> Option<(u32, u32, &T)> {
        self.entries.get(&id).map(|e| (e.width, e.height, &e.tex))
    }

    #[cfg(test)]
    pub fn ref_count(&self, id: TextureId) -> usize {
        self.entries.get(&id).map_or(0, |e| e.refs)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn drain_all(&mut self) -> Vec<T> {
        self.by_key.clear();
        self.entries.drain().map(|(_, e)| e.tex).collect()
    }
}

pub struct TexturePool {
    core: TexturePoolCore<OwnedTexture>,
}

impl TexturePool {
    pub fn new() -> Self {
        Self {
            core: TexturePoolCore::new(),
        }
    }

    pub fn contains_key(&self, key: &TextureKey) -> bool {
        self.core.contains_key(key)
    }

    pub fn acquire(&mut self, key: &TextureKey) -> Option<TextureId> {
        self.core.acquire(key)
    }

    pub fn insert_rgba(
        &mut self,
        gl: &glow::Context,
        key: TextureKey,
        width: u32,
        height: u32,
        pixels: &[u8],
    ) -> Result<TextureId> {
        if let Some(id) = self.core.acquire(&key) {
            return Ok(id);
        }
        let mut tex = OwnedTexture::new(gl)?;
        tex.set_sampler_params(gl, key.filter, key.wrap);
        tex.upload_rgba(gl, width, height, pixels);
        let (id, leftover) = self.core.insert_or_acquire(key, width, height, tex);
        if let Some(unused) = leftover {
            unused.destroy(gl);
        }
        Ok(id)
    }

    pub fn release(&mut self, gl: &glow::Context, id: TextureId) {
        if let Some(tex) = self.core.release(id) {
            tex.destroy(gl);
        }
    }

    pub fn texture_info(&self, id: TextureId) -> Option<(glow::Texture, u32, u32)> {
        self.core.get(id).map(|(w, h, t)| (t.handle, w, h))
    }

    pub fn len(&self) -> usize {
        self.core.len()
    }

    pub fn estimated_gpu_bytes(&self) -> u64 {
        self.core
            .entries
            .values()
            .map(|e| u64::from(e.width) * u64::from(e.height) * 4)
            .sum()
    }

    pub fn destroy(mut self, gl: &glow::Context) {
        for tex in self.core.drain_all() {
            tex.destroy(gl);
        }
    }
}

impl Default for TexturePool {
    fn default() -> Self {
        Self::new()
    }
}

struct Nv12PboRing {
    y_pbos: [glow::Buffer; 2],
    uv_pbos: [glow::Buffer; 2],
    index: usize,
    y_bytes: usize,
    uv_bytes: usize,
}

impl Nv12PboRing {
    fn new(gl: &glow::Context) -> Result<Self> {
        let y_pbos = [
            unsafe { gl.create_buffer().map_err(|e| anyhow::anyhow!("{e}"))? },
            unsafe { gl.create_buffer().map_err(|e| anyhow::anyhow!("{e}"))? },
        ];
        let uv_pbos = [
            unsafe { gl.create_buffer().map_err(|e| anyhow::anyhow!("{e}"))? },
            unsafe { gl.create_buffer().map_err(|e| anyhow::anyhow!("{e}"))? },
        ];
        Ok(Self {
            y_pbos,
            uv_pbos,
            index: 0,
            y_bytes: 0,
            uv_bytes: 0,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn upload_nv12(
        &mut self,
        gl: &glow::Context,
        y_tex: &mut OwnedTexture,
        uv_tex: &mut OwnedTexture,
        width: u32,
        height: u32,
        y_plane: &[u8],
        uv_plane: &[u8],
    ) {
        let uv_w = width / 2;
        let uv_h = height / 2;
        y_tex.alloc_storage(gl, glow::R8 as i32, glow::RED, width, height);
        uv_tex.alloc_storage(gl, glow::RG8 as i32, glow::RG, uv_w, uv_h);

        let y_len = y_plane.len();
        let uv_len = uv_plane.len();
        if self.y_bytes != y_len || self.uv_bytes != uv_len {
            for pbo in self.y_pbos {
                unsafe {
                    gl.bind_buffer(glow::PIXEL_UNPACK_BUFFER, Some(pbo));
                    gl.buffer_data_size(glow::PIXEL_UNPACK_BUFFER, y_len as i32, glow::STREAM_DRAW);
                }
            }
            for pbo in self.uv_pbos {
                unsafe {
                    gl.bind_buffer(glow::PIXEL_UNPACK_BUFFER, Some(pbo));
                    gl.buffer_data_size(
                        glow::PIXEL_UNPACK_BUFFER,
                        uv_len as i32,
                        glow::STREAM_DRAW,
                    );
                }
            }
            unsafe { gl.bind_buffer(glow::PIXEL_UNPACK_BUFFER, None) };
            self.y_bytes = y_len;
            self.uv_bytes = uv_len;
        }

        let slot = self.index;
        self.index = 1 - self.index;

        unsafe {
            gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);

            upload_pbo_plane(
                gl,
                self.y_pbos[slot],
                y_tex.handle,
                width as i32,
                height as i32,
                glow::RED,
                y_plane,
            );
            upload_pbo_plane(
                gl,
                self.uv_pbos[slot],
                uv_tex.handle,
                uv_w as i32,
                uv_h as i32,
                glow::RG,
                uv_plane,
            );

            gl.bind_buffer(glow::PIXEL_UNPACK_BUFFER, None);
            gl.bind_texture(glow::TEXTURE_2D, None);
        }
    }

    fn destroy(self, gl: &glow::Context) {
        unsafe {
            for b in self.y_pbos {
                gl.delete_buffer(b);
            }
            for b in self.uv_pbos {
                gl.delete_buffer(b);
            }
        }
    }
}

unsafe fn upload_pbo_plane(
    gl: &glow::Context,
    pbo: glow::Buffer,
    tex: glow::Texture,
    width: i32,
    height: i32,
    format: u32,
    data: &[u8],
) {
    unsafe {
        gl.bind_buffer(glow::PIXEL_UNPACK_BUFFER, Some(pbo));
        let ptr = gl.map_buffer_range(
            glow::PIXEL_UNPACK_BUFFER,
            0,
            data.len() as i32,
            glow::MAP_WRITE_BIT | glow::MAP_INVALIDATE_BUFFER_BIT,
        );
        if !ptr.is_null() {
            let dst = slice::from_raw_parts_mut(ptr, data.len());
            dst.copy_from_slice(data);
            gl.unmap_buffer(glow::PIXEL_UNPACK_BUFFER);
            gl.bind_texture(glow::TEXTURE_2D, Some(tex));
            gl.tex_sub_image_2d(
                glow::TEXTURE_2D,
                0,
                0,
                0,
                width,
                height,
                format,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::BufferOffset(0),
            );
        } else {
            gl.bind_buffer(glow::PIXEL_UNPACK_BUFFER, None);
            gl.bind_texture(glow::TEXTURE_2D, Some(tex));
            gl.tex_sub_image_2d(
                glow::TEXTURE_2D,
                0,
                0,
                0,
                width,
                height,
                format,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(Some(data)),
            );
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotMediaKind {
    None = 0,
    Rgba = 1,
    Nv12 = 2,
}

#[derive(Debug, Clone, Copy)]
struct SlotBinding {
    kind: SlotMediaKind,
    width: u32,
    height: u32,
    fit: FitMode,
    framebuffer: bool,
    colorspace: ColorSpace,
    rgba_tex: Option<glow::Texture>,
    y_tex: Option<glow::Texture>,
    uv_tex: Option<glow::Texture>,
}

pub struct MediaSlotGpu {
    pub kind: SlotMediaKind,
    pub width: u32,
    pub height: u32,
    pub fit: FitMode,
    pub colorspace: ColorSpace,
    rgba_tex: Option<OwnedTexture>,
    y_tex: Option<OwnedTexture>,
    uv_tex: Option<OwnedTexture>,
    pbo_ring: Option<Nv12PboRing>,
    cuda_textures: Option<crate::gpu::cuda::CudaTextures>,
    cuda_staging: Option<crate::gpu::cuda::CudaTextures>,
    cuda_binding: Option<(glow::Texture, glow::Texture)>,
}

static CUDA_CLEANUP_FAILED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

thread_local! {
    // This renderer is confined to its EGL thread. ManuallyDrop keeps failed
    // driver registrations and their libraries alive even at thread shutdown.
    static RETIRED_CUDA: std::cell::RefCell<std::mem::ManuallyDrop<Vec<crate::gpu::cuda::CudaTextures>>> =
        const { std::cell::RefCell::new(std::mem::ManuallyDrop::new(Vec::new())) };
}

pub(crate) fn retry_retired_cuda(gl: &glow::Context) {
    RETIRED_CUDA.with(|retired| {
        retired.borrow_mut().retain_mut(|textures| {
            if let Err(err) = textures.destroy(gl) {
                tracing::debug!("retired CUDA cleanup still pending: {err:#}");
                true
            } else {
                false
            }
        });
    });
}
fn retire_owned<T>(
    owner: &mut Option<T>,
    retired: &mut Vec<T>,
    clear: impl FnOnce(&mut T) -> Result<()>,
) -> bool {
    if let Some(mut value) = owner.take()
        && let Err(err) = clear(&mut value)
    {
        tracing::error!("CUDA texture cleanup failed; retaining resources for retry: {err:#}");
        retired.push(value);
        return true;
    }
    false
}

impl MediaSlotGpu {
    pub fn empty() -> Self {
        Self {
            kind: SlotMediaKind::None,
            width: 0,
            height: 0,
            fit: FitMode::Cover,
            colorspace: ColorSpace::Bt709Tv,
            rgba_tex: None,
            y_tex: None,
            uv_tex: None,
            pbo_ring: None,
            cuda_textures: None,
            cuda_staging: None,
            cuda_binding: None,
        }
    }

    fn as_binding(&self) -> SlotBinding {
        SlotBinding {
            kind: self.kind,
            width: self.width,
            height: self.height,
            fit: self.fit,
            colorspace: self.colorspace,
            framebuffer: false,
            rgba_tex: self.rgba_tex.as_ref().map(|t| t.handle),
            y_tex: self
                .cuda_binding
                .map(|t| t.0)
                .or_else(|| self.y_tex.as_ref().map(|t| t.handle)),
            uv_tex: self
                .cuda_binding
                .map(|t| t.1)
                .or_else(|| self.uv_tex.as_ref().map(|t| t.handle)),
        }
    }

    pub fn set_rgba(
        &mut self,
        gl: &glow::Context,
        width: u32,
        height: u32,
        fit: FitMode,
        pixels: &[u8],
    ) -> Result<()> {
        if self.rgba_tex.is_none() {
            self.rgba_tex = Some(OwnedTexture::new(gl)?);
        }
        if let Some(tex) = &mut self.rgba_tex {
            tex.upload_rgba(gl, width, height, pixels);
        }
        // The old CUDA binding remains usable until the replacement exists.
        self.retire_cuda(gl);
        self.clear_cpu_nv12(gl);
        self.kind = SlotMediaKind::Rgba;
        self.width = width;
        self.height = height;
        self.fit = fit;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn set_nv12(
        &mut self,
        gl: &glow::Context,
        width: u32,
        height: u32,
        fit: FitMode,
        colorspace: ColorSpace,
        y_plane: &[u8],
        uv_plane: &[u8],
    ) -> Result<()> {
        if self.y_tex.is_none() {
            self.y_tex = Some(OwnedTexture::new(gl)?);
        }
        if self.uv_tex.is_none() {
            self.uv_tex = Some(OwnedTexture::new(gl)?);
        }
        if self.pbo_ring.is_none() {
            self.pbo_ring = Some(Nv12PboRing::new(gl)?);
        }
        if let (Some(y_tex), Some(uv_tex), Some(ring)) =
            (&mut self.y_tex, &mut self.uv_tex, &mut self.pbo_ring)
        {
            ring.upload_nv12(gl, y_tex, uv_tex, width, height, y_plane, uv_plane);
        }
        // Do not let a failed CUDA unregister prevent CPU fallback.
        self.retire_cuda(gl);
        self.clear_rgba(gl);
        self.kind = SlotMediaKind::Nv12;
        self.width = width;
        self.height = height;
        self.fit = fit;
        self.colorspace = colorspace;
        Ok(())
    }

    pub fn set_cuda(
        &mut self,
        gl: &glow::Context,
        frame: &crate::media::gst_cuda::CudaFrame,
        fit: FitMode,
    ) -> Result<()> {
        if CUDA_CLEANUP_FAILED.load(std::sync::atomic::Ordering::Acquire) {
            bail!("CUDA uploads disabled after a graphics-resource cleanup failure");
        }
        if self.cuda_staging.is_none() {
            self.cuda_staging = Some(crate::gpu::cuda::CudaTextures::new()?);
        }
        let binding =
            frame.with_mapped(|mapped| self.cuda_staging.as_mut().unwrap().upload(gl, mapped))?;
        std::mem::swap(&mut self.cuda_textures, &mut self.cuda_staging);
        self.clear_rgba(gl);
        self.clear_cpu_nv12(gl);
        self.cuda_binding = Some(binding);
        self.kind = SlotMediaKind::Nv12;
        self.width = frame.width;
        self.height = frame.height;
        self.fit = fit;
        self.colorspace = frame.colorspace;
        Ok(())
    }

    pub fn clear(&mut self, gl: &glow::Context) {
        if let Err(err) = self.try_clear(gl) {
            tracing::error!("CUDA texture cleanup failed: {err:#}");
        }
    }

    pub fn try_clear(&mut self, gl: &glow::Context) -> Result<()> {
        self.clear_nv12(gl)?;
        self.clear_rgba(gl);
        self.kind = SlotMediaKind::None;
        self.width = 0;
        self.height = 0;
        Ok(())
    }

    pub fn estimated_gpu_bytes(&self) -> u64 {
        match self.kind {
            SlotMediaKind::None => 0,
            SlotMediaKind::Rgba => u64::from(self.width) * u64::from(self.height) * 4,
            SlotMediaKind::Nv12 => {
                let y_uv = (u64::from(self.width) * u64::from(self.height) * 3) / 2;
                let cuda_mult = if self.cuda_textures.is_some() && self.cuda_staging.is_some() {
                    2
                } else {
                    1
                };
                let pbo_bytes = self
                    .pbo_ring
                    .as_ref()
                    .map_or(0, |r| 2 * (r.y_bytes as u64 + r.uv_bytes as u64));
                y_uv * cuda_mult + pbo_bytes
            }
        }
    }

    fn clear_rgba(&mut self, gl: &glow::Context) {
        if let Some(tex) = self.rgba_tex.take() {
            tex.destroy(gl);
        }
    }

    fn clear_nv12(&mut self, gl: &glow::Context) -> Result<()> {
        self.clear_cuda(gl)?;
        self.clear_cpu_nv12(gl);
        Ok(())
    }

    fn clear_cuda(&mut self, gl: &glow::Context) -> Result<()> {
        self.retire_cuda(gl);
        Ok(())
    }

    fn retire_cuda(&mut self, gl: &glow::Context) {
        self.cuda_binding = None;
        RETIRED_CUDA.with(|retired| {
            let mut retired = retired.borrow_mut();
            let active_failed = retire_owned(&mut self.cuda_textures, &mut retired, |textures| {
                textures.destroy(gl)
            });
            let staging_failed = retire_owned(&mut self.cuda_staging, &mut retired, |textures| {
                textures.destroy(gl)
            });
            if active_failed || staging_failed {
                CUDA_CLEANUP_FAILED.store(true, std::sync::atomic::Ordering::Release);
            }
        });
    }

    fn clear_cpu_nv12(&mut self, gl: &glow::Context) {
        if let Some(tex) = self.y_tex.take() {
            tex.destroy(gl);
        }
        if let Some(tex) = self.uv_tex.take() {
            tex.destroy(gl);
        }
        if let Some(ring) = self.pbo_ring.take() {
            ring.destroy(gl);
        }
    }

    pub fn destroy(mut self, gl: &glow::Context) {
        self.clear(gl);
    }
}

struct ResolveTarget {
    fbo: glow::Framebuffer,
    color_tex: OwnedTexture,
}

impl ResolveTarget {
    fn new(gl: &glow::Context, width: u32, height: u32) -> Result<Self> {
        let mut color_tex = OwnedTexture::new(gl)?;
        color_tex.alloc_storage(gl, glow::RGBA8 as i32, glow::RGBA, width, height);
        let fbo = unsafe {
            gl.create_framebuffer()
                .map_err(|e| anyhow::anyhow!("glCreateFramebuffer failed: {e}"))?
        };
        unsafe {
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo));
            gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D,
                Some(color_tex.handle),
                0,
            );
            let status = gl.check_framebuffer_status(glow::FRAMEBUFFER);
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            if status != glow::FRAMEBUFFER_COMPLETE {
                gl.delete_framebuffer(fbo);
                color_tex.destroy(gl);
                bail!("resolve framebuffer incomplete: 0x{status:x}");
            }
        }
        Ok(Self { fbo, color_tex })
    }

    fn ensure_size(&mut self, gl: &glow::Context, width: u32, height: u32) {
        if self.color_tex.width != width || self.color_tex.height != height {
            self.color_tex
                .alloc_storage(gl, glow::RGBA8 as i32, glow::RGBA, width, height);
        }
    }

    fn destroy(self, gl: &glow::Context) {
        unsafe { gl.delete_framebuffer(self.fbo) };
        self.color_tex.destroy(gl);
    }
}

pub struct DrawParams<'a> {
    pub width: u32,
    pub height: u32,
    pub time_secs: f32,
    pub delta_secs: f32,
    pub progress: f32,
    pub frame_index: i32,
    pub params: [f32; 4],
    pub output_index: i32,
    pub output_name: &'a str,
    pub slot0: &'a MediaSlotGpu,
    pub slot1: Option<&'a MediaSlotGpu>,
    pub custom_program: Option<&'a GlProgram>,
    pub transition_program: Option<&'a GlProgram>,
    pub in_transition: bool,
    pub custom_textures: &'a BTreeMap<String, TextureId>,
    pub custom_uniforms: &'a BTreeMap<String, UniformValue>,
    pub transient_uniforms: &'a BTreeMap<String, UniformValue>,
}

pub struct Renderer {
    pub gl: glow::Context,
    vao: glow::VertexArray,
    dummy_tex: OwnedTexture,
    passthrough_program: GlProgram,
    resolve_targets: HashMap<String, ResolveTarget>,
    max_texture_units: u32,
    max_texture_size: u32,
}

type SlotUniformLocs<'a> = (
    Option<&'a glow::UniformLocation>,
    Option<&'a glow::UniformLocation>,
    Option<&'a glow::UniformLocation>,
    Option<&'a glow::UniformLocation>,
    Option<&'a glow::UniformLocation>,
    Option<&'a glow::UniformLocation>,
    Option<&'a glow::UniformLocation>,
);

impl Renderer {
    pub fn new(gl: glow::Context) -> Result<Self> {
        let vao = unsafe {
            gl.create_vertex_array()
                .map_err(|e| anyhow::anyhow!("glCreateVertexArray failed: {e}"))?
        };
        let mut dummy_tex = OwnedTexture::new(&gl)?;
        dummy_tex.upload_rgba(&gl, 1, 1, &[0, 0, 0, 255]);
        let passthrough_program = GlProgram::compile(&gl, BUILTIN_PASSTHROUGH)?;
        let max_texture_units =
            unsafe { gl.get_parameter_i32(glow::MAX_TEXTURE_IMAGE_UNITS) }.max(8) as u32;
        let max_texture_size =
            unsafe { gl.get_parameter_i32(glow::MAX_TEXTURE_SIZE) }.max(2048) as u32;

        Ok(Self {
            gl,
            vao,
            dummy_tex,
            passthrough_program,
            resolve_targets: HashMap::new(),
            max_texture_units,
            max_texture_size,
        })
    }

    pub fn max_texture_size(&self) -> u32 {
        self.max_texture_size
    }

    pub fn resolve_target_gpu_bytes(&self, output_name: &str) -> u64 {
        self.resolve_targets.get(output_name).map_or(0, |t| {
            u64::from(t.color_tex.width) * u64::from(t.color_tex.height) * 4
        })
    }

    pub fn remove_output_cache(&mut self, output_name: &str) {
        if let Some(target) = self.resolve_targets.remove(output_name) {
            target.destroy(&self.gl);
        }
    }

    #[cfg(test)]
    pub(crate) fn resolve_target_count(&self) -> usize {
        self.resolve_targets.len()
    }

    fn ensure_resolve_target(
        &mut self,
        output_name: &str,
        width: u32,
        height: u32,
    ) -> Result<(glow::Framebuffer, glow::Texture)> {
        let w = width.max(1);
        let h = height.max(1);
        let target = match self.resolve_targets.get_mut(output_name) {
            Some(t) => {
                t.ensure_size(&self.gl, w, h);
                t
            }
            None => {
                let created = ResolveTarget::new(&self.gl, w, h)?;
                self.resolve_targets
                    .entry(output_name.to_string())
                    .or_insert(created)
            }
        };
        Ok((target.fbo, target.color_tex.handle))
    }

    pub fn draw(&mut self, params: &DrawParams<'_>, texture_pool: &TexturePool) -> Result<()> {
        let w = params.width.max(1);
        let h = params.height.max(1);
        let raw_slot0 = params.slot0.as_binding();
        let raw_slot1 = params.slot1.map(MediaSlotGpu::as_binding);

        if params.in_transition
            && let Some(custom) = params.custom_program
            && let Some(transition) = params.transition_program
        {
            // Render the transition into an offscreen FBO first, then feed it as
            // slot0 (and iChannel0 for Shadertoy) into the active custom shader.
            let (fbo, color_tex) = self.ensure_resolve_target(params.output_name, w, h)?;
            unsafe {
                self.gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo));
                self.gl.viewport(0, 0, w as i32, h as i32);
                self.gl.clear_color(0.0, 0.0, 0.0, 1.0);
                self.gl.clear(glow::COLOR_BUFFER_BIT);
                self.gl.use_program(Some(transition.handle));
                self.bind_common_uniforms(transition, params, raw_slot0, raw_slot1, None);
                self.bind_user_bindings(transition, params, texture_pool);
                self.gl.bind_vertex_array(Some(self.vao));
                self.gl.draw_arrays(glow::TRIANGLES, 0, 3);
                self.gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            }

            let resolved_slot0 = SlotBinding {
                kind: SlotMediaKind::Rgba,
                width: w,
                height: h,
                fit: params.slot0.fit,
                colorspace: params.slot0.colorspace,
                framebuffer: true,
                rgba_tex: Some(color_tex),
                y_tex: None,
                uv_tex: None,
            };
            let ichannel0_tex = (custom.assembled.mode == ShaderMode::Shadertoy
                && custom.uses_ichannel0)
                .then_some(color_tex);

            unsafe {
                self.gl.bind_framebuffer(glow::FRAMEBUFFER, None);
                self.gl.viewport(0, 0, w as i32, h as i32);
                self.gl.clear_color(0.0, 0.0, 0.0, 1.0);
                self.gl.clear(glow::COLOR_BUFFER_BIT);
                self.gl.use_program(Some(custom.handle));
                self.bind_common_uniforms(custom, params, resolved_slot0, None, ichannel0_tex);
                self.bind_user_bindings(custom, params, texture_pool);
                self.gl.bind_vertex_array(Some(self.vao));
                self.gl.draw_arrays(glow::TRIANGLES, 0, 3);
                self.gl.bind_vertex_array(None);
                self.gl.use_program(None);
            }
            return Ok(());
        }

        let explicit_program = if params.in_transition {
            params.transition_program
        } else {
            params.custom_program
        };
        let need_ichannel0 = {
            let p = explicit_program.unwrap_or(&self.passthrough_program);
            p.assembled.mode == ShaderMode::Shadertoy
                && p.uses_ichannel0
                && params.slot0.kind != SlotMediaKind::None
        };

        let mut ichannel0_tex: Option<glow::Texture> = None;
        if need_ichannel0 {
            let (fbo, color_tex) = self.ensure_resolve_target(params.output_name, w, h)?;
            unsafe {
                self.gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo));
                self.gl.viewport(0, 0, w as i32, h as i32);
                self.gl.use_program(Some(self.passthrough_program.handle));
                self.bind_common_uniforms(&self.passthrough_program, params, raw_slot0, None, None);
                self.gl.bind_vertex_array(Some(self.vao));
                self.gl.draw_arrays(glow::TRIANGLES, 0, 3);
                self.gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            }
            ichannel0_tex = Some(color_tex);
        }

        let program = explicit_program.unwrap_or(&self.passthrough_program);

        unsafe {
            self.gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            self.gl.viewport(0, 0, w as i32, h as i32);
            self.gl.clear_color(0.0, 0.0, 0.0, 1.0);
            self.gl.clear(glow::COLOR_BUFFER_BIT);
            self.gl.use_program(Some(program.handle));
            self.bind_common_uniforms(program, params, raw_slot0, raw_slot1, ichannel0_tex);
            self.bind_user_bindings(program, params, texture_pool);
            self.gl.bind_vertex_array(Some(self.vao));
            self.gl.draw_arrays(glow::TRIANGLES, 0, 3);
            self.gl.bind_vertex_array(None);
            self.gl.use_program(None);
        }
        Ok(())
    }

    unsafe fn bind_common_uniforms(
        &self,
        program: &GlProgram,
        params: &DrawParams<'_>,
        slot0: SlotBinding,
        slot1: Option<SlotBinding>,
        ichannel0: Option<glow::Texture>,
    ) {
        let gl = &self.gl;
        let locs = &program.locs;
        unsafe {
            gl.uniform_2_f32(
                locs.u_resolution.as_ref(),
                params.width.max(1) as f32,
                params.height.max(1) as f32,
            );
            gl.uniform_1_f32(locs.u_time.as_ref(), params.time_secs);
            gl.uniform_1_f32(locs.u_delta.as_ref(), params.delta_secs);
            gl.uniform_1_f32(locs.u_progress.as_ref(), params.progress);
            gl.uniform_1_i32(locs.u_frame.as_ref(), params.frame_index);
            gl.uniform_4_f32(
                locs.u_params.as_ref(),
                params.params[0],
                params.params[1],
                params.params[2],
                params.params[3],
            );
            gl.uniform_1_i32(locs.u_output_index.as_ref(), params.output_index);
            gl.uniform_1_i32(locs.u_slot0_framebuffer.as_ref(), slot0.framebuffer as i32);

            // Slot 0 -> texture units 0, 1, 2
            self.bind_slot(
                slot0,
                [0, 1, 2],
                (
                    locs.u_tex0_rgba.as_ref(),
                    locs.u_tex0_y.as_ref(),
                    locs.u_tex0_uv.as_ref(),
                    locs.u_slot0_kind.as_ref(),
                    locs.u_slot0_size.as_ref(),
                    locs.u_slot0_fit.as_ref(),
                    locs.u_slot0_cs.as_ref(),
                ),
            );

            // Slot 1 -> texture units 3, 4, 5
            if let Some(s1) = slot1 {
                self.bind_slot(
                    s1,
                    [3, 4, 5],
                    (
                        locs.u_tex1_rgba.as_ref(),
                        locs.u_tex1_y.as_ref(),
                        locs.u_tex1_uv.as_ref(),
                        locs.u_slot1_kind.as_ref(),
                        locs.u_slot1_size.as_ref(),
                        locs.u_slot1_fit.as_ref(),
                        locs.u_slot1_cs.as_ref(),
                    ),
                );
            } else {
                gl.uniform_1_i32(locs.u_slot1_kind.as_ref(), 0);
            }

            // Optional Shadertoy iChannel0 -> texture unit 6
            gl.active_texture(glow::TEXTURE6);
            gl.bind_texture(
                glow::TEXTURE_2D,
                Some(ichannel0.unwrap_or(self.dummy_tex.handle)),
            );
            gl.uniform_1_i32(locs.i_channel0.as_ref(), 6);
        }
    }

    unsafe fn bind_user_bindings(
        &self,
        program: &GlProgram,
        params: &DrawParams<'_>,
        texture_pool: &TexturePool,
    ) {
        let gl = &self.gl;
        for (next_unit, sampler) in (7_u32..).zip(program.user_samplers.iter()) {
            if next_unit >= self.max_texture_units {
                break;
            }
            let (tex_handle, tex_w, tex_h) = params
                .custom_textures
                .get(&sampler.name)
                .and_then(|id| texture_pool.texture_info(*id))
                .unwrap_or((self.dummy_tex.handle, 1, 1));
            unsafe {
                gl.active_texture(glow::TEXTURE0 + next_unit);
                gl.bind_texture(glow::TEXTURE_2D, Some(tex_handle));
                gl.uniform_1_i32(Some(&sampler.location), next_unit as i32);
                if let Some(ref size_loc) = sampler.size_location {
                    gl.uniform_2_f32(Some(size_loc), tex_w.max(1) as f32, tex_h.max(1) as f32);
                }
            }
        }

        for uniform in &program.user_uniforms {
            if let Some(value) = params
                .transient_uniforms
                .get(&uniform.name)
                .or_else(|| params.custom_uniforms.get(&uniform.name))
            {
                let _ = upload_uniform(gl, &uniform.location, uniform.utype, value);
            }
        }
    }

    unsafe fn bind_slot(&self, slot: SlotBinding, units: [u32; 3], locs: SlotUniformLocs<'_>) {
        let gl = &self.gl;
        let dummy = self.dummy_tex.handle;
        unsafe {
            gl.active_texture(glow::TEXTURE0 + units[0]);
            gl.bind_texture(glow::TEXTURE_2D, Some(slot.rgba_tex.unwrap_or(dummy)));
            gl.uniform_1_i32(locs.0, units[0] as i32);

            gl.active_texture(glow::TEXTURE0 + units[1]);
            gl.bind_texture(glow::TEXTURE_2D, Some(slot.y_tex.unwrap_or(dummy)));
            gl.uniform_1_i32(locs.1, units[1] as i32);

            gl.active_texture(glow::TEXTURE0 + units[2]);
            gl.bind_texture(glow::TEXTURE_2D, Some(slot.uv_tex.unwrap_or(dummy)));
            gl.uniform_1_i32(locs.2, units[2] as i32);

            gl.uniform_1_i32(locs.3, slot.kind as i32);
            gl.uniform_2_f32(locs.4, slot.width.max(1) as f32, slot.height.max(1) as f32);
            gl.uniform_1_i32(locs.5, slot.fit.as_gl_int());
            gl.uniform_1_i32(locs.6, slot.colorspace.as_gl_int());
        }
    }

    pub fn destroy(self) {
        for (_, target) in self.resolve_targets {
            target.destroy(&self.gl);
        }
        self.passthrough_program.destroy(&self.gl);
        self.dummy_tex.destroy(&self.gl);
        unsafe {
            self.gl.delete_vertex_array(self.vao);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failed_cleanup_retains_owner_for_retry() {
        let mut owner = Some(7);
        let mut retired = Vec::new();
        assert!(retire_owned(&mut owner, &mut retired, |_| Err(
            anyhow::anyhow!("context unavailable")
        )));
        assert_eq!(owner, None);
        assert_eq!(retired, vec![7]);
        let mut next = Some(9);
        assert!(!retire_owned(&mut next, &mut retired, |_| Ok(())));
        assert_eq!(next, None);
        assert_eq!(retired, vec![7]);
    }

    #[test]
    #[ignore = "requires an NVIDIA CUDA/GLES device, GStreamer nvcodec and ffmpeg"]
    fn cuda_video_preserves_orientation_and_loops() {
        use crate::media::gst_cuda::CudaDecoder;
        use std::time::{Duration, Instant};

        let path = std::env::temp_dir().join(format!("glwall-cuda-{}.mp4", std::process::id()));
        let status = std::process::Command::new("ffmpeg")
            .args([
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=red:s=128x128:r=10:d=0.4",
                "-vf",
                "drawbox=x=0:y=64:w=128:h=64:color=blue:t=fill",
                "-threads",
                "1",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
            ])
            .arg(&path)
            .status()
            .unwrap();
        assert!(status.success(), "failed to create video fixture");

        let egl = crate::gpu::egl::EglState::new_test_pbuffer(8, 8).unwrap();
        let gl = egl.create_glow_context().unwrap();
        let mut renderer = Renderer::new(gl).unwrap();
        let mut slot = MediaSlotGpu::empty();
        let mut decoder = CudaDecoder::open(&path, ColorSpace::Bt601Tv).unwrap();
        let deadline = Instant::now() + Duration::from_secs(15);
        let mut frames = 0;
        while frames < 7 && Instant::now() < deadline {
            if let Some(frame) = decoder.pull(Duration::from_millis(100)).unwrap() {
                slot.set_cuda(&renderer.gl, &frame, FitMode::Stretch)
                    .unwrap();
                frames += 1;
            }
        }
        assert_eq!(
            frames, 7,
            "CUDA decoder failed to produce frames across a loop"
        );
        let pool = TexturePool::new();
        let textures = BTreeMap::new();
        let uniforms = BTreeMap::new();
        let mut params = DrawParams {
            width: 8,
            height: 8,
            time_secs: 0.0,
            delta_secs: 0.0,
            progress: 1.0,
            frame_index: 0,
            params: [0.0; 4],
            output_index: 0,
            output_name: "cuda-test",
            slot0: &slot,
            slot1: None,
            custom_program: None,
            transition_program: None,
            in_transition: false,
            custom_textures: &textures,
            custom_uniforms: &uniforms,
            transient_uniforms: &BTreeMap::new(),
        };
        renderer.draw(&params, &pool).unwrap();
        let mut pixels = [0u8; 8 * 8 * 4];
        unsafe {
            renderer.gl.read_pixels(
                0,
                0,
                8,
                8,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelPackData::Slice(Some(&mut pixels)),
            );
        }
        let pixel = |x: usize, y: usize| &pixels[(y * 8 + x) * 4..(y * 8 + x) * 4 + 4];
        let top = pixel(4, 6);
        let bottom = pixel(4, 1);
        assert!(top[0] > 150 && top[2] < 100, "expected red top: {top:?}");
        assert!(
            bottom[2] > 150 && bottom[0] < 100,
            "expected blue bottom: {bottom:?}"
        );

        let mut previous = MediaSlotGpu::empty();
        previous
            .set_rgba(
                &renderer.gl,
                8,
                8,
                FitMode::Stretch,
                &[0, 255, 0, 255].repeat(64),
            )
            .unwrap();
        let transition =
            GlProgram::compile(&renderer.gl, crate::gpu::shader::BUILTIN_FADE).unwrap();
        let postprocess = GlProgram::compile(&renderer.gl, BUILTIN_PASSTHROUGH).unwrap();
        params.slot1 = Some(&previous);
        params.transition_program = Some(&transition);
        params.custom_program = Some(&postprocess);
        params.in_transition = true;
        params.progress = 0.0;
        renderer.draw(&params, &pool).unwrap();
        unsafe {
            renderer.gl.read_pixels(
                0,
                0,
                8,
                8,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelPackData::Slice(Some(&mut pixels)),
            );
        }
        let old = &pixels[(4 * 8 + 4) * 4..(4 * 8 + 4) * 4 + 4];
        assert!(
            old[1] > 200 && old[0] < 20 && old[2] < 20,
            "expected old green image: {old:?}"
        );
        params.progress = 1.0;
        renderer.draw(&params, &pool).unwrap();
        unsafe {
            renderer.gl.read_pixels(
                0,
                0,
                8,
                8,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelPackData::Slice(Some(&mut pixels)),
            );
        }
        let top = &pixels[(6 * 8 + 4) * 4..(6 * 8 + 4) * 4 + 4];
        let bottom = &pixels[(8 + 4) * 4..(8 + 4) * 4 + 4];
        assert!(
            top[0] > 150 && top[2] < 100,
            "expected red top after transition: {top:?}"
        );
        assert!(
            bottom[2] > 150 && bottom[0] < 100,
            "expected blue bottom after transition: {bottom:?}"
        );
        previous.destroy(&renderer.gl);
        transition.destroy(&renderer.gl);
        postprocess.destroy(&renderer.gl);
        drop(decoder);
        slot.destroy(&renderer.gl);
        pool.destroy(&renderer.gl);
        renderer.destroy();
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn wallpaper_keeps_orientation_through_transition_with_postprocessing() {
        let egl = crate::gpu::egl::EglState::new_test_pbuffer(8, 8).unwrap();
        let gl = egl.create_glow_context().unwrap();
        let mut renderer = Renderer::new(gl).unwrap();
        let mut image = Vec::new();
        for y in 0..8 {
            for x in 0..8 {
                image.extend_from_slice(match (y < 4, x < 4) {
                    (true, true) => &[255, 0, 0, 255],
                    (true, false) => &[0, 255, 0, 255],
                    (false, true) => &[0, 0, 255, 255],
                    (false, false) => &[255, 255, 0, 255],
                });
            }
        }
        let mut slot = MediaSlotGpu::empty();
        slot.set_rgba(&renderer.gl, 8, 8, FitMode::Stretch, &image)
            .unwrap();
        let custom = GlProgram::compile(&renderer.gl, BUILTIN_PASSTHROUGH).unwrap();
        let custom_uv = GlProgram::compile(
            &renderer.gl,
            "vec4 wallpaper(vec2 uv) { return texture(u_tex0_rgba, wallpaperUv(uv)); }",
        )
        .unwrap();
        let shadertoy = GlProgram::compile(
            &renderer.gl,
            "void mainImage(out vec4 c, in vec2 p) { c = texture(iChannel0, p / iResolution.xy); }",
        )
        .unwrap();
        let pool = TexturePool::new();
        let textures = BTreeMap::new();
        let uniforms = BTreeMap::new();
        let mut failures = Vec::new();
        for (name, source) in [
            ("fade", crate::gpu::shader::BUILTIN_FADE),
            ("dissolve", crate::gpu::shader::BUILTIN_DISSOLVE),
            ("pixelate", crate::gpu::shader::BUILTIN_PIXELATE),
            ("wipe", crate::gpu::shader::BUILTIN_WIPE),
        ] {
            let transition = GlProgram::compile(&renderer.gl, source).unwrap();
            for (postprocess, program) in [
                ("none", None),
                ("sampleWallpaper", Some(&custom)),
                ("wallpaperUv", Some(&custom_uv)),
                ("iChannel0", Some(&shadertoy)),
            ] {
                for in_transition in [false, true] {
                    for progress in [0.0, 0.5, 1.0] {
                        let params = DrawParams {
                            width: 8,
                            height: 8,
                            time_secs: 0.0,
                            delta_secs: 0.0,
                            progress,
                            frame_index: 0,
                            params: [0.0; 4],
                            output_index: 0,
                            output_name: "test",
                            slot0: &slot,
                            slot1: Some(&slot),
                            custom_program: program,
                            transition_program: Some(&transition),
                            in_transition,
                            custom_textures: &textures,
                            custom_uniforms: &uniforms,
                            transient_uniforms: &BTreeMap::new(),
                        };
                        renderer.draw(&params, &pool).unwrap();
                        let mut pixels = [0u8; 8 * 8 * 4];
                        unsafe {
                            renderer.gl.read_pixels(
                                0,
                                0,
                                8,
                                8,
                                glow::RGBA,
                                glow::UNSIGNED_BYTE,
                                glow::PixelPackData::Slice(Some(&mut pixels)),
                            );
                        }
                        // glReadPixels begins at the bottom-left.
                        let pixel =
                            |x: usize, y: usize| &pixels[(y * 8 + x) * 4..(y * 8 + x) * 4 + 4];
                        let corners = [pixel(1, 6), pixel(6, 6), pixel(1, 1), pixel(6, 1)];
                        if corners
                            != [
                                &[255, 0, 0, 255][..],
                                &[0, 255, 0, 255][..],
                                &[0, 0, 255, 255][..],
                                &[255, 255, 0, 255][..],
                            ]
                        {
                            failures.push(format!("{name} postprocess={postprocess} in_transition={in_transition} progress={progress}: {corners:?}"));
                        }
                    }
                }
            }
            transition.destroy(&renderer.gl);
        }
        slot.destroy(&renderer.gl);
        custom.destroy(&renderer.gl);
        custom_uv.destroy(&renderer.gl);
        shadertoy.destroy(&renderer.gl);
        pool.destroy(&renderer.gl);
        renderer.destroy();
        assert!(
            failures.is_empty(),
            "orientation failures:\n{}",
            failures.join("\n")
        );
    }

    fn test_key(name: &str, filter: TextureFilter) -> TextureKey {
        TextureKey {
            path: PathBuf::from(name),
            filter,
            wrap: TextureWrap::ClampToEdge,
            flip_y: false,
            fingerprint: TextureFingerprint::default(),
        }
    }

    #[test]
    fn texture_pool_refcounts_and_eviction() {
        let mut pool: TexturePoolCore<&'static str> = TexturePoolCore::new();
        let k1 = test_key("/tmp/depth_a.png", TextureFilter::Linear);
        let k2 = test_key("/tmp/depth_a.png", TextureFilter::Nearest);

        assert!(pool.acquire(&k1).is_none());
        let (id1, extra) = pool.insert_or_acquire(k1.clone(), 512, 288, "tex_linear");
        assert!(extra.is_none());
        assert_eq!(pool.ref_count(id1), 1);

        // Second acquire of the exact same key shares ID and bumps refcount.
        assert_eq!(pool.acquire(&k1), Some(id1));
        assert_eq!(pool.ref_count(id1), 2);

        // Different filter creates a separate pool entry so outputs don't stomp sampler state.
        let (id2, extra2) = pool.insert_or_acquire(k2.clone(), 512, 288, "tex_nearest");
        assert!(extra2.is_none());
        assert_ne!(id1, id2);
        assert_eq!(pool.len(), 2);

        // First release decrements without evicting.
        assert!(pool.release(id1).is_none());
        assert_eq!(pool.ref_count(id1), 1);

        // Final release evicts and returns underlying handle for destruction.
        assert_eq!(pool.release(id1), Some("tex_linear"));
        assert!(pool.acquire(&k1).is_none());
        assert_eq!(pool.len(), 1);
    }

    #[test]
    fn texture_key_changes_when_file_updated_in_place() {
        let dir = std::env::temp_dir().join(format!("glwall-texkey-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("depth.pgm");
        std::fs::write(&path, b"P5\n2 2\n255\n\x10\x20\x30\x40").unwrap();

        let spec = TextureSpec {
            path: path.clone(),
            filter: TextureFilter::Linear,
            wrap: TextureWrap::ClampToEdge,
            flip_y: false,
        };
        let key1 = TextureKey::from_spec(&spec);

        let mut pool: TexturePoolCore<&'static str> = TexturePoolCore::new();
        let (id1, _) = pool.insert_or_acquire(key1.clone(), 2, 2, "first");
        assert_eq!(pool.acquire(&key1), Some(id1));

        // Overwrite in place with different contents of the exact same byte length.
        std::fs::write(&path, b"P5\n2 2\n255\n\x90\xa0\xb0\xc0").unwrap();
        let key2 = TextureKey::from_spec(&spec);
        assert_ne!(
            key1, key2,
            "TextureKey must change when file contents change in place"
        );
        assert!(
            pool.acquire(&key2).is_none(),
            "stale cache entry must not be reused after in-place update"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn thousand_image_and_shader_switches_do_not_grow_gpu_objects() {
        let egl = crate::gpu::egl::EglState::new_test_pbuffer(16, 16).unwrap();
        let gl = egl.create_glow_context().unwrap();
        let mut renderer = Renderer::new(gl).unwrap();
        let mut pool = TexturePool::new();
        let mut slot0 = MediaSlotGpu::empty();

        let img_a = [200u8, 40, 20, 255].repeat(32 * 32);
        let img_b = [20u8, 120, 220, 255].repeat(32 * 32);
        let aux_px = [128u8, 64, 32, 255].repeat(16 * 16);

        let mut current_prog: Option<GlProgram> = None;
        let mut max_program_id = 0u32;
        let mut max_texture_id = 0u32;
        let mut warm_private_dirty_kb: Option<u64> = None;

        for i in 0..1000usize {
            let mut slot1 = Some(std::mem::replace(&mut slot0, MediaSlotGpu::empty()));
            let pixels = if i % 2 == 0 { &img_a } else { &img_b };
            slot0
                .set_rgba(&renderer.gl, 32, 32, FitMode::Cover, pixels)
                .unwrap();
            if let Some(ref tex) = slot0.rgba_tex {
                max_texture_id = max_texture_id.max(tex.handle.0.get());
            }

            let shader_src = format!(
                "uniform sampler2D u_aux;\nuniform float u_mix;\nvec4 wallpaper(vec2 uv) {{\n    vec4 a = sampleWallpaper(uv);\n    vec4 b = texture(u_aux, uv);\n    return mix(a, b, 0.25 + clamp(u_mix, 0.0, 0.5) + {:.6});\n}}",
                (i % 7) as f32 * 0.0001
            );
            let next_prog = GlProgram::compile(&renderer.gl, &shader_src).unwrap();
            max_program_id = max_program_id.max(next_prog.handle.0.get());
            if let Some(old_prog) = current_prog.replace(next_prog) {
                let old_h = old_prog.handle;
                old_prog.destroy(&renderer.gl);
                if i % 100 == 0 {
                    unsafe { renderer.gl.flush() };
                    assert!(
                        !unsafe { renderer.gl.is_program(old_h) },
                        "replaced GL program was not deleted"
                    );
                }
            }

            let tr_prog =
                GlProgram::compile(&renderer.gl, crate::gpu::shader::BUILTIN_FADE).unwrap();
            let tr_handle = tr_prog.handle;
            max_program_id = max_program_id.max(tr_handle.0.get());

            let aux_key = TextureKey {
                path: PathBuf::from(format!("/virtual/aux_{i}.png")),
                filter: TextureFilter::Linear,
                wrap: TextureWrap::ClampToEdge,
                flip_y: false,
                fingerprint: TextureFingerprint {
                    size: 1024,
                    mtime_sec: i as i64,
                    mtime_nsec: 0,
                    content_hash: i as u64,
                },
            };
            let aux_id = pool
                .insert_rgba(&renderer.gl, aux_key, 16, 16, &aux_px)
                .unwrap();
            let aux_handle = pool.texture_info(aux_id).map(|(h, _, _)| h);
            if let Some(h) = aux_handle {
                max_texture_id = max_texture_id.max(h.0.get());
            }

            let mut custom_textures = BTreeMap::new();
            custom_textures.insert("u_aux".to_string(), aux_id);
            let mut custom_uniforms = BTreeMap::new();
            custom_uniforms.insert("u_mix".to_string(), UniformValue::Float(0.2));

            let params = DrawParams {
                width: 16,
                height: 16,
                time_secs: i as f32 * 0.016,
                delta_secs: 0.016,
                progress: 0.5,
                frame_index: i as i32,
                params: [0.0; 4],
                output_index: 0,
                output_name: "switch-test",
                slot0: &slot0,
                slot1: slot1.as_ref(),
                custom_program: current_prog.as_ref(),
                transition_program: Some(&tr_prog),
                in_transition: true,
                custom_textures: &custom_textures,
                custom_uniforms: &custom_uniforms,
                transient_uniforms: &BTreeMap::new(),
            };
            renderer.draw(&params, &pool).unwrap();

            // Promote done: clean up old slot1, transition program, resolve target, and aux tex.
            let s1_handle = slot1
                .as_ref()
                .and_then(|s| s.rgba_tex.as_ref())
                .map(|t| t.handle);
            if let Some(s1) = slot1.take() {
                s1.destroy(&renderer.gl);
            }
            tr_prog.destroy(&renderer.gl);
            pool.release(&renderer.gl, aux_id);
            renderer.remove_output_cache("switch-test");

            if i % 100 == 0 {
                unsafe { renderer.gl.flush() };
                assert!(
                    !unsafe { renderer.gl.is_program(tr_handle) },
                    "transition GL program was not deleted"
                );
                if let Some(h) = s1_handle {
                    assert!(
                        !unsafe { renderer.gl.is_texture(h) },
                        "replaced slot1 GL texture was not deleted"
                    );
                }
                if let Some(h) = aux_handle {
                    assert!(
                        !unsafe { renderer.gl.is_texture(h) },
                        "released aux GL texture was not deleted"
                    );
                }
            }

            if i == 100 {
                unsafe { renderer.gl.finish() };
                warm_private_dirty_kb = read_self_private_dirty_kb();
            }
        }

        unsafe { renderer.gl.finish() };
        assert_eq!(pool.len(), 0);
        assert_eq!(renderer.resolve_target_count(), 0);
        assert!(
            max_program_id < 100,
            "GL program handles grew without reuse across 1,000 switches: max_program_id={max_program_id}"
        );
        assert!(
            max_texture_id < 100,
            "GL texture handles grew without reuse across 1,000 switches: max_texture_id={max_texture_id}"
        );
        if let (Some(warm), Some(end)) = (warm_private_dirty_kb, read_self_private_dirty_kb()) {
            let delta = end.saturating_sub(warm);
            assert!(
                delta < 16 * 1024,
                "excessive Private_Dirty growth across 900 switches: warm={warm} kB, end={end} kB (delta={delta} kB)"
            );
        }

        if let Some(prog) = current_prog.take() {
            prog.destroy(&renderer.gl);
        }
        slot0.destroy(&renderer.gl);
        pool.destroy(&renderer.gl);
        renderer.destroy();
    }

    fn read_self_private_dirty_kb() -> Option<u64> {
        let raw = std::fs::read_to_string("/proc/self/smaps_rollup").ok()?;
        raw.lines().find_map(|line| {
            let rest = line.strip_prefix("Private_Dirty:")?;
            rest.split_whitespace().next()?.parse::<u64>().ok()
        })
    }
}
