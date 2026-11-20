use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::PathBuf;
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TextureKey {
    pub path: PathBuf,
    pub filter: TextureFilter,
    pub wrap: TextureWrap,
    pub flip_y: bool,
}

impl TextureKey {
    pub fn from_spec(spec: &TextureSpec) -> Self {
        let canonical = std::fs::canonicalize(&spec.path).unwrap_or_else(|_| spec.path.clone());
        Self {
            path: canonical,
            filter: spec.filter,
            wrap: spec.wrap,
            flip_y: spec.flip_y,
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
            y_tex: self.y_tex.as_ref().map(|t| t.handle),
            uv_tex: self.uv_tex.as_ref().map(|t| t.handle),
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
        self.clear_nv12(gl);
        if self.rgba_tex.is_none() {
            self.rgba_tex = Some(OwnedTexture::new(gl)?);
        }
        if let Some(tex) = &mut self.rgba_tex {
            tex.upload_rgba(gl, width, height, pixels);
        }
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
        self.clear_rgba(gl);
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
        self.kind = SlotMediaKind::Nv12;
        self.width = width;
        self.height = height;
        self.fit = fit;
        self.colorspace = colorspace;
        Ok(())
    }

    pub fn clear(&mut self, gl: &glow::Context) {
        self.clear_rgba(gl);
        self.clear_nv12(gl);
        self.kind = SlotMediaKind::None;
        self.width = 0;
        self.height = 0;
    }

    fn clear_rgba(&mut self, gl: &glow::Context) {
        if let Some(tex) = self.rgba_tex.take() {
            tex.destroy(gl);
        }
    }

    fn clear_nv12(&mut self, gl: &glow::Context) {
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
}

pub struct Renderer {
    pub gl: glow::Context,
    vao: glow::VertexArray,
    dummy_tex: OwnedTexture,
    passthrough_program: GlProgram,
    resolve_targets: HashMap<String, ResolveTarget>,
    max_texture_units: u32,
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

        Ok(Self {
            gl,
            vao,
            dummy_tex,
            passthrough_program,
            resolve_targets: HashMap::new(),
            max_texture_units,
        })
    }

    pub fn remove_output_cache(&mut self, output_name: &str) {
        if let Some(target) = self.resolve_targets.remove(output_name) {
            target.destroy(&self.gl);
        }
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
            if let Some(value) = params.custom_uniforms.get(&uniform.name) {
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
}
