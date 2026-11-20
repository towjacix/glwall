use std::collections::HashMap;
use std::slice;

use anyhow::{Result, bail};
use glow::HasContext;

use crate::config::{ColorSpace, FitMode};
use crate::gpu::shader::{
    AssembledShader, BUILTIN_PASSTHROUGH, ShaderMode, VERTEX_SHADER_SRC, assemble_fragment_shader,
};

pub struct GlProgram {
    pub handle: glow::Program,
    pub assembled: AssembledShader,
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
        Ok(Self {
            handle: program,
            assembled,
            locs,
        })
    }

    pub fn destroy(self, gl: &glow::Context) {
        unsafe { gl.delete_program(self.handle) };
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
        unsafe {
            gl.bind_texture(glow::TEXTURE_2D, Some(handle));
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MIN_FILTER,
                glow::LINEAR as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MAG_FILTER,
                glow::LINEAR as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_WRAP_S,
                glow::CLAMP_TO_EDGE as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_WRAP_T,
                glow::CLAMP_TO_EDGE as i32,
            );
            gl.bind_texture(glow::TEXTURE_2D, None);
        }
        Ok(Self {
            handle,
            width: 0,
            height: 0,
        })
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
        unsafe { gl.delete_texture(self.handle) };
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
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
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
}

pub struct Renderer {
    pub gl: glow::Context,
    vao: glow::VertexArray,
    dummy_tex: OwnedTexture,
    passthrough_program: GlProgram,
    resolve_targets: HashMap<String, ResolveTarget>,
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

        Ok(Self {
            gl,
            vao,
            dummy_tex,
            passthrough_program,
            resolve_targets: HashMap::new(),
        })
    }

    pub fn remove_output_cache(&mut self, output_name: &str) {
        if let Some(target) = self.resolve_targets.remove(output_name) {
            target.destroy(&self.gl);
        }
    }

    pub fn draw(&mut self, params: &DrawParams<'_>) -> Result<()> {
        let program = if params.in_transition {
            params
                .transition_program
                .unwrap_or(&self.passthrough_program)
        } else {
            params.custom_program.unwrap_or(&self.passthrough_program)
        };

        let mut ichannel0_tex: Option<glow::Texture> = None;
        if program.assembled.mode == ShaderMode::Shadertoy
            && program.assembled.uses_ichannel0
            && params.slot0.kind != SlotMediaKind::None
        {
            let (fbo, color_tex) = {
                let target = match self.resolve_targets.get_mut(params.output_name) {
                    Some(t) => {
                        t.ensure_size(&self.gl, params.width.max(1), params.height.max(1));
                        t
                    }
                    None => {
                        let created = ResolveTarget::new(
                            &self.gl,
                            params.width.max(1),
                            params.height.max(1),
                        )?;
                        self.resolve_targets
                            .entry(params.output_name.to_string())
                            .or_insert(created)
                    }
                };
                (target.fbo, target.color_tex.handle)
            };

            unsafe {
                self.gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo));
                self.gl.viewport(
                    0,
                    0,
                    params.width.max(1) as i32,
                    params.height.max(1) as i32,
                );
                self.gl.use_program(Some(self.passthrough_program.handle));
                self.bind_common_uniforms(&self.passthrough_program, params, None);
                self.gl.bind_vertex_array(Some(self.vao));
                self.gl.draw_arrays(glow::TRIANGLES, 0, 3);
                self.gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            }
            ichannel0_tex = Some(color_tex);
        }

        unsafe {
            self.gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            self.gl.viewport(
                0,
                0,
                params.width.max(1) as i32,
                params.height.max(1) as i32,
            );
            self.gl.clear_color(0.0, 0.0, 0.0, 1.0);
            self.gl.clear(glow::COLOR_BUFFER_BIT);
            self.gl.use_program(Some(program.handle));
            self.bind_common_uniforms(program, params, ichannel0_tex);
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

            // Slot 0 -> texture units 0, 1, 2
            self.bind_slot(
                params.slot0,
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
            if let Some(slot1) = params.slot1 {
                self.bind_slot(
                    slot1,
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

    unsafe fn bind_slot(&self, slot: &MediaSlotGpu, units: [u32; 3], locs: SlotUniformLocs<'_>) {
        let gl = &self.gl;
        let dummy = self.dummy_tex.handle;
        unsafe {
            gl.active_texture(glow::TEXTURE0 + units[0]);
            gl.bind_texture(
                glow::TEXTURE_2D,
                Some(slot.rgba_tex.as_ref().map_or(dummy, |t| t.handle)),
            );
            gl.uniform_1_i32(locs.0, units[0] as i32);

            gl.active_texture(glow::TEXTURE0 + units[1]);
            gl.bind_texture(
                glow::TEXTURE_2D,
                Some(slot.y_tex.as_ref().map_or(dummy, |t| t.handle)),
            );
            gl.uniform_1_i32(locs.1, units[1] as i32);

            gl.active_texture(glow::TEXTURE0 + units[2]);
            gl.bind_texture(
                glow::TEXTURE_2D,
                Some(slot.uv_tex.as_ref().map_or(dummy, |t| t.handle)),
            );
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
