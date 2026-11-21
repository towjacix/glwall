//! Optional NVIDIA CUDA-to-GLES NV12 transfer. No frame pixels pass through host RAM.
use std::ffi::{c_int, c_uint, c_void};
use std::ptr;

use anyhow::{Context, Result, anyhow, bail};
use glow::HasContext;
use libloading::Library;

use crate::media::gst_cuda::{CudaSourceInFlight, MappedCudaFrame};

type CuResult = c_int;
type Resource = *mut c_void;
type ContextHandle = *mut c_void;

#[repr(C)]
#[derive(Default)]
struct Copy2d {
    src_x: usize,
    src_y: usize,
    src_type: c_uint,
    src_host: *const c_void,
    src_device: u64,
    src_array: *mut c_void,
    src_pitch: usize,
    dst_x: usize,
    dst_y: usize,
    dst_type: c_uint,
    dst_host: *mut c_void,
    dst_device: u64,
    dst_array: *mut c_void,
    dst_pitch: usize,
    width: usize,
    height: usize,
}

struct Api {
    _cuda: Library,
    _gst_cuda: Library,
    _gobject: Library,
    _egl: Library,
    init: unsafe extern "C" fn(c_uint) -> CuResult,
    register: unsafe extern "C" fn(*mut Resource, c_uint, c_uint, c_uint) -> CuResult,
    unregister: unsafe extern "C" fn(Resource) -> CuResult,
    map: unsafe extern "C" fn(c_uint, *mut Resource, *mut c_void) -> CuResult,
    unmap: unsafe extern "C" fn(c_uint, *mut Resource, *mut c_void) -> CuResult,
    array: unsafe extern "C" fn(*mut *mut c_void, Resource, c_uint, c_uint) -> CuResult,
    copy: unsafe extern "C" fn(*const Copy2d) -> CuResult,
    synchronize: unsafe extern "C" fn() -> CuResult,
    push: unsafe extern "C" fn(ContextHandle) -> c_int,
    pop: unsafe extern "C" fn(*mut ContextHandle) -> c_int,
    object_ref: unsafe extern "C" fn(*mut c_void) -> *mut c_void,
    object_unref: unsafe extern "C" fn(*mut c_void),
    current_egl: unsafe extern "C" fn() -> *mut c_void,
}

impl Api {
    fn load() -> Result<Self> {
        unsafe {
            let cuda = Library::new("libcuda.so.1").context("CUDA driver unavailable")?;
            let gst_cuda =
                Library::new("libgstcuda-1.0.so.0").context("GStreamer CUDA unavailable")?;
            let gobject = Library::new("libgobject-2.0.so.0").context("GObject unavailable")?;
            let egl = Library::new("libEGL.so.1").context("EGL unavailable")?;
            let init = *cuda.get(b"cuInit\0")?;
            let register = *cuda.get(b"cuGraphicsGLRegisterImage\0")?;
            let unregister = *cuda.get(b"cuGraphicsUnregisterResource\0")?;
            let map = *cuda.get(b"cuGraphicsMapResources\0")?;
            let unmap = *cuda.get(b"cuGraphicsUnmapResources\0")?;
            let array = *cuda.get(b"cuGraphicsSubResourceGetMappedArray\0")?;
            let copy = *cuda.get(b"cuMemcpy2D_v2\0")?;
            let synchronize = *cuda.get(b"cuCtxSynchronize\0")?;
            let push = *gst_cuda.get(b"gst_cuda_context_push\0")?;
            let pop = *gst_cuda.get(b"gst_cuda_context_pop\0")?;
            let object_ref = *gobject.get(b"g_object_ref\0")?;
            let object_unref = *gobject.get(b"g_object_unref\0")?;
            let current_egl = *egl.get(b"eglGetCurrentContext\0")?;
            let api = Self {
                _cuda: cuda,
                _gst_cuda: gst_cuda,
                _gobject: gobject,
                _egl: egl,
                init,
                register,
                unregister,
                map,
                unmap,
                array,
                copy,
                synchronize,
                push,
                pop,
                object_ref,
                object_unref,
                current_egl,
            };
            check("cuInit", (api.init)(0))?;
            Ok(api)
        }
    }
}

fn check(operation: &str, code: CuResult) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(anyhow!("{operation} failed with CUDA error {code}"))
    }
}

fn combine_results(primary: Result<()>, cleanup: Result<()>) -> Result<()> {
    match (primary, cleanup) {
        (Err(err), Err(cleanup)) => Err(err.context(format!("cleanup also failed: {cleanup:#}"))),
        (Err(err), _) | (_, Err(err)) => Err(err),
        _ => Ok(()),
    }
}

fn finish_mapped_copy(
    copy: Result<()>,
    sync: impl FnOnce() -> Result<()>,
    unmap: impl FnOnce() -> Result<()>,
) -> Result<()> {
    let sync = sync();
    let unmap = unmap();
    match sync {
        Ok(()) => combine_results(copy, unmap),
        Err(sync_err) => {
            let mut detail = format!("{sync_err:#}");
            if let Err(err) = copy {
                detail.push_str(&format!("; copy failed: {err:#}"));
            }
            if let Err(err) = unmap {
                detail.push_str(&format!("; unmap failed: {err:#}"));
            }
            Err(CudaSourceInFlight(anyhow!("{detail}")).into())
        }
    }
}

/// True when textures were created but graphics-resource registration did not
/// complete for both planes. Upload must reinitialize rather than pass a null
/// `CUgraphicsResource` to `cuGraphicsMapResources`.
fn has_unregistered_resources(
    textures: Option<&[glow::Texture; 2]>,
    resources: &[Resource; 2],
) -> bool {
    textures.is_some() && resources.contains(&ptr::null_mut())
}

/// The graphics resources are tied to the current EGL context. The owner must
/// call `destroy` with that context current before destroying the EGL context.
pub struct CudaTextures {
    api: Api,
    egl_context: *mut c_void,
    cuda_context: ContextHandle,
    textures: Option<[glow::Texture; 2]>,
    resources: [Resource; 2],
    width: u32,
    height: u32,
}

impl CudaTextures {
    pub fn new() -> Result<Self> {
        let api = Api::load()?;
        let egl_context = unsafe { (api.current_egl)() };
        if egl_context.is_null() {
            bail!("no current EGL context for CUDA interop");
        }
        Ok(Self {
            api,
            egl_context,
            cuda_context: ptr::null_mut(),
            textures: None,
            resources: [ptr::null_mut(); 2],
            width: 0,
            height: 0,
        })
    }

    fn push(&self) -> Result<()> {
        if self.cuda_context.is_null() || unsafe { (self.api.push)(self.cuda_context) } == 0 {
            bail!("failed to activate GStreamer CUDA context");
        }
        Ok(())
    }

    fn pop(&self) -> Result<()> {
        let mut popped = ptr::null_mut();
        if unsafe { (self.api.pop)(&mut popped) } == 0 {
            bail!("failed to restore CUDA context");
        }
        Ok(())
    }

    fn clear(&mut self, gl: &glow::Context) -> Result<()> {
        if unsafe { (self.api.current_egl)() } != self.egl_context {
            bail!("CUDA textures destroyed with a different EGL context");
        }
        let mut error = None;
        if !self.cuda_context.is_null() {
            self.push()?;
            for resource in &mut self.resources {
                if !resource.is_null() {
                    if let Err(err) = check("cuGraphicsUnregisterResource", unsafe {
                        (self.api.unregister)(*resource)
                    }) {
                        error.get_or_insert(err);
                    } else {
                        *resource = ptr::null_mut();
                    }
                }
            }
            if let Err(err) = self.pop() {
                error.get_or_insert(err);
            }
            if let Some(err) = error {
                return Err(err);
            }
            unsafe {
                (self.api.object_unref)(self.cuda_context);
            }
            self.cuda_context = ptr::null_mut();
        }
        if let Some(textures) = self.textures.take() {
            unsafe {
                for texture in textures {
                    gl.delete_texture(texture);
                }
            }
        }
        self.width = 0;
        self.height = 0;
        Ok(())
    }

    pub fn upload(
        &mut self,
        gl: &glow::Context,
        frame: &MappedCudaFrame,
    ) -> Result<(glow::Texture, glow::Texture)> {
        if unsafe { (self.api.current_egl)() } != self.egl_context {
            bail!("wrong EGL context for CUDA interop");
        }
        if frame.context.is_null()
            || frame.device_ptr == 0
            || frame.width < 2
            || frame.height < 2
            || !frame.width.is_multiple_of(2)
            || !frame.height.is_multiple_of(2)
            || frame.y_stride < frame.width as usize
            || frame.uv_stride < frame.width as usize
        {
            bail!("invalid CUDA NV12 frame layout");
        }
        if self.textures.is_none()
            || self.cuda_context != frame.context
            || self.width != frame.width
            || self.height != frame.height
            || has_unregistered_resources(self.textures.as_ref(), &self.resources)
        {
            self.clear(gl)?;
            self.cuda_context = unsafe { (self.api.object_ref)(frame.context) };
            if self.cuda_context.is_null() {
                bail!("failed to retain CUDA context");
            }
            let y = unsafe { gl.create_texture().map_err(|e| anyhow!("{e}"))? };
            let uv = match unsafe { gl.create_texture() } {
                Ok(texture) => texture,
                Err(err) => {
                    unsafe {
                        gl.delete_texture(y);
                    }
                    return Err(anyhow!("{err}"));
                }
            };
            self.textures = Some([y, uv]);
            self.width = frame.width;
            self.height = frame.height;
            unsafe {
                for (texture, width, height, internal, format) in [
                    (y, frame.width, frame.height, glow::R8, glow::RED),
                    (uv, frame.width / 2, frame.height / 2, glow::RG8, glow::RG),
                ] {
                    gl.bind_texture(glow::TEXTURE_2D, Some(texture));
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
                    gl.tex_image_2d(
                        glow::TEXTURE_2D,
                        0,
                        internal as i32,
                        width as i32,
                        height as i32,
                        0,
                        format,
                        glow::UNSIGNED_BYTE,
                        glow::PixelUnpackData::Slice(None),
                    );
                }
                gl.bind_texture(glow::TEXTURE_2D, None);
            }
            // Roll back on any failure so a retry cannot observe textures that
            // were never registered as CUDA graphics resources.
            let init = (|| {
                self.push()?;
                let mut registration_error = None;
                for (index, texture) in [y, uv].iter().enumerate() {
                    if let Err(err) = check("cuGraphicsGLRegisterImage", unsafe {
                        (self.api.register)(
                            &mut self.resources[index],
                            texture.0.get(),
                            glow::TEXTURE_2D,
                            2,
                        )
                    }) {
                        registration_error = Some(err);
                        break;
                    }
                }
                let registration = registration_error.map_or(Ok(()), Err);
                combine_results(registration, self.pop())
            })();
            if init.is_err() {
                return Err(combine_results(init, self.clear(gl)).unwrap_err());
            }
        }
        self.push()?;
        let result = (|| {
            check("cuGraphicsMapResources", unsafe {
                (self.api.map)(2, self.resources.as_mut_ptr(), ptr::null_mut())
            })?;
            let result = (|| {
                for (index, offset, pitch, height) in [
                    (0, frame.y_offset, frame.y_stride, frame.height),
                    (1, frame.uv_offset, frame.uv_stride, frame.height / 2),
                ] {
                    let mut array = ptr::null_mut();
                    check("cuGraphicsSubResourceGetMappedArray", unsafe {
                        (self.api.array)(&mut array, self.resources[index], 0, 0)
                    })?;
                    let source = frame
                        .device_ptr
                        .checked_add(offset as u64)
                        .ok_or_else(|| anyhow!("CUDA source offset overflow"))?;
                    let copy = Copy2d {
                        src_type: 2,
                        src_device: source,
                        src_pitch: pitch,
                        dst_type: 3,
                        dst_array: array,
                        width: frame.width as usize,
                        height: height as usize,
                        ..Default::default()
                    };
                    check("cuMemcpy2D", unsafe { (self.api.copy)(&copy) })?;
                }
                Ok(())
            })();
            // Device-to-device cuMemcpy2D can return before the source reads
            // complete, including when a later plane copy fails.
            finish_mapped_copy(
                result,
                || check("cuCtxSynchronize", unsafe { (self.api.synchronize)() }),
                || {
                    check("cuGraphicsUnmapResources", unsafe {
                        (self.api.unmap)(2, self.resources.as_mut_ptr(), ptr::null_mut())
                    })
                },
            )
        })();
        let pop = self.pop();
        combine_results(result, pop)?;
        let textures = self.textures.context("CUDA textures missing")?;
        Ok((textures[0], textures[1]))
    }

    pub fn destroy(&mut self, gl: &glow::Context) -> Result<()> {
        self.clear(gl)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_copy_failure_still_synchronizes_before_unmapping() {
        let order = std::cell::RefCell::new(Vec::new());
        let err = finish_mapped_copy(
            Err(anyhow!("second plane failed")),
            || {
                order.borrow_mut().push("sync");
                Ok(())
            },
            || {
                order.borrow_mut().push("unmap");
                Err(anyhow!("unmap failed"))
            },
        )
        .unwrap_err();
        assert_eq!(*order.borrow(), ["sync", "unmap"]);
        let message = format!("{err:#}");
        assert!(message.contains("second plane failed"));
        assert!(message.contains("unmap failed"));
    }

    #[test]
    fn failed_sync_marks_source_for_quarantine_even_if_unmap_fails() {
        let err = finish_mapped_copy(
            Err(anyhow!("copy failed")),
            || Err(anyhow!("sync failed")),
            || Err(anyhow!("unmap failed")),
        )
        .unwrap_err();
        assert!(err.downcast_ref::<CudaSourceInFlight>().is_some());
        let message = format!("{err:#}");
        for expected in ["copy failed", "sync failed", "unmap failed"] {
            assert!(message.contains(expected));
        }
    }

    #[test]
    fn reports_primary_and_cleanup_errors_without_losing_marker() {
        let primary: anyhow::Error = CudaSourceInFlight(anyhow!("sync failed")).into();
        let err = combine_results(Err(primary), Err(anyhow!("pop failed"))).unwrap_err();
        assert!(err.downcast_ref::<CudaSourceInFlight>().is_some());
        let message = format!("{err:#}");
        assert!(message.contains("sync failed"));
        assert!(message.contains("pop failed"));
    }

    #[test]
    fn partially_registered_resources_force_reinitialization() {
        let texture = glow::Texture(std::num::NonZeroU32::new(7).unwrap());
        let textures = Some([texture, texture]);
        let registered = 0xdeadbeefusize as Resource;
        assert!(has_unregistered_resources(
            textures.as_ref(),
            &[registered, ptr::null_mut()]
        ));
        assert!(!has_unregistered_resources(
            textures.as_ref(),
            &[registered, registered]
        ));
        // No textures yet means the first re-init condition already applies.
        assert!(!has_unregistered_resources(
            None,
            &[ptr::null_mut(), ptr::null_mut()]
        ));
    }
}
