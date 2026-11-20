use std::ffi::c_void;
use std::ptr;

use anyhow::{Context, Result, anyhow};
use glow::HasContext;
use khronos_egl as egl;
use wayland_egl::WlEglSurface;

const EGL_PLATFORM_WAYLAND_KHR: egl::Enum = 0x31D8;
const EGL_PLATFORM_SURFACELESS_MESA: egl::Enum = 0x31DD;

pub type EglInstance = egl::DynamicInstance<egl::EGL1_4>;

pub struct EglState {
    api: EglInstance,
    display: egl::Display,
    config: egl::Config,
    context: egl::Context,
    fallback_pbuffer: Option<egl::Surface>,
}

impl EglState {
    pub fn new_wayland(wl_display_ptr: *mut c_void) -> Result<Self> {
        let lib = unsafe { libloading::Library::new("libEGL.so.1") }
            .context("failed to dynamically load libEGL.so.1")?;
        let api = unsafe { EglInstance::load_required_from(lib) }
            .context("failed to load EGL 1.4 entry points from libEGL.so.1")?;

        let display = if let Some(egl15) = api.upcast::<egl::EGL1_5>() {
            match unsafe {
                egl15.get_platform_display(
                    EGL_PLATFORM_WAYLAND_KHR,
                    wl_display_ptr,
                    &[egl::ATTRIB_NONE],
                )
            } {
                Ok(dpy) => dpy,
                Err(_) => unsafe { api.get_display(wl_display_ptr) }
                    .ok_or_else(|| anyhow!("eglGetDisplay returned NO_DISPLAY for Wayland"))?,
            }
        } else {
            unsafe { api.get_display(wl_display_ptr) }
                .ok_or_else(|| anyhow!("eglGetDisplay returned NO_DISPLAY for Wayland"))?
        };

        api.initialize(display)
            .context("eglInitialize failed on Wayland display")?;
        api.bind_api(egl::OPENGL_ES_API)
            .context("eglBindAPI(OPENGL_ES_API) failed")?;

        let config_attribs = [
            egl::SURFACE_TYPE,
            egl::WINDOW_BIT,
            egl::RENDERABLE_TYPE,
            egl::OPENGL_ES3_BIT,
            egl::RED_SIZE,
            8,
            egl::GREEN_SIZE,
            8,
            egl::BLUE_SIZE,
            8,
            egl::ALPHA_SIZE,
            8,
            egl::NONE,
        ];

        let config = choose_single_config(&api, display, &config_attribs)
            .context("no RGBA8 OpenGL ES 3.0 EGLConfig found for Wayland")?;
        let context = create_gles3_context(&api, display, config)?;

        // Bind surfaceless if supported; ignore failure because Wayland surfaces will bind on first draw.
        let _ = api.make_current(display, None, None, Some(context));

        Ok(Self {
            api,
            display,
            config,
            context,
            fallback_pbuffer: None,
        })
    }

    pub fn new_surfaceless() -> Result<Self> {
        let lib = unsafe { libloading::Library::new("libEGL.so.1") }
            .context("failed to dynamically load libEGL.so.1")?;
        let api = unsafe { EglInstance::load_required_from(lib) }
            .context("failed to load EGL 1.4 entry points from libEGL.so.1")?;

        let display = if let Some(egl15) = api.upcast::<egl::EGL1_5>() {
            match unsafe {
                egl15.get_platform_display(
                    EGL_PLATFORM_SURFACELESS_MESA,
                    ptr::null_mut(),
                    &[egl::ATTRIB_NONE],
                )
            } {
                Ok(dpy) => dpy,
                Err(_) => unsafe { api.get_display(egl::DEFAULT_DISPLAY) }
                    .ok_or_else(|| anyhow!("eglGetDisplay(DEFAULT_DISPLAY) returned NO_DISPLAY"))?,
            }
        } else {
            unsafe { api.get_display(egl::DEFAULT_DISPLAY) }
                .ok_or_else(|| anyhow!("eglGetDisplay(DEFAULT_DISPLAY) returned NO_DISPLAY"))?
        };

        api.initialize(display)
            .context("eglInitialize failed on surfaceless EGL display")?;
        api.bind_api(egl::OPENGL_ES_API)
            .context("eglBindAPI(OPENGL_ES_API) failed")?;

        let surfaceless_attribs = [
            egl::RENDERABLE_TYPE,
            egl::OPENGL_ES3_BIT,
            egl::RED_SIZE,
            8,
            egl::GREEN_SIZE,
            8,
            egl::BLUE_SIZE,
            8,
            egl::NONE,
        ];

        if let Ok(config) = choose_single_config(&api, display, &surfaceless_attribs)
            && let Ok(context) = create_gles3_context(&api, display, config)
        {
            if api.make_current(display, None, None, Some(context)).is_ok() {
                return Ok(Self {
                    api,
                    display,
                    config,
                    context,
                    fallback_pbuffer: None,
                });
            }
            let _ = api.destroy_context(display, context);
        }

        // Fall back to a 1x1 Pbuffer surface when EGL_KHR_surfaceless_context is unavailable.
        let pbuffer_attribs = [
            egl::SURFACE_TYPE,
            egl::PBUFFER_BIT,
            egl::RENDERABLE_TYPE,
            egl::OPENGL_ES3_BIT,
            egl::RED_SIZE,
            8,
            egl::GREEN_SIZE,
            8,
            egl::BLUE_SIZE,
            8,
            egl::NONE,
        ];
        let config = choose_single_config(&api, display, &pbuffer_attribs)
            .context("no Pbuffer OpenGL ES 3.0 EGLConfig found")?;
        let context = create_gles3_context(&api, display, config)?;
        let pbuffer = api
            .create_pbuffer_surface(display, config, &[egl::WIDTH, 1, egl::HEIGHT, 1, egl::NONE])
            .context("failed to create 1x1 EGL pbuffer surface")?;
        api.make_current(display, Some(pbuffer), Some(pbuffer), Some(context))
            .context("failed to make 1x1 EGL pbuffer current")?;

        Ok(Self {
            api,
            display,
            config,
            context,
            fallback_pbuffer: Some(pbuffer),
        })
    }

    pub fn create_glow_context(&self) -> Result<glow::Context> {
        let gl = unsafe {
            glow::Context::from_loader_function(|symbol| {
                self.api
                    .get_proc_address(symbol)
                    .map_or(ptr::null(), |f| f as *const c_void)
            })
        };
        tracing::info!("OpenGL renderer initialized: {}", unsafe {
            gl.get_parameter_string(glow::RENDERER)
        });
        Ok(gl)
    }

    pub fn create_window_surface(&self, wl_egl: &WlEglSurface) -> Result<egl::Surface> {
        let native_win = wl_egl.ptr() as egl::NativeWindowType;
        if let Some(egl15) = self.api.upcast::<egl::EGL1_5>()
            && let Ok(surface) = unsafe {
                egl15.create_platform_window_surface(
                    self.display,
                    self.config,
                    native_win,
                    &[egl::ATTRIB_NONE],
                )
            }
        {
            return Ok(surface);
        }
        unsafe {
            self.api
                .create_window_surface(self.display, self.config, native_win, None)
        }
        .context("eglCreateWindowSurface failed")
    }

    pub fn make_current_surface(
        &self,
        surface: egl::Surface,
        swap_interval_initialized: &mut bool,
    ) -> Result<()> {
        self.api
            .make_current(
                self.display,
                Some(surface),
                Some(surface),
                Some(self.context),
            )
            .context("eglMakeCurrent failed for Wayland surface")?;
        if !*swap_interval_initialized {
            // NVIDIA egl-wayland flickers if swap_interval is reset per frame; configure once per surface.
            self.api
                .swap_interval(self.display, 0)
                .context("eglSwapInterval(0) failed")?;
            *swap_interval_initialized = true;
        }
        Ok(())
    }

    pub fn make_current_surfaceless(&self) {
        let surf = self.fallback_pbuffer;
        let _ = self
            .api
            .make_current(self.display, surf, surf, Some(self.context));
    }

    pub fn unbind_context(&self) {
        let _ = self.api.make_current(self.display, None, None, None);
    }

    pub fn swap_buffers(&self, surface: egl::Surface) -> Result<()> {
        self.api
            .swap_buffers(self.display, surface)
            .context("eglSwapBuffers failed")
    }

    pub fn destroy_surface(&self, surface: egl::Surface) {
        self.unbind_context();
        let _ = self.api.destroy_surface(self.display, surface);
    }
}

impl Drop for EglState {
    fn drop(&mut self) {
        let _ = self.api.make_current(self.display, None, None, None);
        if let Some(pbuffer) = self.fallback_pbuffer.take() {
            let _ = self.api.destroy_surface(self.display, pbuffer);
        }
        let _ = self.api.destroy_context(self.display, self.context);
        let _ = self.api.terminate(self.display);
    }
}

fn choose_single_config(
    api: &EglInstance,
    display: egl::Display,
    attribs: &[egl::Int],
) -> Result<egl::Config> {
    let mut configs = Vec::with_capacity(8);
    api.choose_config(display, attribs, &mut configs)
        .context("eglChooseConfig failed")?;
    configs
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("eglChooseConfig returned 0 matching configurations"))
}

fn create_gles3_context(
    api: &EglInstance,
    display: egl::Display,
    config: egl::Config,
) -> Result<egl::Context> {
    let ctx_attribs = [
        egl::CONTEXT_MAJOR_VERSION,
        3,
        egl::CONTEXT_MINOR_VERSION,
        0,
        egl::NONE,
    ];
    api.create_context(display, config, None, &ctx_attribs)
        .context("eglCreateContext(GLES 3.0) failed")
}
