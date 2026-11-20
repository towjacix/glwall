use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result};
use smithay_client_toolkit::compositor::{
    CompositorHandler, CompositorState, FrameCallbackData, Region,
};
use smithay_client_toolkit::dispatch2::Dispatch2;
use smithay_client_toolkit::output::{OutputHandler, OutputState};
use smithay_client_toolkit::registry::{ProvidesRegistryState, RegistryState};
use smithay_client_toolkit::shell::WaylandSurface;
use smithay_client_toolkit::shell::wlr_layer::{
    Anchor, KeyboardInteractivity, Layer, LayerShell, LayerShellHandler, LayerSurface,
    LayerSurfaceConfigure,
};
use smithay_client_toolkit::{delegate_dispatch2, delegate_registry, registry_handlers};
use wayland_client::globals::registry_queue_init;
use wayland_client::protocol::{wl_output, wl_surface};
use wayland_client::{Connection, EventQueue, Proxy, QueueHandle, event_created_child};
use wayland_egl::WlEglSurface;
use wayland_protocols::wp::fractional_scale::v1::client::{
    wp_fractional_scale_manager_v1::WpFractionalScaleManagerV1,
    wp_fractional_scale_v1::{self, WpFractionalScaleV1},
};
use wayland_protocols::wp::viewporter::client::{
    wp_viewport::{self, WpViewport},
    wp_viewporter::{self, WpViewporter},
};
use wayland_protocols_wlr::foreign_toplevel::v1::client::{
    zwlr_foreign_toplevel_handle_v1::{self, ZwlrForeignToplevelHandleV1},
    zwlr_foreign_toplevel_manager_v1::{self, ZwlrForeignToplevelManagerV1},
};

use crate::gpu::egl::EglState;

pub struct WaylandHandles<'a, S: WaylandAppSink> {
    pub output_state: &'a OutputState,
    pub compositor_state: &'a CompositorState,
    pub layer_shell: &'a LayerShell,
    pub viewporter: Option<&'a WpViewporter>,
    pub fractional_scale_manager: Option<&'a WpFractionalScaleManagerV1>,
    pub qh: &'a QueueHandle<WaylandContext<S>>,
}

pub trait WaylandAppSink: Sized + 'static {
    fn on_output_added(&mut self, handles: &WaylandHandles<'_, Self>, output: wl_output::WlOutput);
    fn on_output_updated(
        &mut self,
        handles: &WaylandHandles<'_, Self>,
        output: &wl_output::WlOutput,
    );
    fn on_output_removed(
        &mut self,
        handles: &WaylandHandles<'_, Self>,
        output: &wl_output::WlOutput,
    );
    fn on_surface_configure(
        &mut self,
        handles: &WaylandHandles<'_, Self>,
        surface: &wl_surface::WlSurface,
        logical_w: u32,
        logical_h: u32,
    );
    fn on_surface_scale_120(&mut self, surface: &wl_surface::WlSurface, scale_120: u32);
    fn on_surface_closed(&mut self, surface: &wl_surface::WlSurface);
    fn on_surface_frame(&mut self, surface: &wl_surface::WlSurface, time_ms: u32);
    fn on_fullscreen_outputs_changed(&mut self, fullscreen_outputs: HashSet<String>);
}

#[derive(Debug, Clone)]
struct TrackedToplevel {
    current_fullscreen: bool,
    pending_fullscreen: bool,
    current_outputs: Vec<wl_output::WlOutput>,
    pending_outputs: Vec<wl_output::WlOutput>,
}

impl TrackedToplevel {
    fn new() -> Self {
        Self {
            current_fullscreen: false,
            pending_fullscreen: false,
            current_outputs: Vec::new(),
            pending_outputs: Vec::new(),
        }
    }
}

pub struct ViewporterGlobal;
pub struct ViewportSurfaceData;
pub struct FractionalScaleGlobal;
pub struct FractionalScaleData(pub wl_surface::WlSurface);
pub struct ToplevelManagerData;
pub struct ToplevelHandleData;

pub struct WaylandContext<S: WaylandAppSink> {
    pub registry_state: RegistryState,
    pub output_state: OutputState,
    pub compositor_state: CompositorState,
    pub layer_shell: LayerShell,
    pub viewporter: Option<WpViewporter>,
    pub fractional_scale_manager: Option<WpFractionalScaleManagerV1>,
    pub _toplevel_manager: Option<ZwlrForeignToplevelManagerV1>,
    toplevels: HashMap<wayland_backend::client::ObjectId, TrackedToplevel>,
    pub qh: QueueHandle<Self>,
    pub sink: S,
}

impl<S: WaylandAppSink> WaylandContext<S> {
    pub fn connect<F>(make_sink: F) -> Result<(Connection, EventQueue<Self>, Self)>
    where
        F: FnOnce(&Connection) -> Result<S>,
    {
        let conn = Connection::connect_to_env()
            .context("failed to connect to Wayland display (is WAYLAND_DISPLAY set?)")?;
        let sink = make_sink(&conn)?;
        let (globals, event_queue) =
            registry_queue_init::<Self>(&conn).context("failed to initialize Wayland registry")?;
        let qh = event_queue.handle();

        let registry_state = RegistryState::new(&globals);
        let output_state = OutputState::new(&globals, &qh);
        let compositor_state =
            CompositorState::bind(&globals, &qh).context("compositor missing wl_compositor")?;
        let layer_shell = LayerShell::bind(&globals, &qh).context(
            "compositor missing zwlr_layer_shell_v1 (required for desktop background surfaces)",
        )?;

        let viewporter = globals
            .bind::<WpViewporter, _, _>(&qh, 1..=1, ViewporterGlobal)
            .ok();
        let fractional_scale_manager = globals
            .bind::<WpFractionalScaleManagerV1, _, _>(&qh, 1..=1, FractionalScaleGlobal)
            .ok();
        let toplevel_manager = globals
            .bind::<ZwlrForeignToplevelManagerV1, _, _>(&qh, 2..=3, ToplevelManagerData)
            .ok();

        let ctx = Self {
            registry_state,
            output_state,
            compositor_state,
            layer_shell,
            viewporter,
            fractional_scale_manager,
            _toplevel_manager: toplevel_manager,
            toplevels: HashMap::new(),
            qh,
            sink,
        };

        Ok((conn, event_queue, ctx))
    }

    fn recompute_fullscreen(&mut self) {
        let mut fullscreen_outputs = HashSet::new();
        for tl in self.toplevels.values() {
            if !tl.current_fullscreen {
                continue;
            }
            for out in &tl.current_outputs {
                if let Some(name) = self.output_state.info(out).and_then(|i| i.name) {
                    fullscreen_outputs.insert(name);
                }
            }
        }
        self.sink.on_fullscreen_outputs_changed(fullscreen_outputs);
    }
}

pub struct SurfaceHandle {
    pub layer: LayerSurface,
    pub viewport: Option<WpViewport>,
    pub fractional_scale: Option<WpFractionalScaleV1>,
    pub wl_egl: Option<WlEglSurface>,
    pub egl_surface: Option<khronos_egl::Surface>,
    pub swap_interval_set: bool,
    pub logical_w: u32,
    pub logical_h: u32,
    pub scale_120: u32,
    pub phys_w: u32,
    pub phys_h: u32,
    pub configured: bool,
    pub frame_pending: bool,
    pub dirty: bool,
}

impl SurfaceHandle {
    pub fn create<S: WaylandAppSink>(
        handles: &WaylandHandles<'_, S>,
        output: &wl_output::WlOutput,
        initial_logical: (u32, u32),
        initial_scale: i32,
    ) -> Self {
        let wl_surface = handles.compositor_state.create_surface(handles.qh);

        if let Ok(empty_region) = Region::new(handles.compositor_state) {
            wl_surface.set_input_region(Some(empty_region.wl_region()));
        }

        let viewport = handles
            .viewporter
            .map(|vp| vp.get_viewport(&wl_surface, handles.qh, ViewportSurfaceData));

        let fractional_scale = handles.fractional_scale_manager.map(|fsm| {
            fsm.get_fractional_scale(
                &wl_surface,
                handles.qh,
                FractionalScaleData(wl_surface.clone()),
            )
        });

        let layer = handles.layer_shell.create_layer_surface(
            handles.qh,
            wl_surface,
            Layer::Background,
            Some("glwall:bg"),
            Some(output),
        );
        layer.set_anchor(Anchor::TOP | Anchor::BOTTOM | Anchor::LEFT | Anchor::RIGHT);
        layer.set_exclusive_zone(-1);
        layer.set_keyboard_interactivity(KeyboardInteractivity::None);
        layer.commit();

        let scale_120 = (initial_scale.max(1) as u32) * 120;
        let logical_w = initial_logical.0.max(1);
        let logical_h = initial_logical.1.max(1);
        let (phys_w, phys_h) = compute_physical_size(logical_w, logical_h, scale_120);

        Self {
            layer,
            viewport,
            fractional_scale,
            wl_egl: None,
            egl_surface: None,
            swap_interval_set: false,
            logical_w,
            logical_h,
            scale_120,
            phys_w,
            phys_h,
            configured: false,
            frame_pending: false,
            dirty: true,
        }
    }

    pub fn ensure_egl(&mut self, egl: &EglState) -> Result<()> {
        if self.wl_egl.is_none() {
            let wl_egl = WlEglSurface::new(
                self.layer.wl_surface().id(),
                self.phys_w as i32,
                self.phys_h as i32,
            )
            .context("WlEglSurface::new failed")?;
            let egl_surface = egl.create_window_surface(&wl_egl)?;
            self.wl_egl = Some(wl_egl);
            self.egl_surface = Some(egl_surface);
        }
        Ok(())
    }

    pub fn update_geometry(&mut self, logical_w: u32, logical_h: u32, scale_120: u32) {
        self.logical_w = logical_w.max(1);
        self.logical_h = logical_h.max(1);
        self.scale_120 = scale_120.max(30);
        let (new_pw, new_ph) =
            compute_physical_size(self.logical_w, self.logical_h, self.scale_120);
        let size_changed = new_pw != self.phys_w || new_ph != self.phys_h;
        self.phys_w = new_pw;
        self.phys_h = new_ph;

        if let Some(vp) = &self.viewport {
            vp.set_destination(self.logical_w as i32, self.logical_h as i32);
        }
        if size_changed && let Some(wl_egl) = &self.wl_egl {
            wl_egl.resize(self.phys_w as i32, self.phys_h as i32, 0, 0);
        }
        self.dirty = true;
    }

    pub fn request_frame_if_idle<S: WaylandAppSink>(
        &mut self,
        qh: &QueueHandle<WaylandContext<S>>,
    ) {
        if !self.frame_pending {
            let wl_surf = self.layer.wl_surface();
            wl_surf.frame(qh, FrameCallbackData(wl_surf.clone()));
            self.frame_pending = true;
        }
    }

    /// Strict NVIDIA-safe teardown order:
    /// unbind EGLContext -> destroy EGLSurface -> drop WlEglSurface -> destroy viewport/fractional_scale -> drop LayerSurface.
    pub fn teardown(mut self, egl: &EglState) {
        if let Some(egl_surf) = self.egl_surface.take() {
            egl.destroy_surface(egl_surf);
        }
        self.wl_egl.take();
        if let Some(fs) = self.fractional_scale.take() {
            fs.destroy();
        }
        if let Some(vp) = self.viewport.take() {
            vp.destroy();
        }
        drop(self.layer);
    }
}

pub fn compute_physical_size(logical_w: u32, logical_h: u32, scale_120: u32) -> (u32, u32) {
    let pw = ((u64::from(logical_w.max(1)) * u64::from(scale_120.max(30)) + 60) / 120) as u32;
    let ph = ((u64::from(logical_h.max(1)) * u64::from(scale_120.max(30)) + 60) / 120) as u32;
    (pw.max(1), ph.max(1))
}

impl<S: WaylandAppSink> ProvidesRegistryState for WaylandContext<S> {
    fn registry(&mut self) -> &mut RegistryState {
        &mut self.registry_state
    }
    registry_handlers!(OutputState);
}

impl<S: WaylandAppSink> CompositorHandler for WaylandContext<S> {
    fn scale_factor_changed(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        surface: &wl_surface::WlSurface,
        new_factor: i32,
    ) {
        if self.fractional_scale_manager.is_none() {
            self.sink
                .on_surface_scale_120(surface, (new_factor.max(1) as u32) * 120);
        }
    }

    fn transform_changed(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _new_transform: wl_output::Transform,
    ) {
    }

    fn frame(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        surface: &wl_surface::WlSurface,
        time: u32,
    ) {
        self.sink.on_surface_frame(surface, time);
    }

    fn surface_enter(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _output: &wl_output::WlOutput,
    ) {
    }

    fn surface_leave(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _output: &wl_output::WlOutput,
    ) {
    }
}

impl<S: WaylandAppSink> OutputHandler for WaylandContext<S> {
    fn output_state(&mut self) -> &mut OutputState {
        &mut self.output_state
    }

    fn new_output(
        &mut self,
        _conn: &Connection,
        qh: &QueueHandle<Self>,
        output: wl_output::WlOutput,
    ) {
        let handles = WaylandHandles {
            output_state: &self.output_state,
            compositor_state: &self.compositor_state,
            layer_shell: &self.layer_shell,
            viewporter: self.viewporter.as_ref(),
            fractional_scale_manager: self.fractional_scale_manager.as_ref(),
            qh,
        };
        self.sink.on_output_added(&handles, output);
    }

    fn update_output(
        &mut self,
        _conn: &Connection,
        qh: &QueueHandle<Self>,
        output: wl_output::WlOutput,
    ) {
        let handles = WaylandHandles {
            output_state: &self.output_state,
            compositor_state: &self.compositor_state,
            layer_shell: &self.layer_shell,
            viewporter: self.viewporter.as_ref(),
            fractional_scale_manager: self.fractional_scale_manager.as_ref(),
            qh,
        };
        self.sink.on_output_updated(&handles, &output);
    }

    fn output_destroyed(
        &mut self,
        _conn: &Connection,
        qh: &QueueHandle<Self>,
        output: wl_output::WlOutput,
    ) {
        let handles = WaylandHandles {
            output_state: &self.output_state,
            compositor_state: &self.compositor_state,
            layer_shell: &self.layer_shell,
            viewporter: self.viewporter.as_ref(),
            fractional_scale_manager: self.fractional_scale_manager.as_ref(),
            qh,
        };
        self.sink.on_output_removed(&handles, &output);
    }
}

impl<S: WaylandAppSink> LayerShellHandler for WaylandContext<S> {
    fn closed(&mut self, _conn: &Connection, _qh: &QueueHandle<Self>, layer: &LayerSurface) {
        self.sink.on_surface_closed(layer.wl_surface());
    }

    fn configure(
        &mut self,
        _conn: &Connection,
        qh: &QueueHandle<Self>,
        layer: &LayerSurface,
        configure: LayerSurfaceConfigure,
        _serial: u32,
    ) {
        let handles = WaylandHandles {
            output_state: &self.output_state,
            compositor_state: &self.compositor_state,
            layer_shell: &self.layer_shell,
            viewporter: self.viewporter.as_ref(),
            fractional_scale_manager: self.fractional_scale_manager.as_ref(),
            qh,
        };
        self.sink.on_surface_configure(
            &handles,
            layer.wl_surface(),
            configure.new_size.0,
            configure.new_size.1,
        );
    }
}

impl<S: WaylandAppSink> Dispatch2<WpViewporter, WaylandContext<S>> for ViewporterGlobal {
    fn event(
        &self,
        _: &mut WaylandContext<S>,
        _: &WpViewporter,
        _: wp_viewporter::Event,
        _: &Connection,
        _: &QueueHandle<WaylandContext<S>>,
    ) {
        unreachable!("wp_viewporter has no events")
    }
}

impl<S: WaylandAppSink> Dispatch2<WpViewport, WaylandContext<S>> for ViewportSurfaceData {
    fn event(
        &self,
        _: &mut WaylandContext<S>,
        _: &WpViewport,
        _: wp_viewport::Event,
        _: &Connection,
        _: &QueueHandle<WaylandContext<S>>,
    ) {
        unreachable!("wp_viewport has no events")
    }
}

impl<S: WaylandAppSink> Dispatch2<WpFractionalScaleManagerV1, WaylandContext<S>>
    for FractionalScaleGlobal
{
    fn event(
        &self,
        _: &mut WaylandContext<S>,
        _: &WpFractionalScaleManagerV1,
        _: wayland_protocols::wp::fractional_scale::v1::client::wp_fractional_scale_manager_v1::Event,
        _: &Connection,
        _: &QueueHandle<WaylandContext<S>>,
    ) {
        unreachable!("wp_fractional_scale_manager_v1 has no events")
    }
}

impl<S: WaylandAppSink> Dispatch2<WpFractionalScaleV1, WaylandContext<S>> for FractionalScaleData {
    fn event(
        &self,
        state: &mut WaylandContext<S>,
        _: &WpFractionalScaleV1,
        event: wp_fractional_scale_v1::Event,
        _: &Connection,
        _: &QueueHandle<WaylandContext<S>>,
    ) {
        if let wp_fractional_scale_v1::Event::PreferredScale { scale } = event {
            state.sink.on_surface_scale_120(&self.0, scale.max(30));
        }
    }
}

impl<S: WaylandAppSink> Dispatch2<ZwlrForeignToplevelManagerV1, WaylandContext<S>>
    for ToplevelManagerData
{
    fn event(
        &self,
        state: &mut WaylandContext<S>,
        _manager: &ZwlrForeignToplevelManagerV1,
        event: zwlr_foreign_toplevel_manager_v1::Event,
        _: &Connection,
        _: &QueueHandle<WaylandContext<S>>,
    ) {
        match event {
            zwlr_foreign_toplevel_manager_v1::Event::Toplevel { toplevel } => {
                state
                    .toplevels
                    .insert(toplevel.id(), TrackedToplevel::new());
            }
            zwlr_foreign_toplevel_manager_v1::Event::Finished => {
                // Server destroys the manager immediately upon emitting finished.
            }
            _ => {}
        }
    }

    event_created_child!(
        WaylandContext<S>,
        ZwlrForeignToplevelManagerV1,
        [
            zwlr_foreign_toplevel_manager_v1::EVT_TOPLEVEL_OPCODE =>
                (ZwlrForeignToplevelHandleV1, ToplevelHandleData)
        ]
    );
}

impl<S: WaylandAppSink> Dispatch2<ZwlrForeignToplevelHandleV1, WaylandContext<S>>
    for ToplevelHandleData
{
    fn event(
        &self,
        state: &mut WaylandContext<S>,
        handle: &ZwlrForeignToplevelHandleV1,
        event: zwlr_foreign_toplevel_handle_v1::Event,
        _: &Connection,
        _: &QueueHandle<WaylandContext<S>>,
    ) {
        let id = handle.id();
        match event {
            zwlr_foreign_toplevel_handle_v1::Event::OutputEnter { output } => {
                if let Some(tl) = state.toplevels.get_mut(&id)
                    && !tl.pending_outputs.contains(&output)
                {
                    tl.pending_outputs.push(output);
                }
            }
            zwlr_foreign_toplevel_handle_v1::Event::OutputLeave { output } => {
                if let Some(tl) = state.toplevels.get_mut(&id) {
                    tl.pending_outputs.retain(|o| o != &output);
                }
            }
            zwlr_foreign_toplevel_handle_v1::Event::State { state: raw_states } => {
                let fullscreen = raw_states
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|chunk| u32::from_ne_bytes(*chunk))
                    .any(|v| v == zwlr_foreign_toplevel_handle_v1::State::Fullscreen as u32);
                if let Some(tl) = state.toplevels.get_mut(&id) {
                    tl.pending_fullscreen = fullscreen;
                }
            }
            zwlr_foreign_toplevel_handle_v1::Event::Done => {
                if let Some(tl) = state.toplevels.get_mut(&id) {
                    tl.current_fullscreen = tl.pending_fullscreen;
                    tl.current_outputs = tl.pending_outputs.clone();
                }
                state.recompute_fullscreen();
            }
            zwlr_foreign_toplevel_handle_v1::Event::Closed => {
                state.toplevels.remove(&id);
                handle.destroy();
                state.recompute_fullscreen();
            }
            _ => {}
        }
    }
}

delegate_registry!(@<S: WaylandAppSink> WaylandContext<S>);
delegate_dispatch2!(@<S: WaylandAppSink> WaylandContext<S>);
