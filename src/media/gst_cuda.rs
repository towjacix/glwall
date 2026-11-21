//! Optional GStreamer NVDEC backend. Libraries are loaded at runtime so the
//! regular FFmpeg backend remains available on systems without GStreamer CUDA.
use std::ffi::{CStr, CString, c_char, c_int, c_uint, c_void};
use std::path::Path;
use std::ptr;
use std::sync::{Arc, LazyLock};
use std::time::Duration;

use anyhow::{Result, anyhow, bail};
use libloading::Library;

use crate::config::ColorSpace;

type Ptr = *mut c_void;

#[repr(C)]
struct GError {
    domain: c_uint,
    code: c_int,
    message: *const c_char,
}
#[repr(C)]
struct MiniObject {
    ty: usize,
    refcount: c_int,
    lockstate: c_int,
    flags: c_uint,
    copy: Ptr,
    dispose: Ptr,
    free: Ptr,
    private_uint: c_uint,
    private_ptr: Ptr,
}
#[repr(C)]
struct GstMessageHeader {
    mini: MiniObject,
    ty: c_uint,
}
#[repr(C)]
struct GstMemory {
    mini: MiniObject,
    allocator: Ptr,
    parent: Ptr,
    maxsize: usize,
    align: usize,
    offset: usize,
    size: usize,
}
#[repr(C)]
struct GstCudaMemory {
    memory: GstMemory,
    context: Ptr,
}
#[repr(C)]
struct GstMeta {
    flags: c_uint,
    info: Ptr,
}
#[repr(C)]
struct GstVideoMeta {
    meta: GstMeta,
    buffer: Ptr,
    flags: c_uint,
    format: c_int,
    id: c_int,
    width: c_uint,
    height: c_uint,
    n_planes: c_uint,
    offset: [usize; 4],
    stride: [c_int; 4],
}
#[repr(C)]
struct GstMapInfo {
    memory: Ptr,
    flags: c_uint,
    data: Ptr,
    size: usize,
    maxsize: usize,
    user_data: [Ptr; 4],
    reserved: [Ptr; 4],
}

struct Api {
    _gst: Library,
    _app: Library,
    _video: Library,
    _cuda: Library,
    _glib: Library,
    init: unsafe extern "C" fn(Ptr, Ptr, *mut *mut GError) -> c_int,
    parse: unsafe extern "C" fn(*const c_char, *mut *mut GError) -> Ptr,
    get_by_name: unsafe extern "C" fn(Ptr, *const c_char) -> Ptr,
    set_state: unsafe extern "C" fn(Ptr, c_int) -> c_int,
    get_bus: unsafe extern "C" fn(Ptr) -> Ptr,
    pop_filtered: unsafe extern "C" fn(Ptr, c_uint) -> Ptr,
    parse_error: unsafe extern "C" fn(Ptr, *mut *mut GError, *mut *mut c_char),
    seek: unsafe extern "C" fn(Ptr, c_int, c_uint, i64) -> c_int,
    unref_object: unsafe extern "C" fn(Ptr),
    unref_mini: unsafe extern "C" fn(Ptr),
    pull: unsafe extern "C" fn(Ptr, u64) -> Ptr,
    sample_buffer: unsafe extern "C" fn(Ptr) -> Ptr,
    memory_count: unsafe extern "C" fn(Ptr) -> c_uint,
    peek_memory: unsafe extern "C" fn(Ptr, c_uint) -> Ptr,
    video_meta: unsafe extern "C" fn(Ptr) -> *const GstVideoMeta,
    is_cuda: unsafe extern "C" fn(Ptr) -> c_int,
    cuda_sync: unsafe extern "C" fn(Ptr),
    map: unsafe extern "C" fn(Ptr, *mut GstMapInfo, c_uint) -> c_int,
    unmap: unsafe extern "C" fn(Ptr, *mut GstMapInfo),
    error_free: unsafe extern "C" fn(*mut GError),
    free: unsafe extern "C" fn(Ptr),
}

impl Api {
    fn load() -> Result<Self> {
        unsafe {
            let gst = Library::new("libgstreamer-1.0.so.0")?;
            let app = Library::new("libgstapp-1.0.so.0")?;
            let video = Library::new("libgstvideo-1.0.so.0")?;
            let cuda = Library::new("libgstcuda-1.0.so.0")?;
            let glib = Library::new("libglib-2.0.so.0")?;
            let init = *gst.get(b"gst_init_check\0")?;
            let parse = *gst.get(b"gst_parse_launch\0")?;
            let get_by_name = *gst.get(b"gst_bin_get_by_name\0")?;
            let set_state = *gst.get(b"gst_element_set_state\0")?;
            let get_bus = *gst.get(b"gst_element_get_bus\0")?;
            let pop_filtered = *gst.get(b"gst_bus_pop_filtered\0")?;
            let parse_error = *gst.get(b"gst_message_parse_error\0")?;
            let seek = *gst.get(b"gst_element_seek_simple\0")?;
            let unref_object = *gst.get(b"gst_object_unref\0")?;
            let unref_mini = *gst.get(b"gst_mini_object_unref\0")?;
            let pull = *app.get(b"gst_app_sink_try_pull_sample\0")?;
            let sample_buffer = *gst.get(b"gst_sample_get_buffer\0")?;
            let memory_count = *gst.get(b"gst_buffer_n_memory\0")?;
            let peek_memory = *gst.get(b"gst_buffer_peek_memory\0")?;
            let video_meta = *video.get(b"gst_buffer_get_video_meta\0")?;
            let is_cuda = *cuda.get(b"gst_is_cuda_memory\0")?;
            let cuda_sync = *cuda.get(b"gst_cuda_memory_sync\0")?;
            let map = *gst.get(b"gst_memory_map\0")?;
            let unmap = *gst.get(b"gst_memory_unmap\0")?;
            let error_free = *glib.get(b"g_error_free\0")?;
            let free = *glib.get(b"g_free\0")?;
            let api = Self {
                _gst: gst,
                _app: app,
                _video: video,
                _cuda: cuda,
                _glib: glib,
                init,
                parse,
                get_by_name,
                set_state,
                get_bus,
                pop_filtered,
                parse_error,
                seek,
                unref_object,
                unref_mini,
                pull,
                sample_buffer,
                memory_count,
                peek_memory,
                video_meta,
                is_cuda,
                cuda_sync,
                map,
                unmap,
                error_free,
                free,
            };
            let mut error = ptr::null_mut();
            if (api.init)(ptr::null_mut(), ptr::null_mut(), &mut error) == 0 {
                return Err(anyhow!(
                    api.take_error(error, "GStreamer initialization failed")
                ));
            }
            Ok(api)
        }
    }

    unsafe fn take_error(&self, error: *mut GError, fallback: &str) -> String {
        if error.is_null() {
            return fallback.to_owned();
        }
        let message = unsafe {
            if (*error).message.is_null() {
                fallback.to_owned()
            } else {
                CStr::from_ptr((*error).message)
                    .to_string_lossy()
                    .into_owned()
            }
        };
        unsafe {
            (self.error_free)(error);
        }
        message
    }
}

fn api() -> Result<Arc<Api>> {
    // GStreamer remains initialized for the life of the process. Keep the
    // loaded libraries alive for all plugin callbacks and retained samples.
    static API: LazyLock<std::result::Result<Arc<Api>, String>> =
        LazyLock::new(|| Api::load().map(Arc::new).map_err(|e| format!("{e:#}")));
    API.clone().map_err(|e| anyhow!(e))
}

pub struct MappedCudaFrame {
    pub context: Ptr,
    pub device_ptr: u64,
    pub width: u32,
    pub height: u32,
    pub y_stride: usize,
    pub uv_stride: usize,
    pub y_offset: usize,
    pub uv_offset: usize,
}

pub struct CudaFrame {
    api: Arc<Api>,
    sample: Ptr,
    memory: Ptr,
    pub width: u32,
    pub height: u32,
    pub colorspace: ColorSpace,
    y_stride: usize,
    uv_stride: usize,
    y_offset: usize,
    uv_offset: usize,
}

// GstSample is a refcounted immutable miniobject; the sample keeps its CUDA
// memory and associated GstCudaContext alive while it crosses the mailbox.
unsafe impl Send for CudaFrame {}

impl std::fmt::Debug for CudaFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaFrame")
            .field("width", &self.width)
            .field("height", &self.height)
            .finish_non_exhaustive()
    }
}

impl CudaFrame {
    fn from_sample(api: Arc<Api>, sample: Ptr, colorspace: ColorSpace) -> Result<Self> {
        let result = (|| unsafe {
            let buffer = (api.sample_buffer)(sample);
            if buffer.is_null() || (api.memory_count)(buffer) != 1 {
                bail!("CUDA frame must contain one memory object");
            }
            let memory = (api.peek_memory)(buffer, 0);
            if memory.is_null() || (api.is_cuda)(memory) == 0 {
                bail!("decoder returned non-CUDA memory");
            }
            let meta = (api.video_meta)(buffer);
            if meta.is_null() {
                bail!("CUDA frame has no video layout metadata");
            }
            let meta = &*meta;
            if meta.width < 2
                || meta.height < 2
                || !meta.width.is_multiple_of(2)
                || !meta.height.is_multiple_of(2)
                || meta.n_planes != 2
                || meta.stride[0] < meta.width as i32
                || meta.stride[1] < meta.width as i32
            {
                bail!("unsupported CUDA NV12 frame layout");
            }
            Ok(Self {
                api: Arc::clone(&api),
                sample,
                memory,
                width: meta.width,
                height: meta.height,
                colorspace,
                y_stride: meta.stride[0] as usize,
                uv_stride: meta.stride[1] as usize,
                y_offset: meta.offset[0],
                uv_offset: meta.offset[1],
            })
        })();
        if result.is_err() {
            unsafe {
                (api.unref_mini)(sample);
            }
        }
        result
    }

    pub fn with_mapped<T>(&self, f: impl FnOnce(&MappedCudaFrame) -> Result<T>) -> Result<T> {
        let mut map: GstMapInfo = unsafe { std::mem::zeroed() };
        unsafe {
            (self.api.cuda_sync)(self.memory);
        }
        if unsafe { (self.api.map)(self.memory, &mut map, (1 << 17) | 1) } == 0 {
            bail!("failed to map CUDA video frame");
        }
        struct Guard<'a> {
            api: &'a Api,
            memory: Ptr,
            map: &'a mut GstMapInfo,
        }
        impl Drop for Guard<'_> {
            fn drop(&mut self) {
                unsafe {
                    (self.api.unmap)(self.memory, self.map);
                }
            }
        }
        let guard = Guard {
            api: &self.api,
            memory: self.memory,
            map: &mut map,
        };
        for (offset, stride, rows) in [
            (self.y_offset, self.y_stride, self.height as usize),
            (self.uv_offset, self.uv_stride, self.height as usize / 2),
        ] {
            let end = stride
                .checked_mul(rows - 1)
                .and_then(|n| offset.checked_add(n))
                .and_then(|n| n.checked_add(self.width as usize))
                .ok_or_else(|| anyhow!("CUDA frame layout overflow"))?;
            if end > guard.map.size {
                bail!("CUDA frame plane exceeds mapped memory");
            }
        }
        let context = unsafe { (*(self.memory as *const GstCudaMemory)).context };
        let frame = MappedCudaFrame {
            context,
            device_ptr: guard.map.data as u64,
            width: self.width,
            height: self.height,
            y_stride: self.y_stride,
            uv_stride: self.uv_stride,
            y_offset: self.y_offset,
            uv_offset: self.uv_offset,
        };
        f(&frame)
    }
}

impl Drop for CudaFrame {
    fn drop(&mut self) {
        unsafe {
            (self.api.unref_mini)(self.sample);
        }
    }
}

pub struct CudaDecoder {
    api: Arc<Api>,
    pipeline: Ptr,
    sink: Ptr,
    bus: Ptr,
    colorspace: ColorSpace,
}

impl CudaDecoder {
    pub fn open(path: &Path, colorspace: ColorSpace) -> Result<Self> {
        let api = api()?;
        let path = path
            .to_str()
            .ok_or_else(|| anyhow!("GStreamer requires a UTF-8 path"))?;
        if path.chars().any(char::is_control) {
            bail!("unsupported control character in GStreamer path");
        }
        let escaped = path.replace('\\', "\\\\").replace('"', "\\\"");
        let desc = CString::new(format!(
            "filesrc location=\"{escaped}\" ! decodebin ! video/x-raw(memory:CUDAMemory),format=NV12 ! appsink name=frames max-buffers=2 drop=false sync=false"
        ))?;
        let mut error = ptr::null_mut();
        let pipeline = unsafe { (api.parse)(desc.as_ptr(), &mut error) };
        if !error.is_null() || pipeline.is_null() {
            let message =
                unsafe { api.take_error(error, "could not create CUDA decoder pipeline") };
            if !pipeline.is_null() {
                unsafe {
                    (api.unref_object)(pipeline);
                }
            }
            bail!("{message}");
        }
        let sink = unsafe { (api.get_by_name)(pipeline, c"frames".as_ptr()) };
        let bus = unsafe { (api.get_bus)(pipeline) };
        if sink.is_null() || bus.is_null() {
            unsafe {
                if !sink.is_null() {
                    (api.unref_object)(sink);
                }
                if !bus.is_null() {
                    (api.unref_object)(bus);
                }
                (api.unref_object)(pipeline);
            }
            bail!("CUDA decoder pipeline is missing a sink or bus");
        }
        let decoder = Self {
            api,
            pipeline,
            sink,
            bus,
            colorspace,
        };
        if unsafe { (decoder.api.set_state)(decoder.pipeline, 4) } == 0 {
            bail!("could not start CUDA decoder pipeline");
        }
        Ok(decoder)
    }

    pub fn pull(&mut self, timeout: Duration) -> Result<Option<CudaFrame>> {
        let timeout_ns = timeout.as_nanos().min(u64::MAX as u128) as u64;
        let sample = unsafe { (self.api.pull)(self.sink, timeout_ns) };
        if !sample.is_null() {
            return CudaFrame::from_sample(Arc::clone(&self.api), sample, self.colorspace)
                .map(Some);
        }
        let message = unsafe { (self.api.pop_filtered)(self.bus, 1 | (1 << 1)) };
        if !message.is_null() {
            let kind = unsafe { (*(message as *const GstMessageHeader)).ty };
            if kind == (1 << 1) {
                let mut detail = ptr::null_mut();
                let mut debug = ptr::null_mut();
                unsafe {
                    (self.api.parse_error)(message, &mut detail, &mut debug);
                    (self.api.unref_mini)(message);
                }
                let text = unsafe { self.api.take_error(detail, "CUDA decoder failed") };
                if !debug.is_null() {
                    unsafe {
                        (self.api.free)(debug.cast());
                    }
                }
                bail!("{text}");
            }
            unsafe {
                (self.api.unref_mini)(message);
            }
            // GST_FORMAT_TIME; FLUSH | KEY_UNIT.
            if unsafe { (self.api.seek)(self.pipeline, 3, 1 | 4, 0) } == 0 {
                bail!("CUDA decoder could not loop video");
            }
        }
        Ok(None)
    }
}

impl Drop for CudaDecoder {
    fn drop(&mut self) {
        unsafe {
            (self.api.set_state)(self.pipeline, 1); // GST_STATE_NULL
            (self.api.unref_object)(self.sink);
            (self.api.unref_object)(self.bus);
            (self.api.unref_object)(self.pipeline);
        }
    }
}
