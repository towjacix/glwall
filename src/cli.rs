use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum FitArg {
    Cover,
    Contain,
    Stretch,
    Center,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum HwAccelArg {
    Auto,
    Cuda,
    Vaapi,
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum FilterArg {
    Linear,
    Nearest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum WrapArg {
    #[value(alias = "clamp_to_edge", alias = "clamp")]
    ClampToEdge,
    Repeat,
    #[value(alias = "mirrored_repeat", alias = "mirror")]
    MirroredRepeat,
}

#[derive(Debug, Parser)]
#[command(
    name = "glwall",
    version,
    about = "Wayland OpenGL ES 3.0 / EGL live & custom-shader wallpaper daemon"
)]
pub struct Cli {
    /// Unix socket path for daemon/client IPC (defaults to $XDG_RUNTIME_DIR/glwall.sock)
    #[arg(long, global = true, value_name = "PATH")]
    pub socket: Option<PathBuf>,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Run the Wayland background wallpaper daemon
    Daemon {
        /// Config TOML path (defaults to ~/.config/glwall/config.toml)
        #[arg(long, value_name = "PATH")]
        config: Option<PathBuf>,

        /// Do not restore persisted per-output state on startup
        #[arg(long)]
        no_restore: bool,

        /// Do not write runtime state changes to state.json
        #[arg(long)]
        no_persist: bool,
    },

    /// Set an image, animation, video, or procedural shader (.frag/.glsl)
    Set {
        /// Media file path or procedural .frag shader path
        target: String,

        /// Output name (e.g. DP-1, eDP-1) or '*' for all outputs
        #[arg(long, default_value = "*")]
        output: String,

        /// Fit mode inside output bounds
        #[arg(long, value_enum)]
        fit: Option<FitArg>,

        /// Transition shader preset (fade|dissolve|pixelate|wipe|none) or .frag path
        #[arg(long)]
        transition: Option<String>,

        /// Transition duration in milliseconds
        #[arg(long)]
        duration_ms: Option<u32>,

        /// Post-process .frag shader path or 'none'
        #[arg(long)]
        shader: Option<String>,

        /// Target FPS cap for videos/animated media/shaders
        #[arg(long)]
        fps: Option<f32>,

        /// FFmpeg hardware decode mode for video inputs
        #[arg(long, value_enum)]
        hwaccel: Option<HwAccelArg>,
    },

    /// Attach or clear a live post-processing / procedural .frag shader
    Shader {
        /// Path to .frag shader file or 'none' to clear
        path: String,

        /// Output name or '*' for all outputs
        #[arg(long, default_value = "*")]
        output: String,
    },

    /// Set custom vec4 u_params uniform (values default to 0.0)
    Params {
        /// u_params.x
        x: f32,
        /// u_params.y
        #[arg(default_value_t = 0.0)]
        y: f32,
        /// u_params.z
        #[arg(default_value_t = 0.0)]
        z: f32,
        /// u_params.w
        #[arg(default_value_t = 0.0)]
        w: f32,

        /// Output name or '*' for all outputs
        #[arg(long, default_value = "*")]
        output: String,
    },

    /// Set one or more typed shader uniforms (NAME=VALUE)
    Uniform {
        /// Assignments in NAME=VALUE format (e.g. u_pointer=0.5,0.2 u_strength=0.05)
        #[arg(required = true)]
        assignments: Vec<String>,

        /// Output name or '*' for all outputs
        #[arg(long, default_value = "*")]
        output: String,
    },

    /// Bind or unbind a named auxiliary sampler2D texture
    Texture {
        /// Uniform sampler2D name in the shader (e.g. u_depth)
        name: String,

        /// Path to image file (omit when using --unset)
        path: Option<String>,

        /// Unbind this texture uniform
        #[arg(long)]
        unset: bool,

        /// Output name or '*' for all outputs
        #[arg(long, default_value = "*")]
        output: String,

        /// Texture filtering mode
        #[arg(long, value_enum, default_value_t = FilterArg::Linear)]
        filter: FilterArg,

        /// Texture wrap mode
        #[arg(long, value_enum, default_value_t = WrapArg::ClampToEdge)]
        wrap: WrapArg,

        /// Vertically flip texture rows on upload
        #[arg(long)]
        flip_y: bool,
    },

    /// Atomically apply media, shader, auxiliary textures, and uniforms
    Apply {
        /// Output name or '*' for all outputs
        #[arg(long, default_value = "*")]
        output: String,

        /// Wallpaper media file or standalone procedural shader path
        #[arg(long)]
        media: Option<String>,

        /// Fit mode inside output bounds
        #[arg(long, value_enum)]
        fit: Option<FitArg>,

        /// Post-process .frag shader path or 'none'
        #[arg(long)]
        shader: Option<String>,

        /// Auxiliary texture binding in NAME=PATH[:linear|nearest][:clamp|repeat|mirror][:flip_y] or NAME=none
        #[arg(long = "texture", value_name = "SPEC")]
        textures: Vec<String>,

        /// Typed uniform assignment in NAME=VALUE format
        #[arg(long = "uniform", value_name = "NAME=VALUE")]
        uniforms: Vec<String>,

        /// Transition preset (fade|dissolve|pixelate|wipe|none) or .frag path
        #[arg(long)]
        transition: Option<String>,

        /// Transition duration in milliseconds
        #[arg(long)]
        duration_ms: Option<u32>,

        /// Explicit monotonic revision number for caller synchronization
        #[arg(long)]
        revision: Option<u64>,

        /// Wait until the daemon commits the updated frame to Wayland
        #[arg(long)]
        wait: bool,
    },

    /// Stream live daemon NDJSON events (frame_committed, transition_done, shader_reloaded, etc.)
    Watch {
        /// Output name filter (omit or '*' for all outputs)
        #[arg(long)]
        output: Option<String>,

        /// Event names to subscribe to (repeatable; defaults to all events)
        #[arg(long = "event")]
        events: Vec<String>,
    },

    /// Pause playback / animation with a tracked reason tag
    Pause {
        /// Output name or '*' for global pause
        #[arg(long, default_value = "*")]
        output: String,

        /// Pause reason tag (e.g. manual, battery, occluded)
        #[arg(long, default_value = "manual")]
        reason: String,
    },

    /// Clear a pause reason tag and resume if no other reasons remain
    Resume {
        /// Output name or '*' for global resume
        #[arg(long, default_value = "*")]
        output: String,

        /// Pause reason tag to remove
        #[arg(long, default_value = "manual")]
        reason: String,
    },

    /// Toggle 'manual' pause state
    Toggle {
        /// Output name or '*' for global toggle
        #[arg(long, default_value = "*")]
        output: String,
    },

    /// Force-reload shaders/media on the matching output(s)
    Reload {
        /// Output name or '*' for all outputs
        #[arg(long, default_value = "*")]
        output: String,
    },

    /// Query daemon status and active per-output wallpaper/shader diagnostics
    Status {
        /// Print raw JSON response
        #[arg(long)]
        json: bool,
    },

    /// List discovered Wayland outputs and their logical/physical geometry
    Outputs {
        /// Print raw JSON response
        #[arg(long)]
        json: bool,
    },

    /// Stop the running glwall daemon cleanly
    Stop,

    /// Compile-check one or more .frag files offline using surfaceless EGL/GLES 3.0
    CheckShader {
        /// Print JSON shader reflection schema & compile status
        #[arg(long)]
        json: bool,

        /// Fragment shader file paths to compile
        #[arg(required = true)]
        paths: Vec<PathBuf>,
    },
}
