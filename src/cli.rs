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
        /// Fragment shader file paths to compile
        #[arg(required = true)]
        paths: Vec<PathBuf>,
    },
}
