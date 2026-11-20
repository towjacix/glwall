# glwall

Standalone Wayland OpenGL ES 3.0 / EGL wallpaper daemon and CLI controller with first-class custom `.frag` shader support, zero-draw idle for static images, streaming animated GIF/WebP, FFmpeg NV12 video playback, HiDPI fractional scaling, and fullscreen auto-pause.

## Install

Via [mise](https://mise.jdx.dev/) (no Rust toolchain required on the target machine):

```bash
mise use -g github:towjacix/glwall
```

Or via [cargo-binstall](https://github.com/cargo-bins/cargo-binstall):

```bash
cargo binstall --git https://github.com/towjacix/glwall glwall
```

## Features

- **Custom GLSL Shaders (`.frag`)**: Hot-reload via `inotify` (keeps old shader on compile errors), unified `sampleWallpaper(uv)` / `samplePrevWallpaper(uv)` preamble across images, animations, and NV12 videos, plus automatic **Shadertoy (`mainImage`)** compatibility.
- **Multiple Input Types**:
  - Static images: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp` (via `image`) and `.avif`, `.heif`, `.jxl` (via `ffmpeg`)
  - Animated images: `.gif`, animated `.webp` (streaming frame-by-frame)
  - Video: `.mp4`, `.webm`, `.mkv`, `.mov`, `.m4v`, `.avi` via `ffmpeg` NV12 pipe -> PBO double-buffered `GL_R8`/`GL_RG8` GPU YUV->RGB conversion (`SIGSTOP`/`SIGCONT` pause)
  - Procedural shaders: `.frag` / `.glsl` directly without any media file
- **Wayland Layer-Shell**: `zwlr_layer_shell_v1` (`Layer::Background`, `namespace = "glwall:bg"`), `wp_fractional_scale_v1` + `wp_viewporter`, multi-output hot-plug, shared `EGLContext`, single-callback `wl_surface.frame` pacing.
- **Fullscreen Auto-Pause**: Native `zwlr_foreign_toplevel_management_v1` per-output fullscreen pause tracking with independent pause reasons (`manual`, `fullscreen`, custom IPC tags).
- **Single Binary Daemon + CLI**: NDJSON Unix socket IPC (`$XDG_RUNTIME_DIR/glwall.sock`).

## Quick Start

```bash
# Start daemon
glwall daemon

# Validate shaders offline (surfaceless EGL)
glwall check-shader shaders/*.frag

# Set static image, video, or procedural shader
glwall set ~/Pictures/wallpaper.jpg --transition dissolve --duration-ms 500
glwall set ~/Videos/loop.mp4 --hwaccel auto --fit cover
glwall set shaders/aurora.frag

# Live post-processing shader & runtime params
glwall shader shaders/vignette_grain.frag
glwall params 0.35 0.08 1.0 0.0

# Pause control
glwall pause
glwall resume
glwall pause --reason battery
glwall status
```

## Toolchain

Managed with [mise](https://mise.jdx.dev/):

```bash
mise install
mise run test
mise run build
```
