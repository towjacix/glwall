# glwall

Standalone Wayland OpenGL ES 3.0 / EGL GPU background shader runtime and CLI controller with custom `.frag` shaders, generic named `sampler2D` textures, reflected typed uniforms, atomic `apply` transactions, NDJSON event streaming, streaming animated GIF/WebP, FFmpeg NV12 video playback, HiDPI fractional scaling, and fullscreen auto-pause.

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

- **Custom GLSL Shaders (`.frag`)**: Hot-reload via canonical path `inotify` (keeps old shader on compile errors), unified `sampleWallpaper(uv)`, `samplePrevWallpaper(uv)`, `wallpaperUv(uv)`, and `sampleTextureFitted(tex, tex_size, uv)` preamble across images, animations, and NV12 videos, FBO transition + post-shader composition, plus automatic **Shadertoy (`mainImage`)** compatibility.
- **Generic Named Textures (`sampler2D`)**: Bind arbitrary auxiliary textures (depth maps, masks, noise maps, LUTs) to any `uniform sampler2D u_<name>` with configurable `linear`/`nearest` filtering, `clamp_to_edge`/`repeat`/`mirrored_repeat` wrapping, optional `flip_y`, automatic `vec2 u_<name>_size` population, and ref-counted GPU texture sharing across outputs.
- **Reflected Typed Uniforms**: Active GL uniform reflection via `glGetActiveUniform` supporting `float`, `int`, `bool`, `vec2`, `vec3`, and `vec4` (alongside legacy `vec4 u_params`) with offline schema inspection via `glwall check-shader --json`.
- **Atomic Transactions & Event Streaming**: Non-blocking multi-client NDJSON Unix socket IPC (`$XDG_RUNTIME_DIR/glwall.sock`), monotonic `revision` tracking, `glwall apply --wait` commit ACKs after `eglSwapBuffers`, and `glwall watch` (`frame_committed`, `transition_done`, `shader_reloaded`, `media_error`, `output_added`, `output_removed`) for external layer synchronization.
- **Multiple Input Types**:
  - Static images: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp` (via `image`) and `.avif`, `.heif`, `.heic`, `.jxl` (via `ffmpeg`)
  - Animated images: `.gif`, animated `.webp` (streaming frame-by-frame)
  - Video: `.mp4`, `.webm`, `.mkv`, `.mov`, `.m4v`, `.avi` via `ffmpeg` NV12 pipe -> PBO double-buffered `GL_R8`/`GL_RG8` GPU YUV->RGB conversion (`SIGSTOP`/`SIGCONT` pause)
  - Procedural shaders: `.frag` / `.glsl` directly without any media file
- **Wayland Layer-Shell**: `zwlr_layer_shell_v1` (`Layer::Background`, `namespace = "glwall:bg"`), `wp_fractional_scale_v1` + `wp_viewporter`, multi-output hot-plug, shared `EGLContext`, single-callback `wl_surface.frame` pacing.
- **Fullscreen Auto-Pause**: Native `zwlr_foreign_toplevel_management_v1` per-output fullscreen pause tracking with independent pause reasons (`manual`, `fullscreen`, custom IPC tags).

## Quick Start

```bash
# Start daemon (--no-persist disables state.json writes when externally managed)
glwall daemon

# Inspect & validate shaders offline with reflected uniforms/samplers (surfaceless EGL)
glwall check-shader --json shaders/*.frag

# Set static image, video, or procedural shader
glwall set ~/Pictures/wallpaper.jpg --transition dissolve --duration-ms 500
glwall set ~/Videos/loop.mp4 --hwaccel auto --fit cover
glwall set shaders/aurora.frag

# Atomic media + shader + depth texture + uniform update with commit ACK
glwall apply \
  --media ~/Pictures/wallpaper.jpg \
  --shader shaders/parallax_depth.frag \
  --texture u_depth=~/Pictures/depth.png:linear:clamp_to_edge \
  --uniform u_pointer=0.5,0.5 \
  --uniform u_parallaxStrength=0.04 \
  --wait

# Live updates & event watching
glwall uniform u_pointer=0.72,0.35
glwall texture u_depth --unset
glwall watch --event frame_committed --event shader_reloaded

# Pause control & status
glwall pause
glwall resume
glwall pause --reason battery
glwall status --json
```

## Toolchain

Managed with [mise](https://mise.jdx.dev/):

```bash
mise install
mise run test
mise run build
```
