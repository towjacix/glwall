mod app;
mod cli;
mod config;
mod gpu;
mod ipc;
mod media;
mod pause;
mod watcher;
mod wayland;

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use clap::Parser;
use serde_json::json;
use tracing_subscriber::EnvFilter;

use crate::cli::{Cli, Commands, FilterArg, WrapArg};
use crate::gpu::egl::EglState;
use crate::gpu::renderer::GlProgram;
use crate::gpu::uniforms::{TextureFilter, TextureSpec, TextureWrap, UniformValue};
use crate::ipc::{
    ApplyMedia, ApplyTransition, IpcRequest, IpcResponse, default_socket_path, send_request,
    send_request_with_timeout, stream_events,
};

fn main() -> ExitCode {
    init_logging();
    let cli = Cli::parse();

    match run(cli) {
        Ok(code) => code,
        Err(err) => {
            eprintln!("glwall error: {err:#}");
            ExitCode::FAILURE
        }
    }
}

fn init_logging() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .try_init();
}

impl From<FilterArg> for TextureFilter {
    fn from(arg: FilterArg) -> Self {
        match arg {
            FilterArg::Linear => TextureFilter::Linear,
            FilterArg::Nearest => TextureFilter::Nearest,
        }
    }
}

impl From<WrapArg> for TextureWrap {
    fn from(arg: WrapArg) -> Self {
        match arg {
            WrapArg::ClampToEdge => TextureWrap::ClampToEdge,
            WrapArg::Repeat => TextureWrap::Repeat,
            WrapArg::MirroredRepeat => TextureWrap::MirroredRepeat,
        }
    }
}

fn run(cli: Cli) -> Result<ExitCode> {
    let socket_path = cli.socket.unwrap_or_else(default_socket_path);

    match cli.command {
        Commands::Daemon {
            config,
            no_restore,
            no_persist,
        } => {
            app::run_daemon(config, socket_path, no_restore, no_persist)?;
            Ok(ExitCode::SUCCESS)
        }
        Commands::CheckShader {
            json: json_out,
            paths,
        } => {
            let egl = EglState::new_surfaceless()
                .context("failed to initialize surfaceless/pbuffer EGL for check-shader")?;
            let gl = egl.create_glow_context()?;
            let mut failed = 0usize;
            let mut json_results = Vec::new();
            for path in &paths {
                let src = match fs::read_to_string(path) {
                    Ok(s) => s,
                    Err(err) => {
                        let msg = format!("failed to read {}: {err}", path.display());
                        if json_out {
                            json_results.push(json!({
                                "path": path,
                                "ok": false,
                                "error": msg,
                            }));
                        } else {
                            eprintln!("FAIL {msg}");
                        }
                        failed += 1;
                        continue;
                    }
                };
                match GlProgram::compile(&gl, &src) {
                    Ok(prog) => {
                        let schema = prog.schema();
                        if json_out {
                            json_results.push(json!({
                                "path": path,
                                "ok": true,
                                "schema": schema,
                            }));
                        } else {
                            println!(
                                "OK   {} (mode={}, uses_time={}, iChannel0={}, uniforms={}, samplers={})",
                                path.display(),
                                schema.mode,
                                schema.uses_time,
                                schema.uses_ichannel0,
                                schema.uniforms.len(),
                                schema.samplers.len(),
                            );
                        }
                        prog.destroy(&gl);
                    }
                    Err(err) => {
                        if json_out {
                            json_results.push(json!({
                                "path": path,
                                "ok": false,
                                "error": format!("{err}"),
                            }));
                        } else {
                            eprintln!("FAIL {}:\n{err}", path.display());
                        }
                        failed += 1;
                    }
                }
            }
            if json_out {
                println!("{}", serde_json::to_string_pretty(&json_results)?);
            }
            if failed > 0 {
                bail!("{failed} shader(s) failed compilation");
            }
            Ok(ExitCode::SUCCESS)
        }
        Commands::Set {
            target,
            output,
            fit,
            transition,
            duration_ms,
            shader,
            fps,
            hwaccel,
        } => {
            let resp = send_request(
                &socket_path,
                &IpcRequest::Set {
                    output,
                    target,
                    fit: fit.map(Into::into),
                    transition,
                    duration_ms,
                    shader,
                    fps,
                    hwaccel: hwaccel.map(Into::into),
                },
            )?;
            print_simple_response(resp)
        }
        Commands::Shader { path, output } => {
            let resp = send_request(&socket_path, &IpcRequest::Shader { output, path })?;
            print_simple_response(resp)
        }
        Commands::Params { x, y, z, w, output } => {
            let resp = send_request(
                &socket_path,
                &IpcRequest::Params {
                    output,
                    values: [x, y, z, w],
                },
            )?;
            print_simple_response(resp)
        }
        Commands::Uniform {
            assignments,
            output,
        } => {
            if assignments.len() == 1 {
                let (name, value) = UniformValue::parse_cli(&assignments[0])?;
                let resp = send_request(
                    &socket_path,
                    &IpcRequest::Uniform {
                        output,
                        name,
                        value,
                    },
                )?;
                print_simple_response(resp)
            } else {
                let mut values = BTreeMap::new();
                for spec in &assignments {
                    let (name, value) = UniformValue::parse_cli(spec)?;
                    values.insert(name, value);
                }
                let resp = send_request(&socket_path, &IpcRequest::Uniforms { output, values })?;
                print_simple_response(resp)
            }
        }
        Commands::UniformTransient {
            assignments,
            output,
            session,
            seq,
            clear,
        } => {
            if !clear && assignments.is_empty() {
                bail!("provide at least one NAME=VALUE assignment or pass --clear");
            }
            let mut values = BTreeMap::new();
            for spec in &assignments {
                let (name, value) = UniformValue::parse_cli(spec)?;
                values.insert(name, value);
            }
            let resp = send_request(
                &socket_path,
                &IpcRequest::UniformsTransient {
                    output,
                    session,
                    seq,
                    values,
                    clear,
                },
            )?;
            print_simple_response(resp)
        }
        Commands::Texture {
            name,
            path,
            unset,
            output,
            filter,
            wrap,
            flip_y,
        } => {
            let req = if unset || path.as_deref().is_some_and(|p| p.trim() == "none") {
                IpcRequest::TextureUnset { output, name }
            } else if let Some(path) = path {
                IpcRequest::TextureSet {
                    output,
                    name,
                    path,
                    filter: filter.into(),
                    wrap: wrap.into(),
                    flip_y,
                }
            } else {
                bail!("texture path is required unless --unset is passed");
            };
            let resp = send_request(&socket_path, &req)?;
            print_simple_response(resp)
        }
        Commands::Apply {
            output,
            media,
            fit,
            shader,
            textures,
            uniforms,
            transition,
            duration_ms,
            revision,
            wait,
        } => {
            let mut parsed_textures = BTreeMap::new();
            for raw in &textures {
                let (name, spec) = parse_cli_texture_spec(raw)?;
                parsed_textures.insert(name, spec);
            }
            let mut parsed_uniforms = BTreeMap::new();
            for raw in &uniforms {
                let (name, val) = UniformValue::parse_cli(raw)?;
                parsed_uniforms.insert(name, val);
            }
            let media_obj = media.map(|target| ApplyMedia {
                target,
                fit: fit.map(Into::into),
                fps: None,
                hwaccel: None,
            });
            let tr_obj = if transition.is_some() || duration_ms.is_some() {
                Some(ApplyTransition {
                    name: transition,
                    duration_ms,
                })
            } else {
                None
            };
            let req = IpcRequest::Apply {
                output,
                revision,
                media: media_obj,
                shader,
                textures: parsed_textures,
                uniforms: parsed_uniforms,
                transition: tr_obj,
                wait_commit: wait,
            };
            let timeout = if wait {
                Duration::from_secs(15)
            } else {
                Duration::from_secs(5)
            };
            let resp = send_request_with_timeout(&socket_path, &req, timeout)?;
            print_simple_response(resp)
        }
        Commands::Watch { output, events } => {
            stream_events(&socket_path, output, events, |ev| {
                println!("{}", serde_json::to_string(&ev)?);
                Ok(true)
            })?;
            Ok(ExitCode::SUCCESS)
        }
        Commands::Pause { output, reason } => {
            let resp = send_request(&socket_path, &IpcRequest::Pause { output, reason })?;
            print_simple_response(resp)
        }
        Commands::Resume { output, reason } => {
            let resp = send_request(&socket_path, &IpcRequest::Resume { output, reason })?;
            print_simple_response(resp)
        }
        Commands::Toggle { output } => {
            let resp = send_request(&socket_path, &IpcRequest::Toggle { output })?;
            print_simple_response(resp)
        }
        Commands::Reload { output } => {
            let resp = send_request(&socket_path, &IpcRequest::Reload { output })?;
            print_simple_response(resp)
        }
        Commands::Status { json } => {
            let resp = send_request(&socket_path, &IpcRequest::Status)?;
            print_data_response(resp, json)
        }
        Commands::Outputs { json } => {
            let resp = send_request(&socket_path, &IpcRequest::Outputs)?;
            print_data_response(resp, json)
        }
        Commands::Stop => {
            let resp = send_request(&socket_path, &IpcRequest::Stop)?;
            print_simple_response(resp)
        }
    }
}

fn parse_cli_texture_spec(raw: &str) -> Result<(String, Option<TextureSpec>)> {
    let Some((raw_name, raw_rest)) = raw.split_once('=') else {
        bail!(
            "invalid --texture '{raw}': expected NAME=PATH[:filter][:wrap][:flip_y] or NAME=none"
        );
    };
    let name = raw_name.trim();
    if name.is_empty() {
        bail!("invalid --texture '{raw}': empty texture name");
    }
    let rest = raw_rest.trim();
    if rest.is_empty() || rest.eq_ignore_ascii_case("none") {
        return Ok((name.to_string(), None));
    }

    let mut path_str = rest;
    let mut filter = TextureFilter::default();
    let mut wrap = TextureWrap::default();
    let mut flip_y = false;

    while let Some((prefix, suffix)) = path_str.rsplit_once(':') {
        let opt = suffix.trim();
        if let Ok(f) = TextureFilter::parse_str(opt) {
            filter = f;
            path_str = prefix;
        } else if let Ok(w) = TextureWrap::parse_str(opt) {
            wrap = w;
            path_str = prefix;
        } else if opt.eq_ignore_ascii_case("flip_y") || opt.eq_ignore_ascii_case("flipy") {
            flip_y = true;
            path_str = prefix;
        } else {
            break;
        }
    }

    Ok((
        name.to_string(),
        Some(TextureSpec {
            path: PathBuf::from(path_str),
            filter,
            wrap,
            flip_y,
        }),
    ))
}

fn print_simple_response(resp: IpcResponse) -> Result<ExitCode> {
    if !resp.ok {
        let msg = resp
            .message
            .unwrap_or_else(|| "request rejected".to_string());
        if let Some(code) = resp.code {
            bail!("[{code}] {msg}");
        }
        bail!("{msg}");
    }
    if let Some(msg) = resp.message {
        println!("{msg}");
    } else if let Some(data) = resp.data {
        println!("{}", serde_json::to_string_pretty(&data)?);
    }
    Ok(ExitCode::SUCCESS)
}

fn print_data_response(resp: IpcResponse, json_mode: bool) -> Result<ExitCode> {
    if !resp.ok {
        let msg = resp
            .message
            .unwrap_or_else(|| "request rejected".to_string());
        if let Some(code) = resp.code {
            bail!("[{code}] {msg}");
        }
        bail!("{msg}");
    }
    if let Some(data) = resp.data {
        if json_mode {
            println!("{}", serde_json::to_string_pretty(&data)?);
        } else if let Some(arr) = data.get("outputs").and_then(|v| v.as_array()) {
            for item in arr {
                let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("?");
                let phys = item.get("physical_size");
                let target = item.get("target").and_then(|v| v.as_str()).unwrap_or("-");
                let shader = item.get("shader").and_then(|v| v.as_str()).unwrap_or("-");
                let paused = item.get("paused").and_then(|v| v.as_bool());
                if let Some(p) = paused {
                    let rev = item.get("revision").and_then(|v| v.as_u64()).unwrap_or(0);
                    let reasons = item
                        .get("pause_reasons")
                        .map(ToString::to_string)
                        .unwrap_or_else(|| "[]".to_string());
                    println!(
                        "{name}: rev={rev} phys={} target={target} shader={shader} paused={p} reasons={reasons}",
                        phys.map(ToString::to_string).unwrap_or_default()
                    );
                } else {
                    let scale = item.get("scale").and_then(|v| v.as_f64()).unwrap_or(1.0);
                    println!(
                        "{name}: phys={} scale={scale:.2}",
                        phys.map(ToString::to_string).unwrap_or_default()
                    );
                }
            }
        } else {
            println!("{}", serde_json::to_string_pretty(&data)?);
        }
    }
    Ok(ExitCode::SUCCESS)
}
