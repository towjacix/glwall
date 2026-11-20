mod app;
mod cli;
mod config;
mod gpu;
mod ipc;
mod media;
mod pause;
mod watcher;
mod wayland;

use std::fs;
use std::process::ExitCode;

use anyhow::{Context, Result, bail};
use clap::Parser;
use tracing_subscriber::EnvFilter;

use crate::cli::{Cli, Commands};
use crate::gpu::egl::EglState;
use crate::gpu::renderer::GlProgram;
use crate::ipc::{IpcRequest, IpcResponse, default_socket_path, send_request};

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

fn run(cli: Cli) -> Result<ExitCode> {
    let socket_path = cli.socket.unwrap_or_else(default_socket_path);

    match cli.command {
        Commands::Daemon { config, no_restore } => {
            app::run_daemon(config, socket_path, no_restore)?;
            Ok(ExitCode::SUCCESS)
        }
        Commands::CheckShader { paths } => {
            let egl = EglState::new_surfaceless()
                .context("failed to initialize surfaceless/pbuffer EGL for check-shader")?;
            let gl = egl.create_glow_context()?;
            let mut failed = 0usize;
            for path in &paths {
                let src = fs::read_to_string(path)
                    .with_context(|| format!("failed to read {}", path.display()))?;
                match GlProgram::compile(&gl, &src) {
                    Ok(prog) => {
                        println!(
                            "OK   {} (mode={:?}, uses_time={}, iChannel0={})",
                            path.display(),
                            prog.assembled.mode,
                            prog.assembled.uses_time,
                            prog.assembled.uses_ichannel0
                        );
                        prog.destroy(&gl);
                    }
                    Err(err) => {
                        eprintln!("FAIL {}:\n{err}", path.display());
                        failed += 1;
                    }
                }
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

fn print_simple_response(resp: IpcResponse) -> Result<ExitCode> {
    if !resp.ok {
        let msg = resp
            .message
            .unwrap_or_else(|| "request rejected".to_string());
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
                    let reasons = item
                        .get("pause_reasons")
                        .map(ToString::to_string)
                        .unwrap_or_else(|| "[]".to_string());
                    println!(
                        "{name}: phys={} target={target} shader={shader} paused={p} reasons={reasons}",
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
