use std::fs;
use std::io::{BufRead, BufReader, ErrorKind, Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::{FitMode, HwAccel};

const MAX_IPC_LINE_BYTES: usize = 64 * 1024;
const IPC_IO_TIMEOUT: Duration = Duration::from_secs(2);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum IpcRequest {
    Set {
        output: String,
        target: String,
        fit: Option<FitMode>,
        transition: Option<String>,
        duration_ms: Option<u32>,
        shader: Option<String>,
        fps: Option<f32>,
        hwaccel: Option<HwAccel>,
    },
    Shader {
        output: String,
        path: String,
    },
    Params {
        output: String,
        values: [f32; 4],
    },
    Pause {
        output: String,
        reason: String,
    },
    Resume {
        output: String,
        reason: String,
    },
    Toggle {
        output: String,
    },
    Reload {
        output: String,
    },
    Status,
    Outputs,
    Stop,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IpcResponse {
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl IpcResponse {
    pub fn ok_msg(msg: impl Into<String>) -> Self {
        Self {
            ok: true,
            message: Some(msg.into()),
            data: None,
        }
    }

    pub fn ok_data(data: Value) -> Self {
        Self {
            ok: true,
            message: None,
            data: Some(data),
        }
    }

    pub fn err(msg: impl Into<String>) -> Self {
        Self {
            ok: false,
            message: Some(msg.into()),
            data: None,
        }
    }
}

pub fn default_socket_path() -> PathBuf {
    if let Ok(runtime) = std::env::var("XDG_RUNTIME_DIR")
        && !runtime.trim().is_empty()
    {
        return PathBuf::from(runtime).join("glwall.sock");
    }
    let uid = unsafe { libc::geteuid() };
    PathBuf::from(format!("/tmp/glwall-{uid}.sock"))
}

pub fn bind_ipc_listener(socket_path: &Path) -> Result<UnixListener> {
    if let Some(parent) = socket_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create socket dir {}", parent.display()))?;
    }

    if socket_path.exists() {
        match UnixStream::connect(socket_path) {
            Ok(_) => bail!("glwall daemon already running on {}", socket_path.display()),
            Err(err)
                if matches!(
                    err.kind(),
                    ErrorKind::ConnectionRefused | ErrorKind::NotFound
                ) =>
            {
                let _ = fs::remove_file(socket_path);
            }
            Err(err) => {
                // Other errors (e.g. ENOTSOCK) should still not clobber non-sockets blindly.
                return Err(err).with_context(|| {
                    format!(
                        "existing path {} is not a connectable glwall socket",
                        socket_path.display()
                    )
                });
            }
        }
    }

    let listener = UnixListener::bind(socket_path)
        .with_context(|| format!("failed to bind IPC socket {}", socket_path.display()))?;
    listener
        .set_nonblocking(true)
        .context("failed to set non-blocking IPC listener")?;
    Ok(listener)
}

pub fn read_ipc_request(stream: &mut UnixStream) -> Result<IpcRequest> {
    stream.set_nonblocking(false)?;
    stream.set_read_timeout(Some(IPC_IO_TIMEOUT))?;
    stream.set_write_timeout(Some(IPC_IO_TIMEOUT))?;

    let mut line = String::new();
    let limited =
        BufReader::new(stream.take((MAX_IPC_LINE_BYTES + 1) as u64)).read_line(&mut line)?;
    if limited == 0 {
        bail!("empty IPC request");
    }
    if line.len() > MAX_IPC_LINE_BYTES {
        bail!("IPC request exceeds {MAX_IPC_LINE_BYTES} bytes");
    }
    serde_json::from_str(line.trim_end()).context("invalid NDJSON IPC request")
}

pub fn write_ipc_response(stream: &mut UnixStream, resp: &IpcResponse) -> Result<()> {
    let mut payload = serde_json::to_vec(resp)?;
    payload.push(b'\n');
    stream.write_all(&payload)?;
    stream.flush()?;
    Ok(())
}

pub fn send_request(socket_path: &Path, req: &IpcRequest) -> Result<IpcResponse> {
    let mut stream = UnixStream::connect(socket_path).with_context(|| {
        format!(
            "failed to connect to glwall socket {} (is `glwall daemon` running?)",
            socket_path.display()
        )
    })?;
    stream.set_read_timeout(Some(Duration::from_secs(5)))?;
    stream.set_write_timeout(Some(IPC_IO_TIMEOUT))?;

    let mut line = serde_json::to_vec(req)?;
    line.push(b'\n');
    stream.write_all(&line)?;
    stream.flush()?;

    let mut resp_line = String::new();
    let read = BufReader::new((&mut stream).take((MAX_IPC_LINE_BYTES + 1) as u64))
        .read_line(&mut resp_line)?;
    if read == 0 {
        bail!("glwall daemon closed IPC connection without a response");
    }
    if resp_line.len() > MAX_IPC_LINE_BYTES {
        bail!("IPC response exceeded {MAX_IPC_LINE_BYTES} bytes");
    }
    serde_json::from_str(resp_line.trim_end()).context("invalid JSON response from daemon")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ndjson_request_response_roundtrip() {
        let req = IpcRequest::Set {
            output: "DP-1".to_string(),
            target: "/tmp/test.mp4".to_string(),
            fit: Some(FitMode::Cover),
            transition: Some("dissolve".to_string()),
            duration_ms: Some(500),
            shader: Some("/tmp/crt.frag".to_string()),
            fps: Some(60.0),
            hwaccel: Some(HwAccel::Cuda),
        };
        let json = serde_json::to_string(&req).unwrap();
        let parsed: IpcRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, req);

        let resp = IpcResponse::ok_msg("updated");
        let resp_json = serde_json::to_string(&resp).unwrap();
        let parsed_resp: IpcResponse = serde_json::from_str(&resp_json).unwrap();
        assert_eq!(parsed_resp, resp);
    }

    #[test]
    fn socket_roundtrip_accepts_and_replies() {
        let sock_path =
            std::env::temp_dir().join(format!("glwall-ipc-test-{}.sock", std::process::id()));
        let _ = fs::remove_file(&sock_path);

        let listener = bind_ipc_listener(&sock_path).unwrap();
        let server_thread = std::thread::spawn(move || {
            loop {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        let req = read_ipc_request(&mut stream).unwrap();
                        assert_eq!(req, IpcRequest::Status);
                        write_ipc_response(&mut stream, &IpcResponse::ok_msg("alive")).unwrap();
                        break;
                    }
                    Err(e) if e.kind() == ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(5));
                    }
                    Err(e) => panic!("accept failed: {e}"),
                }
            }
        });

        let resp = send_request(&sock_path, &IpcRequest::Status).unwrap();
        assert!(resp.ok);
        assert_eq!(resp.message.as_deref(), Some("alive"));

        server_thread.join().unwrap();
        let _ = fs::remove_file(&sock_path);
    }
}
