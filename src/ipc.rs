use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fs;
use std::io::{BufRead, BufReader, ErrorKind, Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use calloop::{
    EventSource, Interest, Mode, Poll, PostAction, Readiness, RegistrationToken, Token,
    TokenFactory, generic::Generic,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::{FitMode, HwAccel};
use crate::gpu::uniforms::{TextureFilter, TextureSpec, TextureWrap, UniformValue};

const MAX_IPC_LINE_BYTES: usize = 64 * 1024;
const MAX_SUBSCRIBER_BUFFER_BYTES: usize = 2 * 1024 * 1024;
const IPC_IO_TIMEOUT: Duration = Duration::from_secs(2);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ApplyMedia {
    pub target: String,
    #[serde(default)]
    pub fit: Option<FitMode>,
    #[serde(default)]
    pub fps: Option<f32>,
    #[serde(default)]
    pub hwaccel: Option<HwAccel>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApplyTransition {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub duration_ms: Option<u32>,
}

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
    Uniform {
        output: String,
        name: String,
        value: UniformValue,
    },
    Uniforms {
        output: String,
        values: BTreeMap<String, UniformValue>,
    },
    TextureSet {
        output: String,
        name: String,
        path: String,
        #[serde(default)]
        filter: TextureFilter,
        #[serde(default)]
        wrap: TextureWrap,
        #[serde(default)]
        flip_y: bool,
    },
    TextureUnset {
        output: String,
        name: String,
    },
    Apply {
        output: String,
        #[serde(default)]
        revision: Option<u64>,
        #[serde(default)]
        media: Option<ApplyMedia>,
        #[serde(default)]
        shader: Option<String>,
        #[serde(default)]
        textures: BTreeMap<String, Option<TextureSpec>>,
        #[serde(default)]
        uniforms: BTreeMap<String, UniformValue>,
        #[serde(default)]
        transition: Option<ApplyTransition>,
        #[serde(default)]
        wait_commit: bool,
    },
    Subscribe {
        #[serde(default)]
        output: Option<String>,
        #[serde(default)]
        events: Vec<String>,
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum IpcEvent {
    FrameCommitted {
        output: String,
        revision: u64,
        frame: i32,
    },
    TransitionDone {
        output: String,
        revision: u64,
    },
    ShaderReloaded {
        output: String,
        revision: u64,
        ok: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    MediaError {
        output: String,
        message: String,
    },
    OutputAdded {
        output: String,
    },
    OutputRemoved {
        output: String,
    },
}

impl IpcEvent {
    pub fn event_name(&self) -> &'static str {
        match self {
            Self::FrameCommitted { .. } => "frame_committed",
            Self::TransitionDone { .. } => "transition_done",
            Self::ShaderReloaded { .. } => "shader_reloaded",
            Self::MediaError { .. } => "media_error",
            Self::OutputAdded { .. } => "output_added",
            Self::OutputRemoved { .. } => "output_removed",
        }
    }

    pub fn output_name(&self) -> &str {
        match self {
            Self::FrameCommitted { output, .. }
            | Self::TransitionDone { output, .. }
            | Self::ShaderReloaded { output, .. }
            | Self::MediaError { output, .. }
            | Self::OutputAdded { output }
            | Self::OutputRemoved { output } => output,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IpcClientId(pub u64);

#[derive(Debug, Clone)]
pub struct SubscriptionFilter {
    pub output: Option<String>,
    pub events: Vec<String>,
}

impl SubscriptionFilter {
    pub fn matches(&self, ev: &IpcEvent) -> bool {
        if let Some(ref out) = self.output {
            let trimmed = out.trim();
            if !trimmed.is_empty() && trimmed != "*" && trimmed != ev.output_name() {
                return false;
            }
        }
        self.events.is_empty() || self.events.iter().any(|e| e == ev.event_name())
    }
}

pub enum DispatchOutcome {
    Reply(IpcResponse),
    Deferred,
}

pub struct ClientState {
    stream: UnixStream,
    token: Option<RegistrationToken>,
    want_write: Arc<AtomicBool>,
    read_buf: Vec<u8>,
    write_buf: VecDeque<u8>,
    read_eof: bool,
    closed: bool,
    subscription: Option<SubscriptionFilter>,
    pending_acks: usize,
}

impl ClientState {
    fn new(stream: UnixStream, want_write: Arc<AtomicBool>) -> Self {
        Self {
            stream,
            token: None,
            want_write,
            read_buf: Vec::new(),
            write_buf: VecDeque::new(),
            read_eof: false,
            closed: false,
            subscription: None,
            pending_acks: 0,
        }
    }

    fn push_bytes(&mut self, bytes: &[u8]) {
        if self.closed {
            return;
        }
        if self.write_buf.len() + bytes.len() > MAX_SUBSCRIBER_BUFFER_BYTES {
            self.closed = true;
            return;
        }
        self.write_buf.extend(bytes);
    }

    fn flush(&mut self) -> bool {
        while !self.write_buf.is_empty() {
            let (first, _) = self.write_buf.as_slices();
            if first.is_empty() {
                break;
            }
            match (&self.stream).write(first) {
                Ok(0) => {
                    self.closed = true;
                    break;
                }
                Ok(n) => {
                    drop(self.write_buf.drain(..n));
                }
                Err(err) if err.kind() == ErrorKind::WouldBlock => break,
                Err(_) => {
                    self.closed = true;
                    break;
                }
            }
        }
        let need_write = !self.closed && !self.write_buf.is_empty();
        self.want_write.store(need_write, Ordering::Release);
        self.closed
    }

    fn should_remove(&self) -> bool {
        self.closed
            || (self.read_eof
                && self.write_buf.is_empty()
                && self.pending_acks == 0
                && self.subscription.is_none())
    }
}

pub struct IpcClientSource {
    pub id: IpcClientId,
    generic: Generic<UnixStream>,
    want_write: Arc<AtomicBool>,
}

impl EventSource for IpcClientSource {
    type Event = (IpcClientId, Readiness);
    type Metadata = ();
    type Ret = PostAction;
    type Error = std::io::Error;

    fn process_events<F>(
        &mut self,
        readiness: Readiness,
        token: Token,
        mut callback: F,
    ) -> std::result::Result<PostAction, Self::Error>
    where
        F: FnMut(Self::Event, &mut Self::Metadata) -> Self::Ret,
    {
        let client_id = self.id;
        let want_write = Arc::clone(&self.want_write);
        let action = self.generic.process_events(readiness, token, |r, _| {
            let action = callback((client_id, r), &mut ());
            Ok(action)
        })?;
        if matches!(action, PostAction::Remove) {
            return Ok(PostAction::Remove);
        }

        let want_write = want_write.load(Ordering::Acquire);
        if self.generic.interest.writable != want_write || !self.generic.interest.readable {
            self.generic.interest = if want_write {
                Interest::BOTH
            } else {
                Interest::READ
            };
            return Ok(PostAction::Reregister);
        }
        Ok(action)
    }

    fn register(
        &mut self,
        poll: &mut Poll,
        token_factory: &mut TokenFactory,
    ) -> calloop::Result<()> {
        self.generic.register(poll, token_factory)
    }

    fn reregister(
        &mut self,
        poll: &mut Poll,
        token_factory: &mut TokenFactory,
    ) -> calloop::Result<()> {
        let desired = if self.want_write.load(Ordering::Acquire) {
            Interest::BOTH
        } else {
            Interest::READ
        };
        self.generic.interest = desired;
        self.generic.reregister(poll, token_factory)
    }

    fn unregister(&mut self, poll: &mut Poll) -> calloop::Result<()> {
        self.generic.unregister(poll)
    }
}

pub struct IpcHub {
    next_id: u64,
    clients: HashMap<IpcClientId, ClientState>,
    pending_reregister: Vec<IpcClientId>,
}

impl Default for IpcHub {
    fn default() -> Self {
        Self::new()
    }
}

impl IpcHub {
    pub fn new() -> Self {
        Self {
            next_id: 1,
            clients: HashMap::new(),
            pending_reregister: Vec::new(),
        }
    }

    pub fn accept_stream(&mut self, stream: UnixStream) -> Result<(IpcClientId, IpcClientSource)> {
        stream
            .set_nonblocking(true)
            .context("failed to set non-blocking IPC client stream")?;
        let cloned = stream
            .try_clone()
            .context("failed to clone IPC client stream")?;
        let id = IpcClientId(self.next_id);
        self.next_id += 1;

        let want_write = Arc::new(AtomicBool::new(false));
        let source = IpcClientSource {
            id,
            generic: Generic::new(stream, Interest::READ, Mode::Level),
            want_write: Arc::clone(&want_write),
        };
        self.clients
            .insert(id, ClientState::new(cloned, want_write));
        Ok((id, source))
    }

    pub fn set_client_token(&mut self, id: IpcClientId, token: RegistrationToken) {
        if let Some(client) = self.clients.get_mut(&id) {
            client.token = Some(token);
        }
    }

    pub fn is_client_alive(&self, id: IpcClientId) -> bool {
        self.clients.get(&id).is_some_and(|c| !c.closed)
    }

    pub fn retain_pending_ack(&mut self, id: IpcClientId) {
        if let Some(client) = self.clients.get_mut(&id) {
            client.pending_acks += 1;
        }
    }

    pub fn complete_pending_ack(&mut self, id: IpcClientId, resp: &IpcResponse) {
        if let Some(client) = self.clients.get_mut(&id) {
            client.pending_acks = client.pending_acks.saturating_sub(1);
        }
        self.send_response(id, resp);
    }

    pub fn send_response(&mut self, id: IpcClientId, resp: &IpcResponse) {
        let Ok(mut payload) = serde_json::to_vec(resp) else {
            return;
        };
        payload.push(b'\n');
        self.enqueue_and_flush(id, &payload);
    }

    pub fn broadcast_event(&mut self, ev: &IpcEvent) {
        let matching_ids: Vec<IpcClientId> = self
            .clients
            .iter()
            .filter_map(|(&id, client)| {
                client
                    .subscription
                    .as_ref()
                    .is_some_and(|sub| sub.matches(ev))
                    .then_some(id)
            })
            .collect();
        if matching_ids.is_empty() {
            return;
        }
        let Ok(mut payload) = serde_json::to_vec(ev) else {
            return;
        };
        payload.push(b'\n');
        for id in matching_ids {
            self.enqueue_and_flush(id, &payload);
        }
    }

    fn enqueue_and_flush(&mut self, id: IpcClientId, bytes: &[u8]) {
        let Some(client) = self.clients.get_mut(&id) else {
            return;
        };
        client.push_bytes(bytes);
        let closed = client.flush();
        if !closed && !client.write_buf.is_empty() {
            self.pending_reregister.push(id);
        }
        if client.should_remove() {
            client.closed = true;
            self.pending_reregister.push(id);
        }
    }

    pub fn drain_pending_reregisters(&mut self) -> Vec<(IpcClientId, RegistrationToken, bool)> {
        let ids = std::mem::take(&mut self.pending_reregister);
        let mut out = Vec::with_capacity(ids.len());
        for id in ids {
            if let Some(client) = self.clients.get(&id)
                && let Some(token) = client.token
            {
                out.push((id, token, client.closed));
            }
        }
        out
    }

    pub fn remove_client(&mut self, id: IpcClientId) {
        self.clients.remove(&id);
    }

    pub fn poll_client_lines(
        &mut self,
        id: IpcClientId,
        readiness: Readiness,
    ) -> (Vec<String>, bool) {
        let Some(client) = self.clients.get_mut(&id) else {
            return (Vec::new(), true);
        };

        if readiness.error {
            client.closed = true;
            return (Vec::new(), true);
        }

        if readiness.readable {
            let mut tmp = [0u8; 4096];
            loop {
                match (&client.stream).read(&mut tmp) {
                    Ok(0) => {
                        client.read_eof = true;
                        break;
                    }
                    Ok(n) => {
                        client.read_buf.extend_from_slice(&tmp[..n]);
                        if client.read_buf.len() > MAX_IPC_LINE_BYTES * 4 {
                            let err = IpcResponse::err(format!(
                                "IPC request exceeded {MAX_IPC_LINE_BYTES} bytes"
                            ));
                            if let Ok(mut payload) = serde_json::to_vec(&err) {
                                payload.push(b'\n');
                                client.push_bytes(&payload);
                            }
                            client.read_buf.clear();
                            client.read_eof = true;
                            break;
                        }
                    }
                    Err(err) if err.kind() == ErrorKind::WouldBlock => break,
                    Err(_) => {
                        client.closed = true;
                        return (Vec::new(), true);
                    }
                }
            }
        }

        let mut lines = Vec::new();
        while let Some(pos) = client.read_buf.iter().position(|&b| b == b'\n') {
            let mut raw_line: Vec<u8> = client.read_buf.drain(..=pos).collect();
            raw_line.pop();
            if raw_line.last() == Some(&b'\r') {
                raw_line.pop();
            }
            if raw_line.len() > MAX_IPC_LINE_BYTES {
                let err =
                    IpcResponse::err(format!("IPC request exceeds {MAX_IPC_LINE_BYTES} bytes"));
                if let Ok(mut payload) = serde_json::to_vec(&err) {
                    payload.push(b'\n');
                    client.push_bytes(&payload);
                }
                client.read_eof = true;
                break;
            }
            if raw_line.is_empty() {
                continue;
            }
            match String::from_utf8(raw_line) {
                Ok(s) => lines.push(s),
                Err(_) => {
                    let err = IpcResponse::err("invalid UTF-8 in NDJSON request");
                    if let Ok(mut payload) = serde_json::to_vec(&err) {
                        payload.push(b'\n');
                        client.push_bytes(&payload);
                    }
                }
            }
        }

        if readiness.writable {
            client.flush();
        }

        let remove_now = lines.is_empty() && client.should_remove();
        (lines, remove_now)
    }

    pub fn handle_raw_line<F>(&mut self, id: IpcClientId, raw: &str, mut on_request: F) -> bool
    where
        F: FnMut(IpcClientId, IpcRequest) -> DispatchOutcome,
    {
        match serde_json::from_str::<IpcRequest>(raw.trim_end()) {
            Ok(IpcRequest::Subscribe { output, events }) => {
                if let Some(client) = self.clients.get_mut(&id) {
                    client.subscription = Some(SubscriptionFilter { output, events });
                }
                self.send_response(id, &IpcResponse::ok_msg("subscribed"));
            }
            Ok(req) => match on_request(id, req) {
                DispatchOutcome::Reply(resp) => {
                    self.send_response(id, &resp);
                }
                DispatchOutcome::Deferred => {
                    if let Some(client) = self.clients.get_mut(&id) {
                        client.flush();
                    }
                }
            },
            Err(err) => {
                self.send_response(
                    id,
                    &IpcResponse::err(format!("invalid NDJSON IPC request: {err}")),
                );
            }
        }

        self.clients.get(&id).is_none_or(ClientState::should_remove)
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

pub fn send_request(socket_path: &Path, req: &IpcRequest) -> Result<IpcResponse> {
    send_request_with_timeout(socket_path, req, Duration::from_secs(5))
}

pub fn send_request_with_timeout(
    socket_path: &Path,
    req: &IpcRequest,
    read_timeout: Duration,
) -> Result<IpcResponse> {
    let mut stream = UnixStream::connect(socket_path).with_context(|| {
        format!(
            "failed to connect to glwall socket {} (is `glwall daemon` running?)",
            socket_path.display()
        )
    })?;
    stream.set_read_timeout(Some(read_timeout))?;
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

pub fn stream_events<F>(
    socket_path: &Path,
    output: Option<String>,
    events: Vec<String>,
    mut on_event: F,
) -> Result<()>
where
    F: FnMut(IpcEvent) -> Result<bool>,
{
    let mut stream = UnixStream::connect(socket_path).with_context(|| {
        format!(
            "failed to connect to glwall socket {} (is `glwall daemon` running?)",
            socket_path.display()
        )
    })?;
    stream.set_write_timeout(Some(IPC_IO_TIMEOUT))?;
    stream.set_read_timeout(Some(Duration::from_secs(5)))?;

    let req = IpcRequest::Subscribe { output, events };
    let mut line = serde_json::to_vec(&req)?;
    line.push(b'\n');
    stream.write_all(&line)?;
    stream.flush()?;

    let mut reader = BufReader::new(stream);
    let mut ack_line = String::new();
    let ack_read = (&mut reader)
        .take((MAX_IPC_LINE_BYTES + 1) as u64)
        .read_line(&mut ack_line)?;
    if ack_read == 0 {
        bail!("glwall daemon closed IPC connection before subscribe ACK");
    }
    let ack: IpcResponse =
        serde_json::from_str(ack_line.trim_end()).context("invalid subscribe ACK JSON")?;
    if !ack.ok {
        bail!(
            "subscribe rejected: {}",
            ack.message.unwrap_or_else(|| "unknown error".into())
        );
    }

    reader.get_ref().set_read_timeout(None)?;
    loop {
        let mut ev_line = String::new();
        let read = (&mut reader)
            .take((MAX_IPC_LINE_BYTES + 1) as u64)
            .read_line(&mut ev_line)?;
        if read == 0 {
            break;
        }
        if ev_line.len() > MAX_IPC_LINE_BYTES {
            bail!("IPC event exceeded {MAX_IPC_LINE_BYTES} bytes");
        }
        let trimmed = ev_line.trim_end();
        if trimmed.is_empty() {
            continue;
        }
        let ev: IpcEvent =
            serde_json::from_str(trimmed).context("invalid NDJSON event from daemon")?;
        if !on_event(ev)? {
            break;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ndjson_request_response_and_events_roundtrip() {
        let mut textures = BTreeMap::new();
        textures.insert(
            "u_depth".to_string(),
            Some(TextureSpec {
                path: PathBuf::from("/tmp/depth.png"),
                filter: TextureFilter::Linear,
                wrap: TextureWrap::ClampToEdge,
                flip_y: false,
            }),
        );
        textures.insert("u_old".to_string(), None);

        let mut uniforms = BTreeMap::new();
        uniforms.insert("u_pointer".to_string(), UniformValue::Vec2([0.3, 0.7]));
        uniforms.insert("u_parallaxStrength".to_string(), UniformValue::Float(0.05));

        let req = IpcRequest::Apply {
            output: "DP-1".to_string(),
            revision: Some(42),
            media: Some(ApplyMedia {
                target: "/tmp/test.png".to_string(),
                fit: Some(FitMode::Cover),
                fps: None,
                hwaccel: None,
            }),
            shader: Some("/tmp/parallax.frag".to_string()),
            textures,
            uniforms,
            transition: Some(ApplyTransition {
                name: Some("fade".to_string()),
                duration_ms: Some(300),
            }),
            wait_commit: true,
        };
        let json = serde_json::to_string(&req).unwrap();
        let parsed: IpcRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, req);

        let ev = IpcEvent::FrameCommitted {
            output: "DP-1".to_string(),
            revision: 42,
            frame: 9,
        };
        let ev_json = serde_json::to_string(&ev).unwrap();
        let parsed_ev: IpcEvent = serde_json::from_str(&ev_json).unwrap();
        assert_eq!(parsed_ev, ev);
    }

    #[test]
    fn ipc_hub_handles_partial_reads_pipelining_and_subscriptions() {
        let (server_sock, mut client_sock) = UnixStream::pair().unwrap();
        let (sub_server, sub_client) = UnixStream::pair().unwrap();
        sub_client.set_nonblocking(false).unwrap();

        let mut hub = IpcHub::new();
        let (id1, _src1) = hub.accept_stream(server_sock).unwrap();
        let (id_sub, _src_sub) = hub.accept_stream(sub_server).unwrap();

        // Register subscription on id_sub.
        hub.handle_raw_line(
            id_sub,
            r#"{"op":"subscribe","output":"DP-1","events":["frame_committed"]}"#,
            |_, _| unreachable!(),
        );
        let mut sub_reader = BufReader::new(sub_client);
        let mut sub_ack = String::new();
        sub_reader.read_line(&mut sub_ack).unwrap();
        assert!(serde_json::from_str::<IpcResponse>(&sub_ack).unwrap().ok);

        // Write a partial request, then the remainder + a second pipelined request.
        client_sock.write_all(br#"{"op":"sta"#).unwrap();
        let (lines_a, remove_a) = hub.poll_client_lines(
            id1,
            Readiness {
                readable: true,
                writable: false,
                error: false,
            },
        );
        assert!(lines_a.is_empty());
        assert!(!remove_a);

        client_sock
            .write_all(b"tus\"}\n{\"op\":\"outputs\"}\n")
            .unwrap();
        let (lines_b, _) = hub.poll_client_lines(
            id1,
            Readiness {
                readable: true,
                writable: false,
                error: false,
            },
        );
        assert_eq!(lines_b.len(), 2);

        let mut seen_ops = Vec::new();
        for line in &lines_b {
            hub.handle_raw_line(id1, line, |_, req| {
                seen_ops.push(req);
                DispatchOutcome::Reply(IpcResponse::ok_msg("ok"))
            });
        }
        assert_eq!(seen_ops, vec![IpcRequest::Status, IpcRequest::Outputs]);

        let mut cli_reader = BufReader::new(client_sock);
        let mut l1 = String::new();
        let mut l2 = String::new();
        cli_reader.read_line(&mut l1).unwrap();
        cli_reader.read_line(&mut l2).unwrap();
        assert!(serde_json::from_str::<IpcResponse>(&l1).unwrap().ok);
        assert!(serde_json::from_str::<IpcResponse>(&l2).unwrap().ok);

        // Broadcast matching and non-matching events to subscriber.
        hub.broadcast_event(&IpcEvent::FrameCommitted {
            output: "eDP-1".into(),
            revision: 1,
            frame: 1,
        });
        hub.broadcast_event(&IpcEvent::TransitionDone {
            output: "DP-1".into(),
            revision: 1,
        });
        hub.broadcast_event(&IpcEvent::FrameCommitted {
            output: "DP-1".into(),
            revision: 2,
            frame: 5,
        });

        let mut ev_line = String::new();
        sub_reader.read_line(&mut ev_line).unwrap();
        let ev: IpcEvent = serde_json::from_str(ev_line.trim_end()).unwrap();
        assert_eq!(
            ev,
            IpcEvent::FrameCommitted {
                output: "DP-1".into(),
                revision: 2,
                frame: 5,
            }
        );
    }

    #[test]
    fn ipc_client_source_removes_on_disconnect_without_spinning() {
        struct LoopState {
            hub: IpcHub,
            callbacks: usize,
            removed: bool,
        }

        let (server_sock, mut client_sock) = UnixStream::pair().unwrap();
        client_sock.set_nonblocking(false).unwrap();

        let mut event_loop: calloop::EventLoop<'static, LoopState> =
            calloop::EventLoop::try_new().unwrap();
        let handle = event_loop.handle();
        let mut state = LoopState {
            hub: IpcHub::new(),
            callbacks: 0,
            removed: false,
        };

        let (client_id, source) = state.hub.accept_stream(server_sock).unwrap();
        let token = handle
            .insert_source(source, |(id, readiness), _, s: &mut LoopState| {
                s.callbacks += 1;
                let (lines, mut remove) = s.hub.poll_client_lines(id, readiness);
                for line in lines {
                    remove = remove
                        || s.hub.handle_raw_line(id, &line, |_, _| {
                            DispatchOutcome::Reply(IpcResponse::ok_msg("alive"))
                        });
                }
                if remove {
                    s.hub.remove_client(id);
                    s.removed = true;
                    PostAction::Remove
                } else {
                    PostAction::Continue
                }
            })
            .unwrap();
        state.hub.set_client_token(client_id, token);

        client_sock.write_all(b"{\"op\":\"status\"}\n").unwrap();
        event_loop
            .dispatch(Some(Duration::from_millis(200)), &mut state)
            .unwrap();

        let mut resp_line = String::new();
        BufReader::new(&client_sock)
            .read_line(&mut resp_line)
            .unwrap();
        assert!(
            serde_json::from_str::<IpcResponse>(resp_line.trim_end())
                .unwrap()
                .ok
        );
        assert!(!state.removed);

        // Dropping client triggers EOF -> source must return PostAction::Remove and never spin.
        drop(client_sock);
        for _ in 0..3 {
            event_loop
                .dispatch(Some(Duration::from_millis(20)), &mut state)
                .unwrap();
            if state.removed {
                break;
            }
        }
        assert!(state.removed);
        assert!(state.hub.clients.is_empty());

        let count_after_remove = state.callbacks;
        event_loop
            .dispatch(Some(Duration::from_millis(25)), &mut state)
            .unwrap();
        assert_eq!(state.callbacks, count_after_remove);
    }
}
