use std::net::SocketAddr;
use std::sync::Arc;

use dashmap::DashMap;
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::Message;
use tracing::{info, warn};

use crate::routing::Route;
use crate::stt::SttClient;

// ──────────────────────────────────────────────
//  Protocol: first message must be Register
// ──────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    Client,       // Unity frontend
    Stt,          // STT server
    Orchestrator, // main inference
    Tts,          // TTS server
}

#[derive(Debug, Deserialize)]
pub struct Envelope {
    #[serde(rename = "type")]
    pub msg_type: String,

    /// Registration
    pub role: Option<Role>,

    /// Routing target (for explicit routing)
    pub to: Option<Role>,

    /// Actual payload (forwarded as-is)
    pub payload: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
#[allow(dead_code)] // protocol type; messages are currently built inline via serde_json::json!
pub struct OutEnvelope {
    #[serde(rename = "type")]
    pub msg_type: String,
    pub from: Role,
    pub payload: serde_json::Value,
}

// ──────────────────────────────────────────────
//  Peer handle: channel-based sender
// ──────────────────────────────────────────────

#[derive(Clone)]
pub struct PeerHandle {
    pub id: uuid::Uuid,
    #[allow(dead_code)] // retained as peer metadata; not read yet
    pub role: Role,
    pub tx: mpsc::UnboundedSender<Message>,
}

// ──────────────────────────────────────────────
//  Hub
// ──────────────────────────────────────────────

#[derive(Clone)]
pub struct Hub {
    /// role -> list of connected peers (supports multiple clients)
    peers: Arc<DashMap<Role, Vec<PeerHandle>>>,

    /// HTTP STT backend. When set, client audio is buffered here and POSTed on
    /// `audio_end` instead of being streamed to a WebSocket STT peer.
    stt: Option<Arc<SttClient>>,
}

impl Hub {
    pub fn new() -> Self {
        let stt = SttClient::from_env().map(Arc::new);
        if stt.is_none() {
            info!("STT_URL not set: client audio is routed to a WebSocket STT peer");
        }
        Self {
            peers: Arc::new(DashMap::new()),
            stt,
        }
    }

    pub async fn handle_connection(
        &self,
        ws: tokio_tungstenite::WebSocketStream<tokio::net::TcpStream>,
        peer: SocketAddr,
    ) {
        let (mut ws_tx, mut ws_rx) = ws.split();
        let (tx, mut rx) = mpsc::unbounded_channel::<Message>();

        // ── Phase 1: Wait for registration ──
        let role = match self.wait_for_registration(&mut ws_rx, &peer).await {
            Some(r) => r,
            None => {
                warn!("[{peer}] No valid registration, dropping");
                return;
            }
        };

        info!("[{peer}] Registered as {role:?}");

        // ── Register peer ──
        let peer_id = uuid::Uuid::new_v4();
        let handle = PeerHandle {
            id: peer_id,
            role,
            tx: tx.clone(),
        };
        self.peers.entry(role).or_default().push(handle);

        // ── Send ack ──
        let ack = serde_json::json!({
            "type": "registered",
            "role": role,
        });
        let _ = tx.send(Message::Text(ack.to_string().into()));

        // ── Outbound task: channel → WS ──
        let out_task = tokio::spawn(async move {
            while let Some(msg) = rx.recv().await {
                if ws_tx.send(msg).await.is_err() {
                    break;
                }
            }
        });

        // Per-connection utterance buffer (HTTP STT mode, clients only)
        let http_stt = if role == Role::Client { self.stt.clone() } else { None };
        let mut audio_buf: Vec<u8> = Vec::new();
        let mut overflow_warned = false;

        // ── Inbound loop: WS → route ──
        while let Some(Ok(msg)) = ws_rx.next().await {
            match msg {
                Message::Text(text) => {
                    if let Some(stt) = &http_stt {
                        if self.handle_audio_control(&text, stt, &mut audio_buf, &tx, &peer) {
                            overflow_warned = false;
                            continue;
                        }
                    }
                    self.handle_text_message(&text, role, &peer);
                }
                Message::Binary(data) => {
                    if let Some(stt) = &http_stt {
                        if audio_buf.len() + data.len() <= stt.max_buffer_bytes {
                            audio_buf.extend_from_slice(&data);
                        } else if !overflow_warned {
                            warn!("[{peer}] Audio buffer full, dropping audio until audio_end");
                            overflow_warned = true;
                        }
                        continue;
                    }
                    self.handle_binary_message(data.to_vec(), role, &peer);
                }
                Message::Close(_) => break,
                _ => {}
            }
        }

        // ── Cleanup ──
        self.remove_peer(role, peer_id);
        out_task.abort();
        info!("[{peer}] {role:?} disconnected");
    }

    // ──────────────────────────────────────────────
    //  Registration handshake
    // ──────────────────────────────────────────────

    async fn wait_for_registration(
        &self,
        rx: &mut futures_util::stream::SplitStream<
            tokio_tungstenite::WebSocketStream<tokio::net::TcpStream>,
        >,
        peer: &SocketAddr,
    ) -> Option<Role> {
        // Give 5 seconds to register
        let timeout = tokio::time::timeout(std::time::Duration::from_secs(5), rx.next()).await;

        match timeout {
            Ok(Some(Ok(Message::Text(text)))) => {
                let env: Envelope = serde_json::from_str(&text).ok()?;
                if env.msg_type == "register" {
                    env.role
                } else {
                    warn!("[{peer}] First message was not register");
                    None
                }
            }
            _ => {
                warn!("[{peer}] Registration timeout or error");
                None
            }
        }
    }

    // ──────────────────────────────────────────────
    //  HTTP STT: audio_start / audio_end
    // ──────────────────────────────────────────────

    /// Returns true if the message was an audio control message and has been handled.
    fn handle_audio_control(
        &self,
        text: &str,
        stt: &Arc<SttClient>,
        audio_buf: &mut Vec<u8>,
        client_tx: &mpsc::UnboundedSender<Message>,
        peer: &SocketAddr,
    ) -> bool {
        let Ok(env) = serde_json::from_str::<Envelope>(text) else {
            return false;
        };

        match env.msg_type.as_str() {
            "audio_start" => {
                audio_buf.clear();
                true
            }
            "audio_end" => {
                let pcm = std::mem::take(audio_buf);
                if pcm.is_empty() {
                    warn!("[{peer}] audio_end with no buffered audio");
                    return true;
                }

                let secs = pcm.len() as f32 / (stt.sample_rate as f32 * 2.0);
                info!("[{peer}] audio_end: {secs:.2}s of audio → STT");

                let hub = self.clone();
                let stt = stt.clone();
                let client_tx = client_tx.clone();
                let peer = *peer;
                tokio::spawn(async move {
                    hub.run_stt(stt, pcm, client_tx, peer).await;
                });
                true
            }
            _ => false,
        }
    }

    async fn run_stt(
        &self,
        stt: Arc<SttClient>,
        pcm: Vec<u8>,
        client_tx: mpsc::UnboundedSender<Message>,
        peer: SocketAddr,
    ) {
        match stt.transcribe(pcm).await {
            Ok(text) => {
                info!("[{peer}] STT: {text:?}");

                // Transcript back to the requesting client (for display)
                let result = serde_json::json!({
                    "type": "stt_result",
                    "from": Role::Stt,
                    "payload": { "text": text },
                });
                let _ = client_tx.send(Message::Text(result.to_string().into()));

                if text.is_empty() {
                    return;
                }

                // Hand off to the orchestrator as a normal user_input
                let input = serde_json::json!({
                    "type": "user_input",
                    "from": Role::Client,
                    "payload": { "text": text, "source": "stt" },
                });
                self.send_to_role(Role::Orchestrator, Message::Text(input.to_string().into()));
            }
            Err(e) => {
                warn!("[{peer}] {e}");
                let err = serde_json::json!({
                    "type": "error",
                    "from": Role::Stt,
                    "payload": { "message": e },
                });
                let _ = client_tx.send(Message::Text(err.to_string().into()));
            }
        }
    }

    // ──────────────────────────────────────────────
    //  Message handling
    // ──────────────────────────────────────────────

    fn handle_text_message(&self, text: &str, from: Role, peer: &SocketAddr) {
        let env: Envelope = match serde_json::from_str(text) {
            Ok(e) => e,
            Err(e) => {
                warn!("[{peer}] Invalid JSON: {e}");
                return;
            }
        };

        // Explicit routing: "to" field
        if let Some(target) = env.to {
            let out = serde_json::json!({
                "type": env.msg_type,
                "from": from,
                "payload": env.payload,
            });
            self.send_to_role(target, Message::Text(out.to_string().into()));
            return;
        }

        // Implicit routing: use routing table
        if let Some(target) = Route::resolve_text(from, &env.msg_type) {
            let out = serde_json::json!({
                "type": env.msg_type,
                "from": from,
                "payload": env.payload,
            });
            self.send_to_role(target, Message::Text(out.to_string().into()));
        } else {
            warn!("[{peer}] No route for {from:?} msg_type={}", env.msg_type);
        }
    }

    fn handle_binary_message(&self, data: Vec<u8>, from: Role, peer: &SocketAddr) {
        // Binary routing: use static routing table
        if let Some(target) = Route::resolve_binary(from) {
            self.send_to_role(target, Message::Binary(data.into()));
        } else {
            warn!("[{peer}] No binary route for {from:?}");
        }
    }

    // ──────────────────────────────────────────────
    //  Send helpers
    // ──────────────────────────────────────────────

    fn send_to_role(&self, role: Role, msg: Message) {
        if let Some(peers) = self.peers.get(&role) {
            for handle in peers.iter() {
                if handle.tx.send(msg.clone()).is_err() {
                    warn!("Failed to send to {role:?} peer (channel closed)");
                }
            }
        } else {
            warn!("No peers registered for {role:?}");
        }
    }

    fn remove_peer(&self, role: Role, id: uuid::Uuid) {
        if let Some(mut peers) = self.peers.get_mut(&role) {
            peers.retain(|h| h.id != id);
            if peers.is_empty() {
                drop(peers);
                self.peers.remove(&role);
            }
        }
    }
}
