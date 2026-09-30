use crate::hub::Role;

/// Static routing rules for the delta-edge hub.
///
/// Text messages can route by (sender_role, msg_type).
/// Binary messages route by sender_role only.
///
///  STT  ──text──→  Orchestrator
///  Orchestrator ──inference_result──→  Client
///  Orchestrator ──tts_request──→  Tts
///  Tts  ──tts_start/tts_done──→  Client
///  Tts  ──binary(audio)──→  Client
pub struct Route;

impl Route {
    /// Resolve text message destination based on sender + message type.
    pub fn resolve_text(from: Role, msg_type: &str) -> Option<Role> {
        match (from, msg_type) {
            // STT finished transcription → orchestrator
            (Role::Stt, "stt_result") => Some(Role::Orchestrator),

            // Orchestrator sends inference result → client (Unity)
            (Role::Orchestrator, "inference_result") => Some(Role::Client),

            // Orchestrator fires TTS request → TTS server
            (Role::Orchestrator, "tts_request") => Some(Role::Tts),

            // TTS metadata → client
            (Role::Tts, "tts_start") => Some(Role::Client),
            (Role::Tts, "tts_done") => Some(Role::Client),

            // Client sends input → orchestrator
            (Role::Client, "user_input") => Some(Role::Orchestrator),

            // Client sends audio → STT
            (Role::Client, "audio_input") => Some(Role::Stt),

            _ => None,
        }
    }

    /// Resolve binary message destination based on sender.
    pub fn resolve_binary(from: Role) -> Option<Role> {
        match from {
            // TTS audio chunks → client
            Role::Tts => Some(Role::Client),

            // Client raw audio → STT
            Role::Client => Some(Role::Stt),

            _ => None,
        }
    }
}
