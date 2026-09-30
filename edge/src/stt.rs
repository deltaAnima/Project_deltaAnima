use std::time::Duration;

use serde_json::Value;

/// HTTP client for the STT server.
///
/// The edge buffers raw PCM16 LE mono audio from a client, then on `audio_end`
/// POSTs it here and relays the transcript.
///
/// Env:
///   STT_URL           full endpoint, e.g. http://127.0.0.1:8080/transcribe  (unset → HTTP STT disabled)
///   STT_FORMAT        multipart (default) | wav | pcm
///   STT_FILE_FIELD    multipart field name (default "file")
///   STT_TEXT_FIELD    JSON response field holding the transcript (default: text/transcript/result)
///   STT_SAMPLE_RATE   sample rate of incoming PCM (default 16000)
///   STT_TIMEOUT_SECS  request timeout (default 30)
///   STT_MAX_SECONDS   max buffered audio per utterance (default 60)
pub struct SttClient {
    http: reqwest::Client,
    url: String,
    format: RequestFormat,
    file_field: String,
    text_field: Option<String>,
    pub sample_rate: u32,
    /// Max buffered audio per utterance (bytes). Guards against a client that never sends audio_end.
    pub max_buffer_bytes: usize,
}

#[derive(Debug, Clone, Copy)]
enum RequestFormat {
    /// multipart/form-data with a WAV file part
    Multipart,
    /// raw WAV body (Content-Type: audio/wav)
    Wav,
    /// raw PCM16 LE body (Content-Type: application/octet-stream)
    Pcm,
}

impl SttClient {
    pub fn from_env() -> Option<Self> {
        let url = std::env::var("STT_URL").ok().filter(|u| !u.is_empty())?;

        let format = match std::env::var("STT_FORMAT").unwrap_or_default().as_str() {
            "wav" => RequestFormat::Wav,
            "pcm" => RequestFormat::Pcm,
            _ => RequestFormat::Multipart,
        };
        let file_field = std::env::var("STT_FILE_FIELD").unwrap_or_else(|_| "file".into());
        let text_field = std::env::var("STT_TEXT_FIELD").ok().filter(|f| !f.is_empty());
        let sample_rate = env_parse("STT_SAMPLE_RATE", 16000);
        let timeout = env_parse("STT_TIMEOUT_SECS", 30);
        let max_secs: usize = env_parse("STT_MAX_SECONDS", 60);

        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(timeout))
            .build()
            .expect("Failed to build HTTP client");

        tracing::info!("HTTP STT enabled: {url} ({format:?}, {sample_rate} Hz)");

        Some(Self {
            http,
            url,
            format,
            file_field,
            text_field,
            sample_rate,
            max_buffer_bytes: sample_rate as usize * 2 * max_secs,
        })
    }


    pub async fn transcribe(&self, pcm: Vec<u8>) -> Result<String, String> {
        let req = match self.format {
            RequestFormat::Multipart => {
                let part = reqwest::multipart::Part::bytes(wav_bytes(&pcm, self.sample_rate))
                    .file_name("audio.wav")
                    .mime_str("audio/wav")
                    .map_err(|e| e.to_string())?;
                let form = reqwest::multipart::Form::new().part(self.file_field.clone(), part);
                self.http.post(&self.url).multipart(form)
            }
            RequestFormat::Wav => self
                .http
                .post(&self.url)
                .header(reqwest::header::CONTENT_TYPE, "audio/wav")
                .body(wav_bytes(&pcm, self.sample_rate)),
            RequestFormat::Pcm => self
                .http
                .post(&self.url)
                .header(reqwest::header::CONTENT_TYPE, "application/octet-stream")
                .body(pcm),
        };

        let resp = req.send().await.map_err(|e| format!("STT request failed: {e}"))?;
        let status = resp.status();
        let body = resp
            .text()
            .await
            .map_err(|e| format!("STT response read failed: {e}"))?;

        if !status.is_success() {
            let snippet: String = body.chars().take(300).collect();
            return Err(format!("STT HTTP {status}: {snippet}"));
        }

        Ok(self.extract_text(&body))
    }

    /// Accepts `{"text": "..."}`-style JSON, a bare JSON string, or plain text.
    fn extract_text(&self, body: &str) -> String {
        let Ok(v) = serde_json::from_str::<Value>(body) else {
            return body.trim().to_owned();
        };
        if let Some(s) = v.as_str() {
            return s.trim().to_owned();
        }
        let found = match &self.text_field {
            Some(f) => v.get(f).and_then(Value::as_str),
            None => ["text", "transcript", "result"]
                .iter()
                .find_map(|k| v.get(*k).and_then(Value::as_str)),
        };
        found.unwrap_or_default().trim().to_owned()
    }
}

fn env_parse<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Wrap PCM16 LE mono in a 44-byte WAV header.
fn wav_bytes(pcm: &[u8], sample_rate: u32) -> Vec<u8> {
    let data_len = pcm.len() as u32;
    let byte_rate = sample_rate * 2;
    let mut out = Vec::with_capacity(44 + pcm.len());
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes()); // block align
    out.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    out.extend_from_slice(pcm);
    out
}
