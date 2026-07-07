[← Back to components](./README.md) · [← Back to root](../../README.md)

# Orchestrator

The orchestrator is the **server layer** that drives a conversational turn. It wraps the
[`Reminh`](./reminh.md) coordination logic, connects to the [edge](./edge-server.md) as a
registered peer, calls the VLM for inference, and fires TTS. Where `Reminh` decides *what* the
prompts are, the orchestrator decides *when* things run and *where* results go.

Built on **FastAPI** + `asyncio`. It runs two interfaces in parallel: the **edge path**
(production) and a set of **legacy direct-WS/HTTP routes** kept for local debugging.

---

## Startup

On launch the orchestrator brings up the whole inference stack in order:

1. **Isolated model servers** (`setup_tmux`) — spins up the VLM (llama.cpp), TTS, and STT as
   separate processes in a tmux session, each pinned to its GPU via `CUDA_VISIBLE_DEVICES`. This
   keeps each model in its own process/GPU rather than co-loaded in the orchestrator.
2. **Handlers** — `VisionLangHandler` (VLM client) and `ReminhLogicHandler` (the turn pipeline).
3. **Reminh** — the persona coordinator (loads deltaEGO, Fuli, prompts).
4. **Edge client** — connects to the edge and registers as role `orchestrator`.

On shutdown it closes the edge connection and **persists memory** (`Fuli.save_db()`), so a clean
stop never loses conversation state.

> The VLM server is launched with the flags that back the project's latency claims:
> `-fa on` (flash attention), `-ctk q8_0 -ctv q8_0` (**KV-cache quantized to q8**),
> `--ctx-size 131072` (128K context), `--n-gpu-layers -1` (fully offloaded), thinking disabled.
> See [design-decisions ADR-03](../design-decisions.md).

---

## The VLM

Inference runs on a **vision-language model** (Qwen3-VL-class, served via llama.cpp), not a
text-only LLM. Both stages of a turn — the VAD emotion read and the in-character response — use
the **same VLM**, and the pipeline accepts an optional image (`image_base64`) alongside text, so
a turn can be multimodal. Throughout the docs "VLM" refers to this model.

---

## Edge message dispatch

The orchestrator registers as `orchestrator` and listens for edge messages. `on_edge_message`
dispatches by type:

| Incoming (from edge) | Action |
|---|---|
| `stt_result` | Speech transcribed by STT → run the inference pipeline on the text |
| `user_input` | Text (optionally image) typed by the Unity client → run the pipeline |
| _other_ | Logged as unhandled |

Both entry points build a `ReminhInferenceRequest` and call `handle_main_inference_edge`.

---

## Turn pipeline (edge path)

`handle_main_inference_edge` is the production path. It runs the two-stage pipeline and pushes
results back through the edge:

1. **VAD inference.** `Remi.get_VAD_prompt()` builds the emotion-read system prompt; the VLM is
   called with the user text. Output is the raw VAD vector. If generation fails, an `error` is
   routed to the client and the turn aborts.
2. **Reminh inference.** `Remi.get_Reminh_prompt(vad_raw, user_text)` internally refines the VAD
   through deltaEGO and retrieves memory (see [reminh.md](./reminh.md)), producing the
   in-character system prompt. The refined emotion is snapshotted
   (`get_Reminh_last_emotion().copy()`). The VLM is called a second time for the actual reply.
3. **Emit result.** The response text + captured emotion are routed to the client as
   `inference_result` (`edge.send_routed("inference_result", …)`).
4. **Fire-and-forget TTS.** A `tts_request` is routed to the TTS server with the response text
   and the full TTS parameter set — **without awaiting audio**. The orchestrator's turn ends
   here; TTS streams audio chunks straight to the client through the edge, never back through the
   orchestrator.

The fire-and-forget split is the key latency decision: the orchestrator doesn't sit blocked
relaying audio. Once it has handed text to TTS, it's free — TTS → edge → client is a separate
flow.

```
stt_result / user_input
        │
        ▼
   [1] VAD prompt ──▶ VLM ──▶ raw VAD
        │
        ▼
   [2] Reminh prompt (deltaEGO refine + RAG) ──▶ VLM ──▶ reply
        │
        ├──▶ [3] inference_result ─▶ edge ─▶ client
        └──▶ [4] tts_request ─▶ edge ─▶ TTS ─▶ (audio) ─▶ client
```

---

## Edge client

`EdgeClient` is a thin WebSocket wrapper:

- **`connect()`** — opens the socket and immediately sends `{type: "register", role}`, then waits
  for the `registered` ack (the edge's [registration-first](./edge-server.md#connection-lifecycle)
  contract).
- **`send_routed(msg_type, payload, to=None)`** — sends a routed envelope. With no `to`, the edge
  resolves the destination from its routing table; with `to`, it's addressed explicitly (used for
  `error → client`).
- **`start_listener(handler)`** — spawns an async task relaying inbound edge messages to
  `on_edge_message`.

---

## Legacy / debug interfaces

Alongside the edge path, the orchestrator keeps direct interfaces for development:

- **`/ws/reminh`** — a direct Unity↔orchestrator WebSocket (pre-edge). Uses `handle_main_inference`,
  which is the same pipeline but **relays TTS audio itself** (`_stream_tts`) instead of
  fire-and-forget. Useful for testing without the edge.
- **`/stt/callback`** — a legacy HTTP callback from STT.
- **`/discord/chat`** — Discord text path (HTTP, no TTS). Reuses the two-stage pipeline with the
  `discord_txt` prompt and persists memory. Shows the persona working across surfaces.
- **`/admin/reload`** — hot-reloads the prompt YAML.
- Requests are validated with Pydantic (`ReminhInferenceRequest`, `STTRequest`, `TTSRequest`,
  `MainAiRequest`, `ReloadYamlRequest`) via a discriminated `TypeAdapter`.

> The legacy WS path and the edge path share one pipeline body; only the transport and the TTS
> handling differ (self-relay vs. fire-and-forget). This is why the edge migration didn't require
> rewriting the inference logic.

---

## TTS request parameters

The orchestrator owns the TTS streaming parameters (env-overridable), passed on every
`tts_request`: speaker, language, `pcm_s16le` format, and chunk-timing controls
(`emit_every_frames`, `decode_window_frames`, `overlap_samples`, `first_emit_frames`). See
[tts.md](./tts.md) for what these mean on the synthesis side.

---

## Note on the README branch

This README lives on an earlier branch: the `setup_tmux` script still launches STT on the main
node (and that block is commented out). In the current system **STT runs on the subnode's RTX
3050** — see [infrastructure.md](../infrastructure.md#placement-rationale). The startup ordering
above is otherwise accurate.

---

## Related

- Turn logic / prompts: [reminh.md](./reminh.md)
- Routing plane it registers with: [edge-server.md](./edge-server.md)
- Why two VLM calls, fire-and-forget: [design-decisions.md](../design-decisions.md)