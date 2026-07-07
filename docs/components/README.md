[← Back to root](../../README.md)

# Components

Per-service deep dives for deltaAnima. Each document covers one component's internals — what it
does, how it's built, and the decisions behind it. For how they fit together in a turn, start
with [architecture.md](../architecture.md).

---

## Request path

The components below, in the order a turn touches them:

| Component | Language | Role |
|---|---|---|
| [Edge server](./edge-server.md) | Rust | WebSocket routing plane — role-based registration, static routing |
| [STT](./stt.md) | Python | Event-based Whisper transcription (subnode, RTX 3050) |
| [Orchestrator](./orchestrator.md) | Python | Drives the turn — edge client, two-stage VLM calls, fire-and-forget TTS |
| [Reminh](./reminh.md) | Python / C++ | Persona backend — prompt assembly, emotion + memory coordination |
| [deltaEGO](./deltaego.md) | C++ (AVX-512) | Deterministic emotion engine — SIMD emotion search + personality physics |
| [TTS](./tts.md) | Python | Finetuned Qwen3-TTS streaming synthesis |

Reminh's memory layer (**Fuli**, RAG) is documented inside [reminh.md](./reminh.md#memory-layer-fuli)
rather than as a separate file.

---

## Boundaries at a glance

Not every component is a network service. The orchestrator hosts two of the most important pieces
**in-process**:

- **Network services** (routed via the edge): client, STT, orchestrator, TTS. The orchestrator
  calls the **VLM** (llama.cpp) directly.
- **In-process modules** (inside the orchestrator): **deltaEGO** (C++ via pybind11) and **Fuli**
  (Python RAG).

See [architecture.md](../architecture.md#system-model) for the full boundary table.

---

## Status

| Doc | Owner | State |
|---|---|---|
| edge-server.md | Justin | ✅ |
| orchestrator.md | Justin | ✅ |
| reminh.md | Justin | ✅ |
| deltaego.md | Justin | ✅ |
| stt.md | Mark | ⬜ authored by STT owner |
| tts.md | Jay | ⬜ authored by TTS owner |

STT and TTS were built by teammates ([contributions](../../README.md#the-team--technical-contributions));
their component docs are best written by their authors.