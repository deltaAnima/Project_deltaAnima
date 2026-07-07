[← Back to components](./README.md) · [← Back to root](../../README.md)

# Reminh — persona backend

`Reminh` is the persona backend: the class that coordinates a full conversational turn. It owns
the emotion engine (deltaEGO), the memory system (Fuli), and prompt assembly (PromptHandler),
and drives the **two-stage emotion pipeline** described in
[architecture.md](../architecture.md). This document covers how those pieces are wired and the
turn logic; the memory layer has its own section at the end.

> Scope note: `Reminh` is the coordination logic. The network/server layer that receives text
> from the edge and calls these methods is the [orchestrator](./orchestrator.md); the two are
> often referred to together as "the orchestrator."

---

## Responsibilities

`Reminh` holds three in-process modules and sequences them per turn:

| Module | Type | Purpose |
|---|---|---|
| `EmotionModule` | `delta_ego_core.deltaEGO` (C++ / pybind11) | Refine a raw VAD vector into a concrete emotion |
| `MemoryHandler` | `Fuli` (Python) | Store and retrieve conversation memory (RAG) |
| `PromptModule` | `PromptHandler` (Python) | Assemble the VAD-guess and response prompts |

All three are constructed in `__init__` from YAML config, so persona traits, memory sizing, and
emotion physics are data-driven rather than hardcoded.

---

## Configuration

Three YAML files drive the persona, loaded at construction:

- **`Reminh_config.yaml`** — memory sizing (`queue_len`, `top_k`, `update_cnt`) and personality:
  the **OCEAN** traits (O/C/E/A/N) and a **base VAD area** (V/A/D + `radius`) that anchors the
  persona's default emotional center.
- **`Reminh_physics.yaml`** — the deltaEGO physics parameters (passed to the C++ engine;
  hot-reloadable via `reload_physics_config()`).
- **`ReminhPrompt.yaml`** — the persona text (core/appearance/reactions), guidelines, per-channel
  constraints, and the VAD-inference system template. Loaded by `PromptHandler`.

The OCEAN traits and base-VAD area mean the persona has a **default emotional disposition** — the
deltaEGO engine is seeded with where this character sits emotionally at rest, not a neutral zero.

---

## Turn logic

A turn spans two entry points on `Reminh`, one per VLM call, plus a persist step.

### 1. `get_VAD_prompt(user_text)` — build the emotion-guess prompt

Delegates to `PromptHandler.get_vad_prompt(user_text)`, which returns the **persona reference**
plus a **VAD-inference system template** with the user's text interpolated in. This is the prompt
for the first (emotion-reading) VLM call.

The persona reference here is intentionally minimal — core/appearance/reactions only, no
retrieved memories. The emotional read is of the *present* moment (see
[design-decisions ADR-02](../design-decisions.md)).

### 2. `get_Reminh_prompt(VAD_result_raw, user_input, source, user_name)` — refine + build response prompt

This is where the two-stage pipeline happens, in order:

1. **Parse VAD.** `_parse_vad_json()` extracts the VAD JSON from the VLM's raw output. It is
   defensive — it strips ``` ``` fences and, if the model mixed narrative with JSON, regex-
   extracts the first `{...}` block. On any failure it falls back to a neutral `(0, 0, 0)` rather
   than throwing, so a malformed VLM response degrades gracefully instead of crashing the turn.
2. **Refine emotion (deltaEGO).** The parsed V/A/D is passed to
   `EmotionModule.process_stimulus(V, A, D)`, the C++ engine, which returns an emotion JSON. The
   `emotion_term` is extracted (falling back to `calm` if missing) and cached in `last_emotion`.
3. **Retrieve memory (Fuli).** `MemoryHandler.retrieve(user_input)` returns relevant memories —
   note this runs *after* emotion, so memory conditions only the response, never the emotional
   read.
4. **Assemble prompt.** `PromptModule` builds the final in-character prompt with the retrieved
   memories and the refined mood, branched by `source` (`unity` vs `discord_txt`).

### 3. `set_Reminh_memory(User_Name, AI_output)` — persist the turn

After the response is generated, the turn is written to Fuli via `add_memory()`, carrying the
user input, the AI response, the emotion term (`AI_status`), and the **full deltaEGO analysis**
(`deltaEGO_analysis`). Emotion is stored with the memory, so it accumulates across turns rather
than being discarded.

---

## Multi-channel prompts

`PromptHandler` produces different prompts per surface from the same persona data:

- **`get_reminh_prompt`** (Unity/TTS) — identity block, constraints/examples, dynamic context
  (memories + mood), and a final directive tuned for spoken, short, soft replies. Explicit
  anti-patterns are baked in ("don't act like a poem bot," "focus on the current question").
- **`get_discord_Text_prompt`** (Discord text) — adds an **expert-mode switch**: on CS/technical
  questions the persona drops the lyrical style and answers precisely in Markdown, and is told to
  **ignore RAG memories** when they're off-topic for the current question.
- **`get_vad_prompt`** (emotion read) — persona reference + VAD-inference system template with
  `{user_text}` interpolated.

The design keeps one persona definition and adapts *presentation* per channel, rather than
maintaining separate personas.

---

## Memory layer (Fuli)

`Fuli` is the RAG module — internally described as "RAG 1.0 but better." It is a **two-tier**
memory system, not a flat vector store.

### Tiers

1. **Recent queue** — a fixed-length `deque` (default 10) holding raw recent turns. Always
   returned in full, preserving exact recent conversational flow.
2. **Long-term vector store** — when the queue overflows, the oldest turn is popped, embedded,
   and added to a FAISS index. Retrieval is top-k by cosine similarity.

### Embedding + search

- **Model:** bge-m3, dense 1024-dim, L2-normalized. Runs on `cuda:1` (the RTX 3090) — see
  [infrastructure.md](../infrastructure.md#placement-rationale) for why embedding is pinned there.
- **Index:** `faiss.IndexFlatIP` (inner product on normalized vectors = cosine similarity).
- **Threshold:** retrieved memories are kept only if cosine **≥ 0.7**, filtering weak matches.
- **Embedded text:** both sides of a turn are embedded together (`User: … AI: …`) so retrieval
  matches on the full exchange, not just the user line.

### Retrieval output

`retrieve()` returns a formatted blend: a `[Reminh's Past Memories]` block (long-term hits, with
date and the emotion tag the AI held at the time) followed by a `[Recent Conversation Flow]`
block (the queue). The response prompt therefore sees both durable context and immediate flow.

### Persistence

FAISS index and memories are written to disk (`.index` + `.json`) every `mem_update_cnt` new
long-term memories, and reloaded on startup — so memory survives restarts. Memories are typed
with Pydantic (`GeneralMem`, `dialog`, `StateTokens`), carrying `impressiveness` and
stress/reward `state_tokens` alongside the text.

> Naming: `Fuli` and `Reminh` are references from *Honkai: Star Rail*.

---

## Related

- Turn flow in context: [architecture.md](../architecture.md#request-lifecycle)
- Why two-stage emotion / persona-only read: [design-decisions.md](../design-decisions.md)
- Emotion engine internals: [deltaego.md](./deltaego.md)