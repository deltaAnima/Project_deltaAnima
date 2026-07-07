[← Back to README](../README.md)

# Design decisions

This document records the **why** behind deltaAnima's architecture in ADR (Architecture
Decision Record) style. Each entry states the decision, the alternative that was rejected, the
reasoning, and what was traded away. Facts about *how* the system is built live in
[architecture.md](./architecture.md) and [infrastructure.md](./infrastructure.md); this file is
only about the choices.

---

## Guiding principle

> **Build the most emotionally realistic AI persona possible.**

Most of the decisions below follow from this one goal. The persona should not feel like a
sentiment classifier emitting "happy / sad / angry." It should hold nuanced, in-between states —
*flustered, wanting to hide, faintly hopeful* — and express them in the moment. That ambition
drives the two-stage emotion model, the deliberate exclusion of memory from the emotional read,
and the willingness to spend a second VLM call. The rest is performance and ownership.

---

## ADR-01 — Two-stage emotion: VLM guesses VAD, deltaEGO refines

**Decision.** The VLM produces a 3-D **VAD** (Valence-Arousal-Dominance) vector, which is then
passed through deltaEGO — a C++ engine — to resolve a precise emotion rather than letting the
VLM name the emotion directly.

**Rejected alternative.** Ask the VLM to output an emotion word ("sadness") in one shot.

**Why.** This inherits the philosophy of deltaEGO v1 (the original k-d tree module): a persona
should hold *detailed* emotion, not one-dimensional labels. Representing emotion as a point in a
continuous 3-D space means it can be controlled with **mathematical algorithms** — nearest-
neighbor search, physics weighting — to land on fine-grained, realistic states instead of a
fixed vocabulary. A VAD vector is also tunable and reproducible in a way a free-text label is
not.

**Tradeoff.** More moving parts and an extra C++ dependency in the path. A single VLM label
would be simpler, but it caps emotional resolution at whatever words the model reaches for.

---

## ADR-02 — Exclude memory (RAG) from the emotional read

**Decision.** The VAD-guess prompt contains the persona and the user input only. Retrieved
memories are **not** included at this stage.

**Rejected alternative.** Feed RAG context into the emotion guess too.

**Why.** Human emotion is treated here as largely **immediate and one-dimensional in the
moment** — a reaction to *what just happened*, not a weighted average of history. To capture
which emotion the persona feels *right now*, the read is kept clean of past context that would
otherwise drag the response toward prior states.

**Tradeoff.** The emotional read loses long-term continuity — the persona reacts fresh each
time rather than carrying a slow-moving mood. That is an accepted, deliberate consequence of
prioritizing in-the-moment reactivity. (Memory still conditions the *response* in stage 4.)

---

## ADR-03 — Two separate VLM calls (emotion, then response)

**Decision.** A turn issues two VLM calls: one to read emotion, one to generate the reply.

**Rejected alternative.** A single call that decides emotion and responds at once, saving tokens
and a round-trip.

**Why.** The fine-grained emotion control in ADR-01 requires the emotion to be resolved
*before* the response is generated, so it can condition that response. Given the project's motto
— the most realistic emotion possible — the second call is essential, not optional.

**Tradeoff.** Roughly double the inference cost and added latency. This is mitigated rather than
avoided: **KV-cache quantization**, **flash attention**, and **minimal token budgets on the
VAD-inference call** keep the two-call design fast enough to be practical.

---

## ADR-04 — TTS receives text only (no emotion params)

**Decision.** Only the response text is sent to TTS. Emotion is baked into the wording upstream,
not passed as a synthesis parameter.

**Why.** It keeps the synthesis boundary simple, and the emotional content already lives in the
generated text.

**Status.** Passing emotion through to TTS is **planned for a future update** — at which point
the refined deltaEGO emotion could drive prosody directly. For now, text-only.

---

## ADR-05 — deltaEGO in C++ with SIMD (not Python)

**Decision.** The emotion engine is implemented in C++ with AVX-512 SIMD, exposed to Python via
pybind11.

**Rejected alternative.** Implement the emotion search in Python.

**Why.** Two reasons, stated honestly. First, **maximum optimization** wherever the design
allows control over it. Second, **deliberate over-engineering as a learning exercise** — the
project is partly a vehicle for going deep on low-level systems. The technical fit is real,
though: KNN over a flat array maps almost perfectly onto SIMD acceleration, so the choice isn't
only academic.

**Tradeoff.** More implementation complexity and a native build in the toolchain versus a few
lines of NumPy. Accepted, because both the performance and the learning were the point.

> Implementation details (k-d tree, AVX-512, int16 quantization) live in
> [components/deltaego.md](./components/deltaego.md). This entry is only about *why C++*.

---

## ADR-06 — Edge server in Rust

**Decision.** The routing/edge layer is written in Rust.

**Why.** The edge has to manage **many concurrent WebSocket connections**, so concurrency
correctness matters — and Rust is well suited to exactly that. Learning Rust was also an
explicit goal.

**Tradeoff.** A steeper language than, say, a Python/Node WebSocket server, in exchange for
concurrency guarantees and the learning investment.

---

## ADR-07 — CPU/GPU placement, and revising it when the environment changed

**Decision.** Assign models to individual GPUs by size and speed; size the CPU allocation to the
host environment. Notably, this decision was **revised** once the environment changed — which is
the interesting part.

**GPU — per-model assignment.** The budget reality is **two different GPU architectures**
(RTX 5090 + RTX 3090), so the way to extract the most from a mixed setup is per-model placement:
the slowest workload (VLM) on the fastest card (5090), smaller models (TTS, FAISS embedding) on
the 3090. `CUDA_DEVICE_ORDER=PCI_BUS_ID` fixes GPU enumeration so this mapping stays stable
across restarts.

**CPU — a decision that flipped.** The original setup ran Windows 11 + WSL2, and the workload was
pinned to a single CCD (**CCD1**) to avoid contending with the Windows host, which leaned on
CCD0. After migrating to native Ubuntu on a **dedicated, project-only machine**, that reasoning
no longer held — there was no competing host to isolate against. So the placement was **reversed**:
instead of confining to one CCD, the container now spans **24 of 32 threads**
(`cpuset: "4-15,20-31"`), reserving two SMT pairs for the OS and using everything else across
both CCDs.

**Why this entry matters.** The point isn't the specific cpuset — it's that **placement is a
function of the environment, not a fixed rule.** Single-CCD isolation was *correct* under a
contended Windows host; the opposite (maximize cores) is correct on a dedicated Linux box.
Recognizing when a prior decision's premise has expired, and reversing it, is the actual skill.

**Tradeoff.** Manual, hardware- and environment-specific tuning that doesn't transfer cleanly to
a homogeneous or cloud setup, and has to be revisited when the environment changes (as it was).
Justified by squeezing maximum efficiency out of a self-funded, mixed-architecture homelab.

> Full placement details, the compose config, and the code reference are in
> [infrastructure.md](./infrastructure.md#placement-rationale).

---

## ADR-08 — Isolate STT on a separate node/GPU

**Decision.** Move STT off the main node onto the subnode, on its own dedicated GPU (RTX 3050),
rather than co-locating it with inference and TTS.

**Rejected alternative.** Run STT on the main node alongside the VLM and TTS, sharing GPUs.

**Why.** STT is **event-based** — it fires whenever speech is detected, at unpredictable times.
Sharing a GPU with the latency-critical path means a burst of STT activity could momentarily
contend for VRAM/compute and slow inference or TTS mid-conversation. Putting STT on separate
hardware on a separate node removes that coupling entirely: a speech-detection spike can never
degrade the main conversational path, because the two never share a resource.

**Tradeoff.** STT now crosses a node boundary (subnode → edge → main node), adding a small amount
of network hop versus a local call. Accepted, because the isolation guarantee is worth more than
the microseconds saved by co-location — predictable main-path latency beats raw locality.

---

## ADR-09 — Migrate to native Linux (drop WSL2)

**Decision.** Move the main node from Windows 11 + WSL2/Docker to **native Ubuntu 24.04**.

**Context.** TTS real-time factor sat above 1.0 — synthesis slower than real time, a hard blocker
for live conversation. The cause was **not** the model or GPU: it was traced to **WSL2's GPU
passthrough (WDDM) kernel-launch overhead**, confirmed by comparing against a native-Linux node
running the same workload.

**Why.** Removing the WDDM passthrough layer was the only fix that addressed the root cause rather
than working around it. Testing TTS on the faster 5090 was tried first, but that treated the
symptom; the environment was the real problem.

**Result.** TTS throughput improved ~**3×**, RTF dropped to **~0.4** on the 3090, and TTS returned
to its designed card. The native, dedicated setup also enabled the CPU placement revision in
ADR-07.

**Tradeoff.** Gave up the convenience of developing inside Windows (and dual-boot/GPU-sharing with
the desktop OS) in exchange for removing an entire latency-inducing abstraction layer. For a
machine that is now project-dedicated, that trade is clearly worth it.

> Full diagnosis writeup in [infrastructure.md](./infrastructure.md#tts-latency-solved).

---

## ADR-10 — Self-hosted remote access via FortiGate (not Tailscale / Cloudflare)

**Decision.** Remote access runs through a FortiGate firewall with a reverse SSH tunnel, instead
of Tailscale or a Cloudflare-based solution that the team had used before.

**Why.** The goal was to keep **everything under the team's own control** rather than depending
on a third-party access layer. A direct benefit was the **networking knowledge gained** from
owning the setup end to end.

**Tradeoff.** More setup and operational burden than a managed mesh-VPN. Consistent with the
project's broader "own everything end-to-end" stance — the cost is accepted in exchange for
control and understanding.

---

## Roadmap-relevant decisions

| Decision | Status |
|---|---|
| WSL2 → native Linux | **Done** (ADR-09) — resolved the TTS RTF bottleneck |
| STT isolation on subnode/3050 | **Done** (ADR-08) |
| Emotion → TTS prosody | Planned future update (ADR-04) |
| DB server → ESXi subnode | Planned, to offload the main node |