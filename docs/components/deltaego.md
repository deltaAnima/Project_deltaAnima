[← Back to components](./README.md) · [← Back to root](../../README.md)

# deltaEGO — C++ emotion engine

deltaEGO is the deterministic half of the [two-stage emotion model](../design-decisions.md).
It takes the VLM's raw VAD guess and turns it into a concrete, reproducible emotional state — not
by asking a model, but by running a small **physics simulation** over a persona and matching the
result against an emotion database with **AVX-512 SIMD** search.

It's a single-translation-unit C++ engine, built for the Ryzen 9 9950X and exposed to Python via
pybind11 as `delta_ego_core`. Called from [Reminh](./reminh.md) as
`EmotionModule.process_stimulus(V, A, D)`.

## Lineage

deltaEGO is the latest step in a line of prior work, not a fresh start:

- **[Delta_me13_RE](https://github.com/namjuu3913/Delta_me13_RE)** — an early local-LLM persona
  chatbot; the ancestor of the deltaAnima persona idea.
- **[deltaEGO v1](https://github.com/namjuu3913/from-VAD-vector-to-String-of-Emotion_onlyCode)**
  — VAD→emotion mapping: a distillation pipeline (NRC-VAD Lexicon → LLM filtering → vectors) plus
  a **k-d tree** emotion search, benchmarked against Faiss.
- **deltaEGO v2** (this engine) — replaced the k-d tree with a flat AVX-512 SIMD scan (see
  [below](#why-simd-instead-of-a-k-d-tree)) and added the personality physics simulation.

### VAD data & license

The emotion database (`VAD.json`) is distilled from the **NRC-VAD Lexicon** via an
LLM-filtering pipeline (originally ~20k lexicon terms → curated vectors, run on GCP/Docker in
v1). Because of the NRC-VAD Lexicon's terms, **the dataset itself stays private — only the
distillation code is public**. To reproduce, obtain the NRC-VAD license and run the pipeline to
generate your own vectors. The distillation pipeline lives in the
[v1 repo](https://github.com/namjuu3913/from-VAD-vector-to-String-of-Emotion_onlyCode).

---

## What it does per call

`process_stimulus(v, a, d)` runs three things and returns a JSON blob:

1. **Physics** — advance the persona's emotional state from where it currently sits toward the
   new stimulus, shaped by personality (OCEAN) and history.
2. **Emotion lookup** — SIMD nearest-neighbor search of the resulting state against a VAD emotion
   database to name it (`emotion_term` + `similarity`).
3. **Analysis** — stress/reward/lability metrics and a frontend expression tag, all serialized
   out for downstream use and persistence.

The key idea: emotion is a **stateful continuous point** that moves under forces, not a label
picked fresh each turn. The same VAD input produces a different result depending on the persona's
current state and recent history.

---

## Two subsystems

### 1. `Int16Tensor` — the SIMD emotion database

The VAD emotion database (see [VAD data & license](#vad-data--license), loaded from `VAD.json`)
is stored as a flat, quantized, cache-friendly array and searched with hand-written AVX-512.

**Layout.** Each emotion is a `(V, A, D, padding)` quadruple, quantized **float → int16**
(`SCALE = 10000`), packed into a single 64-byte-aligned `short*` array. The 4th lane is zero
padding so each vector is exactly 8 bytes and eight vectors fill one 512-bit register cleanly.

> The `SCALE = 10000` value is deliberate: an earlier version used a larger scale that overflowed
> `int` when distances were squared. 10000 keeps squared distances within range.

**Search (`search_knn`).** This is a **brute-force SIMD linear scan**, not a tree — for a small
emotion table, streaming the whole array through vector registers beats tree traversal and its
branch misprediction. Per 512-bit block it:

1. loads 8 int16 vectors (`_mm512_load_si512`),
2. widens int16 → int32 in two halves (`_mm512_cvtepi16_epi32`) to avoid overflow,
3. computes `(target − query)` and squares it (`_mm512_sub_epi32`, `_mm512_mullo_epi32`),
4. horizontally sums each vector's V²+A²+D² via two shuffle+add passes,
5. writes squared distances to a score buffer.

The tail (item count not a multiple of 8) is handled with a scalar loop. Top-k is a
`std::partial_sort`, and integer squared-distance is converted back to a float **similarity** in
`[0, 1]`.

**Why quantize to int16.** Halves memory vs. float32 (better cache residency) and lets one AVX-512
register process 8 emotions at once with integer ops. The accuracy cost is negligible for an
emotion table at `SCALE = 10000`.

### Why SIMD instead of a k-d tree

deltaEGO **v1 used a hand-built k-d tree**. Benchmarked against Faiss, it lost — the custom tree
ran at ~40× over pure Python, while Faiss (FlatL2, AVX/SIMD) reached ~109×. The reason was
structural: a pointer-based k-d tree **chases pointers all over memory**, so every node visit
risks a cache miss, and the branchy traversal causes branch mispredictions. For a small emotion
table, that overhead dominates.

v2's answer: drop the tree. A **flat, contiguous, aligned int16 array streamed through AVX-512**
turns the search into a linear scan with no pointer chasing, no branches in the hot loop, and 8
comparisons per instruction — closing the gap with Faiss's SIMD **without the Faiss dependency**.
The lesson is the point: v1 measured the bottleneck, v2 removed its root cause by changing the
data layout rather than micro-optimizing the tree.

### 2. `emotionPhysics` — the personality-driven simulation

This is what makes the emotion *move*. It models the persona as a point drifting under forces,
parameterized entirely from `Reminh_physics.yaml` (hot-reloadable).

**Per turn (`orchestrate`):**

1. **`runFullAnalysis`** — compute instantaneous **stress** and **reward** from the current VAD
   (weighted V/A terms, dampened inside a stability radius), **affective lability** from the
   VAD delta vs. history (a sigmoid over the change angle), and cumulative stress/reward over a
   bounded history lookback.
2. **`updatePhysicsWeights`** — derive dynamic weights from **OCEAN** traits: extraversion/
   openness raise positive sensitivity, neuroticism raises negative sensitivity, conscientiousness
   raises resistance and decay (all clamped).
3. **`modulatePhysics`** — adjust those weights by the analysis: high cumulative stress →
   hypersensitivity to negativity; high lability → lower resistance (mental whiplash); high reward
   → faster decay back to baseline.
4. **`updateEmotion`** — apply forces. The stimulus is scaled by sensitivity, the state `lerp`s
   toward it by resistance, then `lerp`s back toward the persona's **default VAD** by the decay
   rate, and is clamped to [-1, 1].

The result is a persona that reacts to input *through the lens of its personality*, carries mood
across turns, and drifts back toward its baseline temperament over time.

**Frontend expression.** For rendering, the state's octant (sign of V/A/D) maps to one of eight
zones (Excited/Happy, Surprise, Relaxed, Calm, Angry, Fear, Disgust, Sad), with a neutral zone
inside a small radius, plus cosine similarity to the octant axis and an intensity score.

---

## Output

`process_stimulus` returns JSON:

```jsonc
{
  "current_state": { "v": ..., "a": ..., "d": ..., "r": ... },
  "emotion_term": "Sad/Lonely",   // from the SIMD KNN lookup
  "similarity":   0.83,
  "analysis": {
    "instant":    { "stress": ..., "reward": ..., "deviation": ... },
    "dynamics":   { "delta": {...}, "lability": ... },
    "cumulative": { "stress": ..., "reward": ..., "total": ... },
    "front":      { "expression": "...", "similarity": ..., "intensity": ... }
  }
}
```

Reminh reads `emotion_term` for the response prompt and persists the full `analysis` with the
memory (see [reminh.md](./reminh.md)).

---

## Build & platform

- **C++20**, single translation unit (`deltaEGO.cpp`), built as a pybind11 module.
- **Flags:** `-O3 -march=native -funroll-loops` (GCC/Clang) / `/O2 /arch:AVX512` (MSVC).
- **Cross-platform primitives:** aligned alloc (`posix_memalign` / `_aligned_malloc`) and core
  pinning (`pthread_setaffinity_np` / `SetThreadAffinityMask`) are wrapped per-OS.
- **Deps:** `nlohmann/json` (VAD.json + output), `yaml-cpp` (physics config).
- Built with CMake (`pybind11_add_module`) or `setup.py`.

> `-march=native` bakes in the build host's ISA (AVX-512 on the 9950X). The binary is tuned for
> this specific CPU — consistent with the engine being hand-optimized for one target rather than
> portable.

---

## Known issues / TODO

Honest current state — these are real and live in the code:

- **`front.similarity` / `front.intensity` are typed `int`** while the functions that fill them
  (`get_cosine_sim`, and the fractional intensity) return doubles — so the cosine similarity is
  **truncated to 0/1** and fractional intensity is lost. Should be `double`.
- **History push ordering** — in `orchestrate`, the current state is pushed to history *after*
  analysis runs, and `updateEmotion` mutates state afterward, so the delta/lability on the very
  first turns is based on an empty/partial history.
- **Cumulative analysis is O(n)** over the history lookback each call (flagged in-code as the slow
  path); a leaky-integrator accumulation would remove the per-call loop.
- **Single emotion term** — `process_stimulus` returns only top-1; a top-3 blend is marked TODO.

Documenting these is deliberate — they show the engine is understood in depth, including where it
still falls short.

---

## Related

- Where it sits in a turn: [architecture.md](../architecture.md#request-lifecycle)
- Why deltaEGO exists (probabilistic read → deterministic refine): [design-decisions ADR-01, ADR-05](../design-decisions.md)
- Caller: [reminh.md](./reminh.md)