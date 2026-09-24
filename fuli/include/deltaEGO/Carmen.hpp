#pragma once

// Internal header — NOT part of deltaEGO's public API (see
// deltaEGO.hpp's own doc comment). Shared between deltaEGO.cpp (which
// owns a Carmen::Carmen via unique_ptr) and Carmen.cpp (which implements
// it), same reason as Ayin.hpp.

namespace deltaEGO {
namespace Carmen {

// Will own turning raw text into a VAD stimulus via the OpenJEV NLI
// model (batching premise/hypothesis pairs across the discretized -1..1
// grid per axis, then converting entailment probabilities into a
// weighted-average (v, a, d) — pre-processing = building those pairs,
// post-processing = the weighted average). See Carmen.cpp: NOT
// implemented yet, this is currently just a scaffold.
class Carmen {
public:
  Carmen();
};

} // namespace Carmen
} // namespace deltaEGO
