#include "deltaEGO/Carmen.hpp"

// deltaEGO::deltaEGO::process_stimulus still takes an explicit (v, a, d)
// from its caller instead of deriving one from text through Carmen —
// wiring the actual OpenJEV HTTP call + grid-search averaging up is
// future work, tracked separately from this reorganization. There is no
// existing OpenJEV-calling code anywhere in this project to move here,
// so Carmen() staying `= default` is accurate, not a placeholder left in
// by mistake.

namespace deltaEGO {
namespace Carmen {

Carmen::Carmen() = default;

} // namespace Carmen
} // namespace deltaEGO
