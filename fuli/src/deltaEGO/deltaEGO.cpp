/**
 * deltaEGO — top-level orchestrator only.
 *
 * The actual engine internals live in their own translation units:
 *   Ayin.cpp   — VAD finding-and-processing (Roland: analysis,
 *                Angela: AVX-512 vector search)
 *   Carmen.cpp — text->VAD via Laya or the 5090 SGLang model
 *
 * This file just owns one of each (via Ayin.hpp/Carmen.hpp) and
 * delegates — plus the shared structs::/func:: implementations declared
 * in deltaEGO.hpp, since Ayin.cpp needs those too and a definition has
 * to live somewhere.
 */
#include "deltaEGO/deltaEGO.hpp"
#include "deltaEGO/Ayin.hpp"
#include "deltaEGO/Carmen.hpp"

#include <cmath>
#include <memory>
#include <string>

#include "Third_Party/json.hpp"

using json = nlohmann::json;

// ==========================================
// structs::/func:: implementations. Declared in deltaEGO.hpp so every
// .cpp that needs them (this one, and Ayin.cpp) shares one definition;
// defined here since deltaEGO.cpp is the "core" file that has to exist
// regardless of how many engine-internal .cpp files come and go.
// ==========================================
namespace deltaEGO {

namespace structs {

inline void to_json(json &j, const VAD_Point &p)
{
  j = json
  {
      {"v", p.V},
      {"a", p.A},
      {"d", p.D},
      {"r", p.radius}
  };
}

inline void from_json(const json &j, VAD_Point &p)
{
  j.at("v").get_to(p.V);
  j.at("a").get_to(p.A);
  j.at("d").get_to(p.D);
  j.at("r").get_to(p.radius);
}

// Depends on VAD_Point's to_json above (for dynamics.delta) — must stay
// defined after it in this file.
inline void to_json(json &j, const AnalysisResult &ar)
{
  j = json
  {
      {"instant", {
          {"stress", ar.instant.stress},
          {"reward", ar.instant.reward},
          {"deviation", ar.instant.deviation},
          {"ratio_total", ar.instant.ratio_total},
          {"stress_ratio", ar.instant.stress_ratio},
          {"reward_ratio", ar.instant.reward_ratio}
      }},
      {"dynamics", {
          {"delta", ar.dynamics.delta},
          {"lability", ar.dynamics.affective_lability}
      }},
      {"cumulative", {
          {"stress", ar.cumulative.stress},
          {"reward", ar.cumulative.reward},
          {"total", ar.cumulative.total},
          {"stress_ratio", ar.cumulative.stress_ratio},
          {"reward_ratio", ar.cumulative.reward_ratio}
      }},
      {"front", {
          {"expression", ar.front.front_expression_name},
          {"similarity", ar.front.similarity},
          {"intensity", ar.front.intensity}
      }}
  };
}

} // namespace structs

namespace func {

float lerp(float target, float current, float resistance)
{
  return target + (current - target) * resistance;
}

double get_distance(const structs::VAD_Point &a, const structs::VAD_Point &b)
{
  return std::sqrt(std::pow(a.V - b.V, 2) + std::pow(a.A - b.A, 2) + std::pow(a.D - b.D, 2));
}

double sigmoid(double x)
{
  if (x >= 0)
    return 1.0 / (1.0 + std::exp(-x));

  double e = std::exp(x);
  return e / (1.0 + e);
}

} // namespace func

// ==========================================
// Public deltaEGO::deltaEGO — owns Ayin::Ayin/Carmen::Carmen directly.
// No Impl class, no VAD math, no search, no physics here — just
// delegation plus (for process_stimulus) assembling the response JSON.
// ==========================================

deltaEGO::deltaEGO(boost::asio::io_context &ioc,
                  const std::string &config_path,
                  float def_V, float def_A, float def_D, float def_radius,
                  std::string host_laya, std::string port_laya, std::string laya_api_key,
                  std::string host_5090, std::string port_5090)
    : ayin_(std::make_unique<Ayin::Ayin>(config_path, def_V, def_A, def_D,def_radius)),
      carmen_(std::make_unique<Carmen::Carmen>(ioc, host_laya, port_laya, laya_api_key,
                                               host_5090, port_5090))
{
}

// Defined here (not defaulted inline in the header) because Ayin::Ayin
// and Carmen::Carmen are only forward-declared in deltaEGO.hpp —
// unique_ptr's deleter needs their complete type to destroy them, which
// is only visible once Ayin.hpp/Carmen.hpp are included, as they are
// above. Same reason the move constructor/assignment below can't be
// defaulted in the header either.
deltaEGO::~deltaEGO() = default;
deltaEGO::deltaEGO(deltaEGO &&) noexcept = default;
deltaEGO &deltaEGO::operator=(deltaEGO &&) noexcept = default;

boost::asio::awaitable<nlohmann::json> deltaEGO::sephirothic_tree(
    std::string context, std::string mood_key, nlohmann::json emotion_policy)
{
  auto flag = [&emotion_policy](const char *key, bool def) {
    if (emotion_policy.is_object() && emotion_policy.contains(key) &&
        emotion_policy.at(key).is_boolean())
      return emotion_policy.at(key).get<bool>();
    return def;
  };
  // Laya replaced OpenJev as the NLI-style estimator; callers still
  // sending the old use_openjev/fallback_to_openjev keys get Laya, so the
  // wire format didn't have to change in lockstep. The new key wins when
  // both are present.
  bool use_laya = flag("use_laya", flag("use_openjev", false));
  bool use_5090 = flag("use_5090", !use_laya);
  bool fallback_to_5090 = flag("fallback_to_5090", false);
  bool fallback_to_laya = flag("fallback_to_laya", flag("fallback_to_openjev", false));

  // co_await, not a plain call — whisper_from_Carmen is a coroutine
  // (it makes network calls to Laya / the 5090 model), so without
  // co_await this would just hand back the unstarted
  // awaitable<VAD_Point> object itself instead of the actual result
  // (and .V/.A/.D wouldn't even compile against that type).
  structs::VAD_Point vad_point = co_await this->carmen_->whisper_from_Carmen(
      context, use_laya, fallback_to_5090, use_5090, fallback_to_laya);

  // process_stimulus itself is still plain/synchronous (no network
  // calls in it) — co_return just wraps its result back into this
  // coroutine's awaitable<std::string>.
  co_return this->process_stimulus(vad_point.V, vad_point.A, vad_point.D, mood_key,
                                   emotion_policy);
}

bool deltaEGO::load_vad_db(const std::string &json_path)
{
  return ayin_->load_vad_db(json_path);
}

nlohmann::json deltaEGO::process_stimulus(float v, float a, float d,
                                          const std::string &mood_key,
                                          const nlohmann::json &emotion_policy)
{
  structs::VAD_Point input = {v, a, d, 1.0f};

  // carmen_ isn't wired into this path yet (see Carmen.hpp's comment) —
  // the stimulus still comes directly from the caller's v/a/d arguments.
  // Once OpenJEV integration lands, this is where deltaEGO would decide
  // between the caller-supplied stimulus and one Carmen derives from raw
  // text.
  Ayin::Ayin::ProcessResult result = this->ayin_->Process(mood_key, input, emotion_policy);

  json j_out;
  j_out["current_state"] = result.current_state;
  j_out["emotion_term"] = result.emotion_term;
  j_out["similarity"] = result.similarity;
  j_out["analysis"] = result.analysis;

  return j_out;
}

bool deltaEGO::reload_config()
{
  return ayin_->reload_config();
}

nlohmann::json deltaEGO::Check5090Health() const
{
  return carmen_->Check5090Health();
}

} // namespace deltaEGO
