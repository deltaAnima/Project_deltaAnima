#include "deltaEGO/Carmen.hpp"

#include <cstdio>
#include <iostream>
#include <stdexcept>

// Carmen's constructor is still fully defined inline in Carmen.hpp (it's
// just a one-line forward into temp/Gebura's member initializers) — this
// file is for whisper_from_Carmen(), which has actual logic.

namespace deltaEGO {
namespace Carmen {

namespace {

// clients::VadEstimate (v, a, d) -> structs::VAD_Point (v, a, d, radius).
// radius isn't something a VAD estimator has an opinion on — 1.0f
// matches how deltaEGO.cpp itself builds an input stimulus point
// elsewhere (process_stimulus's VAD_Point{v, a, d, 1.0f}).
structs::VAD_Point ToVadPoint(const clients::VadEstimate &estimate)
{
  return structs::VAD_Point{estimate.v, estimate.a, estimate.d, 1.0f};
}

// --- OpenJEV grid-search config ------------------------------------------
// -1.0..1.0 in 0.1 steps = 21 candidate values per axis. PLACEHOLDER
// hypothesis wording below — the project discussed using real,
// psychologically-calibrated templates for this, not written yet at the
// time this was wired up. Swap ValenceHypothesis/ArousalHypothesis/
// DominanceHypothesis's wording freely; nothing else here depends on the
// exact phrasing.
constexpr float kVadGridMin = -1.0f;
constexpr float kVadGridMax = 1.0f;
constexpr float kVadGridStep = 0.1f;

std::string FormatGridValue(float value)
{
  // "%.1f"-equivalent without pulling in <cstdio> — one decimal place
  // matches kVadGridStep's resolution.
  char buf[16];
  std::snprintf(buf, sizeof(buf), "%.1f", value);
  return std::string(buf);
}

std::string ValenceHypothesis(float value)
{
  return "The emotional tone of this message is rated " + FormatGridValue(value) +
         " on a scale from -1 (very negative) to 1 (very positive).";
}

std::string ArousalHypothesis(float value)
{
  return "The energy level expressed in this message is rated " + FormatGridValue(value) +
         " on a scale from -1 (very calm) to 1 (very excited).";
}

std::string DominanceHypothesis(float value)
{
  return "The sense of control expressed in this message is rated " + FormatGridValue(value) +
         " on a scale from -1 (very submissive) to 1 (very dominant).";
}

} // namespace

// Grid search: for each of the 3 axes, ask OpenJev "does this message
// entail a rating of X?" for every X on the -1..1 grid, then take the
// entailment-probability-weighted average of X as that axis's estimate.
// A near-zero total weight (OpenJev essentially rejected every
// hypothesis on that axis) falls back to 0.0 rather than dividing by
// ~zero.
//
// PERFORMANCE NOTE: this is 3 axes * 21 grid values = 63 sequential
// HTTP round trips per call, each opening a fresh connection (see
// OpenJevClient::Classify's own doc comment — no keep-alive/pooling
// yet). That's almost certainly too slow to sit on a live request path
// as-is; firing the 63 calls concurrently (e.g. via
// boost::asio::experimental::make_parallel_group) instead of one at a
// time is the obvious next step once this needs to be fast, not just
// correct. Written sequentially first to keep the logic easy to read
// and debug.
boost::asio::awaitable<structs::VAD_Point> Carmen::EstimateVadViaOpenJev(
    std::string context) const
{
  float v_sum = 0.0f, v_weight = 0.0f;
  float a_sum = 0.0f, a_weight = 0.0f;
  float d_sum = 0.0f, d_weight = 0.0f;

  for (float value = kVadGridMin; value <= kVadGridMax + 1e-4f; value += kVadGridStep)
  {
    clients::OpenJevScores v_scores =
        co_await this->openjev_.Classify(context, ValenceHypothesis(value));
    v_sum += static_cast<float>(v_scores.entailment) * value;
    v_weight += static_cast<float>(v_scores.entailment);

    clients::OpenJevScores a_scores =
        co_await this->openjev_.Classify(context, ArousalHypothesis(value));
    a_sum += static_cast<float>(a_scores.entailment) * value;
    a_weight += static_cast<float>(a_scores.entailment);

    clients::OpenJevScores d_scores =
        co_await this->openjev_.Classify(context, DominanceHypothesis(value));
    d_sum += static_cast<float>(d_scores.entailment) * value;
    d_weight += static_cast<float>(d_scores.entailment);
  }

  structs::VAD_Point result{};
  result.V = v_weight > 1e-6f ? v_sum / v_weight : 0.0f;
  result.A = a_weight > 1e-6f ? a_sum / a_weight : 0.0f;
  result.D = d_weight > 1e-6f ? d_sum / d_weight : 0.0f;
  result.radius = 1.0f;
  co_return result;
}

boost::asio::awaitable<structs::VAD_Point> Carmen::whisper_from_Carmen(
    std::string context, bool use_jev, bool fallback_to_5090, bool use_5090,
    bool fallback_to_jev)
{
  if (use_jev && fallback_to_jev)
  {
    throw std::runtime_error(
        "Carmen::whisper_from_Carmen() - Cannot use_jev and fallback_to_jev at the same time.");
  }
  if (!use_jev && !fallback_to_jev && !use_5090)
  {
    throw std::runtime_error(
        "Carmen::whisper_from_Carmen() - Must use at least one of use_jev, fallback_to_jev, or use_5090.");
  }
  if (context.empty())
  {
    throw std::runtime_error("Carmen::whisper_from_Carmen() - Context cannot be empty.");
  }

  structs::VAD_Point result{};

  if (use_jev)
  {
    if (fallback_to_5090)
    {
      // co_await is NOT allowed inside a catch block ("await
      // expressions are not permitted in handlers" — a real C++20
      // coroutine rule, not a style choice: a coroutine can't suspend
      // while an exception is mid-unwind). So the fallback call has to
      // happen AFTER the catch block, gated by a flag, instead of
      // directly inside it.
      bool jev_failed = false;
      try
      {
        result = co_await this->EstimateVadViaOpenJev(context);
      }
      catch (const std::exception &e)
      {
        std::cerr << "Carmen::whisper_from_Carmen() - OpenJEV failed, falling back to 5090: " << e.what() << std::endl;
        jev_failed = true;
      }
      if (jev_failed)
      {
        try
        {
          result = ToVadPoint(co_await this->Gebura_5090.EstimateVad(context));
        }
        catch (const std::exception &e)
        {
          std::cerr << "Carmen::whisper_from_Carmen() - jev -> 5090 fallback failed: " << e.what() << std::endl;
          throw; // rethrow to propagate the failure to the caller
        }
      }
    }
    else
    {
      // No fallback requested — let a Gebura failure propagate to the
      // caller instead of silently swallowing it.
      try
      {
        result = co_await this->EstimateVadViaOpenJev(context);
      }
      catch (const std::exception &e)
      {
        std::cerr << "Carmen::whisper_from_Carmen() - OpenJEV failed. No fallback available: " << e.what() << std::endl;
        throw; // rethrow to propagate the failure to the caller
      }
    }
  }
  else if (use_5090)
  {
    if (fallback_to_jev)
    {
      bool NV5090_failed = false;
      try
      {
        result = ToVadPoint(co_await this->Gebura_5090.EstimateVad(context));
      }
      catch(const std::exception& e)
      {
        std::cerr << "Carmen::whisper_from_Carmen() - 5090 failed, falling back to jev: " << e.what() << std::endl;
        NV5090_failed = true;
      }
      if (NV5090_failed)
      {
        try
        {
          result = co_await this->EstimateVadViaOpenJev(context);
        }
        catch (const std::exception &e)
        {
          std::cerr << "Carmen::whisper_from_Carmen() - 5090 -> jev fallback failed: " << e.what() << std::endl;
          throw; // rethrow to propagate the failure to the caller
        }
      }
    }
    else
    {
      // 5090 only, no fallback requested — mirrors the use_jev-only
      // branch above: let a Gebura_5090 failure propagate to the caller
      // instead of silently swallowing it. (This used to be an
      // unconditional throw here — left in from before the
      // fallback_to_jev branch above actually computed a real result —
      // which meant this whole case always failed even on success.)
      try
      {
        result = ToVadPoint(co_await this->Gebura_5090.EstimateVad(context));
      }
      catch (const std::exception &e)
      {
        std::cerr << "Carmen::whisper_from_Carmen() - 5090 failed. No fallback available: " << e.what() << std::endl;
        throw; // rethrow to propagate the failure to the caller
      }
    }
  }
  else
  {
    throw std::runtime_error(
        "Carmen::whisper_from_Carmen() - No valid VAD estimation method selected. This should never happen, check the logic above.");
  }

  co_return result;
}

} // namespace Carmen
} // namespace deltaEGO
