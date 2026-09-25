#include "deltaEGO/Carmen.hpp"

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

} // namespace

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
        result = ToVadPoint(co_await this->temp_jev.EstimateVad(context));
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
        result = ToVadPoint(co_await this->temp_jev.EstimateVad(context));
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
          result = ToVadPoint(co_await this->temp_jev.EstimateVad(context));
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
