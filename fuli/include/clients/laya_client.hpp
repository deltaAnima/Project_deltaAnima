#pragma once

#include <boost/asio/awaitable.hpp>
#include <boost/asio/io_context.hpp>
#include <string>
#include "Third_Party/json.hpp"
#include "clients/llama_5090_client.hpp" // VadEstimate

namespace clients {

// Async HTTP client for Laya Serve (enlotec/laya-serve), the typed-decision
// encoder that replaced OpenJev as Carmen's primary VAD estimator.
//
// Unlike OpenJev's /classify (ONE premise/hypothesis pair per call, so a
// VAD estimate took 3 axes * 21 grid values = 63 round trips), Laya takes
// a whole set of typed questions about one text in a single forward pass.
// EstimateVad() asks three `score` questions — valence, arousal,
// dominance — each with an ordered list of levels, in ONE
// POST /v1/systemone:
//   {"state": {"body": text},
//    "questions": {"valence": {"type": "score", "instructions": ..,
//                              "criteria": [lowest, .., highest]}, ..}}
//   -> {"answers": {"valence": {"score": 4.7, ..}, ..}, ..}
// `score` is the expected ordinal level (0..levels-1), which is mapped
// linearly onto -1..1 per axis.
//
// Every call has a hard deadline (kLayaTimeoutSeconds in the .cpp) on the
// whole connect/write/read exchange, so an unreachable Laya fails fast and
// Carmen's 5090 fallback kicks in, instead of waiting out the kernel's
// ~2 minute SYN timeout the way the old OpenJev client did.
class LayaClient
{
public:
  // ioc must outlive this object. api_key is LAYA_API_KEY's bearer token;
  // empty means Laya was started without one and no Authorization header
  // is sent.
  LayaClient(boost::asio::io_context &ioc, std::string host, std::string port,
             std::string api_key = "");

  // Throws on connect/timeout failure, a non-200 response (including
  // Laya's 503 when all LAYA_MAX_CONCURRENT slots are busy), or a reply
  // missing any of the three scores.
  boost::asio::awaitable<VadEstimate> EstimateVad(std::string text) const;

  // GET /ready — 200 only once Laya's checkpoints are loaded. Blocking
  // (runs on its own throwaway io_context, with the same deadline as
  // EstimateVad), so HealthChecker::CheckAll can std::async it like the
  // other clients' HealthCheck().
  nlohmann::json HealthCheck() const;

private:
  boost::asio::io_context &ioc_; // unused directly, same as the other clients
  std::string host_;
  std::string port_;
  std::string api_key_;
};

} // namespace clients
