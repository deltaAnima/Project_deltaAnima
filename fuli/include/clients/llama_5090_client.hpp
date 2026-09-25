#pragma once

#include <boost/asio/awaitable.hpp>
#include <boost/asio/io_context.hpp>
#include <string>

#include "Third_Party/json.hpp"

namespace clients {

// A VAD stimulus estimate. Same shape as deltaEGO::structs::VAD_Point's
// (v, a, d) — kept as a plain local struct instead of depending on
// deltaEGO directly, since this client lives in clients/, one layer
// below deltaEGO, and whatever wires it in as Carmen's fallback branch
// is expected to convert this into deltaEGO's own VAD_Point.
struct VadEstimate {
  float v = 0.0f;
  float a = 0.0f;
  float d = 0.0f;
};

// Async HTTP client for the llama.cpp server on the 5090 machine
// (OpenAI-compatible /v1/chat/completions). This is the FALLBACK VAD
// estimator — emotion_policy.use_5090 / fallback_to_5090 (see
// main_orch_files/http_request_format.json) — for when the primary
// OpenJEV path (a teammate's separate branch, not this client) is
// unavailable or fails. Deliberately standalone: nothing here depends
// on deltaEGO/Carmen, so it builds and tests independently of whatever
// Carmen ends up looking like once both branches land — wiring
// Llama5090Client in as Carmen's actual fallback is a follow-up, not
// done here.
//
// Unlike OpenJEV's NLI-based design (score many premise/hypothesis
// pairs, weighted-average the entailment probabilities into a VAD
// point), this asks the LLM directly for one (v, a, d) estimate per
// call, using llama.cpp's schema-constrained JSON output
// (response_format.schema) so the reply is guaranteed valid JSON
// instead of needing to parse free-form text out of a chat response.
class Llama5090Client {
public:
  Llama5090Client(boost::asio::io_context &ioc, std::string host,
                   std::string port);

  // Throws on HTTP failure, a non-2xx response, or a reply that doesn't
  // parse as {"v":..,"a":..,"d":..} even after the schema constraint
  // (a sufficiently broken/quantized local model can still emit
  // something the schema technically allows but that fails our range
  // check). Callers that want "fall back to neutral instead of failing
  // the request" should catch around this, not treat a throw as fatal —
  // this class only estimates, it doesn't decide what happens on failure.
  boost::asio::awaitable<VadEstimate> EstimateVad(std::string text) const;

  // GET /health — llama.cpp's server returns 200 {"status":"ok"} when
  // ready, 503 while the model is still loading. Blocking, same pattern
  // as EmbeddingClient::HealthCheck, so HealthChecker::CheckAll can
  // std::async both the same way.
  nlohmann::json HealthCheck() const;

private:
  boost::asio::io_context &ioc_;
  std::string host_;
  std::string port_;
};

} // namespace clients
