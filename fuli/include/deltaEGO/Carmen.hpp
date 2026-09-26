#pragma once
#include <boost/asio/io_context.hpp>
#include "clients/llama_5090_client.hpp"
#include "clients/openjev_client.hpp"
#include "deltaEGO/deltaEGO.hpp"
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
class Carmen
{
public:
  // ioc is NOT owned — it must be the SAME io_context main.cpp created
  // and calls ioc.run() on. Every network client in this project
  // (EmbeddingClient, RedisDbClient, OpenJevClient, Llama5090Client)
  // takes io_context& the same way, for the same reason: there is
  // exactly one event loop for the whole server, and a coroutine
  // spawned onto any OTHER io_context (e.g. one Carmen constructed for
  // itself) would never actually run, because nothing calls .run() on
  // it — co_await inside openjev_/Gebura_5090's calls would hang
  // forever, not fail loudly.
  Carmen(boost::asio::io_context &ioc,
        std::string host_jev, std::string port_jev,
        std::string host_5090, std::string port_5090)
      : ioc_(ioc), openjev_(ioc_, host_jev, port_jev),
        Gebura_5090(ioc_, host_5090, port_5090) {};

  ~Carmen() = default;

  // Implemented in Carmen.cpp, not here — see that file. (Bare
  // `structs::`, not `deltaEGO::structs::`: this class already lives
  // inside namespace deltaEGO::Carmen, and the qualified form actually
  // fails to compile here — `deltaEGO::` resolves to the CLASS
  // deltaEGO::deltaEGO, same name as the enclosing namespace, not the
  // namespace itself. Same reason Ayin.hpp/Ayin.cpp write bare
  // `structs::`/`func::` throughout.)
  // net::awaitable<...>, not a plain structs::VAD_Point — openjev_'s
  // Classify() and Gebura_5090's EstimateVad() are themselves coroutines
  // (see openjev_client.hpp/llama_5090_client.hpp), so anything that
  // calls them (via co_await) has to be a coroutine too. Callers
  // co_await this the same way Orchestrator co_awaits
  // EmbeddingClient::Embed.
  // Defaults to "5090 only, no OpenJEV" — the 5090 path (validated
  // Reminh-persona prompt, one fast LLM call) is the current basis;
  // OpenJEV's grid-search path still exists (see EstimateVadViaOpenJev)
  // but is opt-in only until its placeholder hypothesis wording and
  // per-call-pair Classify() (batching is the friend's branch's job, not
  // this one's) are sorted out. Pass use_jev=true explicitly to use it.
  boost::asio::awaitable<structs::VAD_Point>
  whisper_from_Carmen(std::string context, bool use_jev = false,
                      bool fallback_to_5090 = false, bool use_5090 = true,
                      bool fallback_to_jev = false);

private:
  // OpenJev's /classify only scores ONE premise/hypothesis pair at a
  // time (see openjev_client.hpp) — it doesn't hand back a VAD triple
  // directly the way Llama5090Client::EstimateVad does. This is the
  // pre/post-processing Carmen's class comment above already promised:
  // build a hypothesis per candidate value on the -1..1 grid for each
  // axis, call openjev_.Classify() for each, and weighted-average the
  // entailment probabilities into one (v, a, d). See Carmen.cpp.
  boost::asio::awaitable<structs::VAD_Point>
  EstimateVadViaOpenJev(std::string context) const;

  boost::asio::io_context &ioc_; // borrowed, not owned — see ctor comment
  clients::OpenJevClient openjev_;      // opt-in VAD estimator (via EstimateVadViaOpenJev's grid search) — pass use_jev=true to whisper_from_Carmen to use it
  clients::Llama5090Client Gebura_5090; // DEFAULT VAD estimator — see whisper_from_Carmen's default arguments
};

} // namespace Carmen
} // namespace deltaEGO
