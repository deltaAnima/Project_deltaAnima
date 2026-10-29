#pragma once
#include <boost/asio/io_context.hpp>
#include "clients/llama_5090_client.hpp"
#include "clients/laya_client.hpp"
#include "deltaEGO/deltaEGO.hpp"
// Internal header — NOT part of deltaEGO's public API (see
// deltaEGO.hpp's own doc comment). Shared between deltaEGO.cpp (which
// owns a Carmen::Carmen via unique_ptr) and Carmen.cpp (which implements
// it), same reason as Ayin.hpp.

namespace deltaEGO {
namespace Carmen {

// Turns raw text into a VAD stimulus via one of two estimators: Laya
// (three `score` questions answered in one forward pass, see
// laya_client.hpp) or the 5090's SGLang model (one schema-constrained
// LLM call), with optional fallback from either to the other.
class Carmen
{
public:
  // ioc is NOT owned — it must be the SAME io_context main.cpp created
  // and calls ioc.run() on. Every network client in this project
  // (EmbeddingClient, RedisDbClient, LayaClient, Llama5090Client)
  // takes io_context& the same way, for the same reason: there is
  // exactly one event loop for the whole server, and a coroutine
  // spawned onto any OTHER io_context (e.g. one Carmen constructed for
  // itself) would never actually run, because nothing calls .run() on
  // it — co_await inside laya_/Gebura_5090's calls would hang
  // forever, not fail loudly.
  Carmen(boost::asio::io_context &ioc,
        std::string host_laya, std::string port_laya, std::string laya_api_key,
        std::string host_5090, std::string port_5090)
      : ioc_(ioc), laya_(ioc_, host_laya, port_laya, laya_api_key),
        Gebura_5090(ioc_, host_5090, port_5090) {};

  ~Carmen() = default;

  // Implemented in Carmen.cpp, not here — see that file. (Bare
  // `structs::`, not `deltaEGO::structs::`: this class already lives
  // inside namespace deltaEGO::Carmen, and the qualified form actually
  // fails to compile here — `deltaEGO::` resolves to the CLASS
  // deltaEGO::deltaEGO, same name as the enclosing namespace, not the
  // namespace itself. Same reason Ayin.hpp/Ayin.cpp write bare
  // `structs::`/`func::` throughout.)
  // net::awaitable<...>, not a plain structs::VAD_Point — laya_'s and
  // Gebura_5090's EstimateVad() are themselves coroutines
  // (see laya_client.hpp/llama_5090_client.hpp), so anything that
  // calls them (via co_await) has to be a coroutine too. Callers
  // co_await this the same way Orchestrator co_awaits
  // EmbeddingClient::Embed.
  // Defaults to "5090 only, no Laya" — the 5090 path (validated
  // Reminh-persona prompt) is the current basis. Pass use_laya=true
  // explicitly to make Laya primary.
  boost::asio::awaitable<structs::VAD_Point>
  whisper_from_Carmen(std::string context, bool use_laya = false,
                      bool fallback_to_5090 = false, bool use_5090 = true,
                      bool fallback_to_laya = false);

  // Forwards to Gebura_5090's own HealthCheck() (blocking, self-contained
  // — same pattern as EmbeddingClient::HealthCheck). laya_ doesn't
  // need an equivalent here: main.cpp already constructs its own
  // standalone LayaClient for the smoke test, so HealthChecker talks
  // to that one directly instead of reaching through Carmen for it.
  // Gebura_5090 has no such standalone twin — it only exists inside
  // Carmen — so this is the one health check that has to cross the
  // Pimpl boundary (see deltaEGO::Check5090Health).
  nlohmann::json Check5090Health() const { return Gebura_5090.HealthCheck(); }

private:
  boost::asio::io_context &ioc_; // borrowed, not owned — see ctor comment
  clients::LayaClient laya_;            // opt-in VAD estimator — pass use_laya=true to whisper_from_Carmen to use it
  clients::Llama5090Client Gebura_5090; // DEFAULT VAD estimator — see whisper_from_Carmen's default arguments
};

} // namespace Carmen
} // namespace deltaEGO
