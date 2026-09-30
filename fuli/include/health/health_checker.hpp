#pragma once

#include <future>

#include "Third_Party/json.hpp"
#include "clients/embedding_client.hpp"
#include "clients/redis_db_client.hpp"
#include "clients/openjev_client.hpp"
#include "deltaEGO/deltaEGO.hpp"
// As more clients get their own HealthCheck(), #include each one here
// and add a const-reference member below, same as embedding_client_.
// deltaEGO doesn't have its own HealthCheck() as such — Check5090Health()
// forwards specifically to Carmen's Gebura_5090 (the 5090 llama.cpp
// fallback VAD estimator), since that client has no standalone instance
// of its own in main.cpp to point at directly (see Carmen.hpp's comment
// on Check5090Health for why openjev_client_ below doesn't need the same
// treatment).

namespace health {

// Aggregates every backend client's HealthCheck() into one combined JSON
// status report. This is the /health route's actual logic — http_server.cpp
// only knows how to route an HTTP request to a handler, it doesn't know
// anything about TEI/Redis/etc, same division of responsibility as
// pipeline::Orchestrator for /character/context.
class HealthChecker {
public:
  // Takes const references to every client we want to health-check.
  // HealthChecker does not own these — whoever constructs it (main.cpp)
  // must keep the real client objects alive for at least as long as this
  // HealthChecker is used, same lifetime rule as Orchestrator.
  HealthChecker(const clients::EmbeddingClient &embedding_client,
                const clients::RedisDbClient &redis_client,
                const clients::OpenJevClient &openjev_client,
                const deltaEGO::deltaEGO &emotion_engine);

  // Runs every registered client's HealthCheck() and returns one combined
  // JSON object, e.g.:
  //   {"embedding": {"status": "healthy", ...}, "redis": {...}}
  //
  // Each individual HealthCheck() (like the one on EmbeddingClient) is a
  // BLOCKING call with its own timeout — it spins up a throwaway
  // io_context internally and runs it until the connect attempt
  // succeeds, fails, or the timer fires. Because of that, CheckAll()
  // must not just call them one after another: if embedding is fine but
  // Redis is down and takes the full 5s timeout to fail, calling them
  // sequentially means every healthy service's status is delayed behind
  // the slow/dead one.
  //
  // CheckAll() itself returns a std::future<json> rather than a plain
  // json — calling it never blocks the caller. Internally (see the .cpp)
  // it fans every client's HealthCheck() out onto its own thread via
  // std::async, then wraps the "wait for all of them and assemble the
  // combined JSON" part in one more std::async so CheckAll() can return
  // immediately without waiting for anything itself. Route handlers
  // (net::awaitable coroutines) are expected to bridge the returned
  // future with util::AwaitFuture (include/util/future_bridge.hpp)
  // instead of calling .get()/.wait() on it directly — see main.cpp's
  // "GET /health" route for the pattern. Calling .get() on it directly
  // from a coroutine would defeat the whole point: it would block
  // whichever io_context thread is running that coroutine until every
  // health check completes.
  std::future<nlohmann::json> CheckAll() const;

private:
  const clients::EmbeddingClient &embedding_client_;
  const clients::RedisDbClient &redis_client_;
  const clients::OpenJevClient &openjev_client_;
  const deltaEGO::deltaEGO &emotion_engine_;
};

} // namespace health
