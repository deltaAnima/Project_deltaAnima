#include "health/health_checker.hpp"
#include <future>

namespace health 
{
  HealthChecker::HealthChecker(const clients::EmbeddingClient &embedding_client,
                                const clients::RedisDbClient &redis_client,
                                const clients::OpenJevClient &openjev_client,
                                const deltaEGO::deltaEGO &emotion_engine)
    : embedding_client_(embedding_client), redis_client_(redis_client),
      openjev_client_(openjev_client), emotion_engine_(emotion_engine)
    {}

  std::future<nlohmann::json> HealthChecker::CheckAll() const
  {
    // Everything below runs on a NEW background thread (via this outer
    // std::async), not on whatever thread calls CheckAll(). That's what
    // makes CheckAll() itself non-blocking to call: we return this
    // outer future immediately, and the caller decides how to wait on
    // it (see util::AwaitFuture in main.cpp's "GET /health" route).
    return std::async(std::launch::async, [this]() {
      // Fan out each client's HealthCheck() onto its OWN thread, so a
      // slow/unhealthy service doesn't hold up the healthy ones.
      auto embedding_future = std::async(std::launch::async, [this]() {
        return embedding_client_.HealthCheck();
      });
      auto redis_future = std::async(std::launch::async, [this]() {
        return redis_client_.HealthCheck();
      });
      auto openjev_future = std::async(std::launch::async, [this]() {
        return openjev_client_.HealthCheck();
      });
      auto llama_5090_future = std::async(std::launch::async, [this]() {
        return emotion_engine_.Check5090Health();
      });

      // .get() here is fine — we're already on a background thread
      // (the outer std::async's), so blocking THIS thread while we wait
      // for each future doesn't freeze the server. Only blocking the
      // io_context thread would.
      nlohmann::json result;
      result["embedding"] = embedding_future.get();
      result["redis"] = redis_future.get();
      result["openjev"] = openjev_future.get();
      result["llama_5090"] = llama_5090_future.get();

      return result;
    });
  }
}