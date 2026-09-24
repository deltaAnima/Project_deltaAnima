#include "health/health_checker.hpp"
#include <future>

namespace health 
{
  HealthChecker::HealthChecker(const clients::EmbeddingClient &embedding_client)
    : embedding_client_(embedding_client) 
    {}

  std::future<nlohmann::json> HealthChecker::CheckAll() const
  {
    // Everything below runs on a NEW background thread (via this outer
    // std::async), not on whatever thread calls CheckAll(). That's what
    // makes CheckAll() itself non-blocking to call: we return this
    // outer future immediately, and the caller decides how to wait on
    // it (see util::AwaitFuture in main.cpp's "GET /health" route).
    return std::async(std::launch::async, [this]() {
      // Fan out each client's HealthCheck() onto its OWN thread too, so
      // a slow/unhealthy service doesn't hold up the healthy ones. With
      // only one client this doesn't show its value yet — once
      // RedisClient (etc.) gets its own HealthCheck(), add its
      // std::async call here the same way, and it'll run concurrently
      // with this one instead of after it.
      auto embedding_future = std::async(std::launch::async, [this]() {
        return embedding_client_.HealthCheck();
      });

      // .get() here is fine — we're already on a background thread
      // (the outer std::async's), so blocking THIS thread while we wait
      // for embedding_future doesn't freeze the server. Only blocking
      // the io_context thread would.
      nlohmann::json result;
      result["embedding"] = embedding_future.get();

      return result;
    });
  }
}