// Entry point: wires every piece built so far into one running server.
//   config (constants.hpp)
//     -> deltaEGO (emotion engine, loads its own YAML + VAD term DB)
//     -> GpuFaissEngine (seeded with fake data — no real ingestion yet)
//     -> EmbeddingClient (talks to TEI bge-m3)
//     -> Orchestrator (the actual /character/context pipeline logic)
//     -> HttpServer (generic HTTP plumbing, routes into Orchestrator)
// then hands control to io_context.run(), which drives every coroutine
// (HTTP sessions, embedding calls, the Faiss search bridge, ...) from
// here on.
#include <boost/asio/co_spawn.hpp>
#include <boost/asio/detached.hpp>
#include <boost/asio/io_context.hpp>
#include <cmath>
#include <iostream>
#include <random>

#include "Third_Party/json.hpp"
#include "clients/embedding_client.hpp"
#include "clients/redis_db_client.hpp"
#include "config/constants.hpp"
#include "deltaEGO/deltaEGO.hpp"
#include "engine/gpu_faiss_engine.hpp"
#include "health/health_checker.hpp"
#include "http/http_server.hpp"
#include "pipeline/memory_retriever.hpp"
#include "pipeline/orchestrator.hpp"
#include "schemas/fuli_schemas.hpp"
#include "util/future_bridge.hpp"

namespace {

// There is no ingestion pipeline yet (nothing writes real memory/knowledge
// vectors into Faiss), so the index starts out genuinely empty. This
// function fills it with `count` random unit vectors purely so
// AsyncSearch has something real to return — useful for proving the
// HTTP -> embed -> Faiss -> response chain actually works end to end.
// DELETE this call in main() once a real ingestion path exists; leaving
// fake data seeded in a "production" run would obviously return garbage
// search results.
void SeedTestVectors(GpuFaissEngine &engine, int dim, int count)
{
  std::mt19937 rng(42); // fixed seed: reproducible test data across runs
  std::normal_distribution<float> dist(0.0f, 1.0f);

  std::vector<int64_t> ids;
  std::vector<float> flat; // row-major: vector i occupies flat[i*dim, i*dim+dim)
  ids.reserve(count);
  flat.reserve(static_cast<size_t>(count) * dim);

  for (int i = 0; i < count; ++i)
  {
    ids.push_back(1000 + i); // arbitrary fake memory ids, just need to be unique
    size_t base = flat.size();
    float norm_sq = 0.0f;

    for (int d = 0; d < dim; ++d)
    {
      float v = dist(rng);
      flat.push_back(v);
      norm_sq += v * v;
    }

    // Normalize to a unit vector — matches the normalize=true we ask TEI
    // for in EmbeddingClient::Embed, so L2 distance here behaves like it
    // will against real embeddings.
    float norm = std::sqrt(norm_sq);

    for (int d = 0; d < dim; ++d)
      flat[base + d] /= norm;
  }

  engine.AddVectors(ids, flat); // blocks until the GPU worker thread adds these
  std::cout << "[Faiss] seeded " << engine.Size()
            << " random test vectors (dim=" << dim << ")" << std::endl;
}

} // namespace

int main()
{
  namespace net = boost::asio;
  using json = nlohmann::json;

  // {1} = single-threaded io_context: exactly one OS thread will run
  // ioc.run() below and every coroutine takes turns on it cooperatively.
  // That's fine for an MVP (coroutines never block this thread — see the
  // AwaitSearch comment in orchestrator.cpp for why that property
  // matters). Bump this to N threads + call ioc.run() from N threads once
  // you need real parallelism across requests, not just concurrency.
  net::io_context ioc{1};

  // deltaEGO owns its own OCEAN/physics config (loaded once, here, at
  // startup) — it does NOT yet read the per-request emotion_policy the
  // client sends (see schemas/fuli_schemas.hpp's emotion_policy_raw
  // comment). def_V/def_A/def_D/def_radius (all zeros/1.0 here) are the
  // character's resting emotional state before any stimulus is applied.
  std::cout << "[Init] loading deltaEGO config: " << config::kDeltaEgoConfigPath
            << std::endl;
  deltaEGO::deltaEGO emotion_engine(config::kDeltaEgoConfigPath, 0.0f, 0.0f,
                                     0.0f, 1.0f);
  if (!emotion_engine.load_vad_db(config::kVadDbPath)) 
  {
    std::cerr << "[Init] warning: failed to load VAD DB from "
              << config::kVadDbPath << std::endl;
  }

  // GpuFaissEngine spins up its own dedicated worker thread internally
  // (see gpu_faiss_engine.cpp) — nothing else to do here except seed it.
  GpuFaissEngine search_engine(config::kEmbeddingDim);
  SeedTestVectors(search_engine, config::kEmbeddingDim, /*count=*/200);

  // EmbeddingClient doesn't open any connection yet — it just remembers
  // where TEI lives; the actual TCP connection happens per-call inside
  // Embed().
  clients::EmbeddingClient embedder(ioc, config::kEmbeddingHost,
                                     config::kEmbeddingPort);

  // RedisDbClient's background connection (see Run()) needs to be
  // started before anything calls Get/SetMemoryMetadata on it.
  clients::RedisDbClient redis_client(ioc, config::kRedisHost,
                                       config::kRedisPort);
  redis_client.Run();

  // MemoryRetriever is the Faiss+Redis join layer — it's what
  // Orchestrator actually talks to for retrieval now, not
  // search_engine/redis_client directly.
  pipeline::MemoryRetriever memory_retriever(search_engine, redis_client);

  // Orchestrator holds references to all of the above — none of them
  // may be destroyed before Orchestrator (and, transitively, before
  // every request coroutine referencing it) is done. Since every one of
  // these objects is a local variable in main() that lives until
  // ioc.run() returns, and ioc.run() is the last thing this function
  // does, that invariant holds automatically here.
  pipeline::Orchestrator orchestrator(embedder, memory_retriever, emotion_engine);

  // Same reference-holding, same lifetime rule as Orchestrator above:
  // health_checker just borrows embedder, doesn't own it.
  health::HealthChecker health_checker(embedder);

  httpsrv::HttpServer server(ioc, config::kListenPort);
  server.RegisterRoute(
      "POST /character/context",
      // This lambda IS the RequestHandler from http/http_server.hpp: it
      // takes the raw request body string and must return
      // net::awaitable<std::string> (the raw response body string).
      // Capturing &orchestrator by reference is safe for the same reason
      // noted above — orchestrator outlives every request.
      [&orchestrator](std::string body) -> net::awaitable<std::string> {
        // json::parse + .get<T>() triggers schemas::from_json(...)
        // (see fuli_schemas.hpp) to build the typed request struct.
        // Throws on malformed JSON — caught by http_server.cpp's Session()
        // and turned into a 500 response, not a crash.
        schemas::FuliContextRequest req =
            json::parse(body).get<schemas::FuliContextRequest>();
        schemas::FuliContextResponse resp =
            co_await orchestrator.HandleContextRequest(std::move(req));
        // json(resp) triggers schemas::to_json(...) the same way.
        co_return json(resp).dump();
      });

  server.RegisterRoute(
      "GET /health",
      // health_checker.CheckAll() returns a std::future<json> immediately
      // (it does NOT block here) — util::AwaitFuture is what actually
      // waits for it, the same non-blocking-to-the-io_context way
      // Orchestrator waits on GpuFaissEngine::AsyncSearch. Without that
      // bridge, calling .get() on the future directly in this coroutine
      // would stall the whole server for as long as the slowest
      // dependency's health check takes (up to its timeout) — every
      // other in-flight request would freeze too.
      [&health_checker](std::string) -> net::awaitable<std::string> {
        nlohmann::json services =
            co_await util::AwaitFuture(health_checker.CheckAll());

        // Overall status is "ok" only if every reported service is
        // healthy. Iterating services.items() (instead of hardcoding
        // "embedding") means this keeps working unchanged as more
        // clients get added to HealthChecker later.
        bool all_healthy = true;
        for (const auto &[name, status] : services.items()) {
          if (status.value("status", "unknown") != "healthy") {
            all_healthy = false;
            break;
          }
        }

        nlohmann::json resp = {
            {"status", all_healthy ? "ok" : "degraded"},
            {"services", services},
        };
        co_return resp.dump();
      });


  // Run() only SCHEDULES the accept loop (via co_spawn) — it does not
  // block. The actual accepting/serving only happens once we call
  // ioc.run() below.
  server.Run();
  std::cout << "[Fuli] orchestrator_server up — POST /character/context on :"
            << config::kListenPort << std::endl;

  // Blocks here, running the event loop, until there's no more work to do
  // (which in practice means "forever", since Listen() loops
  // indefinitely) or an unhandled exception propagates out of a
  // non-detached coroutine.
  ioc.run();
  return 0;
}
