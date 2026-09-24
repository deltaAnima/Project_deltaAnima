#include <boost/asio/co_spawn.hpp>
#include <boost/asio/detached.hpp>
#include <boost/asio/io_context.hpp>
#include <cmath>
#include <iostream>
#include <random>

#include "Third_Party/json.hpp"
#include "clients/embedding_client.hpp"
#include "config/constants.hpp"
#include "deltaEGO/deltaEGO.hpp"
#include "engine/gpu_faiss_engine.hpp"
#include "http/http_server.hpp"
#include "pipeline/orchestrator.hpp"
#include "schemas/fuli_schemas.hpp"

namespace {

// Faiss is installed but empty — seed it with random unit vectors so
// AsyncSearch has something real to return while there's no ingestion
// pipeline yet. Remove once actual memory vectors get written in.
void SeedTestVectors(GpuFaissEngine &engine, int dim, int count) {
  std::mt19937 rng(42);
  std::normal_distribution<float> dist(0.0f, 1.0f);

  std::vector<int64_t> ids;
  std::vector<float> flat;
  ids.reserve(count);
  flat.reserve(static_cast<size_t>(count) * dim);

  for (int i = 0; i < count; ++i) {
    ids.push_back(1000 + i);
    size_t base = flat.size();
    float norm_sq = 0.0f;
    for (int d = 0; d < dim; ++d) {
      float v = dist(rng);
      flat.push_back(v);
      norm_sq += v * v;
    }
    float norm = std::sqrt(norm_sq);
    for (int d = 0; d < dim; ++d)
      flat[base + d] /= norm;
  }

  engine.AddVectors(ids, flat);
  std::cout << "[Faiss] seeded " << engine.Size()
            << " random test vectors (dim=" << dim << ")" << std::endl;
}

} // namespace

int main() {
  namespace net = boost::asio;
  using json = nlohmann::json;

  net::io_context ioc{1};

  std::cout << "[Init] loading deltaEGO config: " << config::kDeltaEgoConfigPath
            << std::endl;
  deltaEGO::deltaEGO emotion_engine(config::kDeltaEgoConfigPath, 0.0f, 0.0f,
                                     0.0f, 1.0f);
  if (!emotion_engine.load_vad_db(config::kVadDbPath)) {
    std::cerr << "[Init] warning: failed to load VAD DB from "
              << config::kVadDbPath << std::endl;
  }

  GpuFaissEngine search_engine(config::kEmbeddingDim);
  SeedTestVectors(search_engine, config::kEmbeddingDim, /*count=*/200);

  clients::EmbeddingClient embedder(ioc, config::kEmbeddingHost,
                                     config::kEmbeddingPort);

  pipeline::Orchestrator orchestrator(embedder, search_engine, emotion_engine);

  httpsrv::HttpServer server(ioc, config::kListenPort);
  server.RegisterRoute(
      "POST /character/context",
      [&orchestrator](std::string body) -> net::awaitable<std::string> {
        schemas::FuliContextRequest req =
            json::parse(body).get<schemas::FuliContextRequest>();
        schemas::FuliContextResponse resp =
            co_await orchestrator.HandleContextRequest(std::move(req));
        co_return json(resp).dump();
      });

  server.Run();
  std::cout << "[Fuli] orchestrator_server up — POST /character/context on :"
            << config::kListenPort << std::endl;

  ioc.run();
  return 0;
}
