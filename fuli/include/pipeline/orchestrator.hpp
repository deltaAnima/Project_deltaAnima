#pragma once

#include <boost/asio/awaitable.hpp>

#include "clients/embedding_client.hpp"
#include "deltaEGO/deltaEGO.hpp"
#include "engine/vector_search_engine.hpp"
#include "schemas/fuli_schemas.hpp"

namespace pipeline {

// MVP pipeline: embed -> Faiss search -> (neutral stimulus) -> assemble.
// Hybrid/graph retrieval, reranking, and the OpenJEV emotion branch are
// deliberately not wired in yet — see conversation notes / TODOs below.
class Orchestrator {
public:
  Orchestrator(clients::EmbeddingClient &embedder,
               IVectorSearchEngine &search_engine,
               deltaEGO::deltaEGO &emotion_engine);

  boost::asio::awaitable<schemas::FuliContextResponse>
  HandleContextRequest(schemas::FuliContextRequest req);

private:
  clients::EmbeddingClient &embedder_;
  IVectorSearchEngine &search_engine_;
  deltaEGO::deltaEGO &emotion_engine_;
};

} // namespace pipeline
