#pragma once

#include <boost/asio/awaitable.hpp>

#include "clients/embedding_client.hpp"
#include "deltaEGO/deltaEGO.hpp"
#include "engine/vector_search_engine.hpp"
#include "schemas/fuli_schemas.hpp"

namespace pipeline {

// This is the actual /character/context request handler — everything
// http_server.cpp does is generic HTTP plumbing that doesn't know or care
// about RAG/emotion; this class is where the domain logic lives.
//
// Current (MVP) pipeline, matching the 5 stages discussed in the project
// notes:
//   [1] (done in main.cpp / http_server.cpp) parse the incoming JSON
//   [2] embed user_input via TEI, unless the caller already supplied
//       dense_vector
//   [3] Faiss GPU search for the top_k nearest memory vectors
//   [4-A] emotion: currently a NEUTRAL (0,0,0) stimulus — the real
//         text -> VAD step (calling the OpenJEV inference server) is
//         deliberately deferred, see HandleContextRequest's comments
//   [4-B] rerank/refine: currently a no-op passthrough — hybrid fusion,
//         graph traversal, and TEI reranking are not implemented yet
//   [5] assemble the response
//
// [4-A] and [4-B] are drawn in the design as running CONCURRENTLY (one on
// a dedicated thread for the CPU-bound physics math, one as an async I/O
// coroutine for network calls), joined before assembly. That's not built
// yet either — right now [4-A] is just a synchronous local call (cheap,
// since it's neutral/no network), so there was nothing to parallelize
// against. Once OpenJEV is wired in as an actual network call, revisit
// this class to fork the two branches for real (see AwaitSearch in the
// .cpp for the kind of bridging that will be needed).
class Orchestrator {
public:
  // None of these are owned by Orchestrator — main.cpp constructs them
  // all and must keep them alive for at least as long as this object
  // (and for as long as any in-flight request coroutine holding a
  // reference to it is still running).
  Orchestrator(clients::EmbeddingClient &embedder,
               IVectorSearchEngine &search_engine,
               deltaEGO::deltaEGO &emotion_engine);

  // The single entry point http_server.cpp's route handler calls. Takes
  // the request by value (it's cheap — mostly small structs plus a
  // moved-in json blob) and returns a fully-assembled response.
  boost::asio::awaitable<schemas::FuliContextResponse>
  HandleContextRequest(schemas::FuliContextRequest req);

private:
  clients::EmbeddingClient &embedder_;
  IVectorSearchEngine &search_engine_;
  deltaEGO::deltaEGO &emotion_engine_;
};

} // namespace pipeline
