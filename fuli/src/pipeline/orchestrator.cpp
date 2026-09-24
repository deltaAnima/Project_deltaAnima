#include "pipeline/orchestrator.hpp"

#include "util/future_bridge.hpp"

namespace pipeline {

namespace net = boost::asio;

// IVectorSearchEngine::AsyncSearch() (see engine/vector_search_engine.hpp)
// returns a std::future<SearchResult>, not a boost::asio::awaitable —
// util::AwaitFuture (include/util/future_bridge.hpp) is what turns that
// into something we can co_await without blocking the io_context thread.
// See that header for the full rationale; this used to be a private
// helper duplicated here until HealthChecker needed the exact same
// bridge for a different std::future<T>, at which point it made sense to
// share one implementation instead of copy-pasting the polling loop
// again.

Orchestrator::Orchestrator(clients::EmbeddingClient &embedder,
                            IVectorSearchEngine &search_engine,
                            deltaEGO::deltaEGO &emotion_engine)
    : embedder_(embedder), search_engine_(search_engine),
      emotion_engine_(emotion_engine) {}

net::awaitable<schemas::FuliContextResponse>
Orchestrator::HandleContextRequest(schemas::FuliContextRequest reqest) 
{
  // --- [2] Embedding ---------------------------------------------------
  // dense_vector is only ever set when something upstream already
  // computed an embedding. FuliHandler's docstring is explicit that this
  // is normally null: "Fuli embeds the input itself" — i.e. WE are
  // expected to call TEI here, which is exactly what the else branch
  // does. The dense_vector path mainly exists for testing (see the
  // conversation's curl example that bypassed TEI entirely) and for any
  // future caller that genuinely already has a vector on hand.
  std::vector<float> query_vector;
  if (reqest.rag_policy.dense_vector.has_value()) 
  {
    query_vector = *reqest.rag_policy.dense_vector;
  }
  else
  {
    query_vector = co_await embedder_.Embed(reqest.user_input);
  }

  // --- [3] First-pass retrieval -----------------------------------------
  // MVP scope: memory-domain dense search only. No knowledge-base search,
  // no hybrid (sparse+dense) fusion, no graph traversal, and no Redis
  // metadata join (session/user scoping, importance/time-decay filtering)
  // yet — see schemas/rag_schemas.hpp for exactly which RAGQueryOrder
  // fields are parsed-but-currently-ignored.
  SearchResult raw = co_await util::AwaitFuture(
      search_engine_.AsyncSearch(query_vector, reqest.rag_policy.top_k));

  // --- [4-A] Emotion branch -----------------------------------------------
  // ALWAYS feeds a neutral (v=0, a=0, d=0) stimulus into deltaEGO right
  // now. The real design calls for extracting a per-request VAD stimulus
  // from user_input via the OpenJEV inference server (see the project's
  // grid-search-over-discretized-VAD-bins design discussed separately),
  // but that integration is explicitly deferred until the MVP plumbing
  // (this file) is proven out first. Swap this line out once OpenJEV is
  // ready to be called.
  std::string emotion_json = emotion_engine_.process_stimulus(0.0f, 0.0f, 0.0f);

  // --- [4-B] Rerank/refine branch ------------------------------------------
  // Deferred: enable_rerank (TEI cross-encoder call), fusion_config
  // (memory+knowledge score blending), and granularity/context_window_size
  // (chunk expansion) are all no-ops here — raw Faiss hits pass straight
  // through unchanged. min_score_threshold is parsed but not enforced yet
  // either.
  schemas::FuliContextResponse resp;
  resp.hits.reserve(raw.ids.size());
  for (size_t i = 0; i < raw.ids.size(); ++i) {
    resp.hits.push_back({raw.ids[i], raw.distances[i]});
  }
  resp.emotion_json = std::move(emotion_json);

  // --- [5] Assemble --------------------------------------------------------
  // (Just returning resp — the actual JSON serialization happens in
  // main.cpp via schemas::to_json, triggered by `json(resp).dump()`.)
  co_return resp;
}

} // namespace pipeline
