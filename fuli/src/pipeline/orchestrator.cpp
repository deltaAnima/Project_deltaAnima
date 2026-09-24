#include "pipeline/orchestrator.hpp"

#include <boost/asio/steady_timer.hpp>
#include <boost/asio/this_coro.hpp>
#include <boost/asio/use_awaitable.hpp>
#include <chrono>

namespace pipeline {

namespace net = boost::asio;

namespace {

// --- Why this function exists ---------------------------------------------
// IVectorSearchEngine::AsyncSearch() (see engine/vector_search_engine.hpp)
// returns a std::future<SearchResult>, NOT a boost::asio::awaitable. That
// interface predates the HTTP/coroutine layer of this server, and
// std::future has no built-in way to "wake up" an asio coroutine when it
// becomes ready — futures and asio's executor model are two unrelated
// concurrency systems that don't talk to each other automatically.
//
// The correct long-term fix is to change the engine's API to complete via
// an asio completion token (so the GPU worker thread posts its result
// directly onto the caller's executor) instead of a bare std::future.
// That's a real refactor across gpu_faiss_engine.hpp/.cpp and the
// IVectorSearchEngine interface, deliberately not done yet.
//
// For now, this is a pragmatic bridge: poll the future every 2ms, but do
// the "waiting" with an asio steady_timer + co_await instead of
// fut.wait()/fut.get() directly. That matters a lot: calling fut.get()
// (or fut.wait()) here would BLOCK the io_context thread solid until the
// GPU search finishes, which would freeze every other in-flight request
// this server is handling (remember: this server may run on as few as
// one io_context thread — see main.cpp's `net::io_context ioc{1}`).
// Looping on a short async timer instead means this coroutine repeatedly
// gives control back to the io_context between checks, so other requests
// keep making progress while this one waits.
net::awaitable<SearchResult> AwaitSearch(std::future<SearchResult> fut) {
  auto executor = co_await net::this_coro::executor;
  net::steady_timer timer(executor);
  while (fut.wait_for(std::chrono::milliseconds(0)) !=
         std::future_status::ready) {
    timer.expires_after(std::chrono::milliseconds(2));
    co_await timer.async_wait(net::use_awaitable);
  }
  co_return fut.get(); // re-throws here if the worker thread set an
                        // exception (see GpuFaissEngine::Impl::AsyncSearch)
}

} // namespace

Orchestrator::Orchestrator(clients::EmbeddingClient &embedder,
                            IVectorSearchEngine &search_engine,
                            deltaEGO::deltaEGO &emotion_engine)
    : embedder_(embedder), search_engine_(search_engine),
      emotion_engine_(emotion_engine) {}

net::awaitable<schemas::FuliContextResponse>
Orchestrator::HandleContextRequest(schemas::FuliContextRequest req) {
  // --- [2] Embedding ---------------------------------------------------
  // dense_vector is only ever set when something upstream already
  // computed an embedding. FuliHandler's docstring is explicit that this
  // is normally null: "Fuli embeds the input itself" — i.e. WE are
  // expected to call TEI here, which is exactly what the else branch
  // does. The dense_vector path mainly exists for testing (see the
  // conversation's curl example that bypassed TEI entirely) and for any
  // future caller that genuinely already has a vector on hand.
  std::vector<float> query_vector;
  if (req.rag_policy.dense_vector.has_value()) {
    query_vector = *req.rag_policy.dense_vector;
  } else {
    query_vector = co_await embedder_.Embed(req.user_input);
  }

  // --- [3] First-pass retrieval -----------------------------------------
  // MVP scope: memory-domain dense search only. No knowledge-base search,
  // no hybrid (sparse+dense) fusion, no graph traversal, and no Redis
  // metadata join (session/user scoping, importance/time-decay filtering)
  // yet — see schemas/rag_schemas.hpp for exactly which RAGQueryOrder
  // fields are parsed-but-currently-ignored.
  SearchResult raw = co_await AwaitSearch(
      search_engine_.AsyncSearch(query_vector, req.rag_policy.top_k));

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
