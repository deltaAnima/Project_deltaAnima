#include "pipeline/orchestrator.hpp"

#include <boost/asio/steady_timer.hpp>
#include <boost/asio/this_coro.hpp>
#include <boost/asio/use_awaitable.hpp>
#include <chrono>

namespace pipeline {

namespace net = boost::asio;

namespace {

// IVectorSearchEngine::AsyncSearch returns std::future<SearchResult> (it
// predates the coroutine pieces of this server). Bridge it onto the
// current executor by polling instead of blocking the io_context thread.
// TODO: replace with a proper completion-token based engine API once
// there's more than one engine implementation to design against.
net::awaitable<SearchResult> AwaitSearch(std::future<SearchResult> fut) {
  auto executor = co_await net::this_coro::executor;
  net::steady_timer timer(executor);
  while (fut.wait_for(std::chrono::milliseconds(0)) !=
         std::future_status::ready) {
    timer.expires_after(std::chrono::milliseconds(2));
    co_await timer.async_wait(net::use_awaitable);
  }
  co_return fut.get();
}

} // namespace

Orchestrator::Orchestrator(clients::EmbeddingClient &embedder,
                            IVectorSearchEngine &search_engine,
                            deltaEGO::deltaEGO &emotion_engine)
    : embedder_(embedder), search_engine_(search_engine),
      emotion_engine_(emotion_engine) {}

net::awaitable<schemas::FuliContextResponse>
Orchestrator::HandleContextRequest(schemas::FuliContextRequest req) {
  // [2] embedding — dense_vector is only ever set when something upstream
  // already computed it; normally Fuli embeds user_input itself.
  std::vector<float> query_vector;
  if (req.rag_policy.dense_vector.has_value()) {
    query_vector = *req.rag_policy.dense_vector;
  } else {
    query_vector = co_await embedder_.Embed(req.user_input);
  }

  // [3] 1st-pass retrieval. MVP: memory-domain dense search only —
  // hybrid/graph tiers and Redis metadata filtering are not wired in yet.
  SearchResult raw = co_await AwaitSearch(
      search_engine_.AsyncSearch(query_vector, req.rag_policy.top_k));

  // [4-A] emotion branch — OpenJEV stimulus extraction is deferred, so
  // this always feeds a neutral (0,0,0) stimulus into deltaEGO for now.
  std::string emotion_json = emotion_engine_.process_stimulus(0.0f, 0.0f, 0.0f);

  // [4-B] rerank/refine — deferred (enable_rerank, fusion, granularity
  // expansion all no-op in the MVP). Candidates pass straight through.
  schemas::FuliContextResponse resp;
  resp.hits.reserve(raw.ids.size());
  for (size_t i = 0; i < raw.ids.size(); ++i) {
    resp.hits.push_back({raw.ids[i], raw.distances[i]});
  }
  resp.emotion_json = std::move(emotion_json);

  co_return resp;
}

} // namespace pipeline
