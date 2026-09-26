#include "pipeline/orchestrator.hpp"
#include <iostream>
#include <boost/stacktrace.hpp>

namespace pipeline {

namespace net = boost::asio;

namespace {

// Shared logging shape for every catch site in this file — `where`
// names the call that failed (e.g. "HandleContextRequest (Embed)") so
// the two-line log + full stack trace can be told apart in stderr
// without needing a debugger attached after the fact.
void LogException(const char *where, const std::exception &e)
{
  std::cerr << "<Orchestrator::" << where << ">\n"
            << "[ERROR] Exception: " << e.what() << "\n"
            << "[STACKTRACE]:\n"
            << boost::stacktrace::stacktrace() << "\n";
}

void LogException(const char *where)
{
  std::cerr << "<Orchestrator::" << where << ">\n"
            << "[ERROR] Unknown exception occurred!\n"
            << "[STACKTRACE]:\n"
            << boost::stacktrace::stacktrace() << "\n";
}

} // namespace

Orchestrator::Orchestrator(clients::EmbeddingClient &embedder,
                            MemoryRetriever &memory_retriever,
                            deltaEGO::deltaEGO &emotion_engine)
    : embedder_(embedder), memory_retriever_(memory_retriever),
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
    // Rethrown after logging — everything downstream (search, response
    // assembly) depends on query_vector, so there's no sensible
    // degraded-but-still-succeeds path if this fails; the request
    // should still end in a 500 (via http_server.cpp's generic catch),
    // just with this structured log/stacktrace on the way out now.
    try
    {
      query_vector = co_await embedder_.Embed(reqest.user_input);
    }
    catch (const std::exception &e)
    {
      LogException("HandleContextRequest (Embed)", e);
      throw;
    }
    catch (...)
    {
      LogException("HandleContextRequest (Embed)");
      throw;
    }
  }

  // --- [3] First-pass retrieval -----------------------------------------
  // MemoryRetriever runs the Faiss search and joins each candidate
  // against Redis metadata, applying session_id/importance_threshold
  // filtering from memory_config. Still MVP scope beyond that: no
  // knowledge-base search, no hybrid (sparse+dense) fusion, no graph
  // traversal, no time-decay re-ranking yet (enable_time_decay/
  // recency_weight are parsed but not applied) — see
  // schemas/rag_schemas.hpp for exactly which RAGQueryOrder fields are
  // still parsed-but-currently-ignored.
  std::vector<pipeline::RetrievedMemory> retrieved;
  try
  {
    retrieved = co_await memory_retriever_.Retrieve(
        query_vector, reqest.rag_policy.top_k, reqest.rag_policy.memory_config);
  }
  catch (const std::exception &e)
  {
    LogException("HandleContextRequest (Retrieve)", e);
    throw;
  }
  catch (...)
  {
    LogException("HandleContextRequest (Retrieve)");
    throw;
  }

  // --- [4-A] Emotion branch -----------------------------------------------
  //TODO: integrate OpenJEV inference server to extract VAD stimulus from user_input
  //Current : only 5090
  std::string emotion_json;
  try
  {
    emotion_json = co_await this->emotion_engine_.sephirothic_tree(reqest.user_input);
  }
  catch (const std::exception &e)
  {
    LogException("HandleContextRequest (sephirothic_tree)", e);
    throw;
  }
  catch (...)
  {
    LogException("HandleContextRequest (sephirothic_tree)");
    throw;
  }

  // --- [4-B] Rerank/refine branch ------------------------------------------
  // Deferred: enable_rerank (TEI cross-encoder call), fusion_config
  // (memory+knowledge score blending), and granularity/context_window_size
  // (chunk expansion) are all no-ops here — MemoryRetriever's already-
  // filtered hits pass straight through unchanged. min_score_threshold is
  // parsed but not enforced yet either.
  schemas::FuliContextResponse resp;
  resp.hits.reserve(retrieved.size());
  for (const auto &mem : retrieved) 
  {
    schemas::MemoryHit hit;
    hit.id = mem.id;
    hit.score = mem.distance;
    hit.user_input = mem.metadata.memory.content.user_input;
    hit.model_response = mem.metadata.memory.content.model_response;
    resp.hits.push_back(std::move(hit));
  }
  resp.emotion_json = std::move(emotion_json);

  // --- [5] Assemble --------------------------------------------------------
  // (Just returning resp — the actual JSON serialization happens in
  // main.cpp via schemas::to_json, triggered by `json(resp).dump()`.)
  co_return resp;
}

boost::asio::awaitable<void>
Orchestrator::HandleContextSaveRequest(schemas::FuliContextSaveRequest req)
{
  // TODO: mem_buffer_ moved to MemoryRetriever's per-user map, but
  // FuliContextSaveRequest has no field yet (user_name/session_id) to
  // look the right entry back up with. Stubbed out — not calling
  // Store() at all — until that correlation is designed, so the build
  // isn't blocked on it in the meantime.
  (void)req;
  co_return;
}
} // namespace pipeline
