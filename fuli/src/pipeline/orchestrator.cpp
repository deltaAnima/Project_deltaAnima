#include "pipeline/orchestrator.hpp"
#include <chrono>
#include <iostream>
#include <boost/format.hpp>
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
void LogException(const char *where, const std::string& details)
{
  std::cerr << "<Orchestrator::" << where << ">\n"
            << "[ERROR] Logic Exception: " << details << "\n"
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
  std::pair<bool, clients::MemoryMetadata*> current_mem
                = this->memory_retriever_.get_memory_buff_(reqest.user_name, true);
  if(current_mem.first == false)
  {
    // get_memory_buff_ returns {false, nullptr} when a buffer for this
    // user already existed going into the retrieve stage — meaning a
    // previous turn's HandleContextSaveRequest never ran (or is still
    // in flight). current_mem.second is null here, so this has to stop
    // the request rather than fall through into the null deref below.
    std::string details = boost::str(boost::format(
        "User name %1%'s memory buffer already exists going into the "
        "retrieve stage — a previous turn's HandleContextSaveRequest "
        "never ran (or is still in flight), so the buffer is stale.")
        % reqest.user_name);
    LogException("HandleContextRequest (Hashing)", details);
    throw std::runtime_error(details);
  }

  // Session ownership lives in MemoryRetriever, not with the caller —
  // the same resolved id is used both as Retrieve()'s session_id filter
  // below and as this turn's new-memory tag in the assemble step, so a
  // conversation's own earlier turns stay recallable across the whole
  // session regardless of whatever (if anything) the caller passed in
  // rag_policy.memory_config.session_id.
  std::string session_id =
      this->memory_retriever_.GetOrCreateSessionId(reqest.user_name, reqest.new_session);

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
      query_vector = co_await this->embedder_.Embed(reqest.user_input);
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
  // session_id is always ours, not the caller's — overwritten here
  // regardless of whatever rag_policy.memory_config.session_id held, so
  // Retrieve()'s filter matches the same session this turn's new memory
  // will be tagged with in the assemble step below.
  schemas::MemoryQueryConfig memory_config =
      reqest.rag_policy.memory_config.value_or(schemas::MemoryQueryConfig{});
  memory_config.session_id = session_id;

  std::vector<pipeline::RetrievedMemory> retrieved;
  try
  {
    retrieved = co_await memory_retriever_.Retrieve(
        query_vector, reqest.rag_policy.top_k, memory_config);
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
  nlohmann::json emotion_json;
  std::string emotion_json_dump;
  try
  {
    emotion_json = co_await this->emotion_engine_.sephirothic_tree(reqest.user_input);
    emotion_json_dump = emotion_json.dump();
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
  // NOT a move — emotion_json_dump is read again below (assemble step,
  // emotion_analysis.deltaEGO_analysis) — moving it here left that
  // field permanently empty in Redis (it was always saving whatever a
  // moved-from std::string happens to leave behind), a real bug this
  // caught by inspecting a saved Redis row rather than something the
  // type system could flag.
  resp.emotion_json = emotion_json_dump;

  // --- [5] Assemble --------------------------------------------------------
  // Fills in current_mem.second (this user's per-turn buffer in
  // MemoryRetriever, keyed by MakeUserId(user_name)) with everything
  // HandleContextSaveRequest will need once the persona's response comes
  // back: the user's side of the turn, the emotion read taken from it,
  // and the original request for provenance. sephirothic_tree's json
  // only has a singular "emotion_term" (not a list), so it's wrapped in
  // a one-element vector to match Memory::Emotion::emotion_terms's shape.
  // VAD_Point's to_json/from_json are declared `inline` inside
  // deltaEGO.cpp, so they're only visible in that translation unit —
  // emotion_json's "current_state" fields have to be pulled out by hand
  // here rather than via emotion_json.get<VAD_Point>().
  current_mem.second->memory.content.user_input = reqest.user_input;
  current_mem.second->emotion_analysis.deltaEGO_analysis = emotion_json_dump;

  // VAD_Point's to_json (deltaEGO.cpp) writes lowercase single-letter
  // keys ("v"/"a"/"d"/"r"), not the struct's own field names — this bit
  // me once already when I wrote V/A/D/radius here the first time.
  const auto &current_state = emotion_json.at("current_state");
  current_mem.second->memory.emotion.current.V = current_state.at("v").get<float>();
  current_mem.second->memory.emotion.current.A = current_state.at("a").get<float>();
  current_mem.second->memory.emotion.current.D = current_state.at("d").get<float>();
  current_mem.second->memory.emotion.current.radius = current_state.at("r").get<float>();
  current_mem.second->memory.emotion.similarity = emotion_json.value("similarity", 0.0f);
  current_mem.second->memory.emotion.emotion_terms =
      {emotion_json.value("emotion_term", std::string())};

  current_mem.second->query.request_query = reqest;
  current_mem.second->metadata.session_id = session_id;

  co_return resp;
}

boost::asio::awaitable<nlohmann::json>
Orchestrator::HandleContextSaveRequest(schemas::FuliContextSaveRequest reqest)
{
  std::pair<bool, clients::MemoryMetadata*> current_mem
                = this->memory_retriever_.get_memory_buff_(reqest.user_name, false);

  if(current_mem.first == false)
  {
    std::string details = boost::str(boost::format(
        "User name %1%'s memory buffer is empty in save stage")
        % reqest.user_name);
    LogException("HandleContextSaveRequest (Hashing)", details);
    throw std::runtime_error(details);
  }

  // HandleContextRequest already filled in the user's half of the turn
  // (memory.content.user_input, memory.user.*, the emotion read, query.
  // request_query) when this buffer was created — the persona's half is
  // only known now that the save request has the actual response.
  // metadata.faiss_id is NOT set here: MemoryRetriever::Store fills it in
  // itself once Redis hands back the freshly-allocated id.
  current_mem.second->memory.content.model_response = reqest.persona_response;
  current_mem.second->memory.user.user_content = current_mem.second->memory.content.user_input;
  current_mem.second->memory.persona.persona_name = reqest.persona_name;
  current_mem.second->memory.persona.persona_content = reqest.persona_response;
  current_mem.second->metadata.timestamp =
      std::chrono::duration_cast<std::chrono::seconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count();

  std::string embedding_container = current_mem.second->memory.content.user_input
                                  + reqest.persona_response;

  // NOT rethrown — matches the shape you gave me earlier: log and
  // continue. A failed Store() still lets the buffer get cleaned up
  // below rather than leaking it or failing the whole save call over a
  // memory that (if this keeps failing) was never going to persist
  // anyway.
  try
  {
    std::vector<float> embedded_context = co_await this->embedder_.Embed(embedding_container);
    co_await this->memory_retriever_.Store(embedded_context, current_mem.second);
  }
  catch (const std::exception& e)
  {
    LogException("HandleContextSaveRequest (Store)", e);
  }
  catch (...)
  {
    LogException("HandleContextSaveRequest (Store)");
  }

  // Captured before erasing below — current_mem.second (and everything
  // it points to) is gone once delete_memory_buff_ runs. Built via
  // MemoryMetadata's to_json (see redis_db_client.hpp) regardless of
  // whether Store() above actually succeeded, so the caller can see
  // what was attempted either way.
  nlohmann::json saved_memory = *current_mem.second;

  // Always erase, success or failure — nothing else ever clears this
  // user's buffer, and HandleContextRequest's "buffer already exists"
  // check would trip on this user's very next turn otherwise.
  this->memory_retriever_.delete_memory_buff_(reqest.user_name);

  co_return saved_memory;
}
} // namespace pipeline
