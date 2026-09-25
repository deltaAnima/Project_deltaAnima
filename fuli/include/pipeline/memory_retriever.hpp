#pragma once

#include <boost/asio/awaitable.hpp>
#include <cstdint>
#include <optional>
#include <vector>

#include "clients/redis_db_client.hpp"
#include "engine/vector_search_engine.hpp"
#include "schemas/rag_schemas.hpp"

namespace pipeline {

// One retrieved memory, after Faiss's raw {id, distance} has been joined
// against Redis and filtered/scored per RAGQueryOrder.memory_config.
struct RetrievedMemory {
  int64_t id = 0;
  float distance = 0.0f; // raw Faiss L2 distance — lower = closer
  clients::MemoryMetadata metadata;
};

// The join layer GpuFaissEngine and RedisDbClient were both missing on
// their own: GpuFaissEngine only does raw ANN search (it has no idea
// what a session_id or importance_threshold is), and RedisDbClient only
// knows how to Get/Set one id's metadata at a time. MemoryRetriever is
// what Ayin is to Roland+Angela — it runs Faiss, joins every candidate
// against Redis, applies the filtering RAGQueryOrder.memory_config
// describes, and hands back a ranked, filtered, metadata-attached
// result list. Orchestrator no longer needs to know GpuFaissEngine or
// RedisDbClient exist at all.
class MemoryRetriever {
public:
  // Neither reference is owned — same lifetime rule as every other
  // class in this project that just borrows its dependencies (see
  // Orchestrator).
  MemoryRetriever(IVectorSearchEngine &search_engine,
                   clients::RedisDbClient &redis);

  // Runs a Faiss search, joins each candidate against Redis metadata,
  // drops anything that fails session_id/importance_threshold, and
  // returns up to top_k survivors sorted by distance (best first).
  // `config` is RAGQueryOrder.memory_config — std::nullopt means "no
  // memory-domain filtering requested", not "search nothing".
  boost::asio::awaitable<std::vector<RetrievedMemory>>
  Retrieve(const std::vector<float> &query_vector, int top_k,
           const std::optional<schemas::MemoryQueryConfig> &config);

  // The write-side counterpart to Retrieve: allocates a fresh id (via
  // Redis' NextId — the counter lives in Redis, not in-process, so ids
  // stay unique across restarts), adds `vector` to Faiss under that id,
  // then writes `metadata` to Redis under the same id (metadata.faiss_id
  // is overwritten with the allocated id regardless of what the caller
  // set it to). Returns the assigned id.
  //
  // This is what main.cpp's SeedTestVectors should eventually be
  // replaced by/call through — SeedTestVectors writes straight to Faiss
  // with fake ids and no Redis row at all, which is exactly why its
  // vectors never show up in Retrieve() once memory_config filtering (or
  // even just the "no metadata -> skip" rule) is involved.
  boost::asio::awaitable<int64_t> Store(const std::vector<float> &vector,
                                         clients::MemoryMetadata metadata);

private:
  IVectorSearchEngine &search_engine_;
  clients::RedisDbClient &redis_;
};

} // namespace pipeline
