#pragma once

#include <boost/asio/awaitable.hpp>
#include <string_view>
#include <functional>
#include <unordered_map>
#include <cstdint>
#include <optional>
#include <vector>
#include <variant>
#include <array>

#include "clients/redis_db_client.hpp"
#include "engine/vector_search_engine.hpp"
#include "schemas/rag_schemas.hpp"

namespace pipeline {

// One retrieved memory, after Faiss's raw {id, distance} has been joined
// against Redis and filtered/scored per RAGQueryOrder.memory_config.
struct RetrievedMemory 
{
  int64_t id = 0;
  float distance = 0.0f; // raw Faiss L2 distance — lower = closer
  clients::MemoryMetadata metadata;
};

struct UserId
{
  uint64_t high{0};
  uint64_t low{0};

  bool operator==(const UserId& other) const = default; // C++20
};

} // namespace pipeline

// A std::hash specialization has to live directly inside namespace std
// (fully qualifying the type) — it cannot be nested inside namespace
// pipeline, even though UserId is only ever used from within pipeline.
// It also has to appear here, before MemoryRetriever's std::unordered_map
// member is declared below: std::unordered_map<UserId, ...> instantiates
// std::hash<UserId> as soon as it's named, so a specialization written
// after that point is "after instantiation" — a separate hard error from
// the wrong-namespace one, not a fix for it. Putting it in the wrong
// scope, or the wrong place, doesn't just fail to be picked up; it
// surfaces as a wall of unrelated "use of deleted function" errors far
// from the real cause.
namespace std {
template <>
struct hash<pipeline::UserId>
{
  size_t operator()(const pipeline::UserId& id) const noexcept
  {
    return id.low ^ (id.high + 0x9e3779b97f4a7c15ULL + (id.low << 6) + (id.low >> 2));
  }
};
} // namespace std

namespace pipeline {

// The join layer GpuFaissEngine and RedisDbClient were both missing on
// their own: GpuFaissEngine only does raw ANN search (it has no idea
// what a session_id or importance_threshold is), and RedisDbClient only
// knows how to Get/Set one id's metadata at a time. MemoryRetriever is
// what Ayin is to Roland+Angela — it runs Faiss, joins every candidate
// against Redis, applies the filtering RAGQueryOrder.memory_config
// describes, and hands back a ranked, filtered, metadata-attached
// result list. Orchestrator no longer needs to know GpuFaissEngine or
// RedisDbClient exist at all.
class MemoryRetriever 
{
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
    clients::MemoryMetadata* metadata);

  std::pair<bool, clients::MemoryMetadata*>
    get_memory_buff_(const std::string& user_name, bool is_retrieve);

  bool delete_memory_buff_(const UserId& user_id);
  bool delete_memory_buff_(const std::string& user_name);

  // Session ownership lives here, not with the caller (Python): the
  // first call for a given user_name mints a fresh UUID and remembers
  // it in session_ids_ (separate from mem_buffer_, which is erased every
  // turn once a save completes — a session has to outlive any single
  // turn). Every later call for that same user_name gets the same id
  // back, so Retrieve()'s session_id filter and a new memory's
  // metadata.session_id stay consistent across a whole conversation.
  // force_new (driven by FuliContextRequest::new_session) skips the
  // lookup and always mints + remembers a fresh id instead — the one
  // lever the caller has over an otherwise-permanent-per-user_name
  // session.
  std::string GetOrCreateSessionId(const std::string& user_name, bool force_new = false);

private:
  inline uint64_t fmix64(uint64_t k);
  inline UserId MakeUserId(std::string_view key, uint64_t seed = 0);

  IVectorSearchEngine &search_engine_;
  clients::RedisDbClient &redis_;

  std::unordered_map<std::string, UserId> name_to_id_;
  std::unordered_map<UserId, std::unique_ptr<clients::MemoryMetadata>> mem_buffer_;
  std::unordered_map<UserId, std::string> session_ids_;
};

} // namespace pipeline
