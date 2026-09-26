#pragma once

// Async Redis client (boost::redis) for memory metadata — the piece
// that will let a future Faiss+Redis retrieval layer turn a bare
// {id, distance} hit from GpuFaissEngine into an actual scoped, scored
// memory (session_id/user_id filtering, importance_threshold,
// enable_time_decay). See the orchestrator.cpp discussion: GpuFaissEngine
// only does raw ANN search and knows nothing about any of that — this
// client is where that metadata actually lives.
#include <boost/asio/awaitable.hpp>
#include <boost/asio/io_context.hpp>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "Third_Party/json.hpp"

#include "schemas/fuli_schemas.hpp"

// Forward-declared instead of #including <boost/redis/connection.hpp> —
// same reasoning as deltaEGO.hpp hiding AVX-512/yaml-cpp: consumers of
// this header (the future retrieval layer, main.cpp) don't need to see
// boost::redis's internals, just call through the pointer.
namespace boost {
namespace redis {
class connection;
} // namespace redis
} // namespace boost

namespace clients {

// One row of memory metadata, keyed in Redis by the same int64 id
// GpuFaissEngine's AsyncSearch returns (see gpu_faiss_engine.cpp's
// id_map_). Faiss only stores the vector — this is where the actual
// text and the fields RAGQueryOrder's MemoryQueryConfig filters on
// (session_id, user_id, importance, timestamp) live.
//
// Types are PascalCase, instance fields are the matching lowercase name
// (Memory -> memory, User -> user, ...) so access reads as
// `meta.memory.content.user_input`, `meta.metadata.importance`, etc.
// They can't share the exact same spelling — a struct and a data member
// with an identical name in the same scope is a real compile error, not
// just a style nit — so the case difference is load-bearing, not
// decorative.
struct MemoryMetadata
{
  struct Memory
  {
    struct Content
    {
      std::string user_input;
      std::string model_response;
    } content;

    struct User
    {
      int user_id = 0;           // unique id of the user (currently TODO)
      std::string user_name;     // name of the user
      std::string user_content;  // same as user input
    } user;

    struct Persona
    {
      std::string persona_name = "Reminh"; // character name
      std::string persona_content;         // same as model response
    } persona;

    struct Emotion
    {
      std::vector<std::string> emotion_terms;
      float intensity = 0.0f;
    } emotion;
  } memory;

  struct EmotionAnalysis
  {
    std::string deltaEGO_analysis; // Result of std::string deltaEGO::process_stimulus
  } emotion_analysis;

  struct Metadata
  {
    std::optional<std::string> session_id;
    float importance = 0.5f;  // matches importance_threshold's 0..1 scale
    int64_t timestamp = 0;    // unix epoch seconds, for time-decay scoring
    int64_t faiss_id = 0;     // the id that GpuFaissEngine uses to retrieve the vector
  } metadata;

  struct Query
  {
    schemas::FuliContextRequest request_query;
  } query;
};

class RedisDbClient 
{
public:
  RedisDbClient(boost::asio::io_context &ioc, std::string host,
                std::string port);
  ~RedisDbClient();

  RedisDbClient(const RedisDbClient &) = delete;
  RedisDbClient &operator=(const RedisDbClient &) = delete;

  // Starts the background connection. boost::redis reconnects on its own
  // if the connection drops — call this once from main(), before any
  // Get/SetMemoryMetadata call, and keep this object alive for the
  // server's whole lifetime (one connection total, not one per request).
  void Run();

  // HGETALL mem:{faiss_id}. Returns std::nullopt if that key doesn't
  // exist (nothing stored for this id yet — e.g. seeded test vectors
  // that were never given real metadata).
  boost::asio::awaitable<std::optional<MemoryMetadata>>
  GetMemoryMetadata(int64_t faiss_id);

  // HSET mem:{faiss_id} ... — overwrites any existing fields for that id.
  boost::asio::awaitable<void> SetMemoryMetadata(int64_t faiss_id,
                                                  const MemoryMetadata &meta);

  // INCR on a dedicated counter key — atomically allocates and returns a
  // fresh, globally unique memory id. This is what a new memory gets
  // stored under, both in Faiss (see MemoryRetriever::Store) and here in
  // Redis, so Faiss's id_map_ and Redis's mem:{id} keys never collide
  // even across restarts (the counter lives in Redis, not in-process).
  boost::asio::awaitable<int64_t> NextId();

  // Blocking (same pattern as EmbeddingClient::HealthCheck, so
  // HealthChecker::CheckAll can std::async both the same way) — a real
  // PING through a throwaway connection, not just a TCP connect, so it
  // actually confirms Redis is speaking RESP and not just that the port
  // happens to be open.
  nlohmann::json HealthCheck() const;

private:
  boost::asio::io_context &ioc_;
  std::string host_;
  std::string port_;
  std::shared_ptr<boost::redis::connection> conn_;

  std::unordered_map<std::string, uint64_t> name_to_id_;
  std::unordered_map<uint64_t, std::unique_ptr<clients::MemoryMetadata>> mem_buffer_;
};

} // namespace clients
