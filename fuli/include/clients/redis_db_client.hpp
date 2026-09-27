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
#include "deltaEGO/deltaEGO.hpp"
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
      struct UserId
      {
        uint64_t high{0};
        uint64_t low{0};
      } user_id;
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
      deltaEGO::structs::VAD_Point current;
      std::vector<std::string> emotion_terms;
      float similarity;
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

// ADL to_json/from_json for MemoryMetadata and every nested struct it's
// built from — same convention as schemas/fuli_schemas.hpp and
// schemas/rag_schemas.hpp (custom functions, not the
// NLOHMANN_DEFINE_TYPE_* macros, so optional/default-value fields can
// use the same defensive .value()/contains() style those files use).
// This gives MemoryMetadata a real, nested JSON shape (mirroring the
// struct itself), which is NOT the same thing as the flat dot-path
// fields RedisDbClient actually stores in a Redis hash (see
// redis_db_client.cpp's FieldsToMetadata/MetadataToFields) — that
// flattening is a Redis-storage detail, not this type's JSON
// representation.
//
// emotion.current (a deltaEGO::structs::VAD_Point) is handled by hand
// here rather than via VAD_Point's own to_json/from_json: those are
// declared `inline` inside deltaEGO.cpp, so they're only visible in
// that one translation unit (see orchestrator.cpp's HandleContextRequest
// for the same limitation) — "v"/"a"/"d"/"r" below matches those
// functions' real key names exactly, for consistency with the JSON
// sephirothic_tree/process_stimulus already produce.
inline void to_json(nlohmann::json &j, const MemoryMetadata::Memory::Content &c) {
  j = nlohmann::json{{"user_input", c.user_input},
                      {"model_response", c.model_response}};
}
inline void from_json(const nlohmann::json &j, MemoryMetadata::Memory::Content &c) {
  c.user_input = j.value("user_input", std::string());
  c.model_response = j.value("model_response", std::string());
}

inline void to_json(nlohmann::json &j, const MemoryMetadata::Memory::User::UserId &id) {
  j = nlohmann::json{{"high", id.high}, {"low", id.low}};
}
inline void from_json(const nlohmann::json &j, MemoryMetadata::Memory::User::UserId &id) {
  id.high = j.value("high", uint64_t{0});
  id.low = j.value("low", uint64_t{0});
}

inline void to_json(nlohmann::json &j, const MemoryMetadata::Memory::User &u) {
  j = nlohmann::json{{"user_id", u.user_id},
                      {"user_name", u.user_name},
                      {"user_content", u.user_content}};
}
inline void from_json(const nlohmann::json &j, MemoryMetadata::Memory::User &u) {
  u.user_id = j.value("user_id", MemoryMetadata::Memory::User::UserId{});
  u.user_name = j.value("user_name", std::string());
  u.user_content = j.value("user_content", std::string());
}

inline void to_json(nlohmann::json &j, const MemoryMetadata::Memory::Persona &p) {
  j = nlohmann::json{{"persona_name", p.persona_name},
                      {"persona_content", p.persona_content}};
}
inline void from_json(const nlohmann::json &j, MemoryMetadata::Memory::Persona &p) {
  p.persona_name = j.value("persona_name", std::string("Reminh"));
  p.persona_content = j.value("persona_content", std::string());
}

inline void to_json(nlohmann::json &j, const MemoryMetadata::Memory::Emotion &e) {
  j = nlohmann::json{
      {"current", {{"v", e.current.V}, {"a", e.current.A},
                   {"d", e.current.D}, {"r", e.current.radius}}},
      {"emotion_terms", e.emotion_terms},
      {"similarity", e.similarity}};
}
inline void from_json(const nlohmann::json &j, MemoryMetadata::Memory::Emotion &e) {
  const auto &current = j.value("current", nlohmann::json::object());
  e.current.V = current.value("v", 0.0f);
  e.current.A = current.value("a", 0.0f);
  e.current.D = current.value("d", 0.0f);
  e.current.radius = current.value("r", 0.0f);
  e.emotion_terms = j.value("emotion_terms", std::vector<std::string>());
  e.similarity = j.value("similarity", 0.0f);
}

inline void to_json(nlohmann::json &j, const MemoryMetadata::Memory &m) {
  j = nlohmann::json{{"content", m.content},
                      {"user", m.user},
                      {"persona", m.persona},
                      {"emotion", m.emotion}};
}
inline void from_json(const nlohmann::json &j, MemoryMetadata::Memory &m) {
  m.content = j.value("content", MemoryMetadata::Memory::Content{});
  m.user = j.value("user", MemoryMetadata::Memory::User{});
  m.persona = j.value("persona", MemoryMetadata::Memory::Persona{});
  m.emotion = j.value("emotion", MemoryMetadata::Memory::Emotion{});
}

// deltaEGO_analysis is a std::string holding a JSON blob (it's
// deltaEGO::process_stimulus's dumped output — see the field's own
// comment), not a real nlohmann::json member — serializing it verbatim
// would double-encode it into an escaped string-within-a-string. Parse
// it back into a real nested object here so callers of MemoryMetadata's
// to_json get proper nested JSON instead of an escaped blob; fall back
// to embedding the raw string if it isn't valid JSON (e.g. legacy/
// malformed data) rather than losing it.
inline void to_json(nlohmann::json &j, const MemoryMetadata::EmotionAnalysis &ea) {
  nlohmann::json parsed = nlohmann::json::parse(ea.deltaEGO_analysis, nullptr,
                                                 /*allow_exceptions=*/false);
  j = nlohmann::json{
      {"deltaEGO_analysis", parsed.is_discarded() ? nlohmann::json(ea.deltaEGO_analysis) : parsed}};
}
inline void from_json(const nlohmann::json &j, MemoryMetadata::EmotionAnalysis &ea) {
  if (!j.contains("deltaEGO_analysis")) {
    ea.deltaEGO_analysis.clear();
    return;
  }
  const nlohmann::json &v = j.at("deltaEGO_analysis");
  // Accept either shape back: a plain string (the legacy/Redis-stored
  // form) or the real nested object to_json now produces — either way,
  // deltaEGO_analysis itself stays a std::string.
  ea.deltaEGO_analysis = v.is_string() ? v.get<std::string>() : v.dump();
}

inline void to_json(nlohmann::json &j, const MemoryMetadata::Metadata &md) {
  j = nlohmann::json{
      {"session_id", md.session_id ? nlohmann::json(*md.session_id) : nlohmann::json(nullptr)},
      {"importance", md.importance},
      {"timestamp", md.timestamp},
      {"faiss_id", md.faiss_id}};
}
inline void from_json(const nlohmann::json &j, MemoryMetadata::Metadata &md) {
  if (j.contains("session_id") && !j.at("session_id").is_null())
    md.session_id = j.at("session_id").get<std::string>();
  md.importance = j.value("importance", 0.5f);
  md.timestamp = j.value("timestamp", int64_t{0});
  md.faiss_id = j.value("faiss_id", int64_t{0});
}

inline void to_json(nlohmann::json &j, const MemoryMetadata::Query &q) {
  j = nlohmann::json{{"request_query", q.request_query}};
}
inline void from_json(const nlohmann::json &j, MemoryMetadata::Query &q) {
  q.request_query = j.value("request_query", schemas::FuliContextRequest{});
}

inline void to_json(nlohmann::json &j, const MemoryMetadata &meta) {
  j = nlohmann::json{{"memory", meta.memory},
                      {"emotion_analysis", meta.emotion_analysis},
                      {"metadata", meta.metadata},
                      {"query", meta.query}};
}
inline void from_json(const nlohmann::json &j, MemoryMetadata &meta) {
  meta.memory = j.value("memory", MemoryMetadata::Memory{});
  meta.emotion_analysis = j.value("emotion_analysis", MemoryMetadata::EmotionAnalysis{});
  meta.metadata = j.value("metadata", MemoryMetadata::Metadata{});
  meta.query = j.value("query", MemoryMetadata::Query{});
}

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
                                                  const MemoryMetadata* meta);

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

};

} // namespace clients
