#pragma once

// Top-level request/response shapes for the /character/context endpoint.
// These mirror exactly what Python's FuliHandler sends and reads:
//
//   payload = {
//       "user_name": ..., "user_key": ..., "scope_key": ...,
//       "user_input": ..., "context": ...,
//       "config": {"rag_policy": {...}, "emotion_policy": {...}},
//   }
//   response.get("hits", [...]), response.get("recent_turns", [...]),
//   response.get("emotion")
//
// See main_orch_files/http_request_format.json for a full worked example
// of the wire format, and RAGHandler.py / FuliHandler.py for how the
// Python side builds/consumes it.
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "Third_Party/json.hpp"
#include "schemas/rag_schemas.hpp"

namespace schemas {

using json = nlohmann::json;

// Fallback for clients that don't send user_key yet — every pre-user_key
// memory was keyed off user_name alone, so "legacy:{user_name}" is the
// key those existing records get migrated onto (see the migration
// script), which keeps old clients connected to their old memories.
inline std::string ResolveUserKey(const json &j, const std::string &user_name) {
  if (j.contains("user_key") && j.at("user_key").is_string() &&
      !j.at("user_key").get<std::string>().empty())
    return j.at("user_key").get<std::string>();
  return "legacy:" + user_name;
}

inline std::optional<std::string> ReadScopeKey(const json &j) {
  if (j.contains("scope_key") && j.at("scope_key").is_string() &&
      !j.at("scope_key").get<std::string>().empty())
    return j.at("scope_key").get<std::string>();
  return std::nullopt;
}

struct FuliContextRequest {
  // Display name only — identity (id hash, buffer/session ownership,
  // retrieval scope) is user_key's job. Names collide and change
  // (Discord nicknames differ per server), platform ids don't.
  std::string user_name;

  // Stable per-platform identity, e.g. "unity:{uuid}" or
  // "discord:{snowflake}". Falls back to "legacy:{user_name}" when the
  // caller doesn't send one (see ResolveUserKey).
  std::string user_key;

  // Where this turn happens, e.g. "discord:{guild}:{channel}" or
  // "discord:dm:{user}". Optional — legacy/Unity clients may omit it.
  std::optional<std::string> scope_key;

  std::string user_input;
  std::optional<std::string> context; // free-form extra context, may be null

  // Session identity/lifetime is owned by MemoryRetriever (see
  // GetOrCreateSessionId), not by the caller — this is the one signal
  // Python has over it: true forces a fresh session id for user_name
  // instead of reusing whatever's on file, e.g. when Python knows this
  // is the start of a new conversation rather than a continuation.
  bool new_session = false;

  RAGQueryOrder rag_policy; // parsed from config.rag_policy

  // config.emotion_policy (OCEAN traits, physics weights, backend flags
  // like use_openjev/use_5090), kept as raw JSON — deltaEGO reads it
  // directly (see deltaEGO::sephirothic_tree): weight sections override
  // config/deltaEGO_default.yaml for this turn, flags pick the VAD
  // backend.
  json emotion_policy_raw = json::object();
};

// Custom from_json (instead of the NLOHMANN_DEFINE_TYPE_* macros) because
// the incoming JSON has a nested "config" wrapper object that doesn't
// map 1:1 onto this struct's flat field layout — rag_policy and
// emotion_policy_raw both live under request["config"], not at the top
// level.
inline void from_json(const json &j, FuliContextRequest &r) {
  j.at("user_name").get_to(r.user_name);
  r.user_key = ResolveUserKey(j, r.user_name);
  r.scope_key = ReadScopeKey(j);
  j.at("user_input").get_to(r.user_input);
  if (j.contains("context") && !j.at("context").is_null())
    r.context = j.at("context").get<std::string>();
  r.new_session = j.value("new_session", false);

  const auto &cfg = j.at("config");
  r.rag_policy = cfg.at("rag_policy").get<RAGQueryOrder>();
  r.emotion_policy_raw = cfg.value("emotion_policy", json::object());
}

// Mirrors from_json's shape (nested "config" wrapper) so storing a
// request (see MemoryMetadata::Query in redis_db_client.hpp) and later
// parsing it back with from_json round-trips correctly.
inline void to_json(json &j, const FuliContextRequest &r) {
  j = json{
      {"user_name", r.user_name},
      {"user_key", r.user_key},
      {"scope_key", r.scope_key ? json(*r.scope_key) : json(nullptr)},
      {"user_input", r.user_input},
      {"context", r.context ? json(*r.context) : json(nullptr)},
      {"new_session", r.new_session},
      {"config",
       {
           {"rag_policy", r.rag_policy},
           {"emotion_policy", r.emotion_policy_raw},
       }},
  };
}

// One retrieved memory/knowledge chunk. `score` is currently the raw
// Faiss L2 distance (lower = more similar), NOT a normalized similarity
// score and NOT re-ranked — see Orchestrator::HandleContextRequest.
// user_input/model_response come from Redis (see
// pipeline::MemoryRetriever + clients::MemoryMetadata::Memory::Content)
// — without them a hit is just an opaque id+score, useless to whatever
// reads this response.
struct MemoryHit {
  int64_t id = 0;
  float score = 0.0f;
  std::string user_input;
  std::string model_response;
};

// One of a conversation's latest turns, straight from the recent:{key}
// list in Redis (see MemoryRetriever::GetRecentTurns) — returned in
// time order regardless of similarity, so "what was just said" never
// depends on winning the Faiss search. id is the turn's memory id
// (0 if Store() failed for it), the same id a MemoryHit would carry.
// user_key/user_name say who spoke: a group channel's list holds every
// member's turns.
struct RecentTurn {
  int64_t id = 0;
  std::string user_key;
  std::string user_name;
  std::string user_input;
  std::string persona_name;
  std::string model_response;
  int64_t timestamp = 0;
};

inline void to_json(json &j, const RecentTurn &t) {
  j = json{{"id", t.id},
           {"user_key", t.user_key},
           {"user_name", t.user_name},
           {"user_input", t.user_input},
           {"persona_name", t.persona_name},
           {"model_response", t.model_response},
           {"timestamp", t.timestamp}};
}
inline void from_json(const json &j, RecentTurn &t) {
  t.id = j.value("id", int64_t{0});
  t.user_key = j.value("user_key", std::string());
  t.user_name = j.value("user_name", std::string());
  t.user_input = j.value("user_input", std::string());
  t.persona_name = j.value("persona_name", std::string());
  t.model_response = j.value("model_response", std::string());
  t.timestamp = j.value("timestamp", int64_t{0});
}

struct FuliContextResponse {
  std::vector<MemoryHit> hits;

  // Oldest first, so it reads as a transcript. Turns listed here are
  // removed from `hits` — the caller never sees the same turn twice.
  std::vector<RecentTurn> recent_turns;

  // deltaEGO::process_stimulus() already returns a fully-formed JSON
  // string (current VAD state + full physics analysis breakdown). Rather
  // than re-modeling that structure as a C++ type, we just store the raw
  // string here and re-parse it once at serialization time (see to_json
  // below). This keeps deltaEGO's output format as the single source of
  // truth instead of duplicating its shape in two places.
  std::string emotion_json = "{}";
};

// to_json is the mirror of from_json: nlohmann calls this automatically
// when you do `json(some_response)` or `j = some_response`.
inline void to_json(json &j, const FuliContextResponse &r) 
{
  j["hits"] = json::array();
  for (const auto &h : r.hits) {
    j["hits"].push_back({{"id", h.id},
                          {"score", h.score},
                          {"user_input", h.user_input},
                          {"model_response", h.model_response}});
  }
  j["recent_turns"] = r.recent_turns;

  // allow_exceptions=false makes json::parse return a "discarded" value
  // instead of throwing if emotion_json somehow isn't valid JSON — we'd
  // rather send back an empty object than crash the whole response.
  j["emotion"] = json::parse(r.emotion_json, nullptr, /*allow_exceptions=*/false);
  if (j["emotion"].is_discarded())
    j["emotion"] = json::object();
}

// Body for the (future) write-side endpoint — see the project discussion
// on why /character/context can't auto-save memories itself: it only
// ever sees user_input, never the model's reply, since that gets
// generated in Python AFTER this server responds. This is what the
// Python side posts back once it has persona_response, so the turn can
// actually be embedded + written via MemoryRetriever::Store.
struct FuliContextSaveRequest
{
  std::string user_name;
  // Same meaning/fallback as FuliContextRequest's — together they name
  // the per-turn buffer HandleContextRequest left behind, so they have
  // to match what the context request sent.
  std::string user_key;
  std::optional<std::string> scope_key;
  std::string persona_name;
  std::string persona_response;
};

// Parses what Python sends. Required fields use .get_to() (throws if
// missing) — same defensive convention as FuliContextRequest::from_json
// above, just simpler here since both fields are plain strings.
inline void from_json(const json &j, FuliContextSaveRequest &r)
{
  j.at("user_name").get_to(r.user_name);
  r.user_key = ResolveUserKey(j, r.user_name);
  r.scope_key = ReadScopeKey(j);
  j.at("persona_name").get_to(r.persona_name);
  j.at("persona_response").get_to(r.persona_response);
}

// Mirror of from_json — mainly so this struct can round-trip if it ever
// gets embedded elsewhere for storage, the same reason
// FuliContextRequest has one (see MemoryMetadata::Query in
// redis_db_client.hpp).
inline void to_json(json &j, const FuliContextSaveRequest &r)
{
  j["user_name"]        = r.user_name;
  j["user_key"]         = r.user_key;
  j["scope_key"]        = r.scope_key ? json(*r.scope_key) : json(nullptr);
  j["persona_name"]     = r.persona_name;
  j["persona_response"] = r.persona_response;
}

// Body for /character/context_cancel — sent instead of the save request
// when a turn won't be saved (inference failed, empty reply). Only the
// identity fields matter: they name the buffer to drop.
struct FuliContextCancelRequest
{
  std::string user_name;
  std::string user_key;
  std::optional<std::string> scope_key;
};

inline void from_json(const json &j, FuliContextCancelRequest &r)
{
  j.at("user_name").get_to(r.user_name);
  r.user_key = ResolveUserKey(j, r.user_name);
  r.scope_key = ReadScopeKey(j);
}

} // namespace schemas
