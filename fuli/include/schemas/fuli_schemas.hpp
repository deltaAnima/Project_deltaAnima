#pragma once

// Top-level request/response shapes for the /character/context endpoint.
// These mirror exactly what Python's FuliHandler sends and reads:
//
//   payload = {
//       "user_name": ..., "user_input": ..., "context": ...,
//       "config": {"rag_policy": {...}, "emotion_policy": {...}},
//   }
//   response.get("hits", [...]), response.get("emotion")
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

struct FuliContextRequest {
  std::string user_name;
  std::string user_input;
  std::optional<std::string> context; // free-form extra context, may be null

  // Session identity/lifetime is owned by MemoryRetriever (see
  // GetOrCreateSessionId), not by the caller — this is the one signal
  // Python has over it: true forces a fresh session id for user_name
  // instead of reusing whatever's on file, e.g. when Python knows this
  // is the start of a new conversation rather than a continuation.
  bool new_session = false;

  RAGQueryOrder rag_policy; // parsed from config.rag_policy

  // config.emotion_policy (OCEAN traits, physics weights, backend_flags
  // like use_openjev/use_5090) is accepted here as raw, unparsed JSON.
  // We deliberately do NOT turn this into a typed struct yet: deltaEGO
  // currently loads its OCEAN/weights once at server startup from
  // config/deltaEGO_default.yaml and ignores anything per-request. Wiring
  // per-character emotion policy overrides (and the OpenJEV text->VAD
  // call) is future work, tracked in the project conversation — not
  // implemented in this MVP.
  json emotion_policy_raw = json::object();
};

// Custom from_json (instead of the NLOHMANN_DEFINE_TYPE_* macros) because
// the incoming JSON has a nested "config" wrapper object that doesn't
// map 1:1 onto this struct's flat field layout — rag_policy and
// emotion_policy_raw both live under request["config"], not at the top
// level.
inline void from_json(const json &j, FuliContextRequest &r) {
  j.at("user_name").get_to(r.user_name);
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

struct FuliContextResponse {
  std::vector<MemoryHit> hits;

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
  std::string persona_name;
  std::string persona_response;
};

// Parses what Python sends. Required fields use .get_to() (throws if
// missing) — same defensive convention as FuliContextRequest::from_json
// above, just simpler here since both fields are plain strings.
inline void from_json(const json &j, FuliContextSaveRequest &r)
{
  j.at("persona_name").get_to(r.persona_name);
  j.at("persona_response").get_to(r.persona_response);
}

// Mirror of from_json — mainly so this struct can round-trip if it ever
// gets embedded elsewhere for storage, the same reason
// FuliContextRequest has one (see MemoryMetadata::Query in
// redis_db_client.hpp).
inline void to_json(json &j, const FuliContextSaveRequest &r)
{
  j["persona_name"]     = r.persona_name;
  j["persona_response"] = r.persona_response;
}

} // namespace schemas
