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

  const auto &cfg = j.at("config");
  r.rag_policy = cfg.at("rag_policy").get<RAGQueryOrder>();
  r.emotion_policy_raw = cfg.value("emotion_policy", json::object());
}

// One retrieved memory/knowledge chunk. `score` is currently the raw
// Faiss L2 distance (lower = more similar), NOT a normalized similarity
// score and NOT re-ranked — see Orchestrator::HandleContextRequest.
struct MemoryHit {
  int64_t id = 0;
  float score = 0.0f;
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
inline void to_json(json &j, const FuliContextResponse &r) {
  j["hits"] = json::array();
  for (const auto &h : r.hits) {
    j["hits"].push_back({{"id", h.id}, {"score", h.score}});
  }

  // allow_exceptions=false makes json::parse return a "discarded" value
  // instead of throwing if emotion_json somehow isn't valid JSON — we'd
  // rather send back an empty object than crash the whole response.
  j["emotion"] = json::parse(r.emotion_json, nullptr, /*allow_exceptions=*/false);
  if (j["emotion"].is_discarded())
    j["emotion"] = json::object();
}

} // namespace schemas
