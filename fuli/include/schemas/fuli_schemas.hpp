#pragma once

// Mirrors the payload FuliHandler._post("/character/context", payload)
// sends and the {"hits": [...], "emotion": {...}} shape it reads back.
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
  std::optional<std::string> context;
  RAGQueryOrder rag_policy;
  // emotion_policy (OCEAN/weights/backend_flags) is accepted but not yet
  // applied — deltaEGO still runs off its startup YAML. Per-character
  // policy override is deferred past the MVP.
  json emotion_policy_raw = json::object();
};

inline void from_json(const json &j, FuliContextRequest &r) {
  j.at("user_name").get_to(r.user_name);
  j.at("user_input").get_to(r.user_input);
  if (j.contains("context") && !j.at("context").is_null())
    r.context = j.at("context").get<std::string>();

  const auto &cfg = j.at("config");
  r.rag_policy = cfg.at("rag_policy").get<RAGQueryOrder>();
  r.emotion_policy_raw = cfg.value("emotion_policy", json::object());
}

struct MemoryHit {
  int64_t id = 0;
  float score = 0.0f; // raw Faiss L2 distance for now, not yet re-ranked
};

struct FuliContextResponse {
  std::vector<MemoryHit> hits;
  // process_stimulus() already returns a JSON string; we pass it through
  // verbatim instead of re-modeling deltaEGO's analysis result in C++.
  std::string emotion_json = "{}";
};

inline void to_json(json &j, const FuliContextResponse &r) {
  j["hits"] = json::array();
  for (const auto &h : r.hits) {
    j["hits"].push_back({{"id", h.id}, {"score", h.score}});
  }
  j["emotion"] = json::parse(r.emotion_json, nullptr, /*allow_exceptions=*/false);
  if (j["emotion"].is_discarded())
    j["emotion"] = json::object();
}

} // namespace schemas
