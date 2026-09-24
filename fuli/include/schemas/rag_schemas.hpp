#pragma once

// C++ mirror of Persona/RAG_schemas.py::RAGQueryOrder. Field names and
// defaults must stay in sync with that file by hand — there is no shared
// codegen between the Python and C++ sides yet.
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "Third_Party/json.hpp"

namespace schemas {

using json = nlohmann::json;

enum class TargetDomain { MEMORY_ONLY, KNOWLEDGE_ONLY, BOTH };
NLOHMANN_JSON_SERIALIZE_ENUM(TargetDomain,
                              {
                                  {TargetDomain::MEMORY_ONLY, "memory"},
                                  {TargetDomain::KNOWLEDGE_ONLY, "knowledge"},
                                  {TargetDomain::BOTH, "both"},
                              })

enum class PipelineLevel { V1_DENSE, V1_5_HYBRID, V2_GRAPH_AWARE };
NLOHMANN_JSON_SERIALIZE_ENUM(
    PipelineLevel, {
                        {PipelineLevel::V1_DENSE, "v1_dense"},
                        {PipelineLevel::V1_5_HYBRID, "v1_5_hybrid"},
                        {PipelineLevel::V2_GRAPH_AWARE, "v2_graph_aware"},
                    })

enum class Granularity { RAW_CHUNK, PARENT_CHUNK, WINDOW_EXPANDED };
NLOHMANN_JSON_SERIALIZE_ENUM(
    Granularity, {
                     {Granularity::RAW_CHUNK, "raw_chunk"},
                     {Granularity::PARENT_CHUNK, "parent_chunk"},
                     {Granularity::WINDOW_EXPANDED, "window_expanded"},
                 })

struct MemoryQueryConfig {
  std::optional<std::string> session_id;
  std::optional<std::string> user_id;
  bool enable_time_decay = true;
  float recency_weight = 0.3f;
  float importance_threshold = 0.5f;
  std::optional<std::string> reference_timestamp; // kept as raw ISO string
};

inline void from_json(const json &j, MemoryQueryConfig &c) {
  if (j.contains("session_id") && !j.at("session_id").is_null())
    c.session_id = j.at("session_id").get<std::string>();
  if (j.contains("user_id") && !j.at("user_id").is_null())
    c.user_id = j.at("user_id").get<std::string>();
  c.enable_time_decay = j.value("enable_time_decay", true);
  c.recency_weight = j.value("recency_weight", 0.3f);
  c.importance_threshold = j.value("importance_threshold", 0.5f);
  if (j.contains("reference_timestamp") &&
      !j.at("reference_timestamp").is_null())
    c.reference_timestamp = j.at("reference_timestamp").get<std::string>();
}

struct KnowledgeQueryConfig {
  std::vector<std::string> collection_names{"default_kb"};
  std::optional<std::map<std::string, float>> sparse_vector;
  float hybrid_alpha = 0.7f;
  std::optional<std::vector<std::string>> seed_entities;
  int max_hops = 1;
};

inline void from_json(const json &j, KnowledgeQueryConfig &c) {
  c.collection_names =
      j.value("collection_names", std::vector<std::string>{"default_kb"});
  if (j.contains("sparse_vector") && !j.at("sparse_vector").is_null())
    c.sparse_vector = j.at("sparse_vector").get<std::map<std::string, float>>();
  c.hybrid_alpha = j.value("hybrid_alpha", 0.7f);
  if (j.contains("seed_entities") && !j.at("seed_entities").is_null())
    c.seed_entities = j.at("seed_entities").get<std::vector<std::string>>();
  c.max_hops = j.value("max_hops", 1);
}

struct DualTrackFusionConfig {
  float memory_weight = 0.4f;
  float knowledge_weight = 0.6f;
  bool interleave_results = false;
};

inline void from_json(const json &j, DualTrackFusionConfig &c) {
  c.memory_weight = j.value("memory_weight", 0.4f);
  c.knowledge_weight = j.value("knowledge_weight", 0.6f);
  c.interleave_results = j.value("interleave_results", false);
}

struct RAGQueryOrder {
  TargetDomain target_domain = TargetDomain::BOTH;
  PipelineLevel pipeline_level = PipelineLevel::V1_5_HYBRID;
  std::optional<std::vector<float>> dense_vector; // null => Fuli embeds it
  int top_k = 5;
  float min_score_threshold = 0.6f;
  std::optional<MemoryQueryConfig> memory_config;
  std::optional<KnowledgeQueryConfig> knowledge_config;
  DualTrackFusionConfig fusion_config;
  Granularity granularity = Granularity::RAW_CHUNK;
  int context_window_size = 1;
  bool enable_rerank = false;
};

inline void from_json(const json &j, RAGQueryOrder &o) {
  o.target_domain = j.value("target_domain", TargetDomain::BOTH);
  o.pipeline_level =
      j.value("pipeline_level", PipelineLevel::V1_5_HYBRID);
  if (j.contains("dense_vector") && !j.at("dense_vector").is_null())
    o.dense_vector = j.at("dense_vector").get<std::vector<float>>();
  o.top_k = j.value("top_k", 5);
  o.min_score_threshold = j.value("min_score_threshold", 0.6f);
  if (j.contains("memory_config") && !j.at("memory_config").is_null())
    o.memory_config = j.at("memory_config").get<MemoryQueryConfig>();
  if (j.contains("knowledge_config") && !j.at("knowledge_config").is_null())
    o.knowledge_config = j.at("knowledge_config").get<KnowledgeQueryConfig>();
  if (j.contains("fusion_config"))
    o.fusion_config = j.at("fusion_config").get<DualTrackFusionConfig>();
  o.granularity = j.value("granularity", Granularity::RAW_CHUNK);
  o.context_window_size = j.value("context_window_size", 1);
  o.enable_rerank = j.value("enable_rerank", false);
}

} // namespace schemas
