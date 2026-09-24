#pragma once

// C++ mirror of Persona/RAG_schemas.py::RAGQueryOrder (a pydantic model on
// the Python side). Every struct/enum here exists purely so we can parse
// the JSON that FuliHandler sends us — there is no shared codegen between
// the Python and C++ sides, so if someone changes a field name or default
// in RAG_schemas.py, this file has to be updated by hand to match.
//
// How the parsing works: nlohmann::json looks for a free function
// `void from_json(const json&, T&)` in the same namespace as T, and calls
// it automatically whenever you do `j.get<T>()` or `j.at("key").get<T>()`.
// That's why every struct below is followed by its own from_json().
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "Third_Party/json.hpp"

namespace schemas {

using json = nlohmann::json;

// --- Enums --------------------------------------------------------------
// Python's RAGQueryOrder uses `str, Enum` classes, so on the wire these
// arrive as plain strings like "memory" or "v1_5_hybrid", not integers.
// NLOHMANN_JSON_SERIALIZE_ENUM wires up from_json/to_json for an enum
// class given a list of (enum value, JSON string) pairs. The first pair
// listed is also the value used if we ever need to serialize an
// out-of-range int back to JSON (not something we rely on here).

// Which corpus to search: memory (conversation history), knowledge
// (documents/wiki), or both (search both, then fuse the two result sets).
enum class TargetDomain { MEMORY_ONLY, KNOWLEDGE_ONLY, BOTH };
NLOHMANN_JSON_SERIALIZE_ENUM(TargetDomain,
                              {
                                  {TargetDomain::MEMORY_ONLY, "memory"},
                                  {TargetDomain::KNOWLEDGE_ONLY, "knowledge"},
                                  {TargetDomain::BOTH, "both"},
                              })

// How sophisticated the retrieval should be. Only V1_DENSE (plain
// embedding similarity search) is actually implemented right now — see
// pipeline/orchestrator.cpp. The other two are accepted but currently
// treated the same as V1_DENSE (no hybrid/graph logic wired in yet).
enum class PipelineLevel { V1_DENSE, V1_5_HYBRID, V2_GRAPH_AWARE };
NLOHMANN_JSON_SERIALIZE_ENUM(
    PipelineLevel, {
                        {PipelineLevel::V1_DENSE, "v1_dense"},
                        {PipelineLevel::V1_5_HYBRID, "v1_5_hybrid"},
                        {PipelineLevel::V2_GRAPH_AWARE, "v2_graph_aware"},
                    })

// How much surrounding context to return per hit. Also accepted-but-unused
// for now — the MVP always returns whatever chunk granularity the stored
// vector represents, no window expansion.
enum class Granularity { RAW_CHUNK, PARENT_CHUNK, WINDOW_EXPANDED };
NLOHMANN_JSON_SERIALIZE_ENUM(
    Granularity, {
                     {Granularity::RAW_CHUNK, "raw_chunk"},
                     {Granularity::PARENT_CHUNK, "parent_chunk"},
                     {Granularity::WINDOW_EXPANDED, "window_expanded"},
                 })

// --- MemoryQueryConfig ---------------------------------------------------
// Options that only make sense when searching conversational memory
// (as opposed to the knowledge base). NONE of the filtering described
// here (time decay, importance threshold, session/user scoping) is
// actually applied yet — the MVP pipeline parses these fields and then
// ignores them. That's the next piece of work after this skeleton.
struct MemoryQueryConfig {
  std::optional<std::string> session_id;
  std::optional<std::string> user_id;
  bool enable_time_decay = true;   // weight older memories lower
  float recency_weight = 0.3f;     // how strongly recency affects score
  float importance_threshold = 0.5f; // drop memories scored below this
  std::optional<std::string> reference_timestamp; // ISO string, kept raw
                                                    // (no datetime parsing
                                                    // yet — "now" is
                                                    // implied when absent)
};

// Every from_json below follows the same defensive pattern:
//   - required fields use j.at("key") (throws if missing — fail loudly)
//   - optional fields check j.contains(...) && !is_null() first
//   - fields with a Python-side default use j.value("key", default)
// This mirrors pydantic's own defaulting behavior so a partial JSON
// payload (like the "config" examples in main_orch_files/) still parses.
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

// to_json's job (unlike from_json's) is just "produce something
// round-trippable" — this exists so RedisDbClient can store the request
// that led to a memory being written (see MemoryMetadata::Query in
// redis_db_client.hpp), not to match any wire format Python expects back.
// std::optional isn't handled automatically by the vendored json.hpp, so
// every optional field below is written out by hand as either its value
// or null, same defensive style as from_json above.
inline void to_json(json &j, const MemoryQueryConfig &c) {
  j = json{
      {"session_id", c.session_id ? json(*c.session_id) : json(nullptr)},
      {"user_id", c.user_id ? json(*c.user_id) : json(nullptr)},
      {"enable_time_decay", c.enable_time_decay},
      {"recency_weight", c.recency_weight},
      {"importance_threshold", c.importance_threshold},
      {"reference_timestamp", c.reference_timestamp
                                   ? json(*c.reference_timestamp)
                                   : json(nullptr)},
  };
}

// --- KnowledgeQueryConfig -------------------------------------------------
// Options for searching the knowledge base (docs/wiki) domain. Like
// MemoryQueryConfig, this parses cleanly but isn't acted on yet — there is
// no knowledge-base search path wired into the orchestrator yet, only
// memory-domain dense search.
struct KnowledgeQueryConfig {
  std::vector<std::string> collection_names{"default_kb"};
  // SPLADE/BM25 style {token: weight} map for sparse retrieval. Building
  // this (and the matching Redis-backed sparse index) is future work —
  // see the RAG 2.0 discussion earlier in the project notes.
  std::optional<std::map<std::string, float>> sparse_vector;
  float hybrid_alpha = 0.7f; // 1.0 = dense only, 0.0 = sparse only
  // Knowledge-graph traversal seeds/hop-count for v2_graph_aware. Also
  // future work — no graph store exists yet.
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

inline void to_json(json &j, const KnowledgeQueryConfig &c) {
  j = json{
      {"collection_names", c.collection_names},
      {"sparse_vector", c.sparse_vector ? json(*c.sparse_vector) : json(nullptr)},
      {"hybrid_alpha", c.hybrid_alpha},
      {"seed_entities", c.seed_entities ? json(*c.seed_entities) : json(nullptr)},
      {"max_hops", c.max_hops},
  };
}

// --- DualTrackFusionConfig -------------------------------------------------
// How to combine memory-domain hits and knowledge-domain hits when
// target_domain == BOTH. Only relevant once both domains are actually
// being searched (not yet — see TargetDomain comment above).
struct DualTrackFusionConfig {
  float memory_weight = 0.4f;
  float knowledge_weight = 0.6f;
  bool interleave_results = false; // [mem1, kb1, mem2, ...] vs score-sorted
};

inline void from_json(const json &j, DualTrackFusionConfig &c) {
  c.memory_weight = j.value("memory_weight", 0.4f);
  c.knowledge_weight = j.value("knowledge_weight", 0.6f);
  c.interleave_results = j.value("interleave_results", false);
}

inline void to_json(json &j, const DualTrackFusionConfig &c) {
  j = json{
      {"memory_weight", c.memory_weight},
      {"knowledge_weight", c.knowledge_weight},
      {"interleave_results", c.interleave_results},
  };
}

// --- RAGQueryOrder (top-level) ---------------------------------------------
// This is the exact struct that config.rag_policy in the incoming JSON
// request deserializes into. See schemas/fuli_schemas.hpp for where it
// plugs into the full request.
struct RAGQueryOrder {
  TargetDomain target_domain = TargetDomain::BOTH;
  PipelineLevel pipeline_level = PipelineLevel::V1_5_HYBRID;

  // If the caller already computed an embedding, it's passed here and we
  // skip calling EmbeddingClient::Embed() entirely (see
  // Orchestrator::HandleContextRequest). Normally this is null/absent —
  // FuliHandler's docstring explicitly says "Fuli embeds the input
  // itself", meaning WE are expected to call TEI, not the caller.
  std::optional<std::vector<float>> dense_vector;

  int top_k = 5;                    // how many hits to return
  float min_score_threshold = 0.6f; // not yet enforced by the pipeline

  // Both configs are optional because the Python side only fills in the
  // one(s) matching target_domain (e.g. knowledge_config stays null when
  // target_domain == MEMORY_ONLY).
  std::optional<MemoryQueryConfig> memory_config;
  std::optional<KnowledgeQueryConfig> knowledge_config;
  DualTrackFusionConfig fusion_config; // always present, has its own defaults

  Granularity granularity = Granularity::RAW_CHUNK;
  int context_window_size = 1;
  bool enable_rerank = false; // not yet wired to the TEI reranker
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

inline void to_json(json &j, const RAGQueryOrder &o) {
  j = json{
      {"target_domain", o.target_domain},
      {"pipeline_level", o.pipeline_level},
      {"dense_vector", o.dense_vector ? json(*o.dense_vector) : json(nullptr)},
      {"top_k", o.top_k},
      {"min_score_threshold", o.min_score_threshold},
      {"memory_config",
       o.memory_config ? json(*o.memory_config) : json(nullptr)},
      {"knowledge_config",
       o.knowledge_config ? json(*o.knowledge_config) : json(nullptr)},
      {"fusion_config", o.fusion_config},
      {"granularity", o.granularity},
      {"context_window_size", o.context_window_size},
      {"enable_rerank", o.enable_rerank},
  };
}

} // namespace schemas
