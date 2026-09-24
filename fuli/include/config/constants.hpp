#pragma once

// MVP-stage hardcoded config. Move to a YAML/env-based service registry
// once there is more than one deployment target.
namespace config {

// bge-m3 (HF text-embeddings-inference) dense output dimension.
inline constexpr int kEmbeddingDim = 1024;

inline constexpr const char *kEmbeddingHost = "127.0.0.1";
inline constexpr const char *kEmbeddingPort = "8080";

inline constexpr unsigned short kListenPort = 8080;

// Paths are relative to the process working directory — run
// orchestrator_server from Fuli_main/ (see README/build notes).
inline constexpr const char *kDeltaEgoConfigPath =
    "config/deltaEGO_default.yaml";
inline constexpr const char *kVadDbPath = "include/VAD_DB/VAD.json";

} // namespace config
