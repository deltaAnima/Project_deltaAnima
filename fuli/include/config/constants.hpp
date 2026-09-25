#pragma once

// MVP-stage hardcoded config. Every downstream service address and the
// embedding dimension live here as plain constants for now — there is no
// YAML/env-based service registry yet. When you add more deployment
// targets (dev/staging/prod, multiple GPU hosts, etc.) this is the file
// that should turn into a loaded config struct instead.
namespace config 
{

// bge-m3 (HF text-embeddings-inference) dense embedding output size.
// This MUST match whatever embedding model you actually point
// EmbeddingClient at — Faiss will silently misbehave (or assert) if you
// feed it vectors of the wrong dimension.
inline constexpr int kEmbeddingDim = 1024;

// Where the TEI (text-embeddings-inference) bge-m3 container listens.
// EmbeddingClient::Embed() connects here over plain HTTP (no TLS) since
// this is all internal, same-host/same-LAN traffic.
inline constexpr const char *kEmbeddingHost = "127.0.0.1";
inline constexpr const char *kEmbeddingPort = "8080";

// Where RedisDbClient connects for memory metadata. Same caveat as
// kEmbeddingHost: if orchestrator_server doesn't run on the same host as
// this Redis instance, change this to that host's real address.
inline constexpr const char *kRedisHost = "127.0.0.1";
inline constexpr const char *kRedisPort = "8080";

// The port THIS server (orchestrator_server) listens on for the
// /character/context endpoint that FuliHandler (the Python side) calls.
inline constexpr unsigned short kListenPort = 8080;

// These are relative filesystem paths, resolved against whatever
// directory you launch the binary from — NOT relative to this header or
// to CMakeLists.txt. Always run `./build/orchestrator_server` from the
// Fuli_main/ project root so these resolve correctly.
inline constexpr const char *kDeltaEgoConfigPath =
    "config/deltaEGO_default.yaml";
inline constexpr const char *kVadDbPath = "include/VAD_DB/VAD.json";

} // namespace config
