#include "pipeline/memory_retriever.hpp"

#include "util/future_bridge.hpp"

namespace pipeline {

namespace net = boost::asio;

namespace {

// Faiss has no idea what session_id/importance_threshold are, so a
// plain top_k search can come back with candidates that all get
// filtered out, leaving fewer than top_k survivors even though better
// (still-unseen) matches exist further down the true ranking.
// Over-fetching gives filtering something to work with. This is a flat
// multiplier, not an adaptive one — fine for the few-hundred-vector MVP
// index; revisit once over-fetching this much is actually expensive.
constexpr int kOversampleFactor = 5;

} // namespace

MemoryRetriever::MemoryRetriever(IVectorSearchEngine &search_engine,
                                  clients::RedisDbClient &redis)
    : search_engine_(search_engine), redis_(redis) {}

net::awaitable<std::vector<RetrievedMemory>> MemoryRetriever::Retrieve(
    const std::vector<float> &query_vector, int top_k,
    const std::optional<schemas::MemoryQueryConfig> &config) {
  int fetch_k = top_k * kOversampleFactor;
  SearchResult raw = co_await util::AwaitFuture(
      search_engine_.AsyncSearch(query_vector, fetch_k));

  std::vector<RetrievedMemory> results;
  results.reserve(static_cast<size_t>(top_k));

  // raw.ids/raw.distances are already sorted nearest-first (that's what
  // Faiss's search() guarantees for L2), so the first top_k candidates
  // that SURVIVE filtering are, by construction, the best top_k overall
  // — nothing further down the list could have a smaller distance.
  for (size_t i = 0; i < raw.ids.size(); ++i) {
    // One Redis round trip per candidate, awaited sequentially. Fine
    // for the handful of candidates an MVP oversample produces —
    // pipelining these (fire every HGETALL, then await them together)
    // is the obvious next step once this is measurably slow.
    std::optional<clients::MemoryMetadata> meta =
        co_await redis_.GetMemoryMetadata(raw.ids[i]);
    if (!meta)
      continue; // nothing stored for this id (e.g. a seeded test vector
                // that was never given real metadata)

    if (config) {
      // Only reject on an explicit mismatch — a memory with no
      // session_id recorded is treated as unscoped rather than
      // excluded. That's a judgment call, not a spec: tighten this to
      // "no session_id means excluded" if strict per-session isolation
      // turns out to matter more than recall.
      if (config->session_id && meta->metadata.session_id &&
          *config->session_id != *meta->metadata.session_id)
        continue;

      if (meta->metadata.importance < config->importance_threshold)
        continue;
    }

    results.push_back({raw.ids[i], raw.distances[i], std::move(*meta)});
    if (static_cast<int>(results.size()) >= top_k)
      break;
  }

  co_return results;
}

net::awaitable<int64_t> MemoryRetriever::Store(const std::vector<float> &vector,
                                                clients::MemoryMetadata metadata) {
  int64_t id = co_await redis_.NextId();

  // AsyncAddVectors (not the blocking AddVectors) — Store() runs on the
  // request path, potentially concurrently with other requests, so it
  // must not block the io_context thread the way AddVectors's blocking
  // .get() would.
  co_await util::AwaitFuture(
      search_engine_.AsyncAddVectors({id}, vector));

  metadata.metadata.faiss_id = id;
  co_await redis_.SetMemoryMetadata(id, metadata);

  co_return id;
}

} // namespace pipeline
