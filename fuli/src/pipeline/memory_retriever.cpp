#include <array>
#include <cstdint>
#include <iostream>

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

#include "config/constants.hpp"
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

// Scopes whose memories must never surface anywhere but the scope
// itself, regardless of MemoryScope policy — a DM recalled in a public
// channel is a leak, not a feature.
constexpr std::array<std::string_view, 1> kPrivateScopePrefixes = {"discord:dm:"};

bool IsPrivateScope(const std::string &scope_key)
{
  for (std::string_view prefix : kPrivateScopePrefixes)
    if (scope_key.starts_with(prefix))
      return true;
  return false;
}

} // namespace

inline uint64_t MemoryRetriever::fmix64(uint64_t k)
{
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;
    return k;
}
inline UserId MemoryRetriever::MakeUserId(std::string_view key, uint64_t seed)
{
  const uint8_t* data = reinterpret_cast<const uint8_t*>(key.data());
  const size_t nblocks = key.size() / 16;

  uint64_t h1 = seed;
  uint64_t h2 = seed;

  const uint64_t c1 = 0x87c37b91114253d5ULL;
  const uint64_t c2 = 0x4cf5ad432745937fULL;

  const uint64_t* blocks = reinterpret_cast<const uint64_t*>(data);
  for (size_t i = 0; i < nblocks; ++i) 
  {
    uint64_t k1 = blocks[i * 2 + 0];
    uint64_t k2 = blocks[i * 2 + 1];

    k1 *= c1; k1 = (k1 << 31) | (k1 >> 33); k1 *= c2; h1 ^= k1;
    h1 = (h1 << 27) | (h1 >> 37); h1 += h2; h1 = h1 * 5 + 0x52dce729;

    k2 *= c2; k2 = (k2 << 33) | (k2 >> 31); k2 *= c1; h2 ^= k2;
    h2 = (h2 << 31) | (h2 >> 33); h2 += h1; h2 = h2 * 5 + 0x38495ab5;
  }

  const uint8_t* tail = data + nblocks * 16;
  uint64_t k1 = 0;
  uint64_t k2 = 0;

  switch (key.size() & 15) 
  {
    case 15: k2 ^= uint64_t(tail[14]) << 48; [[fallthrough]];
    case 14: k2 ^= uint64_t(tail[13]) << 40; [[fallthrough]];
    case 13: k2 ^= uint64_t(tail[12]) << 32; [[fallthrough]];
    case 12: k2 ^= uint64_t(tail[11]) << 24; [[fallthrough]];
    case 11: k2 ^= uint64_t(tail[10]) << 16; [[fallthrough]];
    case 10: k2 ^= uint64_t(tail[ 9]) << 8;  [[fallthrough]];
    case  9: k2 ^= uint64_t(tail[ 8]) << 0;
              k2 *= c2; k2 = (k2 << 33) | (k2 >> 31); k2 *= c1; h2 ^= k2;
              [[fallthrough]];
    case  8: k1 ^= uint64_t(tail[ 7]) << 56; [[fallthrough]];
    case  7: k1 ^= uint64_t(tail[ 6]) << 48; [[fallthrough]];
    case  6: k1 ^= uint64_t(tail[ 5]) << 40; [[fallthrough]];
    case  5: k1 ^= uint64_t(tail[ 4]) << 32; [[fallthrough]];
    case  4: k1 ^= uint64_t(tail[ 3]) << 24; [[fallthrough]];
    case  3: k1 ^= uint64_t(tail[ 2]) << 16; [[fallthrough]];
    case  2: k1 ^= uint64_t(tail[ 1]) << 8;  [[fallthrough]];
    case  1: k1 ^= uint64_t(tail[ 0]) << 0;
            k1 *= c1; k1 = (k1 << 31) | (k1 >> 33); k1 *= c2; h1 ^= k1;
  }

  h1 ^= key.size(); h2 ^= key.size();
  h1 += h2; h2 += h1;
  h1 = fmix64(h1); h2 = fmix64(h2);
  h1 += h2; h2 += h1;

  return UserId{ .high = h1, .low = h2 };
}

UserId MemoryRetriever::MakeConversationId(const std::string& user_key,
                                           const std::optional<std::string>& scope_key)
{
  // \x1f (unit separator) can't appear in either key in practice, so
  // ("a", "b:c") and ("a:b", "c") can't collide into the same id.
  return this->MakeUserId(user_key + '\x1f' + scope_key.value_or(""));
}


MemoryRetriever::MemoryRetriever(IVectorSearchEngine &search_engine,
                                  clients::RedisDbClient &redis)
    : search_engine_(search_engine), redis_(redis) {}

net::awaitable<std::vector<RetrievedMemory>> MemoryRetriever::Retrieve(
    const std::vector<float> &query_vector, int top_k,
    const std::optional<schemas::MemoryQueryConfig> &config) 
{
  int fetch_k = top_k * kOversampleFactor;
  SearchResult raw = co_await util::AwaitFuture(
      this->search_engine_.AsyncSearch(query_vector, fetch_k));

  std::vector<RetrievedMemory> results;
  results.reserve(static_cast<size_t>(top_k));

  // Which filters apply, decided once up front. A policy whose key is
  // missing degrades to the other filter rather than to no filter at
  // all — `scope` without a scope_key falls back to the user filter
  // instead of returning every user's memories.
  bool filter_user = false;
  bool filter_scope = false;
  bool filter_session = false;
  UserId wanted_user{};
  if (config)
  {
    filter_scope = config->scope_key.has_value() &&
                   config->scope != schemas::MemoryScope::USER;
    filter_user = config->user_id.has_value() &&
                  (config->scope != schemas::MemoryScope::SCOPE || !filter_scope);
    // Sessions are per (user_key, scope_key), so a session filter only
    // makes sense for the policy that matches that exact pair — under
    // `scope`/`user` it would throw away every other user's / scope's
    // memories, which is exactly what those policies are for.
    filter_session = config->session_id.has_value() &&
                     config->scope == schemas::MemoryScope::USER_IN_SCOPE;
    if (filter_user)
      wanted_user = this->MakeUserId(*config->user_id);
  }

  // raw.ids/raw.distances are already sorted nearest-first (that's what
  // Faiss's search() guarantees for L2), so the first top_k candidates
  // that SURVIVE filtering are, by construction, the best top_k overall
  // — nothing further down the list could have a smaller distance.
  for (size_t i = 0; i < raw.ids.size(); ++i) 
  {
    // One Redis round trip per candidate, awaited sequentially. Fine
    // for the handful of candidates an MVP oversample produces —
    // pipelining these (fire every HGETALL, then await them together)
    // is the obvious next step once this is measurably slow.
    std::optional<clients::MemoryMetadata> meta =
        co_await this->redis_.GetMemoryMetadata(raw.ids[i]);
    if (!meta)
      continue; // nothing stored for this id (e.g. a seeded test vector
                // that was never given real metadata)

    if (config) 
    {
      // Only reject on an explicit mismatch — a memory with no
      // session_id recorded is treated as unscoped rather than
      // excluded. That's a judgment call, not a spec: tighten this to
      // "no session_id means excluded" if strict per-session isolation
      // turns out to matter more than recall.
      if (filter_session && meta->metadata.session_id &&
          *config->session_id != *meta->metadata.session_id)
        continue;

      // Unlike session_id, a missing user_id/scope_key is NOT treated as
      // unscoped — an unattributed memory showing up in someone else's
      // search is a cross-user leak, so strict wins over recall here.
      if (filter_user &&
          (meta->memory.user.user_id.high != wanted_user.high ||
           meta->memory.user.user_id.low != wanted_user.low))
        continue;

      if (filter_scope && meta->metadata.scope_key != config->scope_key)
        continue;

      if (meta->metadata.importance < config->importance_threshold)
        continue;
    }

    // Outside `if (config)` on purpose: no config means no scope to be
    // "in", so a private memory is excluded rather than let through.
    if (meta->metadata.scope_key && IsPrivateScope(*meta->metadata.scope_key) &&
        (!config || meta->metadata.scope_key != config->scope_key))
      continue;

    results.push_back({raw.ids[i], raw.distances[i], std::move(*meta)});
    if (static_cast<int>(results.size()) >= top_k)
      break;
  }

  co_return results;
}

net::awaitable<int64_t> MemoryRetriever::Store(const std::vector<float> &vector,
                                                clients::MemoryMetadata* metadata) 
{
  int64_t id = co_await this->redis_.NextId();

  // AsyncAddVectors (not the blocking AddVectors) — Store() runs on the
  // request path, potentially concurrently with other requests, so it
  // must not block the io_context thread the way AddVectors's blocking
  // .get() would.
  co_await util::AwaitFuture(
      this->search_engine_.AsyncAddVectors({id}, vector));

  metadata->metadata.faiss_id = id;
  co_await this->redis_.SetMemoryMetadata(id, metadata);

  co_return id;
}

std::pair<bool, std::shared_ptr<clients::MemoryMetadata>>
MemoryRetriever::get_memory_buff_(const std::string& user_key,
                                  const std::optional<std::string>& scope_key,
                                  bool is_retrieve)
{
  UserId conversation_id = this->MakeConversationId(user_key, scope_key);
  auto it = this->mem_buffer_.find(conversation_id);

  if (it != this->mem_buffer_.end())
  {
    if (!is_retrieve)
      return {true, it->second.meta};

    auto age = std::chrono::steady_clock::now() - it->second.created;
    if (age < std::chrono::seconds(config::kStaleBufferSeconds))
    {
      std::cerr << "[WARN] Buffer already exists during retrieve step for: " << user_key
                << " (scope: " << scope_key.value_or("-") << ")\n";
      return {false, nullptr};
    }

    // The caller never saved or cancelled this turn (crashed, lost the
    // response, ...) — drop it rather than lock this conversation out
    // until a restart. Any coroutine still holding it keeps its own
    // shared_ptr copy.
    std::cerr << "[WARN] Taking over stale buffer ("
              << std::chrono::duration_cast<std::chrono::seconds>(age).count()
              << "s old) for: " << user_key << " (scope: " << scope_key.value_or("-") << ")\n";
    this->mem_buffer_.erase(it);
  }
  else if (!is_retrieve)
  {
    return {false, nullptr};
  }

  auto new_mem = std::make_shared<clients::MemoryMetadata>();
    std::cout << "[INFO] New context session created for user: " << user_key << "\n";

  // user_id is the user alone (not the conversation id) — it's what
  // Retrieve's user filter matches against, across every scope.
  UserId user_id = this->MakeUserId(user_key);
  new_mem->memory.user.user_key = user_key;
  new_mem->memory.user.user_id.high = user_id.high;
  new_mem->memory.user.user_id.low  = user_id.low;
  new_mem->metadata.scope_key = scope_key;

  this->mem_buffer_.emplace(conversation_id,
                            ConversationBuffer{new_mem, std::chrono::steady_clock::now()});

  return {true, std::move(new_mem)};
}

bool MemoryRetriever::delete_memory_buff_(const std::string& user_key,
                                          const std::optional<std::string>& scope_key,
                                          const clients::MemoryMetadata* expected)
{
  auto it = this->mem_buffer_.find(this->MakeConversationId(user_key, scope_key));
  if (it == this->mem_buffer_.end())
    return false;
  if (expected && it->second.meta.get() != expected)
    return false;
  this->mem_buffer_.erase(it);
  return true;
}

std::string MemoryRetriever::GetOrCreateSessionId(const std::string& user_key,
                                                  const std::optional<std::string>& scope_key,
                                                  bool force_new)
{
  UserId id = this->MakeConversationId(user_key, scope_key);

  if (!force_new)
  {
    auto it = this->session_ids_.find(id);
    if (it != this->session_ids_.end())
      return it->second;
  }

  boost::uuids::random_generator gen;
  std::string session_id = boost::uuids::to_string(gen());

  this->session_ids_[id] = session_id; // insert or overwrite an existing (force_new) entry
  std::cout << "[INFO] New session " << session_id << " created for user: " << user_key
            << " (scope: " << scope_key.value_or("-") << ")\n";

  return session_id;
}
} // namespace pipeline
