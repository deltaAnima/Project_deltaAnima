#pragma once
#include <vector>
#include <cstdint>
#include <future>

struct SearchResult {
    std::vector<int64_t> ids;
    std::vector<float> distances;
};

class IVectorSearchEngine {
public:
    virtual ~IVectorSearchEngine() = default;
    virtual std::future<SearchResult> AsyncSearch(const std::vector<float>& query_vector, int top_k) = 0;

    // Non-blocking add: returns immediately with a future instead of
    // blocking the calling thread. Bridge it with util::AwaitFuture from
    // a coroutine (see pipeline::MemoryRetriever::Store) rather than
    // calling .get()/.wait() on it directly — same reason AsyncSearch's
    // future needs bridging instead of blocking on directly.
    virtual std::future<void> AsyncAddVectors(const std::vector<int64_t>& ids,
                                               const std::vector<float>& flat_vectors) = 0;
};
