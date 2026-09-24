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
};
