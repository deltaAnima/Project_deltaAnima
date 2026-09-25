#include "engine/vector_search_engine.hpp"

class MockFaissEngine : public IVectorSearchEngine {
public:
    std::future<SearchResult> AsyncSearch(const std::vector<float>& query_vector, int top_k) override {
        std::promise<SearchResult> p;
        SearchResult res;
        for (int i = 0; i < top_k; ++i) {
            res.ids.push_back(i);
            res.distances.push_back(1.0f - (i * 0.1f));
        }
        p.set_value(std::move(res));
        return p.get_future();
    }

    // No real index behind this mock, so there's nothing to add to —
    // just acknowledge immediately. Only used when built with
    // ENABLE_GPU=OFF (see CMakeLists.txt), so this never runs alongside
    // the real GpuFaissEngine.
    std::future<void> AsyncAddVectors(const std::vector<int64_t>&,
                                       const std::vector<float>&) override {
        std::promise<void> p;
        p.set_value();
        return p.get_future();
    }
};
