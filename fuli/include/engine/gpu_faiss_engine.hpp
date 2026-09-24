#pragma once

#include <cstdint>
#include <future>
#include <memory>
#include <vector>

#include "engine/vector_search_engine.hpp"

// Real Faiss GPU flat-index backed implementation of IVectorSearchEngine.
// All GPU calls (add/search) run on one dedicated worker thread — Faiss's
// StandardGpuResources/GpuIndex are not safe to hit concurrently from
// arbitrary caller threads, so every request is funneled through a single
// queue instead of trying to synchronize the index directly.
class GpuFaissEngine : public IVectorSearchEngine {
public:
  explicit GpuFaissEngine(int dim);
  ~GpuFaissEngine() override;

  GpuFaissEngine(const GpuFaissEngine &) = delete;
  GpuFaissEngine &operator=(const GpuFaissEngine &) = delete;

  // ids.size() * dim must equal flat_vectors.size(). Blocks until the
  // vectors are actually added — only meant for startup seeding, not the
  // request path.
  void AddVectors(const std::vector<int64_t> &ids,
                   const std::vector<float> &flat_vectors);

  size_t Size() const;

  std::future<SearchResult> AsyncSearch(const std::vector<float> &query_vector,
                                         int top_k) override;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};
