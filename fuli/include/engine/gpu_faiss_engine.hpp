#pragma once

#include <cstdint>
#include <future>
#include <memory>
#include <string>
#include <vector>

#include "engine/vector_search_engine.hpp"

// Real Faiss GPU flat-index implementation of IVectorSearchEngine (see
// vector_search_engine.hpp for the interface, and mock_faiss_engine.cpp
// for the fake version used when ENABLE_GPU is off in CMake).
//
// "Flat index" means brute-force: every search compares the query against
// every stored vector (no approximate-nearest-neighbor structure like
// HNSW/IVF). That's fine — even great — for a few thousand vectors on a
// GPU, and it means zero index-tuning to get a working MVP. Once the
// memory/knowledge store grows large enough that brute force is too slow,
// swap GpuIndexFlatL2 for something like GpuIndexIVFFlat — the
// IVectorSearchEngine interface doesn't need to change for that.
//
// Concurrency model: all GPU work (add + search) is funneled through ONE
// dedicated background thread instead of being called directly from
// whichever thread happens to invoke AddVectors/AsyncSearch. This matters
// because faiss::gpu::StandardGpuResources (which owns a CUDA stream and
// scratch memory) and the GpuIndexFlatL2 built on top of it are NOT
// documented as safe to call concurrently from multiple host threads.
// Rather than trying to prove that's fine or add fine-grained locking
// around every Faiss call, we sidestep the whole problem: exactly one
// thread ever touches `index_`, and everyone else just drops a task on a
// queue and gets a future/promise back.
class GpuFaissEngine : public IVectorSearchEngine {
public:
  // dim must equal whatever embedding model you're using (see
  // config::kEmbeddingDim) — Faiss doesn't check this for you.
  explicit GpuFaissEngine(int dim);
  ~GpuFaissEngine() override;

  // Not copyable: this owns a GPU index and a live worker thread, neither
  // of which has sane copy semantics.
  GpuFaissEngine(const GpuFaissEngine &) = delete;
  GpuFaissEngine &operator=(const GpuFaissEngine &) = delete;

  // Adds ids.size() vectors to the index. flat_vectors is a row-major
  // flattened array: flat_vectors[i*dim : i*dim+dim] is the vector for
  // ids[i]. Faiss's GpuIndexFlat has no built-in id storage (see the
  // comment on id_map_ in the .cpp) — we track ids ourselves, positionally,
  // in insertion order.
  //
  // This call BLOCKS the calling thread until the add completes (it's
  // implemented as AsyncAddVectors(...).get()). That's fine for
  // startup-time bulk seeding (see main.cpp's SeedTestVectors) but NOT
  // for the request path — use AsyncAddVectors there instead, the same
  // way AsyncSearch is used instead of a blocking search.
  void AddVectors(const std::vector<int64_t> &ids,
                   const std::vector<float> &flat_vectors);

  // Non-blocking version — see IVectorSearchEngine::AsyncAddVectors.
  std::future<void> AsyncAddVectors(const std::vector<int64_t> &ids,
                                     const std::vector<float> &flat_vectors) override;

  // Persists the index to disk: writes "{path}.faiss" (a CPU copy of the
  // index, via faiss::write_index — Faiss's GPU indices don't serialize
  // directly) and "{path}.ids" (our external id_map_, which Faiss's own
  // serialization has no concept of — see the id_map_ comment in the
  // .cpp). Blocking — implemented as AsyncSaveToDisk(path).get(). Fine
  // for startup/shutdown; use AsyncSaveToDisk instead for a periodic
  // autosave loop that runs alongside live request handling. Throws on
  // I/O or Faiss failure.
  void SaveToDisk(const std::string &path);

  // Non-blocking version of SaveToDisk — see IVectorSearchEngine::
  // AsyncAddVectors for why this shape (future instead of blocking)
  // matters once something calls it from inside the io_context.
  std::future<void> AsyncSaveToDisk(const std::string &path);

  // Loads an index previously written by SaveToDisk, REPLACING whatever
  // is currently in this engine — any vectors added before this call are
  // discarded. Returns false (engine left untouched) if either file is
  // missing/corrupt or the saved index's dimension doesn't match dim_
  // this engine was constructed with. Also blocking.
  bool LoadFromDisk(const std::string &path);

  // Number of vectors currently stored.
  size_t Size() const;

  // Non-blocking: posts a search task to the worker thread and returns
  // immediately with a std::future you can poll/wait on later. See
  // pipeline/orchestrator.cpp's AwaitSearch() helper for how this gets
  // bridged into an asio coroutine without blocking the whole server.
  std::future<SearchResult> AsyncSearch(const std::vector<float> &query_vector,
                                         int top_k) override;

private:
  // Pimpl (pointer to implementation): hides all the Faiss/CUDA/threading
  // detail from anyone who just #includes this header, so consumers don't
  // need faiss/CUDA headers or -march flags to compile against
  // GpuFaissEngine — same trick deltaEGO.hpp uses for its own internals.
  class Impl;
  std::unique_ptr<Impl> impl_;
};
