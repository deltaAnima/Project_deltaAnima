#include "engine/gpu_faiss_engine.hpp"

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>

#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/StandardGpuResources.h>

// See the big comment in gpu_faiss_engine.hpp for WHY everything here goes
// through a single worker thread instead of being called directly.
class GpuFaissEngine::Impl {
public:
  explicit Impl(int dim)
      : dim_(dim), resources_(),
        // GpuIndexFlatL2's "construct an empty instance" overload: no
        // pre-existing CPU index to copy from, just (resources, dims,
        // config). L2 = squared Euclidean distance; combined with
        // normalized embeddings (see EmbeddingClient::Embed), this ranks
        // results the same way cosine similarity would.
        index_(&resources_, dim, faiss::gpu::GpuIndexFlatConfig()) {
    // Start the single GPU worker thread. It runs WorkerLoop() for the
    // entire lifetime of this object, pulling tasks off `queue_` one at a
    // time until told to stop in the destructor.
    worker_ = std::thread([this] { WorkerLoop(); });
  }

  ~Impl() {
    // Tell the worker to exit once it's drained the queue, then wait for
    // it to actually finish (join) before this object's members — in
    // particular resources_/index_ — get destroyed. Destroying the Faiss
    // GPU objects while the worker thread might still be mid-call into
    // them would be a use-after-free.
    {
      std::lock_guard<std::mutex> lock(mu_);
      stop_ = true;
    }
    cv_.notify_all();
    worker_.join();
  }

  void AddVectors(const std::vector<int64_t> &ids,
                   const std::vector<float> &flat) {
    if (ids.empty())
      return;
    if (flat.size() != ids.size() * static_cast<size_t>(dim_)) {
      throw std::invalid_argument(
          "AddVectors: flat_vectors.size() != ids.size() * dim");
    }

    // AddVectors is documented as blocking (see header), so we enqueue a
    // task that fulfills this local promise, then immediately wait() on
    // its future right here. This is just "run this on the worker thread
    // and don't come back until it's done" expressed with the same
    // promise/future machinery AsyncSearch uses for the non-blocking case.
    std::promise<void> done;
    auto fut = done.get_future();
    Enqueue([this, ids, flat, &done]() mutable {
      // Faiss's GpuIndexFlat does not store ids itself ("Flat index does
      // not require IDs as there is no storage available for them" — see
      // the faiss header). So WE maintain the id mapping: id_map_[i] is
      // the external memory-id for whatever vector Faiss internally calls
      // index i. Since add() always appends, and search() returns
      // internal indices in the same 0..N-1 space, id_map_ position i
      // and Faiss's internal vector i always refer to the same vector as
      // long as we only ever append (never delete/reorder).
      id_map_.insert(id_map_.end(), ids.begin(), ids.end());
      index_.add(static_cast<faiss::idx_t>(ids.size()), flat.data());
      done.set_value();
    });
    fut.wait(); // blocks THIS (calling) thread — see header docs on why
                // that's fine for seeding but not for the request path.
  }

  size_t Size() const { return id_map_.size(); }

  std::future<SearchResult> AsyncSearch(std::vector<float> query,
                                         int top_k) {
    // shared_ptr (not unique_ptr) because the lambda below is copied into
    // std::function, which requires its captures to be copyable.
    auto promise = std::make_shared<std::promise<SearchResult>>();
    auto fut = promise->get_future();

    if (static_cast<int>(query.size()) != dim_) {
      // Fail fast, synchronously, without even touching the worker
      // thread/GPU — wrong-sized input is a caller bug, not a runtime
      // condition worth a round trip.
      promise->set_exception(std::make_exception_ptr(std::invalid_argument(
          "AsyncSearch: query_vector size != index dim")));
      return fut;
    }

    Enqueue([this, query = std::move(query), top_k, promise]() mutable {
      try {
        SearchResult result;
        if (index_.getNumVecs() == 0) {
          // Nothing indexed yet — return an empty (not an error) result.
          promise->set_value(std::move(result));
          return;
        }
        // Faiss will assert/misbehave if asked for more neighbors than
        // exist, so clamp k to however many vectors are actually stored.
        int k = std::min<int>(top_k, static_cast<int>(index_.getNumVecs()));
        std::vector<float> distances(k);
        std::vector<faiss::idx_t> labels(k);
        // search(n_queries, query_data, k, out_distances, out_labels).
        // We always search with n_queries=1 (one query vector at a time)
        // — batching multiple queries into one search() call is possible
        // and more GPU-efficient, but not something the current
        // request-per-search pipeline needs yet.
        index_.search(1, query.data(), k, distances.data(), labels.data());

        for (int i = 0; i < k; ++i) {
          if (labels[i] < 0)
            continue; // Faiss pads with -1 if it found fewer than k hits
                       // (shouldn't happen given the clamp above, but
                       // cheap to guard against).
          result.ids.push_back(id_map_.at(static_cast<size_t>(labels[i])));
          result.distances.push_back(distances[i]);
        }
        promise->set_value(std::move(result));
      } catch (...) {
        // Any exception inside the worker thread must NOT escape
        // WorkerLoop (an uncaught exception on a detached-from-main
        // thread would call std::terminate and kill the whole process).
        // Instead we hand it to the promise, so it re-throws on the
        // CALLER's side when they call fut.get().
        promise->set_exception(std::current_exception());
      }
    });
    return fut; // returns immediately — the actual search runs
                // asynchronously on the worker thread.
  }

private:
  // Pushes a unit of work onto the queue and wakes the worker thread if
  // it's currently sleeping in cv_.wait().
  void Enqueue(std::function<void()> task) {
    {
      std::lock_guard<std::mutex> lock(mu_);
      queue_.push_back(std::move(task));
    }
    cv_.notify_one();
  }

  // Runs for the entire lifetime of this Impl on its own std::thread.
  // Classic producer/consumer loop: sleep until there's work or a stop
  // request, then drain and run tasks one at a time (never concurrently —
  // that's the whole point of this class).
  void WorkerLoop() {
    for (;;) {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [this] { return stop_ || !queue_.empty(); });
        if (stop_ && queue_.empty())
          return; // drained everything AND told to stop — exit the thread
        task = std::move(queue_.front());
        queue_.pop_front();
      }
      // Run the task OUTSIDE the lock, so producers can keep enqueueing
      // more work (AddVectors/AsyncSearch calls from other threads)
      // while this one executes on the GPU.
      task();
    }
  }

  int dim_;
  faiss::gpu::StandardGpuResources resources_; // owns the CUDA stream +
                                                // scratch memory for this
                                                // index
  faiss::gpu::GpuIndexFlatL2 index_;
  std::vector<int64_t> id_map_; // Faiss internal index position -> our
                                 // external memory id (see AddVectors)

  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<std::function<void()>> queue_; // pending add/search tasks,
                                             // FIFO
  bool stop_ = false;
  std::thread worker_;
};

// --- Public class: just forwards everything to Impl ------------------------
GpuFaissEngine::GpuFaissEngine(int dim) : impl_(std::make_unique<Impl>(dim)) {}
GpuFaissEngine::~GpuFaissEngine() = default; // defined here (not inline in
                                              // the header) because Impl is
                                              // an incomplete type there —
                                              // the destructor needs Impl's
                                              // full definition to know how
                                              // to delete it.

void GpuFaissEngine::AddVectors(const std::vector<int64_t> &ids,
                                 const std::vector<float> &flat_vectors) {
  impl_->AddVectors(ids, flat_vectors);
}

size_t GpuFaissEngine::Size() const { return impl_->Size(); }

std::future<SearchResult>
GpuFaissEngine::AsyncSearch(const std::vector<float> &query_vector,
                             int top_k) {
  return impl_->AsyncSearch(query_vector, top_k);
}
