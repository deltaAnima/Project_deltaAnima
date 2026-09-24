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

class GpuFaissEngine::Impl {
public:
  explicit Impl(int dim)
      : dim_(dim), resources_(),
        index_(&resources_, dim, faiss::gpu::GpuIndexFlatConfig()) {
    worker_ = std::thread([this] { WorkerLoop(); });
  }

  ~Impl() {
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

    std::promise<void> done;
    auto fut = done.get_future();
    Enqueue([this, ids, flat, &done]() mutable {
      id_map_.insert(id_map_.end(), ids.begin(), ids.end());
      index_.add(static_cast<faiss::idx_t>(ids.size()), flat.data());
      done.set_value();
    });
    fut.wait();
  }

  size_t Size() const { return id_map_.size(); }

  std::future<SearchResult> AsyncSearch(std::vector<float> query,
                                         int top_k) {
    auto promise = std::make_shared<std::promise<SearchResult>>();
    auto fut = promise->get_future();

    if (static_cast<int>(query.size()) != dim_) {
      promise->set_exception(std::make_exception_ptr(std::invalid_argument(
          "AsyncSearch: query_vector size != index dim")));
      return fut;
    }

    Enqueue([this, query = std::move(query), top_k, promise]() mutable {
      try {
        SearchResult result;
        if (index_.getNumVecs() == 0) {
          promise->set_value(std::move(result));
          return;
        }
        int k = std::min<int>(top_k, static_cast<int>(index_.getNumVecs()));
        std::vector<float> distances(k);
        std::vector<faiss::idx_t> labels(k);
        index_.search(1, query.data(), k, distances.data(), labels.data());

        for (int i = 0; i < k; ++i) {
          if (labels[i] < 0)
            continue; // faiss pads with -1 when it has fewer than k hits
          result.ids.push_back(id_map_.at(static_cast<size_t>(labels[i])));
          result.distances.push_back(distances[i]);
        }
        promise->set_value(std::move(result));
      } catch (...) {
        promise->set_exception(std::current_exception());
      }
    });
    return fut;
  }

private:
  void Enqueue(std::function<void()> task) {
    {
      std::lock_guard<std::mutex> lock(mu_);
      queue_.push_back(std::move(task));
    }
    cv_.notify_one();
  }

  void WorkerLoop() {
    for (;;) {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [this] { return stop_ || !queue_.empty(); });
        if (stop_ && queue_.empty())
          return;
        task = std::move(queue_.front());
        queue_.pop_front();
      }
      task();
    }
  }

  int dim_;
  faiss::gpu::StandardGpuResources resources_;
  faiss::gpu::GpuIndexFlatL2 index_;
  std::vector<int64_t> id_map_; // faiss local index -> external memory id

  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<std::function<void()>> queue_;
  bool stop_ = false;
  std::thread worker_;
};

GpuFaissEngine::GpuFaissEngine(int dim) : impl_(std::make_unique<Impl>(dim)) {}
GpuFaissEngine::~GpuFaissEngine() = default;

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
