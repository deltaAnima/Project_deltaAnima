/**
 * deltaEGO_ version 2.
 * Upgraded emotion search using CPU cache
 * Optimized only for Ryzen 9 9950x
 */

#ifndef deltaEGO_HPP
#define deltaEGO_HPP
// ==========================================
// 0. Common headers
// ==========================================
#include <cmath>
#include <cstring>     // for memset
#include <immintrin.h> // AVX-512
#include <iostream>
#include <vector>

// JSON Library
#include <nlohmann/json.hpp> // to read VAD.json
using json = nlohmann::json;

// ==========================================
// 1. Platform Detection & Headers
// ==========================================
#ifdef _WIN32
#include <malloc.h>
#include <windows.h>
inline void PinThreadToCore(int coreID) {
  HANDLE threadHandle = GetCurrentThread();
  DWORD_PTR mask = (static_cast<DWORD_PTR>(1) << coreID);
  if (SetThreadAffinityMask(threadHandle, mask) == 0) {
    std::cerr << "[Warning] Core pinning failed for ID " << coreID << std::endl;
  } else {
    std::cout << "[System] Thread pinned to Logical Core " << coreID
              << " (CCD 1 Isolated)" << std::endl;
  }
}
inline void *NPCIE_AlignedMalloc(size_t size, size_t alignment) {
  return _aligned_malloc(size, alignment);
}
inline void NPCIE_AlignedFree(void *ptr) { _aligned_free(ptr); }
#else
#include <pthread.h>
#include <sched.h>
#include <stdlib.h>
#include <unistd.h>
inline void PinThreadToCore(int coreID) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(coreID, &cpuset);
  pthread_t current_thread = pthread_self();
  if (pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset) != 0) {
    std::cerr << "[Warning] Core pinning failed for ID " << coreID << std::endl;
  } else {
    std::cout << "[System] Thread pinned to Logical Core " << coreID
              << " (Linux/WSL)" << std::endl;
  }
}
inline void *NPCIE_AlignedMalloc(size_t size, size_t alignment) {
  void *ptr = nullptr;
  // instead of C++17 aligned_alloc, using posix_memalign for better
  // compatability
  if (posix_memalign(&ptr, alignment, size) != 0) {
    return nullptr;
  }
  return ptr;
}
inline void NPCIE_AlignedFree(void *ptr) { free(ptr); }
#endif
// ==========================================
// 2. Aligned Tensor Class
// ==========================================
namespace npcie {
/**
 * An 1d array contains 2D Tensor data
 * Each data's elements will be quantized from float to short. (without losing
 * accuracy)
 *
 *  Format:
 *
 *              |-------------single data-------------|
 *      [ ... , Valance, Arousal, Dominance, 0(padding) ... ]
 */
class Int16Tensor {
public:
  // saved data-----------------------------------------------------------
  // Main integer array(2bytes)
  short *data = nullptr;

  // MetaData
  std::vector<std::string> term_list; // term of emotions
  std::vector<int> idx_s;             // index
  size_t item_number = 0;
  size_t dimension;

  // Quantization: 0.9999 -> 29997
  const float SCALE = 30000.0f;

  // buffer for score calculation
  std::vector<std::pair<int, int>> score_buffer;
  //----------------------------------------------------------------------

  // Methods--------------------------------------------------------------
  /**
   * Name: Int16Tensor(size_t n, size_t d) constructor
   */
  Int16Tensor(size_t n, size_t d) : item_number(n), dimension(d) {
    size_t total_arr_size = n * d * sizeof(short);

    // 64 byte alignment
    this->data = (short *)NPCIE_AlignedMalloc(total_arr_size, 64);

    if (this->data)
      std::memset(this->data, 0, total_arr_size);

    // resize elements
    this->idx_s.resize(n);
    this->term_list.resize(n);
    this->score_buffer.resize(n);

    std::cout << "[Alloc] Compressed " << n << " vectors into "
              << (total_arr_size / 1024.0f) << " KB (short)." << std::endl;
  }
  /**
   * Name: ~Int16Tensor() deconstructor
   */
  ~Int16Tensor() {
    if (this->data)
      NPCIE_AlignedFree(this->data);
  }

  /**
   *
   */
  void add_data(size_t index, int id, const std::vector<float> &raw_vec) {
    if (index >= item_number)
      return;

    this->idx_s[index] = id; // save id

    for (size_t i = 0; i < this->dimension; ++i) {
      float val = raw_vec[i] * this->SCALE;

      if (val > 32767.0f)
        val = 32767.0f;

      if (val < -32768.0f)
        val = -32768.0f;

      this->data[index * this->dimension + i] = static_cast<short>(val);
    }
  }

  /**
   *  input : query vector, k (default =1)
   *  output: {{"term", score}, {"Rage", 0.85} ...}
   */
  std::vector<std::pair<std::string, float>>
  search_knn(const std::vector<float> &query_raw, int k = 1) {
    std::vector<std::pair<std::string, float>> results;
    if (item_number == 0)
      return results;

    // 1. quantize query (Float -> Short)
    alignas(64) short query_quantized[4] = {0};
    for (size_t i = 0; i < 3; ++i) {
      float val = query_raw[i] * this->SCALE;

      if (val > 32767.0f)
        val = 32767.0f;
      else if (val < -32768.0f)
        val = -32768.0f;

      query_quantized[i] = static_cast<short>(val);
    }

    // 2. calculate score
    for (size_t i = 0; i < item_number; ++i) {
      const short *target =
          &this->data[i *
                      4]; // i * 4 is only operated by bit shift (super fast)
      int score = (int)target[0] * (int)query_quantized[0] +
                  (int)target[1] * (int)query_quantized[1] +
                  (int)target[2] * (int)query_quantized[2] +
                  (int)target[3] * (int)query_quantized[3];

      this->score_buffer[i] = {score, (int)i};
    }

    // 3. get Top-k
    if (k > item_number)
      k = item_number;

    std::partial_sort(
        score_buffer.begin(), score_buffer.begin() + k, score_buffer.end(),
        [](const std::pair<int, int> &a, const std::pair<int, int> &b) {
          return a.first > b.first;
        });

    // 4. get result (recover: Int Score -> Float Similarity)
    results.reserve(k);
    const float DIVISOR = this->SCALE * this->SCALE;

    for (int i = 0; i < k; i++) {
      int idx = this->score_buffer[i].second;
      int raw_score = this->score_buffer[i].first;

      float similarity = (float)raw_score / DIVISOR;

      results.push_back({this->term_list[idx], similarity});
    }

    return results;
  }

  void import_json(const json &j) {
    size_t idx = 0;
    for (const auto &item : j) {
      if (idx >= item_number)
        break;

      term_list[idx] = item["term"].get<std::string>();

      this->idx_s[idx] = (int)idx;

      float vec[3];
      vec[0] = item["valence"].get<float>();
      vec[1] = item["arousal"].get<float>();
      vec[2] = item["dominance"].get<float>();

      // Quantization
      for (int i = 0; i < 3; ++i) {
        float val = vec[i] * this->SCALE;

        // prevent overflow
        if (val > 32767.0f)
          val = 32767.0f;

        if (val < -32768.0f)
          val = -32768.0f;

        // put 0 in [3] as padding (for AVX-512)
        this->data[idx * this->dimension + i] = static_cast<short>(val);
      }
      idx++;
    }
    std::cout << "[Import] Successfully imported " << idx << " items."
              << std::endl;
  }
};
}; // namespace npcie
#endif
