/**
 * Ayin — VAD finding-and-processing pipeline.
 * Upgraded emotion search using CPU cache. Int16Tensor's distance scan is
 * runtime-dispatched: AVX-512 (e.g. Ryzen 9 9950x) -> AVX2 (e.g. Core Ultra
 * 265K) -> scalar, so one binary runs on any x86-64 CPU.
 */
#include "deltaEGO/Ayin.hpp"

#include <cmath>
#include <cstddef>
#include <cstring> // for memset
#include <emmintrin.h>
#include <immintrin.h> // AVX2 / AVX-512 intrinsics
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <fstream>
#include "Third_Party/json.hpp"
#include <yaml-cpp/node/parse.h>
#include <yaml-cpp/yaml.h>

using json = nlohmann::json; // to read VAD.json

// ==========================================
// Platform detection & aligned alloc helpers. Only Int16Tensor (below)
// needs these, so they live here rather than in deltaEGO.cpp — moved
// unchanged when Int16Tensor moved.
// ==========================================
#ifdef _WIN32
#include <malloc.h>
#include <windows.h>
namespace {
void PinThreadToCore(int coreID) {
  HANDLE threadHandle = GetCurrentThread();
  DWORD_PTR mask = (static_cast<DWORD_PTR>(1) << coreID);
  if (SetThreadAffinityMask(threadHandle, mask) == 0) {
    std::cerr << "[Warning] Core pinning failed for ID " << coreID << std::endl;
  } else {
    std::cout << "[System] Thread pinned to Logical Core " << coreID
              << " (CCD 1 Isolated)" << std::endl;
  }
}
void *NPCIE_AlignedMalloc(size_t size, size_t alignment) {
  return _aligned_malloc(size, alignment);
}
void NPCIE_AlignedFree(void *ptr) { _aligned_free(ptr); }
} // namespace
#else
#include <pthread.h>
#include <sched.h>
#include <stdlib.h>
#include <unistd.h>

namespace {

void PinThreadToCore(int coreID)
{
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(coreID, &cpuset);
  pthread_t current_thread = pthread_self();
  if (pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset) != 0)
  {
    std::cerr << "[Warning] Core pinning failed for ID " << coreID << std::endl;
  }
  else
  {
    std::cout << "[System] Thread pinned to Logical Core " << coreID
              << " (Linux/WSL)" << std::endl;
  }
}

void *NPCIE_AlignedMalloc(size_t size, size_t alignment)
{
  void *ptr = nullptr;
  // instead of C++17 aligned_alloc, using posix_memalign for better
  // compatability
  if (posix_memalign(&ptr, alignment, size) != 0)
  {
    return nullptr;
  }
  return ptr;
}

void NPCIE_AlignedFree(void *ptr) { free(ptr); }

} // namespace
#endif

namespace deltaEGO {
namespace Ayin {

// ==========================================
// SIMD distance kernels for Int16Tensor::search_knn.
//
// Each kernel scores `n_blocks` blocks of 8 vectors (8 x 4 shorts = 64 bytes,
// 64-byte aligned) against one quantized query, writing {squared_dist, index}
// pairs into `out[first_index + 0 .. first_index + n_blocks*8)`.
//
// The per-function `target` attribute lets AVX2/AVX-512 code live in the same
// translation unit with no global -march flags; the CPU is checked once at
// runtime (simd::detect_level) so an AVX-512 kernel is never executed on a CPU
// without it (Intel consumer CPUs such as Arrow Lake have no AVX-512).
// ==========================================
// BEGIN-SIMD-KERNELS
namespace simd {

enum class Level { Scalar, AVX2, AVX512 };

inline Level detect_level()
{
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
  __builtin_cpu_init();
  if (__builtin_cpu_supports("avx512f"))
    return Level::AVX512;
  if (__builtin_cpu_supports("avx2"))
    return Level::AVX2;
#endif
  return Level::Scalar;
}

#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))

__attribute__((target("avx512f")))
inline void score_blocks_avx512(const short *data, size_t n_blocks,
                                const int query32[4], std::pair<int, int> *out,
                                size_t first_index)
{
  // load 128 bitss(4 ints) and copy and paste to 512 bit regis 4 times(Broadcast)
  __m128i q_128 = _mm_loadu_si128((const __m128i *)query32);
  __m512i v_q32 = _mm512_broadcast_i32x4(q_128);

  // aligned array to extract results
  alignas(64) int out_lo[16];
  alignas(64) int out_hi[16];

  // handle 8 vectors(32 short = 64 bytes) per loop
  size_t i = first_index;
  for (size_t block = 0; block < n_blocks; ++block, i += 8)
  {
    // laod aligned 64 byte data with _mm512_load_si512
    __m512i vec_target = _mm512_load_si512((__m512i *)&data[i * 4]);

    // to convert 16 bit -> 32 bit, devide data as upper/lower 256 bits
    __m256i target_lo = _mm512_castsi512_si256(vec_target);
    __m256i target_hi = _mm512_extracti64x4_epi64(vec_target, 1);

    //16 bit -> 32 bit  up casting
    __m512i vec_t32_lo = _mm512_cvtepi16_epi32(target_lo);
    __m512i vec_t32_hi = _mm512_cvtepi16_epi32(target_hi);

    // calculate difference (Target - Query)
    __m512i diff_lo = _mm512_sub_epi32(vec_t32_lo, v_q32);
    __m512i diff_hi = _mm512_sub_epi32(vec_t32_hi, v_q32);

    // power (Diff * Diff)
    __m512i square_lo = _mm512_mullo_epi32(diff_lo, diff_lo);
    __m512i square_hi = _mm512_mullo_epi32(diff_hi, diff_hi);

    // 1st : addition with Suffle and Adds
    // in 128 bit lanes, swap lo/hi 64 bit(0x4E) and add
    // [x, y, z, pad] + [z, pad, x, y] = [x+z, y+pad, x+z, y+pad]
    __m512i shuf1_lo = _mm512_shuffle_epi32(square_lo, static_cast<_MM_PERM_ENUM>(_MM_SHUFFLE(1, 0, 3, 2)));
    __m512i shuf1_hi = _mm512_shuffle_epi32(square_hi, static_cast<_MM_PERM_ENUM>(_MM_SHUFFLE(1, 0, 3, 2)));

    __m512i sum1_lo = _mm512_add_epi32(square_lo, shuf1_lo);
    __m512i sum1_hi = _mm512_add_epi32(square_hi, shuf1_hi);

    //2nd : in 128 bit lane, swap close 32 bit (0xB1) and add
    //[x+z, y+pad, ...] + [y+pad, x+z, ...] = [x+y+z+pad, x+y+z+pad, ...]
    __m512i shuf2_lo = _mm512_shuffle_epi32(sum1_lo, static_cast<_MM_PERM_ENUM>(_MM_SHUFFLE(2, 3, 0, 1)));
    __m512i shuf2_hi = _mm512_shuffle_epi32(sum1_hi, static_cast<_MM_PERM_ENUM>(_MM_SHUFFLE(2, 3, 0, 1)));

    __m512i sum2_lo = _mm512_add_epi32(sum1_lo, shuf2_lo);
    __m512i sum2_hi = _mm512_add_epi32(sum1_hi, shuf2_hi);

    //Extract data from SIMD regis
    _mm512_store_epi32(out_lo, sum2_lo);
    _mm512_store_epi32(out_hi, sum2_hi);

    out[i + 0] = {out_lo[0],  (int)(i + 0)};
    out[i + 1] = {out_lo[4],  (int)(i + 1)};
    out[i + 2] = {out_lo[8],  (int)(i + 2)};
    out[i + 3] = {out_lo[12], (int)(i + 3)};

    out[i + 4] = {out_hi[0],  (int)(i + 4)};
    out[i + 5] = {out_hi[4],  (int)(i + 5)};
    out[i + 6] = {out_hi[8],  (int)(i + 6)};
    out[i + 7] = {out_hi[12], (int)(i + 7)};
  }
}

// Same math as the AVX-512 kernel on 256-bit registers: one register holds
// 2 vectors (2 x 4 ints), so a block of 8 vectors takes 4 groups.
__attribute__((target("avx2")))
inline void score_blocks_avx2(const short *data, size_t n_blocks,
                              const int query32[4], std::pair<int, int> *out,
                              size_t first_index)
{
  __m128i q_128 = _mm_loadu_si128((const __m128i *)query32);
  __m256i v_q32 = _mm256_broadcastsi128_si256(q_128);

  size_t i = first_index;
  for (size_t block = 0; block < n_blocks; ++block, i += 8)
  {
    // 2 x 32 bytes = 8 vectors
    __m256i raw0 = _mm256_load_si256((const __m256i *)&data[i * 4]);
    __m256i raw1 = _mm256_load_si256((const __m256i *)&data[i * 4 + 16]);

    // 16 bit -> 32 bit; each group = 2 vectors (one per 128-bit lane)
    __m256i g[4] = {
        _mm256_cvtepi16_epi32(_mm256_castsi256_si128(raw0)),
        _mm256_cvtepi16_epi32(_mm256_extracti128_si256(raw0, 1)),
        _mm256_cvtepi16_epi32(_mm256_castsi256_si128(raw1)),
        _mm256_cvtepi16_epi32(_mm256_extracti128_si256(raw1, 1)),
    };

    for (int k = 0; k < 4; ++k)
    {
      __m256i diff = _mm256_sub_epi32(g[k], v_q32);
      __m256i sq = _mm256_mullo_epi32(diff, diff);
      // horizontal add inside each 128-bit lane: every element = x+y+z+pad
      __m256i s = _mm256_hadd_epi32(sq, sq);
      s = _mm256_hadd_epi32(s, s);

      size_t v = i + (size_t)k * 2;
      out[v + 0] = {_mm256_extract_epi32(s, 0), (int)(v + 0)};
      out[v + 1] = {_mm256_extract_epi32(s, 4), (int)(v + 1)};
    }
  }
}

#endif // x86 GCC/Clang

} // namespace simd
// END-SIMD-KERNELS

/**
 * An 1d array contains 2D Tensor data
 * Each data's element will be quantized from float to short. (without losing
 * accuracy)
 *
 *  Format:
 *
 *              |-------------single data-------------|
 *      [ ... , Valance, Arousal, Dominance, 0(padding) ... ]
 *
 * Owned exclusively by Angela — this is the AVX-512 nearest-neighbor
 * search backing store, not something anyone outside Angela touches.
 */
class Int16Tensor
{
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
  const float SCALE = 10000.0f;

  // buffer for score calculation
  std::vector<std::pair<int, int>> score_buffer;
  //----------------------------------------------------------------------

  // Methods--------------------------------------------------------------
  /**
  * Name: Int16Tensor(size_t n, size_t d) constructor
  */
  Int16Tensor(size_t n, size_t d) : item_number(n), dimension(d)
  {
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
  ~Int16Tensor()
  {
    if (this->data)
      NPCIE_AlignedFree(this->data);
  }

  /**
  * Adds quantized data into allegned array.
  *
  * input:
  *        index:  nth-element
  *        id:     id of the elements
  *        raw_vec:raw_vector values in the container
  */
  void add_data(size_t index, int id, const std::vector<float> &raw_vec)
  {
    if (index >= this->item_number)
      return;

    this->idx_s[index] = id; // save id

    for (size_t i = 0; i < this->dimension; ++i)
    {
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
  search_knn(const std::vector<float> &query_raw, int k = 1)
  {
    std::vector<std::pair<std::string, float>> results;
    if (item_number == 0)
      return results;

    // 1. quantize query (Float -> Short)
    alignas(64) short query_quantized[4] = {0};
    for (size_t i = 0; i < 3; ++i)
    {
      float val = query_raw[i] * this->SCALE;

      if (val > 32767.0f)
        val = 32767.0f;

      else if (val < -32768.0f)
        val = -32768.0f;

      query_quantized[i] = static_cast<short>(val);
    }

    // 2. calculate scores (runtime-dispatched: AVX-512 -> AVX2 -> scalar)
    size_t i = 0;
    size_t n_blocks = item_number / 8;  // handle 8 vectors in one

    // save query vector into 32 bit array (prevent overflow when calculates d0, d1, d2)
    const int query32_arr[4] = {query_quantized[0], query_quantized[1],
                                query_quantized[2], query_quantized[3]};

    static const simd::Level level = simd::detect_level();
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
    if (level == simd::Level::AVX512)
    {
      simd::score_blocks_avx512(this->data, n_blocks, query32_arr,
                                this->score_buffer.data(), 0);
      i = n_blocks * 8;
    }
    else if (level == simd::Level::AVX2)
    {
      simd::score_blocks_avx2(this->data, n_blocks, query32_arr,
                              this->score_buffer.data(), 0);
      i = n_blocks * 8;
    }
#endif

    // handle the data's remainder is not multiples of 8
    // (also everything, on CPUs without AVX2/AVX-512)
    for (; i < item_number; ++i)
    {
      const short *target = &this->data[i * 4];
      int d0 = (int)target[0] - (int)query_quantized[0];
      int d1 = (int)target[1] - (int)query_quantized[1];
      int d2 = (int)target[2] - (int)query_quantized[2];
      int dist_sq = d0 * d0 + d1 * d1 + d2 * d2;
      this->score_buffer[i] = {dist_sq, (int)i};
    }

    // 3. get Top-k
    if ((size_t)k > item_number)
      k = (int)item_number;

    std::partial_sort(
        score_buffer.begin(), score_buffer.begin() + k, score_buffer.end(),
        [](const std::pair<int, int> &a, const std::pair<int, int> &b)
        {
          return a.first < b.first;
        });

    // 4. get result (recover: Int Distance -> Float Similarity)
    results.reserve(k);
    const float DIVISOR = this->SCALE * this->SCALE;

    const float MAX_DIST = std::sqrt(12.0f);

    for (int i = 0; i < k; i++)
    {
      int idx = this->score_buffer[i].second;
      int raw_dist_sq = this->score_buffer[i].first;

      float float_dist_sq = (float)raw_dist_sq / DIVISOR;
      float float_dist = std::sqrt(float_dist_sq);

      float similarity = 1.0f - (float_dist / MAX_DIST);
      similarity = std::max(0.0f, std::min(1.0f, similarity));

      results.push_back({this->term_list[idx], similarity});
    }
    return results;
  }

  void import_json(const json &j)
  {
    size_t idx = 0;
    for (const auto &item : j)
    {
      if (idx >= item_number)
        break;

      term_list[idx] = item["term"].get<std::string>();

      this->idx_s[idx] = (int)idx;

      float vec[3];
      vec[0] = item["valence"].get<float>();
      vec[1] = item["arousal"].get<float>();
      vec[2] = item["dominance"].get<float>();

      // Quantization
      for (int i = 0; i < 3; ++i)
      {
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

/**
 * Angela: owns the "VAD finding" half of Ayin's job — given a VAD state,
 * find the nearest known emotion term via Int16Tensor's SIMD search.
 */
class Angela
{
public:
  Angela() = default;

  ~Angela()
  {
    if (vad_db_)
      delete vad_db_;
  }

  Angela(const Angela &) = delete;
  Angela &operator=(const Angela &) = delete;

  bool LoadVadDb(const std::string &json_path)
  {
    try
    {
      std::ifstream f(json_path);

      if (!f.is_open())
        return false;

      json j = json::parse(f);

      if (vad_db_)
        delete vad_db_;

      vad_db_ = new Int16Tensor(j.size(), 4);
      vad_db_->import_json(j);

      return true;
    }
    catch (...)
    {
      return false;
    }
  }

  // Returns {term, similarity} for the nearest known emotion term to
  // `point`, or {"Unknown", 0.0f} if no DB is loaded yet / nothing found.
  std::pair<std::string, float> FindNearestTerm(const structs::VAD_Point &point)
  {
    if (!vad_db_)
      return {"Unknown", 0.0f};

    std::vector<float> query = {point.V, point.A, point.D};
    auto results = vad_db_->search_knn(query, 1);
    if (results.empty())
      return {"Unknown", 0.0f};

    return results[0];
  }

private:
  Int16Tensor *vad_db_ = nullptr;
};

/**
 * This class will control VAD more humanlike.
 * Key weights:
 *      Big 5 traits
 *        Openness
 *        Conscientiousness
 *        Extraversion
 *        Agreeableness
 *        Neuroticism
 *      Axis
 *        default_emotion_Axis
 *        average_emotion_Axis
 *
 * Owned exclusively by Roland — the pure physics/analysis math, no
 * search, no I/O beyond loading its own YAML.
 */
class emotionPhysics
{
public:
  // Yaml path
  std::string setting_path;

  // Variables---------------------
  // [Big 5 Traits]
  structs::OCEAN ocean;

  // [Dynamic Physics Weights] (Modulated by Analysis)
  structs::DynamicPhysicsWeights DYPhysicsW;

  // [Analysis Config Weights] (From original struct weight)
  structs::AnalysisConfigWeights AnalysisConfigW;

  // [calculate Physics Weights]
  structs::calculatePhysicsWeights CalPyhicsW;

  // 4. Advanced config
  structs::AdvancedConfig AdvanConfig;
  //-------------------------------

  /*
   * Loads weights with YAML
   * */
  emotionPhysics(const std::string& setting_yaml_path)
  {
    this->setting_path = setting_yaml_path;
    this->reloadYAMLSettings(this->setting_path);
  }

  void reloadYAMLSettings(std::string path)
  {
  try
    {
      YAML::Node config = YAML::LoadFile(path);

      // 1. [OCEAN]
      auto oceanNode = config["OCEAN"];
      ocean.Openness          = oceanNode["Openness"].as<float>(0.5f);
      ocean.Conscientiousness = oceanNode["Conscientiousness"].as<float>(0.5f);
      ocean.Extraversion      = oceanNode["Extraversion"].as<float>(0.5f);
      ocean.Agreeableness     = oceanNode["Agreeableness"].as<float>(0.5f);
      ocean.Neuroticism       = oceanNode["Neuroticism"].as<float>(0.5f);

      // 2. [AnalysisConfigWeights]
      auto aw = config["AnalysisWeights"];
      AnalysisConfigW.stabilityRadius   = aw["stabilityRadius"].as<double>();
      AnalysisConfigW.weightA_stress    = aw["weightA_stress"].as<double>();
      AnalysisConfigW.weightV_stress    = aw["weightV_stress"].as<double>();
      AnalysisConfigW.weightV_reward    = aw["weightV_reward"].as<double>();
      AnalysisConfigW.weightA_reward    = aw["weightA_reward"].as<double>();
      AnalysisConfigW.dampening_factor  = aw["dampening_factor"].as<double>();
      AnalysisConfigW.weight_k          = aw["weight_k"].as<double>();
      AnalysisConfigW.theta_0           = aw["theta_0"].as<double>();

      // 3. [calculatePhysicsWeights]
      auto fw = config["FormulaWeights"];
      CalPyhicsW.Sensitivity.pos_extraversion_sensi  = fw["sensitivity"]["pos_extraversion"].as<double>(0.5);
      CalPyhicsW.Sensitivity.pos_openness_sensi      = fw["sensitivity"]["pos_openness"].as<double>(0.2);
      CalPyhicsW.Sensitivity.neg_neuroticism_sensi   = fw["sensitivity"]["neg_neuroticism"].as<double>(0.8);

      CalPyhicsW.resistance.base_resis               = fw["resistance"]["base"].as<double>(0.5);
      CalPyhicsW.resistance.conscientiousness_resis  = fw["resistance"]["conscientiousness"].as<double>(0.4);
      CalPyhicsW.resistance.openness_resis           = fw["resistance"]["openness"].as<double>(-0.1);
      CalPyhicsW.resistance.clamp_min_resis          = fw["resistance"]["clamp_min"].as<float>(0.1f);
      CalPyhicsW.resistance.clamp_max_resis          = fw["resistance"]["clamp_max"].as<float>(0.9f);

      CalPyhicsW.decay.base_decay                    = fw["decay"]["base"].as<double>(0.05);
      CalPyhicsW.decay.conscientiousness_decay       = fw["decay"]["conscientiousness"].as<double>(0.1);
      CalPyhicsW.decay.neuroticism_decay             = fw["decay"]["neuroticism"].as<double>(-0.05);
      CalPyhicsW.decay.clamp_min_decay               = fw["decay"]["clamp_min"].as<float>(0.01f);
      CalPyhicsW.decay.clamp_max_decay               = fw["decay"]["clamp_max"].as<float>(0.3f);

      // 4. [AdvancedConfig]
      auto adv = config["AdvancedConfig"];
      AdvanConfig.stress_threshold_modu           = adv["stress_threshold"].as<double>(0.7);
      AdvanConfig.stress_sensitivity_factor_modu  = adv["stress_sensitivity_factor"].as<double>(1.2);
      AdvanConfig.lability_threshold_advcon       = adv["lability_threshold"].as<double>(0.5);
      AdvanConfig.lability_resistance_mult_modu   = adv["lability_resistance_mult"].as<double>(0.8);
      AdvanConfig.reward_threshold                = adv["reward_threshold"].as<double>(0.6);
      AdvanConfig.reward_decay_boost_modu         = adv["reward_decay_boost"].as<double>(1.5);
      AdvanConfig.history_lookback_size           = adv["history_lookback_size"].as<double>(200.0);
      AdvanConfig.dt_factor                       = adv["dt_factor"].as<double>(0.01);
      AdvanConfig.expression_r                    = adv["expression_r"].as<double>(0.4);
      AdvanConfig.epsilon                         = adv["epsilon"].as<double>(1e-9);
    }
    catch (const std::exception& e)
    {
      std::cerr << "YAML Load Error! : " << e.what() << std::endl;
    }

  this->updatePhysicsWeights();
  }

  // Main Orchestration: Analyze -> Modulate -> Update
  structs::AnalysisResult orchestrate(const std::vector<structs::VAD_Point>& history,
        const structs::VAD_Point& input_stimulus,
        structs::VAD_Point& current_state,
        const structs::VAD_Point& default_state)
  {
        // 1. [Analyze] Run full logic (O1 + On)
        structs::AnalysisResult analysis_result = runFullAnalysis(history, current_state, default_state);

        // initalize physics weight into original values
        this->updatePhysicsWeights();

        // 2. [Modulate] Adjust physics based on precise Ratios & Lability
        modulatePhysics(analysis_result);

        // 3. [Update] Apply forces
        updateEmotion(input_stimulus, current_state, default_state);

        return analysis_result;
  }

  void updatePhysicsWeights()
  {
    // 1. Sensitivity
    this->DYPhysicsW.sensitivity_positive = 1.0f + (this->ocean.Extraversion * this->CalPyhicsW.Sensitivity.pos_extraversion_sensi)
                                      + (this->ocean.Openness * this->CalPyhicsW.Sensitivity.pos_openness_sensi);
    this->DYPhysicsW.sensitivity_negative = 1.0f + (this->ocean.Neuroticism * this->CalPyhicsW.Sensitivity.neg_neuroticism_sensi);

    // 2. resistance
    this->DYPhysicsW.emotion_resistance = this->CalPyhicsW.resistance.base_resis + (this->ocean.Conscientiousness * this->CalPyhicsW.resistance.conscientiousness_resis)
                                                - (this->ocean.Openness * this->CalPyhicsW.resistance.openness_resis);
    this->DYPhysicsW.emotion_resistance = std::clamp(this->DYPhysicsW.emotion_resistance,
                                                    this->CalPyhicsW.resistance.clamp_min_resis,
                                                    this->CalPyhicsW.resistance.clamp_max_resis);

    // 3. bias_decay_rate
    this->DYPhysicsW.bias_decay_rate = this->CalPyhicsW.decay.base_decay + (this->ocean.Conscientiousness * this->CalPyhicsW.decay.conscientiousness_decay)
                                             - (this->ocean.Neuroticism * this->CalPyhicsW.decay.neuroticism_decay);
    this->DYPhysicsW.bias_decay_rate = std::clamp(this->DYPhysicsW.bias_decay_rate, this->CalPyhicsW.decay.clamp_min_decay, this->CalPyhicsW.decay.clamp_max_decay);
  }

  void updateEmotion(const structs::VAD_Point& input_stimulus, structs::VAD_Point& current, const structs::VAD_Point& default_state)
  {
      structs::VAD_Point target_emotion;
      float sens = (input_stimulus.V >= 0) ? this->DYPhysicsW.sensitivity_positive : this->DYPhysicsW.sensitivity_negative;

      target_emotion.V = input_stimulus.V * sens;
      target_emotion.A = input_stimulus.A * sens;
      target_emotion.D = input_stimulus.D;

      current.V = func::lerp(target_emotion.V, current.V, this->DYPhysicsW.emotion_resistance);
      current.A = func::lerp(target_emotion.A, current.A, this->DYPhysicsW.emotion_resistance);
      current.D = func::lerp(target_emotion.D, current.D, this->DYPhysicsW.emotion_resistance);

      current.radius = (std::abs(current.A) + std::abs(current.V)) * 0.5f;

      current.V = func::lerp(current.V, default_state.V, this->DYPhysicsW.bias_decay_rate);
      current.A = func::lerp(current.A, default_state.A, this->DYPhysicsW.bias_decay_rate);
      current.D = func::lerp(current.D, default_state.D, this->DYPhysicsW.bias_decay_rate);

      current.V = std::clamp(current.V, -1.0f, 1.0f);
      current.A = std::clamp(current.A, -1.0f, 1.0f);
      current.D = std::clamp(current.D, -1.0f, 1.0f);
  }


  void modulatePhysics(const structs::AnalysisResult& analysis_result)
  {
    // 1. Stress Ratio Dominance -> Hyper-Sensitivity to Negative
    // If stree ratio is over 80%, it will be sensitive to Negativity
    if (analysis_result.cumulative.stress_ratio > this->AdvanConfig.stress_threshold_modu)
    {
      float factor = static_cast<float>((analysis_result.cumulative.stress_ratio
                                          - this->AdvanConfig.stress_threshold_modu)
                                          * this->AdvanConfig.stress_sensitivity_factor_modu); // 0.0 ~ 0.8
      this->DYPhysicsW.sensitivity_negative *= (1.0f + factor);
    }

    // 2. Affective Lability -> Low Resistance (Mental Whiplash)
    if (analysis_result.dynamics.affective_lability > this->AdvanConfig.lability_threshold_advcon)
    {
       this->DYPhysicsW.emotion_resistance *= static_cast<float>(this->AdvanConfig.lability_resistance_mult_modu);
    }

    // 3. Reward Ratio -> Recovery Boost
    if (analysis_result.cumulative.reward_ratio > this->AdvanConfig.reward_threshold)
    {
        this->DYPhysicsW.bias_decay_rate *= (1.0f + static_cast<float>(analysis_result.cumulative.reward_ratio * this->AdvanConfig.reward_decay_boost_modu));
    }
  }

  structs::AnalysisResult runFullAnalysis(const std::vector<structs::VAD_Point>& history, const structs::VAD_Point& current, const structs::VAD_Point& baseline)
  {
      structs::AnalysisResult res = {};

      // --- 1. O(1) Tasks (Instant) ---

      // A. Deviation & Stress
      double dist = func::get_distance(current, baseline);
      double dampener = (dist <= this->AnalysisConfigW.stabilityRadius) ? this->AnalysisConfigW.dampening_factor : 1.0;

      double stressV = this->AnalysisConfigW.weightV_stress * ((1.0 - current.V) / 2.0);
      double stressA = this->AnalysisConfigW.weightA_stress * current.A;
      res.instant.stress = std::min(1.0, std::max(0.0, stressV + stressA)) * dampener;

      // B. Reward
      double rewardV = this->AnalysisConfigW.weightV_reward * ((current.V + 1.0) / 2.0);
      double rewardA = this->AnalysisConfigW.weightA_reward * current.A;
      res.instant.reward = std::min(1.0, std::max(0.0, rewardV + rewardA));

      // C. Ratio (Instant)
      double total = res.instant.stress + res.instant.reward;
      if (total > this->AdvanConfig.epsilon)
      {
          res.instant.ratio_total = total;
          res.instant.stress_ratio = res.instant.stress / total;
          res.instant.reward_ratio = res.instant.reward / total;
      }

      // D. Dynamics (Lability)
      if (!history.empty())
      {
          const auto& prev = history.back();
          // dt calculation (using radius as timestamp proxy or assuming constant 1.0 if radius is weight)
          double dt = 1.0; // Simplified for robustness

          res.dynamics.delta = { (current.V - prev.V) / (float)dt, (current.A - prev.A) / (float)dt, (current.D - prev.D) / (float)dt, 0 };

          // Sigmoid Lability Logic
          double horizon_h = std::sqrt(res.dynamics.delta.V * res.dynamics.delta.V + res.dynamics.delta.A * res.dynamics.delta.A);
          double theta = std::atan2(res.dynamics.delta.D, horizon_h);
          double z = this->AnalysisConfigW.weight_k * (theta - this->AnalysisConfigW.theta_0);
          res.dynamics.affective_lability = func::sigmoid(z);
      }

      // --- 2. O(n) Tasks (Cumulative) ---
      size_t h_size = history.size();
      size_t start_idx = (h_size > this->AdvanConfig.history_lookback_size) ? h_size - this->AdvanConfig.history_lookback_size : 0; // Lookback 200

      for (size_t i = start_idx; i < h_size; ++i)
      {
          // Re-using instant logic for history items would be slow,
          // so we approximate or could store pre-calculated values in history.
          // Here we perform a simplified accumulation based on stored VAD:

          double h_dist = func::get_distance(history[i], baseline);
          double h_damp = (h_dist <= this->AnalysisConfigW.stabilityRadius) ? this->AnalysisConfigW.dampening_factor : 1.0;

          double sV = this->AnalysisConfigW.weightV_stress * ((1.0 - history[i].V) / 2.0);
          double sA = this->AnalysisConfigW.weightA_stress * history[i].A;
          double inst_s = std::min(1.0, std::max(0.0, sV + sA)) * h_damp;

          double rV = this->AnalysisConfigW.weightV_reward * ((history[i].V + 1.0) / 2.0);
          double rA = this->AnalysisConfigW.weightA_reward * history[i].A;
          double inst_r = std::min(1.0, std::max(0.0, rV + rA));

          res.cumulative.stress += inst_s * this->AdvanConfig.dt_factor; // dt factor
          res.cumulative.reward += inst_r * this->AdvanConfig.dt_factor;
      }

      // Cumulative Ratio
      double c_total = res.cumulative.stress + res.cumulative.reward;
      if (c_total > this->AdvanConfig.epsilon)
      {
          res.cumulative.total = c_total;
          res.cumulative.stress_ratio = res.cumulative.stress / c_total;
          res.cumulative.reward_ratio = res.cumulative.reward / c_total;
      }

      // frontend expression data
      /*
       * 1. is it neutral?
       *    -> if current VAD has less then radius r
       *    If you want to change emotion sensitivity, adjust expression_r
       * */
      double expression_r = this->AdvanConfig.expression_r; // <--- change this
      double current_r = std::sqrt(current.V * current.V + current.A * current.A + current.D * current.D);

      // calculate
      if(current_r <= expression_r)
      {
        // since it is neutral, no intensity or similarity needed
        res.front.front_expression_name = "neutral";
        res.front.intensity = 0;
        res.front.similarity = 0;
      }
      else
      {
        int v_bit = (current.V < 0) ? 1 : 0;
        int a_bit = (current.A < 0) ? 1 : 0;
        int d_bit = (current.D < 0) ? 1 : 0;

        int index = (v_bit << 2) | (a_bit << 1) | (d_bit << 0);

        const char* emotion_zones[8] = {
          "Excited/Happy", // 000 (V+, A+, D+)
          "Surprise",      // 001 (V+, A+, D-)
          "Relaxed",       // 010 (V+, A-, D+)
          "Calm",          // 011 (V+, A-, D-)
          "Angry/Mad",     // 100 (V-, A+, D+)
          "Fear/Anxious",  // 101 (V-, A+, D-)
          "Disgust",       // 110 (V-, A-, D+)
          "Sad/Lonely"     // 111 (V-, A-, D-)
        };

        res.front.front_expression_name = emotion_zones[index];
        structs::VAD_Point target_axis = {
          std::signbit(current.V) ? -1.0f : 1.0f,
          std::signbit(current.A) ? -1.0f : 1.0f,
          std::signbit(current.D) ? -1.0f : 1.0f,
          0.0f
        };

        res.front.similarity = this->get_cosine_sim(current, target_axis, this->AdvanConfig.epsilon);
        res.front.intensity = this->get_intensity(current_r, expression_r);
      }

    return res;
}

  double get_cosine_sim(const structs::VAD_Point& a, const structs::VAD_Point& b, const double epsilon)
  {
    double dot = (a.V * b.V) + (a.A * b.A) + (a.D * b.D);
    double normA = std::sqrt(a.V * a.V + a.A * a.A + a.D * a.D);
    double normB = std::sqrt(b.V * b.V + b.A * b.A + b.D * b.D);

    if (normA < epsilon || normB < epsilon)
      return 0.0;

    return std::clamp(dot / (normA * normB), -1.0, 1.0);
  }

  int get_intensity(const double input, const double criterion)
  {
    if (input <= criterion)
      return 0;
    const double MAX_RADIUS = std::sqrt(3.0);

    double raw_intensity = (input - criterion) / (MAX_RADIUS - criterion) * 100.0;

    return static_cast<int>(std::clamp(raw_intensity, 0.0, 100.0));
  }
};

/**
 * Roland: owns the "VAD processing" half of Ayin's job — wraps
 * emotionPhysics plus the VAD state/history it operates on, and exposes
 * one Update() call that runs a stimulus through the full
 * analyze->modulate->update physics pipeline.
 */
class Roland
{
public:
  Roland(const std::string &config_path, float def_V, float def_A, float def_D,
         float def_radius)
      : physicsEngine(config_path),
        current_state(structs::VAD_Point{def_V, def_A, def_D, def_radius}),
        default_state(structs::VAD_Point{def_V, def_A, def_D, def_radius})
  {
    history.reserve(MAX_HISTORY_SIZE);
  }

  bool ReloadConfig()
  {
    try
    {
      this->physicsEngine.reloadYAMLSettings(this->physicsEngine.setting_path);
      std::cout << "[Roland] Successfully reloaded YAML config: "
                << this->physicsEngine.setting_path << std::endl;
      return true;
    }
    catch (const std::exception &e)
    {
      std::cerr << "[Roland] Failed to reload YAML: " << e.what() << std::endl;
      return false;
    }
  }

  struct UpdateResult
  {
    structs::VAD_Point current_state;
    structs::AnalysisResult analysis;
  };

  // Runs one stimulus through the physics engine and updates
  // history/current_state in place.
  UpdateResult Update(const structs::VAD_Point &input_stimulus)
  {
    UpdateResult result;
    result.analysis = this->physicsEngine.orchestrate(
        this->history, input_stimulus, this->current_state, this->default_state);

    if (this->history.size() >= this->MAX_HISTORY_SIZE)
    {
      this->history.erase(this->history.begin());
    }
    this->history.push_back(this->current_state);

    result.current_state = this->current_state;
    return result;
  }

private:
  emotionPhysics physicsEngine;
  std::vector<structs::VAD_Point> history;
  structs::VAD_Point current_state;
  structs::VAD_Point default_state;
  static constexpr size_t MAX_HISTORY_SIZE = 1000;
};

// ==========================================
// Ayin — out-of-line definitions for the class declared in Ayin.hpp.
// ==========================================

Ayin::Ayin(const std::string &config_path, float def_V, float def_A,
           float def_D, float def_radius)
    : roland_(std::make_unique<Roland>(config_path, def_V, def_A, def_D,
                                        def_radius)),
      angela_(std::make_unique<Angela>())
{
}

// Defined here (not defaulted inline in Ayin.hpp) because Roland/Angela
// are only forward-declared there — unique_ptr's deleter needs their
// complete type, which is only visible in this .cpp.
Ayin::~Ayin() = default;

bool Ayin::load_vad_db(const std::string &json_path)
{
  return angela_->LoadVadDb(json_path);
}

bool Ayin::reload_config()
{
  return roland_->ReloadConfig();
}

Ayin::ProcessResult Ayin::Process(const structs::VAD_Point &input_stimulus)
{
  Roland::UpdateResult update = this->roland_->Update(input_stimulus);
  auto [term, sim] = this->angela_->FindNearestTerm(update.current_state);

  ProcessResult result;
  result.current_state = update.current_state;
  result.analysis = update.analysis;
  result.emotion_term = term;
  result.similarity = sim;
  return result;
}

} // namespace Ayin
} // namespace deltaEGO
