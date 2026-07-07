/**
 * deltaEGO_ version 2.
 * Upgraded emotion search using CPU cache
 * Optimized only for Ryzen 9 9950x
 * ALL-IN-ONE cpp code
 */

// ==========================================
// 0. Common headers
// ==========================================
#include <cmath>
#include <cstddef>
#include <cstring>     // for memset
#include <emmintrin.h>
#include <immintrin.h> // AVX-512
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <chrono>
#include "json.hpp"
#include <immintrin.h>
#include <yaml-cpp/node/parse.h>
#include <yaml-cpp/yaml.h>


// JSON Library
#include "json.hpp" // to read VAD.json
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

inline void PinThreadToCore(int coreID) 
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

inline void *NPCIE_AlignedMalloc(size_t size, size_t alignment) 
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

inline void NPCIE_AlignedFree(void *ptr) { free(ptr); }


#endif
// ==========================================
// 2. Aligned Tensor Class
// ==========================================
namespace deltaEGO {

  struct VAD_Point
  {
    float V;
    float A;
    float D;
    float radius;
  };

  struct Ratio 
  {
        double stress_raw;
        double reward_raw;
        double ratio_total;
        double stress_ratio;
        double reward_ratio;
  };

  struct AnalysisResult 
  {
      struct 
      {
           double stress, reward, deviation;
           double ratio_total, stress_ratio, reward_ratio;
      } instant;

      struct 
      {
          VAD_Point delta;
          double affective_lability;
      } dynamics;

      struct 
      {
          double stress, reward, total;
          double stress_ratio, reward_ratio;
      } cumulative;

      struct
      {
          std::string front_expression_name;
          int similarity;
          int intensity;
          
      } front;
  };
  // [JSON Serialization]
  // VAD
  inline void to_json(json& j, const VAD_Point& p) 
  {
    j = json
    {
        {"v", p.V},
        {"a", p.A},
        {"d", p.D},
        {"r", p.radius}
    };
  }  
  // AnalysisResult
  inline void to_json(json& j, const AnalysisResult& ar) 
  {
    j = json
    {
        {"instant", {
            {"stress", ar.instant.stress},
            {"reward", ar.instant.reward},
            {"deviation", ar.instant.deviation},
            {"ratio_total", ar.instant.ratio_total},
            {"stress_ratio", ar.instant.stress_ratio},
            {"reward_ratio", ar.instant.reward_ratio}
        }},
        {"dynamics", {
            {"delta", ar.dynamics.delta}, 
            {"lability", ar.dynamics.affective_lability}
        }},
        {"cumulative", {
            {"stress", ar.cumulative.stress},
            {"reward", ar.cumulative.reward},
            {"total", ar.cumulative.total},
            {"stress_ratio", ar.cumulative.stress_ratio},
            {"reward_ratio", ar.cumulative.reward_ratio}
        }},
        {"front", {
            {"expression", ar.front.front_expression_name},
            {"similarity", ar.front.similarity},
            {"intensity", ar.front.intensity}
        }}
    };
  }
  inline void from_json(const json& j, VAD_Point& p) 
  {
      j.at("v").get_to(p.V); j.at("a").get_to(p.A); j.at("d").get_to(p.D); j.at("r").get_to(p.radius);
  }
  inline float lerp(float target, float current, float resistance) 
  {
    return target + (current - target) * resistance;
  }
  inline double get_distance(const VAD_Point& a, const VAD_Point& b) 
  {
      return std::sqrt(std::pow(a.V - b.V, 2) + std::pow(a.A - b.A, 2) + std::pow(a.D - b.D, 2));
  }

  inline double sigmoid(double x) 
  {
      if (x >= 0) 
        return 1.0 / (1.0 + std::exp(-x));

      else 
      {
          double e = std::exp(x);
          return e / (1.0 + e);
       }
  }
/**
 * An 1d array contains 2D Tensor data
 * Each data's el How fast Reminh come back to nomal stateements will be quantized from float to short. (without losing
 * accuracy)
 *
 *  Format:
 *
 *              |-------------single data-------------|
 *      [ ... , Valance, Arousal, Dominance, 0(padding) ... ]
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

 
  struct AnalysisResult 
  {
      struct 
      {
           double stress, reward, deviation;
           double ratio_total, stress_ratio, reward_ratio;
      } instant;

      struct 
      {
          VAD_Point delta;
          double affective_lability;
      } dynamics;

      struct 
      {
          double stress, reward, total;
          double stress_ratio, reward_ratio;
      } cumulative;
  };   // Methods--------------------------------------------------------------
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
    *  TODO: ADD REAL AVX-512 CODE --> done
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

      // 2. calculate score AVX-512
      size_t i = 0;
      size_t n_blocks = item_number / 8;  // handle 8 vectors in one 
      
      // save query vector into 32 bit array (prevent overflow when calculates d0, d1, d2)
      int query32_arr[4] = {query_quantized[0], query_quantized[1], query_quantized[2], query_quantized[3]};
      // load 128 bitss(4 ints) and copy and paste to 512 bit regis 4 times(Broadcast)
      __m128i q_128 = _mm_loadu_si128((__m128i*)query32_arr);
      __m512i v_q32 = _mm512_broadcast_i32x4(q_128);

      // aligned array to extract results
      alignas(64) int out_lo[16];
      alignas(64) int out_hi[16];

      // handle 8 vectors(32 short = 64 bytes) per loop
      for(size_t block = 0; block < n_blocks; ++block, i += 8)
      {
        // laod aligned 64 byte data with _mm512_load_si512
        __m512i vec_target = _mm512_load_si512((__m512i*)&this->data[i * 4]);

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
        
        this->score_buffer[i + 0] = {out_lo[0],  (int)(i + 0)};
        this->score_buffer[i + 1] = {out_lo[4],  (int)(i + 1)};
        this->score_buffer[i + 2] = {out_lo[8],  (int)(i + 2)};
        this->score_buffer[i + 3] = {out_lo[12], (int)(i + 3)};
    
        this->score_buffer[i + 4] = {out_hi[0],  (int)(i + 4)};
        this->score_buffer[i + 5] = {out_hi[4],  (int)(i + 5)};
        this->score_buffer[i + 6] = {out_hi[8],  (int)(i + 6)};
        this->score_buffer[i + 7] = {out_hi[12], (int)(i + 7)};
      }
      
      // handle the data's remainder is not multiples of 8
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


// Analysis
struct OCEAN
{
    float Openness;
    float Conscientiousness;
    float Extraversion;
    float Agreeableness;
    float Neuroticism;
};
struct DynamicPhysicsWeights
{
    // sensitivity with emotion_resistance
    float sensitivity_positive;
    float sensitivity_negative;
  // TODO: bias(d_b1) addition needs to be added
    // how much Reminh weights on emotion
    float emotion_resistance;

    // How fast Reminh come back to nomal state
    float bias_decay_rate;
};
struct AnalysisConfigWeights
{
    double stabilityRadius;
    double weightA_stress;
    double weightV_stress;
    double weightV_reward;
    double weightA_reward;
    double dampening_factor;
    double weight_k;
    double theta_0; 
};
struct calculatePhysicsWeights
{
  struct
  {
    // 1. Sensitivity
    double pos_extraversion_sensi;
    double pos_openness_sensi;
    double neg_neuroticism_sensi;
  }Sensitivity;
  struct
  {
    // 2. resistance
    double base_resis;
    double conscientiousness_resis;
    double openness_resis;
    float  clamp_min_resis;
    float  clamp_max_resis;
  }resistance;
  struct
  {
    // 3. decay
    double base_decay;    
    double conscientiousness_decay;
    double neuroticism_decay;
    float  clamp_min_decay;
    float  clamp_max_decay; 
  }decay;
};
struct AdvancedConfig
{
    double stress_threshold_modu;
    double stress_sensitivity_factor_modu;
    double lability_threshold_advcon;
    double lability_resistance_mult_modu;
    double reward_threshold;
    double reward_decay_boost_modu;
    double history_lookback_size;
    double dt_factor;
    double expression_r;
    double epsilon;
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
 */
  //TODO: Make it more advanced
  class emotionPhysics 
  {
  public:
    // Yaml path
    std::string setting_path;

    // Variables---------------------
    // [Big 5 Traits]
    OCEAN ocean;

    // [Dynamic Physics Weights] (Modulated by Analysis)
    DynamicPhysicsWeights DYPhysicsW;

    // [Analysis Config Weights] (From original struct weight)
    AnalysisConfigWeights AnalysisConfigW;

    // [calculate Physics Weights]
    calculatePhysicsWeights CalPyhicsW;

    // 4. Advanced config
    AdvancedConfig AdvanConfig;
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
    AnalysisResult orchestrate(const std::vector<VAD_Point>& history,
          const VAD_Point& input_stimulus,
          VAD_Point& current_state,
          const VAD_Point& default_state)
    {
          // 1. [Analyze] Run full logic (O1 + On)
          AnalysisResult analysis_result = runFullAnalysis(history, current_state, default_state);
          
          // initalize physics weight into original values
          this->updatePhysicsWeights();

          // 2. [Modulate] Adjust physics based on precise Ratios & Lability
          modulatePhysics(analysis_result);

          // 3. [Update] Apply forces
          updateEmotion(input_stimulus, current_state, default_state);

          return analysis_result;
    }

//TODO: FIX_PIN
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

    void updateEmotion(const VAD_Point& input_stimulus, VAD_Point& current, const VAD_Point& default_state)
    {
        VAD_Point target_emotion;
        float sens = (input_stimulus.V >= 0) ? this->DYPhysicsW.sensitivity_positive : this->DYPhysicsW.sensitivity_negative;
        
        target_emotion.V = input_stimulus.V * sens;
        target_emotion.A = input_stimulus.A * sens; 
        target_emotion.D = input_stimulus.D;
        
        current.V = lerp(target_emotion.V, current.V, this->DYPhysicsW.emotion_resistance);
        current.A = lerp(target_emotion.A, current.A, this->DYPhysicsW.emotion_resistance);
        current.D = lerp(target_emotion.D, current.D, this->DYPhysicsW.emotion_resistance);

        current.radius = (std::abs(current.A) + std::abs(current.V)) * 0.5f;

        current.V = lerp(current.V, default_state.V, this->DYPhysicsW.bias_decay_rate);
        current.A = lerp(current.A, default_state.A, this->DYPhysicsW.bias_decay_rate);
        current.D = lerp(current.D, default_state.D, this->DYPhysicsW.bias_decay_rate);

        current.V = std::clamp(current.V, -1.0f, 1.0f);
        current.A = std::clamp(current.A, -1.0f, 1.0f);
        current.D = std::clamp(current.D, -1.0f, 1.0f);
    }

    
    void modulatePhysics(const AnalysisResult& analysis_result) 
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

    AnalysisResult runFullAnalysis(const std::vector<VAD_Point>& history, const VAD_Point& current, const VAD_Point& baseline)
    {
        AnalysisResult res = {};

        // --- 1. O(1) Tasks (Instant) ---
        
        // A. Deviation & Stress
        double dist = get_distance(current, baseline);
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
            res.dynamics.affective_lability = sigmoid(z);
        }

        // --- 2. O(n) Tasks (Cumulative) ---
        // TODO:(Leaky Integrator) this shit is slow as fck and also consumes fuck ton of mem
        size_t h_size = history.size();
        size_t start_idx = (h_size > this->AdvanConfig.history_lookback_size) ? h_size - this->AdvanConfig.history_lookback_size : 0; // Lookback 200

        for (size_t i = start_idx; i < h_size; ++i) 
        {
            // Re-using instant logic for history items would be slow, 
            // so we approximate or could store pre-calculated values in history.
            // Here we perform a simplified accumulation based on stored VAD:
            
            double h_dist = get_distance(history[i], baseline);
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
          // TODO: Make it more precise(좀더 정교하게)
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
          VAD_Point target_axis = {
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

  double get_cosine_sim(const VAD_Point& a, const VAD_Point& b, const double epsilon) 
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

  class deltaEGO 
  {
  private:
    emotionPhysics physicsEngine;
    Int16Tensor* vad_db = nullptr;
    std::vector<VAD_Point> history;
    VAD_Point current_state;
    VAD_Point default_state;
    const size_t MAX_HISTORY_SIZE = 1000;

  public:
    deltaEGO(std::string config_path, float def_V, float def_A, float def_D, float def_radius)
        : physicsEngine(config_path), 
        default_state(VAD_Point{def_V, def_A, def_D, def_radius}), current_state(VAD_Point{def_V, def_A, def_D, def_radius})
    {
      history.reserve(MAX_HISTORY_SIZE);
    }

    ~deltaEGO() 
    { 
      if (vad_db) 
        delete vad_db; 
    }

    bool load_vad_db(const std::string& json_path) 
    {
      try 
      {
        std::ifstream f(json_path);

        if (!f.is_open()) 
          return false;

        json j = json::parse(f);

        if (vad_db) 
          delete vad_db;
                
        vad_db = new Int16Tensor(j.size(), 4);
        vad_db->import_json(j);

        return true;
      } 
      catch (...) 
      { 
        return false; 
      }
    }

    std::string process_stimulus(float v, float a, float d)
    {
      VAD_Point input = { v, a, d, 1.0f };
      AnalysisResult analysis_result = this->physicsEngine.orchestrate(
          this->history, 
          input, 
          this->current_state, 
          this->default_state);

      if (this->history.size() >= this->MAX_HISTORY_SIZE) 
      {
        this->history.erase(this->history.begin());
      }
      this->history.push_back(this->current_state);
      
      //TODO
      std::string emotion_term = "Unknown";
        float similarity = 0.0f;
        if (this->vad_db) 
        {
          std::vector<float> query = {this->current_state.V, this->current_state.A, this->current_state.D};
          auto results = this->vad_db->search_knn(query, 1);
          if (!results.empty())
          {
            emotion_term = results[0].first;
            similarity = results[0].second;
          }
        }

        json j_out;

        j_out["current_state"]  = this->current_state;
        j_out["emotion_term"]   = emotion_term;
        j_out["similarity"]     = similarity;
        j_out["analysis"]       = analysis_result;
        
        return j_out.dump();
    }

    bool reload_config() 
    {
      try 
      {
        this->physicsEngine.reloadYAMLSettings(this->physicsEngine.setting_path);
        std::cout << "[deltaEGO] Successfully reloaded YAML config: " 
                  << this->physicsEngine.setting_path << std::endl;
        return true;
      } 
      catch (const std::exception& e) 
      {
        std::cerr << "[deltaEGO] Failed to reload YAML: " << e.what() << std::endl;
        return false;
      }
    }
  };
}; // namespace deltaEGO

// ==========================================
// [Python Binding] Pybind11 Code
// ==========================================
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using DeltaEGO_Class = deltaEGO::deltaEGO;

PYBIND11_MODULE(delta_ego_core, m) 
{
  m.doc() = "DeltaEGO: High-performance Emotion Engine (Ryzen 9950x Optimized)";
  
  py::class_<DeltaEGO_Class>(m, "deltaEGO")
    .def(py::init<std::string, float, float, float, float>(),
      py::arg("config_path"), py::arg("def_V"), py::arg("def_A"), py::arg("def_D"), py::arg("def_radius"))
    .def("load_vad_db", &DeltaEGO_Class::load_vad_db, "Load VAD JSON database path")

    .def("reload_config", &DeltaEGO_Class::reload_config, "Reload YAML configuration on the fly")

    .def("process_stimulus", &DeltaEGO_Class::process_stimulus, "Process VAD input and return JSON string");
}
