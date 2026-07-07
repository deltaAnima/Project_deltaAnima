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
#include <cstring>     // for memset
#include <immintrin.h> // AVX-512
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <chrono>
#include "json.hpp"
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
    const float SCALE = 30000.0f;

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

      // 2. calculate score
      for (size_t i = 0; i < item_number; ++i) 
      {
        // i * 4 is only operated by bit shift (super fast)
        const short *target = &this->data[i * 4];
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
          [](const std::pair<int, int> &a, const std::pair<int, int> &b) 
          {
            return a.first > b.first;
          });

      // 4. get result (recover: Int Score -> Float Similarity)
      results.reserve(k);
      const float DIVISOR = this->SCALE * this->SCALE;

      for (int i = 0; i < k; i++) 
      {
        int idx = this->score_buffer[i].second;
        int raw_score = this->score_buffer[i].first;

        float similarity = (float)raw_score / DIVISOR;

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
    // Variables---------------------
    // [Big 5 Traits]
    float Openness;
    float Conscientiousness;
    float Extraversion;
    float Agreeableness;
    float Neuroticism;

    // [Dynamic Physics Weights] (Modulated by Analysis)
    // sensitivity with emotion
    float sensitivity_positive;
    float sensitivity_negative;
    // how much Reminh weights on emotion
    float emotion_resistance;
    // How fast Reminh come back to nomal state
    float bias_decay_rate;

    // [Analysis Config Weights] (From original struct weight)
        double stabilityRadius = 0.2;
        double weightA_stress = 0.5;
        double weightV_stress = 0.5;
        double weightV_reward = 0.5;
        double weightA_reward = 0.5;
        double dampening_factor = 0.5;
        double weight_k = 1.0;
        double theta_0 = 0.0;
    //-------------------------------
    
    
    emotionPhysics(float Openness, float Conscientiousness, float Extraversion,
                   float Agreeableness, float Neuroticism)
                   
    {
      // Weights for VAD calculation (basic weights)
      this->Openness          = Openness;
      this->Conscientiousness = Conscientiousness;
      this->Extraversion      = Extraversion;
      this->Agreeableness     = Agreeableness;
      this->Neuroticism       = Neuroticism; 

      this->calculatePhysicsWeights();
    }

    // Main Orchestration: Analyze -> Modulate -> Update
    AnalysisResult orchestrate(const std::vector<VAD_Point>& history,
          const VAD_Point& input_stimulus,
          VAD_Point& current_state,
          const VAD_Point& default_state)
    {
          // 1. [Analyze] Run full logic (O1 + On)
          AnalysisResult analysis_result = runFullAnalysis(history, current_state, default_state);

          // 2. [Modulate] Adjust physics based on precise Ratios & Lability
          modulatePhysics(analysis_result);

          // 3. [Update] Apply forces
          updateEmotion(input_stimulus, current_state, default_state);

          return analysis_result;
    }

    void calculatePhysicsWeights()
    {
      // 1. Sensitivity
      this->sensitivity_positive = 1.0f + (this->Extraversion * 0.5f) + (this->Openness * 0.2f);
      this->sensitivity_negative = 1.0f + (this->Neuroticism * 0.8f);
      // 2. resistance
      this->emotion_resistance = 0.5f + (this->Conscientiousness * 0.4f) - (this->Openness * 0.1f);
      this->emotion_resistance = std::clamp(this->emotion_resistance, 0.1f, 0.9f);
      // 3. bias_decay_rate
      this->bias_decay_rate = 0.05f + (this->Conscientiousness * 0.1f) - (this->Neuroticism * 0.05f);
      this->bias_decay_rate = std::clamp(this->bias_decay_rate, 0.01f, 0.3f);
    }

    void updateEmotion(const VAD_Point& input_stimulus, VAD_Point& current, const VAD_Point& default_state)
    {
        VAD_Point target_emotion;
        float sens = (input_stimulus.V >= 0) ? this->sensitivity_positive : this->sensitivity_negative;
        
        target_emotion.V = input_stimulus.V * sens;
        target_emotion.A = input_stimulus.A * sens; 
        target_emotion.D = input_stimulus.D;
        
        current.V = lerp(target_emotion.V, current.V, this->emotion_resistance);
        current.A = lerp(target_emotion.A, current.A, this->emotion_resistance);
        current.D = lerp(target_emotion.D, current.D, this->emotion_resistance);

        current.radius = (std::abs(current.A) + std::abs(current.V)) * 0.5f;

        current.V = lerp(current.V, default_state.V, this->bias_decay_rate);
        current.A = lerp(current.A, default_state.A, this->bias_decay_rate);
        current.D = lerp(current.D, default_state.D, this->bias_decay_rate);

        current.V = std::clamp(current.V, -1.0f, 1.0f);
        current.A = std::clamp(current.A, -1.0f, 1.0f);
        current.D = std::clamp(current.D, -1.0f, 1.0f);
    }

    void resetPhysicsWeights() 
    {
      this->sensitivity_positive  = 1.0f + (this->Extraversion * 0.5f) + (this->Openness * 0.2f);
      this->sensitivity_negative  = 1.0f + (this->Neuroticism * 0.8f);

      this->emotion_resistance    = 0.5f + (this->Conscientiousness * 0.4f) - (this->Openness * 0.1f);
      this->emotion_resistance    = std::clamp(this->emotion_resistance, 0.1f, 0.95f);
      
      this->bias_decay_rate       = 0.05f + (this->Conscientiousness * 0.1f) - (this->Neuroticism * 0.05f);
      this->bias_decay_rate       = std::clamp(this->bias_decay_rate, 0.01f, 0.3f);
    }
    
    void modulatePhysics(const AnalysisResult& analysis_result) 
    {
      resetPhysicsWeights();

      // 1. Stress Ratio Dominance -> Hyper-Sensitivity to Negative
      // If stree ratio is over 80%, it will be sensitive to Negativity
      if (analysis_result.cumulative.stress_ratio > 0.6) 
      {
        float factor = (float)(analysis_result.cumulative.stress_ratio - 0.6) * 2.0f; // 0.0 ~ 0.8
        sensitivity_negative *= (1.0f + factor);
      }

      // 2. Affective Lability -> Low Resistance (Mental Whiplash)
      if (analysis_result.dynamics.affective_lability > 0.7) 
      {
         emotion_resistance *= 0.7f; 
      }

      // 3. Reward Ratio -> Recovery Boost
      if (analysis_result.cumulative.reward_ratio > 0.5) 
      {
          bias_decay_rate *= (1.0f + (float)analysis_result.cumulative.reward_ratio * 0.5f);
      }
    }

    AnalysisResult runFullAnalysis(const std::vector<VAD_Point>& history, const VAD_Point& current, const VAD_Point& baseline)
    {
        AnalysisResult res = {};

        // --- 1. O(1) Tasks (Instant) ---
        
        // A. Deviation & Stress
        double dist = get_distance(current, baseline);
        double dampener = (dist <= stabilityRadius) ? dampening_factor : 1.0;
        
        double stressV = weightV_stress * ((1.0 - current.V) / 2.0);
        double stressA = weightA_stress * current.A;
        res.instant.stress = std::min(1.0, std::max(0.0, stressV + stressA)) * dampener;

        // B. Reward
        double rewardV = weightV_reward * ((current.V + 1.0) / 2.0);
        double rewardA = weightA_reward * current.A;
        res.instant.reward = std::min(1.0, std::max(0.0, rewardV + rewardA));

        // C. Ratio (Instant)
        double total = res.instant.stress + res.instant.reward;
        if (total > 1e-9) 
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
            double z = weight_k * (theta - theta_0);
            res.dynamics.affective_lability = sigmoid(z);
        }

        // --- 2. O(n) Tasks (Cumulative) ---
        size_t h_size = history.size();
        size_t start_idx = (h_size > 200) ? h_size - 200 : 0; // Lookback 200

        for (size_t i = start_idx; i < h_size; ++i) 
        {
            // Re-using instant logic for history items would be slow, 
            // so we approximate or could store pre-calculated values in history.
            // Here we perform a simplified accumulation based on stored VAD:
            
            double h_dist = get_distance(history[i], baseline);
            double h_damp = (h_dist <= stabilityRadius) ? dampening_factor : 1.0;
            
            double sV = weightV_stress * ((1.0 - history[i].V) / 2.0);
            double sA = weightA_stress * history[i].A;
            double inst_s = std::min(1.0, std::max(0.0, sV + sA)) * h_damp;

            double rV = weightV_reward * ((history[i].V + 1.0) / 2.0);
            double rA = weightA_reward * history[i].A;
            double inst_r = std::min(1.0, std::max(0.0, rV + rA));

            res.cumulative.stress += inst_s * 0.01; // dt factor
            res.cumulative.reward += inst_r * 0.01;
        }

        // Cumulative Ratio
        double c_total = res.cumulative.stress + res.cumulative.reward;
        if (c_total > 1e-9) 
        {
            res.cumulative.total = c_total;
            res.cumulative.stress_ratio = res.cumulative.stress / c_total;
            res.cumulative.reward_ratio = res.cumulative.reward / c_total;
        }

        // frontend expression data
        /*
         * 1. is it neutural?
         *    -> if current VAD has less then radius r
         *    If you want to change emotion sensitivity, adjust expression_r
         * */
        double expression_r = 0.4; // <--- change this
        double current_r = std::sqrt(current.V * current.V + current.A * current.A + current.D * current.D);
        
        // lambda function for calculate similarity(cosine similarity)
        auto get_cosine_sim = [](const VAD_Point& a, const VAD_Point& b) -> double 
        {
          double dot = (a.V * b.V) + (a.A * b.A) + (a.D * b.D);
          double normA = std::sqrt(a.V * a.V + a.A * a.A + a.D * a.D);
          double normB = std::sqrt(b.V * b.V + b.A * b.A + b.D * b.D);
    
          if (normA < 1e-6 || normB < 1e-6) 
            return 0.0;
          
          return dot / (normA * normB);
        };
        // lambda function for calculate intensity
        auto get_intensity = [](const double& input, const double& criterion) -> int
        {
          return static_cast<int>((input - criterion) / (1.0 - criterion) * 100);
        };

        // calculate
        if(current_r <= expression_r)
        {
          // since it is neutural, no intensity or similarity needed
          // TODO: Make it more precise(좀더 정교하게)
          res.front.front_expression_name = "neutural";
          res.front.intensity = 0;
          res.front.similarity = 0;
        }
        else 
        {
          int index = (std::signbit(current.V) << 2) | // 4 (100)
                      (std::signbit(current.A) << 1) | // 2 (010)
                      (std::signbit(current.D) << 0);  // 1 (001)
                                                          //
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
          
          res.front.similarity = get_cosine_sim(current, target_axis);
          res.front.intensity = get_intensity(current_r, expression_r);
        }

      return res;
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
    deltaEGO(float O, float C, float E, float A, float N, float def_V, float def_A, float def_D, float def_radius)
        : physicsEngine(emotionPhysics(O,C,E,A,N)), 
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
      AnalysisResult analysis_result = this->physicsEngine.orchestrate(this->history, input, this->current_state, this->default_state);

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
    .def(py::init<float, float, float, float, float, float, float, float, float>(),
      py::arg("O"), py::arg("C"), py::arg("E"), py::arg("A"), py::arg("N"),
      py::arg("def_V"), py::arg("def_A"), py::arg("def_D"), py::arg("def_radius"))

    .def("load_vad_db", &DeltaEGO_Class::load_vad_db, "Load VAD JSON database path")

    .def("process_stimulus", &DeltaEGO_Class::process_stimulus, "Process VAD input and return JSON string");
}
