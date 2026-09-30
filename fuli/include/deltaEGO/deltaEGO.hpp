/**
 * deltaEGO public API.
 *
 * Ayin and Carmen (defined in deltaEGO.cpp) hold all the actual engine
 * internals — Int16Tensor/AVX-512 search, emotionPhysics, VAD math for
 * Ayin; the future OpenJEV text->VAD pipeline for Carmen. They're only
 * forward-declared here and held by pointer so that consumers of this
 * header do not need to see or link against json/yaml-cpp/AVX intrinsics
 * directly — deltaEGO owns them directly, there is no intermediate Impl
 * class in between.
 */
#pragma once

#include <memory>
#include <string>

// sephirothic_tree/process_stimulus return nlohmann::json by value below
// — same rule as boost::asio::awaitable<T> a few lines down: a return
// type needs the complete type, not just a forward declaration, so this
// has to be a real include rather than relying on whatever some other
// header a consumer happens to include first pulls in ahead of this one.
#include "Third_Party/json.hpp"

// io_context is still only forward-declared — a reference parameter
// doesn't need the complete type. boost::asio::awaitable<T>, though, IS
// included for real below: sephirothic_tree returns one, and a return
// type (unlike a reference parameter) needs its complete definition for
// the compiler to know its size/layout. That's the actual cost of
// sephirothic_tree's coroutine-based public API — accepted here as
// unavoidable, not an oversight.
namespace boost {
namespace asio {
class io_context;
} // namespace asio
} // namespace boost
#include <boost/asio/awaitable.hpp>

namespace deltaEGO {

  namespace structs
  {
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
  }

  namespace func
  {
    float lerp(float target, float current, float resistance);
    double get_distance(const structs::VAD_Point& a, const structs::VAD_Point& b);
    double sigmoid(double x);
  }

  namespace Ayin
  {
    class Ayin;
    class Angela;
    class Roland;
  }
  namespace Carmen
  {
    class Carmen;
  }

  class deltaEGO {
  public:
    // config_path: path to the YAML file with OCEAN/physics weights.
    // def_V/def_A/def_D/def_radius: default (resting) VAD state.
    // ioc: THE server's single io_context (see main.cpp) — forwarded to
    // Carmen, which needs it for its network clients (OpenJEV, the 5090
    // fallback). Not owned; must outlive this deltaEGO instance.
    // host_jev/port_jev/host_5090/port_5090: where those two backends
    // live — see config::kLlama5090Host/Port for the 5090 side.
    deltaEGO(boost::asio::io_context &ioc,
              const std::string &config_path,
              float def_V, float def_A, float def_D, float def_radius,
              std::string host_jev, std::string port_jev,
              std::string host_5090, std::string port_5090);
    ~deltaEGO();

    deltaEGO(const deltaEGO &) = delete;
    deltaEGO &operator=(const deltaEGO &) = delete;
    deltaEGO(deltaEGO &&) noexcept;
    deltaEGO &operator=(deltaEGO &&) noexcept;

    // Same text -> VAD -> emotion-analysis pipeline as process_stimulus,
    // but starting from raw text instead of an already-known (v,a,d) —
    // calls Carmen::whisper_from_Carmen first to derive the stimulus,
    // then feeds that into the same physics/search path process_stimulus
    // uses. Defaults (see whisper_from_Carmen) mean the 5090 machine is
    // the basis unless you explicitly opt into OpenJEV with use_jev=true.
    // Returns an awaitable, not a plain std::string, because
    // whisper_from_Carmen is itself a coroutine (it calls out over the
    // network) — co_await only works inside another coroutine, so this
    // one has to be one too. Callers must co_await this, the same way
    // Orchestrator co_awaits EmbeddingClient::Embed.
    boost::asio::awaitable<nlohmann::json>
    sephirothic_tree(std::string context, bool use_jev = false,
                      bool fallback_to_5090 = false, bool use_5090 = true,
                      bool fallback_to_jev = false);

    // Loads the VAD term database (see VAD_DB/VAD.json) used for nearest
    // neighbor emotion-term lookup. Returns false on parse/IO failure.
    bool load_vad_db(const std::string &json_path);

    // Feeds one VAD stimulus into the physics engine and returns the
    // resulting state + analysis as a JSON string.
    nlohmann::json process_stimulus(float v, float a, float d);

    // Re-reads the YAML config passed to the constructor.
    bool reload_config();

    // Forwards to Carmen's Check5090Health() — exposed here since Carmen
    // is only forward-declared/opaque outside this Pimpl boundary, same
    // reason sephirothic_tree has to be a member here instead of callers
    // reaching into carmen_ directly.
    nlohmann::json Check5090Health() const;

  private:
    std::unique_ptr<Ayin::Ayin> ayin_;
    std::unique_ptr<Carmen::Carmen> carmen_; // unused until OpenJEV integration lands
  };

} // namespace deltaEGO
