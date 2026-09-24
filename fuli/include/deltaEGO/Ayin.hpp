#pragma once

// Internal header — NOT part of deltaEGO's public API (see
// deltaEGO.hpp's own doc comment). This exists purely so deltaEGO.cpp
// (which owns an Ayin::Ayin via unique_ptr) and Ayin.cpp (which
// implements it) can share one definition of the class.
//
// Roland and Angela are only forward-declared here, never defined — Ayin
// holds them by unique_ptr specifically so their complete types (and
// everything they drag in: AVX-512 intrinsics, yaml-cpp) stay contained
// to Ayin.cpp alone. Nothing outside Ayin.cpp ever needs to know Roland
// or Angela exist.

#include <memory>
#include <string>

#include "deltaEGO/deltaEGO.hpp"

namespace deltaEGO {
namespace Ayin {

class Roland; // analysis — defined in Ayin.cpp
class Angela; // vector search — defined in Ayin.cpp

// Mid-level orchestrator: holds Roland + Angela, runs a stimulus through
// Roland first (physics update), feeds the resulting state to Angela
// (nearest-term search), and combines both into one result. No math or
// search logic of its own — see Ayin.cpp for where that actually lives.
class Ayin {
public:
  Ayin(const std::string &config_path, float def_V, float def_A, float def_D,
       float def_radius);
  ~Ayin();

  Ayin(const Ayin &) = delete;
  Ayin &operator=(const Ayin &) = delete;

  bool load_vad_db(const std::string &json_path);
  bool reload_config();

  // Everything deltaEGO::deltaEGO needs back from one processing pass, so
  // it can assemble the response JSON without needing to know how any of
  // it was computed.
  struct ProcessResult {
    structs::VAD_Point current_state;
    structs::AnalysisResult analysis;
    std::string emotion_term;
    float similarity;
  };

  ProcessResult Process(const structs::VAD_Point &input_stimulus);

private:
  std::unique_ptr<Roland> roland_;
  std::unique_ptr<Angela> angela_;
};

} // namespace Ayin
} // namespace deltaEGO
