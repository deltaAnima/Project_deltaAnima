/**
 * deltaEGO public API.
 *
 * All engine internals (Int16Tensor/AVX-512 search, emotionPhysics, VAD math)
 * are hidden behind a Pimpl so that consumers of this header do not need to
 * see or link against json/yaml-cpp/AVX intrinsics directly.
 */
#pragma once

#include <memory>
#include <string>

namespace deltaEGO {

class deltaEGO {
public:
  // config_path: path to the YAML file with OCEAN/physics weights.
  // def_V/def_A/def_D/def_radius: default (resting) VAD state.
  deltaEGO(const std::string &config_path, float def_V, float def_A,
           float def_D, float def_radius);
  ~deltaEGO();

  deltaEGO(const deltaEGO &) = delete;
  deltaEGO &operator=(const deltaEGO &) = delete;
  deltaEGO(deltaEGO &&) noexcept;
  deltaEGO &operator=(deltaEGO &&) noexcept;

  // Loads the VAD term database (see VAD_DB/VAD.json) used for nearest
  // neighbor emotion-term lookup. Returns false on parse/IO failure.
  bool load_vad_db(const std::string &json_path);

  // Feeds one VAD stimulus into the physics engine and returns the
  // resulting state + analysis as a JSON string.
  std::string process_stimulus(float v, float a, float d);

  // Re-reads the YAML config passed to the constructor.
  bool reload_config();

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace deltaEGO
