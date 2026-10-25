#include "config/memory_config.hpp"

#include <algorithm>
#include <iostream>
#include <system_error>

#include <yaml-cpp/yaml.h>

namespace config {

MemoryConfig::MemoryConfig(std::string path) : path_(std::move(path))
{
  this->ReloadIfChanged();
}

const RecentTurnsSettings &MemoryConfig::RecentTurns()
{
  this->ReloadIfChanged();
  return this->recent_turns_;
}

void MemoryConfig::ReloadIfChanged()
{
  std::error_code ec;
  auto mtime = std::filesystem::last_write_time(this->path_, ec);
  if (ec)
  {
    // Only warn once per disappearance, not on every request.
    if (this->loaded_mtime_)
      std::cerr << "[WARN] " << this->path_ << " unreadable (" << ec.message()
                << ") — keeping last loaded memory config\n";
    this->loaded_mtime_.reset();
    return;
  }
  if (this->loaded_mtime_ == mtime)
    return;

  try
  {
    YAML::Node root = YAML::LoadFile(this->path_);
    RecentTurnsSettings next = this->recent_turns_;
    if (auto rt = root["RecentTurns"])
    {
      next.count = rt["count"].as<int>(next.count);
      next.max_stored = rt["max_stored"].as<int>(next.max_stored);
      next.ttl_seconds = rt["ttl_seconds"].as<int>(next.ttl_seconds);
    }
    next.max_stored = std::max(next.max_stored, 1);
    next.count = std::clamp(next.count, 0, next.max_stored);
    next.ttl_seconds = std::max(next.ttl_seconds, 1);

    this->recent_turns_ = next;
    std::cout << "[Config] loaded " << this->path_ << ": recent_turns.count="
              << next.count << " max_stored=" << next.max_stored
              << " ttl_seconds=" << next.ttl_seconds << "\n";
  }
  catch (const std::exception &e)
  {
    std::cerr << "[WARN] failed to parse " << this->path_ << ": " << e.what()
              << " — keeping last loaded memory config\n";
  }
  // Recorded even on a parse failure, so a broken file is reported once
  // rather than re-parsed on every request until it's fixed.
  this->loaded_mtime_ = mtime;
}

} // namespace config
