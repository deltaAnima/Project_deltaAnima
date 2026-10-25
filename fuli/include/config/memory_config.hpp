#pragma once

#include <filesystem>
#include <optional>
#include <string>

namespace config {

struct RecentTurnsSettings
{
  int count = 10;
  int max_stored = 50;
  int ttl_seconds = 7 * 24 * 60 * 60;
};

// config/memory.yaml, hot-reloaded: Get() stats the file and re-parses it
// only when its mtime changed, so editing `count` takes effect on the
// next request without a restart. A file that is missing or fails to
// parse keeps whatever was last loaded (the defaults above if nothing
// ever loaded) instead of failing the request.
//
// Not thread-safe — fine on the single-threaded io_context main() runs.
class MemoryConfig
{
public:
  explicit MemoryConfig(std::string path);

  const RecentTurnsSettings &RecentTurns();

private:
  void ReloadIfChanged();

  std::string path_;
  std::optional<std::filesystem::file_time_type> loaded_mtime_;
  RecentTurnsSettings recent_turns_;
};

} // namespace config
