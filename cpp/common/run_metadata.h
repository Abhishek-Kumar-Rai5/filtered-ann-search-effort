#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace fse {

std::uint64_t Fnv1a64(const std::string& s);
std::string Hex64(std::uint64_t v);

std::uint64_t Fnv1a64Bytes(const void* data, std::size_t size,
                           std::uint64_t seed = 1469598103934665603ULL);

std::string MakeExperimentId(const std::string& config_text);

struct GitInfo {
  std::string commit;
  bool dirty = true;
};
GitInfo GetGitInfo(const std::string& repo_dir);

std::string SourceFingerprint(const std::string& repo_dir);

std::string JsonEscape(const std::string& s);

std::string BuildInfoJson();
std::string HardwareJson();

class JsonObject {
 public:
  JsonObject& Str(const std::string& key, const std::string& value);
  JsonObject& Num(const std::string& key, double value);
  JsonObject& Int(const std::string& key, std::int64_t value);
  JsonObject& Bool(const std::string& key, bool value);
  JsonObject& Raw(const std::string& key, const std::string& json);
  [[nodiscard]] std::string Render() const;

 private:
  std::vector<std::pair<std::string, std::string>> fields_;
};

}
