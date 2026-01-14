#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace fse {

// 64-bit FNV-1a, used for experiment ids and cache keys (not for security).
std::uint64_t Fnv1a64(const std::string& s);
std::string Hex64(std::uint64_t v);
// 64-bit FNV-1a over raw bytes (content hashes of masks, vectors, results).
std::uint64_t Fnv1a64Bytes(const void* data, std::size_t size,
                           std::uint64_t seed = 1469598103934665603ULL);

// "<hash of config text>_<UTC timestamp>", per design doc section 20.
std::string MakeExperimentId(const std::string& config_text);

struct GitInfo {
  std::string commit;  // "none" if the repository has no commits yet
  bool dirty = true;
};
GitInfo GetGitInfo(const std::string& repo_dir);

// SHA-256 over the contents of every non-ignored file in the project's
// source paths (cpp/, cmake/, configs/, python/, CMakeLists.txt), excluding
// submodules. Identifies the exact code of a run even before the repository
// has commits, or when the tree is dirty.
std::string SourceFingerprint(const std::string& repo_dir);

std::string JsonEscape(const std::string& s);

// JSON objects with this build's metadata (compiler, flags, OpenMP, pinned
// ACORN commit / opt level) and the host (CPU model, cores, memory), logged
// by every experiment driver.
std::string BuildInfoJson();
std::string HardwareJson();

// Minimal flat-ish JSON object builder: values are inserted pre-rendered, so
// nested objects are built by passing another builder's Render() output.
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

}  // namespace fse
