#include "common/run_metadata.h"

#include <omp.h>

#include <array>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <memory>
#include <sstream>

#include "common/build_info.h"

namespace fse {
namespace {

std::string RunCommand(const std::string& cmd) {
  std::array<char, 256> buf{};
  std::string out;
  std::unique_ptr<FILE, int (*)(FILE*)> pipe(popen(cmd.c_str(), "r"), pclose);
  if (!pipe) {
    return out;
  }
  while (fgets(buf.data(), static_cast<int>(buf.size()), pipe.get()) !=
         nullptr) {
    out += buf.data();
  }
  while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) {
    out.pop_back();
  }
  return out;
}

std::string FirstLineStartingWith(const std::string& path,
                                  const std::string& key) {
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line)) {
    if (line.starts_with(key)) {
      return line;
    }
  }
  return "";
}

}  // namespace

std::string BuildInfoJson() {
  const BuildInfo b = GetBuildInfo();
  return JsonObject()
      .Str("compiler", b.compiler_id + " " + b.compiler_version)
      .Str("build_type", b.build_type)
      .Str("cxx_flags", b.cxx_flags)
      .Int("cxx_standard", b.cxx_standard)
      .Int("openmp_version", b.openmp_version)
      .Str("hnswlib_commit", b.hnswlib_commit)
      .Bool("with_acorn", b.with_acorn)
      .Str("acorn_commit", b.acorn_commit)
      .Str("acorn_opt_level", b.acorn_opt_level)
      .Render();
}

std::string HardwareJson() {
  std::string model = FirstLineStartingWith("/proc/cpuinfo", "model name");
  const auto colon = model.find(':');
  model = colon == std::string::npos ? model : model.substr(colon + 2);
  return JsonObject()
      .Str("cpu_model", model)
      .Int("omp_num_procs", omp_get_num_procs())
      .Str("mem_total", FirstLineStartingWith("/proc/meminfo", "MemTotal"))
      .Str("loadavg_at_start", FirstLineStartingWith("/proc/loadavg", ""))
      .Render();
}

std::uint64_t Fnv1a64(const std::string& s) {
  std::uint64_t h = 1469598103934665603ULL;
  for (const unsigned char c : s) {
    h ^= c;
    h *= 1099511628211ULL;
  }
  return h;
}

std::uint64_t Fnv1a64Bytes(const void* data, std::size_t size,
                           std::uint64_t seed) {
  std::uint64_t h = seed;
  const auto* p = static_cast<const unsigned char*>(data);
  for (std::size_t i = 0; i < size; ++i) {
    h ^= p[i];
    h *= 1099511628211ULL;
  }
  return h;
}

std::string Hex64(std::uint64_t v) {
  std::array<char, 17> buf{};
  std::snprintf(buf.data(), buf.size(), "%016llx",
                static_cast<unsigned long long>(v));
  return buf.data();
}

std::string MakeExperimentId(const std::string& config_text) {
  const std::time_t now =
      std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  std::tm utc{};
  gmtime_r(&now, &utc);
  std::array<char, 32> ts{};
  std::strftime(ts.data(), ts.size(), "%Y%m%dT%H%M%SZ", &utc);
  return Hex64(Fnv1a64(config_text)).substr(0, 12) + "_" + ts.data();
}

GitInfo GetGitInfo(const std::string& repo_dir) {
  GitInfo info;
  const std::string git = "git -C '" + repo_dir + "' ";
  info.commit = RunCommand(git + "rev-parse --verify -q HEAD 2>/dev/null");
  if (info.commit.empty()) {
    info.commit = "none";
  }
  info.dirty = !RunCommand(git + "status --porcelain 2>/dev/null").empty();
  return info;
}

std::string SourceFingerprint(const std::string& repo_dir) {
  const std::string fp = RunCommand(
      "cd '" + repo_dir +
      "' && git ls-files -co --exclude-standard -- cpp cmake configs python "
      "CMakeLists.txt 2>/dev/null | LC_ALL=C sort | xargs -r sha256sum "
      "2>/dev/null | sha256sum | cut -c1-64");
  return fp.empty() ? "unknown" : fp;
}

std::string JsonEscape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 2);
  for (const char c : s) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          std::array<char, 8> buf{};
          std::snprintf(buf.data(), buf.size(), "\\u%04x", c);
          out += buf.data();
        } else {
          out += c;
        }
    }
  }
  return out;
}

JsonObject& JsonObject::Str(const std::string& key, const std::string& value) {
  fields_.emplace_back(key, "\"" + JsonEscape(value) + "\"");
  return *this;
}

JsonObject& JsonObject::Num(const std::string& key, double value) {
  std::ostringstream os;
  os.precision(17);
  os << value;
  fields_.emplace_back(key, os.str());
  return *this;
}

JsonObject& JsonObject::Int(const std::string& key, std::int64_t value) {
  fields_.emplace_back(key, std::to_string(value));
  return *this;
}

JsonObject& JsonObject::Bool(const std::string& key, bool value) {
  fields_.emplace_back(key, value ? "true" : "false");
  return *this;
}

JsonObject& JsonObject::Raw(const std::string& key, const std::string& json) {
  fields_.emplace_back(key, json);
  return *this;
}

std::string JsonObject::Render() const {
  std::string out = "{";
  for (std::size_t i = 0; i < fields_.size(); ++i) {
    out += (i == 0 ? "\"" : ", \"") + JsonEscape(fields_[i].first) +
           "\": " + fields_[i].second;
  }
  return out + "}";
}

}  // namespace fse
