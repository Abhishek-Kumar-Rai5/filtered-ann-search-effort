#include "feature_extraction/live_features.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace fse {
namespace {

void RequireAtLeast(const std::vector<float>& sq_dists, std::size_t k,
                    const char* who) {
  if (k == 0 || sq_dists.size() < k) {
    throw std::invalid_argument(std::string(who) +
                                ": need 0 < k <= number of distances");
  }
}

}  // namespace

std::vector<double> ComputeCentroid(const FloatMatrix& base) {
  if (base.rows == 0) {
    throw std::invalid_argument("ComputeCentroid: empty base");
  }
  std::vector<double> c(base.dim, 0.0);
  for (std::size_t i = 0; i < base.rows; ++i) {
    const float* row = base.Row(i);
    for (std::size_t d = 0; d < base.dim; ++d) {
      c[d] += row[d];
    }
  }
  for (double& x : c) {
    x /= static_cast<double>(base.rows);
  }
  return c;
}

double CentroidDistance(const float* q, const std::vector<double>& centroid) {
  double s = 0.0;
  for (std::size_t d = 0; d < centroid.size(); ++d) {
    const double diff = static_cast<double>(q[d]) - centroid[d];
    s += diff * diff;
  }
  return std::sqrt(s);
}

double ScoreConcentration(const std::vector<float>& sq_dists, std::size_t k) {
  RequireAtLeast(sq_dists, k, "ScoreConcentration");
  const double dk = std::sqrt(static_cast<double>(sq_dists[k - 1]));
  if (dk == 0.0) {
    return 1.0;
  }
  return std::sqrt(static_cast<double>(sq_dists[0])) / dk;
}

double LidMle(const std::vector<float>& sq_dists, std::size_t k) {
  RequireAtLeast(sq_dists, k, "LidMle");
  if (k < 2) {
    throw std::invalid_argument("LidMle: need k >= 2");
  }
  const double dk = std::sqrt(static_cast<double>(sq_dists[k - 1]));
  if (dk == 0.0) {
    return 0.0;
  }
  double log_sum = 0.0;
  for (std::size_t i = 0; i + 1 < k; ++i) {
    const double di = std::sqrt(static_cast<double>(sq_dists[i]));
    if (di == 0.0) {
      return 0.0;
    }
    log_sum += std::log(di / dk);
  }
  if (log_sum == 0.0) {
    return std::numeric_limits<double>::infinity();
  }
  return -static_cast<double>(k - 1) / log_sum;
}

double PassingFraction(const std::vector<std::size_t>& labels,
                       const std::vector<char>& mask) {
  if (labels.empty()) {
    throw std::invalid_argument("PassingFraction: empty probe");
  }
  std::size_t pass = 0;
  for (const std::size_t id : labels) {
    if (id >= mask.size()) {
      throw std::out_of_range("PassingFraction: label outside mask");
    }
    pass += mask[id] != 0 ? 1 : 0;
  }
  return static_cast<double>(pass) / static_cast<double>(labels.size());
}

}  // namespace fse
