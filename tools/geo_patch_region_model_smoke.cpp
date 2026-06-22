// Geometry patch region-model smoke test.
//
// This is a small C++ parity guard for the Python patch mainline.  It does not
// read production PLY files yet; it verifies the key invariant that a stable
// surface patch can own multiple local chart peaks and that graph-local
// frontier continuity can rescue a candidate that would fail a single global
// plane model.

#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

enum Bucket : int {
  kUnknown = 0,
  kHorizontal = 1,
  kVertical = 2,
  kThinLinear = 3,
  kRoughMixed = 4,
};

constexpr int kMaxPatchPrototypes = 24;
constexpr double kPrototypeUpdateDistance = 0.34;
constexpr double kPrototypeChartUpdatePlaneDistance = 0.18;
constexpr double kPrototypeChartUpdateNormalAngle = 42.0;

struct Vec3 {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
};

struct Voxel {
  Vec3 xyz;
  Vec3 rgb;
  Vec3 normal;
  double roughness = 0.0;
  double planarity = 0.0;
  double linearity = 0.0;
  double local_color_std = 0.0;
  double height_range = 0.0;
  Bucket bucket = kUnknown;
};

struct Signature {
  std::array<double, 9> v{};
};

struct Scores {
  double plane = 0.0;
  double height = 0.0;
  double chart_plane = 0.0;
  double chart_height = 0.0;
};

struct MembershipResult {
  bool ok = false;
  double score = 0.0;
  std::string reason;
  Scores scores;
};

double dot(Vec3 a, Vec3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3 operator+(Vec3 a, Vec3 b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}

Vec3 operator-(Vec3 a, Vec3 b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3 operator*(double s, Vec3 v) {
  return {s * v.x, s * v.y, s * v.z};
}

double norm(Vec3 v) {
  return std::sqrt(dot(v, v));
}

Vec3 normalize(Vec3 v) {
  const double n = norm(v);
  if (n < 1e-9) {
    return {0.0, 0.0, 1.0};
  }
  return {(v.x / n), (v.y / n), (v.z / n)};
}

double clamp01(double v) {
  if (v < 0.0) {
    return 0.0;
  }
  if (v > 1.0) {
    return 1.0;
  }
  return v;
}

double normal_angle_deg(Vec3 a, Vec3 b) {
  const double an = norm(a);
  const double bn = norm(b);
  if (an < 1e-9 || bn < 1e-9) {
    return 0.0;
  }
  const double c = std::abs(dot(a, b) / (an * bn));
  return std::acos(std::max(-1.0, std::min(1.0, c))) * 180.0 / M_PI;
}

Signature local_signature(const std::vector<Voxel>& voxels, int index) {
  const auto& p = voxels.at(index);
  Signature sig;
  sig.v = {
      p.rgb.x / 255.0,
      p.rgb.y / 255.0,
      p.rgb.z / 255.0,
      p.roughness,
      p.planarity,
      p.linearity,
      std::min(p.local_color_std / 128.0, 1.5),
      std::min(p.height_range / 0.55, 1.5),
      std::abs(p.normal.z),
  };
  return sig;
}

double signature_distance(const Signature& a, const Signature& b) {
  double color = 0.0;
  for (int i = 0; i < 3; ++i) {
    const double d = a.v[i] - b.v[i];
    color += d * d;
  }
  double shape = 0.0;
  for (int i = 3; i < 8; ++i) {
    const double d = a.v[i] - b.v[i];
    shape += d * d;
  }
  const double normal_z = std::abs(a.v[8] - b.v[8]);
  return 0.46 * std::sqrt(color) + 0.42 * std::sqrt(shape) + 0.12 * normal_z;
}

struct ChartMetrics {
  double score = 0.0;
  double plane_residual = std::numeric_limits<double>::infinity();
  double normal_angle = 180.0;
  double dz = std::numeric_limits<double>::infinity();
  double combined = std::numeric_limits<double>::infinity();
};

ChartMetrics frontier_chart_metrics(const std::vector<Voxel>& voxels, int source_index, int index, double distance_scale) {
  const Signature sig = local_signature(voxels, index);
  const Signature source_sig = local_signature(voxels, source_index);
  const auto& p = voxels.at(index);
  const auto& source = voxels.at(source_index);
  const double sig_distance = signature_distance(sig, source_sig);
  ChartMetrics out;
  out.score = clamp01(1.0 - sig_distance / std::max(distance_scale, 1e-6));
  out.plane_residual = std::abs(dot(p.xyz - source.xyz, source.normal));
  out.normal_angle = normal_angle_deg(source.normal, p.normal);
  out.dz = std::abs(p.xyz.z - source.xyz.z);
  out.combined = sig_distance;
  out.combined += 0.28 * std::min(out.plane_residual / kPrototypeChartUpdatePlaneDistance, 2.0);
  out.combined += 0.12 * std::min(out.normal_angle / std::max(kPrototypeChartUpdateNormalAngle, 1e-6), 2.0);
  return out;
}

class PatchModel {
 public:
  explicit PatchModel(Bucket seed_bucket) : seed_bucket_(seed_bucket) {}

  void add(const std::vector<Voxel>& voxels, int index, double score = 1.0) {
    const auto& p = voxels.at(index);
    Vec3 n = normalize(p.normal);
    if (count_ > 0 && dot(n, normal()) < 0.0) {
      n = -1.0 * n;
    }
    ++count_;
    xyz_sum_ = xyz_sum_ + p.xyz;
    rgb_sum_ = rgb_sum_ + p.rgb;
    normal_sum_ = normal_sum_ + n;
    roughness_sum_ += p.roughness;
    planarity_sum_ += p.planarity;
    linearity_sum_ += p.linearity;
    local_color_std_sum_ += p.local_color_std;
    height_range_sum_ += p.height_range;
    bucket_counts_[p.bucket] += 1;
    accepted_score_sum_ += score;
    update_prototypes(voxels, index);
  }

  int count() const {
    return count_;
  }

  Vec3 centroid() const {
    return (1.0 / std::max(count_, 1)) * xyz_sum_;
  }

  Vec3 color() const {
    return (1.0 / std::max(count_, 1)) * rgb_sum_;
  }

  Vec3 normal() const {
    return normalize(normal_sum_);
  }

  Bucket dominant_bucket() const {
    Bucket best = seed_bucket_;
    int best_count = -1;
    for (const auto& [bucket, n] : bucket_counts_) {
      if (n > best_count) {
        best = bucket;
        best_count = n;
      }
    }
    return best;
  }

  double dominant_ratio() const {
    int best_count = 0;
    for (const auto& [_, n] : bucket_counts_) {
      best_count = std::max(best_count, n);
    }
    return static_cast<double>(best_count) / std::max(count_, 1);
  }

  double mean_roughness() const {
    return roughness_sum_ / std::max(count_, 1);
  }

  double mean_planarity() const {
    return planarity_sum_ / std::max(count_, 1);
  }

  double mean_linearity() const {
    return linearity_sum_ / std::max(count_, 1);
  }

  double mean_local_color_std() const {
    return local_color_std_sum_ / std::max(count_, 1);
  }

  double mean_height_range() const {
    return height_range_sum_ / std::max(count_, 1);
  }

  ChartMetrics chart_metrics(const std::vector<Voxel>& voxels, int index, double distance_scale) const {
    ChartMetrics best;
    if (prototypes_.empty()) {
      return best;
    }
    const Signature sig = local_signature(voxels, index);
    const auto& p = voxels.at(index);
    for (size_t i = 0; i < prototypes_.size(); ++i) {
      const double sig_distance = signature_distance(sig, prototypes_[i]);
      ChartMetrics chart;
      chart.score = clamp01(1.0 - sig_distance / std::max(distance_scale, 1e-6));
      chart.plane_residual = std::abs(dot(p.xyz - prototype_xyz_[i], prototype_normals_[i]));
      chart.normal_angle = normal_angle_deg(prototype_normals_[i], p.normal);
      chart.dz = std::abs(p.xyz.z - prototype_xyz_[i].z);
      chart.combined = sig_distance;
      chart.combined += 0.28 * std::min(chart.plane_residual / kPrototypeChartUpdatePlaneDistance, 2.0);
      chart.combined += 0.12 * std::min(chart.normal_angle / std::max(kPrototypeChartUpdateNormalAngle, 1e-6), 2.0);
      if (chart.combined < best.combined) {
        best = chart;
      }
    }
    return best;
  }

  void force_single_old_chart(const std::vector<Voxel>& voxels, int index) {
    prototypes_.clear();
    prototype_counts_.clear();
    prototype_xyz_.clear();
    prototype_normals_.clear();
    prototypes_.push_back(local_signature(voxels, index));
    prototype_counts_.push_back(count_);
    prototype_xyz_.push_back(voxels.at(index).xyz);
    prototype_normals_.push_back(normalize(voxels.at(index).normal));
  }

 private:
  void update_prototypes(const std::vector<Voxel>& voxels, int index) {
    const Signature sig = local_signature(voxels, index);
    const auto& p = voxels.at(index);
    const Vec3 n = normalize(p.normal);
    if (prototypes_.empty()) {
      prototypes_.push_back(sig);
      prototype_counts_.push_back(1);
      prototype_xyz_.push_back(p.xyz);
      prototype_normals_.push_back(n);
      return;
    }

    int nearest = 0;
    double nearest_distance = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < prototypes_.size(); ++i) {
      double chart_penalty = 0.0;
      if (seed_bucket_ == kHorizontal || seed_bucket_ == kVertical) {
        const double plane = std::abs(dot(p.xyz - prototype_xyz_[i], prototype_normals_[i]));
        const double angle = normal_angle_deg(prototype_normals_[i], n);
        chart_penalty += 0.35 * std::min(plane / kPrototypeChartUpdatePlaneDistance, 2.0);
        chart_penalty += 0.20 * std::min(angle / std::max(kPrototypeChartUpdateNormalAngle, 1e-6), 2.0);
      }
      const double distance = signature_distance(sig, prototypes_[i]) + chart_penalty;
      if (distance < nearest_distance) {
        nearest_distance = distance;
        nearest = static_cast<int>(i);
      }
    }

    const double nearest_plane = std::abs(dot(p.xyz - prototype_xyz_[nearest], prototype_normals_[nearest]));
    const double nearest_angle = normal_angle_deg(prototype_normals_[nearest], n);
    const bool stable_surface = seed_bucket_ == kHorizontal || seed_bucket_ == kVertical;
    const bool same_chart = nearest_distance <= kPrototypeUpdateDistance &&
                            (!stable_surface ||
                             (nearest_plane <= kPrototypeChartUpdatePlaneDistance &&
                              nearest_angle <= kPrototypeChartUpdateNormalAngle));

    if (same_chart || static_cast<int>(prototypes_.size()) >= kMaxPatchPrototypes) {
      const int old_count = prototype_counts_[nearest];
      const double alpha = 1.0 / static_cast<double>(old_count + 1);
      for (size_t j = 0; j < prototypes_[nearest].v.size(); ++j) {
        prototypes_[nearest].v[j] = (1.0 - alpha) * prototypes_[nearest].v[j] + alpha * sig.v[j];
      }
      prototype_xyz_[nearest] = (1.0 - alpha) * prototype_xyz_[nearest] + alpha * p.xyz;
      Vec3 nn = n;
      if (dot(prototype_normals_[nearest], nn) < 0.0) {
        nn = -1.0 * nn;
      }
      prototype_normals_[nearest] = normalize((1.0 - alpha) * prototype_normals_[nearest] + alpha * nn);
      prototype_counts_[nearest] = old_count + 1;
    } else {
      prototypes_.push_back(sig);
      prototype_counts_.push_back(1);
      prototype_xyz_.push_back(p.xyz);
      prototype_normals_.push_back(n);
    }
  }

  Bucket seed_bucket_;
  int count_ = 0;
  Vec3 xyz_sum_;
  Vec3 rgb_sum_;
  Vec3 normal_sum_;
  double roughness_sum_ = 0.0;
  double planarity_sum_ = 0.0;
  double linearity_sum_ = 0.0;
  double local_color_std_sum_ = 0.0;
  double height_range_sum_ = 0.0;
  double accepted_score_sum_ = 0.0;
  std::map<Bucket, int> bucket_counts_;
  std::vector<Signature> prototypes_;
  std::vector<int> prototype_counts_;
  std::vector<Vec3> prototype_xyz_;
  std::vector<Vec3> prototype_normals_;
};

MembershipResult membership_score(const std::vector<Voxel>& voxels, const PatchModel& model, int index, int source_index = -1) {
  constexpr double kMaxColorDistance = 150.0;
  constexpr double kMaxHeightDelta = 0.06;
  constexpr double kMaxNormalAngle = 35.0;
  constexpr double kMaxPlaneResidual = 0.03;
  constexpr double kStablePlaneFactor = 2.0;
  constexpr double kStableHeightFactor = 2.0;
  constexpr double kPrototypeDistanceScale = 0.54;
  constexpr double kMinSurfaceMembershipScore = 0.45;

  const auto& p = voxels.at(index);
  const Bucket dominant = model.dominant_bucket();
  const Vec3 centroid = model.centroid();
  const Vec3 patch_normal = model.normal();
  const double angle = normal_angle_deg(patch_normal, p.normal);
  const double rgb_dist = norm(model.color() - p.rgb);
  const double dz = std::abs(p.xyz.z - centroid.z);
  const double plane_residual = std::abs(dot(p.xyz - centroid, patch_normal));
  const double rough_delta = std::abs(model.mean_roughness() - p.roughness);
  const double planarity_delta = std::abs(model.mean_planarity() - p.planarity);
  const double linearity_delta = std::abs(model.mean_linearity() - p.linearity);
  const double color_std_delta = std::abs(model.mean_local_color_std() - p.local_color_std);
  const double height_range_delta = std::abs(model.mean_height_range() - p.height_range);

  Scores scores;
  scores.plane = clamp01(1.0 - plane_residual / std::max(kMaxPlaneResidual * 2.5, 1e-6));
  scores.height = clamp01(1.0 - dz / std::max(kMaxHeightDelta * 2.8, 1e-6));
  scores.chart_plane = scores.plane;
  scores.chart_height = scores.height;

  if ((dominant == kHorizontal || dominant == kVertical) && model.dominant_ratio() >= 0.72) {
    if (p.bucket != dominant && p.bucket != kUnknown) {
      return {false, 0.0, "stable_bucket_mismatch", scores};
    }
    if (rgb_dist > kMaxColorDistance * 1.9 && color_std_delta > 55.0) {
      return {false, 0.0, "stable_color_texture_jump", scores};
    }

    const double normal_limit = std::min(kMaxNormalAngle * 1.7, 88.0);
    const bool global_normal_fail = angle > normal_limit;
    const bool global_plane_fail = plane_residual > kMaxPlaneResidual * kStablePlaneFactor;
    const bool global_height_fail = dominant == kHorizontal && dz > kMaxHeightDelta * kStableHeightFactor;
    if (global_normal_fail || global_plane_fail || global_height_fail) {
      ChartMetrics chart = model.chart_metrics(voxels, index, kPrototypeDistanceScale);
      if (source_index >= 0 && (voxels.at(source_index).bucket == dominant || voxels.at(source_index).bucket == kUnknown)) {
        const ChartMetrics frontier = frontier_chart_metrics(voxels, source_index, index, kPrototypeDistanceScale);
        if (frontier.combined < chart.combined) {
          chart = frontier;
        }
      }
      scores.chart_plane = clamp01(1.0 - chart.plane_residual / std::max(kMaxPlaneResidual * 2.5, 1e-6));
      scores.chart_height = clamp01(1.0 - chart.dz / std::max(kMaxHeightDelta * 2.8, 1e-6));
      if (chart.normal_angle > normal_limit) {
        return {false, 0.0, "stable_normal_jump", scores};
      }
      if (chart.plane_residual > kMaxPlaneResidual * kStablePlaneFactor) {
        return {false, 0.0, "stable_plane_residual", scores};
      }
      if (dominant == kHorizontal && chart.dz > kMaxHeightDelta * kStableHeightFactor) {
        return {false, 0.0, "stable_height_jump", scores};
      }
    }

    const double color_score = clamp01(1.0 - rgb_dist / std::max(kMaxColorDistance, 1e-6));
    const double texture_score = 0.62 * color_score + 0.38 * clamp01(1.0 - color_std_delta / 85.0);
    const double shape_score = 0.38 * clamp01(1.0 - rough_delta / 0.24) +
                               0.24 * clamp01(1.0 - linearity_delta / 0.50) +
                               0.24 * clamp01(1.0 - planarity_delta / 0.50) +
                               0.14 * clamp01(1.0 - height_range_delta / 0.28);
    const double total = 0.26 * scores.chart_plane +
                         0.22 * clamp01(1.0 - angle / std::max(kMaxNormalAngle * 1.8, 1e-6)) +
                         0.17 * texture_score +
                         0.12 * 1.0 +
                         0.07 * shape_score +
                         0.03 * scores.chart_height;
    if (total < kMinSurfaceMembershipScore) {
      return {false, total, "membership_score_low", scores};
    }
    return {true, total, "accepted", scores};
  }

  return {true, 1.0, "accepted", scores};
}

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

std::vector<Voxel> make_horizontal_fixture() {
  std::vector<Voxel> voxels;
  const std::vector<Vec3> xyz = {
      {0.0, 0.0, 0.0},
      {0.1, 0.0, 0.0},
      {0.2, 0.0, 0.0},
      {1.0, 0.0, 0.42},
      {1.1, 0.0, 0.42},
      {1.2, 0.0, 0.42},
      {1.3, 0.0, 0.42},
  };
  for (size_t i = 0; i < xyz.size(); ++i) {
    Voxel p;
    p.xyz = xyz[i];
    p.rgb = {110.0 + static_cast<double>(i), 108.0 + static_cast<double>(i % 3), 102.0 + static_cast<double>(i % 4)};
    p.normal = {0.0, 0.0, 1.0};
    p.roughness = 0.02;
    p.planarity = 0.82;
    p.linearity = 0.05;
    p.local_color_std = 8.0;
    p.height_range = 0.02;
    p.bucket = kHorizontal;
    voxels.push_back(p);
  }
  return voxels;
}

void test_frontier_chart_rescues_local_growth() {
  const auto voxels = make_horizontal_fixture();
  PatchModel model(kHorizontal);
  for (int i = 0; i < 6; ++i) {
    model.add(voxels, i);
  }
  model.force_single_old_chart(voxels, 0);

  const auto without_frontier = membership_score(voxels, model, 6);
  const auto with_frontier = membership_score(voxels, model, 6, 5);

  require(!without_frontier.ok, "single old chart should reject the distant local plane");
  require(with_frontier.ok, "frontier chart should rescue graph-local growth");
  require(with_frontier.scores.chart_height > without_frontier.scores.chart_height,
          "frontier chart must improve local height score");
}

void test_region_model_breaks_pairwise_chain_bridge() {
  std::vector<Voxel> voxels;
  for (double z : {0.0, 0.05, 0.70}) {
    Voxel p;
    p.xyz = {z, 0.0, z};
    p.rgb = {100.0 + 10.0 * z, 100.0, 100.0};
    p.normal = {0.0, 0.0, 1.0};
    p.roughness = 0.02;
    p.planarity = 0.82;
    p.linearity = 0.05;
    p.local_color_std = 8.0;
    p.height_range = 0.02;
    p.bucket = kHorizontal;
    voxels.push_back(p);
  }
  std::vector<std::vector<int>> adjacency = {{1}, {0, 2}, {1}};
  std::vector<int> labels(voxels.size(), -1);
  int patch_id = 0;
  for (int seed = 0; seed < static_cast<int>(voxels.size()); ++seed) {
    if (labels[seed] != -1) {
      continue;
    }
    PatchModel model(kHorizontal);
    model.add(voxels, seed);
    labels[seed] = patch_id;
    std::deque<int> queue;
    queue.push_back(seed);
    while (!queue.empty()) {
      const int current = queue.front();
      queue.pop_front();
      for (int nbr : adjacency[current]) {
        if (labels[nbr] != -1) {
          continue;
        }
        const auto result = membership_score(voxels, model, nbr, current);
        if (!result.ok) {
          continue;
        }
        labels[nbr] = patch_id;
        model.add(voxels, nbr, result.score);
        queue.push_back(nbr);
      }
    }
    ++patch_id;
  }
  require(labels[0] == labels[1], "nearby horizontal voxels should merge");
  require(labels[2] != labels[0], "far height jump should not bridge through pairwise chain");
}

}  // namespace

int main() {
  try {
    test_frontier_chart_rescues_local_growth();
    test_region_model_breaks_pairwise_chain_bridge();
  } catch (const std::exception& e) {
    std::cerr << "geo_patch_region_model_smoke failed: " << e.what() << "\n";
    return 1;
  }
  std::cout << "geo_patch_region_model_smoke ok\n";
  return 0;
}
