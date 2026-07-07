#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace geo_patch {

enum Bucket : int16_t {
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

struct RegionArgs {
  double max_color_distance = 150.0;
  double max_height_delta = 0.30;
  double max_normal_angle = 120.0;
  double max_plane_residual = 0.26;
  double stable_surface_ratio = 0.72;
  double stable_plane_factor = 2.4;
  double stable_height_factor = 2.4;
  double prototype_distance_scale = 0.54;
  double min_surface_membership_score = 0.50;
  double min_surface_bridge_score = 0.56;
  bool enable_surface_multimodal_bridge = true;
  double surface_bridge_texture_score = 0.62;
  double surface_bridge_shape_score = 0.24;
  double surface_bridge_prototype_score = 0.48;
  double min_object_membership_score = 0.48;
  double min_rough_membership_score = 0.44;
  double object_color_factor = 1.85;
  double object_texture_delta = 64.0;
  double object_roughness_delta = 0.34;
  double object_texture_weight = 0.30;
  double object_shape_weight = 0.27;
  double object_prototype_weight = 0.12;
  double object_height_weight = 0.12;
  double object_bucket_weight = 0.12;
  double object_normal_weight = 0.06;
  double object_plane_weight = 0.10;
  double rough_texture_weight = 0.36;
  double rough_shape_weight = 0.29;
  double rough_prototype_weight = 0.18;
  double rough_height_weight = 0.04;
  double rough_bucket_weight = 0.14;
  double rough_normal_weight = 0.03;
  double rough_plane_weight = 0.03;
};

struct Signature {
  std::array<double, 9> v{};
};

struct MembershipScores {
  double color = 0.0;
  double color_texture = 0.0;
  double roughness = 0.0;
  double planarity = 0.0;
  double linearity = 0.0;
  double height_range = 0.0;
  double height = 0.0;
  double bucket = 0.0;
  double normal = 0.0;
  double plane = 0.0;
  double prototype = 0.0;
  double chart = 0.0;
  double chart_plane = 0.0;
  double chart_normal = 0.0;
  double chart_height = 0.0;
};

struct MembershipResult {
  bool ok = false;
  double score = 0.0;
  std::string reason;
  MembershipScores scores;
};

inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec3 operator*(double s, Vec3 v) { return {s * v.x, s * v.y, s * v.z}; }
inline double dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline double norm(Vec3 v) { return std::sqrt(dot(v, v)); }
inline double clamp01(double v) { return std::max(0.0, std::min(1.0, v)); }

inline Vec3 normalize(Vec3 v) {
  const double n = norm(v);
  if (n < 1e-9) {
    return {0.0, 0.0, 1.0};
  }
  return {(v.x / n), (v.y / n), (v.z / n)};
}

inline double normal_angle_deg(Vec3 a, Vec3 b) {
  const double an = norm(a);
  const double bn = norm(b);
  if (an < 1e-9 || bn < 1e-9) {
    return 0.0;
  }
  const double c = std::abs(dot(a, b) / (an * bn));
  return std::acos(std::max(-1.0, std::min(1.0, c))) * 180.0 / M_PI;
}

inline bool is_stable_surface(Bucket b) {
  return b == kHorizontal || b == kVertical;
}

inline bool is_fine_object(Bucket b) {
  return b == kRoughMixed || b == kThinLinear || b == kUnknown;
}

inline double bucket_score(Bucket a, Bucket b) {
  if (a == b) {
    return 1.0;
  }
  if (a == kUnknown || b == kUnknown) {
    return 0.72;
  }
  const bool rough_horizontal = (a == kRoughMixed && b == kHorizontal) || (b == kRoughMixed && a == kHorizontal);
  const bool rough_vertical = (a == kRoughMixed && b == kVertical) || (b == kRoughMixed && a == kVertical);
  if (rough_horizontal || rough_vertical) {
    return 0.55;
  }
  return 0.15;
}

inline Signature local_signature(const std::vector<Voxel>& voxels, int index) {
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

inline double signature_distance(const Signature& a, const Signature& b) {
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

inline ChartMetrics frontier_chart_metrics(const std::vector<Voxel>& voxels, int source_index, int index, double distance_scale) {
  const Signature sig = local_signature(voxels, index);
  const Signature source_sig = local_signature(voxels, source_index);
  const auto& p = voxels.at(index);
  const auto& source = voxels.at(source_index);
  const double sig_distance = signature_distance(sig, source_sig);
  ChartMetrics out;
  out.score = clamp01(1.0 - sig_distance / std::max(distance_scale, 1e-6));
  out.plane_residual = std::abs(dot(p.xyz - source.xyz, normalize(source.normal)));
  out.normal_angle = normal_angle_deg(source.normal, p.normal);
  out.dz = std::abs(p.xyz.z - source.xyz.z);
  out.combined = sig_distance;
  out.combined += 0.28 * std::min(out.plane_residual / kPrototypeChartUpdatePlaneDistance, 2.0);
  out.combined += 0.12 * std::min(out.normal_angle / std::max(kPrototypeChartUpdateNormalAngle, 1e-6), 2.0);
  return out;
}

class PatchModel {
 public:
  PatchModel(int /*seed_index*/, Bucket seed_bucket) : seed_bucket_(seed_bucket) {}

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

  int count() const { return count_; }
  double mean_score() const { return accepted_score_sum_ / std::max(count_, 1); }
  Vec3 centroid() const { return (1.0 / std::max(count_, 1)) * xyz_sum_; }
  Vec3 color() const { return (1.0 / std::max(count_, 1)) * rgb_sum_; }
  Vec3 normal() const { return normalize(normal_sum_); }
  double mean_roughness() const { return roughness_sum_ / std::max(count_, 1); }
  double mean_planarity() const { return planarity_sum_ / std::max(count_, 1); }
  double mean_linearity() const { return linearity_sum_ / std::max(count_, 1); }
  double mean_local_color_std() const { return local_color_std_sum_ / std::max(count_, 1); }
  double mean_height_range() const { return height_range_sum_ / std::max(count_, 1); }

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

  double prototype_score(const std::vector<Voxel>& voxels, int index, double distance_scale) const {
    if (prototypes_.empty()) {
      return 0.0;
    }
    const Signature sig = local_signature(voxels, index);
    double best = std::numeric_limits<double>::infinity();
    for (const auto& proto : prototypes_) {
      best = std::min(best, signature_distance(sig, proto));
    }
    return clamp01(1.0 - best / std::max(distance_scale, 1e-6));
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
    prototype_counts_.push_back(std::max(count_, 1));
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
      if (is_stable_surface(seed_bucket_)) {
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
    const bool same_chart = nearest_distance <= kPrototypeUpdateDistance &&
                            (!is_stable_surface(seed_bucket_) ||
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

inline MembershipResult membership_score(
    const std::vector<Voxel>& voxels,
    const PatchModel& model,
    int index,
    const RegionArgs& args,
    int source_index = -1) {
  const auto& p = voxels.at(index);
  const Bucket bucket = p.bucket;
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
  const double prototype = model.prototype_score(voxels, index, args.prototype_distance_scale);

  MembershipScores scores;
  scores.color = clamp01(1.0 - rgb_dist / std::max(args.max_color_distance, 1e-6));
  scores.color_texture = clamp01(1.0 - color_std_delta / 85.0);
  scores.roughness = clamp01(1.0 - rough_delta / 0.24);
  scores.planarity = clamp01(1.0 - planarity_delta / 0.50);
  scores.linearity = clamp01(1.0 - linearity_delta / 0.50);
  scores.height_range = clamp01(1.0 - height_range_delta / 0.28);
  scores.height = clamp01(1.0 - dz / std::max(args.max_height_delta * 2.8, 1e-6));
  scores.bucket = bucket_score(dominant, bucket);
  scores.normal = clamp01(1.0 - angle / std::max(args.max_normal_angle * 1.8, 1e-6));
  scores.plane = clamp01(1.0 - plane_residual / std::max(args.max_plane_residual * 2.5, 1e-6));
  scores.prototype = prototype;
  scores.chart = prototype;
  scores.chart_plane = scores.plane;
  scores.chart_normal = scores.normal;
  scores.chart_height = scores.height;

  const double shape_score = 0.38 * scores.roughness + 0.24 * scores.linearity +
                             0.24 * scores.planarity + 0.14 * scores.height_range;
  const double texture_score = 0.62 * scores.color + 0.38 * scores.color_texture;

  double total = 0.0;
  double threshold = 0.0;
  if (is_stable_surface(dominant) && model.dominant_ratio() >= args.stable_surface_ratio) {
    const bool surface_bridge = args.enable_surface_multimodal_bridge &&
                                is_stable_surface(dominant) &&
                                (bucket == kRoughMixed || bucket == kUnknown || bucket == kThinLinear) &&
                                texture_score >= args.surface_bridge_texture_score &&
                                (shape_score >= args.surface_bridge_shape_score ||
                                 scores.prototype >= args.surface_bridge_prototype_score);
    if (bucket != dominant && bucket != kUnknown && !surface_bridge) {
      return {false, 0.0, "stable_bucket_mismatch", scores};
    }
    if (rgb_dist > args.max_color_distance * 1.9 && color_std_delta > 55.0) {
      return {false, 0.0, "stable_color_texture_jump", scores};
    }
    const double normal_limit = std::min(args.max_normal_angle * 1.7, 88.0);
    const bool global_normal_fail = angle > normal_limit;
    const bool global_plane_fail = plane_residual > args.max_plane_residual * args.stable_plane_factor;
    const bool global_height_fail = dominant == kHorizontal && dz > args.max_height_delta * args.stable_height_factor;
    const bool needs_chart = (global_normal_fail || global_plane_fail || global_height_fail) && !surface_bridge;
    if (needs_chart) {
      ChartMetrics chart = model.chart_metrics(voxels, index, args.prototype_distance_scale);
      if (source_index >= 0) {
        const Bucket source_bucket = voxels.at(source_index).bucket;
        if (source_bucket == dominant || source_bucket == kUnknown) {
          const ChartMetrics frontier = frontier_chart_metrics(voxels, source_index, index, args.prototype_distance_scale);
          if (frontier.combined < chart.combined) {
            chart = frontier;
          }
        }
      }
      scores.chart = chart.score;
      scores.chart_plane = clamp01(1.0 - chart.plane_residual / std::max(args.max_plane_residual * 2.5, 1e-6));
      scores.chart_normal = clamp01(1.0 - chart.normal_angle / std::max(args.max_normal_angle * 1.8, 1e-6));
      scores.chart_height = clamp01(1.0 - chart.dz / std::max(args.max_height_delta * 2.8, 1e-6));
      if (chart.normal_angle > normal_limit) {
        return {false, 0.0, "stable_normal_jump", scores};
      }
      if (chart.plane_residual > args.max_plane_residual * args.stable_plane_factor) {
        return {false, 0.0, "stable_plane_residual", scores};
      }
      if (dominant == kHorizontal && chart.dz > args.max_height_delta * args.stable_height_factor) {
        return {false, 0.0, "stable_height_jump", scores};
      }
    }
    if (surface_bridge) {
      total = 0.52 * texture_score + 0.18 * shape_score + 0.12 * scores.chart +
              0.10 * scores.bucket + 0.08 * scores.chart_height;
      threshold = args.min_surface_bridge_score;
    } else {
      total = 0.26 * scores.chart_plane + 0.22 * scores.chart_normal + 0.17 * texture_score +
              0.13 * scores.chart + 0.12 * scores.bucket + 0.07 * shape_score + 0.03 * scores.chart_height;
      threshold = args.min_surface_membership_score;
    }
  } else {
    if (!is_fine_object(bucket) && is_fine_object(dominant)) {
      return {false, 0.0, "fine_to_stable_bucket_block", scores};
    }
    if (rgb_dist > args.max_color_distance * args.object_color_factor && color_std_delta > args.object_texture_delta) {
      return {false, 0.0, "object_color_texture_jump", scores};
    }
    if (rough_delta > args.object_roughness_delta && color_std_delta > args.object_texture_delta) {
      return {false, 0.0, "object_shape_texture_jump", scores};
    }
    if (dominant == kRoughMixed) {
      const double denom = std::max(
          args.rough_texture_weight + args.rough_shape_weight + args.rough_prototype_weight +
              args.rough_height_weight + args.rough_bucket_weight + args.rough_normal_weight +
              args.rough_plane_weight,
          1e-6);
      total = (args.rough_texture_weight * texture_score + args.rough_shape_weight * shape_score +
               args.rough_prototype_weight * scores.prototype + args.rough_height_weight * scores.height +
               args.rough_bucket_weight * scores.bucket + args.rough_normal_weight * scores.normal +
               args.rough_plane_weight * scores.plane) /
              denom;
      threshold = args.min_rough_membership_score;
    } else {
      const double denom = std::max(
          args.object_texture_weight + args.object_shape_weight + args.object_prototype_weight +
              args.object_height_weight + args.object_bucket_weight + args.object_normal_weight +
              args.object_plane_weight,
          1e-6);
      total = (args.object_texture_weight * texture_score + args.object_shape_weight * shape_score +
               args.object_prototype_weight * scores.prototype + args.object_height_weight * scores.height +
               args.object_bucket_weight * scores.bucket + args.object_normal_weight * scores.normal +
               args.object_plane_weight * scores.plane) /
              denom;
      threshold = dominant == kRoughMixed ? args.min_rough_membership_score : args.min_object_membership_score;
    }
  }

  if (total < threshold) {
    return {false, total, "membership_score_low", scores};
  }
  return {true, total, "accepted", scores};
}

inline std::vector<int> seed_order(const std::vector<Voxel>& voxels) {
  std::vector<int> order(voxels.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
    const bool a_stable = is_stable_surface(voxels[a].bucket);
    const bool b_stable = is_stable_surface(voxels[b].bucket);
    if (a_stable != b_stable) {
      return a_stable > b_stable;
    }
    if (voxels[a].planarity != voxels[b].planarity) {
      return voxels[a].planarity > voxels[b].planarity;
    }
    return voxels[a].roughness < voxels[b].roughness;
  });
  return order;
}

inline std::vector<int32_t> grow_region_model(
    const std::vector<Voxel>& voxels,
    const std::vector<std::vector<int>>& adjacency,
    const RegionArgs& args) {
  if (adjacency.size() != voxels.size()) {
    throw std::runtime_error("adjacency size does not match voxel count");
  }
  std::vector<int32_t> labels(voxels.size(), -1);
  int32_t next_patch_id = 1;

  for (const int seed : seed_order(voxels)) {
    if (labels[seed] != -1) {
      continue;
    }
    PatchModel model(seed, voxels[seed].bucket);
    model.add(voxels, seed, 1.0);
    labels[seed] = next_patch_id;
    std::deque<int> queue;
    queue.push_back(seed);
    while (!queue.empty()) {
      const int current = queue.front();
      queue.pop_front();
      for (const int nbr : adjacency[current]) {
        if (labels[nbr] != -1) {
          continue;
        }
        const auto result = membership_score(voxels, model, nbr, args, current);
        if (!result.ok) {
          continue;
        }
        labels[nbr] = next_patch_id;
        model.add(voxels, nbr, result.score);
        queue.push_back(nbr);
      }
    }
    ++next_patch_id;
  }
  return labels;
}

}  // namespace geo_patch
