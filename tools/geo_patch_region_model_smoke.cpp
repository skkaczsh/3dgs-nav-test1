// Geometry patch region-model smoke test.
//
// This verifies the C++ core invariants used by the Python patch mainline:
// stable surfaces may own multiple local charts, and graph-local frontier
// continuity can rescue a candidate that would fail a single global plane.

#include "geo_patch_region_model_core.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using geo_patch::Bucket;
using geo_patch::PatchModel;
using geo_patch::RegionArgs;
using geo_patch::Vec3;
using geo_patch::Voxel;

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
    p.bucket = geo_patch::kHorizontal;
    voxels.push_back(p);
  }
  return voxels;
}

void test_frontier_chart_rescues_local_growth() {
  const auto voxels = make_horizontal_fixture();
  RegionArgs args;
  args.max_height_delta = 0.06;
  args.max_plane_residual = 0.03;
  args.max_normal_angle = 35.0;
  args.stable_plane_factor = 2.0;
  args.stable_height_factor = 2.0;
  args.min_surface_membership_score = 0.45;

  PatchModel model(0, geo_patch::kHorizontal);
  for (int i = 0; i < 6; ++i) {
    model.add(voxels, i);
  }
  model.force_single_old_chart(voxels, 0);

  const auto without_frontier = geo_patch::membership_score(voxels, model, 6, args);
  const auto with_frontier = geo_patch::membership_score(voxels, model, 6, args, 5);

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
    p.bucket = geo_patch::kHorizontal;
    voxels.push_back(p);
  }
  std::vector<std::vector<int>> adjacency = {{1}, {0, 2}, {1}};
  RegionArgs args;
  args.max_height_delta = 0.06;
  args.max_plane_residual = 0.03;
  args.max_normal_angle = 35.0;
  args.stable_plane_factor = 2.0;
  args.stable_height_factor = 2.0;
  args.min_surface_membership_score = 0.45;

  const auto labels = geo_patch::grow_region_model(voxels, adjacency, args);
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
