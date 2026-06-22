// C++ region-growing backend for build_geo_patch_region_model.py.
//
// Python remains responsible for PLY IO, voxelization, local feature extraction,
// and edge construction.  This binary only consumes voxel features + graph
// edges and returns one patch label per voxel.

#include "geo_patch_region_model_core.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr const char* kInputMagic = "GPRGv1\n";
constexpr const char* kOutputMagic = "GPRGlabels1\n";

struct Cli {
  std::string input;
  std::string output;
  geo_patch::RegionArgs args;
};

void usage() {
  std::cerr << "usage: geo_patch_region_grower --input in.bin --output labels.bin [threshold flags]\n";
}

double parse_double(const char* value, const std::string& flag) {
  try {
    size_t parsed = 0;
    const double out = std::stod(value, &parsed);
    if (parsed != std::strlen(value)) {
      throw std::invalid_argument("trailing characters");
    }
    return out;
  } catch (const std::exception& e) {
    throw std::runtime_error("invalid numeric value for " + flag + ": " + value);
  }
}

Cli parse_cli(int argc, char** argv) {
  Cli cli;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    auto need_value = [&]() -> const char* {
      if (i + 1 >= argc) {
        throw std::runtime_error("missing value for " + flag);
      }
      return argv[++i];
    };
    if (flag == "--input") {
      cli.input = need_value();
    } else if (flag == "--output") {
      cli.output = need_value();
    } else if (flag == "--max-color-distance") {
      cli.args.max_color_distance = parse_double(need_value(), flag);
    } else if (flag == "--max-height-delta") {
      cli.args.max_height_delta = parse_double(need_value(), flag);
    } else if (flag == "--max-normal-angle") {
      cli.args.max_normal_angle = parse_double(need_value(), flag);
    } else if (flag == "--max-plane-residual") {
      cli.args.max_plane_residual = parse_double(need_value(), flag);
    } else if (flag == "--stable-surface-ratio") {
      cli.args.stable_surface_ratio = parse_double(need_value(), flag);
    } else if (flag == "--stable-plane-factor") {
      cli.args.stable_plane_factor = parse_double(need_value(), flag);
    } else if (flag == "--stable-height-factor") {
      cli.args.stable_height_factor = parse_double(need_value(), flag);
    } else if (flag == "--prototype-distance-scale") {
      cli.args.prototype_distance_scale = parse_double(need_value(), flag);
    } else if (flag == "--min-surface-membership-score") {
      cli.args.min_surface_membership_score = parse_double(need_value(), flag);
    } else if (flag == "--min-surface-bridge-score") {
      cli.args.min_surface_bridge_score = parse_double(need_value(), flag);
    } else if (flag == "--enable-surface-multimodal-bridge") {
      cli.args.enable_surface_multimodal_bridge = true;
    } else if (flag == "--disable-surface-multimodal-bridge") {
      cli.args.enable_surface_multimodal_bridge = false;
    } else if (flag == "--surface-bridge-texture-score") {
      cli.args.surface_bridge_texture_score = parse_double(need_value(), flag);
    } else if (flag == "--surface-bridge-shape-score") {
      cli.args.surface_bridge_shape_score = parse_double(need_value(), flag);
    } else if (flag == "--surface-bridge-prototype-score") {
      cli.args.surface_bridge_prototype_score = parse_double(need_value(), flag);
    } else if (flag == "--min-object-membership-score") {
      cli.args.min_object_membership_score = parse_double(need_value(), flag);
    } else if (flag == "--min-rough-membership-score") {
      cli.args.min_rough_membership_score = parse_double(need_value(), flag);
    } else if (flag == "--object-color-factor") {
      cli.args.object_color_factor = parse_double(need_value(), flag);
    } else if (flag == "--object-texture-delta") {
      cli.args.object_texture_delta = parse_double(need_value(), flag);
    } else if (flag == "--object-roughness-delta") {
      cli.args.object_roughness_delta = parse_double(need_value(), flag);
    } else if (flag == "--object-texture-weight") {
      cli.args.object_texture_weight = parse_double(need_value(), flag);
    } else if (flag == "--object-shape-weight") {
      cli.args.object_shape_weight = parse_double(need_value(), flag);
    } else if (flag == "--object-prototype-weight") {
      cli.args.object_prototype_weight = parse_double(need_value(), flag);
    } else if (flag == "--object-height-weight") {
      cli.args.object_height_weight = parse_double(need_value(), flag);
    } else if (flag == "--object-bucket-weight") {
      cli.args.object_bucket_weight = parse_double(need_value(), flag);
    } else if (flag == "--object-normal-weight") {
      cli.args.object_normal_weight = parse_double(need_value(), flag);
    } else if (flag == "--object-plane-weight") {
      cli.args.object_plane_weight = parse_double(need_value(), flag);
    } else if (flag == "--rough-texture-weight") {
      cli.args.rough_texture_weight = parse_double(need_value(), flag);
    } else if (flag == "--rough-shape-weight") {
      cli.args.rough_shape_weight = parse_double(need_value(), flag);
    } else if (flag == "--rough-prototype-weight") {
      cli.args.rough_prototype_weight = parse_double(need_value(), flag);
    } else if (flag == "--rough-height-weight") {
      cli.args.rough_height_weight = parse_double(need_value(), flag);
    } else if (flag == "--rough-bucket-weight") {
      cli.args.rough_bucket_weight = parse_double(need_value(), flag);
    } else if (flag == "--rough-normal-weight") {
      cli.args.rough_normal_weight = parse_double(need_value(), flag);
    } else if (flag == "--rough-plane-weight") {
      cli.args.rough_plane_weight = parse_double(need_value(), flag);
    } else if (flag == "--help" || flag == "-h") {
      usage();
      std::exit(0);
    } else {
      throw std::runtime_error("unknown flag: " + flag);
    }
  }
  if (cli.input.empty() || cli.output.empty()) {
    usage();
    throw std::runtime_error("--input and --output are required");
  }
  return cli;
}

template <typename T>
void read_exact(std::istream& in, T* data, size_t count) {
  in.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(sizeof(T) * count));
  if (!in) {
    throw std::runtime_error("unexpected EOF while reading region-grower input");
  }
}

template <typename T>
void write_exact(std::ostream& out, const T* data, size_t count) {
  out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(sizeof(T) * count));
  if (!out) {
    throw std::runtime_error("failed writing region-grower output");
  }
}

std::vector<geo_patch::Voxel> read_voxels(std::istream& in, int64_t n) {
  std::vector<float> xyz(static_cast<size_t>(n) * 3);
  std::vector<float> rgb(static_cast<size_t>(n) * 3);
  std::vector<float> normal(static_cast<size_t>(n) * 3);
  std::vector<float> roughness(n);
  std::vector<float> planarity(n);
  std::vector<float> linearity(n);
  std::vector<float> local_color_std(n);
  std::vector<float> height_range(n);
  std::vector<int16_t> buckets(n);
  read_exact(in, xyz.data(), xyz.size());
  read_exact(in, rgb.data(), rgb.size());
  read_exact(in, normal.data(), normal.size());
  read_exact(in, roughness.data(), roughness.size());
  read_exact(in, planarity.data(), planarity.size());
  read_exact(in, linearity.data(), linearity.size());
  read_exact(in, local_color_std.data(), local_color_std.size());
  read_exact(in, height_range.data(), height_range.size());
  read_exact(in, buckets.data(), buckets.size());

  std::vector<geo_patch::Voxel> voxels;
  voxels.reserve(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    const size_t j = static_cast<size_t>(i);
    geo_patch::Voxel p;
    p.xyz = {xyz[j * 3 + 0], xyz[j * 3 + 1], xyz[j * 3 + 2]};
    p.rgb = {rgb[j * 3 + 0], rgb[j * 3 + 1], rgb[j * 3 + 2]};
    p.normal = {normal[j * 3 + 0], normal[j * 3 + 1], normal[j * 3 + 2]};
    p.roughness = roughness[j];
    p.planarity = planarity[j];
    p.linearity = linearity[j];
    p.local_color_std = local_color_std[j];
    p.height_range = height_range[j];
    p.bucket = static_cast<geo_patch::Bucket>(buckets[j]);
    voxels.push_back(p);
  }
  return voxels;
}

std::vector<std::vector<int>> read_adjacency(std::istream& in, int64_t n, int64_t m) {
  std::vector<int32_t> src(m);
  std::vector<int32_t> dst(m);
  read_exact(in, src.data(), src.size());
  read_exact(in, dst.data(), dst.size());
  std::vector<std::vector<int>> adjacency(static_cast<size_t>(n));
  for (int64_t i = 0; i < m; ++i) {
    const int32_t a = src[static_cast<size_t>(i)];
    const int32_t b = dst[static_cast<size_t>(i)];
    if (a < 0 || b < 0 || a >= n || b >= n) {
      throw std::runtime_error("edge index out of bounds");
    }
    adjacency[static_cast<size_t>(a)].push_back(b);
    adjacency[static_cast<size_t>(b)].push_back(a);
  }
  return adjacency;
}

void run(const Cli& cli) {
  std::ifstream in(cli.input, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed opening input: " + cli.input);
  }
  char magic[8] = {};
  read_exact(in, magic, std::strlen(kInputMagic));
  if (std::string(magic, std::strlen(kInputMagic)) != kInputMagic) {
    throw std::runtime_error("invalid input magic");
  }
  int64_t n = 0;
  int64_t m = 0;
  read_exact(in, &n, 1);
  read_exact(in, &m, 1);
  if (n <= 0 || m < 0) {
    throw std::runtime_error("invalid n/m in region-grower input");
  }
  auto voxels = read_voxels(in, n);
  auto adjacency = read_adjacency(in, n, m);
  const auto labels = geo_patch::grow_region_model(voxels, adjacency, cli.args);

  std::ofstream out(cli.output, std::ios::binary);
  if (!out) {
    throw std::runtime_error("failed opening output: " + cli.output);
  }
  out.write(kOutputMagic, static_cast<std::streamsize>(std::strlen(kOutputMagic)));
  write_exact(out, &n, 1);
  write_exact(out, labels.data(), labels.size());
}

}  // namespace

int main(int argc, char** argv) {
  try {
    run(parse_cli(argc, argv));
  } catch (const std::exception& e) {
    std::cerr << "geo_patch_region_grower failed: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
