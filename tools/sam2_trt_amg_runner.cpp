// SAM2 TensorRT automatic-mask runner, production-compatible first cut.
//
// This runner writes the same artifact names and JSON schema as
// pure_sam_mask_generator.py. It implements the full-image AMG point grid
// path with TensorRT encoder/point-decoder engines. Crop layers are deliberately
// not enabled yet; keep Python SAM2 as production default until side-by-side
// quality comparison passes.

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <glob.h>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr int kModelSize = 1024;
constexpr int kDecoderBatch = 64;
constexpr int kMasksPerPoint = 3;
constexpr int kLowRes = 256;

class Logger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cerr << "[TRT] " << msg << "\n";
    }
  }
};

void check_cuda(cudaError_t status, const std::string& context) {
  if (status != cudaSuccess) {
    throw std::runtime_error(context + ": " + cudaGetErrorString(status));
  }
}

std::vector<char> read_file(const std::string& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    throw std::runtime_error("failed to open " + path);
  }
  const auto size = input.tellg();
  input.seekg(0, std::ios::beg);
  std::vector<char> data(static_cast<size_t>(size));
  if (!input.read(data.data(), size)) {
    throw std::runtime_error("failed to read " + path);
  }
  return data;
}

size_t dtype_size(nvinfer1::DataType dtype) {
  switch (dtype) {
    case nvinfer1::DataType::kFLOAT:
    case nvinfer1::DataType::kINT32:
      return 4;
    case nvinfer1::DataType::kHALF:
    case nvinfer1::DataType::kBF16:
      return 2;
    case nvinfer1::DataType::kINT8:
    case nvinfer1::DataType::kUINT8:
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kFP8:
      return 1;
    case nvinfer1::DataType::kINT64:
      return 8;
    default:
      throw std::runtime_error("unsupported TensorRT dtype");
  }
}

int64_t volume(const nvinfer1::Dims& dims) {
  int64_t total = 1;
  for (int i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] < 0) {
      throw std::runtime_error("dynamic tensor shape is not supported");
    }
    total *= dims.d[i];
  }
  return total;
}

struct DeviceBuffer {
  void* ptr = nullptr;
  size_t bytes = 0;
  ~DeviceBuffer() {
    if (ptr != nullptr) {
      cudaFree(ptr);
    }
  }
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept : ptr(other.ptr), bytes(other.bytes) {
    other.ptr = nullptr;
    other.bytes = 0;
  }
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      if (ptr != nullptr) {
        cudaFree(ptr);
      }
      ptr = other.ptr;
      bytes = other.bytes;
      other.ptr = nullptr;
      other.bytes = 0;
    }
    return *this;
  }
  void allocate(size_t nbytes) {
    bytes = nbytes;
    check_cuda(cudaMalloc(&ptr, bytes), "cudaMalloc");
  }
};

struct TrtEngine {
  Logger* logger = nullptr;
  std::unique_ptr<nvinfer1::IRuntime> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  std::unique_ptr<nvinfer1::IExecutionContext> context;
  std::unordered_map<std::string, DeviceBuffer> buffers;
  std::unordered_map<std::string, size_t> byte_sizes;

  TrtEngine(Logger& log, const std::string& path) : logger(&log) {
    auto data = read_file(path);
    runtime.reset(nvinfer1::createInferRuntime(log));
    if (!runtime) {
      throw std::runtime_error("createInferRuntime failed");
    }
    engine.reset(runtime->deserializeCudaEngine(data.data(), data.size()));
    if (!engine) {
      throw std::runtime_error("deserializeCudaEngine failed: " + path);
    }
    context.reset(engine->createExecutionContext());
    if (!context) {
      throw std::runtime_error("createExecutionContext failed: " + path);
    }
    for (int i = 0; i < engine->getNbIOTensors(); ++i) {
      const char* name = engine->getIOTensorName(i);
      const auto dims = engine->getTensorShape(name);
      const auto dtype = engine->getTensorDataType(name);
      const size_t bytes = static_cast<size_t>(volume(dims)) * dtype_size(dtype);
      DeviceBuffer buffer;
      buffer.allocate(bytes);
      if (!context->setTensorAddress(name, buffer.ptr)) {
        throw std::runtime_error(std::string("setTensorAddress failed: ") + name);
      }
      byte_sizes[name] = bytes;
      buffers.emplace(name, std::move(buffer));
    }
  }

  void copy_h2d(const std::string& name, const void* data, cudaStream_t stream) {
    check_cuda(cudaMemcpyAsync(buffers.at(name).ptr, data, byte_sizes.at(name),
                               cudaMemcpyHostToDevice, stream),
               "cudaMemcpyAsync H2D " + name);
  }

  void copy_d2h(const std::string& name, void* data, cudaStream_t stream) {
    check_cuda(cudaMemcpyAsync(data, buffers.at(name).ptr, byte_sizes.at(name),
                               cudaMemcpyDeviceToHost, stream),
               "cudaMemcpyAsync D2H " + name);
  }

  void run(cudaStream_t stream) {
    if (!context->enqueueV3(stream)) {
      throw std::runtime_error("TensorRT enqueueV3 failed");
    }
  }
};

struct Args {
  std::string images_glob;
  std::string output_dir;
  std::string encoder_engine = "/root/epfs/sam2_tensorrt/engines/sam2_hiera_l_image_encoder_fp16.plan";
  std::string decoder_engine = "/root/epfs/sam2_tensorrt/engines/sam2_hiera_l_point_decoder_b64_fp16.plan";
  int points_per_side = 32;
  int points_per_batch = 64;
  int min_mask_area = 500;
  float pred_iou_thresh = 0.7f;
  float stability_score_thresh = 0.92f;
  float stability_score_offset = 1.0f;
  float box_nms_thresh = 0.7f;
  float crop_nms_thresh = 0.7f;
  int crop_n_layers = 1;
  float crop_overlap_ratio = 512.0f / 1500.0f;
  bool skip_existing = true;
};

struct Candidate {
  std::vector<uint8_t> mask;
  int x0 = 0;
  int y0 = 0;
  int x1 = 0;
  int y1 = 0;
  int area = 0;
  float iou = 0.0f;
  float stability = 0.0f;
  int point_index = 0;
  int mask_index = 0;
  int crop_area = 0;
};

std::string basename(const std::string& path) {
  return fs::path(path).filename().string();
}

std::string stem(const std::string& path) {
  return fs::path(path).stem().string();
}

std::vector<std::string> expand_glob(const std::string& pattern) {
  glob_t glob_result {};
  const int rc = glob(pattern.c_str(), GLOB_TILDE, nullptr, &glob_result);
  std::vector<std::string> paths;
  if (rc == 0) {
    for (size_t i = 0; i < glob_result.gl_pathc; ++i) {
      paths.emplace_back(glob_result.gl_pathv[i]);
    }
  }
  globfree(&glob_result);
  std::sort(paths.begin(), paths.end());
  return paths;
}

std::vector<float> preprocess_image(const cv::Mat& bgr) {
  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  cv::Mat resized;
  cv::resize(rgb, resized, cv::Size(kModelSize, kModelSize), 0, 0, cv::INTER_LINEAR);
  resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);
  const float mean[3] = {0.485f, 0.456f, 0.406f};
  const float stdv[3] = {0.229f, 0.224f, 0.225f};
  std::vector<float> chw(3 * kModelSize * kModelSize);
  for (int y = 0; y < kModelSize; ++y) {
    const cv::Vec3f* row = resized.ptr<cv::Vec3f>(y);
    for (int x = 0; x < kModelSize; ++x) {
      for (int c = 0; c < 3; ++c) {
        chw[c * kModelSize * kModelSize + y * kModelSize + x] =
            (row[x][c] - mean[c]) / stdv[c];
      }
    }
  }
  return chw;
}

std::vector<std::pair<float, float>> build_point_grid(int n) {
  std::vector<std::pair<float, float>> points;
  points.reserve(static_cast<size_t>(n * n));
  const float offset = 1.0f / (2.0f * n);
  for (int y = 0; y < n; ++y) {
    const float py = offset + (1.0f - 2.0f * offset) * y / std::max(n - 1, 1);
    for (int x = 0; x < n; ++x) {
      const float px = offset + (1.0f - 2.0f * offset) * x / std::max(n - 1, 1);
      points.emplace_back(px, py);
    }
  }
  return points;
}

float box_iou(const Candidate& a, const Candidate& b) {
  const int ix0 = std::max(a.x0, b.x0);
  const int iy0 = std::max(a.y0, b.y0);
  const int ix1 = std::min(a.x1, b.x1);
  const int iy1 = std::min(a.y1, b.y1);
  const int iw = std::max(0, ix1 - ix0 + 1);
  const int ih = std::max(0, iy1 - iy0 + 1);
  const int inter = iw * ih;
  const int a_box_area = std::max(0, a.x1 - a.x0 + 1) * std::max(0, a.y1 - a.y0 + 1);
  const int b_box_area = std::max(0, b.x1 - b.x0 + 1) * std::max(0, b.y1 - b.y0 + 1);
  const int union_area = a_box_area + b_box_area - inter;
  return union_area > 0 ? static_cast<float>(inter) / union_area : 0.0f;
}

std::vector<Candidate> nms(std::vector<Candidate> candidates, float thresh) {
  std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
    return a.iou > b.iou;
  });
  std::vector<Candidate> kept;
  std::vector<uint8_t> suppressed(candidates.size(), 0);
  for (size_t i = 0; i < candidates.size(); ++i) {
    if (suppressed[i]) {
      continue;
    }
    kept.push_back(std::move(candidates[i]));
    for (size_t j = i + 1; j < candidates.size(); ++j) {
      if (!suppressed[j] && box_iou(kept.back(), candidates[j]) > thresh) {
        suppressed[j] = 1;
      }
    }
  }
  return kept;
}

std::vector<Candidate> nms_by_crop_preference(std::vector<Candidate> candidates, float thresh) {
  std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
    const float as = a.crop_area > 0 ? 1.0f / static_cast<float>(a.crop_area) : 0.0f;
    const float bs = b.crop_area > 0 ? 1.0f / static_cast<float>(b.crop_area) : 0.0f;
    return as > bs;
  });
  std::vector<Candidate> kept;
  std::vector<uint8_t> suppressed(candidates.size(), 0);
  for (size_t i = 0; i < candidates.size(); ++i) {
    if (suppressed[i]) {
      continue;
    }
    kept.push_back(std::move(candidates[i]));
    for (size_t j = i + 1; j < candidates.size(); ++j) {
      if (!suppressed[j] && box_iou(kept.back(), candidates[j]) > thresh) {
        suppressed[j] = 1;
      }
    }
  }
  return kept;
}

std::vector<Candidate> resolve_overlaps(std::vector<Candidate> masks, int h, int w, int min_area) {
  std::sort(masks.begin(), masks.end(), [](const Candidate& a, const Candidate& b) {
    return a.area > b.area;
  });
  std::vector<int> owner(static_cast<size_t>(h * w), -1);
  std::vector<float> score(static_cast<size_t>(h * w), -1.0f);
  for (size_t i = 0; i < masks.size(); ++i) {
    const float s = masks[i].iou * masks[i].stability;
    const auto& m = masks[i].mask;
    for (size_t p = 0; p < m.size(); ++p) {
      if (m[p] && s > score[p]) {
        owner[p] = static_cast<int>(i);
        score[p] = s;
      }
    }
  }

  std::vector<Candidate> out;
  for (size_t i = 0; i < masks.size(); ++i) {
    Candidate c = masks[i];
    std::fill(c.mask.begin(), c.mask.end(), 0);
    c.area = 0;
    c.x0 = w;
    c.y0 = h;
    c.x1 = 0;
    c.y1 = 0;
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        const size_t p = static_cast<size_t>(y * w + x);
        if (owner[p] == static_cast<int>(i)) {
          c.mask[p] = 1;
          c.area += 1;
          c.x0 = std::min(c.x0, x);
          c.y0 = std::min(c.y0, y);
          c.x1 = std::max(c.x1, x);
          c.y1 = std::max(c.y1, y);
        }
      }
    }
    if (c.area >= min_area) {
      out.push_back(std::move(c));
    }
  }
  return out;
}

void save_json(const std::string& path, const std::string& image_name,
               const std::string& original_image, const std::vector<Candidate>& masks,
               int h, int w) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("failed to write " + path);
  }
  int64_t total_area = 0;
  for (const auto& m : masks) {
    total_area += m.area;
  }
  out << "{\n";
  out << "  \"image_name\": \"" << image_name << "\",\n";
  out << "  \"original_image\": \"" << original_image << "\",\n";
  out << "  \"num_masks\": " << masks.size() << ",\n";
  out << "  \"masked_ratio\": " << (static_cast<double>(total_area) / (static_cast<double>(h) * w)) << ",\n";
  out << "  \"masks\": [\n";
  for (size_t i = 0; i < masks.size(); ++i) {
    const auto& m = masks[i];
    out << "    {\n";
    out << "      \"segmentation\": [\n";
    for (int y = 0; y < h; ++y) {
      out << "        [";
      for (int x = 0; x < w; ++x) {
        if (x != 0) {
          out << ", ";
        }
        out << (m.mask[static_cast<size_t>(y * w + x)] ? "true" : "false");
      }
      out << "]" << (y + 1 == h ? "\n" : ",\n");
    }
    out << "      ],\n";
    out << "      \"bbox\": [" << m.x0 << ", " << m.y0 << ", " << (m.x1 - m.x0 + 1)
        << ", " << (m.y1 - m.y0 + 1) << "],\n";
    out << "      \"area\": " << m.area << ",\n";
    out << "      \"predicted_iou\": " << m.iou << ",\n";
    out << "      \"stability_score\": " << m.stability << "\n";
    out << "    }" << (i + 1 == masks.size() ? "\n" : ",\n");
  }
  out << "  ],\n";
  out << "  \"individual_dir\": \"\"\n";
  out << "}\n";
}

std::vector<cv::Vec3b> fixed_colors(size_t n) {
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(80, 220);
  std::vector<cv::Vec3b> colors;
  colors.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    colors.emplace_back(static_cast<uint8_t>(dist(rng)), static_cast<uint8_t>(dist(rng)),
                        static_cast<uint8_t>(dist(rng)));
  }
  return colors;
}

void save_visuals(const cv::Mat& bgr, const std::vector<Candidate>& masks,
                  const std::string& overlay_path, const std::string& numbered_path) {
  cv::Mat overlay = bgr.clone();
  cv::Mat numbered = bgr.clone();
  auto colors = fixed_colors(masks.size());
  for (size_t i = 0; i < masks.size(); ++i) {
    cv::Mat mask(bgr.rows, bgr.cols, CV_8UC1, const_cast<uint8_t*>(masks[i].mask.data()));
    cv::Mat color_img(bgr.size(), CV_8UC3, colors[i]);
    color_img.copyTo(overlay, mask);
    cv::addWeighted(bgr, 0.5, overlay, 0.5, 0.0, overlay);
    color_img.copyTo(numbered, mask);
    cv::addWeighted(bgr, 0.4, numbered, 0.6, 0.0, numbered);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    cv::drawContours(overlay, contours, -1, cv::Scalar(255, 255, 255), 1);
    cv::drawContours(numbered, contours, -1, cv::Scalar(255, 255, 255), 2);
    const int cx = (masks[i].x0 + masks[i].x1) / 2;
    const int cy = (masks[i].y0 + masks[i].y1) / 2;
    const std::string label = std::to_string(i + 1);
    int baseline = 0;
    const auto sz = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseline);
    cv::rectangle(numbered, cv::Point(cx - sz.width / 2 - 3, cy - sz.height / 2 - 3),
                  cv::Point(cx + sz.width / 2 + 3, cy + sz.height / 2 + 3),
                  cv::Scalar(0, 0, 0), -1);
    cv::putText(numbered, label, cv::Point(cx - sz.width / 2, cy + sz.height / 2),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
  }
  cv::imwrite(overlay_path, overlay);
  cv::imwrite(numbered_path, numbered);
}

std::vector<Candidate> generate_candidates_for_crop(TrtEngine& encoder, TrtEngine& decoder,
                                                    cudaStream_t stream, const cv::Mat& bgr,
                                                    const Args& args) {
  const int h = bgr.rows;
  const int w = bgr.cols;
  auto input = preprocess_image(bgr);
  std::vector<float> high0(1 * 32 * 256 * 256);
  std::vector<float> high1(1 * 64 * 128 * 128);
  std::vector<float> embed(1 * 256 * 64 * 64);

  encoder.copy_h2d("image", input.data(), stream);
  encoder.run(stream);
  encoder.copy_d2h("high_res_0", high0.data(), stream);
  encoder.copy_d2h("high_res_1", high1.data(), stream);
  encoder.copy_d2h("image_embed", embed.data(), stream);
  check_cuda(cudaStreamSynchronize(stream), "encoder synchronize");

  const auto points = build_point_grid(args.points_per_side);
  std::vector<Candidate> candidates;
  std::vector<float> point_coords(kDecoderBatch * 2, 0.0f);
  std::vector<int64_t> point_labels(kDecoderBatch, 1);
  std::vector<float> low_res(kDecoderBatch * kMasksPerPoint * kLowRes * kLowRes);
  std::vector<float> ious(kDecoderBatch * kMasksPerPoint);

  for (size_t start = 0; start < points.size(); start += args.points_per_batch) {
    const size_t batch_count = std::min<size_t>(args.points_per_batch, points.size() - start);
    std::fill(point_coords.begin(), point_coords.end(), 0.0f);
    std::fill(point_labels.begin(), point_labels.end(), 1);
    for (size_t i = 0; i < batch_count; ++i) {
      point_coords[i * 2 + 0] = points[start + i].first * kModelSize;
      point_coords[i * 2 + 1] = points[start + i].second * kModelSize;
    }
    decoder.copy_h2d("image_embed", embed.data(), stream);
    decoder.copy_h2d("high_res_0", high0.data(), stream);
    decoder.copy_h2d("high_res_1", high1.data(), stream);
    decoder.copy_h2d("point_coords", point_coords.data(), stream);
    decoder.copy_h2d("point_labels", point_labels.data(), stream);
    decoder.run(stream);
    decoder.copy_d2h("low_res_masks", low_res.data(), stream);
    decoder.copy_d2h("iou_predictions", ious.data(), stream);
    check_cuda(cudaStreamSynchronize(stream), "decoder synchronize");

    for (size_t bi = 0; bi < batch_count; ++bi) {
      for (int mi = 0; mi < kMasksPerPoint; ++mi) {
        const float pred_iou = ious[bi * kMasksPerPoint + mi];
        if (pred_iou <= args.pred_iou_thresh) {
          continue;
        }
        const size_t off =
            (bi * kMasksPerPoint + mi) * static_cast<size_t>(kLowRes * kLowRes);
        cv::Mat low(kLowRes, kLowRes, CV_32FC1, low_res.data() + off);
        cv::Mat up;
        cv::resize(low, up, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);
        int inter = 0;
        int uni = 0;
        Candidate c;
        c.mask.assign(static_cast<size_t>(h * w), 0);
        c.x0 = w;
        c.y0 = h;
        c.x1 = 0;
        c.y1 = 0;
        c.iou = pred_iou;
        c.point_index = static_cast<int>(start + bi);
        c.mask_index = mi;
        c.crop_area = h * w;
        for (int y = 0; y < h; ++y) {
          const float* row = up.ptr<float>(y);
          for (int x = 0; x < w; ++x) {
            const float v = row[x];
            if (v > args.stability_score_offset) {
              inter += 1;
            }
            if (v > -args.stability_score_offset) {
              uni += 1;
            }
            if (v > 0.0f) {
              c.mask[static_cast<size_t>(y * w + x)] = 1;
              c.area += 1;
              c.x0 = std::min(c.x0, x);
              c.y0 = std::min(c.y0, y);
              c.x1 = std::max(c.x1, x);
              c.y1 = std::max(c.y1, y);
            }
          }
        }
        c.stability = uni > 0 ? static_cast<float>(inter) / uni : 0.0f;
        if (c.area >= args.min_mask_area && c.stability >= args.stability_score_thresh) {
          candidates.push_back(std::move(c));
        }
      }
    }
  }
  candidates = nms(std::move(candidates), args.box_nms_thresh);
  return candidates;
}

std::vector<std::array<int, 4>> generate_crop_boxes(int h, int w, const Args& args) {
  std::vector<std::array<int, 4>> boxes;
  boxes.push_back({0, 0, w, h});
  const int short_side = std::min(h, w);
  for (int layer = 0; layer < args.crop_n_layers; ++layer) {
    const int n_crops = 1 << (layer + 1);
    const int overlap = static_cast<int>(args.crop_overlap_ratio * short_side * (2.0f / n_crops));
    auto crop_len = [&](int orig_len) {
      return static_cast<int>(std::ceil((overlap * (n_crops - 1) + orig_len) /
                                        static_cast<float>(n_crops)));
    };
    const int crop_w = crop_len(w);
    const int crop_h = crop_len(h);
    for (int ix = 0; ix < n_crops; ++ix) {
      for (int iy = 0; iy < n_crops; ++iy) {
        const int x0 = static_cast<int>((crop_w - overlap) * ix);
        const int y0 = static_cast<int>((crop_h - overlap) * iy);
        boxes.push_back({x0, y0, std::min(x0 + crop_w, w), std::min(y0 + crop_h, h)});
      }
    }
  }
  return boxes;
}

bool near_crop_edge(const Candidate& c, const std::array<int, 4>& crop, int h, int w) {
  constexpr int atol = 20;
  const int gx0 = c.x0 + crop[0];
  const int gy0 = c.y0 + crop[1];
  const int gx1 = c.x1 + crop[0];
  const int gy1 = c.y1 + crop[1];
  const bool near_crop =
      std::abs(gx0 - crop[0]) <= atol || std::abs(gy0 - crop[1]) <= atol ||
      std::abs(gx1 - crop[2]) <= atol || std::abs(gy1 - crop[3]) <= atol;
  const bool near_image =
      std::abs(gx0 - 0) <= atol || std::abs(gy0 - 0) <= atol ||
      std::abs(gx1 - w) <= atol || std::abs(gy1 - h) <= atol;
  return near_crop && !near_image;
}

Candidate uncrop_candidate(const Candidate& c, const std::array<int, 4>& crop, int h, int w) {
  Candidate out = c;
  out.mask.assign(static_cast<size_t>(h * w), 0);
  const int crop_w = crop[2] - crop[0];
  const int crop_h = crop[3] - crop[1];
  out.x0 = w;
  out.y0 = h;
  out.x1 = 0;
  out.y1 = 0;
  out.area = 0;
  for (int y = 0; y < crop_h; ++y) {
    for (int x = 0; x < crop_w; ++x) {
      if (!c.mask[static_cast<size_t>(y * crop_w + x)]) {
        continue;
      }
      const int gx = x + crop[0];
      const int gy = y + crop[1];
      out.mask[static_cast<size_t>(gy * w + gx)] = 1;
      out.area += 1;
      out.x0 = std::min(out.x0, gx);
      out.y0 = std::min(out.y0, gy);
      out.x1 = std::max(out.x1, gx);
      out.y1 = std::max(out.y1, gy);
    }
  }
  out.crop_area = crop_w * crop_h;
  return out;
}

std::vector<Candidate> generate_masks_for_image(TrtEngine& encoder, TrtEngine& decoder,
                                                cudaStream_t stream, const cv::Mat& bgr,
                                                const Args& args) {
  const int h = bgr.rows;
  const int w = bgr.cols;
  std::vector<Candidate> all;
  const auto crop_boxes = generate_crop_boxes(h, w, args);
  for (const auto& crop : crop_boxes) {
    const cv::Rect rect(crop[0], crop[1], crop[2] - crop[0], crop[3] - crop[1]);
    cv::Mat cropped = bgr(rect).clone();
    auto crop_candidates = generate_candidates_for_crop(encoder, decoder, stream, cropped, args);
    for (const auto& c : crop_candidates) {
      if (near_crop_edge(c, crop, h, w)) {
        continue;
      }
      all.push_back(uncrop_candidate(c, crop, h, w));
    }
  }
  if (crop_boxes.size() > 1) {
    all = nms_by_crop_preference(std::move(all), args.crop_nms_thresh);
  }
  all = resolve_overlaps(std::move(all), h, w, args.min_mask_area);
  std::sort(all.begin(), all.end(), [](const Candidate& a, const Candidate& b) {
    return a.area > b.area;
  });
  return all;
}

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto need_value = [&](const std::string& name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("missing value for " + name);
      }
      return argv[++i];
    };
    if (a == "--images") {
      args.images_glob = need_value(a);
    } else if (a == "--output-dir") {
      args.output_dir = need_value(a);
    } else if (a == "--encoder-engine") {
      args.encoder_engine = need_value(a);
    } else if (a == "--decoder-engine") {
      args.decoder_engine = need_value(a);
    } else if (a == "--points-per-side") {
      args.points_per_side = std::stoi(need_value(a));
    } else if (a == "--points-per-batch") {
      args.points_per_batch = std::stoi(need_value(a));
    } else if (a == "--min-mask-area") {
      args.min_mask_area = std::stoi(need_value(a));
    } else if (a == "--pred-iou-thresh") {
      args.pred_iou_thresh = std::stof(need_value(a));
    } else if (a == "--stability-score-thresh") {
      args.stability_score_thresh = std::stof(need_value(a));
    } else if (a == "--crop-n-layers") {
      args.crop_n_layers = std::stoi(need_value(a));
    } else if (a == "--overwrite") {
      args.skip_existing = false;
    } else {
      throw std::runtime_error("unknown argument: " + a);
    }
  }
  if (args.images_glob.empty() || args.output_dir.empty()) {
    throw std::runtime_error("usage: sam2_trt_amg_runner --images 'glob' --output-dir DIR [--overwrite]");
  }
  if (args.points_per_batch != kDecoderBatch) {
    throw std::runtime_error("this build expects --points-per-batch 64 to match decoder engine");
  }
  return args;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parse_args(argc, argv);
    fs::create_directories(args.output_dir);
    const auto images = expand_glob(args.images_glob);
    if (images.empty()) {
      throw std::runtime_error("no images matched " + args.images_glob);
    }

    Logger logger;
    TrtEngine encoder(logger, args.encoder_engine);
    TrtEngine decoder(logger, args.decoder_engine);
    cudaStream_t stream = nullptr;
    check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");

    int ok = 0;
    int skipped = 0;
    int failed = 0;
    for (const auto& image_path : images) {
      const std::string img_name = stem(image_path);
      const fs::path out_dir(args.output_dir);
      const std::string json_path = (out_dir / (img_name + "_sam_masks.json")).string();
      const std::string done_path = (out_dir / (img_name + "_sam_done.flag")).string();
      if (args.skip_existing && fs::exists(done_path)) {
        skipped += 1;
        std::cout << "{\"image\":\"" << img_name << "\",\"status\":\"skipped\"}\n";
        continue;
      }
      try {
        cv::Mat bgr = cv::imread(image_path, cv::IMREAD_COLOR);
        if (bgr.empty()) {
          throw std::runtime_error("failed to read image");
        }
        auto masks = generate_masks_for_image(encoder, decoder, stream, bgr, args);
        if (masks.empty()) {
          std::cout << "{\"image\":\"" << img_name << "\",\"status\":\"no_masks\"}\n";
          failed += 1;
          continue;
        }
        save_visuals(bgr, masks, (out_dir / (img_name + "_sam_masks.png")).string(),
                     (out_dir / (img_name + "_numbered.png")).string());
        save_json(json_path, img_name, basename(image_path), masks, bgr.rows, bgr.cols);
        std::ofstream flag(done_path);
        flag << "{\"processed\":true,\"num_masks\":" << masks.size() << "}\n";
        ok += 1;
        std::cout << "{\"image\":\"" << img_name << "\",\"status\":\"success\",\"num_masks\":"
                  << masks.size() << "}\n";
      } catch (const std::exception& exc) {
        failed += 1;
        std::cerr << "{\"image\":\"" << img_name << "\",\"status\":\"error\",\"error\":\""
                  << exc.what() << "\"}\n";
      }
    }

    check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
    std::cerr << "summary ok=" << ok << " skipped=" << skipped << " failed=" << failed << "\n";
    return failed == 0 ? 0 : 1;
  } catch (const std::exception& exc) {
    std::cerr << "error: " << exc.what() << "\n";
    return 1;
  }
}
