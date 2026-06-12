// Minimal SAM2 TensorRT runtime smoke runner.
//
// This is intentionally small: it verifies that the exported SAM2 TensorRT
// engines can be deserialized by C++ and can execute with static shapes. The
// full AMG runner should build on this instead of shelling out to trtexec.

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class Logger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cerr << "[TRT] " << msg << "\n";
    }
  }
};

struct CudaDeleter {
  void operator()(void* ptr) const {
    if (ptr != nullptr) {
      cudaFree(ptr);
    }
  }
};

using CudaPtr = std::unique_ptr<void, CudaDeleter>;

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

const char* dtype_name(nvinfer1::DataType dtype) {
  switch (dtype) {
    case nvinfer1::DataType::kFLOAT:
      return "float32";
    case nvinfer1::DataType::kHALF:
      return "float16";
    case nvinfer1::DataType::kINT8:
      return "int8";
    case nvinfer1::DataType::kINT32:
      return "int32";
    case nvinfer1::DataType::kBOOL:
      return "bool";
    case nvinfer1::DataType::kUINT8:
      return "uint8";
    case nvinfer1::DataType::kFP8:
      return "fp8";
    case nvinfer1::DataType::kBF16:
      return "bf16";
    case nvinfer1::DataType::kINT64:
      return "int64";
    case nvinfer1::DataType::kINT4:
      return "int4";
    default:
      return "unknown";
  }
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

std::string dims_string(const nvinfer1::Dims& dims) {
  std::string out;
  for (int i = 0; i < dims.nbDims; ++i) {
    if (i != 0) {
      out += "x";
    }
    out += std::to_string(dims.d[i]);
  }
  return out.empty() ? "scalar" : out;
}

int64_t volume(const nvinfer1::Dims& dims) {
  int64_t total = 1;
  for (int i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] < 0) {
      throw std::runtime_error("dynamic tensor shape is not supported by this smoke runner");
    }
    total *= dims.d[i];
  }
  return total;
}

void check_cuda(cudaError_t status, const std::string& context) {
  if (status != cudaSuccess) {
    throw std::runtime_error(context + ": " + cudaGetErrorString(status));
  }
}

struct TensorBinding {
  std::string name;
  nvinfer1::Dims dims;
  nvinfer1::DataType dtype;
  nvinfer1::TensorIOMode mode;
  size_t bytes = 0;
  CudaPtr device;
};

void inspect_engine(const std::string& path, bool run) {
  Logger logger;
  auto data = read_file(path);
  std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
  if (!runtime) {
    throw std::runtime_error("failed to create TensorRT runtime");
  }
  std::unique_ptr<nvinfer1::ICudaEngine> engine(
      runtime->deserializeCudaEngine(data.data(), data.size()));
  if (!engine) {
    throw std::runtime_error("failed to deserialize " + path);
  }

  std::cout << "engine " << path << "\n";
  std::cout << "  io_tensors " << engine->getNbIOTensors() << "\n";

  std::vector<TensorBinding> bindings;
  bindings.reserve(static_cast<size_t>(engine->getNbIOTensors()));
  for (int i = 0; i < engine->getNbIOTensors(); ++i) {
    const char* name = engine->getIOTensorName(i);
    TensorBinding binding;
    binding.name = name;
    binding.dims = engine->getTensorShape(name);
    binding.dtype = engine->getTensorDataType(name);
    binding.mode = engine->getTensorIOMode(name);
    binding.bytes = static_cast<size_t>(volume(binding.dims)) * dtype_size(binding.dtype);
    std::cout << "  "
              << (binding.mode == nvinfer1::TensorIOMode::kINPUT ? "input " : "output ")
              << binding.name << " " << dims_string(binding.dims) << " "
              << dtype_name(binding.dtype) << " bytes=" << binding.bytes << "\n";
    bindings.emplace_back(std::move(binding));
  }

  if (!run) {
    return;
  }

  std::unique_ptr<nvinfer1::IExecutionContext> context(engine->createExecutionContext());
  if (!context) {
    throw std::runtime_error("failed to create execution context");
  }

  cudaStream_t stream = nullptr;
  check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");
  for (auto& binding : bindings) {
    void* ptr = nullptr;
    check_cuda(cudaMalloc(&ptr, binding.bytes), "cudaMalloc " + binding.name);
    binding.device.reset(ptr);
    check_cuda(cudaMemsetAsync(ptr, 0, binding.bytes, stream), "cudaMemsetAsync " + binding.name);
    if (!context->setTensorAddress(binding.name.c_str(), ptr)) {
      throw std::runtime_error("setTensorAddress failed for " + binding.name);
    }
  }
  if (!context->enqueueV3(stream)) {
    throw std::runtime_error("enqueueV3 failed for " + path);
  }
  check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
  check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
  std::cout << "  run ok\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::vector<std::string> engines;
    bool run = false;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--engine" && i + 1 < argc) {
        engines.emplace_back(argv[++i]);
      } else if (arg == "--run") {
        run = true;
      } else {
        std::cerr << "usage: " << argv[0] << " --engine path.plan [--engine path.plan] [--run]\n";
        return 2;
      }
    }
    if (engines.empty()) {
      std::cerr << "usage: " << argv[0] << " --engine path.plan [--engine path.plan] [--run]\n";
      return 2;
    }
    for (const auto& engine : engines) {
      inspect_engine(engine, run);
    }
  } catch (const std::exception& exc) {
    std::cerr << "error: " << exc.what() << "\n";
    return 1;
  }
  return 0;
}
