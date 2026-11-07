#include "../src/inference/ort_net.cc"
#include <cstdlib>
#include <limits>
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "line %d: %s\n", __LINE__, #x); std::exit(1); } } while (0)
namespace {
size_t count = 3;
ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
float data[3] = {1, 2, 3};
int released = 0;
OrtStatus* ORT_API_CALL Shape(const OrtValue*, OrtTensorTypeAndShapeInfo** p) noexcept {
  *p = reinterpret_cast<OrtTensorTypeAndShapeInfo*>(1); return nullptr;
}
OrtStatus* ORT_API_CALL Count(const OrtTensorTypeAndShapeInfo*, size_t* p) noexcept { *p = count; return nullptr; }
OrtStatus* ORT_API_CALL Type(const OrtTensorTypeAndShapeInfo*, ONNXTensorElementDataType* p) noexcept { *p = type; return nullptr; }
OrtStatus* ORT_API_CALL Data(OrtValue*, void** p) noexcept { *p = data; return nullptr; }
void ORT_API_CALL Release(OrtTensorTypeAndShapeInfo*) noexcept { ++released; }
}
int main() {
  OrtApi mock{};
  mock.GetTensorTypeAndShape = Shape;
  mock.GetTensorShapeElementCount = Count;
  mock.GetTensorElementType = Type;
  mock.GetTensorMutableData = Data;
  mock.ReleaseTensorTypeAndShapeInfo = Release;
  a2f::api = &mock;
  auto* value = reinterpret_cast<OrtValue*>(1);
  float out[3] = {-1, -1, -1};
  std::string err, info;
  CHECK(!a2f::ReadFloats(nullptr, 3, out, err, "test"));
  count = 2;
  CHECK(!a2f::ReadFloats(value, 3, out, err, "test") && out[0] == -1);
  count = 3; type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
  CHECK(!a2f::ReadFloats(value, 3, out, err, "test") && out[0] == -1);
  type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  for (float invalid : {std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::quiet_NaN()}) {
    data[2] = invalid;
    CHECK(!a2f::ReadFloats(value, 3, out, err, "test") && out[0] == -1);
  }
  data[2] = 3;
  CHECK(a2f::ReadFloats(value, 3, out, err, "test") && out[2] == 3);
  CHECK(released == 5);
  // Caller-owned outputs skip only the copy, never shape/type/finite checks.
  CHECK(a2f::ReadFloats(value, 3, data, err, "bound output"));
  count = 2;
  CHECK(!a2f::ReadFloats(value, 3, data, err, "bound output"));
  count = 3; type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
  CHECK(!a2f::ReadFloats(value, 3, data, err, "bound output"));
  type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  data[2] = std::numeric_limits<float>::quiet_NaN();
  CHECK(!a2f::ReadFloats(value, 3, data, err, "bound output"));
  data[2] = 3;
  // No startup file exists in the API: a failed GPU attempt is reset explicitly.
  a2f::DisableGpu("injected session failure");
  a2f::OrtOptions options;
  for (bool independent : {false, true}) {
    options.independent_device = independent;
    CHECK(!a2f::CreateOrtNet(L"", L"", options, err, info));
    CHECK(!a2f::CreateOrtNet3(L"", L"", options, err, info));
    CHECK(!a2f::CreatePerformanceNet(L"", L"", options, err, info));
    CHECK(!a2f::CreateOrtEmotionNet(L"", L"", options, err));
    CHECK(err.find("injected session failure") != std::string::npos);
  }
  a2f::OrtOptions cpu; cpu.gpu = false;
  a2f::GpuAttempt cpu_attempt(cpu, err);
  CHECK(cpu_attempt.allowed() && cpu_attempt.Finish(true));
  a2f::RetryGpu();
  CHECK(a2f::GpuAllowed(err));
  for (bool independent : {false, true}) {
    options.independent_device = independent;
    a2f::GpuAttempt retry(options, err);
    CHECK(retry.allowed() && retry.Finish(true));
  }
  puts("tensor validation and explicit GPU retry across all models: pass");
}
