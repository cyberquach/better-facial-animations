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
  // Both provider routes arm and complete the same guard before touching ORT.
  wchar_t temp[MAX_PATH], path[MAX_PATH];
  CHECK(GetTempPathW(MAX_PATH, temp));
  CHECK(GetTempFileNameW(temp, L"bfa", 0, path) && DeleteFileW(path));
  a2f::SetGpuStartupMarker(path);
  for (bool independent : {false, true}) {
    a2f::OrtOptions options; options.independent_device = independent;
    a2f::GpuAttempt attempt(options, err);
    CHECK(attempt.allowed() && GetFileAttributesW(path) != INVALID_FILE_ATTRIBUTES);
    CHECK(attempt.Finish(true) && GetFileAttributesW(path) == INVALID_FILE_ATTRIBUTES);
  }
  a2f::DisableGpu("injected session failure");
  a2f::OrtOptions options;
  for (bool independent : {false, true}) {
    options.independent_device = independent;
  // None can touch the deliberately incomplete ORT mock after any model fails.
  CHECK(!a2f::CreateOrtNet(L"", L"", options, err, info));
  CHECK(!a2f::CreateOrtNet3(L"", L"", options, err, info));
  CHECK(!a2f::CreatePerformanceNet(L"", L"", options, err, info));
  CHECK(!a2f::CreateOrtEmotionNet(L"", L"", options, err));
  CHECK(err.find("injected session failure") != std::string::npos);
  }
  // A GPU failure must not block a CPU session's guard or create a marker.
  a2f::OrtOptions cpu; cpu.gpu = false;
  a2f::GpuAttempt cpu_attempt(cpu, err);
  CHECK(cpu_attempt.allowed() && cpu_attempt.Finish(true));
  CHECK(GetFileAttributesW(path) == INVALID_FILE_ATTRIBUTES);
  puts("tensor validation and shared GPU failure gate across all models: pass");
}
