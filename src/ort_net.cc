// ONNX Runtime session wrapper; session options are tuned for minimum VRAM.
#include "ort_net.h"

#include <windows.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>

#include "live_core.h"
#include "onnxruntime_c_api.h"

// dml_provider_factory.h needs DirectML.h (MSVC SAL annotations); only its
// first entry is used.
struct OrtDmlApi {
  OrtStatus*(ORT_API_CALL* SessionOptionsAppendExecutionProvider_DML)(
      OrtSessionOptions*, int device_id);
};

namespace a2f {
namespace {

const OrtApi* api;
OrtEnv* env;
std::string version;

// Consumes s; on error stores "<what>: <message>" in err and returns false.
bool Check(OrtStatus* s, std::string& err, const char* what) {
  if (!s) return true;
  err = std::string(what) + ": " + api->GetErrorMessage(s);
  api->ReleaseStatus(s);
  return false;
}

class OrtNet : public Net {
 public:
  OrtNet(OrtSession* sess, OrtMemoryInfo* mem, int batch)
      : sess_(sess), mem_(mem), batch_(batch) {}
  ~OrtNet() override {
    api->ReleaseSession(sess_);
    api->ReleaseMemoryInfo(mem_);
  }
  int Batch() const override { return batch_; }
  bool Run(const float* x, const float* emo, int b, float* out,
           std::string& err) override {
    int64_t sx[3] = {b, 1, kWindow}, se[3] = {b, 1, kNumEmo};
    OrtValue *vx = nullptr, *ve = nullptr, *vy = nullptr;
    bool ok = Check(api->CreateTensorWithDataAsOrtValue(
                        mem_, const_cast<float*>(x),
                        static_cast<size_t>(b) * kWindow * 4, sx, 3,
                        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &vx),
                    err, "input") &&
              Check(api->CreateTensorWithDataAsOrtValue(
                        mem_, const_cast<float*>(emo),
                        static_cast<size_t>(b) * kNumEmo * 4, se, 3,
                        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &ve),
                    err, "emotion");
    if (ok) {
      const char* in[] = {"input", "emotion"};
      const char* outn[] = {"result"};
      const OrtValue* ins[] = {vx, ve};
      ok =
          Check(api->Run(sess_, nullptr, in, ins, 2, outn, 1, &vy), err, "run");
    }
    if (ok) {
      float* p = nullptr;
      ok = Check(api->GetTensorMutableData(vy, reinterpret_cast<void**>(&p)),
                 err, "output");
      if (ok) memcpy(out, p, static_cast<size_t>(b) * kNumOut * 4);
    }
    if (vy) api->ReleaseValue(vy);
    if (vx) api->ReleaseValue(vx);
    if (ve) api->ReleaseValue(ve);
    return ok;
  }

 private:
  OrtSession* sess_;
  OrtMemoryInfo* mem_;
  int batch_;
};

}  // namespace

std::unique_ptr<Net> CreateOrtNet(const std::wstring& dll_dir,
                                  const std::wstring& model,
                                  const OrtOptions& o, std::string& err,
                                  std::string& info) {
  if (!api) {
    // Full path + altered search path: DirectML.dll /
    // onnxruntime_providers_shared.dll resolve from the same folder.
    HMODULE h = LoadLibraryExW((dll_dir + L"\\onnxruntime.dll").c_str(),
                               nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!h) {
      err = "cannot load onnxruntime.dll (error " +
            std::to_string(GetLastError()) + ")";
      return nullptr;
    }
    auto base = reinterpret_cast<const OrtApiBase* (*)()>(
        GetProcAddress(h, "OrtGetApiBase"));
    if (!base || !(api = base()->GetApi(ORT_API_VERSION))) {
      err = "onnxruntime.dll: wrong version";
      api = nullptr;
      return nullptr;
    }
    if (!Check(api->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "A2FLive", &env), err,
               "env"))
      return nullptr;
    version = base()->GetVersionString();
  }
  const int batch = std::max(1, o.batch);
  OrtSessionOptions* so = nullptr;
  OrtSession* sess = nullptr;
  OrtMemoryInfo* mem = nullptr;
  bool ok = Check(api->CreateSessionOptions(&so), err, "options");
  if (ok && o.gpu) {
    const OrtDmlApi* dml = nullptr;
    ok =
        Check(api->DisableMemPattern(so), err, "mem pattern") &&
        Check(api->SetSessionExecutionMode(so, ORT_SEQUENTIAL), err, "mode") &&
        Check(api->GetExecutionProviderApi(
                  "DML", ORT_API_VERSION, reinterpret_cast<const void**>(&dml)),
              err, "DML api") &&
        Check(dml->SessionOptionsAppendExecutionProvider_DML(so, 0), err,
              "DirectML");
    // DML's arena rounds allocations up to powers of two: off saves ~16-32 MB
    // at small batches.
    if (ok)
      ok = Check(
          api->AddSessionConfigEntry(so, "ep.dml.disable_memory_arena", "1"),
          err, "arena");
    // The batch is always the same: let DirectML plan one static shape (also
    // 4x faster at batch 1-2).
    for (const char* dim : {"input_dynamic_axes_1", "emotion_dynamic_axes_1"})
      if (ok)
        ok = Check(api->AddFreeDimensionOverrideByName(so, dim, batch), err,
                   "batch override");
  } else if (ok) {
    ok = Check(api->SetIntraOpNumThreads(so, 2), err, "threads") &&
         Check(api->SetInterOpNumThreads(so, 1), err, "threads");
  }
  if (ok)
    ok = Check(api->CreateSession(env, model.c_str(), so, &sess), err,
               "session");
  if (ok)
    ok = Check(
        api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem),
        err, "memory info");
  if (so) api->ReleaseSessionOptions(so);
  if (!ok) {
    if (sess) api->ReleaseSession(sess);
    if (mem) api->ReleaseMemoryInfo(mem);
    return nullptr;
  }
  info = "onnxruntime " + version +
         (o.gpu ? ", DirectML device 0" : ", CPU 2 threads") + ", batch " +
         std::to_string(batch);
  return std::make_unique<OrtNet>(sess, mem, batch);
}

}  // namespace a2f
