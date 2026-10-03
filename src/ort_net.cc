// ONNX Runtime session wrapper; session options are tuned for minimum VRAM.
#include "ort_net.h"

#include <d3d12.h>
#include <dxgi1_4.h>
#include <windows.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

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
const char* version = "";  // the runtime's own string

// Consumes s; on error stores "<what>: <message>" in err and returns false.
bool Check(OrtStatus* s, std::string& err, const char* what) {
  if (!s) return true;
  err = std::string(what) + ": " + api->GetErrorMessage(s);
  api->ReleaseStatus(s);
  return false;
}

// Adapter 0's Direct3D 12 device: one per adapter per process, so the very
// device DirectML runs on. Null without Direct3D 12 (d3d12.dll is not
// imported: the plugin loads without it).
ID3D12Device* OpenDevice() {
  using CreateFn =
      HRESULT(WINAPI*)(IUnknown*, D3D_FEATURE_LEVEL, REFIID, void**);
  static const HMODULE d3d12 = LoadLibraryW(L"d3d12.dll");
  auto create = d3d12 ? reinterpret_cast<CreateFn>(reinterpret_cast<void*>(
                            GetProcAddress(d3d12, "D3D12CreateDevice")))
                      : nullptr;
  IDXGIFactory1* factory = nullptr;
  IDXGIAdapter* adapter = nullptr;
  ID3D12Device* device = nullptr;
  if (create &&
      SUCCEEDED(CreateDXGIFactory1(__uuidof(IDXGIFactory1),
                                   reinterpret_cast<void**>(&factory))) &&
      SUCCEEDED(factory->EnumAdapters(0, &adapter)))
    create(adapter, D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device),
           reinterpret_cast<void**>(&device));
  if (adapter) adapter->Release();
  if (factory) factory->Release();
  return device;
}

// " [device removed reason XXXXXXXX]" when the device was lost, for
// DiagnoseNet; empty otherwise.
std::string RemovedReason(ID3D12Device* device) {
  HRESULT r = device ? device->GetDeviceRemovedReason() : S_OK;
  if (r == S_OK) return "";
  char b[48];
  snprintf(b, sizeof b, " [device removed reason %08lX]",
           static_cast<unsigned long>(r));
  return b;
}

class OrtNet : public Net {
 public:
  // device (may be null): held to ask why it was lost when a run fails.
  OrtNet(OrtSession* sess, OrtMemoryInfo* mem, int batch, ID3D12Device* device)
      : sess_(sess), mem_(mem), batch_(batch), device_(device) {}
  ~OrtNet() override {
    api->ReleaseSession(sess_);
    api->ReleaseMemoryInfo(mem_);
    // released last and at once: a lost device held on would be handed out
    // again to the next session
    if (device_) device_->Release();
  }
  int Batch() const override { return batch_; }
  bool Run(const float* x, const float* emo, int b, float* out,
           std::string& err) override {
    int64_t sx[3] = {b, 1, kWindow}, se[3] = {b, 1, kNumEmo};
    OrtValue *vx = nullptr, *ve = nullptr, *vy = nullptr;
    bool ok = Check(api->CreateTensorWithDataAsOrtValue(
                        mem_, const_cast<float*>(x),
                        static_cast<size_t>(b) * kWindow * sizeof(float), sx, 3,
                        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &vx),
                    err, "input") &&
              Check(api->CreateTensorWithDataAsOrtValue(
                        mem_, const_cast<float*>(emo),
                        static_cast<size_t>(b) * kNumEmo * sizeof(float), se, 3,
                        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &ve),
                    err, "emotion");
    if (ok) {
      const char* in[] = {"input", "emotion"};
      const char* outn[] = {"result"};
      const OrtValue* ins[] = {vx, ve};
      ok =
          Check(api->Run(sess_, nullptr, in, ins, 2, outn, 1, &vy), err, "run");
      if (!ok) {
        failed = true;
        err += RemovedReason(device_);
      }
    }
    if (ok) {
      float* p = nullptr;
      ok = Check(api->GetTensorMutableData(vy, reinterpret_cast<void**>(&p)),
                 err, "output");
      if (ok) memcpy(out, p, static_cast<size_t>(b) * kNumOut * sizeof(float));
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
  ID3D12Device* device_;
};

// Loads onnxruntime.dll from dll_dir, once.
bool LoadApi(const std::wstring& dll_dir, std::string& err) {
  if (api) return true;
  // Full path + altered search path: DirectML.dll /
  // onnxruntime_providers_shared.dll resolve from the same folder.
  HMODULE h = LoadLibraryExW((dll_dir + L"\\onnxruntime.dll").c_str(), nullptr,
                             LOAD_WITH_ALTERED_SEARCH_PATH);
  if (!h) {
    err = "cannot load onnxruntime.dll (error " +
          std::to_string(GetLastError()) + ")";
    return false;
  }
  auto base = reinterpret_cast<const OrtApiBase* (*)()>(
      reinterpret_cast<void*>(GetProcAddress(h, "OrtGetApiBase")));
  if (!base || !(api = base()->GetApi(ORT_API_VERSION))) {
    err = "onnxruntime.dll: wrong version";
    api = nullptr;
    FreeLibrary(h);
    return false;
  }
  if (!Check(api->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "BetterFacialAnimations",
                            &env),
             err, "env")) {
    api = nullptr;
    FreeLibrary(h);
    return false;
  }
  // The successful module and environment live as long as the worker.
  version = base()->GetVersionString();
  return true;
}

// A session of model on DirectML (adapter 0; device receives its Direct3D
// device, held to ask why it was lost) or the CPU, with the named free
// dimensions fixed: let DirectML plan one static shape (also 4x faster at
// batch 1-2).
bool CreateSession(const std::wstring& model, bool gpu, int batch,
                   OrtSession*& sess, OrtMemoryInfo*& mem,
                   ID3D12Device*& device, std::string& err) {
  OrtSessionOptions* so = nullptr;
  sess = nullptr;
  mem = nullptr;
  device = gpu ? OpenDevice() : nullptr;
  bool ok = Check(api->CreateSessionOptions(&so), err, "options");
  if (ok && gpu) {
    const OrtDmlApi* dml = nullptr;
    ok =
        Check(api->DisableMemPattern(so), err, "mem pattern") &&
        Check(api->SetSessionExecutionMode(so, ORT_SEQUENTIAL), err, "mode") &&
        Check(api->GetExecutionProviderApi(
                  "DML", ORT_API_VERSION, reinterpret_cast<const void**>(&dml)),
              err, "DML api") &&
        Check(dml->SessionOptionsAppendExecutionProvider_DML(so, 0), err,
              "DirectML");
    // DML's memory arena stays on: it keeps the work buffers of a run
    // allocated between runs (~16 MB more resident than freeing them), so a
    // loaded session never has to find VRAM again.
    for (const char* name : {"input_dynamic_axes_1", "emotion_dynamic_axes_1"})
      if (ok)
        ok = Check(api->AddFreeDimensionOverrideByName(so, name, batch), err,
                   "shape override");
  } else if (ok) {
    // Not ORT_ENABLE_ALL: its layout optimizations fail on the casts the CPU
    // provider inserts around the FP16 network's operators
    // (SimplifiedLayerNormFusion looks up a node argument that is gone).
    ok = Check(api->SetIntraOpNumThreads(so, 2), err, "threads") &&
         Check(api->SetInterOpNumThreads(so, 1), err, "threads") &&
         Check(api->SetSessionGraphOptimizationLevel(so, ORT_ENABLE_EXTENDED),
               err, "optimization");
  }
  if (ok)
    ok = Check(api->CreateSession(env, model.c_str(), so, &sess), err,
               "session");
  if (ok)
    ok = Check(
        api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem),
        err, "memory info");
  if (so) api->ReleaseSessionOptions(so);
  if (ok) return true;
  if (sess) api->ReleaseSession(sess);
  if (mem) api->ReleaseMemoryInfo(mem);
  err += RemovedReason(device);
  if (device) device->Release();
  return false;
}

}  // namespace

std::unique_ptr<Net> CreateOrtNet(const std::wstring& dll_dir,
                                  const std::wstring& model,
                                  const OrtOptions& o, std::string& err,
                                  std::string& info) {
  if (!LoadApi(dll_dir, err)) return nullptr;
  const int batch = std::max(1, o.batch);
  OrtSession* sess;
  OrtMemoryInfo* mem;
  ID3D12Device* device;
  // The batch is always the same.
  if (!CreateSession(model, o.gpu, batch, sess, mem, device, err))
    return nullptr;
  info = std::string("onnxruntime ") + version +
         (o.gpu ? ", DirectML device 0" : ", CPU 2 threads") + ", batch " +
         std::to_string(batch);
  auto net = std::make_unique<OrtNet>(sess, mem, batch, device);
  // One run now: a session that cannot run fails here and not on the first
  // line, and DirectML allocates its work buffers.
  std::vector<float> x(static_cast<size_t>(batch) * kWindow),
      emo(static_cast<size_t>(batch) * kNumEmo),
      y(static_cast<size_t>(batch) * kNumOut);
  if (!net->Run(x.data(), emo.data(), batch, y.data(), err)) return nullptr;
  return net;
}

GpuState QueryGpu() {
  GpuState g;
  IDXGIFactory1* factory = nullptr;
  if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1),
                                reinterpret_cast<void**>(&factory))))
    return g;
  IDXGIAdapter1* adapter = nullptr;
  for (UINT i = 0; factory->EnumAdapters1(i, &adapter) == S_OK; ++i) {
    DXGI_ADAPTER_DESC1 d{};
    adapter->GetDesc1(&d);
    char name[256] = "";
    WideCharToMultiByte(CP_UTF8, 0, d.Description, -1, name, sizeof name,
                        nullptr, nullptr);
    const double mb = d.DedicatedVideoMemory / 1048576.0;
    if (i == 0) {
      g.adapter = name;
      g.dedicated_mb = mb;
      IDXGIAdapter3* adapter3 = nullptr;
      DXGI_QUERY_VIDEO_MEMORY_INFO vm{};
      if (SUCCEEDED(adapter->QueryInterface(
              __uuidof(IDXGIAdapter3), reinterpret_cast<void**>(&adapter3))) &&
          SUCCEEDED(adapter3->QueryVideoMemoryInfo(
              0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &vm))) {
        g.budget_mb = vm.Budget / 1048576.0;
        g.used_mb = vm.CurrentUsage / 1048576.0;
        g.known = vm.Budget != 0;
      }
      if (adapter3) adapter3->Release();
    } else if (!(d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
               mb > std::max(g.dedicated_mb, g.bigger_mb)) {
      g.bigger_adapter = name;
      g.bigger_mb = mb;
    }
    adapter->Release();
  }
  factory->Release();
  return g;
}

}  // namespace a2f
