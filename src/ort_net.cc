// ONNX Runtime session wrapper; session options are tuned for minimum VRAM.
#include "ort_net.h"

#define PSAPI_VERSION 2  // K32EnumProcessModules, in kernel32
#include <d3d12.h>
#include <dxgi1_4.h>
#include <windows.h>
#include <psapi.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <cwctype>
#include <initializer_list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "live_core.h"
#include "onnxruntime_c_api.h"

// dml_provider_factory.h needs DirectML.h (MSVC SAL annotations); the entries
// not used are placeholders.
struct OrtDmlApi {
  OrtStatus*(ORT_API_CALL* SessionOptionsAppendExecutionProvider_DML)(
      OrtSessionOptions*, int device_id);
  OrtStatus*(ORT_API_CALL* SessionOptionsAppendExecutionProvider_DML1)(
      OrtSessionOptions*, IUnknown* dml_device, ID3D12CommandQueue* queue);
  void* CreateGPUAllocationFromD3DResource;
  void* FreeGPUAllocation;
  void* GetD3D12ResourceFromAllocation;
  void* SessionOptionsAppendExecutionProvider_DML2;
  // The DirectML device (IDMLDevice, no reference added) the options'
  // provider was made with.
  OrtStatus*(ORT_API_CALL* GetDMLDevice)(OrtSessionOptions*, IUnknown** device);
  OrtStatus*(ORT_API_CALL* GetDMLCommandQueue)(OrtSessionOptions*,
                                             ID3D12CommandQueue** queue);
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

bool CheckHr(HRESULT hr, std::string& err, const char* what) {
  if (SUCCEEDED(hr)) return true;
  char code[16];
  snprintf(code, sizeof code, "%08lX", static_cast<unsigned long>(hr));
  err = std::string(what) + ": " + code;
  return false;
}

// Owns a reference to a COM object.
struct ComRelease {
  void operator()(IUnknown* p) const { p->Release(); }
};
template <typename T>
using ComPtr = std::unique_ptr<T, ComRelease>;

// The object's interface T, or null when it has none.
template <typename T>
ComPtr<T> As(IUnknown* object) {
  T* out = nullptr;
  if (FAILED(object->QueryInterface(__uuidof(T),
                                    reinterpret_cast<void**>(&out))))
    out = nullptr;
  return ComPtr<T>(out);
}

ComPtr<IDXGIFactory1> DxgiFactory() {
  IDXGIFactory1* factory = nullptr;
  if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1),
                                reinterpret_cast<void**>(&factory))))
    factory = nullptr;
  return ComPtr<IDXGIFactory1>(factory);
}

// Owns an onnxruntime object.
struct OrtRelease {
  void operator()(OrtSessionOptions* p) const {
    api->ReleaseSessionOptions(p);
  }
  void operator()(OrtSession* p) const { api->ReleaseSession(p); }
  void operator()(OrtMemoryInfo* p) const { api->ReleaseMemoryInfo(p); }
  void operator()(OrtValue* p) const { api->ReleaseValue(p); }
};
template <typename T>
using OrtPtr = std::unique_ptr<T, OrtRelease>;

// An export of d3d12.dll (not imported: the plugin loads without it).
FARPROC D3d12Export(const char* id) {
  static const HMODULE d3d12 = LoadLibraryW(L"d3d12.dll");
  return d3d12 ? GetProcAddress(d3d12, id) : nullptr;
}

// Keep the API-provided identity unless Debug/UnwrapDevice is explicitly set.
std::atomic<bool> use_wrapper{true};

// Retain each wrapper until all session/device references have been released.
using Wrappers = std::vector<ComPtr<IUnknown>>;
thread_local Wrappers* wrappers_kept = nullptr;

class ScopedWrappers {
 public:
  explicit ScopedWrappers(Wrappers& wrappers) : previous_(wrappers_kept) {
    wrappers_kept = &wrappers;
  }
  ~ScopedWrappers() { wrappers_kept = previous_; }
  ScopedWrappers(const ScopedWrappers&) = delete;
  ScopedWrappers& operator=(const ScopedWrappers&) = delete;

 private:
  Wrappers* const previous_;
};

// Consumes GetDevice's reference. Only unwrap when this thread has an owner
// that can retain the wrapper. No import table or shared vtable is modified.
void SelectDevice(ComPtr<IUnknown> real, void** out) {
  IUnknown* const wrapper = static_cast<IUnknown*>(*out);
  if (real.get() != wrapper) {
    if (use_wrapper || !wrappers_kept) return;
    wrappers_kept->emplace_back(wrapper);
  } else {
    wrapper->Release();
  }
  *out = real.release();
}

// Keep the API-provided device by default. The explicit debug opt-out uses
// a fence's parent to select a native device, retaining its original wrapper.
HRESULT WINAPI CreateSelectedDevice(IUnknown* adapter, D3D_FEATURE_LEVEL level,
                                REFIID riid, void** out) {
  using Fn = HRESULT(WINAPI*)(IUnknown*, D3D_FEATURE_LEVEL, REFIID, void**);
  static const auto create = reinterpret_cast<Fn>(
      reinterpret_cast<void*>(D3d12Export("D3D12CreateDevice")));
  if (!create) return E_NOINTERFACE;
  const HRESULT hr = create(adapter, level, riid, out);
  if (FAILED(hr) || !out) return hr;  // no out: only asked whether it can
  if (use_wrapper) return hr;  // Keep the API-provided identity by default.
  IUnknown* const got = static_cast<IUnknown*>(*out);
  const ComPtr<ID3D12Device> device = As<ID3D12Device>(got);
  ID3D12Fence* made = nullptr;
  if (!device || FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                                            __uuidof(ID3D12Fence),
                                            reinterpret_cast<void**>(&made))))
    return hr;
  const ComPtr<ID3D12Fence> fence(made);
  void* real = nullptr;
  if (FAILED(fence->GetDevice(riid, &real))) return hr;
  SelectDevice(ComPtr<IUnknown>(static_cast<IUnknown*>(real)), out);
  return hr;
}

// Adapter 0's Direct3D 12 device: one per adapter per process, so the very
// device DirectML runs on. Null without Direct3D 12. Its wrapper, if any, is
// kept by the caller's ScopedWrappers.
ComPtr<ID3D12Device> OpenDevice() {
  const ComPtr<IDXGIFactory1> factory = DxgiFactory();
  IDXGIAdapter* first = nullptr;
  if (!factory || FAILED(factory->EnumAdapters(0, &first))) return nullptr;
  const ComPtr<IDXGIAdapter> adapter(first);
  ID3D12Device* device = nullptr;
  if (FAILED(CreateSelectedDevice(adapter.get(), D3D_FEATURE_LEVEL_11_0,
                              __uuidof(ID3D12Device),
                              reinterpret_cast<void**>(&device))))
    device = nullptr;
  return ComPtr<ID3D12Device>(device);
}

// IDMLDevice::GetDeviceRemovedReason (DirectML.h is not included: the 16th
// entry of the interface, after IUnknown's 3 and IDMLObject's 4).
HRESULT DmlRemovedReason(IUnknown* dml) {
  using Fn = HRESULT(STDMETHODCALLTYPE*)(IUnknown*);
  return dml ? (*reinterpret_cast<Fn**>(dml))[15](dml) : S_OK;
}

// " [device removed reason XXXXXXXX]" when the Direct3D device was lost and
// " [DirectML removed reason XXXXXXXX]" when DirectML's own device stopped
// (it does so by itself when one of its calls fails inside), for
// DiagnoseNet; empty otherwise.
std::string RemovedReason(ID3D12Device* device, IUnknown* dml) {
  std::string out;
  char b[56];
  HRESULT r = device ? device->GetDeviceRemovedReason() : S_OK;
  if (r != S_OK) {
    snprintf(b, sizeof b, " [device removed reason %08lX]",
             static_cast<unsigned long>(r));
    out += b;
  }
  r = DmlRemovedReason(dml);
  if (r != S_OK) {
    snprintf(b, sizeof b, " [DirectML removed reason %08lX]",
             static_cast<unsigned long>(r));
    out += b;
  }
  return out;
}

class OrtNet : public Net {
 public:
  // device, dml (may be null): held to ask why they were lost when a run
  // fails.
  OrtNet(OrtPtr<OrtSession> sess, OrtPtr<OrtMemoryInfo> mem, int batch,
         ComPtr<ID3D12Device> device, ComPtr<IUnknown> dml,
         ComPtr<ID3D12CommandQueue> queue, Wrappers wrappers)
      : wrappers_(std::move(wrappers)),
        device_(std::move(device)),
        dml_(std::move(dml)),
        queue_(std::move(queue)),
        mem_(std::move(mem)),
        sess_(std::move(sess)),
        batch_(batch) {}
  int Batch() const override { return batch_; }
  bool Run(const float* x, const float* emo, int b, float* out,
           std::string& err) override {
    // null (with err) when it cannot be made
    const auto tensor = [&](const float* data, int width, const char* what) {
      const int64_t shape[3] = {b, 1, width};
      OrtValue* value = nullptr;
      if (!Check(api->CreateTensorWithDataAsOrtValue(
                     mem_.get(), const_cast<float*>(data),
                     static_cast<size_t>(b) * width * sizeof(float), shape, 3,
                     ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &value),
                 err, what))
        value = nullptr;
      return OrtPtr<OrtValue>(value);
    };
    const OrtPtr<OrtValue> vx = tensor(x, kWindow, "input");
    if (!vx) return false;
    const OrtPtr<OrtValue> ve = tensor(emo, kNumEmo, "emotion");
    if (!ve) return false;
    const char* in[] = {"input", "emotion"};
    const char* outn[] = {"result"};
    const OrtValue* ins[] = {vx.get(), ve.get()};
    OrtValue* result = nullptr;
    if (!Check(api->Run(sess_.get(), nullptr, in, ins, 2, outn, 1, &result),
               err, "run")) {
      failed = true;
      err += RemovedReason(device_.get(), dml_.get());
      return false;
    }
    const OrtPtr<OrtValue> vy(result);
    float* p = nullptr;
    if (!Check(api->GetTensorMutableData(vy.get(),
                                         reinterpret_cast<void**>(&p)),
               err, "output"))
      return false;
    memcpy(out, p, static_cast<size_t>(b) * kNumOut * sizeof(float));
    return true;
  }

 private:
  // Destroyed in reverse: the session first, then the device, at once (a
  // lost device held on would be handed out again to the next session), and
  // the wrappers it came out of last.
  Wrappers wrappers_;
  ComPtr<ID3D12Device> device_;
  ComPtr<IUnknown> dml_;
  ComPtr<ID3D12CommandQueue> queue_;
  OrtPtr<OrtMemoryInfo> mem_;
  OrtPtr<OrtSession> sess_;
  int batch_;
};

// The folded v3.0 network: one window per run.
class OrtNet3 : public Net {
 public:
  // device, dml: as OrtNet's.
  OrtNet3(OrtPtr<OrtSession> sess, OrtPtr<OrtMemoryInfo> mem,
          ComPtr<ID3D12Device> device, ComPtr<IUnknown> dml,
         ComPtr<ID3D12CommandQueue> queue, Wrappers wrappers)
      : wrappers_(std::move(wrappers)),
        device_(std::move(device)),
        dml_(std::move(dml)),
        queue_(std::move(queue)),
        mem_(std::move(mem)),
        sess_(std::move(sess)) {}
  int Version() const override { return 3; }
  int Batch() const override { return 1; }
  bool Run(const float*, const float*, int, float*, std::string& err) override {
    err = "not a v2.3 network";
    return false;
  }
  bool Run3(const float* window, const float* identity, const float* emotion,
            const float* state, float* skin, float* tongue, float* jaw_eyes,
            float* next_state, std::string& err) override {
    constexpr int kIn = 4, kOut = 4;
    static const int64_t kShapes[kIn][4] = {{1, kWindow3},
                                            {1, kNumIdentity3},
                                            {1, kFrames3, kNumMod},
                                            {2, 2, 1, 256}};
    static const size_t kRanks[kIn] = {2, 2, 3, 4};
    static const size_t kSizes[kIn] = {kWindow3, kNumIdentity3,
                                       kFrames3 * kNumMod, kState3};
    static const char* const kInNames[kIn] = {"window", "identity", "emotion",
                                              "input_latents"};
    static const char* const kOutNames[kOut] = {"skin_latent", "tongue_latent",
                                                "jaw_eyes", "output_latents"};
    const float* data[kIn] = {window, identity, emotion, state};
    float* dst[kOut] = {skin, tongue, jaw_eyes, next_state};
    const size_t out_size[kOut] = {kFrames3 * kLatent3, kFrames3 * kLatent3,
                                   kFrames3 * kJawEyes3, kState3};
    OrtPtr<OrtValue> in[kIn];
    const OrtValue* ins[kIn] = {};
    for (int i = 0; i < kIn; ++i) {
      OrtValue* value = nullptr;
      if (!Check(api->CreateTensorWithDataAsOrtValue(
                     mem_.get(), const_cast<float*>(data[i]),
                     kSizes[i] * sizeof(float), kShapes[i], kRanks[i],
                     ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &value),
                 err, kInNames[i]))
        return false;
      in[i].reset(value);
      ins[i] = value;
    }
    OrtValue* results[kOut] = {};
    const bool ran = Check(api->Run(sess_.get(), nullptr, kInNames, ins, kIn,
                                    kOutNames, kOut, results),
                           err, "run");
    OrtPtr<OrtValue> out[kOut];
    for (int i = 0; i < kOut; ++i) out[i].reset(results[i]);
    if (!ran) {
      failed = true;
      err += RemovedReason(device_.get(), dml_.get());
      return false;
    }
    for (int i = 0; i < kOut; ++i) {
      float* p = nullptr;
      if (!Check(api->GetTensorMutableData(out[i].get(),
                                           reinterpret_cast<void**>(&p)),
                 err, kOutNames[i]))
        return false;
      memcpy(dst[i], p, out_size[i] * sizeof(float));
    }
    return true;
  }

 private:
  // As OrtNet's: the session goes first, the wrappers last.
  Wrappers wrappers_;
  ComPtr<ID3D12Device> device_;
  ComPtr<IUnknown> dml_;
  ComPtr<ID3D12CommandQueue> queue_;
  OrtPtr<OrtMemoryInfo> mem_;
  OrtPtr<OrtSession> sess_;
};

void (*ort_log)(const std::string&);

// onnxruntime's own warnings and errors (DirectML's failures while a session
// loads are among them).
void ORT_API_CALL OrtLog(void*, OrtLoggingLevel severity, const char* category,
                         const char*, const char* location,
                         const char* message) {
  static std::atomic<int> lines{0};
  // each GPU session says this three times; it is harmless
  if (message && strstr(message, "can't constant fold")) return;
  if (!ort_log || ++lines > 200) return;
  ort_log(std::string("onnxruntime ") +
          (severity >= ORT_LOGGING_LEVEL_ERROR ? "error" : "warning") + " [" +
          (category ? category : "") + "] " + (location ? location : "") +
          ": " + (message ? message : ""));
}

// GPU-only load. ORT delay-loads DirectML by name, so load the packaged
// module before touching its GPU provider. Never load or probe it on CPU.
// DMLCreateDevice1(device, DML_CREATE_DEVICE_FLAGS, DML_FEATURE_LEVEL, ...).
using CreateDml = HRESULT(WINAPI*)(ID3D12Device*, UINT, UINT, REFIID, void**);
CreateDml create_dml;
// IDMLDevice and DML_FEATURE_LEVEL_5_0 from DirectML.h, matching ORT 1.24.
constexpr GUID kIidDmlDevice = {
    0x6dbd6437,
    0x96fd,
    0x423f,
    {0xa9, 0x8c, 0xae, 0x5e, 0x7c, 0x2a, 0x57, 0x3f}};
constexpr UINT kDmlFeatureLevel5_0 = 0x5000;
bool LoadDirectMl(const std::wstring& dll_dir, std::string& err) {
  if (create_dml) return true;
  const HMODULE module = LoadLibraryExW((dll_dir + L"\\DirectML.dll").c_str(),
                                       nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
  if (!module) {
    err = "cannot load DirectML.dll (error " + std::to_string(GetLastError()) + ")";
    return false;
  }
  create_dml = reinterpret_cast<CreateDml>(
      reinterpret_cast<void*>(GetProcAddress(module, "DMLCreateDevice1")));
  if (!create_dml) {
    FreeLibrary(module);
    err = "DirectML.dll: DMLCreateDevice1 unavailable";
    return false;
  }
  // Successful modules stay loaded as long as the worker and its sessions.
  return true;
}

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
  if (!Check(api->CreateEnvWithCustomLogger(OrtLog, nullptr,
                                            ORT_LOGGING_LEVEL_WARNING,
                                            "BetterFacialAnimations", &env),
             err, "env")) {
    api = nullptr;
    FreeLibrary(h);
    return false;
  }
  // The successful module and environment live as long as the worker.
  version = base()->GetVersionString();
  return true;
}

// What a session is made of. Destroyed in reverse: the session first, the
// queue/device after it and the wrappers the devices came out of last.
struct Session {
  Wrappers wrappers;  // of device and of the device onnxruntime made
  ComPtr<ID3D12Device> device;  // held to ask why it was lost; null on the CPU
  ComPtr<IUnknown> dml_device;  // the session's DirectML device, or null
  ComPtr<ID3D12CommandQueue> queue;
  OrtPtr<OrtMemoryInfo> mem;
  OrtPtr<OrtSession> sess;
};

// A session of model on DirectML (adapter 0, with the driver's metacommands
// or without) or the CPU, with the free dimensions dims fixed to batch: let
// DirectML plan one static shape (also 4x faster at batch 1-2). False (with err, and
// out empty) when it cannot be made.
bool CreateSession(const std::wstring& model, bool gpu, bool metacommands,
                   std::initializer_list<const char*> dims, int batch,
                   Session& out, std::string& err) {
  Session s;
  const ScopedWrappers keep(s.wrappers);
  OrtSessionOptions* options = nullptr;
  bool ok = Check(api->CreateSessionOptions(&options), err, "options");
  const OrtPtr<OrtSessionOptions> so(options);
  if (ok && gpu) {
    const OrtDmlApi* dml = nullptr;
    ok = Check(api->DisableMemPattern(so.get()), err, "mem pattern") &&
         Check(api->SetSessionExecutionMode(so.get(), ORT_SEQUENTIAL), err,
               "mode") &&
         Check(api->GetExecutionProviderApi(
                   "DML", ORT_API_VERSION,
                   reinterpret_cast<const void**>(&dml)),
               err, "DML api");
    if (ok && metacommands) {
      s.device = OpenDevice();
      if (!s.device) {
        err = "DirectML: cannot create D3D12 device";
        ok = false;
      }
      IUnknown* made = nullptr;
      if (ok)
        ok = CheckHr(create_dml(s.device.get(), 0, kDmlFeatureLevel5_0,
                                kIidDmlDevice, reinterpret_cast<void**>(&made)),
                     err, "DML device");
      s.dml_device.reset(made);
      D3D12_COMMAND_QUEUE_DESC desc{};
      desc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
      desc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
      // No DISABLE_GPU_TIMEOUT: preserve Windows' normal recovery policy.
      ID3D12CommandQueue* queue = nullptr;
      if (ok) ok = CheckHr(s.device->CreateCommandQueue(
          &desc, __uuidof(ID3D12CommandQueue), reinterpret_cast<void**>(&queue)),
          err, "DML compute queue");
      s.queue.reset(queue);
      if (ok) {
        s.queue->SetName(L"Better Facial Animations inference");
        // ORT validates the COM identities of the queue and DML parents.
        ok = Check(dml->SessionOptionsAppendExecutionProvider_DML1(
                       so.get(), s.dml_device.get(), s.queue.get()), err,
                   "DirectML explicit device/queue");
      }
    } else if (ok) {
      // ORT 1.24's DML1 API cannot disable metacommands. Preserve one bounded
      // supported fallback; ORT owns this device/queue, with no import patch
      // or unwrapping. Never mix it with objects from the explicit path.
      const char* const keys[] = {"device_id", "disable_metacommands"};
      const char* const values[] = {"0", "true"};
      ok = Check(api->SessionOptionsAppendExecutionProvider(so.get(), "DML",
                                                            keys, values, 2),
                 err, "DirectML without metacommands");
      IUnknown* device = nullptr;
      ID3D12CommandQueue* queue = nullptr;
      if (ok)
        ok = Check(dml->GetDMLDevice(so.get(), &device), err, "DML device query");
      if (ok)
        ok = Check(dml->GetDMLCommandQueue(so.get(), &queue), err, "DML queue query");
      if (ok && (!device || !queue)) {
        err = "DirectML: provider returned no device or queue";
        ok = false;
      }
      if (ok) {
        device->AddRef();
        s.dml_device.reset(device);
        queue->AddRef();
        s.queue.reset(queue);
        ID3D12Device* parent = nullptr;
        ok = CheckHr(queue->GetDevice(__uuidof(ID3D12Device),
                                     reinterpret_cast<void**>(&parent)), err, "DML queue parent");
        s.device.reset(parent);
      }
    }
    // DML's memory arena stays on: it keeps the work buffers of a run
    // allocated between runs (~16 MB more resident than freeing them). This
    // avoids reallocation; Windows can still evict resident GPU memory.
    for (const char* name : dims) {
      if (ok) {
        ok = Check(api->AddFreeDimensionOverrideByName(so.get(), name, batch),
                   err, "shape override");
      }
    }
  } else if (ok) {
    // Not ORT_ENABLE_ALL: its layout optimizations fail on the casts the CPU
    // provider inserts around the FP16 network's operators
    // (SimplifiedLayerNormFusion looks up a node argument that is gone).
    ok = Check(api->SetIntraOpNumThreads(so.get(), 2), err, "threads") &&
         Check(api->SetInterOpNumThreads(so.get(), 1), err, "threads") &&
         Check(api->SetSessionGraphOptimizationLevel(so.get(),
                                                     ORT_ENABLE_EXTENDED),
               err, "optimization");
  }
  if (ok) {
    OrtSession* sess = nullptr;
    ok = Check(api->CreateSession(env, model.c_str(), so.get(), &sess), err,
               "session");
    s.sess.reset(sess);
  }
  if (ok) {
    OrtMemoryInfo* mem = nullptr;
    ok = Check(
        api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem),
        err, "memory info");
    s.mem.reset(mem);
  }
  if (!ok) {
    err += RemovedReason(s.device.get(), s.dml_device.get());
    return false;
  }
  out = std::move(s);
  return true;
}

}  // namespace

std::unique_ptr<Net> CreateOrtNet(const std::wstring& dll_dir,
                                  const std::wstring& model,
                                  const OrtOptions& o, std::string& err,
                                  std::string& info) {
  if (o.gpu && !LoadDirectMl(dll_dir, err)) return nullptr;
  if (!LoadApi(dll_dir, err)) return nullptr;
  const int batch = std::max(1, o.batch);
  Session s;
  // The batch is always the same.
  if (!CreateSession(model, o.gpu, o.metacommands,
                     {"input_dynamic_axes_1", "emotion_dynamic_axes_1"}, batch,
                     s, err))
    return nullptr;
  info = std::string("Audio2Face-3D v2.3, onnxruntime ") + version +
         (!o.gpu ? ", CPU 2 threads"
          : o.metacommands ? ", DirectML device 0, explicit compute queue, metacommands on"
                           : ", DirectML device 0, provider-owned queue, metacommands off") +
         ", batch " + std::to_string(batch);
  auto net = std::make_unique<OrtNet>(std::move(s.sess), std::move(s.mem),
                                      batch, std::move(s.device),
                                      std::move(s.dml_device), std::move(s.queue),
                                      std::move(s.wrappers));
  // One run now: a session that cannot run fails here and not on the first
  // line, and DirectML allocates its work buffers.
  std::vector<float> x(static_cast<size_t>(batch) * kWindow),
      emo(static_cast<size_t>(batch) * kNumEmo),
      y(static_cast<size_t>(batch) * kNumOut);
  if (!net->Run(x.data(), emo.data(), batch, y.data(), err)) return nullptr;
  return net;
}

std::unique_ptr<Net> CreateOrtNet3(const std::wstring& dll_dir,
                                   const std::wstring& model,
                                   const OrtOptions& o, std::string& err,
                                   std::string& info) {
  if (o.gpu && !LoadDirectMl(dll_dir, err)) return nullptr;
  if (!LoadApi(dll_dir, err)) return nullptr;
  Session s;
  if (!CreateSession(model, o.gpu, o.metacommands,
                     {"window_dynamic_axes_1", "identity_dynamic_axes_1",
                      "emotion_dynamic_axes_1", "input_latents_dynamic_axes_1"},
                     1, s, err))
    return nullptr;
  info = std::string("Audio2Face-3D v3.0, onnxruntime ") + version +
         (!o.gpu ? ", CPU 2 threads"
          : o.metacommands ? ", DirectML device 0, explicit compute queue, metacommands on"
                           : ", DirectML device 0, provider-owned queue, metacommands off");
  auto net = std::make_unique<OrtNet3>(std::move(s.sess), std::move(s.mem),
                                       std::move(s.device),
                                       std::move(s.dml_device), std::move(s.queue),
                                       std::move(s.wrappers));
  // One run now, as CreateOrtNet's.
  std::vector<float> x(kWindow3), emo(kFrames3 * kNumMod), state(kState3),
      skin(kFrames3 * kLatent3), tongue(kFrames3 * kLatent3),
      jaw(kFrames3 * kJawEyes3), next(kState3);
  const float identity[kNumIdentity3] = {1, 0, 0};
  if (!net->Run3(x.data(), identity, emo.data(), state.data(), skin.data(),
                 tongue.data(), jaw.data(), next.data(), err))
    return nullptr;
  return net;
}

namespace {
std::string Utf8(const wchar_t* w) {
  char b[600] = "";
  WideCharToMultiByte(CP_UTF8, 0, w, -1, b, sizeof b, nullptr, nullptr);
  return b;
}

// "a.b.c.d" of a module file; empty when it has no version.
std::string FileVersion(const wchar_t* path) {
  // version.dll is not imported: the plugin loads without it
  static const HMODULE dll = LoadLibraryW(L"version.dll");
  using SizeFn = DWORD(WINAPI*)(const wchar_t*, DWORD*);
  using InfoFn = BOOL(WINAPI*)(const wchar_t*, DWORD, DWORD, void*);
  using QueryFn = BOOL(WINAPI*)(const void*, const wchar_t*, void**, UINT*);
  auto get = [](const char* name) {
    return dll ? reinterpret_cast<void*>(GetProcAddress(dll, name)) : nullptr;
  };
  static const auto size =
      reinterpret_cast<SizeFn>(get("GetFileVersionInfoSizeW"));
  static const auto info = reinterpret_cast<InfoFn>(get("GetFileVersionInfoW"));
  static const auto query = reinterpret_cast<QueryFn>(get("VerQueryValueW"));
  if (!size || !info || !query) return "";
  DWORD unused = 0;
  const DWORD n = size(path, &unused);
  std::vector<uint8_t> data(n);
  VS_FIXEDFILEINFO* f = nullptr;
  UINT len = 0;
  if (!n || !info(path, 0, n, data.data()) ||
      !query(data.data(), L"\\", reinterpret_cast<void**>(&f), &len) || !f)
    return "";
  char b[64];
  snprintf(b, sizeof b, "%u.%u.%u.%u", HIWORD(f->dwFileVersionMS),
           LOWORD(f->dwFileVersionMS), HIWORD(f->dwFileVersionLS),
           LOWORD(f->dwFileVersionLS));
  return b;
}
}  // namespace

void SetOrtLog(void (*log)(const std::string&)) { ort_log = log; }

void SetUseDeviceWrapper(bool use) { use_wrapper = use; }

// GpuReport's lines about the graphics: the adapters and their drivers and
// memory budget. No diagnostic D3D12 device is created.
static void ReportGraphics(std::vector<std::string>& out) {
  char b[900];
  // every adapter, with its driver version
  if (const ComPtr<IDXGIFactory1> factory = DxgiFactory()) {
    IDXGIAdapter1* next = nullptr;
    for (UINT i = 0; factory->EnumAdapters1(i, &next) == S_OK; ++i) {
      const ComPtr<IDXGIAdapter1> adapter(next);
      DXGI_ADAPTER_DESC1 d{};
      adapter->GetDesc1(&d);
      LARGE_INTEGER v{};
      char driver[64] = "unknown";
      if (SUCCEEDED(adapter->CheckInterfaceSupport(__uuidof(IDXGIDevice), &v)))
        snprintf(driver, sizeof driver, "%u.%u.%u.%u", HIWORD(v.HighPart),
                 LOWORD(v.HighPart), HIWORD(v.LowPart), LOWORD(v.LowPart));
      snprintf(b, sizeof b,
               "adapter %u: %s, vendor %04X device %04X, driver %s, %.0f MB "
               "dedicated%s",
               i, Utf8(d.Description).c_str(), d.VendorId, d.DeviceId, driver,
               d.DedicatedVideoMemory / 1048576.0,
               d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE ? ", software" : "");
      out.push_back(b);
    }
  }
  const GpuState g = QueryGpu();
  if (g.known) {
    snprintf(b, sizeof b,
             "VRAM on adapter 0: the game uses %.0f MB of the %.0f MB Windows "
             "grants it",
             g.used_mb, g.budget_mb);
    out.push_back(b);
  }
}

std::vector<std::string> GpuReport(bool probe_gpu) {
  std::vector<std::string> out;
  char b[900];
  // Windows (and Wine / Proton, where Direct3D 12 is vkd3d-proton)
  const HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
  RTL_OSVERSIONINFOW os{};
  os.dwOSVersionInfoSize = sizeof os;
  using VersionFn = LONG(WINAPI*)(RTL_OSVERSIONINFOW*);
  if (auto get = reinterpret_cast<VersionFn>(
          reinterpret_cast<void*>(GetProcAddress(ntdll, "RtlGetVersion"))))
    get(&os);
  using WineFn = const char*(CDECL*)();
  auto wine = reinterpret_cast<WineFn>(
      reinterpret_cast<void*>(GetProcAddress(ntdll, "wine_get_version")));
  MEMORYSTATUSEX ram{};
  ram.dwLength = sizeof ram;
  GlobalMemoryStatusEx(&ram);
  snprintf(b, sizeof b,
           "system: Windows %lu.%lu build %lu%s%s; memory %.0f MB free of "
           "%.0f MB, commit %.0f MB free of %.0f MB",
           os.dwMajorVersion, os.dwMinorVersion, os.dwBuildNumber,
           wine ? ", Wine " : "", wine ? wine() : "",
           ram.ullAvailPhys / 1048576.0, ram.ullTotalPhys / 1048576.0,
           ram.ullAvailPageFile / 1048576.0, ram.ullTotalPageFile / 1048576.0);
  out.push_back(b);
  if (probe_gpu)
    ReportGraphics(out);
  else
    out.push_back("GPU diagnostics skipped");
  // The modules in the game that are not Windows' own (mods, overlays,
  // wrappers of Direct3D) and those the model runs on, with their versions.
  wchar_t windows[MAX_PATH] = L"";
  const size_t wlen = GetWindowsDirectoryW(windows, MAX_PATH);
  std::vector<HMODULE> mods(4096);
  DWORD bytes = 0;
  if (EnumProcessModules(GetCurrentProcess(), mods.data(),
                         static_cast<DWORD>(mods.size() * sizeof(HMODULE)),
                         &bytes)) {
    mods.resize(std::min(mods.size(), bytes / sizeof(HMODULE)));
    static const wchar_t* const kAlways[] = {
        L"d3d12.dll",      L"d3d12core.dll",    L"dxgi.dll",
        L"d3d11.dll",      L"dxcore.dll",       L"directml.dll",
        L"msvcp140.dll",   L"msvcp140_1.dll",   L"vcruntime140.dll",
        L"vcruntime140_1.dll"};
    int listed = 0;
    for (HMODULE m : mods) {
      wchar_t path[MAX_PATH] = L"";
      if (!GetModuleFileNameW(m, path, MAX_PATH)) continue;
      const wchar_t* name = wcsrchr(path, L'\\');
      name = name ? name + 1 : path;
      bool show = _wcsnicmp(path, windows, wlen) != 0;
      for (const wchar_t* a : kAlways) show |= !_wcsicmp(name, a);
      // the graphics driver's own modules
      std::wstring lower = path;
      for (auto& c : lower) c = static_cast<wchar_t>(towlower(c));
      show |= lower.find(L"\\driverstore\\") != std::wstring::npos;
      if (!show) continue;
      if (++listed > 400) {
        out.push_back("module: (more not listed)");
        break;
      }
      const std::string v = FileVersion(path);
      out.push_back("module: " + Utf8(path) + (v.empty() ? "" : " " + v));
    }
  }
  return out;
}

GpuState QueryGpu() {
  GpuState g;
  const ComPtr<IDXGIFactory1> factory = DxgiFactory();
  if (!factory) return g;
  IDXGIAdapter1* next = nullptr;
  for (UINT i = 0; factory->EnumAdapters1(i, &next) == S_OK; ++i) {
    const ComPtr<IDXGIAdapter1> adapter(next);
    DXGI_ADAPTER_DESC1 d{};
    adapter->GetDesc1(&d);
    char name[256] = "";
    WideCharToMultiByte(CP_UTF8, 0, d.Description, -1, name, sizeof name,
                        nullptr, nullptr);
    const double mb = d.DedicatedVideoMemory / 1048576.0;
    if (i == 0) {
      g.adapter = name;
      g.dedicated_mb = mb;
      const ComPtr<IDXGIAdapter3> adapter3 =
          As<IDXGIAdapter3>(adapter.get());
      DXGI_QUERY_VIDEO_MEMORY_INFO vm{};
      if (adapter3 &&
          SUCCEEDED(adapter3->QueryVideoMemoryInfo(
              0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &vm))) {
        g.budget_mb = vm.Budget / 1048576.0;
        g.used_mb = vm.CurrentUsage / 1048576.0;
        g.known = vm.Budget != 0;
      }
    } else if (!(d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
               mb > std::max(g.dedicated_mb, g.bigger_mb)) {
      g.bigger_adapter = name;
      g.bigger_mb = mb;
    }
  }
  return g;
}

}  // namespace a2f
