// ONNX Runtime session wrapper; session options are tuned for minimum VRAM.
#include "a2f/inference/ort_net.h"
#include "a2f/inference/device_factory.h"

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

#include "a2f/inference/live_core.h"
#include "onnxruntime_c_api.h"

// dml_provider_factory.h needs DirectML.h (MSVC SAL annotations); the entries
// not used are placeholders.
struct OrtDmlApi {
  OrtStatus*(ORT_API_CALL* SessionOptionsAppendExecutionProvider_DML)(
      OrtSessionOptions*, int device_id);
  OrtStatus*(ORT_API_CALL* SessionOptionsAppendExecutionProvider_DML1)(
      OrtSessionOptions*, IUnknown*, ID3D12CommandQueue*);
  void* CreateGPUAllocationFromD3DResource;
  void* FreeGPUAllocation;
  void* GetD3D12ResourceFromAllocation;
  void* SessionOptionsAppendExecutionProvider_DML2;
  // The DirectML device (IDMLDevice, no reference added) the options'
  // provider was made with.
  OrtStatus*(ORT_API_CALL* GetDMLDevice)(OrtSessionOptions*, IUnknown** device);
  void* GetDMLCommandQueue;
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

// Set when D3D12CreateDevice handed out a wrapper of the device.
std::atomic<bool> device_wrapped{false};
// Set when onnxruntime.dll creates its device with CreateRealDevice.
bool device_redirected;
// The ini's UseDeviceWrapper: CreateRealDevice returns the wrapper as it is.
std::atomic<bool> use_wrapper{false};

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
// that can retain the wrapper, including when the import runs on another thread.
void SelectDevice(ComPtr<IUnknown> real, void** out) {
  IUnknown* const wrapper = static_cast<IUnknown*>(*out);
  if (real.get() != wrapper) {
    device_wrapped = true;
    if (use_wrapper || !wrappers_kept) return;
    wrappers_kept->emplace_back(wrapper);
  } else {
    wrapper->Release();
  }
  *out = real.release();
}

// D3D12CreateDevice, returning the device Direct3D itself made. ReShade hooks
// that function and hands out a wrapper of the device, on which DirectML
// loses the device on some systems (887A0005). A wrapper passes fence
// creation on to the device, and the fence names the device that made it.
HRESULT WINAPI CreateRealDevice(IUnknown* adapter, D3D_FEATURE_LEVEL level,
                                REFIID riid, void** out) {
  using Fn = HRESULT(WINAPI*)(IUnknown*, D3D_FEATURE_LEVEL, REFIID, void**);
  static const auto create = reinterpret_cast<Fn>(
      reinterpret_cast<void*>(D3d12Export("D3D12CreateDevice")));
  if (!create) return E_NOINTERFACE;
  const HRESULT hr = create(adapter, level, riid, out);
  if (FAILED(hr) || !out) return hr;  // no out: only asked whether it can
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

// Makes onnxruntime.dll create its Direct3D 12 device with CreateRealDevice,
// by replacing the entry of its delay-load import table for D3D12CreateDevice
// (imported by ordinal). False when it has no such import.
bool RedirectCreateDevice(HMODULE ort) {
  const FARPROC create = D3d12Export("D3D12CreateDevice");
  if (!create) return false;
  auto* base = reinterpret_cast<uint8_t*>(ort);
  auto* nt = reinterpret_cast<IMAGE_NT_HEADERS*>(
      base + reinterpret_cast<IMAGE_DOS_HEADER*>(base)->e_lfanew);
  const IMAGE_DATA_DIRECTORY& dir =
      nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT];
  if (!dir.VirtualAddress) return false;
  for (auto* d = reinterpret_cast<IMAGE_DELAYLOAD_DESCRIPTOR*>(
           base + dir.VirtualAddress);
       d->DllNameRVA; ++d) {
    if (_stricmp(reinterpret_cast<char*>(base + d->DllNameRVA), "d3d12.dll"))
      continue;
    auto* name =
        reinterpret_cast<IMAGE_THUNK_DATA*>(base + d->ImportNameTableRVA);
    auto* slot =
        reinterpret_cast<IMAGE_THUNK_DATA*>(base + d->ImportAddressTableRVA);
    for (; name->u1.AddressOfData; ++name, ++slot) {
      const char* const id =
          IMAGE_SNAP_BY_ORDINAL(name->u1.Ordinal)
              ? reinterpret_cast<const char*>(IMAGE_ORDINAL(name->u1.Ordinal))
              : reinterpret_cast<IMAGE_IMPORT_BY_NAME*>(
                    base + name->u1.AddressOfData)
                    ->Name;
      if (D3d12Export(id) != create) continue;
      DWORD protect;
      if (!VirtualProtect(slot, sizeof *slot, PAGE_READWRITE, &protect))
        return false;
      slot->u1.Function = reinterpret_cast<ULONGLONG>(&CreateRealDevice);
      VirtualProtect(slot, sizeof *slot, protect, &protect);
      return true;
    }
  }
  return false;
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
  if (FAILED(CreateRealDevice(adapter.get(), D3D_FEATURE_LEVEL_11_0,
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
         ComPtr<ID3D12Device> device, ComPtr<IUnknown> dml, Wrappers wrappers)
      : wrappers_(std::move(wrappers)),
        device_(std::move(device)),
        dml_(std::move(dml)),
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
  OrtPtr<OrtMemoryInfo> mem_;
  OrtPtr<OrtSession> sess_;
  int batch_;
};

// The folded v3.0 network: one window per run.
class OrtNet3 : public Net {
 public:
  // device, dml: as OrtNet's.
  OrtNet3(OrtPtr<OrtSession> sess, OrtPtr<OrtMemoryInfo> mem,
          ComPtr<ID3D12Device> device, ComPtr<IUnknown> dml, Wrappers wrappers)
      : wrappers_(std::move(wrappers)),
        device_(std::move(device)),
        dml_(std::move(dml)),
        mem_(std::move(mem)),
        sess_(std::move(sess)) {}
  int Version() const override { return kNetV3; }
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
  OrtPtr<OrtMemoryInfo> mem_;
  OrtPtr<OrtSession> sess_;
};

class PerformanceNet : public Net {
 public:
  PerformanceNet(OrtPtr<OrtSession> sess, OrtPtr<OrtMemoryInfo> mem,
                 ComPtr<ID3D12Device> device, ComPtr<IUnknown> dml,
                 Wrappers wrappers)
      : wrappers_(std::move(wrappers)),
        device_(std::move(device)),
        dml_(std::move(dml)),
        mem_(std::move(mem)),
        sess_(std::move(sess)) {}
  int Version() const override { return kNetPerformance; }
  int Batch() const override { return 1; }
  bool Run(const float*, const float*, int, float*,
           std::string& err) override {
    err = "not a v2.3 network";
    return false;
  }
  bool RunPerformance(const float* mel, int64_t identity, const float* emotion,
                      float* motion, std::string& err) override {
    constexpr int kNumInputs = 3;
    const int64_t mel_shape[] = {1, kPerformanceWindowFrames, kMelBins};
    const int64_t identity_shape[] = {1};
    const int64_t emotion_shape[] = {1, kPerformanceWindowFrames, kNumMod};
    const void* data[kNumInputs] = {mel, &identity, emotion};
    const size_t bytes[kNumInputs] = {
        sizeof(float) * kPerformanceWindowFrames * kMelBins, sizeof identity,
        sizeof(float) * kPerformanceWindowFrames * kNumMod};
    const int64_t* shapes[kNumInputs] = {mel_shape, identity_shape,
                                         emotion_shape};
    const size_t ranks[kNumInputs] = {3, 1, 3};
    const ONNXTensorElementDataType types[kNumInputs] = {
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT};
    const char* input_names[kNumInputs] = {"mel", "identity", "emotion"};
    const char* output_names[] = {"normalized_motion"};
    OrtPtr<OrtValue> inputs[kNumInputs];
    const OrtValue* input_values[kNumInputs];
    for (int i = 0; i < kNumInputs; ++i) {
      OrtValue* value = nullptr;
      if (!Check(api->CreateTensorWithDataAsOrtValue(
                     mem_.get(), const_cast<void*>(data[i]), bytes[i],
                     shapes[i], ranks[i], types[i], &value),
                 err, input_names[i]))
        return false;
      inputs[i].reset(value);
      input_values[i] = value;
    }
    OrtValue* raw_output = nullptr;
    const bool ran =
        Check(api->Run(sess_.get(), nullptr, input_names, input_values,
                       kNumInputs, output_names, 1, &raw_output),
              err, "performance run");
    OrtPtr<OrtValue> output(raw_output);
    if (!ran) {
      failed = true;
      err += RemovedReason(device_.get(), dml_.get());
      return false;
    }
    OrtTensorTypeAndShapeInfo* shape = nullptr;
    if (!Check(api->GetTensorTypeAndShape(output.get(), &shape), err,
               "performance shape"))
      return false;
    size_t count = 0;
    ONNXTensorElementDataType type;
    const bool described =
        Check(api->GetTensorShapeElementCount(shape, &count), err,
              "performance count") &&
        Check(api->GetTensorElementType(shape, &type), err,
              "performance type");
    api->ReleaseTensorTypeAndShapeInfo(shape);
    if (!described) return false;
    constexpr size_t kOutputCount =
        kPerformanceWindowFrames * kPerformanceChannels;
    if (count != kOutputCount || type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      err = "wrong performance model output shape/type";
      return false;
    }
    float* values = nullptr;
    if (!Check(api->GetTensorMutableData(output.get(),
                                         reinterpret_cast<void**>(&values)),
               err, "performance output"))
      return false;
    std::copy_n(values, kOutputCount, motion);
    return true;
  }

 private:
  // Session first on teardown, then device references and wrappers.
  Wrappers wrappers_;
  ComPtr<ID3D12Device> device_;
  ComPtr<IUnknown> dml_;
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

// Loads onnxruntime.dll from dll_dir, once.
HMODULE ort_module;
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
  ort_module = h;
  return true;
}

bool GraphicsCheck(HRESULT hr, const char* stage, std::string& err) {
  if (SUCCEEDED(hr)) return true;
  char code[16];
  snprintf(code, sizeof code, "%08lX", static_cast<unsigned long>(hr));
  err = std::string("independent device: ") + stage + " failed (" + code +
        "); CPU fallback; shared-device fallback disabled";
  return false;
}

// get is injectable for the offline factory contract. No global state is
// imported/applied, and no ordinary D3D12CreateDevice call is made here.
ComPtr<ID3D12Device> MakeIndependentDevice(independent::GetInterface get,
                                          std::string& err) {
  if (!get) {
    GraphicsCheck(E_NOINTERFACE, "D3D12GetInterface unavailable", err);
    return nullptr;
  }
  independent::Factory* raw = nullptr;
  const HRESULT found = get(independent::kFactoryClass,
                            independent::kFactoryInterface,
                            reinterpret_cast<void**>(&raw));
  const ComPtr<independent::Factory> factory(raw);
  if (!GraphicsCheck(found, "device factory", err)) return nullptr;
  if (!factory) {
    GraphicsCheck(E_NOINTERFACE, "null device factory", err);
    return nullptr;
  }
  if (!GraphicsCheck(factory->SetFlags(independent::kStrictFlags),
                      "strict isolation flags", err)) return nullptr;
  // Default adapter, as with the legacy adapter-0 provider. Let the factory
  // select it; diagnostics do not need to enumerate DXGI adapters first.
  ID3D12Device* raw_device = nullptr;
  const HRESULT made = factory->CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0,
      __uuidof(ID3D12Device), reinterpret_cast<void**>(&raw_device));
  ComPtr<ID3D12Device> device(raw_device);
  if (!GraphicsCheck(made, "CreateDevice (runtime/driver support required)", err))
    return nullptr;
  if (!device) GraphicsCheck(E_NOINTERFACE, "null independent device", err);
  return device;
}

// Only the inference worker accesses this context. Model sessions borrow COM
// references, but never own the last reference. Keep exactly one context until
// process exit, including across CPU fallback and idle model unloading. Do not
// run its destructor during DLL/process teardown: driver destruction can block
// Skyrim's renderer even when called on the inference worker.
struct IndependentContext {
  ComPtr<ID3D12Device> device;
  ComPtr<IUnknown> dml;
  ComPtr<ID3D12CommandQueue> queue;
  bool attempted = false;
  std::string failure;

  bool Reject(const std::string& reason, std::string& err) {
    failure = reason + "; GPU disabled until restart";
    err = failure;
    if (ort_log) ort_log(failure);
    return false;
  }

  bool Healthy(std::string& err) {
    if (!failure.empty()) { err = failure; return false; }
    const auto reason = RemovedReason(device.get(), dml.get());
    if (!reason.empty()) return Reject("independent context:" + reason, err);
    return true;
  }

  ComPtr<ID3D12Device> Acquire(independent::GetInterface get, std::string& err) {
    if (!Healthy(err)) return nullptr;
    if (!attempted) {
      attempted = true;
      device = MakeIndependentDevice(get, err);
      if (!device) { Reject(err, err); return nullptr; }
      if (ort_log) ort_log("independent GPU device retained until process exit");
    }
    device->AddRef();
    return ComPtr<ID3D12Device>(device.get());
  }
};

IndependentContext& IndependentGpu() {
  static auto* context = new IndependentContext;
  return *context;
}

ComPtr<ID3D12Device> OpenIndependentDevice(std::string& err) {
  return IndependentGpu().Acquire(reinterpret_cast<independent::GetInterface>(
      reinterpret_cast<void*>(D3d12Export("D3D12GetInterface"))), err);
}

// Initialize the shared DML device and queue once. Keep partial initialization
// too: retrying must not destroy a device or accumulate abandoned contexts.
bool InitializeIndependentProvider(IndependentContext& context,
                                   independent::CreateDml create,
                                   std::string& err) {
  if (!context.Healthy(err)) return false;
  if (context.queue) return true;
  if (!create) {
    GraphicsCheck(E_NOINTERFACE, "DMLCreateDevice unavailable", err);
    return context.Reject(err, err);
  }
  IUnknown* raw_dml = nullptr;
  const HRESULT made = create(context.device.get(), 0, independent::kDmlDevice,
                              reinterpret_cast<void**>(&raw_dml));
  context.dml.reset(raw_dml);
  if (!GraphicsCheck(made, "DMLCreateDevice", err))
    return context.Reject(err, err);
  if (!context.dml) {
    GraphicsCheck(E_NOINTERFACE, "null DirectML device", err);
    return context.Reject(err, err);
  }
  D3D12_COMMAND_QUEUE_DESC desc{};
  desc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
  desc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
  // Keep normal GPU timeout behavior. Do not request DISABLE_GPU_TIMEOUT.
  ID3D12CommandQueue* raw_queue = nullptr;
  const HRESULT queued = context.device->CreateCommandQueue(&desc,
      __uuidof(ID3D12CommandQueue), reinterpret_cast<void**>(&raw_queue));
  context.queue.reset(raw_queue);
  if (!GraphicsCheck(queued, "CreateCommandQueue", err))
    return context.Reject(err, err);
  if (!context.queue) {
    GraphicsCheck(E_NOINTERFACE, "null compute queue", err);
    return context.Reject(err, err);
  }
  if (ort_log) ort_log("independent DirectML device and compute queue retained");
  return true;
}

bool AppendIndependentProvider(const std::wstring& dll_dir,
                                const OrtDmlApi* dml, OrtSessionOptions* so,
                                ID3D12Device* device,
                                ComPtr<IUnknown>& dml_device, std::string& err) {
  auto& context = IndependentGpu();
  if (!context.Healthy(err)) return false;
  if (!dml || !dml->SessionOptionsAppendExecutionProvider_DML1) {
    GraphicsCheck(E_NOINTERFACE, "ONNX Runtime DML1 unavailable", err);
    return false;
  }
  // Retained for the process lifetime, as is the ORT module. Never change
  // the game's DLL search policy or globally selected D3D12 SDK version.
  static const HMODULE module = LoadLibraryExW(
      (dll_dir + L"\\DirectML.dll").c_str(), nullptr,
      LOAD_WITH_ALTERED_SEARCH_PATH);
  const auto create = module ? reinterpret_cast<independent::CreateDml>(
      reinterpret_cast<void*>(GetProcAddress(module, "DMLCreateDevice"))) : nullptr;
  if (!InitializeIndependentProvider(context, create, err)) return false;
  if (device != context.device.get()) {
    err = "independent provider: unexpected device";
    return false;
  }
  context.dml->AddRef();
  dml_device.reset(context.dml.get());
  // ORT retains these too; releasing a session frees its model and buffers,
  // while the process context keeps the device and queue alive.
  return Check(dml->SessionOptionsAppendExecutionProvider_DML1(
                   so, context.dml.get(), context.queue.get()), err,
               "independent device: DirectML DML1");
}

// What a session is made of. Destroyed in reverse: the session first, the
// borrowed device references after it and wrappers last. The independent
// context retains its own references regardless of this session's lifetime.
struct Session {
  Wrappers wrappers;  // of device and of the device onnxruntime made
  ComPtr<ID3D12Device> device;  // held to ask why it was lost; null on the CPU
  ComPtr<IUnknown> dml_device;  // the session's DirectML device, or null
  OrtPtr<OrtMemoryInfo> mem;
  OrtPtr<OrtSession> sess;
};

// A session of model on DirectML (adapter 0, with the driver's metacommands
// or without) or the CPU, with the free dimensions dims fixed to batch: let
// DirectML plan one static shape (also 4x faster at batch 1-2). False (with err, and
// out empty) when it cannot be made.
bool CreateSession(const std::wstring& model, bool gpu, bool metacommands,
                   bool independent_device, const std::wstring& dll_dir,
                   std::initializer_list<const char*> dims, int batch,
                   Session& out, std::string& err) {
  Session s;
  const ScopedWrappers keep(s.wrappers);
  if (gpu && independent_device) {
    if (!metacommands) {
      err = "independent device: DML1 does not support disabling metacommands";
      return false;
    }
    s.device = OpenIndependentDevice(err);
    if (!s.device) return false;
  } else if (gpu) {
    static bool redirect_tried;
    if (!redirect_tried) {
      redirect_tried = true;
      device_redirected = RedirectCreateDevice(ort_module);
      if (!device_redirected && ort_log)
        ort_log("onnxruntime.dll does not import D3D12CreateDevice as "
                "expected: a wrapper of Direct3D 12 (ReShade) stays in the "
                "model's way");
    }
    s.device = OpenDevice();
  }
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
    if (ok && independent_device) {
      ok = AppendIndependentProvider(dll_dir, dml, so.get(), s.device.get(),
                                     s.dml_device, err);
    } else if (ok && metacommands) {
      ok = Check(dml->SessionOptionsAppendExecutionProvider_DML(so.get(), 0),
                 err, "DirectML");
    } else if (ok) {
      // only the provider's key / value form can turn them off
      const char* const keys[] = {"device_id", "disable_metacommands"};
      const char* const values[] = {"0", "true"};
      ok = Check(api->SessionOptionsAppendExecutionProvider(so.get(), "DML",
                                                            keys, values, 2),
                 err, "DirectML");
    }
    // the session's DirectML device (handed out without a reference), to ask
    // it why it stopped
    IUnknown* dml_device = nullptr;
    if (ok && !independent_device &&
        !dml->GetDMLDevice(so.get(), &dml_device) && dml_device) {
      dml_device->AddRef();
      s.dml_device.reset(dml_device);
    }
    // DML's memory arena stays on: it keeps the work buffers of a run
    // allocated between runs (~16 MB more resident than freeing them), so a
    // loaded session never has to find VRAM again.
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

// The execution provider part of a session's info line.
std::string DescribeProvider(const OrtOptions& o) {
  std::string description =
      !o.gpu           ? ", CPU 2 threads"
      : o.metacommands ? ", DirectML device 0, metacommands on"
                       : ", DirectML device 0, metacommands off";
  if (o.gpu && o.independent_device)
    description += ", independent device, private compute queue";
  return description;
}

}  // namespace

std::unique_ptr<Net> CreateOrtNet(const std::wstring& dll_dir,
                                  const std::wstring& model,
                                  const OrtOptions& o, std::string& err,
                                  std::string& info) {
  if (!LoadApi(dll_dir, err)) return nullptr;
  const int batch = std::max(1, o.batch);
  Session s;
  // The batch is always the same.
  if (!CreateSession(model, o.gpu, o.metacommands,
                     o.independent_device, dll_dir,
                     {"input_dynamic_axes_1", "emotion_dynamic_axes_1"}, batch,
                     s, err))
    return nullptr;
  info = std::string("Audio2Face-3D v2.3, onnxruntime ") + version +
         DescribeProvider(o) + ", batch " + std::to_string(batch);
  auto net = std::make_unique<OrtNet>(std::move(s.sess), std::move(s.mem),
                                      batch, std::move(s.device),
                                      std::move(s.dml_device),
                                      std::move(s.wrappers));
  // One run now: a session that cannot run fails here and not on the first
  // line, and DirectML allocates its work buffers.
  std::vector<float> x(static_cast<size_t>(batch) * kWindow),
      emo(static_cast<size_t>(batch) * kNumEmo),
      y(static_cast<size_t>(batch) * kNumOut);
  if (!net->Run(x.data(), emo.data(), batch, y.data(), err)) return nullptr;
  return net;
}

std::unique_ptr<Net> CreatePerformanceNet(const std::wstring& dll_dir,
                                          const std::wstring& model,
                                          const OrtOptions& o, std::string& err,
                                          std::string& info) {
  if (!LoadApi(dll_dir, err)) return nullptr;
  Session s;
  if (!CreateSession(model, o.gpu, o.metacommands, o.independent_device,
                     dll_dir, {"batch"}, 1, s, err))
    return nullptr;
  auto net = std::make_unique<PerformanceNet>(
      std::move(s.sess), std::move(s.mem), std::move(s.device),
      std::move(s.dml_device), std::move(s.wrappers));
  // One run now, as CreateOrtNet's.
  std::vector<float> mel(kPerformanceWindowFrames * kMelBins),
      emotion(kPerformanceWindowFrames * kNumMod),
      motion(kPerformanceWindowFrames * kPerformanceChannels);
  if (!net->RunPerformance(mel.data(), 0, emotion.data(), motion.data(), err))
    return nullptr;
  info = std::string("Performance Thin120, 80,729 parameters, 30 fps") +
         DescribeProvider(o);
  return net;
}

std::unique_ptr<Net> CreateOrtNet3(const std::wstring& dll_dir,
                                   const std::wstring& model,
                                   const OrtOptions& o, std::string& err,
                                   std::string& info) {
  if (!LoadApi(dll_dir, err)) return nullptr;
  Session s;
  if (!CreateSession(model, o.gpu, o.metacommands,
                     o.independent_device, dll_dir,
                     {"window_dynamic_axes_1", "identity_dynamic_axes_1",
                      "emotion_dynamic_axes_1", "input_latents_dynamic_axes_1"},
                     1, s, err))
    return nullptr;
  info = std::string("Audio2Face-3D v3.0, onnxruntime ") + version +
         DescribeProvider(o);
  auto net = std::make_unique<OrtNet3>(std::move(s.sess), std::move(s.mem),
                                       std::move(s.device),
                                       std::move(s.dml_device),
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

bool DeviceRedirected() { return device_redirected; }

bool IndependentDeviceSupported(std::string& err) {
  err.clear();
  const bool supported = bool(OpenIndependentDevice(err));
  // The probe does not choose a fallback; its caller selects the policy.
  const auto policy = err.find("; CPU fallback;");
  if (policy != std::string::npos) err.resize(policy);
  return supported;
}

void SetUseDeviceWrapper(bool use) { use_wrapper = use; }
bool DeviceWrapped() { return device_wrapped; }

// GpuReport's lines about the graphics: the adapters and their drivers and
// what Direct3D 12 offers on adapter 0, for which a device is created.
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
  // what Direct3D 12 offers DirectML on adapter 0
  Wrappers wrappers;  // released after the device
  const ScopedWrappers keep(wrappers);
  if (const ComPtr<ID3D12Device> device = OpenDevice()) {
    const D3D_FEATURE_LEVEL levels[] = {
        D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_12_0,
        D3D_FEATURE_LEVEL_12_1};
    D3D12_FEATURE_DATA_FEATURE_LEVELS fl{};
    fl.NumFeatureLevels = 4;
    fl.pFeatureLevelsRequested = levels;
    device->CheckFeatureSupport(D3D12_FEATURE_FEATURE_LEVELS, &fl, sizeof fl);
    D3D12_FEATURE_DATA_D3D12_OPTIONS4 o4{};
    const bool has4 = SUCCEEDED(device->CheckFeatureSupport(
        D3D12_FEATURE_D3D12_OPTIONS4, &o4, sizeof o4));
    snprintf(b, sizeof b,
             "Direct3D 12 on adapter 0: feature level %X, 16-bit shader "
             "operations %s, device removed reason %08lX",
             static_cast<unsigned>(fl.MaxSupportedFeatureLevel),
             !has4 ? "unknown" : o4.Native16BitShaderOpsSupported ? "yes" : "no",
             static_cast<unsigned long>(device->GetDeviceRemovedReason()));
    out.push_back(b);
    const ComPtr<ID3D12Device5> device5 = As<ID3D12Device5>(device.get());
    UINT n = 0;
    if (device5 && SUCCEEDED(device5->EnumerateMetaCommands(&n, nullptr))) {
      std::vector<D3D12_META_COMMAND_DESC> descs(n);
      std::string names;
      if (n && SUCCEEDED(device5->EnumerateMetaCommands(&n, descs.data())))
        for (UINT i = 0; i < n && i < 60; ++i)
          names += (i ? ", " : ": ") +
                   (descs[i].Name ? Utf8(descs[i].Name) : std::string("?"));
      out.push_back("driver metacommands offered: " + std::to_string(n) +
                    names);
    } else {
      out.push_back("driver metacommands offered: unknown");
    }
  } else {
    out.push_back("Direct3D 12 on adapter 0: no device");
  }
  if (device_wrapped)
    out.push_back(std::string("Direct3D 12 hands out a wrapper of its device "
                              "(ReShade does): the model runs on ") +
                  (use_wrapper ? "the wrapper"
                               : "the device itself (the wrapper once a "
                                 "session on it is lost)"));
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
