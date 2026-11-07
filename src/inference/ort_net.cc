#include "a2f/inference/diagnostics.h"
// ONNX Runtime session wrapper; session options are tuned for minimum VRAM.
#include "a2f/inference/device_factory.h"
#include "a2f/inference/gpu_retry.h"
#include "a2f/inference/ort_net.h"
#include "a2f/inference/performance_features.h"
#include "a2f/inference/windows_prerequisite.h"

#define PSAPI_VERSION 2  // K32EnumProcessModules, in kernel32
#include <d3d12.h>
#include <dxgi1_4.h>
#include <psapi.h>
#include <windows.h>

#include "a2f/runtime/test_hooks.h"
#ifndef RRF_SUBKEY_WOW6464KEY
#define RRF_SUBKEY_WOW6464KEY 0x00010000
#endif

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <cwctype>
#include <initializer_list>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "a2f/inference/live_core.h"
#include "onnxruntime_c_api.h"

namespace {
HRESULT CreateFactory(REFIID iid, void** result) {
#ifdef ENABLE_COMMONLIBSSE_TESTING
  if (bfa::testing::dxgi_factory)
    return bfa::testing::dxgi_factory(iid, result);
#endif
  return CreateDXGIFactory1(iid, result);
}
}  // namespace

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
  if (api->GetErrorCode)
    err += " [ORT error " +
           std::to_string(static_cast<int>(api->GetErrorCode(s))) + "]";
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
  if (FAILED(
          object->QueryInterface(__uuidof(T), reinterpret_cast<void**>(&out))))
    out = nullptr;
  return ComPtr<T>(out);
}

ComPtr<IDXGIFactory1> DxgiFactory() {
  IDXGIFactory1* factory = nullptr;
  if (FAILED(CreateFactory(__uuidof(IDXGIFactory1),
                           reinterpret_cast<void**>(&factory))))
    factory = nullptr;
  return ComPtr<IDXGIFactory1>(factory);
}

// Owns an onnxruntime object.
struct OrtRelease {
  void operator()(OrtSessionOptions* p) const { api->ReleaseSessionOptions(p); }
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
std::atomic<bool> use_wrapper{false};
// Set when onnxruntime.dll creates its device with CreateRealDevice.
bool device_redirected;

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
// that can retain the wrapper, including when the import runs on another
// thread.
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
              : reinterpret_cast<IMAGE_IMPORT_BY_NAME*>(base +
                                                        name->u1.AddressOfData)
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
std::string NativeError(const char* operation, HRESULT hr) {
  char code[16];
  snprintf(code, sizeof code, "%08lX", static_cast<unsigned long>(hr));
  return std::string(operation) + " failed [HRESULT 0x" + code + "]";
}

ComPtr<IDXGIAdapter1> OpenAdapter(int index, std::string& error) {
  diagnostics::Progress(diagnostics::Stage::kAdapter);
  IDXGIFactory1* raw = nullptr;
  HRESULT hr =
      CreateFactory(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&raw));
  const ComPtr<IDXGIFactory1> factory(raw);
  if (FAILED(hr) || !factory) {
    error = NativeError("CreateDXGIFactory1", FAILED(hr) ? hr : E_POINTER);
    return nullptr;
  }
  IDXGIAdapter1* selected = nullptr;
  hr = index < 0 ? E_INVALIDARG : factory->EnumAdapters1(index, &selected);
  ComPtr<IDXGIAdapter1> adapter(selected);
  if (FAILED(hr) || !adapter) {
    error = "GPU " + std::to_string(index) + " unavailable: " +
            NativeError("EnumAdapters1", FAILED(hr) ? hr : E_POINTER);
    return nullptr;
  }
  DXGI_ADAPTER_DESC1 desc{};
  hr = adapter->GetDesc1(&desc);
  if (FAILED(hr) || (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) {
    error = "GPU " + std::to_string(index) +
            (FAILED(hr) ? ": " + NativeError("GetDesc1", hr)
                        : " is a software adapter");
    return nullptr;
  }
  return adapter;
}

ComPtr<ID3D12Device> OpenDevice(std::string* error = nullptr, int index = 0) {
  std::string detail;
  auto adapter = OpenAdapter(index, detail);
  const auto fail = [&](const char* operation, HRESULT code) {
    if (error) *error = "shared device: " + NativeError(operation, code);
  };
  if (!adapter) {
    if (error) *error = "shared device: " + detail;
    return nullptr;
  }
  diagnostics::Progress(diagnostics::Stage::kDevice);
  ID3D12Device* device = nullptr;
  const HRESULT hr = CreateRealDevice(adapter.get(), D3D_FEATURE_LEVEL_11_0,
                                      __uuidof(ID3D12Device),
                                      reinterpret_cast<void**>(&device));
  ComPtr<ID3D12Device> result(device);
  if (FAILED(hr) || !result) {
    fail("D3D12CreateDevice", FAILED(hr) ? hr : E_POINTER);
    return nullptr;
  }
  return result;
}

// Capture the loader error at the failing call, including on later retries.
struct RuntimeModule {
  HMODULE module;
  DWORD error;
  explicit RuntimeModule(const std::wstring& path)
      : module(LoadLibraryExW(path.c_str(), nullptr,
                              LOAD_WITH_ALTERED_SEARCH_PATH)),
        error(module ? ERROR_SUCCESS : GetLastError()) {}
};

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

// Validate before copying: malformed/non-finite output must never reach face
// geometry. A successful ORT Run alone does not establish usable output. A
// failed validation fails the line only: the session and device stay.
bool ReadFloats(OrtValue* value, size_t expected, float* dst, std::string& err,
                const char* name) {
  if (!value) {
    err = std::string(name) + ": missing tensor";
    return false;
  }
  OrtTensorTypeAndShapeInfo* shape = nullptr;
  if (!Check(api->GetTensorTypeAndShape(value, &shape), err, name))
    return false;
  size_t count = 0;
  ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  const bool valid =
      Check(api->GetTensorShapeElementCount(shape, &count), err, name) &&
      Check(api->GetTensorElementType(shape, &type), err, name);
  api->ReleaseTensorTypeAndShapeInfo(shape);
  if (!valid) return false;
  if (count != expected || type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    err = std::string(name) + ": wrong output shape/type";
    return false;
  }
  float* data = nullptr;
  if (!Check(api->GetTensorMutableData(value, reinterpret_cast<void**>(&data)),
             err, name))
    return false;
  if (!data || !std::all_of(data, data + count,
                            [](float f) { return std::isfinite(f); })) {
    err = std::string(name) + ": non-finite or missing output data";
    return false;
  }
  if (data != dst) std::copy_n(data, count, dst);
  return true;
}

bool CanRun(ID3D12Device* device, std::string& err) {
  return !device || GpuAllowed(err);
}

// What a session is made of. Destroyed in reverse: the session first, the
// borrowed device references after it and wrappers last. The independent
// context retains its own references regardless of this session's lifetime.
struct Session {
  bool independent = false, failed = false;
  Wrappers wrappers;            // of device and of the device onnxruntime made
  ComPtr<ID3D12Device> device;  // held to ask why it was lost; null on the CPU
  ComPtr<IUnknown> dml_device;  // the session's DirectML device, or null
  OrtPtr<OrtMemoryInfo> mem;
  OrtPtr<OrtSession> sess;
};

// Failed sessions release their resources before retrying. An independent
// session that lost its device latches the context instead: its resources
// stay quarantined until RetryGpu creates the context again.
void RunFailed(Session& s, std::string& err) {
  s.failed = true;
  if (!s.device) return;
  std::string blocked;
  if (!GpuAllowed(blocked)) {
    err = blocked;
    return;
  }
  err += RemovedReason(s.device.get(), s.dml_device.get());
  if (s.independent && IndependentContextFailure(err)) DisableGpu(err);
}
bool FailRun(Session& s, bool& failed, std::string& err) {
  failed = true;
  RunFailed(s, err);
  return false;
}
void RetainFailedSession(Session& s) {
  if (!s.device) return;
  std::string err;
  if (!GpuAllowed(err)) {
    (void)s.sess.release();
    (void)s.mem.release();
    (void)s.device.release();
    (void)s.dml_device.release();
    for (auto& wrapper : s.wrappers) (void)wrapper.release();
    return;
  }
  if (s.failed) {
    // Shared failures release their session before another GPU attempt.
    s.sess.reset();
    s.mem.reset();
    s.dml_device.reset();
    s.device.reset();
    s.wrappers.clear();
  }
}

class OrtNet : public Net {
 public:
  OrtNet(Session s, int batch) : s_(std::move(s)), batch_(batch) {}
  ~OrtNet() override { RetainFailedSession(s_); }
  int Batch() const override { return batch_; }
  bool Run(const float* x, const float* emo, int b, float* out,
           std::string& err) override {
    if (!CanRun(s_.device.get(), err)) return FailRun(s_, failed, err);
    // null (with err) when it cannot be made
    const auto tensor = [&](const float* data, int width, const char* what) {
      const int64_t shape[3] = {b, 1, width};
      OrtValue* value = nullptr;
      if (!Check(api->CreateTensorWithDataAsOrtValue(
                     s_.mem.get(), const_cast<float*>(data),
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
    if (!Check(api->Run(s_.sess.get(), nullptr, in, ins, 2, outn, 1, &result),
               err, "run"))
      return FailRun(s_, failed, err);
    const OrtPtr<OrtValue> vy(result);
    return ReadFloats(vy.get(), static_cast<size_t>(b) * kNumOut, out, err,
                      "output");
  }

 private:
  Session s_;
  int batch_;
};

// The folded v3.0 network: one window per run.
class OrtNet3 : public Net {
 public:
  explicit OrtNet3(Session s) : s_(std::move(s)) {}
  ~OrtNet3() override { RetainFailedSession(s_); }
  int Version() const override { return kNetV3; }
  int Batch() const override { return 1; }
  bool Run(const float*, const float*, int, float*, std::string& err) override {
    err = "not a v2.3 network";
    return false;
  }
  bool Run3(const float* window, const float* identity, const float* emotion,
            const float* state, float* skin, float* tongue, float* jaw_eyes,
            float* next_state, std::string& err) override {
    if (!CanRun(s_.device.get(), err)) return FailRun(s_, failed, err);
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
                     s_.mem.get(), const_cast<float*>(data[i]),
                     kSizes[i] * sizeof(float), kShapes[i], kRanks[i],
                     ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &value),
                 err, kInNames[i]))
        return false;
      in[i].reset(value);
      ins[i] = value;
    }
    OrtValue* results[kOut] = {};
    const bool ran = Check(api->Run(s_.sess.get(), nullptr, kInNames, ins, kIn,
                                    kOutNames, kOut, results),
                           err, "run");
    OrtPtr<OrtValue> out[kOut];
    for (int i = 0; i < kOut; ++i) out[i].reset(results[i]);
    if (!ran) return FailRun(s_, failed, err);
    for (int i = 0; i < kOut; ++i)
      if (!ReadFloats(out[i].get(), out_size[i], dst[i], err, kOutNames[i]))
        return false;
    return true;
  }

 private:
  Session s_;
};

class PerformanceNet : public Net {
 public:
  explicit PerformanceNet(Session s) : s_(std::move(s)) {}
  ~PerformanceNet() override { RetainFailedSession(s_); }
  int Version() const override { return kNetPerformance; }
  int Batch() const override { return 1; }
  bool Run(const float*, const float*, int, float*, std::string& err) override {
    err = "not a v2.3 network";
    return false;
  }
  bool RunPerformance(const float* mel, int64_t identity, const float* emotion,
                      const PerformanceState& state, float* motion,
                      PerformanceState& next, std::string& err) override {
    if (!CanRun(s_.device.get(), err)) return FailRun(s_, failed, err);
    if (identity < 0 || identity > 1 || state.frames < 0 ||
        state.frames > kPerformanceMemoryFrames ||
        state.keys.size() != kPerformanceStateSize ||
        state.values.size() != kPerformanceStateSize ||
        next.keys.size() != kPerformanceStateSize ||
        next.values.size() != kPerformanceStateSize) {
      err = "invalid performance state";
      return false;
    }
    constexpr int kNumInputs = 6, kNumOutputs = 3;
    const int64_t mel_shape[] = {1, kPerformanceWindowFrames, kMelBins};
    const int64_t scalar_shape[] = {1};
    const int64_t emotion_shape[] = {1, kPerformanceWindowFrames, kNumMod};
    const int64_t state_shape[] = {6, 1, 4, kPerformanceMemoryFrames, 64};
    const void* data[kNumInputs] = {mel, &identity, emotion, state.keys.data(),
                                   state.values.data(), &state.frames};
    const size_t bytes[kNumInputs] = {
        sizeof(float) * kPerformanceWindowFrames * kMelBins, sizeof identity,
        sizeof(float) * kPerformanceWindowFrames * kNumMod,
        sizeof(float) * kPerformanceStateSize,
        sizeof(float) * kPerformanceStateSize, sizeof state.frames};
    const int64_t* shapes[kNumInputs] = {mel_shape,     scalar_shape,
                                         emotion_shape, state_shape,
                                         state_shape,   scalar_shape};
    const size_t ranks[kNumInputs] = {3, 1, 3, 5, 5, 1};
    const ONNXTensorElementDataType types[kNumInputs] = {
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64};
    const char* input_names[kNumInputs] = {
        "mel", "identity", "emotion", "past_k", "past_v", "past_frames"};
    const char* output_names[kNumOutputs] = {"normalized_motion", "next_k",
                                             "next_v"};
    OrtPtr<OrtValue> inputs[kNumInputs];
    const OrtValue* input_values[kNumInputs];
    for (int i = 0; i < kNumInputs; ++i) {
      OrtValue* value = nullptr;
      if (!Check(api->CreateTensorWithDataAsOrtValue(
                     s_.mem.get(), const_cast<void*>(data[i]), bytes[i],
                     shapes[i], ranks[i], types[i], &value),
                 err, input_names[i]))
        return false;
      inputs[i].reset(value);
      input_values[i] = value;
    }
    const int64_t motion_shape[] = {1, kPerformanceHopFrames,
                                    kPerformanceChannels};
    const int64_t* output_shapes[] = {motion_shape, state_shape, state_shape};
    const size_t output_ranks[] = {3, 5, 5};
    const size_t output_counts[] = {
        kPerformanceHopFrames * kPerformanceChannels, kPerformanceStateSize,
        kPerformanceStateSize};
    float* output_data[] = {motion, next.keys.data(), next.values.data()};
    const auto overlaps = [](const void* a, size_t an, const void* b,
                             size_t bn) {
      const auto x = reinterpret_cast<uintptr_t>(a);
      const auto y = reinterpret_cast<uintptr_t>(b);
      return x <= y ? y - x < an : x - y < bn;
    };
    bool disjoint = true;
    for (int i = 0; i < kNumOutputs; ++i) {
      const size_t size = output_counts[i] * sizeof(float);
      for (int j = 0; j < kNumInputs; ++j)
        disjoint &= !overlaps(output_data[i], size, data[j], bytes[j]);
      for (int j = 0; j < i; ++j)
        disjoint &= !overlaps(output_data[i], size, output_data[j],
                              output_counts[j] * sizeof(float));
    }
    // These are per-line scratch buffers, published only after success. Let
    // ORT write them directly, avoiding temporary CPU outputs and copies.
    // Aliased callers retain the allocate/validate/copy path.
    OrtPtr<OrtValue> outputs[kNumOutputs];
    OrtValue* results[kNumOutputs] = {};
    if (disjoint) {
      for (int i = 0; i < kNumOutputs; ++i) {
        if (!Check(api->CreateTensorWithDataAsOrtValue(
                       s_.mem.get(), output_data[i],
                       output_counts[i] * sizeof(float), output_shapes[i],
                       output_ranks[i], ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                       &results[i]),
                   err, output_names[i]))
          return false;
        outputs[i].reset(results[i]);
      }
    }
    const bool ran =
        Check(api->Run(s_.sess.get(), nullptr, input_names, input_values,
                       kNumInputs, output_names, kNumOutputs, results),
              err, "performance run");
    if (!disjoint)
      for (int i = 0; i < kNumOutputs; ++i) outputs[i].reset(results[i]);
    if (!ran) return FailRun(s_, failed, err);
    if (!ReadFloats(outputs[0].get(),
                    kPerformanceHopFrames * kPerformanceChannels, motion, err,
                    "performance output") ||
        !ReadFloats(outputs[1].get(), kPerformanceStateSize, next.keys.data(),
                    err, "performance keys") ||
        !ReadFloats(outputs[2].get(), kPerformanceStateSize, next.values.data(),
                    err, "performance values"))
      return false;
    next.frames = std::min<int64_t>(kPerformanceMemoryFrames,
                                    state.frames + kPerformanceHopFrames);
    return true;
  }

 private:
  Session s_;
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
  diagnostics::Progress(diagnostics::Stage::kRuntime);
  if (api) return true;
  static bool attempted = false;
  static std::string failure;
  static auto failure_stage = diagnostics::Stage::kRuntime;
  if (attempted) {
    diagnostics::Progress(failure_stage);
    err = failure;
    return false;
  }
  attempted = true;
  // Full path + altered search path: DirectML.dll /
  // onnxruntime_providers_shared.dll resolve from the same folder.
  HMODULE h = LoadLibraryExW((dll_dir + L"\\onnxruntime.dll").c_str(), nullptr,
                             LOAD_WITH_ALTERED_SEARCH_PATH);
  if (!h) {
    err = "cannot load onnxruntime.dll (error " +
          std::to_string(GetLastError()) + ")";
    failure = err;
    return false;
  }
  // Retain even an unusable module: repeated load/unload can invalidate a
  // crash logger's concurrently captured module handles.
  ort_module = h;
  failure_stage = diagnostics::Stage::kRuntimeApi;
  diagnostics::Progress(failure_stage);
  auto base = reinterpret_cast<const OrtApiBase* (*)()>(
      reinterpret_cast<void*>(GetProcAddress(h, "OrtGetApiBase")));
  if (!base || !(api = base()->GetApi(ORT_API_VERSION))) {
    err = "onnxruntime.dll: wrong version";
    api = nullptr;
    failure = err;
    return false;
  }
  failure_stage = diagnostics::Stage::kEnvironment;
  diagnostics::Progress(failure_stage);
  if (!Check(api->CreateEnvWithCustomLogger(OrtLog, nullptr,
                                            ORT_LOGGING_LEVEL_WARNING,
                                            "BetterFacialAnimations", &env),
             err, "env")) {
    api = nullptr;
    failure = err;
    return false;
  }
  // The successful module and environment live as long as the worker.
  version = base()->GetVersionString();
  ort_module = h;
  return true;
}

std::string StageError(HRESULT hr, const char* stage) {
  char code[16];
  snprintf(code, sizeof code, "%08lX", static_cast<unsigned long>(hr));
  return std::string("independent device: ") + stage + " failed (" + code + ")";
}

bool GraphicsCheck(HRESULT hr, const char* stage, std::string& err) {
  if (SUCCEEDED(hr)) return true;
  err = StageError(hr, stage) + "; shared-device fallback disabled";
  return false;
}

// Owns raw. A successful call that produced nothing is reported as
// E_NOINTERFACE at the same stage.
template <class T>
ComPtr<T> Obtained(HRESULT hr, T* raw, const char* stage, std::string& err) {
  ComPtr<T> out(raw);
  if (!GraphicsCheck(hr, stage, err)) return nullptr;
  if (!out) GraphicsCheck(E_NOINTERFACE, stage, err);
  return out;
}

// Resolve the mod-manager-visible runtime directory, preserving Unicode paths.
// CreateDeviceFactory accepts an absolute UTF-8 SDK path with a trailing slash.
std::string AgilityPath(const std::wstring& dll_dir) {
  const auto relative = dll_dir + L"\\Agility\\";
  const DWORD size = GetFullPathNameW(relative.c_str(), 0, nullptr, nullptr);
  if (!size) return {};
  std::vector<wchar_t> full(size);
  const DWORD written =
      GetFullPathNameW(relative.c_str(), size, full.data(), nullptr);
  if (!written || written >= size) return {};
  const int bytes =
      WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, full.data(), written,
                          nullptr, 0, nullptr, nullptr);
  if (!bytes) return {};
  std::string path(bytes, '\0');
  if (!WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, full.data(), written,
                           path.data(), bytes, nullptr, nullptr))
    return {};
  return path;
}

// This changes only our factory, never the process SDK selection. The SDK
// configuration is released normally; FreeUnusedSDKs is deliberately never
// called, preserving the existing process-lifetime runtime/resource policy.
ComPtr<independent::Factory> AgilityFactory(independent::GetInterface get,
                                            const std::string& path,
                                            std::string& err) {
  independent::Configuration* raw = nullptr;
  const HRESULT queried =
      get(independent::kConfigurationClass,
          independent::kConfigurationInterface, reinterpret_cast<void**>(&raw));
  const auto config =
      Obtained(queried, raw, "Agility SDK configuration query", err);
  if (!config) return nullptr;
  independent::Factory* made = nullptr;
  const HRESULT created = config->CreateDeviceFactory(
      independent::kAgilityVersion, path.c_str(),
      independent::kFactoryInterface, reinterpret_cast<void**>(&made));
  return Obtained(created, made, "Agility CreateDeviceFactory (SDK 619)", err);
}

// get is injectable for the offline factory contract. No global state is
// imported/applied, and no ordinary D3D12CreateDevice call is made here.
// A non-empty dll_dir enables the bundled Agility runtime when the default
// factory is unavailable.
ComPtr<ID3D12Device> MakeIndependentDevice(independent::GetInterface get,
                                           std::string& err,
                                           IUnknown* adapter = nullptr,
                                           const std::wstring& dll_dir = {},
                                           bool force_bundled_agility = false) {
  diagnostics::Progress(diagnostics::Stage::kFactory);
  auto api_status = [](const std::string& result) {
    diagnostics::State().IndependentApi(result);
    if (ort_log) ort_log("Independent Devices runtime API: " + result);
  };
  api_status("query in progress (no device created yet)");
  if (ort_log) ort_log("GPU startup: obtaining independent device factory");
  if (!get) {
    diagnostics::State().System(GpuReport(false));
    api_status(
        "unavailable: D3D12GetInterface export missing; driver support not "
        "tested");
    GraphicsCheck(E_NOINTERFACE, "runtime API unavailable: D3D12GetInterface",
                  err);
    return nullptr;
  }
  independent::Factory* raw = nullptr;
  const HRESULT found = force_bundled_agility
      ? E_NOINTERFACE
      : get(independent::kFactoryClass, independent::kFactoryInterface,
            reinterpret_cast<void**>(&raw));
  ComPtr<independent::Factory> factory(raw);
  if (FAILED(found)) factory.reset();
  const HRESULT direct = FAILED(found) ? found : factory ? S_OK : E_NOINTERFACE;
  const char* const stage = found == E_NOINTERFACE
                                ? "runtime API unavailable: device factory"
                                : "runtime API query: device factory";
  std::string status =
      direct == E_NOINTERFACE ? "unavailable: " : "query failed: ";
  if (force_bundled_agility && dll_dir.empty()) {
    GraphicsCheck(E_INVALIDARG, "forced Agility runtime directory missing", err);
    api_status("query failed: " + err);
    return nullptr;
  }
  if (FAILED(direct) && !dll_dir.empty()) {
    const std::string path = AgilityPath(dll_dir);
    if (ort_log)
      ort_log((force_bundled_agility
                   ? std::string("ForceBundledAgility=1: default factory skipped")
                   : StageError(direct, stage)) +
              "; trying bundled Agility SDK 619 at " + path);
    std::string recovery;
    factory = AgilityFactory(get, path, recovery);
    if (factory) {
      if (ort_log)
        ort_log("GPU startup: bundled Agility factory obtained (SDK 619)");
    } else {
      // Both failures, one trailing policy suffix (carried by recovery).
      err = force_bundled_agility ? "forced Agility: " + recovery
                                  : StageError(direct, stage) + "; " + recovery;
      status = "query failed: ";
    }
  } else if (FAILED(direct)) {
    GraphicsCheck(direct, stage, err);
  }
  // Capture the runtime actually selected by the factory query, including on
  // failure. Passive reporting does not create a device or load a graphics
  // runtime.
  diagnostics::State().System(GpuReport(false));
  if (!factory) {
    api_status(status + err + "; driver support not tested");
    return nullptr;
  }
  api_status("available (factory obtained; does not establish driver support)");
  diagnostics::Progress(diagnostics::Stage::kFlags);
  if (!GraphicsCheck(factory->SetFlags(independent::kStrictFlags),
                     "strict isolation flags", err))
    return nullptr;
  // Null preserves the default-adapter path. Explicit adapters use the same
  // DXGI enumeration order as the shared DirectML provider.
  ID3D12Device* raw_device = nullptr;
  if (ort_log) ort_log("GPU startup: creating independent D3D12 device");
  diagnostics::Progress(diagnostics::Stage::kDevice);
  const HRESULT made = factory->CreateDevice(
      adapter, D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device),
      reinterpret_cast<void**>(&raw_device));
  return Obtained(made, raw_device,
                  "device creation: CreateDevice (runtime API available)", err);
}

// Only the inference worker accesses this context. Model sessions borrow COM
// references, but never own the last reference. Keep the healthy context until
// process exit, including across explicit CPU use and idle model unloading. Do
// not run its destructor during DLL/process teardown: driver destruction can
// block Skyrim's renderer even when called on the inference worker.
struct IndependentContext {
  ComPtr<ID3D12Device> device;
  ComPtr<IUnknown> dml;
  ComPtr<ID3D12CommandQueue> queue;
  bool attempted = false;

  // Latches err as the process-wide failure and returns the latched text.
  bool Reject(std::string& err) {
    DisableGpu(err);
    GpuAllowed(err);
    return false;
  }

  bool Healthy(std::string& err) {
    if (!GpuAllowed(err)) return false;
    const auto reason = RemovedReason(device.get(), dml.get());
    if (reason.empty()) return true;
    err = "independent context:" + reason;
    return Reject(err);
  }

  ComPtr<ID3D12Device> Acquire(independent::GetInterface get, std::string& err,
                               IUnknown* adapter = nullptr,
                               const std::wstring& dll_dir = {},
                                           bool force_bundled_agility = false) {
    if (!Healthy(err)) return nullptr;
    if (!attempted) {
      attempted = true;
      device = MakeIndependentDevice(get, err, adapter, dll_dir,
                                     force_bundled_agility);
      if (!device) {
        Reject(err);
        return nullptr;
      }
      if (ort_log)
        ort_log("independent GPU device retained until process exit");
    }
    device->AddRef();
    return ComPtr<ID3D12Device>(device.get());
  }
};

int independent_adapter = 0;  // worker-owned, last attempted context
IndependentContext& IndependentGpu(int index = 0) {
  static auto& contexts = *new std::map<int, IndependentContext*>;
  auto& context = contexts[index];
  if (!context) context = new IndependentContext;
  return *context;
}

// A failed independent context (not a failed session on a healthy one)
// protects its existing sessions until RetryGpu. No files or cross-launch
// state participate in selection.
std::string gpu_failure;

class GpuAttempt {
 public:
  GpuAttempt(const OrtOptions& options, std::string& err)
      : active_(options.gpu),
        independent_(options.independent_device),
        err_(err) {
    allowed_ = !active_ || GpuAllowed(err_);
    finished_ = !allowed_;
  }
  bool allowed() const { return allowed_; }
  bool Finish(bool warmed) {
    if (!active_) return warmed;
    finished_ = true;
    if (warmed) {
      if (ort_log) ort_log("GPU startup: warmup complete; GPU ready");
    } else {
      Latch();
    }
    return warmed;
  }
  ~GpuAttempt() {
    if (active_ && !finished_) Latch();
  }

 private:
  void Latch() {
    if (independent_ && IndependentContextFailure(err_)) DisableGpu(err_);
  }

 private:
  bool active_, independent_, allowed_ = true, finished_ = false;
  std::string& err_;
};

ComPtr<ID3D12Device> OpenIndependentDevice(const std::wstring& dll_dir,
                                           std::string& err, int index = 0,
                                           bool force_bundled_agility = false) {
  independent_adapter = index;
  auto& context = IndependentGpu(index);
  ComPtr<IDXGIAdapter1> adapter;
  // The default path deliberately needs no DXGI discovery. Nonzero selections
  // must resolve exactly; never replace a missing adapter with the default.
  if (index != 0 && !context.attempted) {
    adapter = OpenAdapter(index, err);
    if (!adapter) return nullptr;
  }
  diagnostics::Progress(diagnostics::Stage::kFactory);
  return context.Acquire(
      reinterpret_cast<independent::GetInterface>(
          reinterpret_cast<void*>(D3d12Export("D3D12GetInterface"))),
      err, adapter.get(), dll_dir, force_bundled_agility);
}

// Initialize the shared DML device and queue once. Keep partial initialization
// too: retrying must not destroy a device or accumulate abandoned contexts.
bool InitializeIndependentProvider(IndependentContext& context,
                                   independent::CreateDml create,
                                   std::string& err) {
  diagnostics::Progress(diagnostics::Stage::kDirectML);
  if (!context.Healthy(err)) return false;
  if (context.queue) return true;
  if (!create) {
    GraphicsCheck(E_NOINTERFACE, "DMLCreateDevice unavailable", err);
    return context.Reject(err);
  }
  IUnknown* raw_dml = nullptr;
  if (ort_log) ort_log("GPU startup: creating DirectML device");
  const HRESULT made = create(context.device.get(), 0, independent::kDmlDevice,
                              reinterpret_cast<void**>(&raw_dml));
  context.dml.reset(raw_dml);
  if (!GraphicsCheck(made, "DMLCreateDevice", err)) return context.Reject(err);
  if (!context.dml) {
    GraphicsCheck(E_NOINTERFACE, "null DirectML device", err);
    return context.Reject(err);
  }
  D3D12_COMMAND_QUEUE_DESC desc{};
  desc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
  desc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
  // Keep normal GPU timeout behavior. Do not request DISABLE_GPU_TIMEOUT.
  ID3D12CommandQueue* raw_queue = nullptr;
  if (ort_log) ort_log("GPU startup: creating private compute queue");
  diagnostics::Progress(diagnostics::Stage::kQueue);
  const HRESULT queued =
      context.device->CreateCommandQueue(&desc, __uuidof(ID3D12CommandQueue),
                                         reinterpret_cast<void**>(&raw_queue));
  context.queue.reset(raw_queue);
  if (!GraphicsCheck(queued, "CreateCommandQueue", err))
    return context.Reject(err);
  if (!context.queue) {
    GraphicsCheck(E_NOINTERFACE, "null compute queue", err);
    return context.Reject(err);
  }
  if (ort_log)
    ort_log("independent DirectML device and compute queue retained");
  return true;
}

bool AppendIndependentProvider(const std::wstring& dll_dir,
                               const OrtDmlApi* dml, OrtSessionOptions* so,
                               ID3D12Device* device,
                               ComPtr<IUnknown>& dml_device, std::string& err,
                               int index = 0) {
  auto& context = IndependentGpu(index);
  if (!context.Healthy(err)) return false;
  if (!dml || !dml->SessionOptionsAppendExecutionProvider_DML1) {
    GraphicsCheck(E_NOINTERFACE, "ONNX Runtime DML1 unavailable", err);
    return false;
  }
  // Retained for the process lifetime, as is the ORT module. Never change
  // the game's DLL search policy or globally selected D3D12 SDK version.
  if (!context.queue) {
    diagnostics::Progress(diagnostics::Stage::kDirectML);
    static const RuntimeModule module(dll_dir + L"\\DirectML.dll");
    if (!module.module) {
      err = "independent device: LoadLibraryExW(DirectML.dll) failed [Win32 " +
            std::to_string(module.error) + "]";
      return false;
    }
    const auto create =
        reinterpret_cast<independent::CreateDml>(reinterpret_cast<void*>(
            GetProcAddress(module.module, "DMLCreateDevice")));
    if (!create) {
      const DWORD code = GetLastError();
      err =
          "independent device: GetProcAddress(DMLCreateDevice) failed [Win32 " +
          std::to_string(code) + "]";
      return false;
    }
    if (!InitializeIndependentProvider(context, create, err)) return false;
  }
  if (device != context.device.get()) {
    err = "independent provider: unexpected device";
    return false;
  }
  context.dml->AddRef();
  dml_device.reset(context.dml.get());
  // ORT retains these too; releasing a session frees its model and buffers,
  // while the process context keeps the device and queue alive.
  diagnostics::Progress(diagnostics::Stage::kProvider);
  return Check(dml->SessionOptionsAppendExecutionProvider_DML1(
                   so, context.dml.get(), context.queue.get()),
               err, "independent device: DirectML DML1");
}

bool AppendSharedProvider(const OrtDmlApi* dml, OrtSessionOptions* options,
                          bool metacommands, int adapter, std::string& err) {
  if (adapter < 0) {
    err = "invalid GPU adapter index";
    return false;
  }
  if (metacommands)
    return Check(
        dml->SessionOptionsAppendExecutionProvider_DML(options, adapter), err,
        "DirectML");
  const char* const keys[] = {"device_id", "disable_metacommands"};
  const std::string id = std::to_string(adapter);
  const char* const values[] = {id.c_str(), "true"};
  return Check(api->SessionOptionsAppendExecutionProvider(options, "DML", keys,
                                                          values, 2),
               err, "DirectML");
}

// A session of model on DirectML (selected adapter, with the driver's
// metacommands or without) or the CPU, with the free dimensions dims fixed
// (name, size): let DirectML plan one static shape (also 4x faster at batch
// 1-2). False (with err, and out empty) when it cannot be made.
using Dims = std::initializer_list<std::pair<const char*, int>>;
bool CreateSession(const std::wstring& model, bool gpu, bool metacommands,
                   bool independent_device, const std::wstring& dll_dir,
                   Dims dims, Session& out, std::string& err,
                   int gpu_adapter = 0, bool force_bundled_agility = false) {
  Session s;
  s.independent = independent_device;
  const ScopedWrappers keep(s.wrappers);
  if (gpu && independent_device) {
    if (!metacommands) {
      err = "independent device: DML1 does not support disabling metacommands";
      return false;
    }
    s.device = OpenIndependentDevice(dll_dir, err, gpu_adapter,
                                     force_bundled_agility);
    if (!s.device) return false;
  } else if (gpu) {
    diagnostics::Progress(diagnostics::Stage::kDevice);
    if (ort_log) ort_log("GPU startup: creating 4.0-style shared D3D12 device");
    static bool redirect_tried;
    if (!redirect_tried) {
      redirect_tried = true;
      device_redirected = RedirectCreateDevice(ort_module);
      if (!device_redirected && ort_log)
        ort_log(
            "onnxruntime.dll does not import D3D12CreateDevice as "
            "expected: a wrapper of Direct3D 12 (ReShade) stays in the "
            "model's way");
    }
    s.device = OpenDevice(&err, gpu_adapter);
    if (!s.device) return false;
    // ORT delay-loads DirectML by name; pin our bundled library first.
    diagnostics::Progress(diagnostics::Stage::kDirectML);
    static const RuntimeModule module(dll_dir + L"\\DirectML.dll");
    if (!module.module) {
      err = "shared device: LoadLibraryExW(DirectML.dll) failed [Win32 " +
            std::to_string(module.error) + "]";
      return false;
    }
  }
  diagnostics::Progress(diagnostics::Stage::kOptions);
  OrtSessionOptions* options = nullptr;
  bool ok = Check(api->CreateSessionOptions(&options), err, "options");
  OrtPtr<OrtSessionOptions> so(options);
  // Bound CPU helper work for both providers; idle threads sleep.
  if (ok) {
    ok = Check(api->SetIntraOpNumThreads(so.get(), 2), err, "threads") &&
         Check(api->SetInterOpNumThreads(so.get(), 1), err, "threads") &&
         Check(api->AddSessionConfigEntry(
                   so.get(), "session.intra_op.allow_spinning", "0"),
               err, "CPU idle") &&
         Check(api->AddSessionConfigEntry(
                   so.get(), "session.inter_op.allow_spinning", "0"),
               err, "CPU idle");
  }
  if (ok && gpu) {
    diagnostics::Progress(diagnostics::Stage::kProvider);
    const OrtDmlApi* dml = nullptr;
    ok =
        Check(api->DisableMemPattern(so.get()), err, "mem pattern") &&
        Check(api->SetSessionExecutionMode(so.get(), ORT_SEQUENTIAL), err,
              "mode") &&
        Check(api->GetExecutionProviderApi(
                  "DML", ORT_API_VERSION, reinterpret_cast<const void**>(&dml)),
              err, "DML api");
    if (ok && independent_device) {
      ok = AppendIndependentProvider(dll_dir, dml, so.get(), s.device.get(),
                                     s.dml_device, err, gpu_adapter);
    } else if (ok) {
      ok = AppendSharedProvider(dml, so.get(), metacommands, gpu_adapter, err);
    }
    // the session's DirectML device (handed out without a reference), to ask
    // it why it stopped
    IUnknown* dml_device = nullptr;
    if (ok && !independent_device) {
      OrtStatus* status = dml->GetDMLDevice(so.get(), &dml_device);
      if (status)
        api->ReleaseStatus(status);
      else if (dml_device) {
        dml_device->AddRef();
        s.dml_device.reset(dml_device);
      }
    }
    // DML's memory arena stays on: it keeps the work buffers of a run
    // allocated between runs (~16 MB more resident than freeing them), so a
    // loaded session never has to find VRAM again.
    for (const auto& [name, size] : dims) {
      if (ok) {
        ok = Check(api->AddFreeDimensionOverrideByName(so.get(), name, size),
                   err, "shape override");
      }
    }
  } else if (ok) {
    // Not ORT_ENABLE_ALL: its layout optimizations fail on the casts the CPU
    // provider inserts around the FP16 network's operators
    // (SimplifiedLayerNormFusion looks up a node argument that is gone).
    ok = Check(
        api->SetSessionGraphOptimizationLevel(so.get(), ORT_ENABLE_EXTENDED),
        err, "optimization");
  }
  if (ok) {
    diagnostics::Progress(diagnostics::Stage::kSession);
    OrtSession* sess = nullptr;
    if (gpu && ort_log) ort_log("GPU startup: compiling ONNX model session");
    ok = Check(api->CreateSession(env, model.c_str(), so.get(), &sess), err,
               "session");
    s.sess.reset(sess);
  }
  if (ok) {
    diagnostics::Progress(diagnostics::Stage::kMemory);
    OrtMemoryInfo* mem = nullptr;
    ok = Check(
        api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem),
        err, "memory info");
    s.mem.reset(mem);
  }
  if (!ok) {
    if (gpu) err += RemovedReason(s.device.get(), s.dml_device.get());
    // Resources on a failed independent context stay quarantined (driver
    // teardown there can block the renderer); GpuAttempt latches the context.
    // Everything else is destroyed in order.
    if (gpu && independent_device && IndependentContextFailure(err)) {
      (void)so.release();
      (void)new Session(std::move(s));
    }
    return false;
  }
  if (gpu && ort_log) ort_log("GPU startup: session compiled; model warmup");
  diagnostics::Progress(diagnostics::Stage::kWarmup);
  out = std::move(s);
  return true;
}

// The execution provider part of a session's info line.
std::string DescribeProvider(const OrtOptions& o) {
  std::string description =
      !o.gpu
          ? ", CPU 2 threads"
          : ", DirectML device " + std::to_string(o.gpu_adapter) +
                (o.metacommands ? ", metacommands on" : ", metacommands off");
  if (o.gpu && o.independent_device)
    description += ", independent device, private compute queue";
  else if (o.gpu)
    description +=
        use_wrapper ? ", shared device, wrapper" : ", shared device, unwrapped";
  return description;
}
class OrtEmotionNet : public EmotionNet {
 public:
  OrtEmotionNet(Session s, bool performance)
      : s_(std::move(s)), performance_(performance) {}
  ~OrtEmotionNet() override { RetainFailedSession(s_); }
  bool Run(const float* pcm, float* logits, std::string& err) override {
    if (!CanRun(s_.device.get(), err)) return false;
    std::vector<float> mel;
    if (performance_) EmotionFeatures(pcm, mel);
    const int64_t raw_shape[] = {1, kVoiceWindow};
    const int64_t mel_shape[] = {1, 48, 188};
    const float* data = performance_ ? mel.data() : pcm;
    const size_t count = performance_ ? mel.size() : kVoiceWindow;
    OrtValue* input = nullptr;
    if (!Check(
            api->CreateTensorWithDataAsOrtValue(
                s_.mem.get(), const_cast<float*>(data), count * sizeof(float),
                performance_ ? mel_shape : raw_shape, performance_ ? 3 : 2,
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input),
            err, "sentiment input"))
      return false;
    const OrtPtr<OrtValue> x(input);
    const OrtValue* inputs[] = {x.get()};
    const char* in[] = {performance_ ? "mel" : "input_values"};
    const char* out[] = {performance_ ? "logits" : "output"};
    OrtValue* output = nullptr;
    const bool ran =
        Check(api->Run(s_.sess.get(), nullptr, in, inputs, 1, out, 1, &output),
              err, "sentiment run");
    const OrtPtr<OrtValue> y(output);
    if (!ran) {
      RunFailed(s_, err);
      return false;
    }
    return ReadFloats(y.get(), kNumVoiceEmo, logits, err, "sentiment output");
  }

 private:
  Session s_;
  bool performance_;
};

}  // namespace

std::unique_ptr<EmotionNet> CreateOrtEmotionNet(const std::wstring& dll_dir,
                                                const std::wstring& model,
                                                const OrtOptions& o,
                                                std::string& err,
                                                bool performance) {
  if (!LoadApi(dll_dir, err)) return nullptr;
  GpuAttempt attempt(o, err);
  if (!attempt.allowed()) return nullptr;
  Session session;
  if (!CreateSession(
          model, o.gpu, o.metacommands, o.independent_device, dll_dir,
          {{"batch_size", 1}, {"seq_len", kVoiceWindow}, {"batch", 1}}, session,
          err, o.gpu_adapter, o.force_bundled_agility))
    return nullptr;
  auto net = std::make_unique<OrtEmotionNet>(std::move(session), performance);
  std::vector<float> silence(kVoiceWindow);
  float logits[kNumVoiceEmo];
  if (!attempt.Finish(net->Run(silence.data(), logits, err))) return nullptr;
  return net;
}

std::unique_ptr<Net> CreateOrtNet(const std::wstring& dll_dir,
                                  const std::wstring& model,
                                  const OrtOptions& o, std::string& err,
                                  std::string& info) {
  if (!LoadApi(dll_dir, err)) return nullptr;
  GpuAttempt attempt(o, err);
  if (!attempt.allowed()) return nullptr;
  const int batch = std::max(1, o.batch);
  Session s;
  // The batch is always the same.
  if (!CreateSession(
          model, o.gpu, o.metacommands, o.independent_device, dll_dir,
          {{"input_dynamic_axes_1", batch}, {"emotion_dynamic_axes_1", batch}},
          s, err, o.gpu_adapter, o.force_bundled_agility))
    return nullptr;
  info = std::string("Audio2Face-3D v2.3, onnxruntime ") + version +
         DescribeProvider(o) + ", batch " + std::to_string(batch);
  auto net = std::make_unique<OrtNet>(std::move(s), batch);
  // One run now: a session that cannot run fails here and not on the first
  // line, and DirectML allocates its work buffers.
  std::vector<float> x(static_cast<size_t>(batch) * kWindow),
      emo(static_cast<size_t>(batch) * kNumEmo),
      y(static_cast<size_t>(batch) * kNumOut);
  if (!attempt.Finish(net->Run(x.data(), emo.data(), batch, y.data(), err)))
    return nullptr;
  return net;
}

std::unique_ptr<Net> CreatePerformanceNet(const std::wstring& dll_dir,
                                          const std::wstring& model,
                                          const OrtOptions& o, std::string& err,
                                          std::string& info) {
  if (!LoadApi(dll_dir, err)) return nullptr;
  GpuAttempt attempt(o, err);
  if (!attempt.allowed()) return nullptr;
  Session s;
  if (!CreateSession(model, o.gpu, o.metacommands, o.independent_device,
                     dll_dir, {{"batch", 1}}, s, err, o.gpu_adapter, o.force_bundled_agility))
    return nullptr;
  auto net = std::make_unique<PerformanceNet>(std::move(s));
  // One run now, as CreateOrtNet's.
  std::vector<float> mel(kPerformanceWindowFrames * kMelBins),
      emotion(kPerformanceWindowFrames * kNumMod),
      motion(kPerformanceHopFrames * kPerformanceChannels);
  PerformanceState state, next;
  if (!attempt.Finish(net->RunPerformance(mel.data(), 0, emotion.data(),
                                          state, motion.data(), next, err)))
    return nullptr;
  info =
      std::string("Performance Block AttnRes, 4,008,025 parameters, 30 fps") +
      DescribeProvider(o);
  return net;
}

std::unique_ptr<Net> CreateOrtNet3(const std::wstring& dll_dir,
                                   const std::wstring& model,
                                   const OrtOptions& o, std::string& err,
                                   std::string& info) {
  if (!LoadApi(dll_dir, err)) return nullptr;
  GpuAttempt attempt(o, err);
  if (!attempt.allowed()) return nullptr;
  Session s;
  if (!CreateSession(model, o.gpu, o.metacommands, o.independent_device,
                     dll_dir,
                     {{"window_dynamic_axes_1", 1},
                      {"identity_dynamic_axes_1", 1},
                      {"emotion_dynamic_axes_1", 1},
                      {"input_latents_dynamic_axes_1", 1}},
                     s, err, o.gpu_adapter, o.force_bundled_agility))
    return nullptr;
  info = std::string("Audio2Face-3D v3.0, onnxruntime ") + version +
         DescribeProvider(o);
  auto net = std::make_unique<OrtNet3>(std::move(s));
  // One run now, as CreateOrtNet's.
  std::vector<float> x(kWindow3), emo(kFrames3 * kNumMod), state(kState3),
      skin(kFrames3 * kLatent3), tongue(kFrames3 * kLatent3),
      jaw(kFrames3 * kJawEyes3), next(kState3);
  const float identity[kNumIdentity3] = {1, 0, 0};
  if (!attempt.Finish(net->Run3(x.data(), identity, emo.data(), state.data(),
                                skin.data(), tongue.data(), jaw.data(),
                                next.data(), err)))
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
void RetryGpu() {
  // The worker has already unloaded both model sessions. Failed resources and
  // their context stay alive: driver teardown must not block the renderer.
  // A healthy context is reused across explicit CPU/GPU/model changes.
  if (!gpu_failure.empty()) {
    auto& context = IndependentGpu(independent_adapter);
    (void)context.device.release();
    (void)context.dml.release();
    (void)context.queue.release();
    context.attempted = false;
  }
  gpu_failure.clear();
}
bool GpuAllowed(std::string& err) {
  if (gpu_failure.empty()) return true;
  err = gpu_failure;
  return false;
}
void DisableGpu(const std::string& reason) {
  if (!gpu_failure.empty()) return;
  gpu_failure = reason + "; GPU attempt failed";
  if (ort_log) ort_log(gpu_failure);
}

bool DeviceWrapped() { return device_wrapped; }
void SetUseDeviceWrapper(bool use) { use_wrapper = use; }

bool DeviceRedirected() { return device_redirected; }

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
             !has4                              ? "unknown"
             : o4.Native16BitShaderOpsSupported ? "yes"
                                                : "no",
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
    out.push_back(
        "Direct3D 12 hands out a wrapper of its device (ReShade "
        "does): the model runs on the device itself");
}

namespace {
using WineFn = const char*(CDECL*)();
// Windows (and Wine / Proton, where Direct3D 12 is vkd3d-proton).
independent::WindowsVersion ReadWindowsVersion(RTL_OSVERSIONINFOEXW& os,
                                               WineFn& wine) {
  const HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
  os = {};
  os.dwOSVersionInfoSize = sizeof os;
  using VersionFn = LONG(WINAPI*)(RTL_OSVERSIONINFOW*);
  independent::WindowsVersion version;
  if (auto get = reinterpret_cast<VersionFn>(
          reinterpret_cast<void*>(GetProcAddress(ntdll, "RtlGetVersion"))))
    version.known = get(reinterpret_cast<RTL_OSVERSIONINFOW*>(&os)) == 0;
  version.major = os.dwMajorVersion;
  version.minor = os.dwMinorVersion;
  version.build = os.dwBuildNumber;
  version.workstation = os.wProductType == VER_NT_WORKSTATION;
  wine = reinterpret_cast<WineFn>(
      reinterpret_cast<void*>(GetProcAddress(ntdll, "wine_get_version")));
  version.wine = wine != nullptr;
  // RtlGetVersion omits the cumulative-update revision. Trust UBR only when
  // the registry build agrees with the running OS; never classify Proton by it.
  if (version.known && !version.wine) {
    constexpr auto key = L"SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion";
    wchar_t build[32] = {}, expected[32] = {};
    DWORD size = sizeof build, revision = 0, revision_size = sizeof revision;
    swprintf(expected, 32, L"%lu", os.dwBuildNumber);
    if (RegGetValueW(HKEY_LOCAL_MACHINE, key, L"CurrentBuildNumber",
                     RRF_RT_REG_SZ | RRF_SUBKEY_WOW6464KEY, nullptr, build,
                     &size) == ERROR_SUCCESS &&
        wcscmp(build, expected) == 0 &&
        RegGetValueW(HKEY_LOCAL_MACHINE, key, L"UBR",
                     RRF_RT_REG_DWORD | RRF_SUBKEY_WOW6464KEY, nullptr,
                     &revision, &revision_size) == ERROR_SUCCESS) {
      version.revision = revision;
      version.revision_known = true;
    }
  }
  return version;
}
}  // namespace

std::string IndependentPrerequisite() {
  RTL_OSVERSIONINFOEXW os;
  WineFn wine;
  return independent::WindowsPrerequisiteReport(ReadWindowsVersion(os, wine));
}

std::vector<std::string> GpuReport(bool probe_gpu) {
  std::vector<std::string> out;
  char b[900];
  RTL_OSVERSIONINFOEXW os;
  WineFn wine;
  const independent::WindowsVersion version = ReadWindowsVersion(os, wine);
  const std::string revision = version.revision_known
                                   ? "." + std::to_string(version.revision)
                                   : ".? (update revision unknown)";
  MEMORYSTATUSEX ram{};
  ram.dwLength = sizeof ram;
  GlobalMemoryStatusEx(&ram);
  snprintf(b, sizeof b,
           "system: Windows %lu.%lu build %lu%s%s%s; memory %.0f MB free of "
           "%.0f MB, commit %.0f MB free of %.0f MB",
           os.dwMajorVersion, os.dwMinorVersion, os.dwBuildNumber,
           revision.c_str(), wine ? ", Wine " : "", wine ? wine() : "",
           ram.ullAvailPhys / 1048576.0, ram.ullTotalPhys / 1048576.0,
           ram.ullAvailPageFile / 1048576.0, ram.ullTotalPageFile / 1048576.0);
  out.push_back(b);
  out.push_back(independent::WindowsPrerequisiteReport(version));
  for (const wchar_t* name : {L"d3d12.dll", L"d3d12core.dll"}) {
    const HMODULE module = GetModuleHandleW(name);
    wchar_t path[MAX_PATH] = {};
    const DWORD length =
        module ? GetModuleFileNameW(module, path, MAX_PATH) : 0;
    if (length && length < MAX_PATH) {
      const auto v = FileVersion(path);
      out.push_back("loaded runtime: " + Utf8(path) +
                    (v.empty() ? " (version unknown)" : " " + v));
    } else {
      out.push_back("loaded runtime: " + Utf8(name) +
                    (module ? " (path/version unknown)"
                            : " not loaded (availability not tested)"));
    }
  }
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
        L"d3d12.dll",         L"d3d12core.dll",  L"dxgi.dll",
        L"d3d11.dll",         L"dxcore.dll",     L"directml.dll",
        L"msvcp140.dll",      L"msvcp140_1.dll", L"vcruntime140.dll",
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

GpuAdapterList EnumerateGpuAdapters() {
  GpuAdapterList result;
  result.requested = result.complete = true;
  IDXGIFactory1* raw = nullptr;
  HRESULT hr =
      CreateFactory(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&raw));
  const ComPtr<IDXGIFactory1> factory(raw);
  if (FAILED(hr) || !factory) {
    result.error = NativeError("GPU discovery", FAILED(hr) ? hr : E_POINTER);
    return result;
  }
  for (UINT i = 0;; ++i) {
    IDXGIAdapter1* next = nullptr;
    hr = factory->EnumAdapters1(i, &next);
    const ComPtr<IDXGIAdapter1> adapter(next);
    if (hr == DXGI_ERROR_NOT_FOUND) break;
    if (FAILED(hr) || !adapter) {
      result.error = NativeError("GPU discovery", FAILED(hr) ? hr : E_POINTER);
      break;
    }
    DXGI_ADAPTER_DESC1 desc{};
    hr = adapter->GetDesc1(&desc);
    if (FAILED(hr)) {
      result.error = NativeError("GPU description", hr);
      continue;
    }
    if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
    result.adapters.push_back({static_cast<int>(i), Utf8(desc.Description),
                               desc.DedicatedVideoMemory / 1048576.0});
  }
  if (result.adapters.empty() && result.error.empty())
    result.error = "No hardware GPUs found. CPU remains available.";
  return result;
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
      const ComPtr<IDXGIAdapter3> adapter3 = As<IDXGIAdapter3>(adapter.get());
      DXGI_QUERY_VIDEO_MEMORY_INFO vm{};
      if (adapter3 && SUCCEEDED(adapter3->QueryVideoMemoryInfo(
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
