// ONNX Runtime session wrapper; session options are tuned for minimum VRAM.
#include "ort_net.h"

#define PSAPI_VERSION 2  // K32EnumProcessModules, in kernel32
#include <d3d12.h>
#include <dxgi1_4.h>
#include <windows.h>
#include <psapi.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <atomic>
#include <cstring>
#include <cwchar>
#include <cwctype>
#include <memory>
#include <string>
#include <vector>

#include "live_core.h"
#include "onnxruntime_c_api.h"

// dml_provider_factory.h needs DirectML.h (MSVC SAL annotations); the entries
// not used are placeholders.
struct OrtDmlApi {
  OrtStatus*(ORT_API_CALL* SessionOptionsAppendExecutionProvider_DML)(
      OrtSessionOptions*, int device_id);
  void* SessionOptionsAppendExecutionProvider_DML1;
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

// An export of d3d12.dll (not imported: the plugin loads without it).
FARPROC D3d12Export(const char* id) {
  static const HMODULE d3d12 = LoadLibraryW(L"d3d12.dll");
  return d3d12 ? GetProcAddress(d3d12, id) : nullptr;
}

// Set when D3D12CreateDevice handed out a wrapper of the device.
bool device_wrapped;
// Set when onnxruntime.dll creates its device with CreateRealDevice.
bool device_redirected;

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
  ID3D12Device* device = nullptr;
  ID3D12Fence* fence = nullptr;
  void* real = nullptr;
  if (SUCCEEDED(got->QueryInterface(__uuidof(ID3D12Device),
                                    reinterpret_cast<void**>(&device))) &&
      SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                                    __uuidof(ID3D12Fence),
                                    reinterpret_cast<void**>(&fence))) &&
      SUCCEEDED(fence->GetDevice(riid, &real))) {
    device_wrapped |= real != *out;
    got->Release();
    *out = real;
  }
  if (fence) fence->Release();
  if (device) device->Release();
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
// device DirectML runs on. Null without Direct3D 12.
ID3D12Device* OpenDevice() {
  IDXGIFactory1* factory = nullptr;
  IDXGIAdapter* adapter = nullptr;
  ID3D12Device* device = nullptr;
  if (SUCCEEDED(CreateDXGIFactory1(__uuidof(IDXGIFactory1),
                                   reinterpret_cast<void**>(&factory))) &&
      SUCCEEDED(factory->EnumAdapters(0, &adapter)))
    CreateRealDevice(adapter, D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device),
                     reinterpret_cast<void**>(&device));
  if (adapter) adapter->Release();
  if (factory) factory->Release();
  return device;
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
  OrtNet(OrtSession* sess, OrtMemoryInfo* mem, int batch, ID3D12Device* device,
         IUnknown* dml)
      : sess_(sess), mem_(mem), batch_(batch), device_(device), dml_(dml) {}
  ~OrtNet() override {
    api->ReleaseSession(sess_);
    api->ReleaseMemoryInfo(mem_);
    if (dml_) dml_->Release();
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
        err += RemovedReason(device_, dml_);
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
  IUnknown* dml_;
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
  device_redirected = RedirectCreateDevice(h);
  if (!device_redirected && ort_log)
    ort_log("onnxruntime.dll does not import D3D12CreateDevice as expected: "
            "a wrapper of Direct3D 12 (ReShade) stays in the model's way");
  return true;
}

// A session of model on DirectML (adapter 0, with the driver's metacommands
// or without; device receives its Direct3D device, held to ask why it was
// lost) or the CPU, with the named free
// dimensions fixed: let DirectML plan one static shape (also 4x faster at
// batch 1-2).
bool CreateSession(const std::wstring& model, bool gpu, bool metacommands,
                   int batch, OrtSession*& sess, OrtMemoryInfo*& mem,
                   ID3D12Device*& device, IUnknown*& dml_device,
                   std::string& err) {
  OrtSessionOptions* so = nullptr;
  sess = nullptr;
  mem = nullptr;
  dml_device = nullptr;
  device = gpu ? OpenDevice() : nullptr;
  bool ok = Check(api->CreateSessionOptions(&so), err, "options");
  if (ok && gpu) {
    const OrtDmlApi* dml = nullptr;
    ok =
        Check(api->DisableMemPattern(so), err, "mem pattern") &&
        Check(api->SetSessionExecutionMode(so, ORT_SEQUENTIAL), err, "mode") &&
        Check(api->GetExecutionProviderApi(
                  "DML", ORT_API_VERSION, reinterpret_cast<const void**>(&dml)),
              err, "DML api");
    if (ok && metacommands) {
      ok = Check(dml->SessionOptionsAppendExecutionProvider_DML(so, 0), err,
                 "DirectML");
    } else if (ok) {
      // only the provider's key / value form can turn them off
      const char* const keys[] = {"device_id", "disable_metacommands"};
      const char* const values[] = {"0", "true"};
      ok = Check(
          api->SessionOptionsAppendExecutionProvider(so, "DML", keys, values, 2),
          err, "DirectML");
    }
    // the session's DirectML device, to ask it why it stopped
    if (ok && !dml->GetDMLDevice(so, &dml_device) && dml_device)
      dml_device->AddRef();
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
  err += RemovedReason(device, dml_device);
  if (dml_device) dml_device->Release();
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
  IUnknown* dml;
  // The batch is always the same.
  if (!CreateSession(model, o.gpu, o.metacommands, batch, sess, mem, device,
                     dml, err))
    return nullptr;
  info = std::string("onnxruntime ") + version +
         (!o.gpu ? ", CPU 2 threads"
          : o.metacommands ? ", DirectML device 0, metacommands on"
                           : ", DirectML device 0, metacommands off") +
         ", batch " + std::to_string(batch);
  auto net = std::make_unique<OrtNet>(sess, mem, batch, device, dml);
  // One run now: a session that cannot run fails here and not on the first
  // line, and DirectML allocates its work buffers.
  std::vector<float> x(static_cast<size_t>(batch) * kWindow),
      emo(static_cast<size_t>(batch) * kNumEmo),
      y(static_cast<size_t>(batch) * kNumOut);
  if (!net->Run(x.data(), emo.data(), batch, y.data(), err)) return nullptr;
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

std::vector<std::string> GpuReport() {
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
  // every adapter, with its driver version
  IDXGIFactory1* factory = nullptr;
  if (SUCCEEDED(CreateDXGIFactory1(__uuidof(IDXGIFactory1),
                                   reinterpret_cast<void**>(&factory)))) {
    IDXGIAdapter1* adapter = nullptr;
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) == S_OK; ++i) {
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
      adapter->Release();
    }
    factory->Release();
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
  if (ID3D12Device* device = OpenDevice()) {
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
    ID3D12Device5* device5 = nullptr;
    UINT n = 0;
    if (SUCCEEDED(device->QueryInterface(__uuidof(ID3D12Device5),
                                         reinterpret_cast<void**>(&device5))) &&
        SUCCEEDED(device5->EnumerateMetaCommands(&n, nullptr))) {
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
    if (device5) device5->Release();
    device->Release();
  } else {
    out.push_back("Direct3D 12 on adapter 0: no device");
  }
  if (device_wrapped)
    out.push_back("Direct3D 12 hands out a wrapper of its device (ReShade "
                  "does): the model runs on the device itself");
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
