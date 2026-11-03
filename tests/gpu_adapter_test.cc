// GPU discovery and provider arguments without real hardware or device creation.
#include <windows.h>
#include <dxgi1_4.h>
HRESULT WINAPI TestCreateFactory(REFIID, void**);
#define CreateDXGIFactory1 TestCreateFactory
#include "../src/inference/ort_net.cc"
#undef CreateDXGIFactory1
#include <cstdlib>
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "line %d: %s\n", __LINE__, #x); std::exit(1); } } while (0)
namespace {
template<class Base> class Object : public Base {
 public:
  ULONG refs = 0;
  HRESULT STDMETHODCALLTYPE QueryInterface(REFIID, void**) override { CHECK(false); return E_FAIL; }
  ULONG STDMETHODCALLTYPE AddRef() override { return ++refs; }
  ULONG STDMETHODCALLTYPE Release() override { CHECK(refs); return --refs; }
  HRESULT STDMETHODCALLTYPE SetPrivateData(REFGUID, UINT, const void*) override { CHECK(false); return E_FAIL; }
  HRESULT STDMETHODCALLTYPE SetPrivateDataInterface(REFGUID, const IUnknown*) override { CHECK(false); return E_FAIL; }
  HRESULT STDMETHODCALLTYPE GetPrivateData(REFGUID, UINT*, void*) override { CHECK(false); return E_FAIL; }
  HRESULT STDMETHODCALLTYPE GetParent(REFIID, void**) override { CHECK(false); return E_FAIL; }
};
class Adapter : public Object<IDXGIAdapter1> {
 public:
  bool software = false, fail_desc = false;
  const wchar_t* name = L"Same GPU name";
  size_t dedicated_memory = 8ull * 1024 * 1024 * 1024;
  HRESULT STDMETHODCALLTYPE EnumOutputs(UINT, IDXGIOutput**) override { CHECK(false); return E_FAIL; }
  HRESULT STDMETHODCALLTYPE GetDesc(DXGI_ADAPTER_DESC*) override { CHECK(false); return E_FAIL; }
  HRESULT STDMETHODCALLTYPE CheckInterfaceSupport(REFGUID, LARGE_INTEGER*) override { CHECK(false); return E_FAIL; }
  HRESULT STDMETHODCALLTYPE GetDesc1(DXGI_ADAPTER_DESC1* out) override {
    if (fail_desc) return E_FAIL;
    *out = {};
    wcscpy(out->Description, name);
    out->Flags = software ? DXGI_ADAPTER_FLAG_SOFTWARE : 0;
    out->DedicatedVideoMemory = dedicated_memory;
    return S_OK;
  }
} adapters[3];
class Factory : public Object<IDXGIFactory1> {
 public:
  bool fail_enum = false;
  HRESULT STDMETHODCALLTYPE EnumAdapters(UINT, IDXGIAdapter**) override { CHECK(false); return E_FAIL; }
  HRESULT STDMETHODCALLTYPE MakeWindowAssociation(HWND, UINT) override { CHECK(false); return E_FAIL; }
  HRESULT STDMETHODCALLTYPE GetWindowAssociation(HWND*) override { CHECK(false); return E_FAIL; }
  HRESULT STDMETHODCALLTYPE CreateSwapChain(IUnknown*, DXGI_SWAP_CHAIN_DESC*, IDXGISwapChain**) override { CHECK(false); return E_FAIL; }
  HRESULT STDMETHODCALLTYPE CreateSoftwareAdapter(HMODULE, IDXGIAdapter**) override { CHECK(false); return E_FAIL; }
  HRESULT STDMETHODCALLTYPE EnumAdapters1(UINT index, IDXGIAdapter1** out) override {
    *out = nullptr;
    if (fail_enum) return E_FAIL;
    if (index >= 3) return DXGI_ERROR_NOT_FOUND;
    adapters[index].AddRef(); *out = &adapters[index]; return S_OK;
  }
  BOOL STDMETHODCALLTYPE IsCurrent() override { CHECK(false); return FALSE; }
} factory;
bool fail_factory = false;
HRESULT WINAPI MakeFactory(REFIID iid, void** out) {
  CHECK(iid == __uuidof(IDXGIFactory1));
  *out = nullptr;
  if (fail_factory) return E_FAIL;
  factory.AddRef(); *out = &factory; return S_OK;
}
int seen_id = -1, provider_calls = 0;
OrtStatus* ORT_API_CALL Append(OrtSessionOptions*, int index) {
  seen_id = index; ++provider_calls; return nullptr;
}
OrtStatus* ORT_API_CALL AppendOptions(OrtSessionOptions*, const char* provider,
    const char* const* keys, const char* const* values, size_t count) noexcept {
  CHECK(!strcmp(provider, "DML") && count == 2);
  CHECK(!strcmp(keys[0], "device_id") && !strcmp(keys[1], "disable_metacommands"));
  CHECK(!strcmp(values[1], "true"));
  seen_id = atoi(values[0]); ++provider_calls; return nullptr;
}
}
HRESULT WINAPI TestCreateFactory(REFIID iid, void** out) {
  return MakeFactory(iid, out);
}
int main() {
  using namespace a2f;
  const auto before = GetModuleHandleW(L"d3d12.dll");
  adapters[1].software = true;
  const auto list = EnumerateGpuAdapters();
  CHECK(list.complete && list.requested && list.error.empty());
  CHECK(list.adapters.size() == 2 && list.adapters[0].index == 0 && list.adapters[1].index == 2);
  CHECK(list.adapters[1].dedicated_mb == 8192 && list.adapters[0].name == list.adapters[1].name);
  adapters[0].name = L"Intel(R) UHD Graphics 770";
  adapters[0].dedicated_memory = 0;
  adapters[2].name = L"NVIDIA GeForce RTX 4070";
  const auto hybrid = EnumerateGpuAdapters();
  CHECK(hybrid.adapters.size() == 2 && hybrid.error.empty());
  CHECK(hybrid.adapters[0].index == 0 && hybrid.adapters[0].dedicated_mb == 0 &&
        hybrid.adapters[0].name == "Intel(R) UHD Graphics 770");
  CHECK(hybrid.adapters[1].index == 2 &&
        hybrid.adapters[1].name == "NVIDIA GeForce RTX 4070");
  std::string err;
  { auto selected = OpenAdapter(2, err); CHECK(selected.get() == &adapters[2]); }
  CHECK(!OpenAdapter(1, err) && err.find("software") != std::string::npos);
  CHECK(!OpenAdapter(8, err) && err.find("GPU 8 unavailable") != std::string::npos);
  CHECK(!OpenAdapter(-1, err));
  adapters[2].fail_desc = true;
  CHECK(!OpenAdapter(2, err));
  CHECK(!EnumerateGpuAdapters().error.empty());
  adapters[2].fail_desc = false;
  factory.fail_enum = true;
  CHECK(!EnumerateGpuAdapters().error.empty());
  factory.fail_enum = false;
  fail_factory = true;
  CHECK(!EnumerateGpuAdapters().error.empty());
  CHECK(!OpenAdapter(2, err));
  CHECK(!factory.refs && !adapters[0].refs && !adapters[1].refs && !adapters[2].refs);
  CHECK(before == GetModuleHandleW(L"d3d12.dll"));
  OrtApi fake{}; fake.SessionOptionsAppendExecutionProvider = AppendOptions; api = &fake;
  OrtDmlApi dml{}; dml.SessionOptionsAppendExecutionProvider_DML = Append;
  for (int index : {0, 2, 9}) for (bool meta : {false, true}) {
    CHECK(AppendSharedProvider(&dml, nullptr, meta, index, err));
    CHECK(seen_id == index);
  }
  CHECK(!AppendSharedProvider(&dml, nullptr, true, -1, err) && provider_calls == 6);
  OrtOptions options; options.gpu_adapter = 2;
  CHECK(DescribeProvider(options).find("device 2") != std::string::npos);
  puts("GPU adapter discovery, exact selection, provider variants and failure reporting passed");
}
