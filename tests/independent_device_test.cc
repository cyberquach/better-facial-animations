// Production factory contract without a GPU, graphics hooks or third-party DLLs.
#include "../src/inference/ort_net.cc"
#include <cstdlib>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "line %d: %s\n", __LINE__, #x); std::exit(1); } } while (0)

namespace {
class Factory : public a2f::independent::Factory {
 public:
  ULONG refs = 0;
  int creates = 0;
  UINT flags = 0;
  HRESULT flag_result = S_OK, create_result = DXGI_ERROR_UNSUPPORTED;
  bool null_success = false;
  HRESULT STDMETHODCALLTYPE QueryInterface(REFIID, void**) override { return E_NOINTERFACE; }
  ULONG STDMETHODCALLTYPE AddRef() override { return ++refs; }
  ULONG STDMETHODCALLTYPE Release() override { CHECK(refs); return --refs; }
  HRESULT STDMETHODCALLTYPE InitializeFromGlobalState() override { CHECK(false); return E_FAIL; }
  HRESULT STDMETHODCALLTYPE ApplyToGlobalState() override { CHECK(false); return E_FAIL; }
  HRESULT STDMETHODCALLTYPE SetFlags(UINT f) override { flags = f; return flag_result; }
  UINT STDMETHODCALLTYPE GetFlags() override { return flags; }
  HRESULT STDMETHODCALLTYPE GetConfigurationInterface(REFCLSID, REFIID, void**) override { CHECK(false); return E_FAIL; }
  HRESULT STDMETHODCALLTYPE EnableExperimentalFeatures(UINT, const IID*, void*, UINT*) override { CHECK(false); return E_FAIL; }
  HRESULT STDMETHODCALLTYPE CreateDevice(IUnknown* adapter, D3D_FEATURE_LEVEL level,
                                         REFIID iid, void** out) override {
    ++creates;
    CHECK(!adapter && level == D3D_FEATURE_LEVEL_11_0);
    CHECK(iid == __uuidof(ID3D12Device));
    CHECK(flags == 4);  // neither return an existing device nor store a singleton
    // Opaque identity only; the success test releases ownership without
    // dereferencing it. No fake graphics implementation is exercised.
    *out = SUCCEEDED(create_result) && !null_success ? &identity : nullptr;
    return create_result;
  }
  int identity = 0;
} factory;
OrtStatus* ORT_API_CALL NeverDml1(OrtSessionOptions*, IUnknown*,
                                  ID3D12CommandQueue*) {
  CHECK(false);  // never reached without a DirectML device
  return nullptr;
}
HRESULT get_result = S_OK;
bool null_factory = false;
HRESULT WINAPI Get(REFCLSID clsid, REFIID iid, void** out) {
  CHECK(clsid == a2f::independent::kFactoryClass);
  CHECK(iid == a2f::independent::kFactoryInterface);
  *out = nullptr;
  if (SUCCEEDED(get_result) && !null_factory) { factory.AddRef(); *out = &factory; }
  return get_result;
}
}
int main() {
  using namespace a2f;
  std::string err;
  CHECK(!MakeIndependentDevice(nullptr, err));
  CHECK(err.find("unavailable") != std::string::npos);
  get_result = E_NOINTERFACE;
  CHECK(!MakeIndependentDevice(Get, err) && !factory.refs && !factory.creates);
  get_result = S_OK;
  null_factory = true;
  CHECK(!MakeIndependentDevice(Get, err));
  null_factory = false;
  factory.flag_result = E_INVALIDARG;
  CHECK(!MakeIndependentDevice(Get, err) && !factory.refs && !factory.creates);
  factory.flag_result = S_OK;
  for (HRESULT failure : {DXGI_ERROR_UNSUPPORTED, DXGI_ERROR_ALREADY_EXISTS,
                           DXGI_ERROR_DEVICE_REMOVED, E_OUTOFMEMORY}) {
    factory.create_result = failure;
    const int before = factory.creates;
    CHECK(!MakeIndependentDevice(Get, err));
    CHECK(factory.creates == before + 1 && !factory.refs);
    CHECK(err.find("shared-device fallback disabled") != std::string::npos);
  }
  factory.create_result = S_OK;
  factory.null_success = true;
  CHECK(!MakeIndependentDevice(Get, err) && !factory.refs);
  factory.null_success = false;
  auto device = MakeIndependentDevice(Get, err);
  CHECK(device.release() == reinterpret_cast<ID3D12Device*>(&factory.identity));
  CHECK(!factory.refs);
  OrtDmlApi dml{};
  ComPtr<IUnknown> dml_device;
  CHECK(!AppendIndependentProvider(L"", &dml, nullptr, nullptr, dml_device, err));
  CHECK(err.find("DML1 unavailable") != std::string::npos && !dml_device);
  // DML1 present but no DirectML.dll beside the model: fails before DML1,
  // the device or the queue are touched.
  dml.SessionOptionsAppendExecutionProvider_DML1 = NeverDml1;
  wchar_t temp[MAX_PATH];
  CHECK(GetTempPathW(MAX_PATH, temp));
  CHECK(!AppendIndependentProvider(std::wstring(temp) + L"bfa-no-directml",
                                   &dml, nullptr, nullptr, dml_device, err));
  CHECK(err.find("DMLCreateDevice unavailable") != std::string::npos &&
        !dml_device);
  // An unsupported metacommand mode must fail before touching any graphics API.
  Session session;
  CHECK(!CreateSession(L"", true, false, true, L"", {}, 1, session, err));
  CHECK(err.find("does not support disabling metacommands") != std::string::npos);
  CHECK(!DeviceRedirected());
  puts("independent factory isolation, failures and ownership passed");
}
