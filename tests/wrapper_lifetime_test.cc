// Exercise production ownership without a GPU or third-party graphics DLL.
#include "../src/inference/ort_net.cc"

#include <cstdlib>
#include <thread>
#include <type_traits>

namespace {
void Check(bool ok, const char* expression, int line) {
  if (ok) return;
  fprintf(stderr, "line %d: %s\n", line, expression);
  std::exit(1);
}
#define CHECK(expression) Check((expression), #expression, __LINE__)

class Object : public IUnknown {
 public:
  HRESULT STDMETHODCALLTYPE QueryInterface(REFIID, void**) override {
    return E_NOINTERFACE;
  }
  ULONG STDMETHODCALLTYPE AddRef() override { return ++refs; }
  ULONG STDMETHODCALLTYPE Release() override {
    CHECK(refs > 0);
    return --refs;
  }
  ULONG refs = 1;
};

Object wrapper;
int releases = 0;
void ORT_API_CALL ReleaseSession(OrtSession*) {
  CHECK(wrapper.refs == 1);
  ++releases;
}
void ORT_API_CALL ReleaseMemory(OrtMemoryInfo*) {
  CHECK(wrapper.refs == 1);
  ++releases;
}

void TestSelection() {
  using namespace a2f;
  {
    Object wrapped, real;
    Wrappers wrappers;
    {
      const ScopedWrappers scope(wrappers);
      void* result = &wrapped;
      SelectDevice(ComPtr<IUnknown>(&real), &result);
      CHECK(result == &real && wrappers.size() == 1u);
      static_cast<IUnknown*>(result)->Release();
      CHECK(real.refs == 0 && wrapped.refs == 1u);
    }
    wrappers.clear();
    CHECK(wrapped.refs == 0);
  }
  Object wrapped, real;
  void* result = &wrapped;
  SelectDevice(ComPtr<IUnknown>(&real), &result);
  CHECK(result == &wrapped && real.refs == 0 && wrapped.refs == 1);
  wrapped.Release();

  // An unwrapped device has two references to the same interface.
  Object native;
  native.AddRef();
  result = &native;
  Wrappers wrappers;
  const ScopedWrappers scope(wrappers);
  SelectDevice(ComPtr<IUnknown>(&native), &result);
  CHECK(result == &native && native.refs == 1 && wrappers.empty());
  native.Release();
}

void TestScopes() {
  using namespace a2f;
  static_assert(!std::is_copy_constructible<ScopedWrappers>::value);
  Wrappers outer, inner;
  CHECK(!wrappers_kept);
  {
    const ScopedWrappers first(outer);
    {
      const ScopedWrappers second(inner);
      CHECK(wrappers_kept == &inner);
    }
    CHECK(wrappers_kept == &outer);
    std::thread other([] {
      CHECK(!a2f::wrappers_kept);
      TestSelection();  // Out-of-scope calls keep their wrapper on this thread.
    });
    other.join();
    CHECK(wrappers_kept == &outer);
  }
  CHECK(!wrappers_kept);
}

int redirect_logs = 0;
void Log(const std::string& line) {
  if (line.find("D3D12CreateDevice") != std::string::npos) ++redirect_logs;
}
OrtStatus* ORT_API_CALL NoOptions(OrtSessionOptions**) noexcept {
  return reinterpret_cast<OrtStatus*>(1);
}
const char* ORT_API_CALL Message(const OrtStatus*) noexcept { return "mock"; }
void ORT_API_CALL DropStatus(OrtStatus*) noexcept {}

// The shared-device path patches onnxruntime.dll's D3D12CreateDevice import
// once per process; the independent device never touches it. This module
// has no delay import to patch, which the log reports.
void TestRedirect() {
  using namespace a2f;
  OrtApi mock{};
  mock.CreateSessionOptions = NoOptions;
  mock.GetErrorMessage = Message;
  mock.ReleaseStatus = DropStatus;
  api = &mock;
  ort_module = GetModuleHandleW(nullptr);
  SetOrtLog(Log);
  std::string err;
  Session s;
  CHECK(!CreateSession(L"", true, true, true, L"", {}, s, err));
  CHECK(redirect_logs == 0 && !DeviceRedirected());
  CHECK(!CreateSession(L"", true, true, false, L"", {}, s, err));
  CHECK(redirect_logs == 1 && !DeviceRedirected());
  CHECK(!CreateSession(L"", true, true, false, L"", {}, s, err));
  CHECK(redirect_logs == 1);
  SetOrtLog(nullptr);
  ort_module = nullptr;
  api = nullptr;
}

void TestTeardown() {
  using namespace a2f;
  OrtApi mock{};
  mock.ReleaseSession = ReleaseSession;
  mock.ReleaseMemoryInfo = ReleaseMemory;
  api = &mock;
  // Both model versions and a partially constructed session must release
  // their ORT resources before the retained wrapper, including after moves.
  for (int version : {0, 2, 3}) {
    wrapper.refs = 1;
    releases = 0;
    {
      Session s;
      s.wrappers.emplace_back(&wrapper);
      s.sess.reset(reinterpret_cast<OrtSession*>(1));
      s.mem.reset(reinterpret_cast<OrtMemoryInfo*>(1));
      if (version == 2) {
        OrtNet net(std::move(s), 1);
      } else if (version == 3) {
        OrtNet3 net(std::move(s));
      }
    }
    CHECK(releases == 2 && wrapper.refs == 0);
  }
  api = nullptr;
}

void TestFailedRetention() {
  using namespace a2f;
  OrtApi mock{};
  mock.ReleaseSession = ReleaseSession;
  mock.ReleaseMemoryInfo = ReleaseMemory;
  api = &mock;
  Object device, dml;
  wrapper.refs = 1;
  releases = 0;
  DisableGpu("injected shared GPU failure");
  {
    // Only IUnknown ownership is exercised; no D3D12 methods are invoked.
    Session s;
    s.device.reset(reinterpret_cast<ID3D12Device*>(&device));
    s.dml_device.reset(&dml);
    s.wrappers.emplace_back(&wrapper);
    s.sess.reset(reinterpret_cast<OrtSession*>(1));
    s.mem.reset(reinterpret_cast<OrtMemoryInfo*>(1));
    OrtNet3 net(std::move(s));
  }
  CHECK(releases == 0 && wrapper.refs == 1 && device.refs == 1 && dml.refs == 1);
  api = nullptr;
}
}  // namespace

int main() {
  TestScopes();
  TestSelection();
  TestTeardown();
  TestRedirect();
  TestFailedRetention();
  puts("wrapper unwrapping, nested/thread scopes, model teardown and "
       "one-time import redirect passed");
}
