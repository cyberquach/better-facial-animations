// Exercise production ownership without a GPU or third-party graphics DLL.
#include "../src/ort_net.cc"

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
  for (bool keep_wrapper : {false, true}) {
    SetUseDeviceWrapper(keep_wrapper);
    Object wrapped, real;
    Wrappers wrappers;
    {
      const ScopedWrappers scope(wrappers);
      void* result = &wrapped;
      SelectDevice(ComPtr<IUnknown>(&real), &result);
      CHECK(result == (keep_wrapper ? &wrapped : &real));
      CHECK(wrappers.size() == (keep_wrapper ? 0u : 1u));
      static_cast<IUnknown*>(result)->Release();
      CHECK(real.refs == 0);
      CHECK(wrapped.refs == (keep_wrapper ? 0u : 1u));
    }
    wrappers.clear();
    CHECK(wrapped.refs == 0);
  }
  SetUseDeviceWrapper(false);
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
        OrtNet net(std::move(s.sess), std::move(s.mem), 1,
                   std::move(s.device), std::move(s.dml_device),
                   std::move(s.wrappers));
      } else if (version == 3) {
        OrtNet3 net(std::move(s.sess), std::move(s.mem),
                    std::move(s.device), std::move(s.dml_device),
                    std::move(s.wrappers));
      }
    }
    CHECK(releases == 2 && wrapper.refs == 0);
  }
  api = nullptr;
}
}  // namespace

int main() {
  TestScopes();
  TestSelection();
  TestTeardown();
  puts("wrapper selection, nested/thread scopes and model teardown passed");
}
