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
Object dml;
// A real COM interface shape: releasing ORT must precede our queue/device
// references, including when a Session is moved into either model wrapper.
class Queue : public ID3D12CommandQueue {
 public:
  ULONG refs = 1;
  HRESULT STDMETHODCALLTYPE QueryInterface(REFIID, void**) override { return E_NOINTERFACE; }
  ULONG STDMETHODCALLTYPE AddRef() override { return ++refs; }
  ULONG STDMETHODCALLTYPE Release() override {
    CHECK(refs == 1 && releases == 2 && wrapper.refs == 1 && dml.refs == 1);
    return --refs;
  }
  HRESULT STDMETHODCALLTYPE GetPrivateData(REFGUID, UINT*, void*) override { return E_NOTIMPL; }
  HRESULT STDMETHODCALLTYPE SetPrivateData(REFGUID, UINT, const void*) override { return E_NOTIMPL; }
  HRESULT STDMETHODCALLTYPE SetPrivateDataInterface(REFGUID, const IUnknown*) override { return E_NOTIMPL; }
  HRESULT STDMETHODCALLTYPE SetName(const WCHAR*) override { return S_OK; }
  HRESULT STDMETHODCALLTYPE GetDevice(REFIID, void**) override { return E_NOTIMPL; }
  void STDMETHODCALLTYPE UpdateTileMappings(ID3D12Resource*, UINT,
      const D3D12_TILED_RESOURCE_COORDINATE*, const D3D12_TILE_REGION_SIZE*,
      ID3D12Heap*, UINT, const D3D12_TILE_RANGE_FLAGS*, const UINT*, const UINT*,
      D3D12_TILE_MAPPING_FLAGS) override {}
  void STDMETHODCALLTYPE CopyTileMappings(ID3D12Resource*,
      const D3D12_TILED_RESOURCE_COORDINATE*, ID3D12Resource*,
      const D3D12_TILED_RESOURCE_COORDINATE*, const D3D12_TILE_REGION_SIZE*,
      D3D12_TILE_MAPPING_FLAGS) override {}
  void STDMETHODCALLTYPE ExecuteCommandLists(UINT, ID3D12CommandList* const*) override {}
  void STDMETHODCALLTYPE SetMarker(UINT, const void*, UINT) override {}
  void STDMETHODCALLTYPE BeginEvent(UINT, const void*, UINT) override {}
  void STDMETHODCALLTYPE EndEvent() override {}
  HRESULT STDMETHODCALLTYPE Signal(ID3D12Fence*, UINT64) override { return E_NOTIMPL; }
  HRESULT STDMETHODCALLTYPE Wait(ID3D12Fence*, UINT64) override { return E_NOTIMPL; }
  HRESULT STDMETHODCALLTYPE GetTimestampFrequency(UINT64*) override { return E_NOTIMPL; }
  HRESULT STDMETHODCALLTYPE GetClockCalibration(UINT64*, UINT64*) override { return E_NOTIMPL; }
#ifdef WIDL_EXPLICIT_AGGREGATE_RETURNS
  D3D12_COMMAND_QUEUE_DESC* STDMETHODCALLTYPE GetDesc(D3D12_COMMAND_QUEUE_DESC* out) override {
    *out = {};
    return out;
  }
#else
  D3D12_COMMAND_QUEUE_DESC STDMETHODCALLTYPE GetDesc() override { return {}; }
#endif
} queue;

void ORT_API_CALL ReleaseSession(OrtSession*) {
  CHECK(wrapper.refs == 1 && dml.refs == 1 && queue.refs == 1);
  ++releases;
}
void ORT_API_CALL ReleaseMemory(OrtMemoryInfo*) {
  CHECK(wrapper.refs == 1 && dml.refs == 1 && queue.refs == 1);
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
    wrapper.refs = dml.refs = queue.refs = 1;
    releases = 0;
    {
      Session s;
      s.wrappers.emplace_back(&wrapper);
      s.dml_device.reset(&dml);
      s.queue.reset(&queue);
      s.sess.reset(reinterpret_cast<OrtSession*>(1));
      s.mem.reset(reinterpret_cast<OrtMemoryInfo*>(1));
      if (version == 2) {
        OrtNet net(std::move(s.sess), std::move(s.mem), 1,
                   std::move(s.device), std::move(s.dml_device), std::move(s.queue),
                   std::move(s.wrappers));
      } else if (version == 3) {
        OrtNet3 net(std::move(s.sess), std::move(s.mem),
                    std::move(s.device), std::move(s.dml_device), std::move(s.queue),
                    std::move(s.wrappers));
      }
    }
    CHECK(releases == 2 && wrapper.refs == 0 && dml.refs == 0 && queue.refs == 0);
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
