// Production factory contract without a GPU, graphics hooks or third-party DLLs.
#include "../src/inference/ort_net.cc"
#include <cstdlib>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "line %d: %s\n", __LINE__, #x); std::exit(1); } } while (0)

namespace {
// Offline COM fakes: every unanticipated graphics call fails immediately.
class Queue : public ID3D12CommandQueue {
 public:
  ULONG refs = 0;
  int final_releases = 0;
  HRESULT STDMETHODCALLTYPE QueryInterface(REFIID, void**) override { CHECK(false); return E_NOINTERFACE; }
  ULONG STDMETHODCALLTYPE AddRef() override { return ++refs; }
  ULONG STDMETHODCALLTYPE Release() override { CHECK(refs); if (!--refs) ++final_releases; return refs; }
  HRESULT STDMETHODCALLTYPE GetPrivateData(REFGUID guid, UINT *data_size, void *data) override { CHECK(false); return {}; }
  HRESULT STDMETHODCALLTYPE SetPrivateData(REFGUID guid, UINT data_size, const void *data) override { CHECK(false); return {}; }
  HRESULT STDMETHODCALLTYPE SetPrivateDataInterface(REFGUID guid, const IUnknown *data) override { CHECK(false); return {}; }
  HRESULT STDMETHODCALLTYPE SetName(const WCHAR *name) override { CHECK(false); return {}; }
  HRESULT STDMETHODCALLTYPE GetDevice(REFIID riid, void **device) override { CHECK(false); return {}; }
  void STDMETHODCALLTYPE UpdateTileMappings(ID3D12Resource *resource, UINT region_count, const D3D12_TILED_RESOURCE_COORDINATE *region_start_coordinates, const D3D12_TILE_REGION_SIZE *region_sizes, ID3D12Heap *heap, UINT range_count, const D3D12_TILE_RANGE_FLAGS *range_flags, const UINT *heap_range_offsets, const UINT *range_tile_counts, D3D12_TILE_MAPPING_FLAGS flags) override { CHECK(false); }
  void STDMETHODCALLTYPE CopyTileMappings(ID3D12Resource *dst_resource, const D3D12_TILED_RESOURCE_COORDINATE *dst_region_start_coordinate, ID3D12Resource *src_resource, const D3D12_TILED_RESOURCE_COORDINATE *src_region_start_coordinate, const D3D12_TILE_REGION_SIZE *region_size, D3D12_TILE_MAPPING_FLAGS flags) override { CHECK(false); }
  void STDMETHODCALLTYPE ExecuteCommandLists(UINT command_list_count, ID3D12CommandList *const *command_lists) override { CHECK(false); }
  void STDMETHODCALLTYPE SetMarker(UINT metadata, const void *data, UINT size) override { CHECK(false); }
  void STDMETHODCALLTYPE BeginEvent(UINT metadata, const void *data, UINT size) override { CHECK(false); }
  void STDMETHODCALLTYPE EndEvent() override { CHECK(false); }
  HRESULT STDMETHODCALLTYPE Signal(ID3D12Fence *fence, UINT64 value) override { CHECK(false); return {}; }
  HRESULT STDMETHODCALLTYPE Wait(ID3D12Fence *fence, UINT64 value) override { CHECK(false); return {}; }
  HRESULT STDMETHODCALLTYPE GetTimestampFrequency(UINT64 *frequency) override { CHECK(false); return {}; }
  HRESULT STDMETHODCALLTYPE GetClockCalibration(UINT64 *gpu_timestamp, UINT64 *cpu_timestamp) override { CHECK(false); return {}; }
#ifdef WIDL_EXPLICIT_AGGREGATE_RETURNS
  D3D12_COMMAND_QUEUE_DESC* STDMETHODCALLTYPE GetDesc(D3D12_COMMAND_QUEUE_DESC* result) override { CHECK(false); return nullptr; }
#else
  D3D12_COMMAND_QUEUE_DESC STDMETHODCALLTYPE GetDesc() override { CHECK(false); return {}; }
#endif
} queue;
class Device : public ID3D12Device {
 public:
  ULONG refs = 0;
  int final_releases = 0;
  HRESULT STDMETHODCALLTYPE QueryInterface(REFIID, void**) override { CHECK(false); return E_NOINTERFACE; }
  ULONG STDMETHODCALLTYPE AddRef() override { return ++refs; }
  ULONG STDMETHODCALLTYPE Release() override { CHECK(refs); if (!--refs) ++final_releases; return refs; }
  HRESULT STDMETHODCALLTYPE GetPrivateData(REFGUID guid, UINT *data_size, void *data) override { CHECK(false); return {}; }
  HRESULT STDMETHODCALLTYPE SetPrivateData(REFGUID guid, UINT data_size, const void *data) override { CHECK(false); return {}; }
  HRESULT STDMETHODCALLTYPE SetPrivateDataInterface(REFGUID guid, const IUnknown *data) override { CHECK(false); return {}; }
  HRESULT STDMETHODCALLTYPE SetName(const WCHAR *name) override { CHECK(false); return {}; }
  HRESULT removed = S_OK, queue_result = S_OK;
  int queues = 0;
  UINT STDMETHODCALLTYPE GetNodeCount() override { CHECK(false); return {}; }
  HRESULT STDMETHODCALLTYPE CreateCommandQueue(const D3D12_COMMAND_QUEUE_DESC *desc, REFIID riid, void **command_queue) override { ++queues; CHECK(desc->Type == D3D12_COMMAND_LIST_TYPE_COMPUTE && desc->Priority == D3D12_COMMAND_QUEUE_PRIORITY_NORMAL && desc->Flags == D3D12_COMMAND_QUEUE_FLAG_NONE); CHECK(riid == __uuidof(ID3D12CommandQueue)); *command_queue = nullptr; if (SUCCEEDED(queue_result)) { queue.AddRef(); *command_queue = &queue; } return queue_result; }
  HRESULT STDMETHODCALLTYPE CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE type, REFIID riid, void **command_allocator) override { CHECK(false); return {}; }
  HRESULT STDMETHODCALLTYPE CreateGraphicsPipelineState(const D3D12_GRAPHICS_PIPELINE_STATE_DESC *desc, REFIID riid, void **pipeline_state) override { CHECK(false); return {}; }
  HRESULT STDMETHODCALLTYPE CreateComputePipelineState(const D3D12_COMPUTE_PIPELINE_STATE_DESC *desc, REFIID riid, void **pipeline_state) override { CHECK(false); return {}; }
  HRESULT STDMETHODCALLTYPE CreateCommandList(UINT node_mask, D3D12_COMMAND_LIST_TYPE type, ID3D12CommandAllocator *command_allocator, ID3D12PipelineState *initial_pipeline_state, REFIID riid, void **command_list) override { CHECK(false); return {}; }
  HRESULT STDMETHODCALLTYPE CheckFeatureSupport(D3D12_FEATURE feature, void *feature_data, UINT feature_data_size) override { CHECK(false); return {}; }
  HRESULT STDMETHODCALLTYPE CreateDescriptorHeap(const D3D12_DESCRIPTOR_HEAP_DESC *desc, REFIID riid, void **descriptor_heap) override { CHECK(false); return {}; }
  UINT STDMETHODCALLTYPE GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE descriptor_heap_type) override { CHECK(false); return {}; }
  HRESULT STDMETHODCALLTYPE CreateRootSignature(UINT node_mask, const void *bytecode, SIZE_T bytecode_length, REFIID riid, void **root_signature) override { CHECK(false); return {}; }
  void STDMETHODCALLTYPE CreateConstantBufferView(const D3D12_CONSTANT_BUFFER_VIEW_DESC *desc, D3D12_CPU_DESCRIPTOR_HANDLE descriptor) override { CHECK(false); }
  void STDMETHODCALLTYPE CreateShaderResourceView(ID3D12Resource *resource, const D3D12_SHADER_RESOURCE_VIEW_DESC *desc, D3D12_CPU_DESCRIPTOR_HANDLE descriptor) override { CHECK(false); }
  void STDMETHODCALLTYPE CreateUnorderedAccessView(ID3D12Resource *resource, ID3D12Resource *counter_resource, const D3D12_UNORDERED_ACCESS_VIEW_DESC *desc, D3D12_CPU_DESCRIPTOR_HANDLE descriptor) override { CHECK(false); }
  void STDMETHODCALLTYPE CreateRenderTargetView(ID3D12Resource *resource, const D3D12_RENDER_TARGET_VIEW_DESC *desc, D3D12_CPU_DESCRIPTOR_HANDLE descriptor) override { CHECK(false); }
  void STDMETHODCALLTYPE CreateDepthStencilView(ID3D12Resource *resource, const D3D12_DEPTH_STENCIL_VIEW_DESC *desc, D3D12_CPU_DESCRIPTOR_HANDLE descriptor) override { CHECK(false); }
  void STDMETHODCALLTYPE CreateSampler(const D3D12_SAMPLER_DESC *desc, D3D12_CPU_DESCRIPTOR_HANDLE descriptor) override { CHECK(false); }
  void STDMETHODCALLTYPE CopyDescriptors(UINT dst_descriptor_range_count, const D3D12_CPU_DESCRIPTOR_HANDLE *dst_descriptor_range_offsets, const UINT *dst_descriptor_range_sizes, UINT src_descriptor_range_count, const D3D12_CPU_DESCRIPTOR_HANDLE *src_descriptor_range_offsets, const UINT *src_descriptor_range_sizes, D3D12_DESCRIPTOR_HEAP_TYPE descriptor_heap_type) override { CHECK(false); }
  void STDMETHODCALLTYPE CopyDescriptorsSimple(UINT descriptor_count, const D3D12_CPU_DESCRIPTOR_HANDLE dst_descriptor_range_offset, const D3D12_CPU_DESCRIPTOR_HANDLE src_descriptor_range_offset, D3D12_DESCRIPTOR_HEAP_TYPE descriptor_heap_type) override { CHECK(false); }
#ifdef WIDL_EXPLICIT_AGGREGATE_RETURNS
  D3D12_RESOURCE_ALLOCATION_INFO* STDMETHODCALLTYPE GetResourceAllocationInfo(D3D12_RESOURCE_ALLOCATION_INFO* result, UINT visible_mask, UINT reource_desc_count, const D3D12_RESOURCE_DESC *resource_descs) override { CHECK(false); return nullptr; }
#else
  D3D12_RESOURCE_ALLOCATION_INFO STDMETHODCALLTYPE GetResourceAllocationInfo(UINT visible_mask, UINT reource_desc_count, const D3D12_RESOURCE_DESC *resource_descs) override { CHECK(false); return {}; }
#endif
#ifdef WIDL_EXPLICIT_AGGREGATE_RETURNS
  D3D12_HEAP_PROPERTIES* STDMETHODCALLTYPE GetCustomHeapProperties(D3D12_HEAP_PROPERTIES* result, UINT node_mask, D3D12_HEAP_TYPE heap_type) override { CHECK(false); return nullptr; }
#else
  D3D12_HEAP_PROPERTIES STDMETHODCALLTYPE GetCustomHeapProperties(UINT node_mask, D3D12_HEAP_TYPE heap_type) override { CHECK(false); return {}; }
#endif
  HRESULT STDMETHODCALLTYPE CreateCommittedResource(const D3D12_HEAP_PROPERTIES *heap_properties, D3D12_HEAP_FLAGS heap_flags, const D3D12_RESOURCE_DESC *desc, D3D12_RESOURCE_STATES initial_state, const D3D12_CLEAR_VALUE *optimized_clear_value, REFIID riid, void **resource) override { CHECK(false); return {}; }
  HRESULT STDMETHODCALLTYPE CreateHeap(const D3D12_HEAP_DESC *desc, REFIID riid, void **heap) override { CHECK(false); return {}; }
  HRESULT STDMETHODCALLTYPE CreatePlacedResource(ID3D12Heap *heap, UINT64 heap_offset, const D3D12_RESOURCE_DESC *desc, D3D12_RESOURCE_STATES initial_state, const D3D12_CLEAR_VALUE *optimized_clear_value, REFIID riid, void **resource) override { CHECK(false); return {}; }
  HRESULT STDMETHODCALLTYPE CreateReservedResource(const D3D12_RESOURCE_DESC *desc, D3D12_RESOURCE_STATES initial_state, const D3D12_CLEAR_VALUE *optimized_clear_value, REFIID riid, void **resource) override { CHECK(false); return {}; }
  HRESULT STDMETHODCALLTYPE CreateSharedHandle(ID3D12DeviceChild *object, const SECURITY_ATTRIBUTES *attributes, DWORD access, const WCHAR *name, HANDLE *handle) override { CHECK(false); return {}; }
  HRESULT STDMETHODCALLTYPE OpenSharedHandle(HANDLE handle, REFIID riid, void **object) override { CHECK(false); return {}; }
  HRESULT STDMETHODCALLTYPE OpenSharedHandleByName(const WCHAR *name, DWORD access, HANDLE *handle) override { CHECK(false); return {}; }
  HRESULT STDMETHODCALLTYPE MakeResident(UINT object_count, ID3D12Pageable *const *objects) override { CHECK(false); return {}; }
  HRESULT STDMETHODCALLTYPE Evict(UINT object_count, ID3D12Pageable *const *objects) override { CHECK(false); return {}; }
  HRESULT STDMETHODCALLTYPE CreateFence(UINT64 initial_value, D3D12_FENCE_FLAGS flags, REFIID riid, void **fence) override { CHECK(false); return {}; }
  HRESULT STDMETHODCALLTYPE GetDeviceRemovedReason() override { return removed; }
  void STDMETHODCALLTYPE GetCopyableFootprints(const D3D12_RESOURCE_DESC *desc, UINT first_sub_resource, UINT sub_resource_count, UINT64 base_offset, D3D12_PLACED_SUBRESOURCE_FOOTPRINT *layouts, UINT *row_count, UINT64 *row_size, UINT64 *total_bytes) override { CHECK(false); }
  HRESULT STDMETHODCALLTYPE CreateQueryHeap(const D3D12_QUERY_HEAP_DESC *desc, REFIID riid, void **heap) override { CHECK(false); return {}; }
  HRESULT STDMETHODCALLTYPE SetStablePowerState(WINBOOL enable) override { CHECK(false); return {}; }
  HRESULT STDMETHODCALLTYPE CreateCommandSignature(const D3D12_COMMAND_SIGNATURE_DESC *desc, ID3D12RootSignature *root_signature, REFIID riid, void **command_signature) override { CHECK(false); return {}; }
  void STDMETHODCALLTYPE GetResourceTiling(ID3D12Resource *resource, UINT *total_tile_count, D3D12_PACKED_MIP_INFO *packed_mip_info, D3D12_TILE_SHAPE *standard_tile_shape, UINT *sub_resource_tiling_count, UINT first_sub_resource_tiling, D3D12_SUBRESOURCE_TILING *sub_resource_tilings) override { CHECK(false); }
#ifdef WIDL_EXPLICIT_AGGREGATE_RETURNS
  LUID* STDMETHODCALLTYPE GetAdapterLuid(LUID* result) override { CHECK(false); return nullptr; }
#else
  LUID STDMETHODCALLTYPE GetAdapterLuid() override { CHECK(false); return {}; }
#endif
} device;
// IDMLDevice health query uses the documented slot 15 COM ABI.
class Dml : public IUnknown {
 public:
  ULONG refs = 0;
  int final_releases = 0;
  HRESULT STDMETHODCALLTYPE QueryInterface(REFIID, void**) override { CHECK(false); return E_NOINTERFACE; }
  ULONG STDMETHODCALLTYPE AddRef() override { return ++refs; }
  ULONG STDMETHODCALLTYPE Release() override { CHECK(refs); if (!--refs) ++final_releases; return refs; }
  HRESULT removed = S_OK;
  virtual void STDMETHODCALLTYPE Unused3() { CHECK(false); }
  virtual void STDMETHODCALLTYPE Unused4() { CHECK(false); }
  virtual void STDMETHODCALLTYPE Unused5() { CHECK(false); }
  virtual void STDMETHODCALLTYPE Unused6() { CHECK(false); }
  virtual void STDMETHODCALLTYPE Unused7() { CHECK(false); }
  virtual void STDMETHODCALLTYPE Unused8() { CHECK(false); }
  virtual void STDMETHODCALLTYPE Unused9() { CHECK(false); }
  virtual void STDMETHODCALLTYPE Unused10() { CHECK(false); }
  virtual void STDMETHODCALLTYPE Unused11() { CHECK(false); }
  virtual void STDMETHODCALLTYPE Unused12() { CHECK(false); }
  virtual void STDMETHODCALLTYPE Unused13() { CHECK(false); }
  virtual void STDMETHODCALLTYPE Unused14() { CHECK(false); }
  virtual HRESULT STDMETHODCALLTYPE GetDeviceRemovedReason() { return removed; }
} fake_dml;
int dml_creates = 0;
HRESULT dml_result = S_OK;
HRESULT WINAPI CreateDml(ID3D12Device* dev, UINT flags, REFIID iid, void** out) {
  CHECK(dev == &device && !flags && iid == a2f::independent::kDmlDevice);
  ++dml_creates;
  *out = nullptr;
  if (SUCCEEDED(dml_result)) { fake_dml.AddRef(); *out = &fake_dml; }
  return dml_result;
}
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
    *out = nullptr;
    if (SUCCEEDED(create_result) && !null_success) {
      device.AddRef(); *out = &device;
    }
    return create_result;
  }
} factory;
OrtStatus* ORT_API_CALL NeverDml1(OrtSessionOptions*, IUnknown*,
                                  ID3D12CommandQueue*) {
  CHECK(false);  // never reached without a DirectML device
  return nullptr;
}
a2f::ComPtr<IUnknown> provider_dml;
a2f::ComPtr<ID3D12CommandQueue> provider_queue;
int attachments = 0;
OrtStatus* ORT_API_CALL AttachDml1(OrtSessionOptions*, IUnknown* dml,
                                ID3D12CommandQueue* cq) {
  CHECK(dml == &fake_dml && cq == &queue);
  CHECK(!provider_dml && !provider_queue);
  dml->AddRef(); provider_dml.reset(dml);
  cq->AddRef(); provider_queue.reset(cq);
  ++attachments;
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
  auto reference = MakeIndependentDevice(Get, err);
  CHECK(reference.get() == &device && device.refs == 1);
  reference.reset();
  CHECK(device.refs == 0 && device.final_releases == 1);
  device.final_releases = 0;
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
  // Failed initialization is sticky, including partial DML/queue creation.
  for (int stage = 0; stage < 3; ++stage) {
    IndependentContext failed;
    const int creates_before = factory.creates;
    factory.create_result = stage == 0 ? E_FAIL : S_OK;
    dml_result = stage == 1 ? E_FAIL : S_OK;
    device.queue_result = E_FAIL;
    auto ref = failed.Acquire(Get, err);
    if (stage == 0) CHECK(!ref);
    else {
      CHECK(ref);
      CHECK(!InitializeIndependentProvider(failed, CreateDml, err));
      ref.reset();
      CHECK(device.refs == 1);  // failed context still owns the device
      CHECK(fake_dml.refs == (stage == 2 ? 1u : 0u));
    }
    const int creates_dml_before = dml_creates;
    const int creates_queue_before = device.queues;
    const auto failure = err;
    CHECK(!failed.Acquire(Get, err) && err == failure);
    CHECK(!InitializeIndependentProvider(failed, CreateDml, err));
    CHECK(factory.creates == creates_before + 1);
    CHECK(dml_creates == creates_dml_before && device.queues == creates_queue_before);
  }
  factory.create_result = dml_result = device.queue_result = S_OK;
  device.final_releases = fake_dml.final_releases = queue.final_releases = 0;
  // Reset the empty test singleton after the missing-library test; production
  // never resets this context. The successful context below is left alive.
  auto& context = IndependentGpu();
  CHECK(!context.device && !context.dml && !context.queue);
  context.failure.clear();
  const int creates_before = factory.creates;
  auto probe = context.Acquire(Get, err);
  CHECK(probe.get() == &device && device.refs == 2);
  probe.reset();
  CHECK(device.refs == 1 && !device.final_releases);
  CHECK(InitializeIndependentProvider(context, CreateDml, err));
  const int creates_dml_before = dml_creates, creates_queue_before = device.queues;
  dml.SessionOptionsAppendExecutionProvider_DML1 = AttachDml1;
  for (int model = 0; model < 12; ++model) {
    {
      Session next;
      next.device = OpenIndependentDevice(err);
      CHECK(next.device.get() == &device);
      CHECK(AppendIndependentProvider(L"", &dml, nullptr, next.device.get(),
                                      next.dml_device, err));
      // Model release also drops the simulated ORT provider references.
      provider_dml.reset(); provider_queue.reset();
    }
    CHECK(device.refs == 1 && fake_dml.refs == 1 && queue.refs == 1);
    CHECK(!device.final_releases && !fake_dml.final_releases && !queue.final_releases);
    { Session cpu; }  // no model/device reference exists during CPU operation
    CHECK(IndependentDeviceSupported(err));  // probe reuses the same device
  }
  CHECK(attachments == 12 && factory.creates == creates_before + 1);
  CHECK(dml_creates == creates_dml_before && device.queues == creates_queue_before);
  // DML-only loss disables the retained context even if D3D12 is healthy.
  fake_dml.removed = DXGI_ERROR_DEVICE_REMOVED;
  CHECK(!OpenIndependentDevice(err));
  CHECK(err.find("DirectML removed reason") != std::string::npos);
  fake_dml.removed = S_OK;
  CHECK(!OpenIndependentDevice(err));  // latch persists, never silently revived
  CHECK(err.find("GPU disabled until restart") != std::string::npos);
  CHECK(!AppendIndependentProvider(L"", &dml, nullptr, &device, dml_device, err));
  CHECK(attachments == 12 && factory.creates == creates_before + 1);
  CHECK(device.refs == 1 && fake_dml.refs == 1 && queue.refs == 1);
  CHECK(!device.final_releases && !fake_dml.final_releases && !queue.final_releases);
  // Independently exercise D3D12 removal before any DirectML allocation.
  {
    IndependentContext lost;
    auto ref = lost.Acquire(Get, err);
    CHECK(ref);
    device.removed = DXGI_ERROR_DEVICE_REMOVED;
    CHECK(!lost.Acquire(Get, err));
    CHECK(err.find("device removed reason") != std::string::npos);
    device.removed = S_OK;
    CHECK(!lost.Acquire(Get, err));
  }
  puts("independent context reuse, failure latch and session ownership passed");
  puts("independent factory isolation, failures and ownership passed");
}
