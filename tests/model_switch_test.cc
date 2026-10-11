// Real worker selection, cache invalidation and mapping lifetime with fake
// nets.
#include <cstdlib>

#include "a2f/inference/ort_net.h"
#define CHECK(x)                                      \
  do {                                                \
    if (!(x)) {                                       \
      fprintf(stderr, "line %d: %s\n", __LINE__, #x); \
      std::exit(1);                                   \
    }                                                 \
  } while (0)
namespace a2f {
int alive = 0, destroyed = 0, created = 0;
bool fail_gpu = false;
struct SwitchNet : Net {
  int version;
  explicit SwitchNet(int v) : version(v) {
    CHECK(++alive == 1);
    ++created;
  }
  ~SwitchNet() override {
    --alive;
    ++destroyed;
  }
  int Version() const override { return version; }
  int Batch() const override { return 1; }
  bool Run(const float*, const float*, int, float*, std::string&) override {
    return true;
  }
};
std::unique_ptr<Net> SwitchCreate(int version, bool gpu, std::string& err,
                                  std::string& info) {
  if (gpu && fail_gpu) {
    err = "test GPU failure";
    return {};
  }
  info = std::to_string(version) + (gpu ? " GPU" : " CPU");
  return std::make_unique<SwitchNet>(version);
}
std::unique_ptr<Net> Switch2(const std::wstring&, const std::wstring&,
                             const OrtOptions& o, std::string& e,
                             std::string& i) {
  return SwitchCreate(2, o.gpu, e, i);
}
std::unique_ptr<Net> Switch3(const std::wstring&, const std::wstring&,
                             const OrtOptions& o, std::string& e,
                             std::string& i) {
  return SwitchCreate(3, o.gpu, e, i);
}
std::unique_ptr<Net> Switch4(const std::wstring&, const std::wstring&,
                             const OrtOptions& o, std::string& e,
                             std::string& i) {
  return SwitchCreate(4, o.gpu, e, i);
}
}  // namespace a2f
#define CreateOrtNet Switch2
#define CreateOrtNet3 Switch3
#define CreatePerformanceNet Switch4
#include "../src/plugin/better_facial_animations.cc"
#undef CreateOrtNet
#undef CreateOrtNet3
#undef CreatePerformanceNet

void Select(int network, bool cpu = false) {
  using namespace bfa;
  {
    ScopedLock guard(settings_lock);
    for (const Param& p : kParams) {
      if (!strcmp(p.key, "UsePerformanceModel")) *p.flag = network == 4;
      if (!strcmp(p.key, "UseOldModel")) *p.flag = network == 2;
      if (!strcmp(p.key, "UseV3Model")) *p.flag = network == 3;
      if (!strcmp(p.key, "ForceCPU")) *p.flag = cpu;
    }
  }
  WakeWorker();
}
int main() {
  using namespace bfa;
  wchar_t temp[MAX_PATH], root[MAX_PATH], cwd[32768];
  CHECK(GetCurrentDirectoryW(32768, cwd));
  CHECK(GetTempPathW(MAX_PATH, temp));
  CHECK(GetTempFileNameW(temp, L"bfa", 0, root));
  CHECK(DeleteFileW(root) && CreateDirectoryW(root, nullptr));
  CHECK(SetCurrentDirectoryW(root));
  for (const wchar_t* dir :
       {L"Data", L"Data\\SKSE", L"Data\\SKSE\\Plugins", kPluginDir})
    CHECK(CreateDirectoryW(dir, nullptr));
  for (const wchar_t* name : {L"A2FDirect.bin", L"A2FDirect3.bin"}) {
    CHECK(CopyFileW((std::wstring(cwd) + L"\\dist\\" + name).c_str(),
                    (PluginDir() + L"\\" + name).c_str(), TRUE));
  }
  for (const wchar_t* name :
       {L"performance.onnx", L"a2f3_fp16.onnx", L"network_fp16.onnx"}) {
    HANDLE f = CreateFileW((PluginDir() + L"\\" + name).c_str(), GENERIC_WRITE,
                           0, nullptr, CREATE_NEW, 0, nullptr);
    CHECK(f != INVALID_HANDLE_VALUE);
    CloseHandle(f);
  }
  config.network = 3;
  config.gpu = true;
  config.model = L"a2f3_fp16.onnx";
  config.independent_device = true;
  config.gpu_device = 3;
  std::string error;
  CHECK(model.Load(PluginDir() + L"\\A2FDirect3.bin", error));
  std::atomic_store(&active_model, SnapshotModel());
  ready = true;
  a2f::NetKeeper keeper(true, CreateNet, Now);
  WorkerStep(keeper, false);
  CHECK(keeper.on_gpu());
  auto completed =
      Request("same", 0, 0, false, false, SnapshotTune(), 0, world_epoch);
  queue.clear();
  completed->state = State::kReady;
  auto pending =
      Request("pending", 0, 0, false, false, SnapshotTune(), 0, world_epoch);
  const Job obsolete = queue.front();
  auto old_mapping = completed->model_state->mapping;
  metacommands.state = a2f::Metacommands::kOff;
  metacommands.error = "previous session";
  Select(4);
  CHECK(!JobAllowed(obsolete));
  WorkerStep(keeper, false);
  CHECK(SnapshotModel()->network == 4 && keeper.on_gpu() && keeper.loaded());
  CHECK(a2f::destroyed == 1 && a2f::alive == 1);
  // The new session decides metacommands for itself.
  CHECK(metacommands.state == a2f::Metacommands::kUntried &&
        metacommands.error.empty());
  CHECK(queue.empty() && cache.empty() && pending->state == State::kFailed);
  CHECK(completed->state == State::kReady &&
        completed->model_state->network == 3);
  CHECK(SnapshotModel()->mapping ==
        old_mapping);  // v3/performance share mapping
  auto performance =
      Request("same", 0, 0, false, false, SnapshotTune(), 0, world_epoch);
  CHECK(performance != completed && performance->model_state->network == 4);
  queue.clear();
  performance->state = State::kReady;
  // A missing geometry file leaves the previous session usable and reports
  // failure.
  CHECK(MoveFileW((PluginDir() + L"\\A2FDirect.bin").c_str(),
                  (PluginDir() + L"\\old.bin").c_str()));
  Select(2);
  WorkerStep(keeper, false);
  CHECK(SnapshotModel()->network == 4 && keeper.loaded() &&
        !stats.model_error.empty());
  // The selection goes back to the running model: new lines are generated
  // with it instead of being cancelled, and the worker does not retry.
  CHECK(SnapshotTune().SelectedModel() == 4);
  auto retained = Request("retained", 0, 0, false, false, SnapshotTune(), 0,
                          world_epoch);
  CHECK(retained->model_state->network == 4 && JobAllowed(queue.front()));
  queue.clear();
  const int before_failed_retry = a2f::created;
  WorkerStep(keeper, false);
  CHECK(a2f::created == before_failed_retry && !stats.model_error.empty());
  CHECK(MoveFileW((PluginDir() + L"\\old.bin").c_str(),
                  (PluginDir() + L"\\A2FDirect.bin").c_str()));
  Select(2);
  WorkerStep(keeper, false);
  CHECK(SnapshotModel()->network == 2 &&
        SnapshotModel()->mapping->network == 2);
  CHECK(completed->model_state->mapping == old_mapping &&
        old_mapping->network == 3);
  CHECK(stats.model_error.empty() && keeper.on_gpu());
  Select(2, true);
  WorkerStep(keeper, false);
  CHECK(!keeper.on_gpu() && keeper.loaded() && !SnapshotModel()->gpu);
  // Rapid selections coalesce to the final choice; no intermediate session.
  int before = a2f::created;
  Select(3);
  Select(2);
  Select(4);
  WorkerStep(keeper, false);
  CHECK(SnapshotModel()->network == 4 && a2f::created == before + 1);
  // GPU failure retains normal CPU fallback and diagnostics after switching.
  a2f::fail_gpu = true;
  Select(3);
  WorkerStep(keeper, false);
  CHECK(SnapshotModel()->network == 3 && keeper.loaded() && !keeper.on_gpu());
  CHECK(!stats.net_error.empty() && !stats.switching_model);
  a2f::fail_gpu = false;
  Select(4);
  WorkerStep(keeper, false);
  CHECK(stats.net_error.empty() && stats.lines == 0 && stats.gen_sec == 0);
  // Disabled switches never load a session until enabled again.
  for (const Param& p : kParams)
    if (!strcmp(p.key, "Enabled")) *p.flag = false;
  Select(2);
  WorkerStep(keeper, false);
  CHECK(SnapshotModel()->network == 2 && !keeper.loaded() && a2f::alive == 0);
  for (const Param& p : kParams)
    if (!strcmp(p.key, "Enabled")) *p.flag = true;
  WorkerStep(keeper, false);
  CHECK(keeper.loaded());
  keeper.Unload();
  CHECK(a2f::alive == 0);
  for (const wchar_t* name :
       {L"A2FDirect.bin", L"A2FDirect3.bin", L"performance.onnx",
        L"a2f3_fp16.onnx", L"network_fp16.onnx"})
    CHECK(DeleteFileW((PluginDir() + L"\\" + name).c_str()));
  for (const wchar_t* dir :
       {kPluginDir, L"Data\\SKSE\\Plugins", L"Data\\SKSE", L"Data"})
    CHECK(RemoveDirectoryW(dir));
  CHECK(SetCurrentDirectoryW(cwd) && RemoveDirectoryW(root));
  puts(
      "live model switching, resource lifetime, mapping retention and failure "
      "recovery passed");
}
