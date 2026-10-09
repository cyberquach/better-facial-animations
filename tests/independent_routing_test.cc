// Exercise the plugin's real mode selection without loading graphics DLLs.
#include "a2f/inference/ort_net.h"
#include <cstdlib>
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "line %d: %s\n", __LINE__, #x); std::exit(1); } } while (0)
namespace a2f {
int creates = 0, version_seen = 0;
OrtOptions seen;
class Cpu : public Net {
 public:
  int Batch() const override { return 1; }
  bool Run(const float*, const float*, int, float*, std::string&) override { return true; }
};
bool allow_cpu = false;
std::unique_ptr<Net> Probe2(const std::wstring&, const std::wstring&,
                           const OrtOptions& o, std::string& err, std::string&) {
  ++creates; seen = o; version_seen = 2;
  if (!o.gpu && allow_cpu) return std::make_unique<Cpu>();
  err = "injected failure";
  return nullptr;
}
std::unique_ptr<Net> Probe3(const std::wstring& dir, const std::wstring& model,
                           const OrtOptions& o, std::string& err, std::string& info) {
  auto result = Probe2(dir, model, o, err, info); version_seen = 3; return result;
}
std::unique_ptr<Net> ProbePerformance(const std::wstring& dir,
                                      const std::wstring& model,
                                      const OrtOptions& o, std::string& err,
                                      std::string& info) {
  auto result = Probe2(dir, model, o, err, info);
  version_seen = 4;
  return result;
}
}
#define CreatePerformanceNet ProbePerformance
#define CreateOrtNet Probe2
#define CreateOrtNet3 Probe3
#include "../src/plugin/better_facial_animations.cc"
#undef CreatePerformanceNet
#undef CreateOrtNet
#undef CreateOrtNet3
int main() {
  using namespace bfa;
  CHECK(!config.independent_device);
  // Use a temporary game-like directory rather than changing the repository.
  wchar_t temp[MAX_PATH], root[MAX_PATH], old[32768];
  CHECK(GetCurrentDirectoryW(32768, old));
  CHECK(GetTempPathW(MAX_PATH, temp));
  CHECK(GetTempFileNameW(temp, L"bfa", 0, root));
  CHECK(DeleteFileW(root) && CreateDirectoryW(root, nullptr));
  CHECK(SetCurrentDirectoryW(root));
  for (const wchar_t* dir : {L"Data", L"Data\\SKSE", L"Data\\SKSE\\Plugins",
                             L"Data\\SKSE\\Plugins\\BetterFacialAnimations"})
    CHECK(CreateDirectoryW(dir, nullptr));
  config.model = L"test.onnx";
  const auto file = PluginDir() + L"\\" + config.model;
  HANDLE f = CreateFileW(file.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW, 0, nullptr);
  CHECK(f != INVALID_HANDLE_VALUE); CloseHandle(f);
  config.independent_device = true;
  for (int network : {2, 3, 4}) {
    config.network = network;
    model.network = network == 4 ? 3 : network;
    for (bool gpu : {false, true}) {
      const int before = a2f::creates;
      std::string err, info;
      CHECK(!CreateNet(gpu, err, info));
      CHECK(a2f::creates == before + 1 && a2f::version_seen == network);
      CHECK(a2f::seen.gpu == gpu && a2f::seen.independent_device);
      CHECK(a2f::seen.metacommands);
    }
  }
  a2f::allow_cpu = true;
  double now = 0;
  a2f::NetKeeper keeper(true, CreateNet, [&] { return now; });
  CHECK(keeper.Ensure() && !keeper.on_gpu() && keeper.attempts() == 1);
  CHECK(keeper.error().find("independent device:") == 0);
  const int before_retry = a2f::creates;
  CHECK(keeper.Ensure() && a2f::creates == before_retry);
  now = keeper.retry_at();
  CHECK(keeper.Ensure() && !keeper.on_gpu() && keeper.attempts() == 2);
  CHECK(a2f::creates == before_retry + 1); // retry retains the CPU session
  const auto why = a2f::DiagnoseNet(true, keeper.error(), 2, {});
  CHECK(!why.empty() && why[0].find("Independent GPU device") == 0);
  CHECK(metacommands.state == a2f::Metacommands::kUntried);
  CHECK(!a2f::DeviceRedirected());
  config.network = 4;
  model.network = 3; // Performance uses the v3 geometry mapping.
  for (bool independent : {false, true}) {
    config.independent_device = independent;
    const int before = a2f::creates;
    std::string err, info;
    CHECK(CreateNet(false, err, info));
    CHECK(a2f::creates == before + 1 && a2f::version_seen == 4);
    CHECK(!a2f::DeviceRedirected());
  }
  CHECK(DeleteFileW(file.c_str()));
  for (const wchar_t* dir : {L"Data\\SKSE\\Plugins\\BetterFacialAnimations",
                             L"Data\\SKSE\\Plugins", L"Data\\SKSE", L"Data"})
    CHECK(RemoveDirectoryW(dir));
  CHECK(SetCurrentDirectoryW(old) && RemoveDirectoryW(root));
  puts("both networks: strict GPU path and CPU selection without legacy retries passed");
}
