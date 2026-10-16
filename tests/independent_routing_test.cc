// Exercise the plugin's real mode selection without loading graphics DLLs.
#include "a2f/inference/ort_net.h"
#include <cstdlib>
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "line %d: %s\n", __LINE__, #x); std::exit(1); } } while (0)
namespace a2f {
int creates = 0, version_seen = 0;
OrtOptions seen;
class Cpu : public Net {
 public:
  int version = 2;
  int Version() const override { return version; }
  int Batch() const override { return 1; }
  bool Run(const float*, const float*, int, float*, std::string&) override { return true; }
};
bool allow_cpu = false, allow_gpu = false;
std::unique_ptr<Net> Probe2(const std::wstring&, const std::wstring&,
                           const OrtOptions& o, std::string& err, std::string&) {
  ++creates; seen = o; version_seen = 2;
  if ((o.gpu && allow_gpu) || (!o.gpu && allow_cpu)) return std::make_unique<Cpu>();
  err = "injected failure";
  return nullptr;
}
std::unique_ptr<Net> Probe3(const std::wstring& dir, const std::wstring& model,
                           const OrtOptions& o, std::string& err, std::string& info) {
  auto result = Probe2(dir, model, o, err, info); version_seen = 3;
  if (result) static_cast<Cpu*>(result.get())->version = 3;
  return result;
}
std::unique_ptr<Net> ProbePerformance(const std::wstring& dir,
                                      const std::wstring& model,
                                      const OrtOptions& o, std::string& err,
                                      std::string& info) {
  auto result = Probe2(dir, model, o, err, info);
  version_seen = 4;
  if (result) static_cast<Cpu*>(result.get())->version = 4;
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
  config.gpu = false; // direct CPU selection honors the requested model
  for (bool independent : {false, true}) {
    config.independent_device = independent;
    for (int network : {2, 3, 4}) {
    config.network = network;
    model.network = network == 4 ? 3 : network;
    for (bool gpu : {false, true}) {
      const int before = a2f::creates;
      std::string err, info;
      CHECK(!CreateNet(gpu, err, info));
      CHECK(a2f::creates == before + (gpu && !independent ? 2 : 1) && a2f::version_seen == network);
      CHECK(a2f::seen.gpu == gpu && a2f::seen.independent_device == config.independent_device);
      CHECK(a2f::seen.metacommands == (!gpu || independent));
    }
  }
  }
  // Every selected model survives GPU failure on either route, without
  // requiring any alternate model file to exist.
  a2f::allow_cpu = true;
  for (bool independent : {false, true}) {
    config.independent_device = independent;
    for (int network : {2, 3, 4}) {
      config.network = network;
      config.gpu = true;
      model.network = network == 4 ? 3 : network;
      double time = 0;
      a2f::NetKeeper automatic(true, CreateNet, [&] { return time; });
      CHECK(!automatic.Ensure() && !automatic.loaded());
      CHECK(a2f::version_seen == network && a2f::seen.gpu);
      CHECK(config.network == network && config.model == L"test.onnx");
      config.gpu = false;
      std::string error, information;
      auto explicit_cpu = CreateNet(false, error, information);
      CHECK(explicit_cpu && explicit_cpu->Version() == network);
    }
  }
  config.independent_device = true;
  config.network = 4;
  a2f::allow_cpu = true;
  double now = 0;
  a2f::NetKeeper keeper(true, CreateNet, [&] { return now; });
  CHECK(!keeper.Ensure() && !keeper.loaded());
  CHECK(keeper.error().find("independent device:") == 0);
  const int before_retry = a2f::creates;
  CHECK(!keeper.Ensure() && a2f::creates == before_retry);
  now = 1e6;
  CHECK(!keeper.Ensure() && !keeper.loaded());
  CHECK(a2f::creates == before_retry); // waits for explicit retry
  const auto why = a2f::DiagnoseNet(true, keeper.error(), {});
  CHECK(!why.empty() && why[0].find("GPU model unavailable") == 0);
  CHECK(!a2f::DeviceRedirected());
  config.network = 4;
  model.network = 3; // Performance uses the v3 geometry mapping.
  // IndependentGpuDevice is explicit and survives saving. Retired Auto/GpuDevice
  // keys must never override it. ForceCPU keeps the chosen route for live opt-in.
  for (bool independent : {false, true}) {
  for (const char* mode : {"0", "1", "2", "3", "99"}) {
    const char* ini = ".\\Data\\SKSE\\Plugins\\BetterFacialAnimations.ini";
    CHECK(WritePrivateProfileStringA("General", "GpuDevice", mode, ini));
    CHECK(WritePrivateProfileStringA("General", "AutoGpuDevice", mode, ini));
    CHECK(WritePrivateProfileStringA("General", "IndependentGpuDevice", independent ? "1" : "0", ini));
    CHECK(WritePrivateProfileStringA("Debug", "UseDeviceWrapper", "1", ini));
    LoadIni();
    CHECK(config.independent_device == independent);
    SaveIni();
    LoadIni();
    CHECK(config.independent_device == independent);
    config.model = L"test.onnx";
    config.network = 4;
    model.network = 3;
    for (bool gpu : {false, true}) {
      const int before = a2f::creates;
      std::string err, info;
      a2f::allow_gpu = true;
      CHECK(CreateNet(gpu, err, info));
      CHECK(a2f::creates == before + 1 && a2f::version_seen == 4);
      CHECK(a2f::seen.gpu == gpu && a2f::seen.independent_device == config.independent_device);
      CHECK(!a2f::DeviceRedirected());
    }
  }
  }
  CHECK(DeleteFileW(file.c_str()));
  CHECK(DeleteFileA("Data\\SKSE\\Plugins\\BetterFacialAnimations.ini"));
  for (const wchar_t* dir : {L"Data\\SKSE\\Plugins\\BetterFacialAnimations",
                             L"Data\\SKSE\\Plugins", L"Data\\SKSE", L"Data"})
    CHECK(RemoveDirectoryW(dir));
  CHECK(SetCurrentDirectoryW(old) && RemoveDirectoryW(root));
  puts("all models: explicit independent/shared paths, CPU opt-in and persistent settings passed");
}
