#include "runtime_fixture.h"
// Production SimpleIni persistence across immediate process exit and restart.
#include <cstdarg>

#include "../src/plugin/better_facial_animations.cc"

#define CHECK(x)                                                 \
  do {                                                           \
    if (!(x)) {                                                  \
      fprintf(stderr, "line %d: %s (error %lu)\n", __LINE__, #x, \
              GetLastError());                                   \
      std::exit(1);                                              \
    }                                                            \
  } while (0)

namespace {
constexpr char kRelativeIni[] =
    "Data\\SKSE\\Plugins\\BetterFacialAnimations.ini";
constexpr char kInstaller[] =
    "Data\\SKSE\\Plugins\\BetterFacialAnimations\\Installer";
std::string report;
int failures = 0;

std::string FullPath(const char* path) {
  char full[32768];
  const DWORD n = GetFullPathNameA(path, sizeof full, full, nullptr);
  CHECK(n && n < sizeof full);
  return full;
}

void MakeTree() {
  CHECK(CreateDirectoryA("Data", nullptr));
  CHECK(CreateDirectoryA("Data\\SKSE", nullptr));
  CHECK(CreateDirectoryA("Data\\SKSE\\Plugins", nullptr));
}

void RemoveTree() {
  CHECK(DeleteFileA(kRelativeIni));
  CHECK(RemoveDirectoryA("Data\\SKSE\\Plugins"));
  CHECK(RemoveDirectoryA("Data\\SKSE"));
  CHECK(RemoveDirectoryA("Data"));
}

void WriteText(const std::string& path, const char* text) {
  FILE* file = fopen(path.c_str(), "wb");
  CHECK(file);
  CHECK(fputs(text, file) >= 0);
  CHECK(fclose(file) == 0);
}

void Child(const char* phase, const char* root) {
  char exe[32768];
  CHECK(GetModuleFileNameA(nullptr, exe, sizeof exe));
  std::string command = "\"" + std::string(exe) + "\" " + phase;
  STARTUPINFOA startup{};
  startup.cb = sizeof startup;
  PROCESS_INFORMATION process{};
  CHECK(CreateProcessA(exe, command.data(), nullptr, nullptr, FALSE, 0, nullptr,
                       root, &startup, &process));
  const DWORD waited = WaitForSingleObject(process.hProcess, 15000);
  if (waited != WAIT_OBJECT_0) TerminateProcess(process.hProcess, 1);
  CHECK(waited == WAIT_OBJECT_0);
  DWORD code;
  CHECK(GetExitCodeProcess(process.hProcess, &code));
  CloseHandle(process.hThread);
  CloseHandle(process.hProcess);
  CHECK(code == 0);
}
}  // namespace

extern "C" void TestReport(const char* fmt, ...) {
  char buffer[4096];
  va_list args;
  va_start(args, fmt);
  vsnprintf(buffer, sizeof buffer, fmt, args);
  va_end(args);
  report = buffer;
  ++failures;
}

int main(int argc, char** argv) {
  bfa::testing::report = TestReport;
  bfa::testing::address = [](uint64_t) -> uint64_t { return 0; };
  using namespace bfa;
  if (argc == 2) {
    const auto ini = FullPath(kRelativeIni);
    // Derive expected edits from fresh defaults, before loading saved values.
    std::vector<std::string> expected;
    for (const Param& p : kParams) {
      char value[64];
      if (p.value)
        snprintf(value, sizeof value, "%.9g", p.lo + (p.hi - p.lo) * .25f);
      expected.emplace_back(p.value ? value : *p.flag ? "0" : "1");
    }
    LoadIni();
    if (!strcmp(argv[1], "installer-write") ||
        !strcmp(argv[1], "installer-read")) {
      CHECK(config.independent_device && config.gpu);
      CHECK(SnapshotTune().SelectedModel() == a2f::kNetV2);
      CHECK(!SnapshotTune().sentiment_analysis &&
            SnapshotTune().performance_sentiment);
      CHECK(SnapshotTune().vanilla_emotions);  // Explicit main INI wins.
      CHECK(SnapshotTune().emotion_device == kCpuDevice);
      CHECK(SnapshotTune().potato_mode && !SnapshotTune().player);
      CHECK(!SnapshotTune().prefetch && SnapshotTune().disable_tongue);
      CHECK(!SnapshotTune().conversation_gaze && !SnapshotTune().logging);
      CHECK(failures == 0);
      if (!strcmp(argv[1], "installer-write")) {
        SaveIni();
        CHECK(GetPrivateProfileIntA("General", "UseOldModel", -1,
                                    ini.c_str()) == 1);
        CHECK(GetPrivateProfileIntA("General", "ForceCPU", -1, ini.c_str()) ==
              0);
        // A later installer default cannot overwrite a menu-saved choice.
        WriteText(
            std::string(kInstaller) + "\\FaceModel.ini",
            "[General]\nEnabled=0\nUseOldModel=0\nUsePerformanceModel=1\n");
        WriteText(std::string(kInstaller) + "\\EmotionDevice.ini",
                  "[General]\nEmotionDevice=0\n");
      } else {
        CHECK(SnapshotTune().enabled);
        const auto fragment = std::string(kInstaller) + "\\Startup.ini";
        HANDLE held = CreateFileA(fragment.c_str(), GENERIC_READ, 0, nullptr,
                                  OPEN_EXISTING, 0, nullptr);
        CHECK(held != INVALID_HANDLE_VALUE);
        SetNpcHotkey(70);
        LoadIni();
        CHECK(NpcHotkey() == 70 && failures == 1);
        CHECK(report.find(fragment) != std::string::npos);
        SaveIni();
        CHECK(failures == 2);
        CHECK(GetPrivateProfileIntA("General", "OverridesHotkey", -1,
                                    ini.c_str()) == 0);
        CloseHandle(held);
      }
    } else if (!strcmp(argv[1], "write")) {
      CHECK(config.independent_device);
      {
        ScopedLock guard(settings_lock);
        for (size_t i = 0; i < kParams.size(); ++i) {
          const Param& p = kParams[i];
          if (p.value) *p.value = std::strtof(expected[i].c_str(), nullptr);
          if (p.flag) *p.flag = expected[i] == "1";
        }
      }
      SetSelectedModel(a2f::kNetPerformance);
      CHECK(SnapshotTune().emotion_device == 0);
      SetEmotionDevice(3);
      SetNpcHotkey(67);
      NpcSetting npc{0, "missing.esp|001234", "Absent NPC", true};
      npc.values["UpperLip"] = .42f;
      SetNpcOverride(npc);
      SaveIni();
      CHECK(GetPrivateProfileIntA("General", "OverridesHotkey", -1,
                                  ini.c_str()) == 67);
      CHECK(GetPrivateProfileIntA("Debug", "Logging", -1, ini.c_str()) == 0);
      CHECK(failures == 0);
      ExitProcess(0);  // No waiting, shutdown callback or destructor flush.
    } else if (!strcmp(argv[1], "read")) {
      CHECK(config.independent_device && !config.gpu);
      CHECK(SnapshotTune().emotion_device == 3);
      CHECK(SnapshotTune().SelectedModel() == a2f::kNetPerformance);
      CHECK(!SnapshotTune().performance_sentiment &&
            !SnapshotTune().sentiment_analysis);
      CHECK(!SnapshotTune().enabled);
      CHECK(NpcHotkey() == 67);
      for (size_t i = 0; i < kParams.size(); ++i) {
        const Param& p = kParams[i];
        if (p.value)
          CHECK(std::fabs(*p.value -
                          std::strtof(expected[i].c_str(), nullptr)) < 1e-5f);
        if (p.flag && p.sec != Sec::kModel)
          CHECK(*p.flag == (expected[i] == "1"));
      }
      // Late-loaded NPC settings use the same file after restart.
      LoadNpcSettings();
      const auto overrides = SnapshotOverrides();
      CHECK(overrides.size() == 1 && overrides[0].blocked);
      CHECK(overrides[0].values.at("UpperLip") == .42f);
      LoadIni();
      CHECK(NpcHotkey() == 67 && SnapshotTune().force_cpu);
    } else if (!strcmp(argv[1], "legacy-devices")) {
      // An old CPU-only INI must not silently enable emotion GPU work.
      CHECK(SnapshotTune().force_cpu &&
            SnapshotTune().emotion_device == kCpuDevice);
      SetProcessingDevice(false, 3);
      CHECK(SnapshotTune().emotion_device == kCpuDevice);
      SaveIni();
      LoadIni();
      CHECK(!SnapshotTune().force_cpu && SnapshotTune().gpu_adapter == 3);
      CHECK(SnapshotTune().emotion_device == kCpuDevice);
      SetEmotionDevice(2);
      SetProcessingDevice(true, 3);
      SaveIni();
      LoadIni();
      CHECK(SnapshotTune().force_cpu && SnapshotTune().emotion_device == 2);
    } else if (!strcmp(argv[1], "failure")) {
      // Deny writes as an ordinary user; a failed save must be reported even
      // with optional logging disabled.
      HANDLE held = CreateFileA(ini.c_str(), GENERIC_READ, FILE_SHARE_READ,
                                nullptr, OPEN_EXISTING, 0, nullptr);
      CHECK(held != INVALID_HANDLE_VALUE);
      SetNpcHotkey(68);
      SaveIni();
      CHECK(failures == 1);
      CHECK(report.find("settings save failed") != std::string::npos);
      CHECK(report.find(kRelativeIni) != std::string::npos);
      CHECK(report.find("SimpleIni") != std::string::npos);
      CHECK(GetPrivateProfileIntA("General", "OverridesHotkey", -1,
                                  ini.c_str()) == 67);
      CloseHandle(held);
      SaveIni();
      CHECK(GetPrivateProfileIntA("General", "OverridesHotkey", -1,
                                  ini.c_str()) == 68);
      CHECK(failures == 1);
      // A file that cannot be read must not reset live choices or be replaced.
      held = CreateFileA(ini.c_str(), GENERIC_READ, 0, nullptr, OPEN_EXISTING,
                         0, nullptr);
      CHECK(held != INVALID_HANDLE_VALUE);
      SetNpcHotkey(69);
      LoadIni();
      CHECK(NpcHotkey() == 69 && failures == 2);
      SaveIni();
      CHECK(failures == 3 &&
            report.find("settings load failed") != std::string::npos);
      CloseHandle(held);
      CHECK(GetPrivateProfileIntA("General", "OverridesHotkey", -1,
                                  ini.c_str()) == 68);
    } else {
      CHECK(false);
    }
    return 0;
  }
  char old[MAX_PATH], temp[MAX_PATH], root[MAX_PATH];
  CHECK(GetCurrentDirectoryA(MAX_PATH, old));
  CHECK(GetTempPathA(MAX_PATH, temp));
  CHECK(GetTempFileNameA(temp, "bfs", 0, root));
  CHECK(DeleteFileA(root) && CreateDirectoryA(root, nullptr));
  CHECK(SetCurrentDirectoryA(root));
  MakeTree();
  const auto ini = FullPath(kRelativeIni);
  CHECK(WritePrivateProfileStringA("General", "IniVersion", "9", ini.c_str()));
  CHECK(WritePrivateProfileStringA("General", "IndependentGpuDevice", "1",
                                   ini.c_str()));
  CHECK(WritePrivateProfileStringA("Unowned", "Keep", "yes", ini.c_str()));
  {
    FILE* file = fopen(ini.c_str(), "ab");
    CHECK(file);
    fputs(
        "\r\n; Keep this user comment: "
        "caf\xc3\xa9\r\n[Extra]\r\nText=caf\xc3\xa9\r\n",
        file);
    CHECK(fclose(file) == 0);
  }
  Child("write", root);
  Child("read", root);
  Child("failure", root);
  char value[16];
  GetPrivateProfileStringA("Unowned", "Keep", "", value, sizeof value,
                           ini.c_str());
  CHECK(!strcmp(value, "yes"));
  {
    FILE* file = fopen(ini.c_str(), "rb");
    CHECK(file);
    std::string content;
    char buffer[4096];
    size_t n;
    while ((n = fread(buffer, 1, sizeof buffer, file)))
      content.append(buffer, n);
    CHECK(fclose(file) == 0);
    CHECK(content.find("; Keep this user comment: caf\xc3\xa9") !=
          std::string::npos);
    const auto text = content.find("Text");
    CHECK(text != std::string::npos);
    CHECK(content.find("caf\xc3\xa9", text) < content.find('\n', text));
  }
  // Independent defaults compose; ordinary installations above have none.
  CHECK(
      CreateDirectoryA("Data\\SKSE\\Plugins\\BetterFacialAnimations", nullptr));
  CHECK(CreateDirectoryA(kInstaller, nullptr));
  const std::pair<const char*, const char*> fragments[] = {
      {"GraphicsRoute", "[General]\nIndependentGpuDevice=1\n"},
      {"Startup", "[General]\nForceCPU=1\n"},
      {"FaceModel",
       "[General]\nEnabled=1\nUseOldModel=1\nUsePerformanceModel=0\n"},
      {"EmotionModel",
       "[General]\nSentimentAnalysis=0\nUsePerformanceEmotion=1\n"},
      {"EmotionDevice", "[General]\nEmotionDevice=-1\n"},
      {"VanillaEmotions", "[General]\nVanillaEmotions=0\n"},
      {"ActiveOnly", "[General]\nPotatoMode=1\n"},
      {"PlayerAnimation", "[General]\nPlayerFacialAnimations=0\n"},
      {"PrepareEarly", "[General]\nPrefetch=0\n"},
      {"Tongue", "[LocalAnimationStrength]\nDisableTongue=1\n"},
      {"Gaze", "[LocalAnimationStrength]\nConversationGaze=0\n"},
      {"Logging", "[Debug]\nLogging=0\n"},
  };
  for (const auto& [name, text] : fragments)
    WriteText(std::string(kInstaller) + "\\" + name + ".ini", text);
  WriteText(kRelativeIni,
            "[General]\nIniVersion=9\nForceCPU=0\nVanillaEmotions=1\n");
  Child("installer-write", root);
  Child("installer-read", root);
  for (const auto& [name, text] : fragments)
    CHECK(
        DeleteFileA((std::string(kInstaller) + "\\" + name + ".ini").c_str()));
  CHECK(RemoveDirectoryA(kInstaller));
  CHECK(RemoveDirectoryA("Data\\SKSE\\Plugins\\BetterFacialAnimations"));
  WriteText(kRelativeIni,
            "[General]\nIniVersion=9\nForceCPU=1\nGpuAdapter=2\n");
  Child("legacy-devices", root);
  RemoveTree();
  CHECK(SetCurrentDirectoryA(old) && RemoveDirectoryA(root));
  puts("immediate settings save, fresh restart and I/O failures: pass");
}
