// Checks every supported runtime: SKSE would load the built plugin there, and
// each Address Library id in ids.h resolves in that runtime's file. VR has no
// file (its ids are exe offsets), so there only the single ids are checked:
// each has an offset.
//   runtimes_test <BetterFacialAnimations.dll> <dir with Data\SKSE\Plugins>
// ./build.sh test builds and runs it (under wine on Linux).
#include <windows.h>

#include <cstdint>
#include <cstdio>

#include "../src/ids.h"
#include "../src/skse.h"

// skse.cc names its log after the plugin.
extern "C" {
skse::SKSEPluginVersionData SKSEPlugin_Version = {1, 0, "runtimes_test"};
}

namespace {

using skse::Version;

constexpr uint32_t kSupported[] = {
    skse::kVrRuntime,    Version(1, 5, 97),
    Version(1, 6, 353),  Version(1, 6, 640),
    Version(1, 6, 659, skse::kGog), Version(1, 6, 1170),
    Version(1, 6, 1179, skse::kGog),
    Version(1, 7, 99),   Version(1, 7, 104),
};
// Neighbours of the supported ones, which the plugin has no ids or layout for.
constexpr uint32_t kUnsupported[] = {
    Version(1, 3, 64),
    Version(1, 4, 15),
    Version(1, 5, 80),
    Version(1, 6, 629),
    Version(1, 6, 659),
    Version(1, 6, 353, skse::kGog),
    Version(1, 6, 1130),
    Version(1, 6, 1170, skse::kGog),
};

int failures;

void Check(bool ok, const char* what, uint32_t runtime, uint64_t id = 0) {
  if (ok) return;
  ++failures;
  printf("FAIL %u.%u.%u%s: %s %llu\n", runtime >> 24, (runtime >> 16) & 0xFF,
         (runtime >> 4) & 0xFFF, (runtime & 0xF) == skse::kGog ? " GOG" : "",
         what, static_cast<unsigned long long>(id));
}

// What SKSE does with the plugin: SE's and VR's call SKSEPlugin_Query, AE's
// matches the runtime against SKSEPlugin_Version (whose versionIndependence is
// 0).
bool SkseLoads(HMODULE dll, uint32_t runtime) {
  if (runtime < Version(1, 6, 0)) {
    auto query = reinterpret_cast<bool (*)(const skse::SKSEInterface*,
                                           skse::PluginInfo*)>(
        GetProcAddress(dll, "SKSEPlugin_Query"));
    skse::SKSEInterface iface = {};
    iface.runtimeVersion = runtime;
    skse::PluginInfo info = {};
    return query && query(&iface, &info) && info.infoVersion == 1 &&
           info.name && *info.name;
  }
  auto* data = reinterpret_cast<const skse::SKSEPluginVersionData*>(
      GetProcAddress(dll, "SKSEPlugin_Version"));
  if (!data || data->dataVersion != 1) return false;
  for (uint32_t v : data->compatibleVersions) {
    if (!v) break;
    if (v == runtime) return true;
  }
  return false;
}

// Every id resolves unless ids.h marks it absent on this runtime.
template <size_t N>
void CheckIds(const skse::Rel (&ids)[N], uint32_t runtime) {
  for (const skse::Rel& id : ids)
    Check(!id.Get() || skse::Addr(id), "id not in Address Library", runtime,
          id.Get());
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    printf("usage: %s <plugin dll> <dir with Data\\SKSE\\Plugins>\n", argv[0]);
    return 2;
  }
  HMODULE dll = LoadLibraryA(argv[1]);
  if (!dll) {
    printf("FAIL cannot load %s (error %lu)\n", argv[1], GetLastError());
    return 1;
  }
  if (!SetCurrentDirectoryA(argv[2])) {
    printf("FAIL no directory %s\n", argv[2]);
    return 1;
  }
  for (uint32_t runtime : kUnsupported)
    Check(!SkseLoads(dll, runtime), "SKSE would load the plugin", runtime);
  for (uint32_t runtime : kSupported) {
    Check(SkseLoads(dll, runtime), "SKSE would not load the plugin", runtime);
    skse::SetRuntime(runtime);
    if (!skse::LoadAddressLibrary()) {
      Check(false, "Address Library not loaded", runtime);
      continue;
    }
    CheckIds(kIds, runtime);
    for (const skse::Rel& id : kIds) Check(id.Get(), "id missing", runtime);
    CheckIds(kOptionalIds, runtime);
    CheckIds(kSpeakCallers, runtime);
    CheckIds(kResponseCallers, runtime);
    CheckIds(kFaceCallers, runtime);
    CheckIds(kMorphCallers, runtime);
  }
  printf("%s: %zu runtimes, %d failures\n", failures ? "FAILED" : "ok",
         sizeof(kSupported) / sizeof(kSupported[0]), failures);
  return failures ? 1 : 0;
}
