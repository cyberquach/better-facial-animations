// Log, Address Library loader and engine helpers declared in skse.h.
#include "a2f/runtime/skse.h"

#include <shlobj.h>
#include <windows.h>

#include <algorithm>
#include <atomic>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace skse {
namespace {

FILE* log_file;
std::atomic<bool> log_on{false};
uint32_t runtime;
const wchar_t* my_games = L"Skyrim Special Edition";

// Opens the log on first use; false when there is none.
bool OpenLog() {
  static std::atomic<bool> tried{false};
  if (log_file || tried.exchange(true)) return log_file != nullptr;
  wchar_t* docs = nullptr;
  wchar_t path[MAX_PATH];
  if (SHGetKnownFolderPath(FOLDERID_Documents, 0, nullptr, &docs) != S_OK)
    return false;
  _snwprintf(path, MAX_PATH, L"%ls\\My Games\\%ls\\SKSE\\%hs.log", docs,
             my_games, SKSEPlugin_Version.pluginName);
  CoTaskMemFree(docs);
  log_file = _wfopen(path, L"w");
  return log_file != nullptr;
}

void Write(const char* fmt, va_list a) {
  vfprintf(log_file, fmt, a);
  fputc('\n', log_file);
  fflush(log_file);
}
}  // namespace

void Log(const char* fmt, ...) {
#ifdef ENABLE_COMMONLIBSSE_TESTING
  if (bfa::testing::log) {
    va_list args;
    va_start(args, fmt);
    char text[16384];
    vsnprintf(text, sizeof(text), fmt, args);
    va_end(args);
    bfa::testing::log("%s", text);
    return;
  }
#endif
  if (!log_on || !OpenLog()) return;
  va_list a;
  va_start(a, fmt);
  Write(fmt, a);
  va_end(a);
}

void Report(const char* fmt, ...) {
#ifdef ENABLE_COMMONLIBSSE_TESTING
  if (bfa::testing::report) {
    va_list args;
    va_start(args, fmt);
    char text[16384];
    vsnprintf(text, sizeof(text), fmt, args);
    va_end(args);
    bfa::testing::report("%s", text);
    return;
  }
#endif
  static std::atomic<int> lines{0};
  if (!OpenLog()) return;
  const int n = ++lines;
  if (n > kMaxReportLines) return;
  if (n == kMaxReportLines) fmt = "(no more diagnostics are written)";
  va_list a;
  va_start(a, fmt);
  Write(fmt, a);
  va_end(a);
}

uint32_t Runtime() { return runtime; }

void SetRuntime(uint32_t runtime_version) {
  runtime = runtime_version;
#ifdef ENABLE_COMMONLIBSSE_TESTING
  REL::Module::mock(
      REL::Version::unpack(runtime_version),
      runtime_version < Version(1, 5, 0)   ? REL::Module::Runtime::VR
      : runtime_version < Version(1, 6, 0) ? REL::Module::Runtime::SE
                                           : REL::Module::Runtime::AE,
      L"SkyrimSE.exe", reinterpret_cast<uintptr_t>(GetModuleHandleW(nullptr)));
  REL::IDDB::reset();
#endif
  // SKSE's folder per store, from the version's last field (skse_version.h)
  switch (runtime_version & 0xF) {
    case 1:
      my_games = L"Skyrim Special Edition GOG";
      break;
    case 2:
      my_games = L"Skyrim Special Edition EPIC";
      break;
  }
  if (IsVr()) my_games = L"Skyrim VR";  // its last field (1) is not a store
}

bool IsSe() {
  return runtime >= Version(1, 5, 0) && runtime < Version(1, 6, 0);
}
bool IsVr() { return runtime && runtime < Version(1, 5, 0); }

void SetLogging(bool on) { log_on = on; }

namespace {
// CommonLib terminates the game on an unknown id (and on a missing library
// file): the ids the loaded library has, so lookups can fail softly instead.
std::vector<uint32_t> known_ids;

std::wstring AddressLibraryFile() {
  const REL::Version version(runtime >> 24, (runtime >> 16) & 255,
                             (runtime >> 4) & 4095, 0);
  return std::format(L"Data\\SKSE\\Plugins\\{}-{}.bin",
                     IsSe() ? L"version" : L"versionlib",
                     version.wstring(L"-"));
}
}  // namespace

// The pinned CommonLib IDDB owns all decoding and runtime selection.
bool LoadAddressLibrary() {
  if (IsVr()) return true;
  const std::wstring file = AddressLibraryFile();
#ifdef ENABLE_COMMONLIBSSE_TESTING
  const REL::Version version(runtime >> 24, (runtime >> 16) & 255,
                             (runtime >> 4) & 4095, 0);
  if (!REL::IDDB::inject(file, version)) return false;
#else
  if (GetFileAttributesW(file.c_str()) == INVALID_FILE_ATTRIBUTES) {
    Log("%ls missing", file.c_str());
    return false;
  }
  (void)REL::IDDB::get();
#endif
  known_ids.clear();
  for (const auto& mapping : REL::Offset2ID())
    if (mapping.id <= UINT32_MAX) known_ids.push_back(mapping.id);
  std::sort(known_ids.begin(), known_ids.end());
  return true;
}
uint64_t ImageBase() { return REL::Module::get().base(); }
uint64_t Addr(uint64_t id) {
#ifdef ENABLE_COMMONLIBSSE_TESTING
  if (bfa::testing::address)
    return REL::Relocation<uintptr_t>(bfa::testing::address(id)).address();
#endif
  if (!id) return 0;
  if (!IsVr() && !std::binary_search(known_ids.begin(), known_ids.end(), id))
    return 0;
  return REL::VariantID(id, id, id).address();
}

const ResId* ActiveLine(uint8_t* fg) {
  auto* entry = reinterpret_cast<RE::BSFaceGenAnimationData*>(fg)->dialogueData;
  if (!entry) return nullptr;
  uint32_t ctrl = entry->refCount;
  if ((((ctrl & 0x70000000u) + 0xD0000000u) & 0xEFFFFFFFu) != 0)
    return nullptr;  // lip resource not loaded
  return reinterpret_cast<const ResId*>(entry);
}

}  // namespace skse
