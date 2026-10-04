// SKSE ABI, log, Address Library and engine layout (SE 1.5.97, AE 1.6.640 and
// 1.6.1170+).
#ifndef SRC_SKSE_H_
#define SRC_SKSE_H_

#include <windows.h>

#include <cstdint>

namespace skse {

// SKSE ABI (skse64 PluginAPI.h); field names are the ABI's.
struct SKSEPluginVersionData {
  uint32_t dataVersion;
  uint32_t pluginVersion;
  char name[256];
  char author[256];
  char supportEmail[252];
  uint32_t versionIndependenceEx;
  uint32_t versionIndependence;
  uint32_t compatibleVersions[16];
  uint32_t seVersionRequired;
};
static_assert(sizeof(SKSEPluginVersionData) == 0x350, "SKSEPluginVersionData");

// SE's SKSE (1.5.97) asks SKSEPlugin_Query for this instead.
struct PluginInfo {
  uint32_t infoVersion;
  const char* name;
  uint32_t version;
};

struct SKSEInterface {
  uint32_t skseVersion, runtimeVersion, editorVersion, isEditor;
  void* (*QueryInterface)(uint32_t);
  uint32_t (*GetPluginHandle)();
  uint32_t (*GetReleaseIndex)();
  const void* (*GetPluginInfo)(const char*);
};

struct SKSEMessage {
  const char* sender;
  uint32_t type;
  uint32_t dataLen;
  void* data;
};

struct SKSEMessagingInterface {
  uint32_t interfaceVersion;
  bool (*RegisterListener)(uint32_t handle, const char* sender,
                           void (*cb)(SKSEMessage*));
};

constexpr uint32_t kInterfaceMessaging = 5;
constexpr uint32_t kMessageDataLoaded = 8;

// SKSE's packed version (skse_version.h); the low nibble is the store.
enum Store : uint32_t { kSteam, kGog, kEpic };
constexpr uint32_t Version(uint32_t major, uint32_t minor, uint32_t build,
                           Store store = kSteam) {
  return (major << 24) | (minor << 16) | (build << 4) | store;
}

// Records the runtime: the My Games folder per store and the Address Library
// file. Call first in SKSEPlugin_Load.
void SetRuntime(uint32_t runtime_version);
// The runtime set above is SE (1.5.x) rather than AE (1.6+); AE until set.
bool IsSe();
// <plugin name>.log in that folder's SKSE directory, created (truncated) by
// the first line written. Log is a no-op while logging is off; Report always
// writes (what a bug report needs), up to kMaxReportLines.
void SetLogging(bool on);
void Log(const char* fmt, ...);
void Report(const char* fmt, ...);
constexpr int kMaxReportLines = 3000;

// A value that differs between SE and AE, like CommonLibSSE-NG's
// RelocationID: an Address Library id or a member offset. An id of 0 does not
// exist on that runtime.
struct Rel {
  uint64_t se, ae;
  uint64_t Get() const { return IsSe() ? se : ae; }
};

// Address Library for the runtime set above (version format 1 on SE,
// versionlib format 2, or 5 from 1.7.99). Also records the executable's image
// base; Addr returns 0 for unknown ids.
bool LoadAddressLibrary();
uint64_t ImageBase();
uint64_t Addr(uint64_t id);
inline uint64_t Addr(Rel id) { return id.Get() ? Addr(id.Get()) : 0; }

// BSFaceGenAnimationData layout.
constexpr uint32_t kFgExpr3 = 0xC0;
constexpr uint32_t kFgMod3 = 0x100;
constexpr uint32_t kFgPhon3 = 0x140;
constexpr uint32_t kFgPhon1Timer = 0x12C;
constexpr uint32_t kFgLock = 0x220;
constexpr uint32_t kFgDialogue = 0x228;

struct KeyframeMultiple {
  void* vtbl;
  uint32_t type;
  float timer;
  float* values;
  uint32_t count;
  bool is_updated;
};

struct ResId {
  uint32_t file;
  uint32_t ext;
  uint32_t dir;
};

inline uint64_t KeyOf(const ResId& r) {
  return (static_cast<uint64_t>(r.dir) << 32) | r.file;
}

// RE::BSSpinLock: recursive spin lock owned by a thread id.
struct SpinLock {
  volatile LONG owner;
  volatile LONG count;
  void Lock();
  void Unlock();
};

// The head's active dialogue line: lip resource id, or nullptr when there is
// no line or its lip resource is not loaded yet.
const ResId* ActiveLine(uint8_t* fg);

}  // namespace skse

// Defined (and exported) by the plugin.
extern "C" skse::SKSEPluginVersionData SKSEPlugin_Version;

#endif  // SRC_SKSE_H_
