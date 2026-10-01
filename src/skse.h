// SKSE ABI, log, Address Library and engine layout (AE 1.6.1170).
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

// Log file in My Games\Skyrim Special Edition\SKSE, created (truncated) the
// first time logging is turned on; Log is a no-op while it is off. True when
// this call created the file.
bool SetLogging(bool on);
void Log(const char* fmt, ...);

// Address Library (versionlib format 2). LoadAddressLibrary also records the
// executable's image base; Addr returns 0 for unknown ids.
bool LoadAddressLibrary(const char* path);
uint64_t ImageBase();
uint64_t Addr(uint64_t id);

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

#endif  // SRC_SKSE_H_
