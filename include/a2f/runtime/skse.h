// SKSE ABI, log, Address Library and engine layout (VR 1.4.15, SE 1.5.97, AE
// 1.6.353, 1.6.640, GOG 1.6.659 and 1.6.1170+).
#ifndef INCLUDE_A2F_RUNTIME_SKSE_H_
#define INCLUDE_A2F_RUNTIME_SKSE_H_

#include "RE/Skyrim.h"
#include "SKSE/Impl/PCH.h"
#include "SKSE/SKSE.h"
#include "a2f/runtime/test_hooks.h"

namespace skse {
using SKSEPluginVersionData = SKSE::PluginVersionData;
using PluginInfo = SKSE::PluginInfo;
using SKSEInterface = SKSE::LoadInterface;
using SKSEMessage = SKSE::MessagingInterface::Message;
using SKSEMessagingInterface = SKSE::MessagingInterface;
using SKSETaskInterface = SKSE::TaskInterface;
constexpr uint32_t kInterfaceTask = SKSE::LoadInterface::kTask;
constexpr uint32_t kInterfaceMessaging = SKSE::LoadInterface::kMessaging;
constexpr uint32_t kMessagePostLoad = SKSE::MessagingInterface::kPostLoad;
constexpr uint32_t kMessagePostPostLoad =
    SKSE::MessagingInterface::kPostPostLoad;
constexpr uint32_t kMessagePreLoadGame = SKSE::MessagingInterface::kPreLoadGame;
constexpr uint32_t kMessagePostLoadGame =
    SKSE::MessagingInterface::kPostLoadGame;
constexpr uint32_t kMessageNewGame = SKSE::MessagingInterface::kNewGame;
constexpr uint32_t kMessageDataLoaded = SKSE::MessagingInterface::kDataLoaded;

// SKSE's packed version (skse_version.h); the low nibble is the store.
enum Store : uint32_t { kSteam, kGog, kEpic };
constexpr uint32_t Version(uint32_t major, uint32_t minor, uint32_t build,
                           Store store = kSteam) {
  return (major << 24) | (minor << 16) | (build << 4) | store;
}
// VR's one runtime; SKSEVR sets the last field to 1 (not a store there).
constexpr uint32_t kVrRuntime = Version(1, 4, 15) | 1;

// Records the runtime: the My Games folder per store and the Address Library
// file. Call first in SKSEPlugin_Load.
void SetRuntime(uint32_t runtime_version);
uint32_t Runtime();
// The runtime set above is SE (1.5.x) / VR (1.4.x); neither (AE) until set.
bool IsSe();
bool IsVr();
// <plugin name>.log in that folder's SKSE directory, created (truncated) by
// the first line written. Log is a no-op while logging is off; Report always
// writes (what a bug report needs), up to kMaxReportLines.
void SetLogging(bool on);
void Log(const char* fmt, ...);
void Report(const char* fmt, ...);
constexpr int kMaxReportLines = 3000;

// A value that differs between SE, AE and VR, like CommonLibSSE-NG's
// RelocationID: an Address Library id or a member offset. An id of 0 does not
// exist on that runtime. VR's ids are offsets into SkyrimVR.exe itself: it has
// one version, and its Address Library lacks most of the ids.
struct Rel {
  uint64_t se, ae, vr;
  uint64_t Get() const { return IsVr() ? vr : IsSe() ? se : ae; }
};

// Address Library for the runtime set above (version format 1 on SE,
// versionlib format 2, or 5 from 1.7.99; none on VR). Also records the
// executable's image base; Addr returns 0 for unknown ids.
bool LoadAddressLibrary();
uint64_t ImageBase();
uint64_t Addr(uint64_t id);
inline uint64_t Addr(Rel id) { return id.Get() ? Addr(id.Get()) : 0; }

using KeyframeMultiple = RE::BSFaceGenKeyframeMultiple;
using ResId = RE::BSResource::ID;
using SpinLock = RE::BSSpinLock;
#ifdef ENABLE_COMMONLIBSSE_TESTING
class ScopedFaceLock : public RE::BSSpinLockGuard {
  static SpinLock& Schedule(SpinLock& lock) {
    if (bfa::testing::before_face_lock) bfa::testing::before_face_lock();
    return lock;
  }

 public:
  explicit ScopedFaceLock(SpinLock& lock)
      : RE::BSSpinLockGuard(Schedule(lock)) {}
};
#else
using ScopedFaceLock = RE::BSSpinLockGuard;
#endif

inline uint64_t KeyOf(const ResId& r) {
  return (static_cast<uint64_t>(r.dir) << 32) | r.file;
}

// The head's active dialogue line: lip resource id, or nullptr when there is
// no line or its lip resource is not loaded yet.
const ResId* ActiveLine(uint8_t* fg);

}  // namespace skse

// Defined (and exported) by the plugin.
extern "C" __declspec(dllexport) skse::SKSEPluginVersionData SKSEPlugin_Version;

#endif  // INCLUDE_A2F_RUNTIME_SKSE_H_
