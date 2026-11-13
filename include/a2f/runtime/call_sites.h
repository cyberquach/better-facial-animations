#pragma once

#include <windows.h>

#include <cstddef>
#include <cstdint>

#include "a2f/runtime/skse.h"

// Call-site hooks: a `call rel32` in the game is redirected through a thunk
// to a hook, and each site keeps its own original target (looked up by the
// hook from its return address, see Orig()).
//
// Another plugin may patch the same site after us and keep our thunk as its
// original. Retake() puts our hook back in front of that plugin and points
// the thunk the plugin kept straight at our original, so the chain is
// site -> hook -> other plugin -> original and never comes back into the
// hook. Looping there froze the game when both hooks tail-jumped: the return
// address stayed the site's and each took the other for its original.
namespace call_sites {

struct Site {
  uint64_t at;  // the call instruction
  void* orig;   // what the hook calls
  void* hook;
  uint8_t* thunk;  // the site's own thunk
  const char* what;
};
// Every ApplyMorphs call in the game is hooked, and Retake() may take a
// second thunk per site; the thunk page holds kThunkPage / kThunkSize = 256.
constexpr int kMaxSites = 64;
constexpr size_t kThunkSize = 16, kThunkPage = 0x1000;
inline Site sites[kMaxSites];
inline int num_sites;
inline SKSE::Trampoline trampoline("Better Facial Animations");
inline uint8_t* thunks;
inline size_t thunk_used;

// jmp [rip+2] -> fn, within rel32 reach below `near_addr`. The target is an
// aligned pointer, so it can be replaced while other threads run the thunk.
inline uint8_t* Thunk(uint64_t near_addr, void* fn) {
  if (!thunks) {
    if (trampoline.empty())
      trampoline.create(64 * 1024, reinterpret_cast<void*>(near_addr + 5));
    thunks = static_cast<uint8_t*>(trampoline.allocate(kThunkPage));
    if (!thunks) return nullptr;
  }
  if (thunk_used + kThunkSize > kThunkPage) return nullptr;
  uint8_t* t = thunks + thunk_used;
  thunk_used += kThunkSize;
  t[0] = 0xFF;
  t[1] = 0x25;
  *reinterpret_cast<uint32_t*>(t + 2) = 2;
  t[6] = t[7] = 0xCC;
  *reinterpret_cast<void**>(t + 8) = fn;
  return t;
}
inline void Retarget(uint8_t* thunk, void* fn) {
  InterlockedExchangePointer(reinterpret_cast<void**>(thunk + 8), fn);
}
inline uint64_t CallTarget(uint64_t at) {
  return at + 5 + *reinterpret_cast<int32_t*>(at + 1);
}
inline bool PatchCall(uint64_t at, void* to) {
  int64_t rel = static_cast<int64_t>(reinterpret_cast<uint64_t>(to) - (at + 5));
  if (rel != static_cast<int32_t>(rel)) return false;
  const int32_t displacement = static_cast<int32_t>(rel);
  REL::safe_write(at + 1, &displacement, sizeof(displacement));
  return true;
}

// Redirects the call at `at` to `hook` through a thunk of its own.
inline bool Hook(uint64_t at, void* hook, const char* what,
                 uint64_t near_addr) {
  if (num_sites >= kMaxSites) return false;
  void* orig = reinterpret_cast<void*>(CallTarget(at));
  uint8_t* thunk = Thunk(near_addr, hook);
  if (!thunk || !PatchCall(at, thunk)) return false;
  sites[num_sites++] = {at, orig, hook, thunk, what};
  return true;
}

// Original target of the site that called `hook`; `fallback` (the callee
// itself) for unknown callers.
//
// A plugin that patches a site after Retake() may call the thunk it kept as
// its original instead of jumping to it: the return address is then inside
// that plugin. The site is the one of this hook that no longer calls its
// thunk while the thunk still reaches the hook. Its original is used only
// when it is the single answer and not the plugin now at the site, which
// would come back into the hook.
template <typename Fn>
Fn Orig(void* ret, void* hook, void* fallback) {
  for (int i = 0; i < num_sites; ++i)
    if (sites[i].at + 5 == reinterpret_cast<uint64_t>(ret))
      return reinterpret_cast<Fn>(sites[i].orig);
  void* orig = nullptr;
  for (int i = 0; i < num_sites; ++i) {
    const Site& s = sites[i];
    if (s.hook != hook || *reinterpret_cast<void**>(s.thunk + 8) != hook)
      continue;
    if (*reinterpret_cast<uint8_t*>(s.at) == 0xE8) {
      const uint64_t target = CallTarget(s.at);
      if (target == reinterpret_cast<uint64_t>(s.thunk)) continue;
      if (target == reinterpret_cast<uint64_t>(s.orig))
        return reinterpret_cast<Fn>(fallback);
    }
    if (orig && orig != s.orig) return reinterpret_cast<Fn>(fallback);
    orig = s.orig;
  }
  return reinterpret_cast<Fn>(orig ? orig : fallback);
}

enum class Retaken {
  kKept,     // still ours
  kChained,  // another plugin patched the site: our hook now runs before it
  kLost,     // the site cannot be patched again: it is the other plugin's
};

// After other plugins loaded. Every step leaves a chain without a loop: the
// thunk the other plugin may call as its original stops reaching the hook
// before the hook starts calling that plugin.
inline Retaken Retake(Site& s, uint64_t near_addr) {
  bool call = *reinterpret_cast<uint8_t*>(s.at) == 0xE8;
  if (call && CallTarget(s.at) == reinterpret_cast<uint64_t>(s.thunk))
    return Retaken::kKept;
  Retarget(s.thunk, s.orig);
  if (!call) return Retaken::kLost;
  void* other = reinterpret_cast<void*>(CallTarget(s.at));
  uint8_t* thunk = Thunk(near_addr, s.hook);
  if (!thunk) return Retaken::kLost;
  s.orig = other;
  if (!PatchCall(s.at, thunk)) return Retaken::kLost;
  s.thunk = thunk;
  return Retaken::kChained;
}

}  // namespace call_sites
