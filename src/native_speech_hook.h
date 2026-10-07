#ifndef SRC_NATIVE_SPEECH_HOOK_H_
#define SRC_NATIVE_SPEECH_HOOK_H_

#include <cstdint>

#include "skse.h"

namespace bfa {
// BFA-supported desktop runtimes only. HookNativeSpeech still checks the entry
// bytes; an enabled runtime is not a claim of tested DBVO playback.
inline constexpr bool NativeSpeechRuntimeSupported(uint32_t runtime) {
  return runtime == skse::Version(1, 5, 97) ||
         runtime == skse::Version(1, 6, 640) ||
         runtime == skse::Version(1, 6, 1170) ||
         runtime == skse::Version(1, 6, 1179, skse::kGog) ||
         runtime == skse::Version(1, 7, 99) ||
         runtime == skse::Version(1, 7, 104);
}

// Skyrim 1.5.97, 1.6.1170 and 1.7.104 SpeakVoiceLine begin with
// mov rax,rsp; mov [rax+20h],r9d. Neither instruction is RIP relative or
// changes RSP. Do not extend to other prologues without relocation review.
inline constexpr uint8_t kNativeSpeechPrefix[] =
    {0x48, 0x8B, 0xC4, 0x44, 0x89, 0x48, 0x20};

// Called at SKSE PostLoad, before gameplay. Refuse existing entry detours;
// existing call-site chains remain intact. The trampoline lives until exit.
void* HookNativeSpeech(uint8_t* entry, void* hook);

}  // namespace bfa

#endif  // SRC_NATIVE_SPEECH_HOOK_H_
