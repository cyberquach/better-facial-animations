// Detours the engine's SpeakVoiceLine entry (see native_speech_hook.h):
// a rel32 jmp over its known prologue, which a trampoline replays.
#include "a2f/speech/native_speech_hook.h"

#include <windows.h>

#include <cstdint>
#include <cstring>

#include "a2f/runtime/call_sites.h"

namespace bfa {

namespace {
constexpr size_t kPrefixSize = sizeof kNativeSpeechPrefix;
constexpr size_t kJmpRel32Size = 5;  // E9 rel32, then NOPs to kPrefixSize
constexpr uint8_t kJmpIndirect[] = {0xFF, 0x25, 0, 0, 0, 0};  // jmp [rip]
// The trampoline: the prefix, jmp [rip], the continuation's address.
constexpr size_t kTrampolineSize =
    kPrefixSize + sizeof kJmpIndirect + sizeof(uint8_t*);
static_assert(kPrefixSize >= kJmpRel32Size);
}  // namespace

void* HookNativeSpeech(uint8_t* entry, void* hook) {
  if (!entry || std::memcmp(entry, kNativeSpeechPrefix, kPrefixSize))
    return nullptr;
  auto* relay = call_sites::Thunk(reinterpret_cast<uint64_t>(entry), hook);
  if (!relay) return nullptr;
  const int64_t delta = reinterpret_cast<int64_t>(relay) -
                        reinterpret_cast<int64_t>(entry + kJmpRel32Size);
  if (delta != static_cast<int32_t>(delta)) return nullptr;
  auto* original =
      static_cast<uint8_t*>(call_sites::trampoline.allocate(kTrampolineSize));
  if (!original) return nullptr;
  std::memcpy(original, kNativeSpeechPrefix, kPrefixSize);
  // jmp [rip] to the first untouched instruction, without clobbering RAX.
  std::memcpy(original + kPrefixSize, kJmpIndirect, sizeof kJmpIndirect);
  auto* continuation = entry + kPrefixSize;
  std::memcpy(original + kPrefixSize + sizeof kJmpIndirect, &continuation,
              sizeof continuation);
  FlushInstructionCache(GetCurrentProcess(), original, kTrampolineSize);
  uint8_t patch[kPrefixSize];
  const int32_t displacement = static_cast<int32_t>(delta);
  patch[0] = 0xE9;
  std::memcpy(patch + 1, &displacement, sizeof displacement);
  std::memset(patch + kJmpRel32Size, 0x90, kPrefixSize - kJmpRel32Size);
  REL::safe_write(reinterpret_cast<uint64_t>(entry), patch, sizeof patch);
  FlushInstructionCache(GetCurrentProcess(), entry, kPrefixSize);
  return original;
}

}  // namespace bfa
