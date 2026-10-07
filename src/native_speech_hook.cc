// Detours the engine's SpeakVoiceLine entry (see native_speech_hook.h):
// a rel32 jmp over its known prologue, which a trampoline replays.
#include "native_speech_hook.h"

#include <windows.h>

#include <cstdint>
#include <cstring>

#include "call_sites.h"

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
  auto* original = static_cast<uint8_t*>(VirtualAlloc(
      nullptr, kTrampolineSize, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
  if (!original) return nullptr;
  std::memcpy(original, kNativeSpeechPrefix, kPrefixSize);
  // jmp [rip] to the first untouched instruction, without clobbering RAX.
  std::memcpy(original + kPrefixSize, kJmpIndirect, sizeof kJmpIndirect);
  auto* continuation = entry + kPrefixSize;
  std::memcpy(original + kPrefixSize + sizeof kJmpIndirect, &continuation,
              sizeof continuation);
  DWORD old;
  if (!VirtualProtect(original, kTrampolineSize, PAGE_EXECUTE_READ, &old)) {
    VirtualFree(original, 0, MEM_RELEASE);
    return nullptr;
  }
  FlushInstructionCache(GetCurrentProcess(), original, kTrampolineSize);
  if (!VirtualProtect(entry, kPrefixSize, PAGE_EXECUTE_READWRITE, &old)) {
    VirtualFree(original, 0, MEM_RELEASE);
    return nullptr;
  }
  const int32_t displacement = static_cast<int32_t>(delta);
  entry[0] = 0xE9;
  std::memcpy(entry + 1, &displacement, sizeof displacement);
  std::memset(entry + kJmpRel32Size, 0x90, kPrefixSize - kJmpRel32Size);
  DWORD ignored;
  VirtualProtect(entry, kPrefixSize, old, &ignored);
  FlushInstructionCache(GetCurrentProcess(), entry, kPrefixSize);
  return original;
}

}  // namespace bfa
