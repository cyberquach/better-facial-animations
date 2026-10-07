// Execute the real entry detour and trampoline with the speech call ABI.
#include "../src/native_speech_hook.h"

#include <windows.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
namespace {
void Check(bool ok, const char* expression, int line) {
  if (ok) return;
  std::fprintf(stderr, "FAIL %d: %s\n", line, expression);
  std::exit(1);
}
#define CHECK(x) Check((x), #x, __LINE__)

using Fn = float (*)(uint8_t*, const char*, void*, uint64_t, uint64_t,
                     uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
                     uint64_t, uint64_t, uint64_t, uint64_t);
Fn original;
unsigned calls, target_calls;
uint8_t actor_storage;
const char voice_storage[] = "DBVO\\Callan\\test.fuz";
int handle_storage;
float Target(uint8_t* actor, const char* voice, void* handle, uint64_t d,
             uint64_t e, uint64_t f, uint64_t g, uint64_t h, uint64_t i,
             uint64_t j, uint64_t k, uint64_t l, uint64_t m, uint64_t n) {
  ++target_calls;
  CHECK(actor == &actor_storage && voice == voice_storage &&
        handle == &handle_storage);
  CHECK(d == 4 && e == 5 && f == 6 && g == 7 && h == 8 && i == 9 &&
        j == 10 && k == 11 && l == 12 && m == 13 && n == 14);
  return 2.5f;
}
float Hook(uint8_t* actor, const char* voice, void* handle, uint64_t d,
             uint64_t e, uint64_t f, uint64_t g, uint64_t h, uint64_t i,
             uint64_t j, uint64_t k, uint64_t l, uint64_t m, uint64_t n) {
  ++calls;
  return original(actor, voice, handle, d, e, f, g, h, i, j, k, l, m, n) + 100.0f;
}
float Invoke(Fn fn) {
  return fn(&actor_storage, voice_storage, &handle_storage,
            4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14);
}
}  // namespace

int main() {
  CHECK(bfa::NativeSpeechRuntimeSupported(skse::Version(1, 5, 97)));
  CHECK(bfa::NativeSpeechRuntimeSupported(skse::Version(1, 6, 640)));
  CHECK(bfa::NativeSpeechRuntimeSupported(skse::Version(1, 6, 1170)));
  CHECK(bfa::NativeSpeechRuntimeSupported(
      skse::Version(1, 6, 1179, skse::kGog)));
  CHECK(bfa::NativeSpeechRuntimeSupported(skse::Version(1, 7, 99)));
  CHECK(bfa::NativeSpeechRuntimeSupported(skse::Version(1, 7, 104)));
  for (uint32_t runtime : {0u, skse::Version(1, 5, 80),
                          skse::Version(1, 6, 659), skse::Version(1, 6, 1179),
                          skse::Version(1, 6, 1179, skse::kEpic),
                          skse::Version(1, 7, 105),
                          skse::kVrRuntime})
    CHECK(!bfa::NativeSpeechRuntimeSupported(runtime));
  auto* code = static_cast<uint8_t*>(VirtualAlloc(nullptr, 4096,
      MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE));
  CHECK(code);
  // Every altered prologue byte must refuse installation without mutation.
  for (size_t i = 0; i < sizeof(bfa::kNativeSpeechPrefix); ++i) {
    std::memcpy(code, bfa::kNativeSpeechPrefix, 7);
    code[i] ^= 1;
    uint8_t before[7];
    std::memcpy(before, code, 7);
    CHECK(!bfa::HookNativeSpeech(code, reinterpret_cast<void*>(&Hook)));
    CHECK(!std::memcmp(before, code, 7));
  }
  std::memcpy(code, bfa::kNativeSpeechPrefix, 7);
  // Jump without disturbing argument registers/stack to the ABI assertion target.
  const uint8_t jump[] = {0xff, 0x25, 0, 0, 0, 0};
  std::memcpy(code + 7, jump, sizeof(jump));
  auto target = &Target;
  std::memcpy(code + 13, &target, sizeof(target));
  FlushInstructionCache(GetCurrentProcess(), code, 21);
  auto fn = reinterpret_cast<Fn>(code);
  CHECK(Invoke(fn) == 2.5f && target_calls == 1);
  original = reinterpret_cast<Fn>(
      bfa::HookNativeSpeech(code, reinterpret_cast<void*>(&Hook)));
  CHECK(original);
  CHECK(Invoke(fn) == 102.5f && calls == 1 && target_calls == 2);
  CHECK(Invoke(original) == 2.5f && calls == 1 && target_calls == 3);
  uint8_t patched[7];
  std::memcpy(patched, code, 7);
  CHECK(!bfa::HookNativeSpeech(code, reinterpret_cast<void*>(&Hook)));
  CHECK(!std::memcmp(patched, code, 7));
  CHECK(Invoke(fn) == 102.5f && calls == 2 && target_calls == 4);
  CHECK(!bfa::HookNativeSpeech(nullptr, reinterpret_cast<void*>(&Hook)));
  std::puts("ok: native speech 14-argument ABI, float return, trampoline "
            "and refusal controls");
}
