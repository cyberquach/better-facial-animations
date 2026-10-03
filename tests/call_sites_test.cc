// Call-site hooks shared with another plugin (src/call_sites.h).
#include "../src/call_sites.h"

#include <cstdio>
#include <cstring>
#include <initializer_list>

namespace {
int failures = 0;
void Expect(bool ok, const char* label) {
  printf("%s %s\n", ok ? "ok" : "FAIL", label);
  if (!ok) ++failures;
}

using Fn = int (*)(int);
int callee_calls, hook_calls;

__attribute__((noinline)) int Callee(int x) {
  ++callee_calls;
  return x + 1;
}
// Like the plugin's hooks, and tail-jumping to its original at -O2. A chain
// that loops would never return: give up after a few rounds.
__attribute__((noinline)) int Hook(int x) {
  auto orig = call_sites::Orig<Fn>(__builtin_return_address(0),
                                   reinterpret_cast<void*>(&Callee));
  if (++hook_calls > 8) return -1;
  return orig(x);
}

// Code written at run time, in rel32 reach of the test's own functions.
uint8_t* code;
size_t code_used;
uint8_t* Code(size_t size) {
  uint8_t* p = code + code_used;
  code_used += (size + 15) & ~size_t{15};
  return p;
}
// `call Callee; ret`: a call site of the game.
uint8_t* MakeSite() {
  uint8_t* s = Code(6);
  s[0] = 0xE8;
  s[5] = 0xC3;
  call_sites::PatchCall(reinterpret_cast<uint64_t>(s),
                        reinterpret_cast<void*>(&Callee));
  return s;
}
// Another plugin's hook: counts, then tail-jumps to the original it saved.
struct Other {
  uint8_t* code;
  int& calls() { return *reinterpret_cast<int*>(code + 16); }
  void*& orig() { return *reinterpret_cast<void**>(code + 24); }
};
Other MakeOther() {
  static const uint8_t kCode[] = {0xFF, 0x05, 10, 0, 0, 0,   // inc [rip+10]
                                  0xFF, 0x25, 12, 0, 0, 0};  // jmp [rip+12]
  Other o{Code(32)};
  std::memcpy(o.code, kCode, sizeof(kCode));
  o.calls() = 0;
  o.orig() = nullptr;
  return o;
}
// The other plugin patches a call site the way SKSE's write_call does.
void OtherHooks(Other& o, uint8_t* site) {
  uint64_t at = reinterpret_cast<uint64_t>(site);
  o.orig() = reinterpret_cast<void*>(call_sites::CallTarget(at));
  call_sites::PatchCall(at, o.code);
}
int Run(uint8_t* site, int x) {
  callee_calls = hook_calls = 0;
  return reinterpret_cast<Fn>(site)(x);
}
}  // namespace

int main() {
  const uint64_t near_addr =
      reinterpret_cast<uint64_t>(GetModuleHandleA(nullptr));
  for (uint64_t a = (near_addr - 0x200000) & ~0xFFFFull;
       a > near_addr - 0x70000000 && !code; a -= 0x10000)
    code = static_cast<uint8_t*>(
        VirtualAlloc(reinterpret_cast<void*>(a), 0x1000,
                     MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE));
  if (!code) {
    printf("FAIL no code page in reach\n");
    return 1;
  }
  uint8_t* shared = MakeSite();  // re-hooked by the other plugin
  uint8_t* ours = MakeSite();    // only ours
  uint8_t* taken = MakeSite();   // replaced by the other plugin's jump
  for (uint8_t* s : {shared, ours, taken})
    Expect(call_sites::Hook(reinterpret_cast<uint64_t>(s),
                            reinterpret_cast<void*>(&Hook), "test", near_addr),
           "site hooked");
  call_sites::Site* sites = call_sites::sites;
  Expect(sites[0].thunk != sites[1].thunk, "every site has its own thunk");
  Expect(Run(shared, 4) == 5 && hook_calls == 1 && callee_calls == 1,
         "hooked site runs the hook, then the original");

  Other other = MakeOther();
  OtherHooks(other, shared);
  Expect(Run(shared, 4) == 5 && other.calls() == 1 && hook_calls == 1 &&
             callee_calls == 1,
         "before the retake the other plugin runs first");

  // The freeze: both hooks tail-jump, and after the retake each had the other
  // as its original.
  Expect(call_sites::Retake(sites[0], near_addr) ==
             call_sites::Retaken::kChained,
         "re-hooked site is chained");
  Expect(call_sites::Retake(sites[1], near_addr) == call_sites::Retaken::kKept,
         "untouched site is kept");
  other.calls() = 0;
  Expect(Run(shared, 4) == 5, "chained site returns the original's result");
  Expect(hook_calls == 1, "chained site runs the hook once");
  Expect(other.calls() == 1, "chained site runs the other plugin once");
  Expect(callee_calls == 1, "chained site runs the original once");
  Expect(Run(ours, 7) == 8 && hook_calls == 1 && callee_calls == 1,
         "untouched site still runs the hook");
  Expect(call_sites::Retake(sites[0], near_addr) == call_sites::Retaken::kKept,
         "a second retake changes nothing");

  // The other plugin replaces the call by a jump to its hook, which still
  // has our thunk as its original.
  Other jumper = MakeOther();
  OtherHooks(jumper, taken);
  taken[0] = 0xE9;
  Expect(call_sites::Retake(sites[2], near_addr) == call_sites::Retaken::kLost,
         "replaced site is lost");
  Expect(Run(taken, 1) == 2 && jumper.calls() == 1 && hook_calls == 0 &&
             callee_calls == 1,
         "lost site runs the other plugin and the original only");

  printf(failures ? "%d FAILED\n" : "all passed\n", failures);
  return failures != 0;
}
