// Exercise the real message handler and hooks without starting Skyrim.
// Poison pointers deliberately make any access to the departing world fail.
#include "../src/better_facial_animations.cc"
#include <cstdlib>

namespace {
void Check(bool ok, const char* expression, int line) {
  if (ok) return;
  fprintf(stderr, "failed: %s:%d: %s\n", __FILE__, line, expression);
  std::exit(1);
}
#define CHECK(expression) Check((expression), #expression, __LINE__)

float* retired_keyframe = nullptr;
bool inject_reset = false;
}  // namespace

// Deterministically schedule a load while RestoreExpressions waits for the
// face lock. The retired keyframe must never be touched on cancellation.
extern "C" void __real__ZN4skse8SpinLock4LockEv(skse::SpinLock*);
extern "C" void __wrap__ZN4skse8SpinLock4LockEv(skse::SpinLock* self) {
  if (inject_reset) {
    inject_reset = false;
    bfa::ResetPlayback(true);
    DWORD old;
    CHECK(VirtualProtect(retired_keyframe, 4096, PAGE_NOACCESS, &old));
  }
  __real__ZN4skse8SpinLock4LockEv(self);
}

namespace {
int original_updates = 0, original_morphs = 0;
bool OriginalUpdate(uint8_t*, float, uint64_t) {
  ++original_updates;
  return true;
}
uint64_t OriginalMorphs(uint8_t*, uint8_t) {
  ++original_morphs;
  return 123;
}
// Native callbacks returning across a world reset must not inspect the actor.
float NativeReset(uint8_t*, const char*, void*, uint64_t, uint64_t,
                  uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
                  uint64_t, uint64_t, uint64_t, uint64_t) {
  bfa::ResetPlayback(true);
  return 2.5f;
}
float NativeReturn(uint8_t*, const char*, void*, uint64_t, uint64_t,
                   uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
                   uint64_t, uint64_t, uint64_t, uint64_t) { return 3.5f; }
// A Win64 caller with shadow space and a real call-site return address.
template <typename Fn>
Fn Hooked(void* original, void* hook) {
  auto* stub = call_sites::Thunk(skse::ImageBase() ? skse::ImageBase() :
      reinterpret_cast<uint64_t>(GetModuleHandleA(nullptr)), original);
  CHECK(stub);
  auto* caller = call_sites::Thunk(reinterpret_cast<uint64_t>(stub), original);
  CHECK(caller);
  const uint8_t code[] = {0x48,0x83,0xEC,0x28, 0xE8,0,0,0,0,
                         0x48,0x83,0xC4,0x28,0xC3};
  memcpy(caller, code, sizeof code);
  const auto at = reinterpret_cast<uint64_t>(caller + 4);
  CHECK(call_sites::PatchCall(at, stub));
  CHECK(call_sites::Hook(at, hook, "test", reinterpret_cast<uint64_t>(stub)));
  return reinterpret_cast<Fn>(caller);
}
void Message(unsigned type, void* data = nullptr) {
  skse::SKSEMessage m{"SKSE", type, 0, data};
  bfa::OnMessage(&m);
}
void Seed() {
  using namespace bfa;
  auto* poison = reinterpret_cast<uint8_t*>(1);
  Playing p;
  p.actor = poison;  // no handle: the old implementation trusted this
  p.entry = std::make_shared<Entry>();
  playing[poison] = p;
  Face f;
  f.actor = poison;
  f.src[0] = 0.75f;
  faces[poison] = f;
  partner_fg = partner_ref = poison;
  partner_female = true;
  Count();
}
void Empty() {
  using namespace bfa;
  CHECK(playing.empty() && faces.empty());
  CHECK(num_playing == 0 && num_faces == 0);
  CHECK(!partner_fg && !partner_ref && !partner_female);
}
}
int main() {
  using namespace bfa;
  ready = true;
  net_loaded = true;
  auto entry = std::make_shared<Entry>();
  cache["owned-audio"] = entry;
  queue.push_back({"owned-audio", "copied/path", {}, false});
  const auto old_epoch = world_epoch.load();
  Seed();
  Message(skse::kMessagePreLoadGame);
  Empty();
  CHECK(loading_game && !CurrentWorld(old_epoch));
  CHECK(ready && net_loaded && cache.at("owned-audio") == entry && queue.size() == 1);
  auto* poison = reinterpret_cast<uint8_t*>(1);
  auto update = Hooked<AnimUpdateFn>(reinterpret_cast<void*>(&OriginalUpdate),
                                    reinterpret_cast<void*>(&AnimUpdateHook));
  auto morph = Hooked<ApplyMorphsFn>(reinterpret_cast<void*>(&OriginalMorphs),
                                    reinterpret_cast<void*>(&ApplyMorphsHook));
  CHECK(update(poison, 0.016f, 0));
  CHECK(morph(poison, 0) == 123);
  CHECK(original_updates == 1 && original_morphs == 1);
  auto speech = std::make_shared<PlayerSpeech>();
  StartPlayerSpeech(speech);  // must not resolve a player while loading
  Empty();
  Message(skse::kMessagePostLoadGame, reinterpret_cast<void*>(1));
  Empty();
  CHECK(!loading_game && !CurrentWorld(old_epoch));
  // An old speech hook returning after a load must not use its old actor.
  BeginSpeech(poison, "old", 0, 1, 0, 0, Tune{}, old_epoch);
  Empty();
  native_speech_original = NativeReset;
  CHECK(NativeSpeakHook(poison, nullptr, nullptr, 0,0,0,0,0,0,0,0,0,0,0) == 2.5f);
  CHECK(speech_depth == 0);
  Empty();
  Message(skse::kMessagePostLoadGame);
  native_speech_original = NativeReturn;
  {
    SpeechCallScope outer;  // call-site hook already owns observation
    CHECK(NativeSpeakHook(poison, nullptr, nullptr, 0,0,0,0,0,0,0,0,0,0,0) == 3.5f);
    CHECK(speech_depth == 1);  // no actor access or duplicate observation
  }
  CHECK(speech_depth == 0);
  // Nothing of the old world is left for a face update to find.
  CHECK(Apply(poison, Tune{}, false, nullptr, 0, a2f::kNoSound, -1) ==
        kNoUpdate);
  CHECK(Expressions(poison, 0.016f, Tune{}, false, nullptr, 0) == kNoUpdate);
  CHECK(!FaceActorOf(poison).handle);
  RestoreExpressions(poison, old_epoch);
  Empty();
  // Idle BFA does not take any FaceGen lock (the pointer is inaccessible).
  CHECK(update(poison, 0.016f, 0));
  CHECK(original_updates == 2);
  Seed();
  Message(skse::kMessagePreLoadGame);
  Message(skse::kMessagePostLoadGame);  // failed load also resumes
  Empty();
  CHECK(!loading_game);
  Seed();
  Message(skse::kMessageNewGame);
  Empty();
  CHECK(!loading_game && CurrentWorld(world_epoch));
  // Repeated loads remain safe and keep copied caches/network alive.
  for (int i = 0; i < 100; ++i) {
    Seed();
    Message(skse::kMessagePreLoadGame);
    Message(skse::kMessagePostLoadGame);
    Empty();
  }
  CHECK(cache.at("owned-audio") == entry && queue.size() == 1 && net_loaded);
  // A missing handle cannot authorize a deferred raw actor dereference,
  // even within the current world generation.
  Seed();
  CHECK(Apply(poison, Tune{}, false, nullptr, 0, a2f::kNoSound, -1) ==
        kNoUpdate);
  CHECK(playing.empty());
  // A line begun after the update looked its actor up is kept for the next.
  Seed();
  playing[poison].actor_handle = 42;
  CHECK(Apply(poison, Tune{}, false, nullptr, 0, a2f::kNoSound, -1) ==
        kKeepUpdating);
  CHECK(playing.size() == 1);
  ResetPlayback(false);
  Empty();
  auto* fg = static_cast<uint8_t*>(VirtualAlloc(
      nullptr, 4096, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
  retired_keyframe = static_cast<float*>(VirtualAlloc(
      nullptr, 4096, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
  CHECK(fg && retired_keyframe);
  auto& ex = Field<skse::KeyframeMultiple>(fg, skse::kFgExpr3);
  ex.values = retired_keyframe;
  ex.count = direct::kNumExpressions;
  Face parked;
  parked.parked = true;
  faces[fg] = parked;
  Count();
  inject_reset = true;
  RestoreExpressions(fg, world_epoch);
  CHECK(!inject_reset);
  Empty();
  CHECK(Field<skse::SpinLock>(fg, skse::kFgLock).count == 0);
  CHECK(VirtualFree(retired_keyframe, 0, MEM_RELEASE));
  CHECK(VirtualFree(fg, 0, MEM_RELEASE));
  puts("load lifecycle passed: stale pointers, hook forwarding, idle lock avoidance, failed/new/repeated loads, old speech rejection, owned cache retention, overlapping reset");
}
