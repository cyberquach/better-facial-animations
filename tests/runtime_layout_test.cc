// Independent actor layouts: never build fixtures using production offsets.
#include "../src/better_facial_animations.cc"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>

namespace {
void Check(bool ok, const char* expression, int line) {
  if (ok) return;
  std::fprintf(stderr, "FAIL %d: %s\n", line, expression);
  std::exit(1);
}
#define CHECK(x) Check((x), #x, __LINE__)

bool sound_playing;
void* expected_handle;
bool SoundPlaying(void* handle) {
  CHECK(handle == expected_handle);
  return sound_playing;
}

void CheckActor(uint32_t runtime, uint32_t process_offset,
                uint32_t race_offset) {
  skse::SetRuntime(runtime);
  CHECK(bfa::kActorProcess.Get() == process_offset);
  CHECK(bfa::kActorRace.Get() == race_offset);
  alignas(16) std::array<uint8_t, 0x300> actor{}, base{}, race{};
  alignas(16) std::array<uint8_t, 0x400> process{}, high{};
  actor[0x1A] = 0x3E;
  base[0x1A] = 0x2B;
  bfa::Field<void*>(actor.data(), 0x40) = base.data();
  bfa::Field<void*>(actor.data(), race_offset) = race.data();
  bfa::Field<uint32_t>(race.data(), 0x108) = 2;
  CHECK(bfa::SpeakerOk(actor.data()));
  for (uint32_t id : {0x13740u, 0x8883Au}) {
    bfa::Field<uint32_t>(race.data(), 0x14) = id;
    CHECK(!bfa::SpeakerOk(actor.data()));
  }
  bfa::Field<uint32_t>(race.data(), 0x14) = 0;
  bfa::Field<uint32_t>(race.data(), 0x108) = 0;
  CHECK(!bfa::SpeakerOk(actor.data()));

  uint32_t sound = 0;
  CHECK(bfa::ActorVoice(actor.data(), sound) == -1);
  CHECK(sound == a2f::kNoSound);
  bfa::Field<void*>(actor.data(), process_offset) = process.data();
  CHECK(bfa::ActorVoice(actor.data(), sound) == -1);
  bfa::Field<void*>(process.data(), 0x10) = high.data();
  expected_handle = high.data() + 0x2FC;
  bfa::Field<uint32_t>(expected_handle, 0) = 42;
  sound_playing = true;
  CHECK(bfa::ActorVoice(actor.data(), sound) == 1 && sound == 42);
  sound_playing = false;
  CHECK(bfa::ActorVoice(actor.data(), sound) == 0 && sound == 42);
}
}  // namespace

extern "C" uint64_t __wrap__ZN4skse4AddrEy(uint64_t id) {
  CHECK(id == kIdSoundIsPlaying.Get());
  return reinterpret_cast<uint64_t>(&SoundPlaying);
}

int main() {
  using skse::Version;
  // Keep this expectation independent of the production support lists.
  for (uint32_t runtime : {
           Version(1, 6, 353), Version(1, 6, 640),
           Version(1, 6, 659, skse::kGog), Version(1, 6, 1170),
           Version(1, 6, 1179, skse::kGog), Version(1, 7, 99),
           Version(1, 7, 104)}) {
    bool declared = false;
    for (uint32_t v : SKSEPlugin_Version.compatibleVersions) {
      if (!v) break;
      declared |= v == runtime;
    }
    CHECK(declared);
    CHECK(bfa::NativeSpeechRuntimeSupported(runtime));
    CheckActor(runtime, runtime == Version(1, 6, 353) ? 0xF0 : 0xF8,
               runtime == Version(1, 6, 353) ? 0x1F0 : 0x1F8);
    CHECK(!skse::IsSe() && !skse::IsVr());
    CHECK(kIdSpeakVoiceLine.Get() == 37542);
    CHECK(kIdSoundIsPlaying.Get() == 67620);
  }
  CheckActor(Version(1, 5, 97), 0xF0, 0x1F0);
  CheckActor(skse::kVrRuntime, 0xF0, 0x1F0);
  // Layout boundary, independently of whether BFA enables these runtimes.
  CheckActor(Version(1, 6, 628), 0xF0, 0x1F0);
  CheckActor(Version(1, 6, 629), 0xF8, 0x1F8);
  std::puts("ok: runtime declarations, AE ids, race and voice layouts");
}
