// Production admission and conversation tracking; only engine lookup is mocked.
#include "../src/plugin/better_facial_animations.cc"

#include <array>
#include <cstdlib>
#include <cstring>

namespace {
#define CHECK(x)                                                    \
  do {                                                              \
    if (!(x)) {                                                     \
      std::fprintf(stderr, "line %d: %s\n", __LINE__, #x);           \
      std::exit(1);                                                 \
    }                                                               \
  } while (0)

std::array<uint8_t, 0x1000> actor_bytes{}, base{}, race{}, file{},
    override_file{}, menu{}, node{}, fg{}, data{};
uint8_t* actor = actor_bytes.data();
uint8_t* manager = menu.data();
uint8_t* handler = data.data();

void SetOwner(const char* name) {
  std::strcpy(reinterpret_cast<char*>(file.data() + bfa::kFileName), name);
}

bool Lookup(const uint32_t& handle, uint8_t** out) {
  *out = handle == 42 ? actor : nullptr;
  if (*out) ++bfa::Field<LONG>(*out, bfa::kRefrRefCount);
  return *out != nullptr;
}
}  // namespace

extern "C" uint64_t __wrap__ZN4skse4AddrEy(uint64_t id) {
  if (id == kIdDataHandler.Get())
    return reinterpret_cast<uint64_t>(&handler);
  if (id == kIdMenuTopicManager.Get())
    return reinterpret_cast<uint64_t>(&manager);
  if (id == kIdLookupReference.Get())
    return reinterpret_cast<uint64_t>(&Lookup);
  return 0;
}

int main() {
  using namespace bfa;
  // Keep persisted race choices isolated from the user's settings.
  char old[MAX_PATH], temp[MAX_PATH], root[MAX_PATH];
  CHECK(GetCurrentDirectoryA(MAX_PATH, old));
  CHECK(GetTempPathA(MAX_PATH, temp));
  CHECK(GetTempFileNameA(temp, "bfa", 0, root));
  CHECK(DeleteFileA(root) && CreateDirectoryA(root, nullptr));
  CHECK(SetCurrentDirectoryA(root));
  CHECK(CreateDirectoryA("Data", nullptr));
  CHECK(CreateDirectoryA("Data\\SKSE", nullptr));
  CHECK(CreateDirectoryA("Data\\SKSE\\Plugins", nullptr));
  constexpr char kIni[] = "Data\\SKSE\\Plugins\\BetterFacialAnimations.ini";
  race[kFormType] = kFtRace;
  uint8_t* records[] = {race.data()};
  Field<BSTArray>(handler, 0x10 + kFtRace * 0x18) = {records, 1, 0, 1};
  for (uint32_t runtime : {skse::Version(1, 5, 97), skse::Version(1, 6, 353),
                           skse::Version(1, 6, 640), skse::Version(1, 6, 1170),
                           skse::Version(1, 4, 15)}) {
    skse::SetRuntime(runtime);
    actor[kFormType] = kFtActor;
    base[kFormType] = kFtNpc;
    Field<uint8_t*>(actor, kRefrBase) = base.data();
    Field<uint8_t*>(actor, kActorRace.Get()) = race.data();
    Field<uint32_t>(race.data(), kRaceFlags) = 2;
    Field<LONG>(actor, kRefrRefCount) = 1;
    Field<uint32_t>(manager, kMtmSpeaker) = 42;
    Field<uint8_t*>(node.data(), kAvUserData.Get()) = actor;
    uint8_t* files[] = {file.data(), override_file.data()};
    struct {
      uint8_t** items;
      uint32_t count;
    } source{files, 2};
    Field<void*>(race.data(), kFormFiles) = &source;
    SetOwner("dWaLfIqKhAjIiT.EsP");
    std::strcpy(reinterpret_cast<char*>(override_file.data() + kFileName),
                "Patch.esp");
    // All three shipped races, both plugin encodings, and a future local ID.
    for (uint32_t id : {0xFE123800u, 0xFE123845u, 0xFE12384Eu, 0x05000800u,
                        0x05000845u, 0x0500084Eu, 0xFE123900u}) {
      Field<uint32_t>(race.data(), kFormId) = id;
      LoadRaceSettings();
      CHECK(SnapshotRaces().size() == 1);
      CHECK(!SnapshotRaces()[0].enabled);
      CHECK(!RaceEnabled(id) && !SpeakerOk(actor));
      SetRaceEnabled(id, true);
      CHECK(SpeakerOk(actor));
      Field<uint32_t>(race.data(), kRaceFlags) = 0;
      CHECK(!SpeakerOk(actor));
      Field<uint32_t>(race.data(), kRaceFlags) = 2;
      TrackPartner(node.data(), fg.data());
      CHECK(partner_ref == actor);
      SaveIni();
      // The same saved choice survives full/light plugin load-order changes.
      const uint32_t moved =
          id ^ ((id >> 24) == 0xFE ? 0x00333000u : 0x0C000000u);
      Field<uint32_t>(race.data(), kFormId) = moved;
      LoadRaceSettings();
      CHECK(RaceEnabled(moved) && SpeakerOk(actor));
      ResetTune();
      CHECK(!RaceEnabled(moved) && !SpeakerOk(actor));
      SaveIni();
      LoadRaceSettings();
      CHECK(!RaceEnabled(moved));
      CHECK(DeleteFileA(kIni));
      Field<uint32_t>(race.data(), kFormId) = id;
      LoadRaceSettings();
      ready = true;
      uint32_t sound = 123;
      constexpr char kVoice[] = "Sound\\Voice\\Skyrim.esm\\MaleKhajiit\\test.fuz";
      CHECK(ObserveSpeech(actor, kVoice, &sound, 0, 0, 4.25f, world_epoch) ==
            4.25f);
      TrackPartner(node.data(), fg.data());
      CHECK(!partner_ref && !partner_fg && !partner_handle);
      CHECK(Expressions(fg.data(), 0.016f, Tune{}, false, actor, 42) ==
            kNoUpdate);
      CHECK(faces.empty() && playing.empty() && queue.empty());
    }
    // Same numeric ID in an unrelated plugin remains eligible, even when
    // Alfiq is the last override rather than the owner.
    SetOwner("Other.esp");
    std::strcpy(reinterpret_cast<char*>(override_file.data() + kFileName),
                "DWAlfiqKhajiit.esp");
    LoadRaceSettings();
    CHECK(SpeakerOk(actor));
    TrackPartner(node.data(), fg.data());
    CHECK(partner_ref == actor && partner_fg == fg.data() &&
          partner_handle == 42);
    SetOwner("DWAlfiqKhajiit.esp");
    LoadRaceSettings();
    TrackPartner(node.data(), fg.data());
    CHECK(!partner_ref && !partner_fg && !partner_handle);
    SetOwner("Skyrim.esm");
    // Khajiit, Argonian, their vampire races and Nord are positive controls.
    for (uint32_t id : {0x13745u, 0x88845u, 0x13740u, 0x8883Au, 0x13746u}) {
      Field<uint32_t>(race.data(), kFormId) = id;
      LoadRaceSettings();
      CHECK(SpeakerOk(actor));
    }
    // Eligibility is independent of cached geometry, names and load order.
    CHECK(!LimitLowerLip(nullptr));
    Field<uint8_t*>(actor, kActorRace.Get()) = nullptr;
    CHECK(!LimitLowerLip(actor));
    Field<uint8_t*>(actor, kActorRace.Get()) = race.data();
    for (uint32_t id : {0x13745u, 0x88845u, 0x13740u, 0x8883Au}) {
      Field<uint32_t>(race.data(), kFormId) = id;
      CHECK(LimitLowerLip(actor));
      Field<uint32_t>(race.data(), kFormId) = id | 0x05000000;
      CHECK(!LimitLowerLip(actor));
    }
    for (uint32_t id : {0u, 0x13746u, 0x13747u, 0x06000800u, 0xFE123ABCu}) {
      Field<uint32_t>(race.data(), kFormId) = id;
      CHECK(!LimitLowerLip(actor));
    }
    Field<uint32_t>(race.data(), kFormId) = 0x13746;
    SetOwner("DWAlfiqKhajiit.esp");
    source.count = 0;
    LoadRaceSettings();
    CHECK(SpeakerOk(actor));
    source.count = 2;
    files[0] = nullptr;
    LoadRaceSettings();
    CHECK(SpeakerOk(actor));
    source.items = nullptr;
    LoadRaceSettings();
    CHECK(SpeakerOk(actor));
    Field<void*>(race.data(), kFormFiles) = nullptr;
    LoadRaceSettings();
    CHECK(SpeakerOk(actor));
    Field<uint32_t>(race.data(), kRaceFlags) = 0;
    CHECK(!SpeakerOk(actor));
    CHECK(!SpeakerOk(nullptr));
    CHECK(Field<LONG>(actor, kRefrRefCount) == 1);
  }
  CHECK(RemoveDirectoryA("Data\\SKSE\\Plugins"));
  CHECK(RemoveDirectoryA("Data\\SKSE"));
  CHECK(RemoveDirectoryA("Data"));
  CHECK(SetCurrentDirectoryA(old) && RemoveDirectoryA(root));
  std::puts("Alfiq defaults: owner identity, opt-in, persistence, reset, "
            "speech, conversation, "
            "SE/AE/VR and ordinary Khajiit controls passed");
}
