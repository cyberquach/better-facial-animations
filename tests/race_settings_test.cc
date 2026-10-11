// Exercise loaded records and live handoff; only engine calls are mocked.
#include "../src/plugin/better_facial_animations.cc"

#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace {
#define CHECK(x)                                                      \
  do {                                                                \
    if (!(x)) {                                                       \
      std::fprintf(stderr, "line %d: %s\n", __LINE__, #x);             \
      std::exit(1);                                                   \
    }                                                                 \
  } while (0)

std::array<uint8_t, 0x1000> data{};
std::array<uint8_t, 0x400> actor_data{}, base{};
uint8_t* handler = data.data();
uint8_t* actor = actor_data.data();

struct Race {
  std::array<uint8_t, 0x500> form{};
  std::array<uint8_t, 0x180> file{};
  uint8_t* files[1] = {file.data()};
  struct {
    uint8_t** items;
    uint32_t count;
  } source{files, 1};
  Race(const char* plugin, uint32_t id, const char* editor) {
    using bfa::Field;
    form[0x1A] = 0x0E;
    Field<uint32_t>(form.data(), 0x14) = id;
    Field<void*>(form.data(), 0x08) = &source;
    Field<const char*>(form.data(), 0x28) = "Test race";
    Field<const char*>(form.data(), 0x2E0) = editor;
    Field<uint32_t>(form.data(), 0x108) = 2;
    std::strcpy(reinterpret_cast<char*>(file.data() + 0x58), plugin);
  }
};

bool Lookup(const uint32_t& handle, uint8_t** out) {
  *out = handle == 42 ? actor : nullptr;
  if (*out) ++bfa::Field<LONG>(*out, bfa::kRefrRefCount);
  return *out != nullptr;
}
}  // namespace

extern "C" uint64_t __wrap__ZN4skse4AddrEy(uint64_t id) {
  if (id == kIdDataHandler.Get()) return reinterpret_cast<uint64_t>(&handler);
  if (id == kIdLookupReference.Get())
    return reinterpret_cast<uint64_t>(&Lookup);
  return 0;
}

int main() {
  using namespace bfa;
  // All INI writes stay inside this contract's temporary working directory.
  char old[MAX_PATH], temp[MAX_PATH], root[MAX_PATH];
  CHECK(GetCurrentDirectoryA(MAX_PATH, old));
  CHECK(GetTempPathA(MAX_PATH, temp));
  CHECK(GetTempFileNameA(temp, "bfr", 0, root));
  CHECK(DeleteFileA(root) && CreateDirectoryA(root, nullptr));
  CHECK(SetCurrentDirectoryA(root));
  CreateDirectoryA("Data", nullptr);
  CreateDirectoryA("Data\\SKSE", nullptr);
  CreateDirectoryA("Data\\SKSE\\Plugins", nullptr);
  constexpr char kIni[] = "Data\\SKSE\\Plugins\\BetterFacialAnimations.ini";
  DeleteFileA(kIni);
  // INIs from before Argonians defaulted on drop their saved off state once.
  WritePrivateProfileStringA("General", "IniVersion", "5", kIni);
  WritePrivateProfileStringA("Races", "Skyrim.esm|013740", "0", kIni);
  WritePrivateProfileStringA("Races", "Skyrim.esm|08883A", "0", kIni);
  LoadIni();
  Race nord("Skyrim.esm", 0x13746, "NordRace");
  Race argonian("Skyrim.esm", 0x13740, "ArgonianRace");
  Race vampire("Skyrim.esm", 0x8883A, "ArgonianRaceVampire");
  Race custom("Custom.esp", 0x05001234, "CustomRace");
  Race alfiq("DWAlfiqKhajiit.esp", 0x06000800, "0_KhajiitAlfiqRace");
  Race light("Light.esl", 0xFE123ABC, "LightRace");
  Race draugr("Skyrim.esm", 0xD53, "DraugrRace");
  Race dragon("Skyrim.esm", 0x12E82, "DragonRace");
  Field<uint32_t>(draugr.form.data(), kRaceFlags) = 0;
  Field<uint32_t>(dragon.form.data(), kRaceFlags) = 1;
  // Other race flags must not hide a custom FaceGen race.
  Field<uint32_t>(custom.form.data(), kRaceFlags) = 3;
  uint8_t* records[] = {custom.form.data(), nullptr, nord.form.data(),
                        light.form.data(), argonian.form.data(),
                        vampire.form.data(), draugr.form.data(),
                        dragon.form.data(), alfiq.form.data()};
  Field<BSTArray>(handler, 0x10 + 0x0E * 0x18) = {records, 9, 0, 9};
  LoadRaceSettings();
  CHECK(SnapshotRaces().size() == 6);
  for (const auto& race : SnapshotRaces())
    CHECK(race.id != 0xD53 && race.id != 0x12E82);
  CHECK(RaceEnabled(0x13746) && RaceEnabled(0x05001234));
  CHECK(RaceEnabled(0x13740) && RaceEnabled(0x8883A));
  SetRaceEnabled(0x8883A, false);
  SetRaceEnabled(0x05001234, false);
  SetRaceEnabled(0xFE123ABC, false);
  SaveIni();
  CHECK(GetPrivateProfileIntA("Races", "Custom.esp|001234", 1, kIni) == 0);
  CHECK(GetPrivateProfileIntA("Races", "Light.esl|000ABC", 1, kIni) == 0);
  // Reordered full and light plugins retain independent saved choices.
  Field<uint32_t>(custom.form.data(), 0x14) = 0x09001234;
  Field<uint32_t>(light.form.data(), 0x14) = 0xFE456ABC;
  LoadRaceSettings();
  CHECK(!RaceEnabled(0x09001234) && !RaceEnabled(0xFE456ABC));
  CHECK(RaceEnabled(0x13740) && !RaceEnabled(0x8883A));
  CHECK(RaceEnabled(0x13746));
  CHECK(SnapshotRaces().size() == 6);

  for (uint32_t runtime : {skse::Version(1, 5, 97),
                           skse::Version(1, 6, 353),
                           skse::Version(1, 6, 1170),
                           skse::Version(1, 4, 15)}) {
    skse::SetRuntime(runtime);
    actor[0x1A] = 0x3E;
    base[0x1A] = 0x2B;
    Field<void*>(actor, 0x40) = base.data();
    Field<void*>(actor, kActorRace.Get()) = argonian.form.data();
    CHECK(SpeakerOk(actor));
    Field<void*>(actor, kActorRace.Get()) = alfiq.form.data();
    CHECK(!SpeakerOk(actor));
    BeginSpeech(actor, "disabled", 0, 2, 0, 0, Tune(), world_epoch);
    CHECK(queue.empty() && playing.empty());
    Field<LONG>(actor, kRefrRefCount) = 1;
    std::array<uint8_t, 0x400> storage{}, node{};
    auto* fg = storage.data();
    float values[direct::kNumExpressions];
    auto& ex = Field<skse::KeyframeMultiple>(fg, skse::kFgExpr3);
    ex.values = values;
    ex.count = direct::kNumExpressions;
    auto seed = [&] {
      std::fill_n(values, direct::kNumExpressions, 0.65f);
      Face face;
      face.actor = actor;
      face.actor_handle = 42;
      face.head_state = State::kReady;
      face.ownership_blend = face.speech_blend = 1;
      face.until = play_clock + 20;
      faces[fg] = face;
      Playing line;
      line.actor = actor;
      line.actor_handle = 42;
      line.entry = std::make_shared<Entry>();
      line.direct_frames = 10;
      playing[fg] = line;
      Count();
    };
    seed();
    CHECK(Apply(fg, Tune(), false, actor, 42, a2f::kNoSound, -1) &
          kFaceChanged);
    CHECK(playing.empty());
    CHECK(Expressions(fg, 0, Tune(), false, actor, 42) & kFaceChanged);
    CHECK(faces.empty());
    for (float value : values) CHECK(value == 0.65f);
    seed();
    Field<uint8_t*>(node.data(), kNodeFg.Get()) = fg;
    AddDirect(node.data(), false);
    CHECK(faces.empty() && playing.empty() && fg[kFgDirty]);
    CHECK(Field<LONG>(actor, kRefrRefCount) == 1);
    SetRaceEnabled(0x06000800, true);
    CHECK(SpeakerOk(actor));
    partner_fg = fg;
    partner_ref = actor;
    partner_handle = 42;
    Expressions(fg, 0.1f, Tune(), false, actor, 42);
    CHECK(faces.count(fg) && faces.at(fg).ownership_blend > 0);
    // The engine's expressions rewritten in one frame (a script's override,
    // as CHIM's) reach the face smoothly (Face::shown, kSrcFollow), not as
    // a snap. The engine's value is put back each frame (RestoreKeyframe).
    {
      Face& f = faces.at(fg);
      int idx = 0;
      while (!direct::PlaysExpression(idx)) ++idx;
      auto frame = [&](float engine) {
        std::fill_n(values, direct::kNumExpressions, engine);
        Expressions(fg, 0.1f, Tune(), false, actor, 42);
      };
      for (int i = 0; i < 12; ++i) frame(0.65f);
      CHECK(f.ownership_blend == 1 && f.speech_blend == 0);
      CHECK(std::fabs(f.shown[idx] - 0.65f) < 1e-4f);
      frame(0);  // override: neutral at once
      CHECK(f.src[idx] == 0 && f.shown[idx] > 0.4f && f.shown[idx] < 0.47f);
      frame(0);
      CHECK(f.shown[idx] > 0.25f && f.shown[idx] < 0.32f);
      for (int i = 0; i < 30; ++i) frame(0);
      CHECK(f.shown[idx] < 0.01f);
      frame(0.9f);  // override: a full expression at once
      CHECK(f.shown[idx] > 0.25f && f.shown[idx] < 0.35f);
      std::fill_n(values, direct::kNumExpressions, 0.65f);
    }
    // A quiet engine must still be asked to tick every frame of a handoff.
    // Cover both mesh expressions and keyframe-only fallback; the last step
    // must rebuild too, but a settled face must stop requesting updates.
    {
      Face& f = faces.at(fg);
      const Tune tune;
      auto frame = [&] {
        std::fill_n(values, direct::kNumExpressions, 0.65f);
        return Expressions(fg, 0.1f, tune, false, actor, 42);
      };
      for (State state : {State::kReady, State::kFailed}) {
        f.head_state = state;
        f.ownership_blend = 1;
        f.speech_blend = 1;
        for (int i = 0; i < direct::kNumExpressions; ++i)
          f.shown[i] = direct::PlaysExpression(i) ? 0.65f : 0.f;
        for (int i = 0; i < 20 && f.speech_blend > 0; ++i) {
          const int updates = frame();
          CHECK(updates & kFaceChanged);
          CHECK(updates & kKeepUpdating);
        }
        CHECK(f.speech_blend == 0);
        CHECK(!(frame() & kKeepUpdating));
        std::fill_n(f.shown, direct::kNumExpressions, 0.f);
        CHECK(frame() & kKeepUpdating);  // a new engine expression
        for (int i = 0; i < 60; ++i) frame();
        CHECK(!(frame() & kKeepUpdating));
      }
      f.head_state = State::kReady;
      auto line = std::make_shared<a2f::Line>();
      std::vector<float> pose(1, 1.f);
      f.transition.Blend(line, Now(), 1, true, true, pose);
      pose[0] = 0;
      f.transition.Blend(nullptr, Now(), 1, true, false, pose);
      CHECK(frame() & kKeepUpdating);  // speech pose still easing to zero
      f.transition = a2f::PoseTransition();
      CHECK(!(frame() & kKeepUpdating));
    }
    // Disabling a race must release every live animation path.
    SetRaceEnabled(0x06000800, false);
    CHECK(!SpeakerOk(actor));
    seed();
    CHECK(Apply(fg, Tune(), false, actor, 42, a2f::kNoSound, -1) &
          kFaceChanged);
    CHECK(playing.empty());
    CHECK(Expressions(fg, 0, Tune(), false, actor, 42) & kFaceChanged);
    CHECK(faces.empty());
    for (float value : values) CHECK(value == 0.65f);
    seed();
    AddDirect(node.data(), false);
    CHECK(faces.empty() && playing.empty());
    BeginSpeech(actor, "disabled", 0, 2, 0, 0, Tune(), world_epoch);
    CHECK(queue.empty() && playing.empty());
    SetRaceEnabled(0x06000800, true);
    CHECK(SpeakerOk(actor));
    // Enabling a creature record never bypasses the FaceGen requirement.
    Field<uint32_t>(alfiq.form.data(), 0x108) = 0;
    CHECK(!SpeakerOk(actor));
    Field<uint32_t>(alfiq.form.data(), 0x108) = 2;
    SetRaceEnabled(0x06000800, false);
    ResetPlayback(false);
  }
  // Race defaults and persistence must preserve the candidate's model choice.
  {
    ScopedLock guard(settings_lock);
    for (const Param& param : kParams) {
      if (std::strcmp(param.key, "UsePerformanceModel") == 0)
        SelectModelCheckbox(param);
    }
  }
  CHECK(SnapshotTune().SelectedModel() == 4);
  ResetTune();
  CHECK(SnapshotTune().SelectedModel() == 4);
  CHECK(RaceEnabled(0x09001234) && RaceEnabled(0x13740));
  CHECK(!RaceEnabled(0x06000800));
  SaveIni();
  LoadIni();
  LoadRaceSettings();
  CHECK(SnapshotTune().SelectedModel() == 4);
  CHECK(RaceEnabled(0x09001234) && RaceEnabled(0x13740));
  CHECK(!RaceEnabled(0x06000800));
  DeleteFileA(kIni);
  CHECK(RemoveDirectoryA("Data\\SKSE\\Plugins"));
  CHECK(RemoveDirectoryA("Data\\SKSE"));
  CHECK(RemoveDirectoryA("Data"));
  CHECK(SetCurrentDirectoryA(old) && RemoveDirectoryA(root));
  std::puts("race settings: discovery, stable persistence, admission, live "
            "handoff, re-enable and defaults passed");
}
