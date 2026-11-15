#include "runtime_fixture.h"
// Exercise loaded records and live handoff; only engine calls are mocked.
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "../src/plugin/better_facial_animations.cc"

namespace {
#define CHECK(x)                                           \
  do {                                                     \
    if (!(x)) {                                            \
      std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); \
      std::exit(1);                                        \
    }                                                      \
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
    Field<const char*>(form.data(), 0x28) = FixtureString("Test race");
    Field<const char*>(form.data(), 0x2E0) = FixtureString(editor);
    Field<uint32_t>(form.data(), 0x108) = 2;
    std::strcpy(reinterpret_cast<char*>(file.data() + 0x58), plugin);
  }
};

bool Lookup(const uint32_t& handle, uint8_t** out) {
  *out = handle == 42 ? actor : nullptr;
  if (*out) ++bfa::Field<LONG>(*out, 0x28);
  return *out != nullptr;
}
}  // namespace

// The engine's ApplyMorphs behind the morph hook: nothing to rebuild here.
uint64_t EngineMorphs(uint8_t*, uint8_t) { return 0; }
// The morph hook on the node, as the engine calls it (the plugin is ready).
void MorphPass(uint8_t* n, bool reset) {
  const bool was = bfa::ready.exchange(true);
  bfa::ApplyMorphsHook(n, reset);
  bfa::ready = was;
}
extern "C" uint64_t TestAddress(uint64_t id) {
  if (id == kIdDataHandler.Get()) return reinterpret_cast<uint64_t>(&handler);
  if (id == kIdLookupReference.Get())
    return reinterpret_cast<uint64_t>(&Lookup);
  if (id == kIdApplyMorphs.Get())
    return reinterpret_cast<uint64_t>(&EngineMorphs);
  return 0;
}

int main() {
  bfa::testing::address = TestAddress;
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
  Field<uint32_t>(draugr.form.data(), 0x108) = 0;
  Field<uint32_t>(dragon.form.data(), 0x108) = 1;
  // Other race flags must not hide a custom FaceGen race.
  Field<uint32_t>(custom.form.data(), 0x108) = 3;
  uint8_t* records[] = {custom.form.data(),   nullptr,
                        nord.form.data(),     light.form.data(),
                        argonian.form.data(), vampire.form.data(),
                        draugr.form.data(),   dragon.form.data(),
                        alfiq.form.data()};
  Field<FixtureArray>(handler, 0x10 + 0x0E * 0x18) = {records, 9, 0, 9};
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

  for (uint32_t runtime :
       {skse::Version(1, 5, 97), skse::Version(1, 6, 353),
        skse::Version(1, 6, 1170), skse::Version(1, 4, 15)}) {
    skse::SetRuntime(runtime);
    actor[0x1A] = 0x3E;
    base[0x1A] = 0x2B;
    Field<void*>(actor, 0x40) = base.data();
    Field<void*>(
        actor, (skse::Runtime() >= skse::Version(1, 6, 629) ? 0x1f8 : 0x1f0)) =
        argonian.form.data();
    CHECK(SpeakerOk(actor));
    Field<void*>(
        actor, (skse::Runtime() >= skse::Version(1, 6, 629) ? 0x1f8 : 0x1f0)) =
        alfiq.form.data();
    CHECK(!SpeakerOk(actor));
    BeginSpeech(actor, "disabled", 0, 2, 0, 0, Tune(), world_epoch);
    CHECK(queue.empty() && playing.empty());
    Field<LONG>(actor, 0x28) = 1;
    std::array<uint8_t, 0x400> storage{}, node{};
    auto* fg = storage.data();
    float values[direct::kNumExpressions];
    auto& ex = FixtureKeyframe(fg, 0xC0);
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
    Field<uint8_t*>(node.data(), (skse::IsVr() ? 0x178 : 0x150)) = fg;
    MorphPass(node.data(), false);
    CHECK(faces.empty() && playing.empty() && fg[0x218]);
    CHECK(Field<LONG>(actor, 0x28) == 1);
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
      CHECK(f.src[idx] == 0 && f.shown[idx] > 0.30f && f.shown[idx] < 0.33f);
      frame(0);
      CHECK(f.shown[idx] == 0.f);
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
    // Work budget: a one-frame override must settle within the advertised
    // full-scale fade, rather than forcing seconds of idle mesh rebuilds.
    for (int hz : {30, 60, 120}) {
      Face& f = faces.at(fg);
      f.head_state = State::kReady;
      f.ownership_blend = 1;
      f.speech_blend = 0;
      f.transition = a2f::PoseTransition();
      f.shown_set = true;
      std::fill_n(f.shown, direct::kNumExpressions, 0.f);
      std::fill_n(f.level, direct::kNumExpressions, 0.f);
      int updates = 0;
      auto frame = [&](float target, float dt) {
        std::fill_n(values, direct::kNumExpressions, target);
        return Expressions(fg, dt, Tune(), false, actor, 42);
      };
      for (int i = 0; i < hz * 5; ++i)
        updates += (frame(0.9f, 1.f / hz) & kKeepUpdating) != 0;
      CHECK(updates > 1);  // preserve smoothing for abrupt overrides
      CHECK(updates <= static_cast<int>(std::ceil(0.3f * hz)));
      CHECK(frame(0.9f, 1.f / hz) == kNoUpdate);
      // Small native changes already have an engine-provided ramp. Following
      // them must not introduce a second fade or leave idle work afterward.
      float target = 0;
      for (int i = 1; i <= hz; ++i) {
        target = 0.9f - 0.6f * i / hz;
        frame(target, 1.f / hz);
        for (int j = 0; j < direct::kNumExpressions; ++j)
          if (direct::PlaysExpression(j)) CHECK(f.shown[j] == target);
      }
      CHECK(frame(target, 1.f / hz) == kNoUpdate);
    }
    // After its last line a face is kept for kFaceGrace, then handed back
    // over kOwnOut and dropped: no held expression lingers, on the player or
    // an NPC. A line of the same speaker within the grace, even one still
    // awaiting its animation, keeps the face without a dip. The partner's
    // face is released the same way once the conversation is over.
    {
      partner_fg = nullptr;
      partner_ref = nullptr;
      partner_handle = 0;
      faces.clear();
      Count();
      auto frame = [&](float dt) {
        testing::clock_offset = testing::clock_offset.load() + dt;
        std::fill_n(values, direct::kNumExpressions, 0.65f);
        return Expressions(fg, dt, Tune(), false, actor, 42);
      };
      auto speak = [&](bool player, State state) {
        Playing line;
        line.actor = actor;
        line.actor_handle = 42;
        line.player = player;
        line.entry = std::make_shared<Entry>();
        line.entry->state = state;
        line.direct_frame = state == State::kReady ? 0 : -1;
        line.direct_frames = 10;
        playing[fg] = line;
        Count();
      };
      for (bool player : {false, true}) {
        speak(player, State::kReady);
        for (int i = 0; i < 20; ++i) frame(0.1f);
        Face& f = faces.at(fg);
        CHECK(f.ownership_blend == 1 && f.speech_blend == 1);
        playing.erase(fg);  // the line is over, nothing follows
        Count();
        const double over = play_clock;
        // the grace: still the plugin's while the engine's expressions
        // fade back in, and kept updating towards the hand-back
        while (play_clock < over + kFaceGrace - 0.15) {
          CHECK(frame(0.1f) & kKeepUpdating);
          CHECK(f.ownership_blend == 1);
        }
        CHECK(f.speech_blend == 0);
        while (faces.count(fg)) {
          CHECK(play_clock < over + kFaceGrace + kOwnOut + 0.3);
          frame(0.1f);
        }
        CHECK(playing.empty());
        for (float value : values) CHECK(value == 0.65f);  // the engine's
      }
      // The next line within the grace: no dip while it is generated, then
      // its animation fades in from the kept face.
      speak(false, State::kReady);
      for (int i = 0; i < 20; ++i) frame(0.1f);
      Face& f = faces.at(fg);
      playing.erase(fg);
      Count();
      for (int i = 0; i < 10; ++i) frame(0.1f);  // 1 s gap
      speak(false, State::kPending);
      Playing& next = playing.at(fg);
      for (int i = 0; i < 15; ++i) {  // 1.5 s of inference
        frame(0.1f);
        CHECK(f.ownership_blend == 1 && &faces.at(fg) == &f);
      }
      CHECK(f.speech_blend == 0);  // the engine's expressions, as between lines
      next.entry->state = State::kReady;
      next.direct_frame = 0;
      for (int i = 0; i < 20; ++i) {
        frame(0.1f);
        CHECK(f.ownership_blend == 1);
      }
      CHECK(f.speech_blend == 1);
      // but inference longer than the grace hands the face back meanwhile
      next.entry->state = State::kPending;
      next.direct_frame = -1;
      for (int i = 0; i < 50; ++i) frame(0.1f);
      CHECK(faces.count(fg) && faces.at(fg).ownership_blend == 0);
      playing.erase(fg);
      faces.clear();
      Count();
      // A first line waits for its animation with the engine's expressions.
      speak(false, State::kPending);
      for (int i = 0; i < 10; ++i) frame(0.1f);
      CHECK(faces.count(fg) && faces.at(fg).ownership_blend == 0);
      playing.erase(fg);
      faces.clear();
      Count();
      // The partner: the plugin's throughout the conversation, handed back
      // within the grace and the fade after it.
      partner_fg = fg;
      partner_ref = actor;
      partner_handle = 42;
      for (int i = 0; i < 20; ++i) frame(0.1f);
      CHECK(faces.at(fg).ownership_blend == 1);
      partner_fg = nullptr;  // the conversation is over
      partner_ref = nullptr;
      partner_handle = 0;
      const double over = play_clock;
      while (play_clock < over + kFaceGrace - 0.15) {
        frame(0.1f);
        CHECK(faces.at(fg).ownership_blend == 1);
      }
      while (faces.count(fg)) {
        CHECK(play_clock < over + kFaceGrace + kOwnOut + 0.3);
        frame(0.1f);
      }
      testing::clock_offset = 0;
      partner_fg = fg;
      partner_ref = actor;
      partner_handle = 42;
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
    MorphPass(node.data(), false);
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
  SelectFaceOption(a2f::kNetPerformance);
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
  // Installer race choices compose with the main INI and remain overridable.
  CHECK(
      CreateDirectoryA("Data\\SKSE\\Plugins\\BetterFacialAnimations", nullptr));
  CHECK(CreateDirectoryA(
      "Data\\SKSE\\Plugins\\BetterFacialAnimations\\Installer", nullptr));
  const char* fragment =
      "Data\\SKSE\\Plugins\\BetterFacialAnimations\\Installer\\Argonian.ini";
  FILE* file = fopen(fragment, "wb");
  CHECK(file);
  CHECK(fputs("[Races]\nSkyrim.esm|013740=0\nSkyrim.esm|08883A=0\n", file) >=
        0);
  CHECK(fclose(file) == 0);
  LoadIni();
  LoadRaceSettings();
  CHECK(!RaceEnabled(0x13740) && !RaceEnabled(0x8883A));
  SetRaceEnabled(0x13740, true);
  SaveIni();
  LoadRaceSettings();
  CHECK(RaceEnabled(0x13740) && !RaceEnabled(0x8883A));
  ResetTune();
  SaveIni();
  LoadIni();
  LoadRaceSettings();
  CHECK(RaceEnabled(0x13740) && RaceEnabled(0x8883A));
  CHECK(!RaceEnabled(0x06000800));
  CHECK(DeleteFileA(fragment));
  CHECK(RemoveDirectoryA(
      "Data\\SKSE\\Plugins\\BetterFacialAnimations\\Installer"));
  CHECK(RemoveDirectoryA("Data\\SKSE\\Plugins\\BetterFacialAnimations"));
  CHECK(DeleteFileA(kIni));
  CHECK(RemoveDirectoryA("Data\\SKSE\\Plugins"));
  CHECK(RemoveDirectoryA("Data\\SKSE"));
  CHECK(RemoveDirectoryA("Data"));
  CHECK(SetCurrentDirectoryA(old) && RemoveDirectoryA(root));
  std::puts(
      "race settings: discovery, stable persistence, admission, live "
      "handoff, re-enable and defaults passed");
}
