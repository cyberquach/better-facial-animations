#include "runtime_fixture.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

#include "../src/plugin/better_facial_animations.cc"
#define CHECK(x)                                      \
  do {                                                \
    if (!(x)) {                                       \
      fprintf(stderr, "FAIL %d: %s\n", __LINE__, #x); \
      std::exit(1);                                   \
    }                                                 \
  } while (0)
namespace {
using namespace bfa;
alignas(16) std::array<uint8_t, 0x500> npc{}, player{}, other{}, process{},
    fg{}, node{}, manager{};
uint8_t* mtm = manager.data();
uint32_t target_handle = 7;
float modifiers[17] = {}, consumed[17] = {};
int forced = -1;
bool other_writer = false, replace_keyframe = false, in_scene = false;
float replacement[17] = {}, baked[17] = {};
void* actor_vtable[0x100] = {};

bool Member(const uint8_t* actor, const uint8_t*) {
  CHECK(actor == npc.data());
  return in_scene;
}
uint32_t* Target(uint8_t* p, uint32_t* out) {
  CHECK(p == process.data());
  *out = target_handle;
  return out;
}
bool Lookup(const uint32_t& h, uint8_t** out) {
  *out = h == 42  ? npc.data()
         : h == 7 ? player.data()
         : h == 8 ? other.data()
                  : nullptr;
  if (*out) InterlockedIncrement(&Field<LONG>(*out, 0x28));
  return *out != nullptr;
}
uint64_t Morph(uint8_t* n, uint8_t force) {
  CHECK(n == node.data());
  forced = force;
  std::copy_n(modifiers, 17, consumed);
  auto& flags = Field<uint16_t>(node.data(), (skse::IsVr() ? 0x188 : 0x160));
  if (force || (flags & 1)) std::copy_n(consumed, 17, baked);
  flags &= ~1;
  fg[0x218] = 0;
  auto& mods = FixtureKeyframe(fg.data(), 0x100);
  if (other_writer) {
    modifiers[9] = 0.8f;
    mods.isUpdated = true;
  }
  if (replace_keyframe) mods.values = replacement;
  return 123;
}
void Toggle(const char* key, bool value) {
  for (const auto& p : kParams)
    if (!strcmp(p.key, key)) {
      *p.flag = value;
      return;
    }
  CHECK(false);
}
void Strength(float value) {
  for (const auto& p : kParams)
    if (!strcmp(p.key, "GazeStrength")) {
      *p.value = value;
      return;
    }
  CHECK(false);
}
// The line's 3 / 6 degree eyes, scaled by strength and clamped at 8, over
// the 30 degree look morphs: Down gains x, Left (0.4) loses y.
void Draw(bool expected, float strength = 1) {
  bool reset = false;
  forced = -1;
  CHECK(GazeMorphs(Morph, node.data(), 0, reset, world_epoch) == 123);
  CHECK(reset == expected && forced == int(expected));
  CHECK(modifiers[8] == 0 && modifiers[9] == 0.4f && modifiers[10] == 0 &&
        modifiers[11] == 0);
  CHECK(modifiers[0] == 0.7f && consumed[0] == 0.7f);
  if (expected) {
    const float x = std::min(3 * strength, 8.f) / 30;
    const float y = std::min(6 * strength, 8.f) / 30;
    CHECK(std::abs(consumed[8] - x) < 1e-5f);
    CHECK(std::abs(consumed[9] - (0.4f - y)) < 1e-5f);
    CHECK(Field<uint16_t>(node.data(), (skse::IsVr() ? 0x188 : 0x160)) & 1);
    CHECK(fg[0x218] == 1);
  } else {
    CHECK(consumed[9] == 0.4f && consumed[8] == 0);
    // The rebuild requested by the previous active call removes baked gaze.
    CHECK(baked[9] == 0.4f && baked[8] == 0);
  }
  for (int c = 0; c < 17; ++c) {
    if (c < 8 || c > 11) CHECK(consumed[c] == modifiers[c]);
  }
  CHECK(Field<uint32_t>(fg.data(), 0x224) == 0);
}
struct Net : a2f::Net {
  int frames = 0, windows = 0;
  int Batch() const override { return 2; }
  bool Run(const float*, const float*, int n, float* out,
           std::string&) override {
    std::fill_n(out, n * a2f::kNumOut, 10.f);
    for (int f = 0; f < n; ++f)
      for (int c = 0; c < 4; ++c)
        out[f * a2f::kNumOut + a2f::kNumOut - 4 + c] = 11.f + c + frames + f;
    frames += n;
    return true;
  }
  bool Run3(const float*, const float*, const float*, const float*, float* skin,
            float* tongue, float* jaw, float* state, std::string&) override {
    std::fill_n(skin, a2f::kFrames3 * a2f::kLatent3, 10.f);
    std::fill_n(tongue, a2f::kFrames3 * a2f::kLatent3, 10.f);
    std::fill_n(jaw, a2f::kFrames3 * a2f::kJawEyes3, 10.f);
    std::fill_n(state, a2f::kState3, 0.f);
    for (int f = 0; f < a2f::kFrames3; ++f)
      for (int c = 0; c < 4; ++c)
        jaw[f * a2f::kJawEyes3 + 15 + c] =
            11.f + c + windows * a2f::kFrames3 + f - a2f::kSkip3;
    ++windows;
    return true;
  }
};
void Generation() {
  Net net;
  std::string err;
  a2f::Line line;
  std::vector<float> pcm(16001, 0.1f), rest(a2f::kNumOut3, 10.f), proj(512),
      tongue(5120);
  CHECK(a2f::Generate(net, pcm.data(), pcm.size(), {}, proj.data(), rest.data(),
                      1, line, err));
  CHECK(line.frames == 31 && line.fps == 30 && line.direct_dim == 26);
  CHECK(line.eyes.size() == line.frames * 4);
  for (size_t i = 0; i < line.eyes.size(); ++i)
    CHECK(line.eyes[i] == 1.f + i % 4 + i / 4);
  for (float f : line.direct) CHECK(f == 0);
  CHECK(a2f::Generate3(net, pcm.data(), pcm.size(), {},
                       {0, proj.data(), tongue.data(), rest.data(), 1}, line,
                       err));
  CHECK(line.frames == 61 && line.fps == 60 && line.eyes.size() == 244);
  CHECK(line.eyes.size() == line.frames * 4);
  for (size_t i = 0; i < line.eyes.size(); ++i)
    CHECK(line.eyes[i] == 1.f + i % 4 + i / 4);
  for (float f : line.direct) CHECK(f == 0);
  CHECK(!a2f::Generate3(net, pcm.data(), 0, {},
                        {0, proj.data(), tongue.data(), rest.data(), 1}, line,
                        err));
  CHECK(line.eyes.empty());
}
}  // namespace
extern "C" uint64_t TestAddress(uint64_t id) {
  if (id == kIdMenuTopicManager.Get()) return reinterpret_cast<uint64_t>(&mtm);
  if (id == kIdHeadtrackTarget.Get())
    return reinterpret_cast<uint64_t>(&Target);
  if (id == kIdLookupReference.Get())
    return reinterpret_cast<uint64_t>(&Lookup);
  return 0;
}
namespace {
void AdmissionAndRestoration() {
  using namespace bfa;
  skse::SetRuntime(skse::Version(1, 6, 1170));
  CHECK(SnapshotTune().conversation_gaze && SnapshotTune().gaze_strength == 1);
  Toggle("ConversationGaze", true);
  ready = true;
  Field<uint32_t>(manager.data(), 0x68) = 42;
  Field<void*>(npc.data(), (skse::Runtime() >= skse::Version(1, 6, 629) ? 0xf8 : 0xf0)) = process.data();
  Field<uint32_t>(npc.data(), 0x14) = 0x123;
  npc[0x1A] = kFtActor;
  Field<void*>(npc.data(), 0) = actor_vtable;
  actor_vtable[(skse::IsVr() ? 0xFB : 0xF9)] = reinterpret_cast<void*>(&Member);
  scene_factions[0] = other.data();
  num_scene_factions = 1;
  Field<uint32_t>(player.data(), 0x14) = 0x14;
  Field<uint32_t>(other.data(), 0x14) = 0x456;
  for (auto* a : {npc.data(), player.data(), other.data()})
    Field<LONG>(a, 0x28) = 1;
  Field<void*>(node.data(), 0x150) = fg.data();
  auto& mods = FixtureKeyframe(fg.data(), 0x100);
  mods.values = modifiers;
  mods.count = 17;
  modifiers[0] = 0.7f;
  modifiers[9] = 0.4f;
  auto line = std::make_shared<a2f::Line>();
  line->frames = 60;
  line->eyes.resize(240);
  for (int i = 0; i < 240; ++i) line->eyes[i] = i % 2 ? 6 : 3;
  Playing p;
  p.actor = npc.data();
  p.actor_handle = 42;
  p.direct_frame = 30;
  p.ready_at = 0;
  p.entry = std::make_shared<Entry>();
  p.entry->state = State::kReady;
  p.entry->line = line;
  playing[fg.data()] = p;
  Draw(true);
  CHECK(!mods.isUpdated);
  Draw(true);  // same baseline, no accumulation
  std::array<uint8_t, 0x100> base{};
  base[0x1A] = kFtNpc;
  Field<uint32_t>(base.data(), 0x14) = 0x05001234;
  Field<void*>(npc.data(), 0x40) = base.data();
  const NpcSetting excluded{0x05001234, "follower.esp|001234", "Follower",
                            true};
  SetNpcBlocked(excluded, true);
  Draw(false);  // even before the speech/morph handoff, disabled override
                // yields gaze
  SetNpcBlocked(excluded, false);
  Draw(true);
  auto customized = excluded;
  customized.blocked = false;
  customized.values = {{"GazeStrength", .5f}, {"ConversationGaze", 1}};
  Toggle("ConversationGaze", false);
  SetNpcOverride(customized);
  Draw(true, .5f);  // per-NPC gaze and strength override the global controls
  customized.values["ConversationGaze"] = 0;
  SetNpcOverride(customized);
  Toggle("ConversationGaze", true);
  Draw(false);
  RemoveNpcOverride(customized);
  Draw(true);
  Strength(0);
  Draw(false);  // zero strength keeps the game's gaze
  Strength(2);
  Draw(true, 2);  // scaled, and clamped at the maximum angle
  Strength(1);
  Draw(true);
  other_writer = true;
  bool reset = false;
  GazeMorphs(Morph, node.data(), 0, reset, world_epoch);
  CHECK(modifiers[9] == 0.8f && modifiers[8] == 0 && modifiers[10] == 0);
  other_writer = false;
  modifiers[9] = 0.4f;
  CHECK(mods.isUpdated);
  mods.isUpdated = false;
  replace_keyframe = true;
  GazeMorphs(Morph, node.data(), 0, reset, world_epoch);
  CHECK(mods.values == replacement);
  for (float value : replacement) CHECK(value == 0);
  replace_keyframe = false;
  mods.values = modifiers;
  std::fill_n(modifiers + 8, 4, 0.f);
  modifiers[9] = 0.4f;
  mods.isUpdated = false;
  Draw(true);
  in_scene = true;
  Draw(false);
  in_scene = false;
  Draw(true);
  target_handle = 8;
  Draw(false);  // scripted look at someone else
  target_handle = 0;
  Draw(false);
  target_handle = 7;
  Draw(true);
  Field<void*>(npc.data(), (skse::Runtime() >= skse::Version(1, 6, 629) ? 0xf8 : 0xf0)) = nullptr;
  Draw(false);
  Field<void*>(npc.data(), (skse::Runtime() >= skse::Version(1, 6, 629) ? 0xf8 : 0xf0)) = process.data();
  Field<uint32_t>(manager.data(), 0x68) = 43;
  Draw(false);
  Field<uint32_t>(manager.data(), 0x68) = 0;
  Draw(false);  // dialogue closed
  Field<uint32_t>(manager.data(), 0x68) = 42;
  Draw(true);
  playing[fg.data()].actor = other.data();
  Draw(false);
  playing[fg.data()].actor = npc.data();
  playing[fg.data()].player = true;
  Draw(false);
  playing[fg.data()].player = false;
  for (State state : {State::kPending, State::kFailed}) {
    playing[fg.data()].entry->state = state;
    Draw(false);
  }
  playing[fg.data()].entry->state = State::kReady;
  playing[fg.data()].entry->line.reset();
  Draw(false);
  playing[fg.data()].entry->line = line;
  playing[fg.data()].external = std::make_shared<PlayerSpeech>();
  playing[fg.data()].external->stopped = true;
  Draw(false);
  playing[fg.data()].external.reset();
  mods.count = 11;
  Draw(false);
  mods.count = 17;
  mods.values = nullptr;
  Draw(false);
  mods.values = modifiers;
  Toggle("Enabled", false);
  Draw(false);
  Toggle("Enabled", true);
  Toggle("ConversationGaze", false);
  Draw(false);
  Toggle("ConversationGaze", true);
  playing[fg.data()].direct_frame = -1;
  Draw(false);
  playing[fg.data()].direct_frame = 0;
  Draw(false);  // audio held/start fade
  playing[fg.data()].direct_frame = 59;
  Draw(false);  // completed fade
  playing[fg.data()].direct_frame = 30;
  line->eyes[120] = std::numeric_limits<float>::quiet_NaN();
  Draw(false);
  line->eyes[120] = 3;
  // Both playback profiles use sync-adjusted frames for late-start fading.
  for (bool female : {false, true}) {
    playing[fg.data()].female = female;
    for (float sync : {-0.3f, 0.3f}) {
      for (const auto& param : kParams) {
        if (!strcmp(param.key, "TimeOffset") && param.profile == int(female))
          *param.value = sync;
      }
      playing[fg.data()].ready_at = 1;
      playing[fg.data()].direct_frame = (1 + sync) * line->fps;
      Draw(false);  // Must not jump on the first late frame.
    }
  }
  ResetTune();
  playing[fg.data()].ready_at = 0;
  playing[fg.data()].direct_frame = 30;
  // SE and early AE have a different actor-process offset than modern AE.
  for (uint32_t runtime : {skse::Version(1, 5, 97), skse::Version(1, 6, 353),
                           skse::Version(1, 6, 1170)}) {
    skse::SetRuntime(runtime);
    Field<void*>(npc.data(), (skse::Runtime() >= skse::Version(1, 6, 629) ? 0xf8 : 0xf0)) = process.data();
    Draw(true);
  }
  reset = false;
  GazeMorphs(Morph, node.data(), 0, reset, world_epoch + 1);
  CHECK(!reset && consumed[9] == 0.4f);
  Draw(true);
  loading_game = true;
  Draw(false);
  loading_game = false;
  skse::SetRuntime(skse::kVrRuntime);
  Draw(false);
  skse::SetRuntime(skse::Version(1, 6, 1170));
  ResetPlayback(false);
  Draw(false);
  for (auto* actor : {npc.data(), player.data(), other.data()})
    CHECK(Field<LONG>(actor, 0x28) == 1);
}

void SamplingAndMorphs() {
  auto line = std::make_shared<a2f::Line>();
  line->frames = 60;
  line->eyes.resize(240);
  for (int i = 0; i < 240; ++i) line->eyes[i] = i % 2 ? 6 : 3;
  float offsets[2];
  CHECK(a2f::GazeOffset(*line, 30, 0, 2, offsets));
  CHECK(std::abs(offsets[1] - 8.f / 30) < 1e-5f);
  a2f::Line varied = *line;
  varied.eyes[31 * 4] = varied.eyes[31 * 4 + 2] = 9;
  CHECK(a2f::GazeOffset(varied, 30.5f, 0, 1, offsets));
  CHECK(std::abs(offsets[0] - 0.2f) < 1e-5f);
  CHECK(!a2f::GazeOffset(varied, 30.5f, 30.5f, 1, offsets));
  CHECK(!a2f::GazeOffset(varied, 60, 0, 1, offsets));
  varied.eyes.pop_back();
  CHECK(!a2f::GazeOffset(varied, 30, 0, 1, offsets));
  float pair[4] = {0.2f, 0.9f, 0.1f, 0.4f};
  float zero[2] = {};
  CHECK(a2f::OffsetGazeMorphs(pair, zero));
  CHECK(pair[0] == 0.2f && pair[3] == 0.4f);
  float move[2] = {0.8f, -0.8f};
  CHECK(a2f::OffsetGazeMorphs(pair, move));
  for (float v : pair) CHECK(v >= 0 && v <= 1);
  // Missing/nonfinite model data and settings cannot touch valid controls.
  for (float invalid : {std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::infinity()}) {
    CHECK(!a2f::GazeOffset(*line, invalid, 0, 1, offsets));
    CHECK(!a2f::GazeOffset(*line, 30, invalid, 1, offsets));
    CHECK(!a2f::GazeOffset(*line, 30, 0, invalid, offsets));
    varied = *line;
    varied.fps = invalid;
    CHECK(!a2f::GazeOffset(varied, 30, 0, 1, offsets));
    varied = *line;
    varied.eyes[122] = invalid;
    CHECK(!a2f::GazeOffset(varied, 30, 0, 1, offsets));
    CHECK(offsets[0] == 0 && offsets[1] == 0);
    float controls[4] = {0, 0.4f, 0, 0};
    float bad_offset[2] = {0.1f, invalid};
    CHECK(!a2f::OffsetGazeMorphs(controls, bad_offset));
    CHECK(controls[0] == 0 && controls[1] == 0.4f);
  }
  CHECK(!a2f::GazeOffset(*line, -1, 0, 1, offsets));
  CHECK(!a2f::GazeOffset(*line, 30, 0, 0, offsets));
  varied = *line;
  varied.fps = 0;
  CHECK(!a2f::GazeOffset(varied, 30, 0, 1, offsets));
  // Opposite eye rotations cancel vergence; signs work in both directions.
  varied = *line;
  for (int f = 0; f < 60; ++f) {
    varied.eyes[f * 4] = -3;
    varied.eyes[f * 4 + 1] = -6;
    varied.eyes[f * 4 + 2] = 3;
    varied.eyes[f * 4 + 3] = 6;
  }
  CHECK(!a2f::GazeOffset(varied, 30, 0, 1, offsets));
  for (int f = 0; f < 60; ++f) {
    varied.eyes[f * 4 + 2] = -3;
    varied.eyes[f * 4 + 3] = -6;
  }
  CHECK(a2f::GazeOffset(varied, 30, 0, 1, offsets));
  CHECK(std::abs(offsets[0] + 0.1f) < 1e-5f &&
        std::abs(offsets[1] + 0.2f) < 1e-5f);
  for (float invalid : {-0.1f, 1.1f, std::numeric_limits<float>::infinity()}) {
    float controls[4] = {0, 0.4f, 0, invalid};
    CHECK(!a2f::OffsetGazeMorphs(controls, move));
    CHECK(controls[0] == 0 && controls[1] == 0.4f);
  }
}
}  // namespace

int main() {
  bfa::testing::address = TestAddress;
  Generation();
  AdmissionAndRestoration();
  SamplingAndMorphs();
  puts(
      "ok: model eye tracks, additive gaze, target/line/runtime gating, "
      "restoration and no accumulation");
}
