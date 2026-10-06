// Production discovery, admission and handoff; only engine calls are mocked.
#include "../src/better_facial_animations.cc"
#include <array>
#include <cstdlib>

namespace {
void Check(bool ok, const char* expression, int line) {
  if (ok) return;
  fprintf(stderr, "failed: %s:%d: %s\n", __FILE__, line, expression);
  std::exit(1);
}
#define CHECK(expression) Check((expression), #expression, __LINE__)

alignas(16) std::array<uint8_t, 0x1000> data{}, actor_data{};
uint8_t* handler = data.data();
uint8_t* actor = actor_data.data();
const uint8_t* membership = nullptr;
int calls = 0;
bool Member(const uint8_t* who, const uint8_t* faction) {
  CHECK(who == actor);
  ++calls;
  return faction == membership;
}
bool Lookup(const uint32_t& h, uint8_t** out) {
  *out = h == 42 ? actor : nullptr;
  if (*out) ++bfa::Field<LONG>(*out, bfa::kRefrRefCount);
  return *out != nullptr;
}
struct Faction {
  std::array<uint8_t, 0x40> form{};
  std::array<uint8_t, 0x180> file{};
  uint8_t* files[1] = {file.data()};
  struct { uint8_t** items; uint32_t count; } source{files, 1};
  Faction(const char* name, uint32_t id) {
    using namespace bfa;
    form[kFormType] = kFtFaction;
    Field<uint32_t>(form.data(), kFormId) = id;
    Field<void*>(form.data(), kFormFiles) = &source;
    strcpy(reinterpret_cast<char*>(file.data() + kFileName), name);
  }
};
}
extern "C" uint64_t __wrap__ZN4skse4AddrEy(uint64_t id) {
  using namespace bfa;
  if (id == kIdDataHandler.Get()) return reinterpret_cast<uint64_t>(&handler);
  if (id == kIdLookupReference.Get())
    return reinterpret_cast<uint64_t>(&Lookup);
  return 0; // Unexpected engine calls must not silently succeed.
}
int main() {
  using namespace bfa;
  for (uint32_t runtime : {skse::Version(1, 5, 97), skse::Version(1, 6, 1170),
                           skse::Version(1, 4, 15)}) {
    skse::SetRuntime(runtime);
    // Preserve the exact phoneme contribution removed from the engine,
    // including partial takeover; malformed tracks must clear old weights.
    std::array<uint8_t, 0x1000> phon_face{};
    float original[direct::kNumPhonemes], authored[direct::kNumPhonemes];
    for (int i = 0; i < direct::kNumPhonemes; ++i) original[i] = .03f * i;
    auto& phon = Field<skse::KeyframeMultiple>(phon_face.data(), skse::kFgPhon3);
    phon.values = original;
    phon.count = direct::kNumPhonemes;
    PlayFrame(phon_face.data(), .4f, authored);
    for (int i = 0; i < direct::kNumPhonemes; ++i) {
      CHECK(std::fabs(authored[i] - .03f * i * .4f) < 1e-6f);
      CHECK(std::fabs(original[i] + authored[i] - .03f * i) < 1e-6f);
    }
    phon.count = 0;
    PlayFrame(phon_face.data(), 1, authored);
    for (float weight : authored) CHECK(weight == 0);

    // The extra tongue weights participate in normal line replacement and
    // release fades, rather than holding the last speech shape forever.
    a2f::PoseTransition tongue_transition;
    auto tongue_line = std::make_shared<a2f::Line>();
    std::vector<float> tongue_pose(1 + direct::kNumMouthCoef + direct::kNumPhonemes, 0);
    tongue_pose.back() = 1;
    CHECK(tongue_transition.Blend(tongue_line, 0, 0, true, true, tongue_pose));
    tongue_pose.assign(tongue_pose.size(), 0);
    CHECK(tongue_transition.Blend(nullptr, .1, .2, true, false, tongue_pose));
    CHECK(tongue_pose.back() == 1);
    tongue_pose.assign(tongue_pose.size(), 0);
    CHECK(!tongue_transition.Blend(nullptr, .4, .2, true, false, tongue_pose));
    CHECK(tongue_pose.back() == 0);

    void* vtable[0x100]{};
    vtable[kVfIsInFaction.Get()] = reinterpret_cast<void*>(&Member);
    Field<void**>(actor, 0) = vtable;
    actor[kFormType] = kFtActor;
    Field<LONG>(actor, kRefrRefCount) = 1;
    Faction sl("SexLab.esm", 0x0500E50F), os("oStIm.EsP", 0xFE123ECA),
        excitement("OStim.esp", 0x08000D93), alias("OStim.esp", 0x08001ECA),
        wrong("Unrelated.esp", 0x09000ECA);
    uint8_t* forms[] = {alias.form.data(), wrong.form.data(), sl.form.data(),
                        os.form.data(), excitement.form.data()};
    BSTArray& factions =
        Field<BSTArray>(handler, kDhFormArrays + kFtFaction * sizeof(BSTArray));
    factions = {forms, 5, 0, 5};
    LoadSceneFactions();
    CHECK(num_scene_factions == 3);
    CHECK(scene_factions[0] == sl.form.data());
    CHECK(scene_factions[1] == os.form.data());
    CHECK(scene_factions[2] == excitement.form.data());
    LoadSceneFactions(); // idempotent; not a second set of stale pointers
    CHECK(num_scene_factions == 3 && scene_factions[0] == sl.form.data());
    membership = nullptr;
    CHECK(!InScene(actor));
    CHECK(!InScene(nullptr));
    calls = 0;
    actor[kFormType] = kFtNpc;
    CHECK(!InScene(actor) && calls == 0);
    actor[kFormType] = kFtActor;
    for (auto* faction :
         {sl.form.data(), os.form.data(), excitement.form.data()}) {
      membership = faction;
      CHECK(InScene(actor));
      // Common admission gate must stop NPC and external/player starts
      // before attempting face/resource engine calls or queueing inference.
      for (uint32_t id : {0x14u, 0x123u}) {
        Field<uint32_t>(actor, kFormId) = id;
        BeginSpeech(actor, "scene", 0, 2, 0, 0, Tune{}, world_epoch);
        CHECK(playing.empty() && queue.empty());
      }
      alignas(16) std::array<uint8_t, 0x300> storage{}, node{};
      auto* fg = storage.data();
      float values[direct::kNumExpressions];
      auto& ex = Field<skse::KeyframeMultiple>(fg, skse::kFgExpr3);
      ex.values = values; ex.count = direct::kNumExpressions;
      auto seed = [&] {
        std::fill_n(values, direct::kNumExpressions, 0.65f);
        Face f; f.actor = actor; f.actor_handle = 42;
        f.head_state = State::kReady;
        f.ownership_blend = f.speech_blend = 1;
        f.until = play_clock + 20;
        auto line = std::make_shared<a2f::Line>();
        // Include retained authored tongue cues in the scene handoff.
        std::vector<float> pose(1 + direct::kNumMouthCoef +
                                    direct::kNumPhonemes, 1);
        f.transition.Blend(line, Now(), 0, true, true, pose);
        faces[fg] = f;
        Playing p; p.actor = actor; p.actor_handle = 42;
        p.entry = std::make_shared<Entry>(); p.direct_frames = 10;
        std::fill_n(p.tongue_phonemes, direct::kNumPhonemes, 0.8f);
        playing[fg] = p;
        Count();
      };
      seed();
      CHECK(Apply(fg, Tune{}, true, actor, 42, a2f::kNoSound, -1) &
            kFaceChanged);
      CHECK(playing.empty());
      // Scene expressions just computed by the original update survive.
      CHECK(Expressions(fg, 0, Tune{}, true, actor, 42) & kFaceChanged);
      CHECK(faces.empty());
      for (float v : values) CHECK(v == 0.65f);
      CHECK(Expressions(fg, 0.016f, Tune{}, true, actor, 42) == kNoUpdate);
      seed();
      Field<uint8_t*>(node.data(), kNodeFg.Get()) = fg;
      // No animation update before this morph call: no head mapping or
      // vertex engine function is available, so proceeding would crash.
      AddDirect(node.data(), false);
      CHECK(faces.empty() && playing.empty() && fg[kFgDirty]);
      for (float v : values) CHECK(v == 0.65f);
      CHECK(Field<LONG>(actor, kRefrRefCount) == 1);
      seed();
      // If no scene writer has replaced our keyframe, restore its saved
      // engine values before forgetting ownership, including a reset call.
      std::fill_n(values, direct::kNumExpressions, 0.f);
      std::fill_n(faces.at(fg).src, direct::kNumExpressions, 0.45f);
      AddDirect(node.data(), true);
      CHECK(faces.empty() && playing.empty());
      for (int i = 0; i < direct::kNumExpressions; ++i)
        CHECK(values[i] == (direct::PlaysExpression(i) ? 0.45f : 0.f));
      CHECK(Field<skse::SpinLock>(fg, skse::kFgLock).count == 0);
      // Leaving the scene permits conversation expression ownership again.
      membership = nullptr;
      CHECK(!InScene(actor));
      partner_fg = fg; partner_ref = actor; partner_handle = 42;
      Expressions(fg, 0.1f, Tune{}, false, actor, 42);
      CHECK(faces.count(fg) && faces.at(fg).ownership_blend > 0);
      ResetPlayback(false);
    }
    factions = {};
    LoadSceneFactions();
    CHECK(num_scene_factions == 0 && !InScene(actor));
    handler = nullptr; LoadSceneFactions(); CHECK(num_scene_factions == 0);
    handler = data.data();
  }
  puts("scene contracts passed: SE/AE/VR dispatch, discovery, absence, "
       "identity, admission, immediate handoff, direct-only handoff, exit");
}
