// Real settings persistence and animation handoff; engine lookups are mocked.
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "../src/plugin/better_facial_animations.cc"

#define CHECK(x)                                           \
  do {                                                     \
    if (!(x)) {                                            \
      std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); \
      std::exit(1);                                        \
    }                                                      \
  } while (0)
namespace {
std::array<uint8_t, 0x1000> data{};
std::array<uint8_t, 0x400> actor_data{}, race{};
uint8_t* handler = data.data();
uint8_t* actor = actor_data.data();
struct Npc {
  std::array<uint8_t, 0x400> form{}, file{};
  uint8_t* files[1] = {file.data()};
  struct {
    uint8_t** items;
    uint32_t count;
  } source{files, 1};
  Npc(const char* plugin, uint32_t id) {
    form[bfa::kFormType] = bfa::kFtNpc;
    bfa::Field<uint32_t>(form.data(), bfa::kFormId) = id;
    bfa::Field<void*>(form.data(), bfa::kFormFiles) = &source;
    bfa::Field<const char*>(form.data(), bfa::kNpcName) = "Val Serano";
    std::strcpy(reinterpret_cast<char*>(file.data() + bfa::kFileName), plugin);
  }
};
bool Lookup(const uint32_t& handle, uint8_t** out) {
  *out = handle == 42 ? actor : nullptr;
  if (*out) ++bfa::Field<LONG>(*out, bfa::kRefrRefCount);
  return *out != nullptr;
}
uint8_t* LookupForm(uint32_t id) { return id == 0x100 ? actor : nullptr; }
}  // namespace
extern "C" uint64_t __wrap__ZN4skse4AddrEy(uint64_t id) {
  if (id == kIdDataHandler.Get()) return reinterpret_cast<uint64_t>(&handler);
  if (id == kIdLookupReference.Get())
    return reinterpret_cast<uint64_t>(&Lookup);
  if (id == kIdLookupFormById.Get())
    return reinterpret_cast<uint64_t>(&LookupForm);
  return 0;
}
int main() {
  using namespace bfa;
  char old[MAX_PATH], temp[MAX_PATH], root[MAX_PATH];
  CHECK(GetCurrentDirectoryA(MAX_PATH, old));
  CHECK(GetTempPathA(MAX_PATH, temp));
  CHECK(GetTempFileNameA(temp, "bfn", 0, root));
  CHECK(DeleteFileA(root) && CreateDirectoryA(root, nullptr));
  CHECK(SetCurrentDirectoryA(root));
  CreateDirectoryA("Data", nullptr);
  CreateDirectoryA("Data\\SKSE", nullptr);
  CreateDirectoryA("Data\\SKSE\\Plugins", nullptr);
  constexpr char ini[] = "Data\\SKSE\\Plugins\\BetterFacialAnimations.ini";
  LoadIni();
  CHECK(NpcHotkey() == 0);
  for (int key : {-1, 1, 256, 99999}) {
    SetNpcHotkey(key);
    CHECK(NpcHotkey() == 0);
  }
  CHECK(WritePrivateProfileStringA("General", "BlacklistHotkey", "66", ini));
  LoadIni();
  CHECK(NpcHotkey() == 66);  // legacy binding migrates without rebinding
  SaveIni();
  LoadIni();
  CHECK(NpcHotkey() == 66);
  CHECK(GetPrivateProfileIntA("General", "OverridesHotkey", 0, ini) == 66);
  CHECK(GetPrivateProfileIntA("General", "BlacklistHotkey", 0, ini) == 0);
  Npc full("Follower.esp", 0x05001234), light("Light.esl", 0xFE123ABC);
  Npc same_local("Other.esp", 0x07001234),
      temporary("Follower.esp", 0xFF001234);
  uint8_t* forms[] = {full.form.data(), light.form.data(),
                      same_local.form.data(), nullptr};
  Field<BSTArray>(handler, kDhFormArrays + kFtNpc * sizeof(BSTArray)) = {
      forms, 4, 0, 4};
  actor[kFormType] = kFtActor;
  Field<uint32_t>(actor, kFormId) = 0x100;
  Field<uint8_t*>(actor, kRefrBase) = full.form.data();
  LoadNpcSettings();
  CHECK(SnapshotOverrides().empty());
  auto npc = DescribeNpc(actor);
  CHECK(npc.key == "follower.esp|001234" && npc.label == "Val Serano");
  CHECK(NpcAllowed(actor));
  SetNpcBlocked(npc, true);
  CHECK(!NpcAllowed(actor));
  Field<uint8_t*>(actor, kRefrBase) = light.form.data();
  CHECK(NpcAllowed(actor));
  SetNpcBlocked(DescribeNpc(actor), true);
  SaveIni();
  CHECK(GetPrivateProfileIntA("Overrides", "Follower.esp|001234", 0, ini) == 1);
  CHECK(GetPrivateProfileIntA("Overrides", "Light.esl|000ABC", 0, ini) == 1);
  Field<uint8_t*>(actor, kRefrBase) = same_local.form.data();
  CHECK(NpcAllowed(actor));  // same local id, different plugin
  Field<uint8_t*>(actor, kRefrBase) = temporary.form.data();
  CHECK(DescribeNpc(actor).key.empty());
  CHECK(DescribeNpc(nullptr).key.empty());
  Field<uint32_t>(actor, kFormId) = kPlayerFormId;
  CHECK(DescribeNpc(actor).key.empty() && NpcAllowed(actor));
  Field<uint32_t>(actor, kFormId) = 0x100;
  // Reordered full/light plugins and differently cased names retain choices.
  Field<uint32_t>(full.form.data(), kFormId) = 0x09001234;
  Field<uint32_t>(light.form.data(), kFormId) = 0xFE456ABC;
  std::strcpy(reinterpret_cast<char*>(full.file.data() + kFileName),
              "FOLLOWER.ESP");
  CHECK(WritePrivateProfileStringA("Blacklist", "Absent.esp|123ABC", "1", ini));
  CHECK(WritePrivateProfileStringA("Blacklist", "Invalid", "1", ini));
  LoadNpcSettings();
  CHECK(SnapshotOverrides().size() == 3);
  Field<uint8_t*>(actor, kRefrBase) = full.form.data();
  CHECK(!NpcAllowed(actor));
  Field<uint8_t*>(actor, kRefrBase) = light.form.data();
  CHECK(!NpcAllowed(actor));
  ResetTune();
  CHECK(!NpcAllowed(actor) && NpcHotkey() == 66);
  SaveIni();
  CHECK(GetPrivateProfileIntA("Overrides", "Absent.esp|123ABC", 0, ini) == 1);
  // Temporarily absent entries are removable, and removing one preserves
  // others.
  for (const auto& entry : SnapshotOverrides())
    if (!entry.id) SetNpcBlocked(entry, false);
  SaveIni();
  LoadNpcSettings();
  CHECK(SnapshotOverrides().size() == 2);
  CHECK(GetPrivateProfileIntA("Blacklist", "Absent.esp|123ABC", 0, ini) == 0);
  CHECK(WritePrivateProfileStringA("Overrides", "Absent.esp|123ABC", "1", ini));
  CHECK(WritePrivateProfileStringA("Override:Absent.esp|123ABC", "MaxJoy",
                                   "0.2", ini));
  LoadNpcSettings();
  SaveIni();
  LoadNpcSettings();
  for (const auto& entry : SnapshotOverrides())
    if (!entry.id) {
      CHECK(entry.values.at("MaxJoy") == .2f);
      RemoveNpcOverride(entry);
    }
  SaveIni();
  LoadNpcSettings();

  for (uint32_t runtime : {skse::Version(1, 5, 97), skse::Version(1, 6, 353),
                           skse::Version(1, 6, 1170), skse::kVrRuntime}) {
    skse::SetRuntime(runtime);
    Field<uint8_t*>(actor, kRefrBase) = full.form.data();
    Field<uint8_t*>(actor, kActorRace.Get()) = race.data();
    Field<uint32_t>(race.data(), kFormId) = 0x13746;
    Field<uint32_t>(race.data(), kRaceFlags) = 2;
    Field<LONG>(actor, kRefrRefCount) = 1;
    npc = DescribeNpc(actor);
    CHECK(npc.blocked && !SpeakerOk(actor));
    CHECK(BeginSpeech(actor, "blocked", 0, 2, 0, 0, Tune(), world_epoch) ==
          BFA_kResultDisabled);
    CHECK(queue.empty() && playing.empty());
    ready = net_loaded = true;
    ProviderActor provider;
    CHECK(ResolveProviderActor(0x100, provider) == BFA_kResultDisabled);
    ready = net_loaded = false;
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
    for (float v : values) CHECK(v == 0.65f);
    seed();
    Field<uint8_t*>(node.data(), kNodeFg.Get()) = fg;
    AddDirect(node.data(), false);
    CHECK(faces.empty() && playing.empty() && fg[kFgDirty]);
    CHECK(Field<LONG>(actor, kRefrRefCount) == 1);
    SetNpcBlocked(npc, false);
    CHECK(SpeakerOk(actor));
    partner_fg = fg;
    partner_ref = actor;
    partner_handle = 42;
    Expressions(fg, .1f, Tune(), false, actor, 42);
    CHECK(faces.count(fg) && faces.at(fg).ownership_blend > 0);
    // Re-blocking also relinquishes an idle conversation face at zero dt.
    SetNpcBlocked(npc, true);
    CHECK(Expressions(fg, 0, Tune(), false, actor, 42) & kFaceChanged);
    CHECK(faces.empty());
    ResetPlayback(false);
  }
  SetNpcBlocked(npc, false);
  SaveIni();
  LoadNpcSettings();
  CHECK(NpcAllowed(actor) && SnapshotOverrides().size() == 1);
  CHECK(SnapshotOverrides()[0].key == "light.esl|000ABC");
  // Sparse values inherit the current profile and persist independently.
  Field<uint8_t*>(actor, kRefrBase) = full.form.data();
  npc = DescribeNpc(actor);
  npc.values = {{"DirectMouth", .4f},    {"SmileEyes", .7f},
                {"VoiceEmotion", .25f},  {"MaxJoy", .3f},
                {"TimeOffset", .1f},     {"DisableTongue", 1},
                {"ConversationGaze", 0}, {"GazeStrength", .8f}};
  SetNpcOverride(npc);
  Tune defaults;
  defaults.male.upper_lip = .8f;
  defaults.female.upper_lip = 1.2f;
  auto face = NpcFaceTune(defaults, actor, false, false);
  CHECK(face.direct_scale[direct::kMouth] == .4f && face.smile_eyes == .7f);
  CHECK(face.upper_lip == .8f && face.time_offset == .1f);
  CHECK(NpcFaceTune(defaults, actor, true, false).upper_lip == 1.2f);
  CHECK(NpcFaceTune(defaults, actor, false, true).voice_emotion == 1);
  SaveIni();
  LoadNpcSettings();
  CHECK(DescribeNpc(actor).values == npc.values && NpcAllowed(actor));
  const auto direct = NpcDirectTune(defaults, actor);
  CHECK(direct.disable_tongue && !direct.conversation_gaze &&
        direct.gaze_strength == .8f);
  defaults.male.upper_lip = 1.4f;
  CHECK(NpcFaceTune(defaults, actor, false, false).upper_lip == 1.4f);
  // New format wins over a conflicting legacy entry; bad floats are
  // ignored/clamped.
  CHECK(WritePrivateProfileStringA("Blacklist", npc.key.c_str(), "1", ini));
  CHECK(WritePrivateProfileStringA("Override:follower.esp|001234", "UpperLip",
                                   "nan", ini));
  CHECK(WritePrivateProfileStringA("Override:follower.esp|001234", "LowerLip",
                                   "99", ini));
  LoadNpcSettings();
  CHECK(NpcAllowed(actor));
  face = NpcFaceTune(defaults, actor, false, false);
  CHECK(face.upper_lip == 1.4f && face.lower_lip == 3);
  SaveIni();
  CHECK(GetPrivateProfileIntA("Blacklist", npc.key.c_str(), 0, ini) == 0);
  // Inference receives effective emotions; cached lines cannot leak an
  // override.
  model.faces.resize(1);
  auto custom = Request("npc_override", 3, 100, false, false, defaults, 0,
                        world_epoch, false, nullptr, {}, actor);
  CHECK(custom && queue.back().emotion.gain == .25f);
  CHECK(queue.back().emotion.maximum[a2f::kJoy] == .3f);
  Field<uint8_t*>(actor, kRefrBase) = same_local.form.data();
  auto normal = Request("npc_override", 3, 100, false, false, defaults, 0,
                        world_epoch, false, nullptr, {}, actor);
  CHECK(normal && normal != custom && queue.back().emotion.gain == 1);
  CHECK(
      NpcFaceTune(defaults, actor, false, false).direct_scale[direct::kMouth] ==
      1);
  Field<uint8_t*>(actor, kRefrBase) = full.form.data();
  npc = DescribeNpc(actor);
  npc.values["VoiceEmotion"] = .6f;
  SetNpcOverride(npc);
  auto revised = Request("npc_override", 3, 100, false, false, defaults, 0,
                         world_epoch, false, nullptr, {}, actor);
  CHECK(revised != custom && queue.back().emotion.gain == .6f);
  ResetTune();
  CHECK(NpcFaceTune(defaults, actor, false, false).voice_emotion == .6f);
  RemoveNpcOverride(npc);
  SaveIni();
  LoadNpcSettings();
  CHECK(DescribeNpc(actor).values.empty() && NpcAllowed(actor));
  CHECK(GetPrivateProfileIntA("Overrides", npc.key.c_str(), 0, ini) == 0);
  CHECK(NpcFaceTune(defaults, actor, false, false).voice_emotion == 1);
  CHECK(!NpcDirectTune(defaults, actor).disable_tongue);
  // Independently exercise every kind of profile field after replacing
  // layout-based access: scalar, region array and emotion-cap array.
  npc.values = {
      {"TimeOffset", .2f},   {"BlendIn", .3f},        {"VoiceSwell", .4f},
      {"FaceFadeIn", .6f},   {"FaceFadeOut", .7f},    {"VoiceEmotion", .8f},
      {"UpperLip", .9f},     {"LowerLip", 1.1f},      {"SmileEyes", 1.2f},
      {"DirectBrows", .1f},  {"DirectEyes", .2f},     {"DirectCheeks", .3f},
      {"DirectMouth", .4f},  {"DirectJaw", .5f},      {"DirectTongue", .6f},
      {"MaxAmazement", .1f}, {"MaxAnger", .2f},       {"MaxCheekiness", .3f},
      {"MaxDisgust", .4f},   {"MaxFear", .5f},        {"MaxGrief", .6f},
      {"MaxJoy", .7f},       {"MaxOutOfBreath", .8f}, {"MaxPain", .9f},
      {"MaxSadness", 1.f}};
  SetNpcOverride(npc);
  SaveIni();
  LoadNpcSettings();
  for (bool female : {false, true}) {
    const auto f = NpcFaceTune(defaults, actor, female, false);
    CHECK(f.time_offset == .2f && f.blend_in == .3f);
    CHECK(f.voice_swell == .4f && f.face_in == .6f && f.face_out == .7f);
    CHECK(f.voice_emotion == .8f && f.upper_lip == .9f);
    CHECK(f.lower_lip == 1.1f && f.smile_eyes == 1.2f);
    CHECK(f.direct_scale[direct::kBrows] == .1f);
    CHECK(f.direct_scale[direct::kEyes] == .2f);
    CHECK(f.direct_scale[direct::kCheeks] == .3f);
    CHECK(f.direct_scale[direct::kMouth] == .4f);
    CHECK(f.direct_scale[direct::kJaw] == .5f);
    CHECK(f.direct_scale[direct::kTongue] == .6f);
    for (int i = 0; i < a2f::kNumMod; ++i)
      CHECK(std::fabs(f.max_emotion[i] - (i + 1) / 10.f) < 1e-6f);
  }
  // Persisted deletion markers must not keep runtime override work enabled.
  for (const auto& entry : SnapshotOverrides()) RemoveNpcOverride(entry);
  CHECK(SnapshotOverrides().empty() && !NpcOverridesActive());
  npc.values = {{"UnknownSetting", 1}, {"UpperLip", NAN}};
  SetNpcOverride(npc);
  CHECK(!NpcOverridesActive());
  npc.values = {{"DirectJaw", .4f}};
  SetNpcOverride(npc);
  CHECK(NpcOverridesActive());
  RemoveNpcOverride(npc);
  CHECK(!NpcOverridesActive());
  SaveIni();
  LoadNpcSettings();
  CHECK(SnapshotOverrides().empty() && !NpcOverridesActive());
  CHECK(GetPrivateProfileIntA("Overrides", npc.key.c_str(), 0, ini) == 0);
  DeleteFileA(ini);
  CHECK(RemoveDirectoryA("Data\\SKSE\\Plugins"));
  CHECK(RemoveDirectoryA("Data\\SKSE"));
  CHECK(RemoveDirectoryA("Data"));
  CHECK(SetCurrentDirectoryA(old) && RemoveDirectoryA(root));
  std::puts(
      "NPC overrides: migration, inheritance, persistence, cache isolation and "
      "live handoff passed");
}
