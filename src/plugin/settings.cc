// The ini and the table of settings declared in plugin.h.
#include <windows.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "a2f/inference/net_keeper.h"
#include "a2f/plugin/plugin.h"
#include "a2f/runtime/engine.h"
#include "a2f/runtime/scoped_lock.h"
#include "a2f/runtime/skse.h"

namespace bfa {

Config& config = *new Config;
namespace {
Tune tune;
// Sorted by runtime id; settings_lock guards it. Outlives the INI writer.
std::vector<RaceSetting>& races = *new std::vector<RaceSetting>;
std::map<std::string, NpcSetting>& npcs =
    *new std::map<std::string, NpcSetting>;
std::unordered_set<uint32_t>& blocked_npcs = *new std::unordered_set<uint32_t>;
std::unordered_map<uint32_t, std::string>& npc_ids =
    *new std::unordered_map<uint32_t, std::string>;
int npc_hotkey = 0;
// Any override resolved to a base in this load order: the per-frame tune
// lookups skip settings_lock while this is false.
std::atomic<bool> npc_overrides{false};

// Races that start disabled: their heads differ too much from the model's.
constexpr const char* kOptInRaceFiles[] = {"DWAlfiqKhajiit.esp"};

bool DefaultRaceEnabled(uint32_t, const char* file = "") {
  for (const char* opt_in : kOptInRaceFiles)
    if (!_stricmp(file, opt_in)) return false;
  return true;
}

// Caller holds settings_lock.
std::vector<RaceSetting>::iterator FindRace(uint32_t id) {
  return std::lower_bound(
      races.begin(), races.end(), id,
      [](const RaceSetting& race, uint32_t value) { return race.id < value; });
}
}  // namespace
SRWLOCK settings_lock = SRWLOCK_INIT;

extern const std::vector<Param> kParams = [] {
  std::vector<Param> params = {
      {"Enabled", nullptr, 0, 0, "Enable facial animations",
       "Untick for the game's own lip-sync, instantly (A/B).", Sec::kGeneral,
       &tune.enabled},
      {"PlayerFacialAnimations", nullptr, 0, 0, "Player facial animations",
       "Automatically detect DBReV or DBVO2 for player dialogue. "
       "DBVO2 prototype for Skyrim. Untick to keep the voice mod's own "
       "animation.",
       Sec::kGeneral, &tune.player},
      {"PotatoMode", nullptr, 0, 0, "Potato Mode",
       "Only animate the player and NPC in an active dialogue window with one another.",
       Sec::kGeneral, &tune.potato_mode},
      {"VanillaEmotions", nullptr, 0, 0, "Vanilla / scene emotion inputs",
       "Pass authored scene and dialogue-mod emotions to the facial model. "
       "On by default. Sentiment Analysis works independently. "
       "Applies during BFA speech.",
       Sec::kGeneral, &tune.vanilla_emotions, -1, true, Param::kNone,
       Group::kSources},
      {"SentimentAnalysis", nullptr, 0, 0, "Sentiment Analysis",
       "Sentiment Analysis is a HEAVY FEATURE: its additional model needs "
       "600 MB of EXTRA VRAM on GPU, substantial processing power and longer "
       "animation generation time. Adds voice-detected emotion to the model "
       "inputs alongside authored scene emotion. Requires emotion_fp16.onnx. "
       "Off by default. Disabling unloads it after current inference. "
       "Force CPU uses system memory and additional CPU processing.",
       Sec::kGeneral, &tune.sentiment_analysis, -1, true, Param::kSentiment,
       Group::kSources},
      {"SentimentMultiplier", &tune.sentiment_multiplier, 0, 2,
       "Sentiment addition multiplier",
       "Scales sentiment added to the model inputs: 0 = none, 0.5 = half, "
       "1 = normal, 2 = double. Requires Sentiment Analysis. Final emotion "
       "limits apply. 0 skips sentiment inference. Applies next line.",
       Sec::kGeneral, nullptr, -1, true, Param::kSentiment, Group::kSources},
      {"Prefetch", nullptr, 0, 0, "Prepare dialogue in advance",
       "Generate a line when its dialogue response is created, before it is "
       "spoken.",
       Sec::kGeneral, &tune.prefetch},
      {"UseV3Model", nullptr, 0, 0, "V3 model",
       "Audio2Face v3.0. The most expressive model, with detailed lip syncing "
       "at 60 fps. Roughly 2x as fast as the Old "
       "model in RTX 4070 GPU tests; speed varies by system and dialogue. "
       "About 250 MB of VRAM on GPU. Switches live; already playing lines "
       "finish with their current animation.",
       Sec::kModel, &tune.v3_model},
      {"UsePerformanceModel", nullptr, 0, 0, "Performance model (Experimental)",
       "Smaller model distilled from V3's output on Skyrim's dialogue. "
       "Fastest to run: roughly 8x as fast as V3 in RTX 4070 GPU tests; "
       "speed varies by system and dialogue. Less expressive and less "
       "accurate at lip syncing; "
       "lip syncing on modded dialogue may be degraded. Runs at 30 fps. "
       "About 80 MB of VRAM on GPU; usage "
       "varies by system. Switches live; already playing lines finish with their "
       "current animation. Eye gaze is unavailable with this model.",
       Sec::kModel, &tune.performance_model},
      {"UseOldModel", nullptr, 0, 0, "Old model",
       "Audio2Face v2.3. Less expressive than V3 and the slowest to run: "
       "roughly half V3's speed in RTX 4070 GPU tests. Speed varies by "
       "system and dialogue. Lower VRAM use than V3, "
       "at about 100 MB of VRAM on GPU. Uses one shared facial "
       "identity at 30 fps. Switches live; already playing lines finish "
       "with their current animation.",
       Sec::kModel, &tune.old_model},
      {"ForceCPU", nullptr, 0, 0, "Force CPU",
       "Run the selected model on CPU. V3 and Old "
       "run much slower on CPU; long lines can start late.\n"
       "Applies live after the current inference call finishes.",
       Sec::kModel, &tune.force_cpu},
      {"DisableTongue", nullptr, 0, 0, "Disable tongue motion",
       "Restore the tongue behavior from before this feature. Applies "
       "immediately.",
       Sec::kDirect, &tune.disable_tongue},
      {"ConversationGaze", nullptr, 0, 0, "Enable eye gaze",
       "Add speech-driven eye motion while the conversation NPC targets you. "
       "Untick to keep the game's eye movement. Applies immediately; SE/AE "
       "only.",
       Sec::kDirect, &tune.conversation_gaze},
      {"GazeStrength", &tune.gaze_strength, 0, 2, "Gaze strength",
       "Speech eye offset strength; 0 preserves the game's gaze.", Sec::kDirect,
       nullptr},
  };
  static char mouth_tip[160];
  snprintf(mouth_tip, sizeof mouth_tip,
           "Lip and mouth skin movement. Default strength: 1.0; %g with "
           "this sex's Expressive Facial Animation edition detected.",
           kExpressiveMouthScale);
  for (int profile = 0; profile < 3; ++profile) {
    FaceTune& face =
        profile == 2 ? tune.player_face : profile ? tune.female : tune.male;
    auto face_param = [&](Param p, float* (*access)(FaceTune&, size_t),
                          size_t index = 0) {
      p.face_value = access;
      p.face_index = index;
      p.value = p.FaceValue(face);
      p.profile = profile;
      return p;
    };
    for (Param p : std::initializer_list<Param>{
             face_param(
                 {"TimeOffset", nullptr, -0.3f, 0.3f, "Sync offset (s)",
                  "Positive = mouth shapes earlier.", Sec::kPlayback, nullptr},
                 [](FaceTune& f, size_t) { return &f.time_offset; }),
             face_param({"BlendIn", nullptr, 0, 0.5f, "Blend-in (s)",
                         "Crossfade from vanilla lip-sync when a line is ready "
                         "after it "
                         "started.",
                         Sec::kPlayback, nullptr},
                        [](FaceTune& f, size_t) { return &f.blend_in; }),
             face_param({"DirectBrows", nullptr, 0, 3, "Brows",
                         "Brow movement.", Sec::kDirect, nullptr},
                        [](FaceTune& f, size_t) {
                          return &f.direct_scale[direct::kBrows];
                        }),
             face_param(
                 {"DirectEyes", nullptr, 0, 3, "Eyes",
                  "Eyelid and eye region movement.", Sec::kDirect, nullptr},
                 [](FaceTune& f, size_t) {
                   return &f.direct_scale[direct::kEyes];
                 }),
             face_param(
                 {"SmileEyes", nullptr, 0, 2, "Smile reaches eyes",
                  "Extra happy-expression movement around the eyes. 0 restores "
                  "the original behavior. Default: 1.0 for every profile. "
                  "Applies immediately.",
                  Sec::kDirect, nullptr},
                 [](FaceTune& f, size_t) { return &f.smile_eyes; }),
             face_param({"DirectCheeks", nullptr, 0, 3, "Cheeks & nose",
                         "Cheek and nose movement.", Sec::kDirect, nullptr},
                        [](FaceTune& f, size_t) {
                          return &f.direct_scale[direct::kCheeks];
                        }),
             face_param({"DirectMouth", nullptr, 0, 3, "Mouth", mouth_tip,
                         Sec::kDirect, nullptr},
                        [](FaceTune& f, size_t) {
                          return &f.direct_scale[direct::kMouth];
                        }),
             face_param({"DirectJaw", nullptr, 0, 3, "Jaw",
                         "How far the lower teeth and gums move with the jaw.",
                         Sec::kDirect, nullptr},
                        [](FaceTune& f, size_t) {
                          return &f.direct_scale[direct::kJaw];
                        }),
             face_param(
                 {"DirectTongue", nullptr, 0, 3, "Tongue",
                  "Tongue articulation strength for all phonemes. 1 uses the "
                  "calibrated default; 0 keeps jaw follow only.",
                  Sec::kDirect, nullptr},
                 [](FaceTune& f, size_t) {
                   return &f.direct_scale[direct::kTongue];
                 }),
             face_param({"UpperLip", nullptr, 0, 3, "Upper lip",
                         "Upper lip movement, multiplied by Mouth strength.",
                         Sec::kDirect, nullptr},
                        [](FaceTune& f, size_t) { return &f.upper_lip; }),
             face_param({"LowerLip", nullptr, 0, 3, "Lower lip",
                         "Lower lip movement, multiplied by Mouth strength.",
                         Sec::kDirect, nullptr},
                        [](FaceTune& f, size_t) { return &f.lower_lip; }),
             // all of these apply from the next line spoken
             face_param(
                 {"VoiceEmotion", nullptr, 0, 3,
                  "Scene emotion input multiplier",
                  "Multiply authored scene emotion before it enters the model. "
                  "1 = unchanged; 0 = none. Requires Vanilla / scene emotion "
                  "inputs enabled. Applies next line.",
                  Sec::kEmotion, nullptr, -1, true, Param::kScene},
                 [](FaceTune& f, size_t) { return &f.voice_emotion; }),
             face_param({"VoiceSwell", nullptr, 0, 2,
                         "Scene emotion boost from voice volume",
                         "Boost authored scene emotion where the voice is "
                         "loud. 0 = steady. Applies from the next line.",
                         Sec::kEmotion, nullptr, -1, true, Param::kScene},
                        [](FaceTune& f, size_t) { return &f.voice_swell; }),
             face_param(
                 {"FaceFadeIn", nullptr, 0, 3,
                  "Fade into generated expressions (s)",
                  "When a line starts: how long the game's own expressions "
                  "take to fade out, leaving the face to Audio2Face.",
                  Sec::kEmotion, nullptr, -1, true, Param::kNone,
                  Group::kTransitions},
                 [](FaceTune& f, size_t) { return &f.face_in; }),
             face_param(
                 {"FaceFadeOut", nullptr, 0, 3,
                  "Restore game expressions after speech (s)",
                  "After a line: how long the game's own expressions take to "
                  "fade back in.",
                  Sec::kEmotion, nullptr, -1, true, Param::kNone,
                  Group::kTransitions},
                 [](FaceTune& f, size_t) { return &f.face_out; }),
         }) {
      params.push_back(p);
    }
    static const char* const kKeys[a2f::kNumMod] = {
        "MaxAmazement", "MaxAnger", "MaxCheekiness", "MaxDisgust", "MaxFear",
        "MaxGrief", "MaxJoy", "MaxOutOfBreath", "MaxPain", "MaxSadness"};
    static const char* const kLabels[a2f::kNumMod] = {
        "Maximum surprise input",       "Maximum anger input",
        "Maximum cheekiness input",     "Maximum disgust input",
        "Maximum fear input",           "Maximum grief input",
        "Maximum happiness input",      "Maximum breathlessness input",
        "Maximum pain input",           "Maximum sadness input"};
    for (int c = 0; c < a2f::kNumMod; ++c) {
      const bool unused =
          c == a2f::kGrief || c == a2f::kOutOfBreath || c == a2f::kPain;
      Param p = face_param(
          {kKeys[c], nullptr, 0, 2, kLabels[c],
           unused
               ? "Maximum input for this emotion, supplied explicitly by "
                 "dialogue mods. 0 blocks this input; 2 is the highest limit."
               : "MAXIMUM amount of this emotion the model can receive after "
                 "scene multipliers, volume boost and added sentiment are "
                 "combined. This is a ceiling. 0 blocks this input; 1 is the "
                 "default; 2 is the highest limit.",
           Sec::kEmotion, nullptr},
          [](FaceTune& f, size_t index) { return &f.max_emotion[index]; }, c);
      p.group = Group::kLimits;
      params.push_back(p);
    }
  }
  params.push_back(
      {"Logging", nullptr, 0, 0, "Logging",
       "Also write the generated and played lines to "
       "BetterFacialAnimations.log (SKSE log folder). The GPU, the driver and "
       "model problems are logged either way.",
       Sec::kDebug, &tune.logging});
  for (Param& p : params)
    if (p.flag == &tune.v3_model) p.stored = false;
  return params;
}();

const ModelInfo& ModelFor(int network) {
  static constexpr ModelInfo kModels[] = {
      {a2f::kNetV3, L"a2f3_fp16.onnx", L"A2FDirect3.bin", a2f::kNetV3,
       a2f::kNetVramMb3},
      {a2f::kNetV2, L"network_fp16.onnx", L"A2FDirect.bin", a2f::kNetV2,
       a2f::kNetVramMb},
      {a2f::kNetPerformance, L"performance.onnx", L"A2FDirect3.bin",
       a2f::kNetV3, a2f::kNetVramMbPerformance},
  };
  for (const ModelInfo& info : kModels)
    if (info.network == network) return info;
  return kModels[0];
}
extern const size_t kNumParams = std::size(kParams);

namespace {

SRWLOCK ini_write_lock = SRWLOCK_INIT;

DWORD WINAPI IniWriter(void* event) {
  for (;;) {
    if (WaitForSingleObject(event, INFINITE) != WAIT_OBJECT_0) return 1;
    // Every edit restarts the quiet period, including edits after menu close.
    DWORD result;
    do {
      result = WaitForSingleObject(event, 250);
    } while (result == WAIT_OBJECT_0);
    if (result != WAIT_TIMEOUT) return 1;
    SaveIni();
  }
}

constexpr char kIni[] = "Data\\SKSE\\Plugins\\BetterFacialAnimations.ini";

// A form's identity across load orders: "<plugin>|<local id>" as the ini
// stores it; empty without a source plugin.
std::string FormKey(uint8_t* form) {
  const char* file = SourceFileName(form);
  if (!file || !*file) return {};
  char suffix[16];
  snprintf(suffix, sizeof suffix, "|%06X",
           LocalFormId(Field<uint32_t>(form, kFormId)));
  return file + std::string(suffix);
}

// NPC keys lower the plugin name so that its spelling cannot split entries;
// the local id stays upper-case hex.
std::string NpcKey(uint8_t* base) {
  if (!base || base[kFormType] != kFtNpc) return {};
  const uint32_t id = Field<uint32_t>(base, kFormId);
  if (!id || (id >> 24) == 0xFF) return {};  // save-created: not stable
  const std::string key = FormKey(base);
  const size_t split = key.find('|');
  if (key.empty() || key.find_first_of("=[]\r\n/\\") != std::string::npos ||
      key.rfind('|') != split)
    return {};
  return Lower(key.substr(0, split).c_str()) + key.substr(split);
}

}  // namespace

std::string Lower(const char* s) {
  std::string n = s;
  for (auto& c : n) c = static_cast<char>(tolower(static_cast<uint8_t>(c)));
  return n;
}

namespace {

std::string NpcLabel(uint8_t* base) {
  const char* name = Field<const char*>(base, kNpcName);
  return name && *name ? name : "Unnamed NPC";
}

// Version 2 renamed Direct to LocalAnimationStrength and changed its defaults.
// Version 3 turned Logging on by default: an older ini's Logging=0 was that
// version's default, not a choice, so it is turned on once.
// Preserve unrelated keys, including startup configuration, during migration.
// Version 4 gives each sex its own sliders; shared values seed both profiles.
// Version 5 adds a player profile and normalizes lips to backend calibration.
// Version 6 turns Argonians on by default and drops their saved off state.
// Version 7 lowers the emotion caps' default to 1 and adds NPC overrides.
constexpr int kIniVersion = 8;

std::string Section(const Param& p) {
  static const char* const kNames[] = {"General",  "General",
                                       "Playback", "LocalAnimationStrength",
                                       "Emotion",  "Debug"};
  std::string section = kNames[static_cast<int>(p.sec)];
  if (p.profile >= 0)
    section += p.profile == 2 ? "Player" : p.profile ? "Female" : "Male";
  return section;
}

}  // namespace

void LoadRaceSettings() {
  std::vector<RaceSetting> loaded;
  const BSTArray* forms = FormArray(kFtRace);
  if (!forms) return;
  for (uint32_t i = 0; i < forms->n; ++i) {
    uint8_t* form = forms->items[i];
    if (!form || form[kFormType] != kFtRace ||
        !(Field<uint32_t>(form, kRaceFlags) & 2))
      continue;
    const uint32_t id = Field<uint32_t>(form, kFormId);
    const char* file = SourceFileName(form);
    const std::string key = FormKey(form);
    if (key.empty()) continue;
    const char* name = Field<const char*>(form, kRaceName);
    const char* editor = Field<const char*>(form, kRaceEditorId);
    std::string label = name && *name ? name : "Unnamed race";
    if (editor && *editor) label += " (" + std::string(editor) + ")";
    const bool default_enabled = DefaultRaceEnabled(id, file);
    const bool enabled = GetPrivateProfileIntA(
        "Races", key.c_str(), default_enabled, kIni) != 0;
    loaded.push_back({id, key, label, enabled, default_enabled});
  }
  std::sort(loaded.begin(), loaded.end(),
            [](const RaceSetting& left, const RaceSetting& right) {
              return left.id < right.id;
            });
  ScopedLock guard(settings_lock);
  races = std::move(loaded);
}

std::vector<RaceSetting> SnapshotRaces() {
  ScopedLock guard(settings_lock, ScopedLock::kShared);
  return races;
}

bool RaceEnabled(uint32_t id) {
  ScopedLock guard(settings_lock, ScopedLock::kShared);
  const auto it = FindRace(id);
  return it != races.end() && it->id == id ? it->enabled
                                         : DefaultRaceEnabled(id);
}

void SetRaceEnabled(uint32_t id, bool enabled) {
  ScopedLock guard(settings_lock);
  const auto it = FindRace(id);
  if (it != races.end() && it->id == id) it->enabled = enabled;
}

void LoadNpcSettings() {
  // Legacy exclusions migrate to disabled overrides. New entries take
  // precedence.
  std::map<std::string, NpcSetting> loaded;
  for (const char* section : {"Blacklist", "Overrides"}) {
    std::vector<char> keys(4096);
    for (;;) {
      const DWORD n = GetPrivateProfileStringA(section, nullptr, "",
                                               keys.data(), keys.size(), kIni);
      if (n < keys.size() - 2) break;
      if (keys.size() >= 1024 * 1024) {
        skse::Report(
            "NPC overrides too large to read; keeping current settings");
        return;
      }
      keys.resize(keys.size() * 2);
    }
    for (const char* p = keys.data(); *p; p += std::strlen(p) + 1) {
      if (GetPrivateProfileIntA(section, p, 0, kIni) != 1) continue;
      std::string key = p;
      const auto split = key.find('|');
      if (split == std::string::npos || !split || key.size() != split + 7 ||
          key.find_first_of("=[]\r\n/\\") != std::string::npos)
        continue;
      bool valid = true;
      for (size_t i = split + 1; i < key.size(); ++i) {
        char& c = key[i];
        if (c >= 'a' && c <= 'f') c -= 'a' - 'A';
        if (!((c >= '0' && c <= '9') || (c >= 'A' && c <= 'F'))) valid = false;
      }
      if (!valid) continue;
      key = Lower(key.substr(0, split).c_str()) + key.substr(split);
      NpcSetting npc{0, key, key, !strcmp(section, "Blacklist")};
      if (!npc.blocked) {
        const std::string detail = "Override:" + key;
        npc.blocked =
            !GetPrivateProfileIntA(detail.c_str(), "Enabled", 1, kIni);
        for (const Param& param : kParams) {
          if (!NpcParamSupported(param)) continue;
          char value[128];
          GetPrivateProfileStringA(detail.c_str(), param.key, "", value,
                                   sizeof value, kIni);
          char* end;
          const float v = std::strtof(value, &end);
          if (end != value && !*end && std::isfinite(v))
            npc.values[param.key] = std::clamp(v, param.flag ? 0.f : param.lo,
                                               param.flag ? 1.f : param.hi);
        }
      }
      loaded[key] = std::move(npc);
    }
  }
  std::unordered_set<uint32_t> ids;
  if (const BSTArray* forms = FormArray(kFtNpc)) {
    for (uint32_t i = 0; i < forms->n; ++i) {
      uint8_t* form = forms->items[i];
      const auto it = loaded.find(NpcKey(form));
      if (it == loaded.end()) continue;
      it->second.id = Field<uint32_t>(form, kFormId);
      it->second.label = NpcLabel(form);
      it->second.female = (Field<uint32_t>(form, kNpcFlags) & 1) != 0;
      if (it->second.blocked) ids.insert(it->second.id);
    }
  }
  ScopedLock guard(settings_lock);
  npcs = std::move(loaded);
  blocked_npcs = std::move(ids);
  npc_ids.clear();
  for (const auto& [key, npc] : npcs)
    if (npc.id && (npc.blocked || !npc.values.empty())) npc_ids[npc.id] = key;
  npc_overrides = !npc_ids.empty();
}

NpcSetting DescribeNpc(uint8_t* actor) {
  if (!actor || actor[kFormType] != kFtActor ||
      Field<uint32_t>(actor, kFormId) == kPlayerFormId)
    return {};
  auto* base = Field<uint8_t*>(actor, kRefrBase);
  const std::string key = NpcKey(base);
  if (key.empty()) return {};
  NpcSetting npc{Field<uint32_t>(base, kFormId), key, NpcLabel(base), false};
  ScopedLock guard(settings_lock, ScopedLock::kShared);
  npc.female = (Field<uint32_t>(base, kNpcFlags) & 1) != 0;
  const auto it = npcs.find(key);
  if (it != npcs.end()) {
    npc.blocked = it->second.blocked;
    npc.values = it->second.values;
  }
  return npc;
}

std::vector<NpcSetting> SnapshotOverrides() {
  ScopedLock guard(settings_lock, ScopedLock::kShared);
  std::vector<NpcSetting> result;
  for (const auto& item : npcs)
    if (item.second.blocked || !item.second.values.empty())
      result.push_back(item.second);
  return result;
}

bool NpcOverridesActive() { return npc_overrides; }

bool NpcAllowed(uint8_t* actor) {
  if (!npc_overrides || !actor ||
      Field<uint32_t>(actor, kFormId) == kPlayerFormId)
    return true;
  auto* base = Field<uint8_t*>(actor, kRefrBase);
  if (!base || base[kFormType] != kFtNpc) return true;
  ScopedLock guard(settings_lock, ScopedLock::kShared);
  return !blocked_npcs.count(Field<uint32_t>(base, kFormId));
}

bool NpcParamSupported(const Param& param) {
  return (param.profile == 0 && param.face_value) ||
         (param.profile == -1 && (param.sec == Sec::kDirect ||
                                 !strcmp(param.key, "SentimentMultiplier") ||
                                 !strcmp(param.key, "SentimentAnalysis")));
}

Tune NpcSharedTune(Tune result, uint8_t* actor) {
  if (!npc_overrides || !actor ||
      Field<uint32_t>(actor, kFormId) == kPlayerFormId)
    return result;
  auto* base = Field<uint8_t*>(actor, kRefrBase);
  if (!base || base[kFormType] != kFtNpc) return result;
  ScopedLock guard(settings_lock, ScopedLock::kShared);
  const auto id = npc_ids.find(Field<uint32_t>(base, kFormId));
  if (id == npc_ids.end()) return result;
  const auto& values = npcs.at(id->second).values;
  auto get = [&](const char* key, float fallback) {
    const auto it = values.find(key);
    return it == values.end() ? fallback : it->second;
  };
  result.disable_tongue = get("DisableTongue", result.disable_tongue) != 0;
  result.conversation_gaze =
      get("ConversationGaze", result.conversation_gaze) != 0;
  result.gaze_strength = get("GazeStrength", result.gaze_strength);
  result.sentiment_analysis =
      result.sentiment_analysis && get("SentimentAnalysis", result.sentiment_analysis) != 0;
  result.sentiment_multiplier =
      get("SentimentMultiplier", result.sentiment_multiplier);
  return result;
}

FaceTune NpcFaceTune(const Tune& defaults, uint8_t* actor, bool female,
                     bool player) {
  FaceTune face = defaults.ForActor(female, player);
  if (!npc_overrides || !actor || player) return face;
  auto* base = Field<uint8_t*>(actor, kRefrBase);
  if (!base || base[kFormType] != kFtNpc) return face;
  ScopedLock guard(settings_lock, ScopedLock::kShared);
  const auto id = npc_ids.find(Field<uint32_t>(base, kFormId));
  if (id == npc_ids.end()) return face;
  const auto& values = npcs.at(id->second).values;
  for (const Param& param : kParams) {
    if (param.profile != 0) continue;
    float* target = param.FaceValue(face);
    if (!target) continue;
    const auto value = values.find(param.key);
    if (value != values.end()) *target = value->second;
  }
  return face;
}

void SetNpcOverride(const NpcSetting& npc) {
  if (npc.key.empty()) return;
  NpcSetting entry = npc;
  // Only known finite settings may enter the runtime or the INI writer.
  entry.values.clear();
  for (const Param& param : kParams) {
    if (!NpcParamSupported(param)) continue;
    const auto value = npc.values.find(param.key);
    if (value != npc.values.end() && std::isfinite(value->second))
      entry.values[param.key] =
          std::clamp(value->second, param.flag ? 0.f : param.lo,
                     param.flag ? 1.f : param.hi);
  }
  ScopedLock guard(settings_lock);
  const bool active = entry.blocked || !entry.values.empty();
  npcs[npc.key] = std::move(entry);
  if (npc.id) {
    if (active)
      npc_ids[npc.id] = npc.key;
    else
      npc_ids.erase(npc.id);
    if (npc.blocked)
      blocked_npcs.insert(npc.id);
    else
      blocked_npcs.erase(npc.id);
    npc_overrides = !npc_ids.empty();
  }
}

void RemoveNpcOverride(const NpcSetting& npc) {
  auto cleared = npc;
  cleared.blocked = false;
  cleared.values.clear();
  SetNpcOverride(cleared);  // tombstone lets the writer remove persisted keys
}

void SetNpcBlocked(const NpcSetting& npc, bool blocked) {
  auto changed = npc;
  {
    ScopedLock guard(settings_lock, ScopedLock::kShared);
    const auto it = npcs.find(npc.key);
    if (it != npcs.end()) changed.values = it->second.values;
  }
  changed.blocked = blocked;
  SetNpcOverride(changed);
}

int NpcHotkey() {
  ScopedLock guard(settings_lock, ScopedLock::kShared);
  return npc_hotkey;
}

void SetNpcHotkey(int key) {
  ScopedLock guard(settings_lock);
  npc_hotkey = key > 1 && key <= 255 ? key : 0;  // Escape always cancels
}

Tune SnapshotTune() {
  ScopedLock guard(settings_lock, ScopedLock::kShared);
  return tune;
}

void SelectModelCheckbox(const Param& p) {
  if (p.flag == &tune.v3_model) tune.SelectModel(a2f::kNetV3);
  if (p.flag == &tune.performance_model)
    tune.SelectModel(a2f::kNetPerformance);
  if (p.flag == &tune.old_model) tune.SelectModel(a2f::kNetV2);
}

void SetSelectedModel(int network) {
  ScopedLock guard(settings_lock);
  tune.SelectModel(network);
}

void ResetTune() {
  ScopedLock guard(settings_lock);
  // a diagnostic switch and the model, not tuning values
  const Tune old = tune;
  tune = Tune();
  tune.logging = old.logging;
  tune.SelectModel(old.SelectedModel());
  tune.force_cpu = old.force_cpu;
  for (RaceSetting& race : races) race.enabled = race.default_enabled;
}

void ApplyLogging() { skse::SetLogging(SnapshotTune().logging); }

bool LogMore(std::atomic<int>& written) {
  return SnapshotTune().logging && ++written <= config.log_lines;
}

std::wstring PluginDir() {
  wchar_t full[MAX_PATH];
  GetFullPathNameW(kPluginDir, MAX_PATH, full, nullptr);
  return full;
}

void LoadIni() {
  const DWORD sentiment_file =
      GetFileAttributesW((PluginDir() + L"\\" + kSentimentModel).c_str());
  config.sentiment_available = sentiment_file != INVALID_FILE_ATTRIBUTES &&
                               !(sentiment_file & FILE_ATTRIBUTE_DIRECTORY);
  // Startup only: no readers or menu callbacks exist yet.
  tune = Tune();
  SetNpcHotkey(GetPrivateProfileIntA(
      "General", "OverridesHotkey",
      GetPrivateProfileIntA("General", "BlacklistHotkey", 0, kIni), kIni));
  const int version =
      GetFileAttributesA(kIni) == INVALID_FILE_ATTRIBUTES
          ? kIniVersion
          : GetPrivateProfileIntA("General", "IniVersion", 0, kIni);
  const bool migrate = version < 2;  // the Direct section is dropped
  config.batch = std::max(
      1, static_cast<int>(GetPrivateProfileIntA("General", "Batch", 2, kIni)));
  config.cache_lines = std::max(8, static_cast<int>(GetPrivateProfileIntA(
                                       "General", "CacheLines", 300, kIni)));
  config.log_lines = GetPrivateProfileIntA("Debug", "LogLines", 400, kIni);
  // UseOldModel was Network=2 (3: v3.0) in earlier 3.0 builds, and ForceCPU
  // GPU=0
  tune.old_model = GetPrivateProfileIntA("General", "Network", 3, kIni) == 2;
  tune.force_cpu = GetPrivateProfileIntA("General", "GPU", 1, kIni) == 0;
  // Explicit route, as in 4.0. No Auto probe or GPU-to-GPU fallback. This
  // key takes precedence over retired GpuDevice/AutoGpuDevice settings.
  config.independent_device =
      GetPrivateProfileIntA("General", "IndependentGpuDevice", 1, kIni) != 0;
  char b[128], d[128];
  for (size_t i = 0; i < kNumParams; ++i) {
    const Param& p = kParams[i];
    if (!p.stored || (migrate && p.sec == Sec::kDirect)) continue;
    if (p.flag) {
      *p.flag =
          GetPrivateProfileIntA(Section(p).c_str(), p.key, *p.flag, kIni) != 0;
      continue;
    }
    const float calibration =
        !strcmp(p.key, "UpperLip")   ? direct::kUpperLipCalibration
        : !strcmp(p.key, "LowerLip") ? direct::kLowerLipCalibration
                                     : 1.f;
    const bool old_lip = version < 5 && p.profile >= 0 && p.profile < 2 &&
                         calibration != 1.f;
    // Missing keys use new defaults; only stored legacy values are converted.
    snprintf(d, sizeof d, "%g", *p.value * (old_lip ? calibration : 1.f));
    if (version < 4 && p.profile >= 0 && p.profile < 2) {
      Param legacy = p;
      legacy.profile = -1;
      GetPrivateProfileStringA(Section(legacy).c_str(), p.key, d, b, sizeof b,
                               kIni);
      snprintf(d, sizeof d, "%s", b);
    }
    GetPrivateProfileStringA(Section(p).c_str(), p.key, d, b, sizeof b, kIni);
    char* end;
    const float value = std::strtof(b, &end);
    if (end != b && *end == '\0' && std::isfinite(value))
      *p.value = std::clamp(old_lip ? std::clamp(value, 0.f, 3.f) / calibration
                                    : value,
                            p.lo, p.hi);
  }
  tune.SelectModel(tune.SelectedModel());  // derives v3_model
  config.network = tune.SelectedModel();
  config.gpu = !tune.force_cpu;
  config.model = ModelFor(config.network).model_file;
  config.direct = ModelFor(config.network).mapping_file;
  if (version < 3) tune.logging = true;
  // Version 7 lowered the emotion caps' default from 2 to 1: a stored cap at
  // the old default was that version's default, not a choice.
  if (version < 7)
    for (const Param& p : kParams)
      if (p.value && p.sec == Sec::kEmotion && !strncmp(p.key, "Max", 3) &&
          *p.value == 2)
        *p.value = 1;
  if (version < 6) {  // Argonians were saved as off while that was the default
    WritePrivateProfileStringA("Races", "Skyrim.esm|013740", nullptr, kIni);
    WritePrivateProfileStringA("Races", "Skyrim.esm|08883A", nullptr, kIni);
  }
  if (version < 8) {  // 4.3: one fixed GPU route; sentiment is shared
    WritePrivateProfileStringA("General", "GpuDevice", nullptr, kIni);
    WritePrivateProfileStringA("General", "AutoGpuDevice", nullptr, kIni);
    for (const char* section :
         {"Emotion", "EmotionMale", "EmotionFemale", "EmotionPlayer"})
      WritePrivateProfileStringA(section, "Sentiment", nullptr, kIni);
  }
  if (version < kIniVersion) SaveIni();
  if (migrate) WritePrivateProfileStringA("Direct", nullptr, nullptr, kIni);
  ApplyLogging();
}
void ScheduleSaveIni() {
  // Process-lifetime event: the writer remains asleep between edits.
  static const HANDLE event = [] {
    HANDLE e = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!e) return static_cast<HANDLE>(nullptr);
    HANDLE thread = CreateThread(nullptr, 0, IniWriter, e, 0, nullptr);
    if (!thread) {
      CloseHandle(e);
      return static_cast<HANDLE>(nullptr);
    }
    CloseHandle(thread);
    return e;
  }();
  if (!event || !SetEvent(event)) {
    skse::Log("settings writer unavailable; saving synchronously");
    SaveIni();
  }
}

void SaveIni() {
  // Serialize writers, but release settings_lock before any filesystem I/O.
  ScopedLock writer(ini_write_lock);
  std::vector<std::string> values;
  std::vector<RaceSetting> saved_races;
  std::vector<NpcSetting> saved_npcs;
  int saved_hotkey;
  {
    ScopedLock guard(settings_lock, ScopedLock::kShared);
    saved_races = races;
    for (const auto& item : npcs) saved_npcs.push_back(item.second);
    saved_hotkey = npc_hotkey;
    values.reserve(kNumParams);
    for (const Param& p : kParams) {
      char value[64];
      if (p.value) snprintf(value, sizeof value, "%.9g", *p.value);
      values.emplace_back(p.value ? value : *p.flag ? "1" : "0");
    }
  }
  char b[64];
  snprintf(b, sizeof b, "%d", kIniVersion);
  WritePrivateProfileStringA("General", "IniVersion", b, kIni);
  WritePrivateProfileStringA("General", "IndependentGpuDevice",
                             config.independent_device ? "1" : "0", kIni);
  snprintf(b, sizeof b, "%d", saved_hotkey);
  WritePrivateProfileStringA("General", "OverridesHotkey", b, kIni);
  WritePrivateProfileStringA("General", "BlacklistHotkey", nullptr, kIni);
  for (const NpcSetting& npc : saved_npcs) {
    const bool active = npc.blocked || !npc.values.empty();
    const std::string section = "Override:" + npc.key;
    if (!active) {
      WritePrivateProfileStringA("Overrides", npc.key.c_str(), nullptr, kIni);
      WritePrivateProfileStringA(section.c_str(), nullptr, nullptr, kIni);
    } else {
      // One write replaces the section, dropping keys no longer overridden.
      std::string body = std::string("Enabled=") + (npc.blocked ? "0" : "1");
      body += '\0';
      for (const auto& [key, value] : npc.values) {
        snprintf(b, sizeof b, "%.9g", value);
        body += key + "=" + b;
        body += '\0';
      }
      WritePrivateProfileSectionA(section.c_str(), body.c_str(), kIni);
      WritePrivateProfileStringA("Overrides", npc.key.c_str(), "1", kIni);
    }
    WritePrivateProfileStringA("Blacklist", npc.key.c_str(), nullptr, kIni);
  }
  for (size_t i = 0; i < kNumParams; ++i) {
    const Param& p = kParams[i];
    if (!p.stored) continue;
    WritePrivateProfileStringA(Section(p).c_str(), p.key, values[i].c_str(),
                               kIni);
  }
  // Only choices that differ from the default: a key per race in the load
  // order would swamp the ini.
  for (const RaceSetting& race : saved_races)
    WritePrivateProfileStringA(
        "Races", race.key.c_str(),
        race.enabled == race.default_enabled ? nullptr
        : race.enabled                       ? "1"
                                             : "0",
        kIni);
  // replaced by UseOldModel and ForceCPU
  WritePrivateProfileStringA("General", "Model", nullptr, kIni);
  WritePrivateProfileStringA("General", "Network", nullptr, kIni);
  WritePrivateProfileStringA("General", "GPU", nullptr, kIni);
}

}  // namespace bfa
