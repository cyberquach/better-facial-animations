// The ini and the table of settings declared in plugin.h.
#include <windows.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
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

#define SI_NO_CONVERSION
#include "a2f/inference/net_keeper.h"
#include "a2f/plugin/plugin.h"
#include "a2f/runtime/engine.h"
#include "a2f/runtime/scoped_lock.h"
#include "a2f/runtime/skse.h"
#include "third_party/simpleini/SimpleIni.h"

namespace bfa {

Config& config = *new Config;
namespace {
Tune tune;
// Sorted by runtime id; settings_lock guards it.
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
// Every Audio2Face model, once: the menu, the ini switches, the selection
// and the files all come from here.
constexpr ModelInfo kFaceModels[] = {
    {a2f::kNetPerformance, L"performance.onnx", L"A2FDirect3.bin", a2f::kNetV3,
     a2f::kNetVramMbPerformance, a2f::kNetVramMbPerformanceFused, "Performance",
     &Tune::performance_model, "UsePerformanceModel", "Performance model",
     "22-million-parameter student trained to reproduce V3's answers on "
     "the training dialogue, with memory within each line. Designed for "
     "familiar dialogue; unfamiliar or generated dialogue may be less "
     "consistent. It does not perfectly reproduce V3. Runs at 30 fps "
     "without speech-driven eye movement. GPU or CPU; about 185 MB of "
     "VRAM on GPU. Switches live; playing lines retain their animation."},
    {a2f::kNetUltraPerformance, L"ultra-performance.onnx", L"A2FDirect3.bin",
     a2f::kNetV3, a2f::kNetVramMbUltraPerformance,
     a2f::kNetVramMbUltraPerformance, "Ultra Performance",
     &Tune::ultra_performance_model, "UseUltraPerformanceModel",
     "Ultra Performance model",
     "The original 80,729-parameter Performance model, renamed. Lowest "
     "processing cost, with reduced detail and less consistent lip sync. "
     "Trained on V3's answers for familiar dialogue; unfamiliar dialogue "
     "may be less reliable. Runs at 30 fps without speech-driven eye "
     "movement. GPU or CPU; only a few MB of VRAM on GPU. Switches "
     "live, and playing lines retain their animation."},
    {a2f::kNetV2, L"network_fp16.onnx", L"A2FDirect.bin", a2f::kNetV2,
     a2f::kNetVramMb, a2f::kNetVramMb, "Legacy", &Tune::old_model,
     "UseOldModel", "Old model",
     "Audio2Face v2.3. Less expressive than V3 and the slowest to run: "
     "roughly half V3's speed in RTX 4070 GPU tests. Speed varies by "
     "system and dialogue. Lower VRAM use than V3, "
     "at about 100 MB of VRAM on GPU. Uses one shared facial "
     "identity at 30 fps. Switches live; already playing lines finish "
     "with their current animation."},
    {a2f::kNetV3, L"a2f3_fp16.onnx", L"A2FDirect3.bin", a2f::kNetV3,
     a2f::kNetVramMb3, a2f::kNetVramMb3Fused, "V3", nullptr, nullptr, nullptr,
     "Audio2Face v3.0. The most expressive model, with detailed lip syncing "
     "at 60 fps. Roughly 2x as fast as the Old "
     "model in RTX 4070 GPU tests; speed varies by system and dialogue. "
     "About 270 MB of VRAM on GPU. Switches live; already playing lines "
     "finish with their current animation."},
};
// Exactly one without a switch, the fallback, and it is last.
static_assert([] {
  for (const ModelInfo& info : kFaceModels)
    if (!info.selected != (&info == &std::rbegin(kFaceModels)[0])) return false;
  return true;
}());
}  // namespace
SRWLOCK settings_lock = SRWLOCK_INIT;

extern const std::vector<Param> kParams = [] {
  const auto Own = [](Param p) {
    p.own_control = true;
    return p;
  };
  std::vector<Param> params = {
      Own({"Enabled", nullptr, 0, 0, "Enable facial animations",
           "Untick for the game's own lip-sync, instantly (A/B).",
           Sec::kGeneral, &tune.enabled}),
      {"PlayerFacialAnimations", nullptr, 0, 0, "Player facial animations",
       "Automatically detect DBReV or DBVO2 for player dialogue. "
       "DBVO2 prototype for Skyrim. Untick to keep the voice mod's own "
       "animation.",
       Sec::kGeneral, &tune.player},
      {"PotatoMode", nullptr, 0, 0, "Potato Mode",
       "Only animate the player and NPC in an active dialogue window with one "
       "another.",
       Sec::kGeneral, &tune.potato_mode},
      Own({"VanillaEmotions", nullptr, 0, 0, "Vanilla / scene emotion inputs",
           "Pass authored scene and dialogue-mod emotions to the facial model. "
           "On by default. Sentiment Analysis works independently. "
           "Applies during BFA speech.",
           Sec::kGeneral, &tune.vanilla_emotions, kProfileShared, true,
           Param::kNone, Group::kSources}),
      Own({"SentimentAnalysis", nullptr, 0, 0, "Sentiment Analysis",
           "Enable the selected Audio2Emotion model to add voice-driven "
           "emotion "
           "alongside authored scene emotion. On GPU, Full uses about 745 MB "
           "of VRAM and Performance about 31 MB. "
           "Performance is enabled by default. Disabling unloads it after "
           "current "
           "inference. "
           "Force CPU uses system memory and additional CPU processing.",
           Sec::kGeneral, &tune.sentiment_analysis, kProfileShared, true,
           Param::kSentiment, Group::kSources}),
      Own({"UsePerformanceEmotion", nullptr, 0, 0,
           "Skyrim-trained emotion model",
           "Experimental distilled Audio2Emotion model for Skyrim facial "
           "animation.",
           Sec::kModel, &tune.performance_sentiment}),
      Own({"SentimentMultiplier", &tune.sentiment_multiplier, 0, 2,
           "Sentiment addition multiplier",
           "Scales sentiment added to the model inputs: 0 = none, 0.5 = half, "
           "1 = normal, 2 = double. Requires Sentiment Analysis. Final emotion "
           "limits apply. 0 skips sentiment inference. Applies next line.",
           Sec::kGeneral, nullptr, kProfileShared, true, Param::kSentiment,
           Group::kSources}),
      {"Prefetch", nullptr, 0, 0, "Prepare dialogue in advance",
       "Generate a line when its dialogue response is created, before it is "
       "spoken.",
       Sec::kGeneral, &tune.prefetch},
  };
  for (const ModelInfo& info : kFaceModels)
    if (info.selected)
      params.push_back(Own({info.ini_key, nullptr, 0, 0, info.label, info.tip,
                            Sec::kModel, &(tune.*info.selected)}));
  params.insert(
      params.end(),
      {
          Own({"ForceCPU", nullptr, 0, 0, "Force CPU",
               "Run the selected model on CPU. V3 and Old "
               "run much slower on CPU; long lines can start late.\n"
               "Applies live after the current inference call finishes.",
               Sec::kModel, &tune.force_cpu}),
          // Drawn under each model's processing device (menu.cc).
          Own({"GraphFusion", nullptr, 0, 0,
               "Use graph fusion for this model",
               "Audio2Face on GPU: DirectML merges the model's steps into "
               "fewer, larger GPU operations, as every version before 7.0 "
               "did. Slightly faster, but keeps extra copies of the model "
               "in VRAM.\n"
               "Measured on an RTX 5070 Ti: V3 about 5% faster for about "
               "140 MB more VRAM (270 -> 410 MB); Performance about 13% "
               "faster for about 130 MB more (185 -> 315 MB); Ultra "
               "Performance about 30% faster for about 1 MB more; Legacy "
               "unchanged. Every model already generates several times "
               "faster than speech, so the speed difference is rarely "
               "noticeable. Off by default; tick it for the speed and VRAM "
               "use of earlier versions.\n"
               "No effect on CPU. Applies live after the current inference "
               "call finishes.",
               Sec::kModel, &tune.face_graph_fusion}),
          Own({"EmotionGraphFusion", nullptr, 0, 0,
               "Use graph fusion for this model",
               "Audio2Emotion on GPU: DirectML merges the model's steps "
               "into fewer, larger GPU operations, as every version before "
               "7.0 did. Slightly faster, but keeps extra copies of the "
               "model in VRAM.\n"
               "Measured on an RTX 5070 Ti: Full about 1% faster for about "
               "600 MB more VRAM (745 -> 1,340 MB); Performance about 8% "
               "faster for about 20 MB more (31 -> 52 MB). Results are "
               "identical either way. Off by default; tick it for the speed "
               "and VRAM use of earlier versions.\n"
               "No effect on CPU. Applies live after the current inference "
               "call finishes, without reloading Audio2Face.",
               Sec::kModel, &tune.emotion_graph_fusion}),
          {"DisableTongue", nullptr, 0, 0, "Disable tongue motion",
           "Restore the tongue behavior from before this feature. Applies "
           "immediately.",
           Sec::kDirect, &tune.disable_tongue},
          {"ConversationGaze", nullptr, 0, 0, "Enable improved eye gaze",
           "The master switch for everything under Gaze. Unticked, the game's "
           "own steady gaze and head tracking are back exactly as without it. "
           "Ticked: the NPC you are talking to looks at you with speech-driven "
           "eye motion, and may look away in the pauses of a line, for the "
           "silence alone or in thought. How often is set per male and female "
           "profile below, and per NPC under Overrides. Applies immediately; "
           "SE/AE only.",
           Sec::kDirect, &tune.conversation_gaze, kProfileShared, true,
           Param::kNone, Group::kGaze},
          {"GazeStrength", &tune.gaze_strength, 0, 2, "Eye motion strength",
           "Strength of the small speech-driven eye motion while they look at "
           "you; 0 for none. Looking away is set separately below.",
           Sec::kDirect, nullptr, kProfileShared, true, Param::kNone,
           Group::kGaze},
          // With ConversationGaze: 0 keeps the eye motion alone, as before 6.1.
          {"GazeBehavior", nullptr, 0, 0, "Look away at pauses",
           "Unticked, they only ever look at you.", Sec::kGaze,
           &tune.gaze_behavior, kProfileShared, true, Param::kNone,
           Group::kAuto, false},
          {"GazeTurnHead", nullptr, 0, 0, "Turn head toward objects",
           "Looking at an object well to the side turns the head, through the "
           "game's own head tracking. Untick to move the eyes only.",
           Sec::kGaze, &tune.gaze_turn_head},
          {"GazeLineOfSight", nullptr, 0, 0, "Only look at what they can see",
           "Objects and people behind a wall or other obstacle are not looked "
           "at.",
           Sec::kGaze, &tune.gaze_line_of_sight, kProfileShared, true,
           Param::kNone, Group::kAuto, false},
          {"GazeAvoidCamera", nullptr, 0, 0, "Never look into the camera",
           "Outside first person, looks keep clear of the camera.", Sec::kGaze,
           &tune.gaze_avoid_camera, kProfileShared, true, Param::kNone,
           Group::kAuto, false},
      });
  // Per male/female profile and per NPC; the player is never the partner.
  static const struct {
    const char* key;
    const char* label;
    const char* tip;
    bool menu;
  } kGazeOdds[] = {
      {"GazeNeutral", "Neutral: holds your eye",
       "Chance that a pause in a neutral line leaves them looking at you: "
       "0 = never, 1 = always. Otherwise they look at something past you, "
       "aside or nearby. A line's start looks away half as often.",
       true},
      {"GazeAngry", "Angry: holds your eye",
       "Chance per pause while angry. High: anger stares you down. "
       "Otherwise, in frustration, a short look low and just past you.",
       true},
      {"GazeHappy", "Happy: holds your eye", "Chance per pause while happy.",
       true},
      {"GazeSad", "Sad or embarrassed: holds your eye",
       "Chance per pause while sad, ashamed or in pain. Otherwise they "
       "mostly look down, and hold it into the next words.",
       true},
      {"GazeAfraid", "Afraid: holds your eye", "Chance per pause while afraid.",
       false},
      {"GazeSurprised", "Surprised: holds your eye",
       "Chance per pause while surprised.", false},
      {"GazeDisgusted", "Disgusted: holds your eye",
       "Chance per pause while disgusted.", false},
  };
  static_assert(std::size(kGazeOdds) == a2f::kNumGazeMoods);
  for (int profile : {kProfileMale, kProfileFemale}) {
    FaceTune& face = tune.ForProfile(profile);
    auto gaze_param = [&](Param p, float* (*access)(FaceTune&, size_t),
                          size_t index, bool menu) {
      p.face_value = access;
      p.face_index = index;
      p.value = p.FaceValue(face);
      p.profile = profile;
      p.menu = menu;
      params.push_back(p);
    };
    gaze_param(
        {"GazeHold", nullptr, 0, 2, "Overall: holds your eye",
         "How firmly they keep looking at you, over every emotion "
         "below and the idle glances past your head. 1 = as set "
         "below, 2 = never looks away, 0 = looks away twice as often.",
         Sec::kGaze, nullptr},
        [](FaceTune& f, size_t) { return &f.gaze_hold; }, 0, true);
    gaze_param(
        {"GazeRestSeconds", nullptr, 0, 15, "Seconds between looks away",
         "After their eyes come back to you they stay at least this "
         "long, whatever the line does.",
         Sec::kGaze, nullptr},
        [](FaceTune& f, size_t) { return &f.gaze_rest; }, 0, true);
    gaze_param(
        {"GazeSettleSeconds", nullptr, 0, 30,
         "Seconds before the first look away",
         "A conversation opens with their eyes on you: no look away "
         "until it has run this long.",
         Sec::kGaze, nullptr},
        [](FaceTune& f, size_t) { return &f.gaze_settle; }, 0, true);
    for (int mood = 0; mood < a2f::kNumGazeMoods; ++mood)
      gaze_param(
          {kGazeOdds[mood].key, nullptr, 0, 1, kGazeOdds[mood].label,
           kGazeOdds[mood].tip, Sec::kGaze, nullptr},
          [](FaceTune& f, size_t i) { return &f.gaze_odds[i]; }, mood,
          kGazeOdds[mood].menu);
  }
  static char mouth_tip[160];
  snprintf(mouth_tip, sizeof mouth_tip,
           "Lip and mouth skin movement. Default strength: 1.0; %g with "
           "this sex's Expressive Facial Animation edition detected.",
           kExpressiveMouthScale);
  for (int profile = 0; profile < kNumProfiles; ++profile) {
    FaceTune& face = tune.ForProfile(profile);
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
                         "Crossfade into a line ready after it started, or "
                         "begun while the face still shows a pose. From "
                         "rest a line is shown at once.",
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
                  Sec::kEmotion, nullptr, kProfileShared, true, Param::kScene},
                 [](FaceTune& f, size_t) { return &f.voice_emotion; }),
             face_param(
                 {"VoiceSwell", nullptr, 0, 2,
                  "Scene emotion boost from voice volume",
                  "Boost authored scene emotion where the voice is "
                  "loud. 0 = steady. Applies from the next line.",
                  Sec::kEmotion, nullptr, kProfileShared, true, Param::kScene},
                 [](FaceTune& f, size_t) { return &f.voice_swell; }),
             face_param(
                 {"FaceFadeIn", nullptr, 0, 3,
                  "Fade into generated expressions (s)",
                  "When a line starts: how long the game's own expressions "
                  "take to fade out, leaving the face to Audio2Face.",
                  Sec::kEmotion, nullptr, kProfileShared, true, Param::kNone,
                  Group::kTransitions},
                 [](FaceTune& f, size_t) { return &f.face_in; }),
             face_param(
                 {"FaceFadeOut", nullptr, 0, 3,
                  "Restore game expressions after speech (s)",
                  "After a line: how long the game's own expressions take to "
                  "fade back in.",
                  Sec::kEmotion, nullptr, kProfileShared, true, Param::kNone,
                  Group::kTransitions},
                 [](FaceTune& f, size_t) { return &f.face_out; }),
         }) {
      params.push_back(p);
    }
    static const char* const kKeys[] = {
        "MaxAmazement", "MaxAnger",  "MaxCheekiness", "MaxDisgust",
        "MaxFear",      "MaxGrief",  "MaxJoy",        "MaxOutOfBreath",
        "MaxPain",      "MaxSadness"};
    static_assert(std::size(kKeys) == a2f::kNumMod);
    static const char* const kLabels[] = {
        "Maximum surprise input",   "Maximum anger input",
        "Maximum cheekiness input", "Maximum disgust input",
        "Maximum fear input",       "Maximum grief input",
        "Maximum happiness input",  "Maximum breathlessness input",
        "Maximum pain input",       "Maximum sadness input"};
    static_assert(std::size(kLabels) == a2f::kNumMod);
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
  return params;
}();

std::span<const ModelInfo> FaceModels() { return kFaceModels; }
const ModelInfo& FallbackFaceModel() { return std::rbegin(kFaceModels)[0]; }
const ModelInfo& ModelFor(int network) {
  for (const ModelInfo& info : kFaceModels)
    if (info.network == network) return info;
  return FallbackFaceModel();
}
int Tune::SelectedModel() const {
  for (const ModelInfo& info : kFaceModels)
    if (info.selected && this->*info.selected) return info.network;
  return FallbackFaceModel().network;
}
void Tune::SelectModel(int network) {
  for (const ModelInfo& info : kFaceModels)
    if (info.selected) this->*info.selected = info.network == network;
}
extern const size_t kNumParams = std::size(kParams);

namespace {

SRWLOCK ini_write_lock = SRWLOCK_INIT;

constexpr char kIni[] = "Data\\SKSE\\Plugins\\BetterFacialAnimations.ini";

bool ReadIniFile(CSimpleIniA& ini, const char* path, bool* missing = nullptr) {
  errno = 0;
  const SI_Error result = ini.LoadFile(path);
  const bool absent = result == SI_FILE && errno == ENOENT;
  if (missing) *missing = absent;
  if (result >= SI_OK || absent) return true;
  skse::Report(
      "settings load failed: %s (SimpleIni %d, errno %d); "
      "keeping current settings",
      path, result, errno);
  return false;
}

bool ReadIni(CSimpleIniA& ini, bool* missing = nullptr) {
  ini.SetUnicode();
  // Fixed, independent installer choices; no directory scans or frame work.
  // Explicit user settings load last and always win, including after menu
  // saves.
  constexpr const char* kChoices[] = {
      "GraphicsRoute",   "Startup",      "FaceModel",       "EmotionModel",
      "VanillaEmotions", "ActiveOnly",   "PlayerAnimation", "PrepareEarly",
      "Tongue",          "Gaze",         "Logging",         "Khajiit",
      "Argonian",        "EmotionDevice", "FaceGraphFusion",
      "EmotionGraphFusion"};
  for (const char* choice : kChoices) {
    const std::string path =
        std::string(
            "Data\\SKSE\\Plugins\\BetterFacialAnimations\\Installer\\") +
        choice + ".ini";
    if (!ReadIniFile(ini, path.c_str())) return false;
  }
  return ReadIniFile(ini, kIni, missing);
}

// A form's identity across load orders: "<plugin>|<local id>" as the ini
// stores it; empty without a source plugin.
std::string FormKey(uint8_t* form) {
  const char* file = SourceFileName(form);
  if (!file || !*file) return {};
  char suffix[16];
  snprintf(suffix, sizeof suffix, "|%06X",
           LocalFormId(reinterpret_cast<RE::TESForm*>(form)->formID));
  return file + std::string(suffix);
}

// NPC keys lower the plugin name so that its spelling cannot split entries;
// the local id stays upper-case hex.
std::string NpcKey(uint8_t* base) {
  if (!base ||
      reinterpret_cast<RE::TESForm*>(base)->formType.underlying() != kFtNpc)
    return {};
  const uint32_t id = reinterpret_cast<RE::TESForm*>(base)->formID;
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
  const char* name = reinterpret_cast<RE::TESNPC*>(base)->fullName.c_str();
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
// Version 8 retires GpuDevice/AutoGpuDevice and the Sentiment sliders.
// Version 9 makes the independent GPU route opt-in: a stored
// IndependentGpuDevice=1 in the user's INI was 4.0's default, not a choice,
// so it is cleared unless this install's FOMOD fragment chose it.
// Version 10 preserves the former Performance choice as Ultra Performance.
constexpr int kIniVersion = 10;

std::string Section(const Param& p) {
  static const char* const kNames[] = {
      "General", "General", "Playback", "LocalAnimationStrength",
      "Emotion", "Debug",   "Gaze"};
  static_assert(std::size(kNames) == static_cast<size_t>(Sec::kGaze) + 1);
  std::string section = kNames[static_cast<int>(p.sec)];
  if (p.profile != kProfileShared) section += kProfiles[p.profile].section;
  return section;
}

}  // namespace

void LoadRaceSettings() {
  ScopedLock reader(ini_write_lock);
  CSimpleIniA ini;
  if (!ReadIni(ini)) return;
  std::vector<RaceSetting> loaded;
  const auto* forms = FormArray(kFtRace);
  if (!forms) return;
  for (uint32_t i = 0; i < forms->size(); ++i) {
    uint8_t* form = reinterpret_cast<uint8_t*>(forms->data()[i]);
    if (!form ||
        reinterpret_cast<RE::TESForm*>(form)->formType.underlying() !=
            kFtRace ||
        !(reinterpret_cast<RE::TESRace*>(form)->data.flags.underlying() & 2))
      continue;
    const uint32_t id = reinterpret_cast<RE::TESForm*>(form)->formID;
    const char* file = SourceFileName(form);
    const std::string key = FormKey(form);
    if (key.empty()) continue;
    const char* name = reinterpret_cast<RE::TESRace*>(form)->fullName.c_str();
    const char* editor =
        reinterpret_cast<RE::TESRace*>(form)->formEditorID.c_str();
    std::string label = name && *name ? name : "Unnamed race";
    if (editor && *editor) label += " (" + std::string(editor) + ")";
    const bool default_enabled = DefaultRaceEnabled(id, file);
    const bool enabled =
        ini.GetLongValue("Races", key.c_str(), default_enabled) != 0;
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
  ScopedLock reader(ini_write_lock);
  CSimpleIniA ini;
  if (!ReadIni(ini)) return;
  // Legacy exclusions migrate to disabled overrides. New entries take
  // precedence.
  std::map<std::string, NpcSetting> loaded;
  for (const char* section : {"Blacklist", "Overrides"}) {
    CSimpleIniA::TNamesDepend keys;
    ini.GetAllKeys(section, keys);
    for (const auto& entry : keys) {
      if (ini.GetLongValue(section, entry.pItem, 0) != 1) continue;
      std::string key = entry.pItem;
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
        npc.blocked = !ini.GetLongValue(detail.c_str(), "Enabled", 1);
        for (const Param& param : kParams) {
          if (!NpcParamSupported(param)) continue;
          const char* value = ini.GetValue(detail.c_str(), param.key, "");
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
  if (const auto* forms = FormArray(kFtNpc)) {
    for (uint32_t i = 0; i < forms->size(); ++i) {
      uint8_t* form = reinterpret_cast<uint8_t*>(forms->data()[i]);
      const auto it = loaded.find(NpcKey(form));
      if (it == loaded.end()) continue;
      it->second.id = reinterpret_cast<RE::TESForm*>(form)->formID;
      it->second.label = NpcLabel(form);
      it->second.female = (reinterpret_cast<RE::TESNPC*>(form)
                               ->actorData.actorBaseFlags.underlying() &
                           1) != 0;
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
  if (!actor ||
      reinterpret_cast<RE::TESForm*>(actor)->formType.underlying() !=
          kFtActor ||
      reinterpret_cast<RE::TESForm*>(actor)->formID == kPlayerFormId)
    return {};
  auto* base = reinterpret_cast<uint8_t*>(
      reinterpret_cast<RE::TESObjectREFR*>(actor)->data.objectReference);
  const std::string key = NpcKey(base);
  if (key.empty()) return {};
  NpcSetting npc{reinterpret_cast<RE::TESForm*>(base)->formID, key,
                 NpcLabel(base), false};
  ScopedLock guard(settings_lock, ScopedLock::kShared);
  npc.female = (reinterpret_cast<RE::TESNPC*>(base)
                    ->actorData.actorBaseFlags.underlying() &
                1) != 0;
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
      reinterpret_cast<RE::TESForm*>(actor)->formID == kPlayerFormId)
    return true;
  auto* base = reinterpret_cast<uint8_t*>(
      reinterpret_cast<RE::TESObjectREFR*>(actor)->data.objectReference);
  if (!base ||
      reinterpret_cast<RE::TESForm*>(base)->formType.underlying() != kFtNpc)
    return true;
  ScopedLock guard(settings_lock, ScopedLock::kShared);
  return !blocked_npcs.count(reinterpret_cast<RE::TESForm*>(base)->formID);
}

bool NpcParamSupported(const Param& param) {
  return (param.profile == kProfileMale && param.face_value) ||
         (param.profile == kProfileShared &&
          (param.sec == Sec::kDirect ||
           !strcmp(param.key, "SentimentMultiplier") ||
           !strcmp(param.key, "SentimentAnalysis")));
}

Tune NpcSharedTune(Tune result, uint8_t* actor) {
  if (!npc_overrides || !actor ||
      reinterpret_cast<RE::TESForm*>(actor)->formID == kPlayerFormId)
    return result;
  auto* base = reinterpret_cast<uint8_t*>(
      reinterpret_cast<RE::TESObjectREFR*>(actor)->data.objectReference);
  if (!base ||
      reinterpret_cast<RE::TESForm*>(base)->formType.underlying() != kFtNpc)
    return result;
  ScopedLock guard(settings_lock, ScopedLock::kShared);
  const auto id = npc_ids.find(reinterpret_cast<RE::TESForm*>(base)->formID);
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
      result.sentiment_analysis &&
      get("SentimentAnalysis", result.sentiment_analysis) != 0;
  result.sentiment_multiplier =
      get("SentimentMultiplier", result.sentiment_multiplier);
  return result;
}

FaceTune NpcFaceTune(const Tune& defaults, uint8_t* actor, bool female,
                     bool player) {
  FaceTune face = defaults.ForActor(female, player);
  if (!npc_overrides || !actor || player) return face;
  auto* base = reinterpret_cast<uint8_t*>(
      reinterpret_cast<RE::TESObjectREFR*>(actor)->data.objectReference);
  if (!base ||
      reinterpret_cast<RE::TESForm*>(base)->formType.underlying() != kFtNpc)
    return face;
  ScopedLock guard(settings_lock, ScopedLock::kShared);
  const auto id = npc_ids.find(reinterpret_cast<RE::TESForm*>(base)->formID);
  if (id == npc_ids.end()) return face;
  const auto& values = npcs.at(id->second).values;
  for (const Param& param : kParams) {
    if (param.profile != kProfileMale) continue;
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

void SetSelectedModel(int network) {
  ScopedLock guard(settings_lock);
  tune.SelectModel(network);
}

void SelectFaceOption(int network) {
  ScopedLock guard(settings_lock);
  tune.enabled = network != kFaceDisabled;
  if (tune.enabled) tune.SelectModel(network);
}

void SelectEmotionOption(int model) {
  ScopedLock guard(settings_lock);
  tune.sentiment_analysis = model != kEmotionDisabled;
  if (tune.sentiment_analysis)
    tune.performance_sentiment = model == kEmotionPerformance;
}

bool SentimentAvailable(const Tune& selected) {
  return selected.performance_sentiment ? config.performance_sentiment_available
                                        : config.sentiment_available;
}

void SetProcessingDevice(bool cpu, int adapter) {
  ScopedLock guard(settings_lock);
  tune.force_cpu = cpu;
  tune.gpu_adapter = std::max(0, adapter);
}

void SetEmotionDevice(int device) {
  ScopedLock guard(settings_lock);
  tune.emotion_device = std::max(kCpuDevice, device);
}

void ResetTune() {
  ScopedLock guard(settings_lock);
  // a diagnostic switch, the models and their devices, not tuning values
  const Tune old = tune;
  tune = Tune();
  tune.logging = old.logging;
  tune.SelectModel(old.SelectedModel());
  tune.enabled = old.enabled;
  tune.sentiment_analysis = old.sentiment_analysis;
  tune.performance_sentiment = old.performance_sentiment;
  tune.force_cpu = old.force_cpu;
  tune.gpu_adapter = old.gpu_adapter;
  tune.emotion_device = old.emotion_device;
  tune.face_graph_fusion = old.face_graph_fusion;
  tune.emotion_graph_fusion = old.emotion_graph_fusion;
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
  CSimpleIniA ini;
  bool missing;
  if (!ReadIni(ini, &missing)) return;
  const DWORD sentiment_file =
      GetFileAttributesW((PluginDir() + L"\\" + kSentimentModel).c_str());
  config.sentiment_available = sentiment_file != INVALID_FILE_ATTRIBUTES &&
                               !(sentiment_file & FILE_ATTRIBUTE_DIRECTORY);
  const DWORD performance_file = GetFileAttributesW(
      (PluginDir() + L"\\" + kPerformanceSentimentModel).c_str());
  config.performance_sentiment_available =
      performance_file != INVALID_FILE_ATTRIBUTES &&
      !(performance_file & FILE_ATTRIBUTE_DIRECTORY);
  // Startup only: no readers or menu callbacks exist yet.
  tune = Tune();
  SetNpcHotkey(
      ini.GetLongValue("General", "OverridesHotkey",
                       ini.GetLongValue("General", "BlacklistHotkey", 0)));
  const int version =
      missing ? kIniVersion : ini.GetLongValue("General", "IniVersion", 0);
  const bool migrate = version < 2;  // the Direct section is dropped
  config.batch =
      std::max(1, static_cast<int>(ini.GetLongValue("General", "Batch", 2)));
  config.cache_lines = std::max(
      8, static_cast<int>(ini.GetLongValue("General", "CacheLines", 300)));
  config.log_lines = ini.GetLongValue("Debug", "LogLines", 400);
  // ForceCPU was GPU=0 in earlier 3.0 builds.
  tune.force_cpu = ini.GetLongValue("General", "GPU", 1) == 0;
  tune.gpu_adapter = std::max(0L, ini.GetLongValue("General", "GpuAdapter", 0));
  // Explicit route, as in 4.0. No Auto probe or GPU-to-GPU fallback. This
  // key takes precedence over retired GpuDevice/AutoGpuDevice settings.
  config.independent_device =
      ini.GetLongValue("General", "IndependentGpuDevice", 0) != 0;
  config.force_bundled_agility =
      ini.GetLongValue("General", "ForceBundledAgility", 0) != 0;
  if (version < 9 && config.independent_device) {
    CSimpleIniA user;
    user.SetUnicode();
    if (ReadIniFile(user, kIni) &&
        user.GetValue("General", "IndependentGpuDevice")) {
      config.independent_device = false;
      skse::Report(
          "IndependentGpuDevice=1 was the 4.0 default: cleared; Independent "
          "GPU mode is opt-in since 5.0 (see GPU-SETUP.txt)");
    }
  }
  char d[128];
  for (size_t i = 0; i < kNumParams; ++i) {
    const Param& p = kParams[i];
    if (!p.stored || (migrate && p.sec == Sec::kDirect)) continue;
    if (p.flag) {
      *p.flag = ini.GetLongValue(Section(p).c_str(), p.key, *p.flag) != 0;
      continue;
    }
    const float calibration =
        !strcmp(p.key, "UpperLip")   ? direct::kUpperLipCalibration
        : !strcmp(p.key, "LowerLip") ? direct::kLowerLipCalibration
                                     : 1.f;
    const bool sex_profile =
        p.profile == kProfileMale || p.profile == kProfileFemale;
    const bool old_lip = version < 5 && sex_profile && calibration != 1.f;
    // Missing keys use new defaults; only stored legacy values are converted.
    snprintf(d, sizeof d, "%g", *p.value * (old_lip ? calibration : 1.f));
    if (version < 4 && sex_profile) {
      Param legacy = p;
      legacy.profile = kProfileShared;
      const std::string legacy_value =
          ini.GetValue(Section(legacy).c_str(), p.key, d);
      snprintf(d, sizeof d, "%s", legacy_value.c_str());
    }
    const char* b = ini.GetValue(Section(p).c_str(), p.key, d);
    char* end;
    const float value = std::strtof(b, &end);
    if (end != b && *end == '\0' && std::isfinite(value))
      *p.value = std::clamp(
          old_lip ? std::clamp(value, 0.f, 3.f) / calibration : value, p.lo,
          p.hi);
  }
  // The model is chosen whole by the last INI that names one, the user's over
  // the installer's: a switch it leaves out is off, so that neither the other
  // INI nor the default outranks its choice. Naming none keeps the default.
  const auto chooses = [](const CSimpleIniA& file) {
    if (file.GetValue("General", "Network")) return true;
    for (const ModelInfo& info : kFaceModels)
      if (info.ini_key && file.GetValue("General", info.ini_key)) return true;
    return false;
  };
  CSimpleIniA user;
  user.SetUnicode();
  const CSimpleIniA* chosen = nullptr;
  if (ReadIniFile(user, kIni) && chooses(user))
    chosen = &user;
  else if (chooses(ini))
    chosen = &ini;
  if (chosen) {
    // UseOldModel was Network=2 (3: V3) in earlier 3.0 builds.
    const bool network2 = chosen->GetLongValue("General", "Network", 3) == 2;
    for (const ModelInfo& info : kFaceModels)
      if (info.selected)
        tune.*info.selected =
            chosen->GetLongValue(
                "General", info.ini_key,
                info.selected == &Tune::old_model && network2) != 0;
    // Before version 10, Performance was the model now Ultra Performance.
    if (version < 10 &&
        !chosen->GetValue("General", "UseUltraPerformanceModel") &&
        tune.performance_model) {
      tune.performance_model = false;
      tune.ultra_performance_model = true;
    }
  }
  tune.SelectModel(tune.SelectedModel());
  // Preserve the former shared choice when upgrading an existing INI. Once
  // saved, each model has its own device; neither follows later changes.
  tune.emotion_device = std::max<long>(
      kCpuDevice,
      ini.GetLongValue("General", "EmotionDevice",
                       tune.force_cpu ? kCpuDevice : tune.gpu_adapter));
  config.network = tune.SelectedModel();
  config.gpu = !tune.force_cpu;
  config.gpu_adapter = tune.gpu_adapter;
  config.graph_fusion = tune.face_graph_fusion;
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
  if (version < kIniVersion) SaveIni();
  ApplyLogging();
}
namespace {
// Writes once each burst of menu edits has been quiet for a quarter second:
// a slider drag edits every frame, and SaveIni re-reads and rewrites the file.
DWORD WINAPI IniWriter(void* event) {
  for (;;) {
    if (WaitForSingleObject(event, INFINITE) != WAIT_OBJECT_0) return 1;
    DWORD result;
    do {
      result = WaitForSingleObject(event, 250);
    } while (result == WAIT_OBJECT_0);
    if (result != WAIT_TIMEOUT) return 1;
    SaveIni();
  }
}
}  // namespace

void ScheduleSaveIni() {
#ifdef ENABLE_COMMONLIBSSE_TESTING
  if (testing::save_ini) {
    SaveIni();
    return;
  }
#endif
  // Process-lifetime event: the writer sleeps between edits.
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
#ifdef ENABLE_COMMONLIBSSE_TESTING
  if (testing::save_ini) {
    testing::save_ini();
    return;
  }
#endif
  // Serialize writers, but release settings_lock before any filesystem I/O.
  ScopedLock writer(ini_write_lock);
  CSimpleIniA ini;
  bool missing;
  // Never replace an existing file we cannot read, or migrate installer-only
  // defaults as if they were a legacy user INI.
  if (!ReadIni(ini, &missing)) return;
  std::vector<std::string> values;
  std::vector<RaceSetting> saved_races;
  std::vector<NpcSetting> saved_npcs;
  int saved_hotkey, saved_gpu_adapter, saved_emotion_device;
  {
    ScopedLock guard(settings_lock, ScopedLock::kShared);
    saved_races = races;
    for (const auto& item : npcs) saved_npcs.push_back(item.second);
    saved_hotkey = npc_hotkey;
    saved_gpu_adapter = tune.gpu_adapter;
    saved_emotion_device = tune.emotion_device;
    values.reserve(kNumParams);
    for (const Param& p : kParams) {
      char value[64];
      if (p.value) snprintf(value, sizeof value, "%.9g", *p.value);
      values.emplace_back(p.value ? value : *p.flag ? "1" : "0");
    }
  }
  const int version =
      missing ? kIniVersion : ini.GetLongValue("General", "IniVersion", 0);
  if (version < 2) ini.Delete("Direct", nullptr);
  if (version < 6) {
    ini.Delete("Races", "Skyrim.esm|013740");
    ini.Delete("Races", "Skyrim.esm|08883A");
  }
  if (version < 8) {
    ini.Delete("General", "GpuDevice");
    ini.Delete("General", "AutoGpuDevice");
    for (const char* section :
         {"Emotion", "EmotionMale", "EmotionFemale", "EmotionPlayer"})
      ini.Delete(section, "Sentiment");
  }
  auto write = [&](const char* section, const char* key, const char* value) {
    if (value)
      ini.SetValue(section, key, value);
    else
      ini.Delete(section, key);
  };
  char b[64];
  snprintf(b, sizeof b, "%d", kIniVersion);
  write("General", "IniVersion", b);
  write("General", "IndependentGpuDevice",
        config.independent_device ? "1" : "0");
  write("General", "ForceBundledAgility",
        config.force_bundled_agility ? "1" : "0");
  snprintf(b, sizeof b, "%d", saved_hotkey);
  write("General", "OverridesHotkey", b);
  snprintf(b, sizeof b, "%d", saved_gpu_adapter);
  write("General", "GpuAdapter", b);
  snprintf(b, sizeof b, "%d", saved_emotion_device);
  write("General", "EmotionDevice", b);
  write("General", "BlacklistHotkey", nullptr);
  for (const NpcSetting& npc : saved_npcs) {
    const bool active = npc.blocked || !npc.values.empty();
    const std::string section = "Override:" + npc.key;
    if (!active) {
      write("Overrides", npc.key.c_str(), nullptr);
      write(section.c_str(), nullptr, nullptr);
    } else {
      // Replace the section, dropping keys no longer overridden.
      ini.Delete(section.c_str(), nullptr);
      write(section.c_str(), "Enabled", npc.blocked ? "0" : "1");
      for (const auto& [key, value] : npc.values) {
        snprintf(b, sizeof b, "%.9g", value);
        write(section.c_str(), key.c_str(), b);
      }
      write("Overrides", npc.key.c_str(), "1");
    }
    write("Blacklist", npc.key.c_str(), nullptr);
  }
  for (size_t i = 0; i < kNumParams; ++i) {
    const Param& p = kParams[i];
    if (!p.stored) continue;
    write(Section(p).c_str(), p.key, values[i].c_str());
  }
  // Preserve explicit race choices, including overrides of installer defaults.
  // Untouched races at the built-in default need no entry.
  for (const RaceSetting& race : saved_races)
    write("Races", race.key.c_str(),
          race.enabled == race.default_enabled &&
                  !ini.GetValue("Races", race.key.c_str())
              ? nullptr
          : race.enabled ? "1"
                         : "0");
  // replaced by UseOldModel and ForceCPU
  write("General", "Model", nullptr);
  write("General", "Network", nullptr);
  write("General", "GPU", nullptr);
  errno = 0;
  const SI_Error result = ini.SaveFile(kIni);
  if (result < SI_OK)
    skse::Report(
        "settings save failed: %s (SimpleIni %d, errno %d); "
        "changes may not survive restart",
        kIni, result, errno);
}

}  // namespace bfa
