// What the plugin's files share: the settings (settings.cc) and the state
// of better_facial_animations.cc that the head cache and the settings page
// use.
#ifndef INCLUDE_A2F_PLUGIN_PLUGIN_H_
#define INCLUDE_A2F_PLUGIN_PLUGIN_H_

#include <windows.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "a2f/animation/direct.h"
#include "a2f/animation/gaze_director.h"
#include "a2f/inference/gpu_state.h"

namespace bfa {

extern const char kVersion[];  // from the VERSION file

enum class State { kPending, kReady, kFailed };

// ---------------- settings ----------------
// Audio2Face menu choice: kFaceDisabled or an a2f::kNet* network.
constexpr int kFaceDisabled = 0;
// Audio2Emotion menu choice.
enum EmotionChoice : int {
  kEmotionDisabled = 0,
  kEmotionFull = 1,
  kEmotionPerformance = 2,  // experimental student
};

struct Config {  // fixed after load
  bool performance_sentiment_available = false;
  bool sentiment_available = false;  // optional model found at settings load
  bool gpu = true;
  int gpu_adapter = 0;              // startup selection
  bool graph_fusion = false;        // startup selection, Audio2Face
  bool independent_device = false;  // fixed GPU route; change INI and restart
  bool force_bundled_agility = false;  // INI-only, independent mode, restart required
  int network = a2f::kNetV3;        // the model at startup, see ModelFor
  int batch = 2;
  int log_lines = 400;
  int cache_lines = 300;
  // Winning EFA head assets, detected at data loaded: male, female. Their
  // mouths were made for less speech: that face's Mouth scale is multiplied.
  bool expressive_faces[2] = {};
  std::wstring model, direct;  // the network's files, by network
};
constexpr float kExpressiveMouthScale = 0.85f;
constexpr int kCpuDevice = -1;
struct FaceTune {
  // how much louder speech adds to a line's emotion
  float voice_swell = 0.25f;
  // s for the game's expressions to fade out when a line starts (the face
  // becomes the model's alone) and back in after it
  float face_in = 0.5f;
  float face_out = 1;
  float time_offset = 0;
  float blend_in = 0.15f;
  float direct_scale[direct::kNumRegions] = {1, 1, 1, 1, 1, 1};
  // how strongly a line's emotion drives the network
  float voice_emotion = 1;
  a2f::EmotionCaps max_emotion = a2f::kDefaultEmotionCaps;
  float upper_lip = 1;
  float lower_lip = 1;
  float smile_eyes = 1.0f;
  // Where this profile looks as the conversation partner (unused for the
  // player): a2f::GazeAwayFromHold, GazeSettings' rest_seconds and player_odds.
  float gaze_hold = 1;
  float gaze_rest = 3;
  float gaze_settle = 3;
  a2f::GazeMoods gaze_odds = a2f::kGazePlayerOdds;
};
// Whose face a setting is for: Param::profile, the ini section's suffix and
// the menu's tab. Gaze settings have no player profile.
enum Profile : int {
  kProfileShared = -1,
  kProfileMale,
  kProfileFemale,
  kProfilePlayer,
  kNumProfiles
};
struct ProfileInfo {
  const char* section;  // the ini section's suffix
  const char* tab;
};
inline constexpr ProfileInfo kProfiles[] = {
    {"Male", "Male NPCs"}, {"Female", "Female NPCs"}, {"Player", "Player"}};
static_assert(std::size(kProfiles) == kNumProfiles);

struct Tune {  // menu; defaults as dist/BetterFacialAnimations.ini
  bool enabled = true;
  bool player = true;
  bool prefetch = true;
  bool potato_mode = false;
  bool vanilla_emotions = true;
  bool sentiment_analysis = true;
  bool performance_sentiment = true;
  float sentiment_multiplier = 1;
  bool disable_tongue = false;
  // The partner's gaze as a whole: the line's eye motion (gaze_strength) and,
  // with gaze_behavior, where they look (a2f/animation/gaze_director.h).
  bool conversation_gaze = true;
  float gaze_strength = 1;
  bool gaze_behavior = true;
  bool gaze_line_of_sight = true;
  bool gaze_avoid_camera = true;
  bool gaze_turn_head = true;
  // At most one is set (SelectModel); none is V3. Performance is the default.
  // The worker applies the selection and force_cpu live (SwitchModel).
  bool old_model = false;
  bool performance_model = true;
  bool ultra_performance_model = false;
  // By FaceModels(): the first whose switch is set, else the fallback.
  int SelectedModel() const;
  void SelectModel(int network);
  // The menu's current choices (see SelectFaceOption, SelectEmotionOption).
  int FaceChoice() const { return enabled ? SelectedModel() : kFaceDisabled; }
  int SelectedEmotion() const {
    return !sentiment_analysis     ? kEmotionDisabled
           : performance_sentiment ? kEmotionPerformance
                                   : kEmotionFull;
  }
  bool force_cpu = false;
  int gpu_adapter = 0;  // remembered while CPU is selected
  int emotion_device =
      0;  // CPU (-1) or DXGI GPU index; independent of Audio2Face
  bool EmotionUsesGpu() const { return emotion_device >= 0; }
  // DirectML graph fusion per model (GPU only): faster for more VRAM. On
  // before 7.0; the worker applies changes live.
  bool face_graph_fusion = false;
  bool emotion_graph_fusion = false;
  bool logging = true;
  FaceTune male, female, player_face;
  const FaceTune& ForActor(bool is_female, bool is_player) const {
    return is_player ? player_face : ForSex(is_female);
  }
  const FaceTune& ForSex(bool is_female) const {
    return is_female ? female : male;
  }
  FaceTune& ForProfile(int profile) {
    return profile == kProfilePlayer   ? player_face
           : profile == kProfileFemale ? female
                                       : male;
  }
};
extern Config& config;
Tune SnapshotTune();  // coherent copy for playback and workers
void SetProcessingDevice(bool cpu, int adapter);
void SetEmotionDevice(int device);
a2f::GpuAdapterList SnapshotGpuAdapters();
void RequestGpuAdapters();  // discovery runs on the inference worker
// An Audio2Face network (see a2f::kNetV2 and the other identifiers): its
// files and footprint, its name in the menu and, except for the fallback, the
// switch that selects it with that switch's ini key, label and description.
// Everything that lists the models reads FaceModels().
struct ModelInfo {
  int network;
  const wchar_t* model_file;
  const wchar_t* mapping_file;
  int mapping_network;  // direct::Model::network of mapping_file
  double vram_mb, fused_vram_mb;  // graph fusion off, on (a2f::kNetVramMb)
  const char* name;
  bool Tune::* selected = nullptr;  // nullptr: the fallback
  const char* ini_key = nullptr;
  const char* label = nullptr;
  const char* tip = nullptr;
};
// In priority order where several switches are set; the fallback is last.
std::span<const ModelInfo> FaceModels();
// Selected when no switch is set (V3); not the default (Tune).
const ModelInfo& FallbackFaceModel();
const ModelInfo& ModelFor(int network);  // the fallback's for an unknown one

extern SRWLOCK
    settings_lock;  // guards all Param pointers; readers use SnapshotTune()

// Param pointers may only be accessed with settings_lock held.
// Every ini key / menu item: a slider (value, lo..hi) or, with flag set, a
// checkbox, in an ini section (kModel is in the ini's General section) and a
// menu group (kAuto: the section's). Menus sort by profile, then group, then
// table order.
enum class Sec {
  kGeneral,
  kModel,
  kPlayback,
  kDirect,
  kEmotion,
  kDebug,
  kGaze
};
enum class Group {
  kGeneral,
  kModel,
  kSources,
  kPlayback,
  kTransitions,
  kMovement,
  kGaze,
  kInputs,
  kLimits,
  kDebug,
  kAuto
};
struct Param {
  // What the control is inert without: the scene emotion toggle, or the
  // optional sentiment model and its toggle.
  enum Needs { kNone, kScene, kSentiment };
  const char* key;
  float* value;
  float lo, hi;
  const char* label;
  const char* tip;
  Sec sec;
  bool* flag;
  int profile = kProfileShared;  // a Profile
  bool stored = true;  // in the ini
  Needs needs = kNone;
  Group group = Group::kAuto;
  bool menu = true;  // false: in the ini only
  // Typed access to a profile field, independent of the global tune's layout.
  float* (*face_value)(FaceTune&, size_t) = nullptr;
  size_t face_index = 0;  // array element, when applicable
  // The settings page draws this one itself (the model dropdowns, the device
  // and the emotion sources), so its list of controls leaves it out.
  bool own_control = false;
  float* FaceValue(FaceTune& face) const {
    return face_value ? face_value(face, face_index) : nullptr;
  }
};
extern const std::vector<Param> kParams;
extern const size_t kNumParams;

// Puts the selection back to network, e.g. after a switch failed.
void SetSelectedModel(int network);
// kFaceDisabled disables output while retaining the last facial model.
void SelectFaceOption(int network);
// An EmotionChoice; kEmotionDisabled retains the last emotion model.
void SelectEmotionOption(int model);
bool SentimentAvailable(const Tune& tune);

struct RaceSetting {
  uint32_t id;
  std::string key, label;
  bool enabled;
  bool default_enabled;
};
void LoadRaceSettings();  // at data loaded, before the menu and worker
std::vector<RaceSetting> SnapshotRaces();
bool RaceEnabled(uint32_t id);
void SetRaceEnabled(uint32_t id, bool enabled);

struct NpcSetting {
  uint32_t id = 0;  // base form, resolved for this load order; 0 when absent
  std::string key, label;
  bool blocked = false;
  bool female = false;
  // Absent keys inherit shared settings or the sex profile.
  std::map<std::string, float> values;
};
void LoadNpcSettings();
// Game thread; copies the base identity and never retains forms.
NpcSetting DescribeNpc(uint8_t* actor);
std::vector<NpcSetting> SnapshotOverrides();
void SetNpcOverride(const NpcSetting& npc);
void RemoveNpcOverride(const NpcSetting& npc);
bool NpcParamSupported(const Param& param);
// The tune for an actor: the defaults, with the actor's override applied.
// Without any override these return the defaults without locking.
Tune NpcSharedTune(Tune defaults, uint8_t* actor);
FaceTune NpcFaceTune(const Tune& defaults, uint8_t* actor, bool female,
                     bool player);
bool NpcOverridesActive();  // any NPC override resolved in this load order
bool NpcAllowed(uint8_t* actor);
void SetNpcBlocked(const NpcSetting& npc, bool blocked);
int NpcHotkey();  // keyboard scan code; 0 = unbound
void SetNpcHotkey(int key);

constexpr wchar_t kPluginDir[] = L"Data\\SKSE\\Plugins\\BetterFacialAnimations";
std::wstring PluginDir();  // kPluginDir as a full path
constexpr wchar_t kSentimentModel[] = L"emotion_fp16.onnx";  // optional
constexpr wchar_t kPerformanceSentimentModel[] = L"emotion_performance.onnx";
void LoadIni();  // at load
void SaveIni();
// Menu edits: saves once the edits pause (sliders change every frame).
void ScheduleSaveIni();
void RequestGpuRetry();  // retry on the worker between inference calls
void WakeWorker();  // after menu settings change, with settings_lock released
void ResetTune();   // menu defaults
// Turns the log on / off as tune.logging says; a new log starts with the
// version.
void ApplyLogging();
// Whether one more log line of a kind may be written: logging is on and
// fewer than LogLines of the kind (counted in written) were.
bool LogMore(std::atomic<int>& written);

// ---------------- state ----------------
// The network is loaded (the status page and the provider API).
extern std::atomic<bool> net_loaded;
// The A2FDirect bin is loaded and the morph hook in place.
extern std::atomic<bool> ready;
extern direct::Model& model;  // startup mapping; immutable once ready
// Immutable selection shared by queued work, playing lines and head fitting.
struct ModelState {
  int network = a2f::kNetV3;
  bool gpu = true;
  std::wstring file;
  std::shared_ptr<const direct::Model> mapping;
  int gpu_adapter = 0;
  bool graph_fusion = false;
};
std::shared_ptr<const ModelState> SnapshotModel();

double Now();  // s
std::string Lower(const char* s);

struct Stats {
  uint32_t lines = 0, failed = 0, late = 0;
  double audio_sec = 0, gen_sec = 0, last_ms = 0;
  std::string net_info, last_error;
  std::string sentiment_info;  // Sentiment Analysis state, set at data loaded
  // The wanted session failed: the diagnosis (problem, cause, advice), the
  // runtime's own message, the attempts so far and when (Now()) the next one
  // is.
  std::vector<std::string> net_error;
  std::string net_detail;
  double net_retry_at = 0;
  // Adapter 0 and its VRAM as the worker last saw them when it published
  // the session's state (the menu never asks the driver).
  a2f::GpuState gpu;
  // The worker's model and device preference, and a switch in progress or
  // the reason the last one failed.
  int network = a2f::kNetV3;
  bool want_gpu = true;
  int gpu_adapter = 0;
  bool graph_fusion = false;
  bool switching_model = false;
  std::string model_error;
  bool net_gpu = false;  // the loaded session is the GPU one
  // Why the plugin is not running (the A2FDirect bin).
  std::string startup_error;
  bool waiting_for_game = false;
};
// The statistics now, and the lines cached and waiting to be generated.
Stats SnapshotStats(size_t& cached, size_t& queued);
// Forgets the generated lines (not those still being generated).
void ClearLines();

}  // namespace bfa

#endif  // INCLUDE_A2F_PLUGIN_PLUGIN_H_
