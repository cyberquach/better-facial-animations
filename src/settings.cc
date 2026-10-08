// The ini and the table of settings declared in plugin.h.
#include <windows.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <string>
#include <vector>

#include "plugin.h"
#include "scoped_lock.h"
#include "skse.h"

namespace bfa {

Config& config = *new Config;
namespace {
Tune tune;
}  // namespace
SRWLOCK settings_lock = SRWLOCK_INIT;

extern const std::vector<Param> kParams = [] {
  std::vector<Param> params = {
      {"Enabled", nullptr, 0, 0, "Enabled",
       "Untick for the game's own lip-sync, instantly (A/B).", Sec::kGeneral,
       &tune.enabled},
      {"PlayerFacialAnimations", nullptr, 0, 0, "Player facial animations",
       "Automatically detect DBReV or DBVO2 for player dialogue. "
       "DBVO2 prototype for Skyrim. Untick to keep the voice mod's own animation.",
       Sec::kGeneral, &tune.player},
      {"PotatoMode", nullptr, 0, 0, "Potato mode",
       "Animate the player and the NPC actively in dialogue with you; "
       "keep the model loaded.", Sec::kGeneral,
       &tune.potato_mode},
      {"Prefetch", nullptr, 0, 0, "Prefetch",
       "Generate a line when its dialogue response is created, before it is "
       "spoken.",
       Sec::kGeneral, &tune.prefetch},
      // both are read at startup
      {"UseOldModel", nullptr, 0, 0, "Use Old Model",
       "Old model: ~100 MB of VRAM, less expressive, slower to inference.\n"
       "V3 model (box unchecked): ~250 MB of VRAM, more expressive, faster to "
       "inference.\n"
       "Restart the game to apply.",
       Sec::kModel, &tune.old_model},
      {"ForceCPU", nullptr, 0, 0, "Force CPU",
       "Never load the model into VRAM: it runs on the CPU, much slower, and "
       "long lines start late.\n"
       "Restart the game to apply.",
       Sec::kModel, &tune.force_cpu},
      {"DisableTongue", nullptr, 0, 0, "Disable tongue motion",
       "Restore the tongue behavior from before this feature. Applies "
       "immediately.",
       Sec::kDirect, &tune.disable_tongue},
  };
  static char mouth_tip[160];
  snprintf(mouth_tip, sizeof mouth_tip,
           "Lip and mouth skin movement. 1 uses %g strength when this "
           "sex's Expressive Facial Animation is detected, otherwise 1.0.",
           kExpressiveMouthScale);
  for (int profile = 0; profile < 3; ++profile) {
    FaceTune& face =
        profile == 2 ? tune.player_face : profile ? tune.female : tune.male;
    for (Param p : std::initializer_list<Param>{
             {"TimeOffset", &face.time_offset, -0.3f, 0.3f, "Sync offset (s)",
              "Positive = mouth shapes earlier.", Sec::kPlayback, nullptr},
             {"BlendIn", &face.blend_in, 0, 0.5f, "Blend-in (s)",
              "Crossfade from vanilla lip-sync when a line is ready after it "
              "started.",
              Sec::kPlayback, nullptr},
             {"DirectBrows", &face.direct_scale[direct::kBrows], 0, 3, "Brows",
              "Brow movement.", Sec::kDirect, nullptr},
             {"DirectEyes", &face.direct_scale[direct::kEyes], 0, 3, "Eyes",
              "Eyelid and eye region movement.", Sec::kDirect, nullptr},
             {"DirectCheeks", &face.direct_scale[direct::kCheeks], 0, 3,
              "Cheeks & nose", "Cheek and nose movement.", Sec::kDirect,
              nullptr},
             {"DirectMouth", &face.direct_scale[direct::kMouth], 0, 3, "Mouth",
              mouth_tip, Sec::kDirect, nullptr},
             {"DirectJaw", &face.direct_scale[direct::kJaw], 0, 3, "Jaw",
              "How far the lower teeth and gums move with the jaw.",
              Sec::kDirect, nullptr},
             {"DirectTongue", &face.direct_scale[direct::kTongue], 0, 3,
              "Tongue",
              "Tongue articulation strength for all phonemes. 1 uses the "
              "calibrated default; 0 keeps jaw follow only.",
              Sec::kDirect, nullptr},
             {"UpperLip", &face.upper_lip, 0, 3, "Upper lip",
              "Upper lip movement, multiplied by Mouth strength.", Sec::kDirect,
              nullptr},
             {"LowerLip", &face.lower_lip, 0, 3, "Lower lip",
              "Lower lip movement, multiplied by Mouth strength.", Sec::kDirect,
              nullptr},
             // all of these apply from the next line spoken
             {"VoiceEmotion", &face.voice_emotion, 0, 3, "Strength",
              "How strongly a line's emotion (the game's) drives the brows, "
              "eyes, "
              "cheeks and mouth. 1 = as given. Applies from the next line.",
              Sec::kEmotion, nullptr},
             {"VoiceSwell", &face.voice_swell, 0, 2, "Swell with volume",
              "How much the emotion grows where the voice is loud. 0 = steady. "
              "Applies from the next line.",
              Sec::kEmotion, nullptr},
             {"FaceFadeIn", &face.face_in, 0, 3, "Fade to the mod's face (s)",
              "When a line starts: how long the game's own expressions take to "
              "fade "
              "out, leaving the face to Audio2Face.",
              Sec::kEmotion, nullptr},
             {"FaceFadeOut", &face.face_out, 0, 3,
              "Fade to the game's face (s)",
              "After a line: how long the game's own expressions take to fade "
              "back "
              "in.",
              Sec::kEmotion, nullptr},
         }) {
      p.profile = profile;
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

// Version 2 renamed Direct to LocalAnimationStrength and changed its defaults.
// Version 3 turned Logging on by default: an older ini's Logging=0 was that
// version's default, not a choice, so it is turned on once.
// Preserve unrelated keys, including startup configuration, during migration.
// Version 4 gives each sex its own sliders; shared values seed both profiles.
// Version 5 adds a player profile and normalizes lips to backend calibration.
constexpr int kIniVersion = 5;

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

Tune SnapshotTune() {
  ScopedLock guard(settings_lock, ScopedLock::kShared);
  return tune;
}

void ResetTune() {
  ScopedLock guard(settings_lock);
  // a diagnostic switch and the model, not tuning values
  const Tune old = tune;
  tune = Tune();
  tune.logging = old.logging;
  tune.old_model = old.old_model;
  tune.force_cpu = old.force_cpu;
}

void ApplyLogging() { skse::SetLogging(SnapshotTune().logging); }

bool LogMore(std::atomic<int>& written) {
  return SnapshotTune().logging && ++written <= config.log_lines;
}

void LoadIni() {
  // Startup only: no readers or menu callbacks exist yet.
  tune = Tune();
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
  config.use_device_wrapper =
      GetPrivateProfileIntA("Debug", "UseDeviceWrapper", 0, kIni) != 0;
  // UseOldModel was Network=2 (3: v3.0) in earlier 3.0 builds, and ForceCPU
  // GPU=0
  tune.old_model = GetPrivateProfileIntA("General", "Network", 3, kIni) == 2;
  tune.force_cpu = GetPrivateProfileIntA("General", "GPU", 1, kIni) == 0;
  char b[128], d[128];
  for (size_t i = 0; i < kNumParams; ++i) {
    const Param& p = kParams[i];
    if (migrate && p.sec == Sec::kDirect) continue;
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
  config.gpu = !tune.force_cpu;
  config.network = tune.old_model ? 2 : 3;
  config.model = config.network == 3 ? L"a2f3_fp16.onnx" : L"network_fp16.onnx";
  config.direct = config.network == 3 ? L"A2FDirect3.bin" : L"A2FDirect.bin";
  if (version < 3) tune.logging = true;
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
  {
    ScopedLock guard(settings_lock, ScopedLock::kShared);
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
  for (size_t i = 0; i < kNumParams; ++i) {
    const Param& p = kParams[i];
    WritePrivateProfileStringA(Section(p).c_str(), p.key, values[i].c_str(),
                               kIni);
  }
  // replaced by UseOldModel and ForceCPU
  WritePrivateProfileStringA("General", "Model", nullptr, kIni);
  WritePrivateProfileStringA("General", "Network", nullptr, kIni);
  WritePrivateProfileStringA("General", "GPU", nullptr, kIni);
}

}  // namespace bfa
