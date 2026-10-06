// The ini and the table of settings declared in plugin.h.
#include <windows.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iterator>

#include "plugin.h"
#include "scoped_lock.h"
#include "skse.h"

namespace bfa {

Config& config = *new Config;
namespace {
Tune tune;
}  // namespace
SRWLOCK settings_lock = SRWLOCK_INIT;

extern const Param kParams[] = {
    {"Enabled", nullptr, 0, 0, "Enabled",
     "Untick for the game's own lip-sync, instantly (A/B).", Sec::kGeneral,
     &tune.enabled},
    {"PlayerFacialAnimations", nullptr, 0, 0, "Player facial animations",
     "Animate the player's voiced dialogue (DBReV 1.4.4+). Untick to keep "
     "the voice mod's own animation.",
     Sec::kGeneral, &tune.player},
    {"Prefetch", nullptr, 0, 0, "Prefetch",
     "Generate a line when its dialogue response is created, before it is "
     "spoken.",
     Sec::kGeneral, &tune.prefetch},
    // both are read at startup
    {"UseOldModel", nullptr, 0, 0, "Use Old Model",
     "Old model: ~100 MB of VRAM, less expressive, slower to inference.\n"
     "V3 model (box unchecked): ~250 MB of VRAM, more expressive, faster to "
     "inference.\n"
     "Save, then restart the game to apply.",
     Sec::kModel, &tune.old_model},
    {"ForceCPU", nullptr, 0, 0, "Force CPU",
     "Never load the model into VRAM: it runs on the CPU, much slower, and "
     "long lines start late.\n"
     "Save, then restart the game to apply.",
     Sec::kModel, &tune.force_cpu},
    {"TimeOffset", &tune.time_offset, -0.3f, 0.3f, "Sync offset (s)",
     "Positive = mouth shapes earlier.", Sec::kPlayback, nullptr},
    {"BlendIn", &tune.blend_in, 0, 0.5f, "Blend-in (s)",
     "Crossfade from vanilla lip-sync when a line is ready after it started.",
     Sec::kPlayback, nullptr},
    {"DirectBrows", &tune.direct_scale[direct::kBrows], 0, 3, "Brows",
     "Brow movement.", Sec::kDirect, nullptr},
    {"DirectEyes", &tune.direct_scale[direct::kEyes], 0, 3, "Eyes",
     "Eyelid and eye region movement.", Sec::kDirect, nullptr},
    {"DirectCheeks", &tune.direct_scale[direct::kCheeks], 0, 3, "Cheeks & nose",
     "Cheek and nose movement.", Sec::kDirect, nullptr},
    {"DirectMouth", &tune.direct_scale[direct::kMouth], 0, 3, "Mouth",
     "Lip and mouth skin movement.", Sec::kDirect, nullptr},
    {"DirectJaw", &tune.direct_scale[direct::kJaw], 0, 3, "Jaw",
     "How far the lower teeth, gums and tongue move with the jaw.",
     Sec::kDirect, nullptr},
    {"DirectTongue", &tune.direct_scale[direct::kTongue], 0, 3, "Tongue",
     "How far the tongue bends.", Sec::kDirect, nullptr},
    // all of these apply from the next line spoken
    {"VoiceEmotion", &tune.voice_emotion, 0, 3, "Strength",
     "How strongly a line's emotion (the game's) drives the brows, eyes, "
     "cheeks and mouth. 1 = as given. Applies from the next line.",
     Sec::kEmotion, nullptr},
    {"VoiceSwell", &tune.voice_swell, 0, 2, "Swell with volume",
     "How much the emotion grows where the voice is loud. 0 = steady. "
     "Applies from the next line.",
     Sec::kEmotion, nullptr},
    {"FaceFadeIn", &tune.face_in, 0, 3, "Fade to the mod's face (s)",
     "When a line starts: how long the game's own expressions take to fade "
     "out, leaving the face to Audio2Face.",
     Sec::kEmotion, nullptr},
    {"FaceFadeOut", &tune.face_out, 0, 3, "Fade to the game's face (s)",
     "After a line: how long the game's own expressions take to fade back "
     "in.",
     Sec::kEmotion, nullptr},
    {"Logging", nullptr, 0, 0, "Logging",
     "Also write the generated and played lines to "
     "BetterFacialAnimations.log (SKSE log folder). The GPU, the driver and "
     "model problems are logged either way.",
     Sec::kDebug, &tune.logging},
};
extern const size_t kNumParams = std::size(kParams);

namespace {

constexpr char kIni[] = "Data\\SKSE\\Plugins\\BetterFacialAnimations.ini";

// Version 2 renamed Direct to LocalAnimationStrength and changed its defaults.
// Version 3 turned Logging on by default: an older ini's Logging=0 was that
// version's default, not a choice, so it is turned on once.
// Preserve unrelated keys, including startup configuration, during migration.
constexpr int kIniVersion = 3;

const char* Section(const Param& p) {
  static const char* const kNames[] = {
      "General", "General", "Playback", "LocalAnimationStrength",
      "Emotion", "Debug"};
  return kNames[static_cast<int>(p.sec)];
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
  char b[128], d[32];
  for (size_t i = 0; i < kNumParams; ++i) {
    const Param& p = kParams[i];
    if (migrate && p.sec == Sec::kDirect) continue;
    if (p.flag) {
      *p.flag = GetPrivateProfileIntA(Section(p), p.key, *p.flag, kIni) != 0;
      continue;
    }
    snprintf(d, sizeof d, "%g", *p.value);
    GetPrivateProfileStringA(Section(p), p.key, d, b, sizeof b, kIni);
    char* end;
    const float value = std::strtof(b, &end);
    if (end != b && *end == '\0' && std::isfinite(value))
      *p.value = std::clamp(value, p.lo, p.hi);
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
void SaveIni() {
  ScopedLock guard(settings_lock, ScopedLock::kShared);
  char b[64];
  snprintf(b, sizeof b, "%d", kIniVersion);
  WritePrivateProfileStringA("General", "IniVersion", b, kIni);
  for (size_t i = 0; i < kNumParams; ++i) {
    const Param& p = kParams[i];
    if (p.value) snprintf(b, sizeof b, "%.4g", *p.value);
    WritePrivateProfileStringA(Section(p), p.key,
                               p.value   ? b
                               : *p.flag ? "1"
                                         : "0",
                               kIni);
  }
  // replaced by UseOldModel and ForceCPU
  WritePrivateProfileStringA("General", "Model", nullptr, kIni);
  WritePrivateProfileStringA("General", "Network", nullptr, kIni);
  WritePrivateProfileStringA("General", "GPU", nullptr, kIni);
}

}  // namespace bfa
