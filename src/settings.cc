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
     "Play Audio2Face lip-sync. Untick to see the game's own lip-sync "
     "instantly (A/B).",
     Sec::kGeneral, &tune.enabled},
    {"PlayerFacialAnimations", nullptr, 0, 0, "Player facial animations",
     "Animate the player's voiced dialogue (DBReV 1.4.4+). Untick to keep "
     "the voice mod's own facial animation; NPCs are unaffected.",
     Sec::kGeneral, &tune.player},
    {"Prefetch", nullptr, 0, 0, "Prefetch",
     "Start generating when a dialogue response is created, before it is "
     "spoken.",
     Sec::kGeneral, &tune.prefetch},
    {"TimeOffset", &tune.time_offset, -0.3f, 0.3f, "Sync offset (s)",
     "Shifts the animation against the voice. Positive = earlier mouth shapes.",
     Sec::kPlayback, nullptr},
    {"BlendIn", &tune.blend_in, 0, 0.5f, "Blend-in (s)",
     "Crossfade from vanilla lip-sync when a line finishes generating after it "
     "started.",
     Sec::kPlayback, nullptr},
    {"DirectBrows", &tune.direct_scale[direct::kBrows], 0, 3, "Brows",
     "Brow deformation.", Sec::kDirect, nullptr},
    {"DirectEyes", &tune.direct_scale[direct::kEyes], 0, 3, "Eyes",
     "Eyelid / eye region deformation.", Sec::kDirect, nullptr},
    {"DirectCheeks", &tune.direct_scale[direct::kCheeks], 0, 3, "Cheeks & nose",
     "Cheek and nose deformation.", Sec::kDirect, nullptr},
    {"DirectMouth", &tune.direct_scale[direct::kMouth], 0, 3, "Mouth",
     "Lip and mouth skin deformation.", Sec::kDirect, nullptr},
    {"DirectJaw", &tune.direct_scale[direct::kJaw], 0, 3, "Jaw",
     "How far the lower teeth, gums and tongue move with Audio2Face's jaw.",
     Sec::kDirect, nullptr},
    {"DirectTongue", &tune.direct_scale[direct::kTongue], 0, 3, "Tongue",
     "How far the tongue bends with Audio2Face's tongue.", Sec::kDirect,
     nullptr},
    // all of these apply from the next line spoken
    {"VoiceEmotion", &tune.voice_emotion, 0, 3, "Emotion strength",
     "How strongly a line's emotion (the game's, for the line) drives the "
     "Audio2Face face: brows, eyes, cheeks, mouth. 1 = as given; the game's "
     "is often only 50 of 100.",
     Sec::kEmotion, nullptr},
    {"VoiceSwell", &tune.voice_swell, 0, 2, "Emotion swells with volume",
     "How much the emotion grows where the voice is loud: at the line's "
     "loudest it is 1 + this times its base strength (0 = steady).",
     Sec::kEmotion, nullptr},
    {"FaceFadeIn", &tune.face_in, 0, 3, "Fade to the mod's face (s)",
     "When a line starts: how long the game's own expressions take to fade "
     "out, leaving the face to Audio2Face.",
     Sec::kEmotion, nullptr},
    {"FaceFadeOut", &tune.face_out, 0, 3, "Fade back to the game's face (s)",
     "After a line: how long the game's own expressions take to fade back "
     "in.",
     Sec::kEmotion, nullptr},
    {"Logging", nullptr, 0, 0, "Logging",
     "Also write the generated and played lines to "
     "BetterFacialAnimations.log (SKSE log folder in My Games). The GPU, "
     "the driver and problems loading the model are written there either "
     "way.",
     Sec::kDebug, &tune.logging},
};
extern const size_t kNumParams = std::size(kParams);

namespace {

constexpr char kIni[] = "Data\\SKSE\\Plugins\\BetterFacialAnimations.ini";
constexpr wchar_t kIniW[] = L"Data\\SKSE\\Plugins\\BetterFacialAnimations.ini";

// Version 2 renamed Direct to LocalAnimationStrength and changed its defaults.
// Version 3 turned Logging on by default: an older ini's Logging=0 was that
// version's default, not a choice, so it is turned on once.
// Preserve unrelated keys, including startup configuration, during migration.
constexpr int kIniVersion = 3;

const char* Section(const Param& p) {
  static const char* const kNames[] = {
      "General", "Playback", "LocalAnimationStrength", "Emotion", "Debug"};
  return kNames[static_cast<int>(p.sec)];
}

}  // namespace

Tune SnapshotTune() {
  ScopedLock guard(settings_lock, ScopedLock::kShared);
  return tune;
}

void ResetTune() {
  ScopedLock guard(settings_lock);
  bool logging = tune.logging;  // a diagnostic switch, not a tuning value
  tune = Tune();
  tune.logging = logging;
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
  config.gpu = GetPrivateProfileIntA("General", "GPU", 1, kIni) != 0;
  config.batch = std::max(
      1, static_cast<int>(GetPrivateProfileIntA("General", "Batch", 2, kIni)));
  config.cache_lines = std::max(8, static_cast<int>(GetPrivateProfileIntA(
                                       "General", "CacheLines", 300, kIni)));
  config.log_lines = GetPrivateProfileIntA("Debug", "LogLines", 400, kIni);
  wchar_t m[260];
  GetPrivateProfileStringW(L"General", L"Model", L"network_fp16.onnx", m, 260,
                           kIniW);
  config.model = m;
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
}

}  // namespace bfa
