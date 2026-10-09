#include "a2f/animation/gaze.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace a2f {
namespace {

constexpr float kFadeSeconds = 0.2f;
constexpr float kMaxStrength = 2;
constexpr float kMaxAngleDegrees = 8;
// A look morph at 1 turns the eye this far; custom eye TRIs may differ.
constexpr float kFullLookDegrees = 30;

}  // namespace

bool GazeOffset(const Line& line, float frame, float ready_frame,
                float strength, float (&offset)[2]) {
  offset[0] = offset[1] = 0;
  if (!line.frames ||
      line.eyes.size() != static_cast<size_t>(line.frames) * kNumEyeAngles ||
      !std::isfinite(frame) || !std::isfinite(ready_frame) ||
      !std::isfinite(strength) || !std::isfinite(line.fps) || line.fps <= 0 ||
      frame < 0 || frame >= line.frames || strength <= 0)
    return false;
  const FrameSample sample = SampleFrame(frame, line.frames);
  const float fade_frames = kFadeSeconds * line.fps;
  const float fade =
      Smoothstep((frame - std::max(0.f, ready_frame)) / fade_frames) *
      Smoothstep((line.frames - 1.f - frame) / fade_frames);
  float angles[kNumEyeAngles];
  for (int angle = 0; angle < kNumEyeAngles; ++angle) {
    const float from =
        line.eyes[static_cast<size_t>(sample.i0) * kNumEyeAngles + angle];
    const float to =
        line.eyes[static_cast<size_t>(sample.i1) * kNumEyeAngles + angle];
    if (!std::isfinite(from) || !std::isfinite(to)) return false;
    angles[angle] = from + (to - from) * sample.a;
    if (!std::isfinite(angles[angle])) return false;
  }
  // Right eye X/Y then left eye X/Y.
  for (int axis = 0; axis < 2; ++axis) {
    const float both_eyes = (angles[axis] + angles[axis + 2]) * 0.5f *
                            std::clamp(strength, 0.f, kMaxStrength);
    offset[axis] = std::clamp(both_eyes, -kMaxAngleDegrees, kMaxAngleDegrees) *
                   fade / kFullLookDegrees;
  }
  return offset[0] != 0 || offset[1] != 0;
}

bool OffsetGazeMorphs(float* values, const float (&offset)[2]) {
  for (float value : offset) {
    if (!std::isfinite(value)) return false;
  }
  for (int morph = 0; morph < kNumGazeMorphs; ++morph) {
    if (!std::isfinite(values[morph]) || values[morph] < 0 || values[morph] > 1)
      return false;
  }
  // Down (0) opposes Up (3) and Right (2) opposes Left (1). The pair's
  // common part stays; the offset moves their signed difference.
  for (int axis = 0; axis < 2; ++axis) {
    if (offset[axis] == 0) continue;
    float& positive = values[axis == 0 ? 0 : 2];
    float& negative = values[axis == 0 ? 3 : 1];
    const float common = std::min(positive, negative);
    const float difference = std::clamp(positive - negative + offset[axis],
                                        common - 1.f, 1.f - common);
    positive = common + std::max(difference, 0.f);
    negative = common + std::max(-difference, 0.f);
  }
  return true;
}

}  // namespace a2f
