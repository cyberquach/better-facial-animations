#ifndef INCLUDE_A2F_ANIMATION_GAZE_H_
#define INCLUDE_A2F_ANIMATION_GAZE_H_

#include "a2f/inference/live_core.h"

namespace a2f {

// The engine's look morphs among the modifier keyframes: LookDown, LookLeft,
// LookRight, LookUp.
constexpr int kFirstGazeMorph = 8;
constexpr int kNumGazeMorphs = 4;

// The line's eye angles at frame, both eyes averaged (the game keeps its
// vergence), faded in from ready_frame and out at the line's end, scaled by
// strength (0..2) and expressed as look morph offsets (X down, Y right).
// False with a zero offset for a missing or invalid track or no movement.
bool GazeOffset(const Line& line, float frame, float ready_frame,
                float strength, float (&offset)[2]);

// Adds offset to the kNumGazeMorphs look values, keeping each in [0, 1] and
// preserving what the opposing pairs have in common. False, with values
// untouched, for invalid values or offsets.
bool OffsetGazeMorphs(float* values, const float (&offset)[2]);

}  // namespace a2f

#endif  // INCLUDE_A2F_ANIMATION_GAZE_H_
