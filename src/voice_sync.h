#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace a2f {

constexpr uint32_t kNoSound = UINT32_MAX;

// When a dialogue line's sound starts. The engine copies sound handles, so
// match the sound ID, not the handle's address. Playing states are 1, 0 or -1
// (unknown: no actor handle or no query).
struct VoiceStart {
  // a stopped voice is not trusted this soon after the request
  static constexpr double kGrace = 0.5;
  // a sound that never plays ends the line after duration + kNeverSlack,
  // at least kNeverMin
  static constexpr double kNeverMin = 5, kNeverSlack = 2;

  uint32_t sound_id = kNoSound;
  double requested_at = 0, duration = 0;
  bool seen = false, fallback = false;
  double at = 0;

  // A known sound is playing only while the actor plays that same ID.
  int State(uint32_t actor_sound, int playing) const {
    if (sound_id == kNoSound) return playing;
    return actor_sound == sound_id && playing == 1 ? 1 : 0;
  }

  // Records the start and returns State(). Without an ID the start falls back
  // to the request after kGrace.
  int Observe(uint32_t actor_sound, int playing, double now) {
    int state = State(actor_sound, playing);
    if (!seen && (state == 1 || (sound_id == kNoSound && state == 0 &&
                                 now - requested_at >= kGrace))) {
      seen = true;
      fallback = state != 1;
      at = fallback ? requested_at : now;
    }
    return state;
  }

  bool Holding(int state) const { return !seen && state == 0; }
  double Elapsed(double now) const { return now - (seen ? at : requested_at); }

  // A line without lip data ends: its sound stopped, or never played.
  bool Ended(int state, double now) const {
    return state == 0 &&
           now - requested_at >
               (seen ? kGrace : std::max(kNeverMin, duration + kNeverSlack));
  }
};

// Line time from the engine lip timer, which may run before the voice starts.
// It holds at zero until then and counts from the timer's value at the start.
struct LipClock {
  static constexpr float kRestart = 0.05f;  // a drop this large restarts it

  bool waited = false, anchored = false;
  float anchor = 0, last_lip = 0;

  // direct_started: wall-clock playback ran before the lip timer appeared;
  // its phase is kept.
  float Update(float lip, double now, const VoiceStart& voice, bool hold,
               bool direct_started) {
    if (hold) {
      waited = true;
      return 0;
    }
    if (lip < last_lip - kRestart) {
      anchor = 0;
      anchored = true;
    } else if (!anchored) {
      anchored = true;
      if (voice.seen && !voice.fallback)
        anchor = waited           ? lip
                 : direct_started ? lip - static_cast<float>(now - voice.at)
                                  : 0;
    }
    last_lip = lip;
    return std::fmax(0.f, lip - anchor);
  }
};

}  // namespace a2f
