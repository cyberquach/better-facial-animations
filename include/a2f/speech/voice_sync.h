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
  // a lip-synced line outlives a stopped-sound report shorter than this
  static constexpr double kLipStop = 0.25;

  uint32_t sound_id = kNoSound;
  double requested_at = 0, duration = 0;
  bool seen = false, fallback = false;
  double at = 0;
  double stopped_at = 0;  // first stopped observation since it last played

  // A known sound is playing only while the actor plays that same ID.
  int State(uint32_t actor_sound, int playing) const {
    if (sound_id == kNoSound) return playing;
    return actor_sound == sound_id && playing == 1 ? 1 : 0;
  }

  // Records the start and returns State(). Without an ID the start falls back
  // to the request after kGrace.
  int Observe(uint32_t actor_sound, int playing, double now) {
    int state = State(actor_sound, playing);
    if (state == 1)
      stopped_at = 0;
    else if (state == 0 && !stopped_at)
      stopped_at = now;
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
  // Ended, and not a single frame's false IsPlaying while the lip track runs.
  bool EndedFor(int state, double now, double hold) const {
    return Ended(state, now) && stopped_at && now - stopped_at >= hold;
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

// Fallback for a verified native voice whose facial timer is accelerated by
// animation updates. This estimates elapsed speech, not the audio device
// cursor. Observe only changes in the lip sample: repeated render callbacks
// must not consume time or look like pauses. A long stationary interval
// contributes at most the next lip step, so menu pauses do not become playback
// jumps.
struct SpeechClock {
  // A newly corrected clock is ahead of the estimate by the acceleration so
  // far: crawl at this fraction of real time until the estimate catches up,
  // instead of freezing on one mouth shape.
  static constexpr float kCatchUp = 0.25f;

  bool initialized = false, corrected = false;
  double changed_at = 0, sampled_at = 0, elapsed = 0;
  double window_time = 0;
  float previous = 0, shown = 0, window_lip = 0;

  float Update(float native, double now) {
    if (!initialized) {
      initialized = true;
      changed_at = sampled_at = now;
      elapsed = window_time = native;
      previous = shown = window_lip = native;
      return shown;
    }
    if (native == previous && now - sampled_at > 0.25) changed_at = now;
    const double dt = std::clamp(now - sampled_at, 0., 0.25);
    sampled_at = now;
    const bool moved = native != previous;
    if (moved) {
      const float step = native - previous;
      elapsed += std::min(std::max(0., now - changed_at),
                          static_cast<double>(std::max(0.f, step)));
      changed_at = now;
      if (step < -LipClock::kRestart && !corrected) {
        elapsed = window_time = native;
        window_lip = native;
      }
      previous = native;
    }
    const double span = elapsed - window_time;
    if (span >= 0.25) {
      // Tolerate scheduling jitter and deliberate modest native rate changes.
      // Once selected, keep the fallback for the rest of this voice line.
      corrected |= native - window_lip > span * 1.5 + 0.1;
      if (!corrected) elapsed = native;
      window_time = elapsed;
      window_lip = native;
    }
    if (!corrected)
      shown = native;
    else if (moved)
      shown = Converge(dt);
    return shown;
  }

  float Continue(double now) {
    // No lip resource remains to signal a pause. Bound a callback gap as the
    // existing face clock does; continue only while the same sound is playing.
    const double step = std::clamp(now - sampled_at, 0., 0.25);
    sampled_at = changed_at = now;
    elapsed += step;
    corrected = true;
    return shown = Converge(step);
  }

  float Converge(double dt) const {
    return std::max(static_cast<float>(elapsed),
                    shown + static_cast<float>(dt) * kCatchUp);
  }
};

}  // namespace a2f
