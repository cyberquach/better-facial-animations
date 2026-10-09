// Voice-start gating and the lip clock (include/a2f/speech/voice_sync.h).
#include "a2f/speech/voice_sync.h"

#include <cmath>
#include <cstdio>
#include <initializer_list>

namespace {
int failures = 0;
void Expect(bool ok, const char* label) {
  printf("%s %s\n", ok ? "ok" : "FAIL", label);
  if (!ok) ++failures;
}
bool Near(double a, double b) { return std::fabs(a - b) < 1e-5; }
}  // namespace

int main() {
  // The lip timer runs while the line's sound is still stopped, then the
  // sound starts (observed in game: Irileth, 00093129_3).
  a2f::VoiceStart voice{172};
  a2f::LipClock clock;
  for (double t : {0.005, 0.5, 1.003}) {
    int state = voice.Observe(172, 0, t);
    Expect(voice.Holding(state), "stopped known sound stays gated");
    Expect(clock.Update(static_cast<float>(t), t, voice, true, false) == 0,
           "lip timer cannot advance the waiting line");
  }
  Expect(
      voice.Observe(172, 1, 2.006) == 1 && !voice.fallback && voice.at == 2.006,
      "playing sound starts the line");
  Expect(clock.Update(1.97f, 2.006, voice, false, false) == 0,
         "lip time before the start is discarded");
  Expect(Near(clock.Update(2.22f, 2.306, voice, false, true), 0.25),
         "lip deltas advance the line");
  Expect(Near(clock.Update(2.22f, 3.006, voice, false, true), 0.25),
         "paused lip timer holds the line");
  voice.Observe(172, 1, 4);
  Expect(voice.at == 2.006 && Near(voice.Elapsed(4), 1.994),
         "later observations keep the start");
  Expect(voice.Observe(173, 1, 5) == 0, "another sound is not this line's");

  a2f::VoiceStart pending{279};
  Expect(pending.Observe(266, 1, 0.8) == 0 && !pending.seen,
         "previous line's sound is ignored");
  Expect(pending.Observe(a2f::kNoSound, -1, 2) == 0 && !pending.seen,
         "missing actor handle keeps a known sound gated");
  Expect(pending.Observe(279, 0, 4) == 0 && !pending.fallback &&
             !pending.Ended(0, 4.9),
         "known sound never falls back");
  Expect(pending.Ended(0, 5.1), "sound that never plays ends the line");
  Expect(pending.Observe(279, 1, 4.5) == 1 && pending.at == 4.5,
         "actor adopting the sound later starts the line");
  Expect(!pending.Ended(1, 6) && pending.Ended(0, 6),
         "line ends when its sound stops");

  a2f::VoiceStart immediate{168, 42};
  immediate.Observe(168, 1, 42);
  Expect(immediate.seen && immediate.at == 42,
         "already playing sound starts at once");

  a2f::VoiceStart unknown{a2f::kNoSound, 100};
  unknown.Observe(a2f::kNoSound, 0, 100.49);
  Expect(!unknown.seen, "unknown sound waits half a second");
  unknown.Observe(a2f::kNoSound, 0, 100.5);
  Expect(unknown.seen && unknown.fallback && unknown.at == 100,
         "unknown sound falls back to the request time");
  a2f::LipClock fallback_clock;
  Expect(fallback_clock.Update(0.7f, 100.7, unknown, false, false) == 0.7f,
         "fallback follows the raw lip timer");

  a2f::VoiceStart unavailable;
  Expect(unavailable.Observe(a2f::kNoSound, -1, 1) == -1 && !unavailable.seen &&
             !unavailable.Holding(-1),
         "no ID and no query play on the wall clock");

  a2f::VoiceStart late{7, 10};
  late.Observe(7, 1, 10);
  a2f::LipClock late_clock;
  Expect(Near(late_clock.Update(0.4f, 10.5, late, false, true), 0.5),
         "lip timer appearing mid-line keeps the wall-clock phase");
  Expect(Near(late_clock.Update(0.5f, 10.6, late, false, true), 0.6),
         "then follows lip deltas");
  Expect(late_clock.Update(0.1f, 10.7, late, false, true) == 0.1f,
         "restarted timer is followed from zero");

  printf("%s: %d failures\n", failures ? "FAIL" : "ok", failures);
  return failures ? 1 : 0;
}
