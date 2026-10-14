// Checks a2f::EmotionTrack, the emotion a line's frames are given: the
// game's emotion at its value and the swell with loudness. ./build.sh test
// builds and runs it.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include "a2f/inference/live_core.h"

namespace {
int failures;
void Expect(bool ok, const char* what) {
  if (!ok) ++failures;
  printf("%s %s\n", ok ? "ok" : "FAIL", what);
}

struct Voice : a2f::EmotionNet {
  int calls = 0, selected = 3;
  bool fail = false, nonfinite = false;
  bool Run(const float*, float* logits, std::string& err) override {
    ++calls;
    if (fail) {
      err = "test failure";
      return false;
    }
    std::fill_n(logits, a2f::kNumVoiceEmo, -100.f);
    logits[selected] = nonfinite ? std::numeric_limits<float>::infinity() : 100;
    return true;
  }
};
constexpr int kJoy = 6, kAnger = 1;
}  // namespace

int main() {
  // 12 s: quiet, then loud from 6 s
  const int frames = 12 * a2f::kFps;
  std::vector<float> pcm(12 * a2f::kSampleRate);
  for (size_t i = 0; i < pcm.size(); ++i)
    pcm[i] = (i < pcm.size() / 2 ? 0.01f : 0.5f) *
             static_cast<float>(std::sin(i * 0.2));
  auto at = [](const std::vector<float>& t, int frame, int c) {
    return t[static_cast<size_t>(frame) * a2f::kNumMod + c];
  };
  std::vector<float> t;

  a2f::Emotion e;  // neutral
  a2f::EmotionTrack(pcm.data(), pcm.size(), frames, e, t);
  float top = 0;
  for (float v : t) top = std::fmax(top, std::fabs(v));
  Expect(t.size() == static_cast<size_t>(frames) * a2f::kNumMod && top == 0,
         "a neutral line has no emotion");

  e.type = 1;  // anger at 50, times 1.5
  e.value = 50;
  e.gain = 1.5f;
  a2f::EmotionTrack(pcm.data(), pcm.size(), frames, e, t);
  Expect(std::fabs(at(t, 0, kAnger) - 0.75f) < 1e-3f &&
             std::fabs(at(t, frames - 1, kAnger) - 0.75f) < 1e-3f &&
             at(t, 100, kJoy) == 0,
         "the game's emotion: value / 100 times the strength, steady");

  e.maximum.fill(2);  // exercise volume swell with the optional legacy limit
  e.swell = 1;
  a2f::EmotionTrack(pcm.data(), pcm.size(), frames, e, t);
  printf("  swell 1: quiet %.3f, loud %.3f\n", at(t, 60, kAnger),
         at(t, 300, kAnger));
  Expect(std::fabs(at(t, 60, kAnger) - 0.75f) < 0.05f &&
             std::fabs(at(t, 300, kAnger) - 1.5f) < 0.05f,
         "the swell: up to 1 + swell times the base where the voice is loud");

  const auto uncapped = t;
  e.maximum.fill(1);
  a2f::EmotionTrack(pcm.data(), pcm.size(), frames, e, t);
  bool clipped = false;
  bool exact = true;
  for (size_t i = 0; i < t.size(); ++i) {
    clipped |= uncapped[i] > 1;
    exact &= t[i] == std::min(uncapped[i], 1.f);
  }
  Expect(clipped && exact, "cap clips loud peaks while preserving quieter frames");
  for (int type = 1; type <= 7; ++type) {
    e.type = type;
    e.value = 100;
    e.gain = 3;
    e.swell = 2;
    for (float cap : {0.f, .25f, 2.f, -1.f, 9.f,
                      std::numeric_limits<float>::quiet_NaN()}) {
      e.maximum.fill(cap);
      a2f::EmotionTrack(pcm.data(), pcm.size(), frames, e, t);
      const float limit = std::isfinite(cap) ? std::clamp(cap, 0.f, 2.f) : 2.f;
      Expect(std::all_of(t.begin(), t.end(), [=](float v) {
               return std::isfinite(v) && v >= 0 && v <= limit;
             }),
             "all channels honor the cap, including swell and invalid bounds");
    }
  }

  // Independent expected Skyrim mapping: default caps limit each mapped
  // channel to full intensity.
  constexpr float kExpected[8][a2f::kNumMod] = {
      {}, {0, 1}, {0, 0, 0, 1}, {0, 0, 0, 0, 1},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
      {0, 0, 0, 0, 0, 0, 1}, {1}, {.3f, 0, .3f}};
  e.maximum = a2f::kDefaultEmotionCaps;
  e.value = 100;
  e.swell = 0;
  for (int type = 0; type <= 7; ++type) {
    e.type = type;
    for (float gain : {.5f, 1.f, 3.f}) {
      e.gain = gain;
      a2f::EmotionTrack(pcm.data(), pcm.size(), frames, e, t);
      bool unchanged = true;
      for (int frame = 0; frame < frames; ++frame) {
        for (int c = 0; c < a2f::kNumMod; ++c) {
          const float expected = std::min(gain * kExpected[type][c], 1.f);
          unchanged &= std::fabs(at(t, frame, c) - expected) < 1e-6f;
        }
      }
      Expect(unchanged, "default caps limit every mapped emotion to one");
    }
  }

  // Puzzled feeds two channels at once: each cap must act independently.
  e.type = 7;
  e.value = 100;
  e.gain = 1;
  e.swell = 0;
  e.maximum = a2f::kDefaultEmotionCaps;
  e.maximum[a2f::kAmazement] = .1f;
  e.maximum[a2f::kJoy] = 0;
  a2f::EmotionTrack(pcm.data(), pcm.size(), frames, e, t);
  Expect(at(t, 0, a2f::kAmazement) == .1f &&
             at(t, 0, a2f::kCheekiness) == .3f,
         "one channel cap preserves the other channel in a mixed emotion");
  e.type = 5;
  a2f::EmotionTrack(pcm.data(), pcm.size(), frames, e, t);
  Expect(at(t, 0, a2f::kJoy) == 0, "joy can be disabled on its own");
  e.type = 4;
  a2f::EmotionTrack(pcm.data(), pcm.size(), frames, e, t);
  Expect(at(t, 0, a2f::kSadness) == 1, "joy cap leaves sadness unchanged");
  Voice voice;
  std::string err;
  e = {};
  e.type = 5;
  e.value = 50;
  e.gain = 1;
  e.maximum.fill(2);
  Expect(a2f::AnalyzeSentiment(voice, pcm.data(), 16000, e.sentiment, err),
         "short audio produces sentiment");
  Expect(voice.calls == 1 && e.sentiment.size() == 30 * a2f::kNumMod,
         "short audio is heard once, not once per frame");
  a2f::EmotionTrack(pcm.data(), 16000, 30, e, t);
  Expect(std::fabs(at(t, 0, kJoy) - 1.5f) < .001f,
         "sentiment ADDS to the same dialogue emotion, not max or replacement");
  e.type = 1;
  a2f::EmotionTrack(pcm.data(), 16000, 30, e, t);
  Expect(at(t, 0, kJoy) == 1 && at(t, 0, kAnger) == .5f,
         "different sentiment and dialogue channels coexist");
  for (float multiplier : {0.f, .25f, .5f, 1.f}) {
    e.sentiment_multiplier = multiplier;
    a2f::EmotionTrack(pcm.data(), 16000, 30, e, t);
    Expect(at(t, 0, kJoy) == multiplier && at(t, 0, kAnger) == .5f,
           "multiplier scales only the added emotion; authored emotion is unchanged");
  }
  e.sentiment[kJoy] = .2f;
  for (float multiplier : {1.5f, 2.f}) {
    e.sentiment_multiplier = multiplier;
    a2f::EmotionTrack(pcm.data(), 16000, 30, e, t);
    Expect(std::fabs(at(t, 0, kJoy) - .2f * multiplier) < .0001f &&
               at(t, 0, kAnger) == .5f,
           "up to double sentiment leaves authored emotion unchanged");
  }
  e.sentiment[kJoy] = .8f;
  a2f::EmotionTrack(pcm.data(), 16000, 30, e, t);
  Expect(std::fabs(at(t, 0, kJoy) - 1.6f) < .0001f,
         "0.8 sentiment times two reaches 1.6 before the final emotion limit");
  e.sentiment[kJoy] = 1;
  a2f::EmotionTrack(pcm.data(), 16000, 30, e, t);
  Expect(at(t, 0, kJoy) == 2 && at(t, 0, kAnger) == .5f,
         "double sentiment reaches two without an intermediate cap");
  e.sentiment_multiplier = std::numeric_limits<float>::quiet_NaN();
  a2f::EmotionTrack(pcm.data(), 16000, 30, e, t);
  Expect(at(t, 0, kJoy) == 0 && at(t, 0, kAnger) == .5f,
         "invalid multiplier suppresses only sentiment");
  e.sentiment_multiplier = 1;
  e.gain = 0;
  a2f::EmotionTrack(pcm.data(), 16000, 30, e, t);
  Expect(at(t, 0, kJoy) == 1 && at(t, 0, kAnger) == 0,
         "dialogue at zero does not suppress sentiment");
  e.type = 0;
  a2f::EmotionTrack(pcm.data(), 16000, 30, e, t);
  Expect(at(t, 0, kJoy) == 1,
         "neutral dialogue can use sentiment; total retains safety cap");
  e.type = 5;
  e.value = 50;
  e.gain = 1;
  e.maximum[kJoy] = .6f;
  a2f::EmotionTrack(pcm.data(), 16000, 30, e, t);
  Expect(at(t, 0, kJoy) == .6f, "combined sentiment obeys the configured cap");
  e.maximum[kJoy] = 0;
  a2f::EmotionTrack(pcm.data(), 16000, 30, e, t);
  Expect(at(t, 0, kJoy) == 0, "sentiment cannot revive a disabled emotion");
  e.maximum.fill(2);
  e.gain = 0;
  e.sentiment[kJoy] = .1f;
  a2f::EmotionTrack(pcm.data(), 16000, 30, e, t);
  Expect(at(t, 0, kJoy) == .1f, "cap preserves probabilities below the limit");
  e.sentiment_multiplier = std::numeric_limits<float>::quiet_NaN();
  a2f::EmotionTrack(pcm.data(), 16000, 30, e, t);
  Expect(at(t, 0, kJoy) == 0, "nonfinite sentiment multiplier is disabled");
  e.sentiment_multiplier = 0;
  a2f::EmotionTrack(pcm.data(), 16000, 30, e, t);
  Expect(at(t, 0, kJoy) == 0, "zero disables the sentiment contribution");
  const int mapping[] = {1, 3, 4, 6, -1, 9};
  for (int label = 0; label < 6; ++label) {
    voice.selected = label;
    Expect(a2f::AnalyzeSentiment(voice, pcm.data(), 16000, e.sentiment, err),
           "all six model labels accepted");
    for (int c = 0; c < a2f::kNumMod; ++c)
      Expect(
          std::fabs(e.sentiment[c] - (c == mapping[label] ? 1.f : 0.f)) < 1e-6f,
          "model labels map correctly and neutral contributes nothing");
  }
  voice.fail = true;
  Expect(
      !a2f::AnalyzeSentiment(voice, pcm.data(), pcm.size(), e.sentiment, err) &&
          e.sentiment.empty(),
      "failure discards sentiment output");
  voice.fail = false;
  voice.nonfinite = true;
  Expect(
      !a2f::AnalyzeSentiment(voice, pcm.data(), pcm.size(), e.sentiment, err) &&
          e.sentiment.empty(),
      "nonfinite logits rejected");
  voice.nonfinite = false;
  int calls = voice.calls;
  Expect(!a2f::AnalyzeSentiment(voice, pcm.data(), pcm.size(), e.sentiment, err,
                                [] { return false; }) &&
             voice.calls == calls,
         "cancelled work never runs the model");
  Expect(!a2f::AnalyzeSentiment(voice, pcm.data(), 0, e.sentiment, err),
         "empty input rejected");
  Expect(
      a2f::AnalyzeSentiment(voice, pcm.data(), pcm.size(), e.sentiment, err) &&
          voice.calls > calls + 1 &&
          e.sentiment.size() == frames * a2f::kNumMod,
      "long audio is sampled across multiple windows");

  printf(failures ? "%d FAILED\n" : "all ok\n", failures);
  return failures != 0;
}
