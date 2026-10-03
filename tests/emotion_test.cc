// Checks a2f::EmotionTrack, the emotion a line's frames are given: the
// game's emotion at its value and the swell with loudness. ./build.sh test
// builds and runs it.
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "../src/live_core.h"

namespace {
int failures;
void Expect(bool ok, const char* what) {
  if (!ok) ++failures;
  printf("%s %s\n", ok ? "ok" : "FAIL", what);
}

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

  e.swell = 1;
  a2f::EmotionTrack(pcm.data(), pcm.size(), frames, e, t);
  printf("  swell 1: quiet %.3f, loud %.3f\n", at(t, 60, kAnger),
         at(t, 300, kAnger));
  Expect(std::fabs(at(t, 60, kAnger) - 0.75f) < 0.05f &&
             std::fabs(at(t, 300, kAnger) - 1.5f) < 0.05f,
         "the swell: up to 1 + swell times the base where the voice is loud");

  printf(failures ? "%d FAILED\n" : "all ok\n", failures);
  return failures != 0;
}
