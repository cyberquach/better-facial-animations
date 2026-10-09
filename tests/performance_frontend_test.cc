// The performance frontend alone: frame count, silence, tone placement and
// cancellation, without a network.
#include "a2f/inference/performance_features.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "a2f/inference/live_core.h"

#define CHECK(x)                                      \
  do {                                                \
    if (!(x)) {                                       \
      fprintf(stderr, "line %d: %s\n", __LINE__, #x); \
      std::exit(1);                                   \
    }                                                 \
  } while (0)

namespace {
constexpr int kMel = a2f::kMelBins;

int Peak(const std::vector<float>& mel, int frame) {
  const auto first = mel.begin() + frame * kMel;
  return static_cast<int>(std::max_element(first, first + kMel) - first);
}
}  // namespace

int main() {
  std::vector<float> mel;
  // ceil(n / 16000 * 30) frames, so one sample is one frame.
  const int expected[][2] = {{1, 1}, {159, 1}, {160, 1}, {16000, 30},
                             {16001, 31}, {64017, 121}};
  for (const auto& [n, frames] : expected) {
    std::vector<float> pcm(n, .01f);
    CHECK(a2f::PerformanceFeatures(pcm.data(), pcm.size(), mel));
    CHECK(mel.size() == size_t(frames) * kMel);
    for (float v : mel) CHECK(std::isfinite(v));
  }
  CHECK(!a2f::PerformanceFeatures(nullptr, 0, mel) || mel.empty());
  // Silence is the log of the power floor in every bin; GeneratePerformance
  // pads windows past the clip with the same value.
  std::vector<float> silence(16000, 0.f);
  CHECK(a2f::PerformanceFeatures(silence.data(), silence.size(), mel));
  for (float v : mel) CHECK(std::abs(v - std::log(1e-6f)) < 1e-5f);
  // A 1 kHz tone peaks in the bin around 1 kHz in every frame, whatever its
  // phase, and far above silence.
  int peak = -1;
  for (int offset : {0, 1, 2, 3, 37}) {
    std::vector<float> tone(16000);
    for (size_t i = 0; i < tone.size(); ++i)
      tone[i] = .5f * std::sin(2 * 3.14159265f * 1000 * (i + offset) / 16000);
    CHECK(a2f::PerformanceFeatures(tone.data(), tone.size(), mel));
    for (int frame = 1; frame < 29; ++frame) {
      const int bin = Peak(mel, frame);
      if (peak < 0) peak = bin;
      CHECK(bin == peak && bin >= 26 && bin <= 29);
      CHECK(mel[frame * kMel + bin] > std::log(1e-6f) + 10);
    }
  }
  // Cancellation returns false and runs nothing more.
  int asked = 0;
  CHECK(!a2f::PerformanceFeatures(silence.data(), silence.size(), mel,
                                  [&] { return ++asked < 3; }));
  CHECK(asked == 3);
  puts("performance frontend frames, silence, tone and cancellation: pass");
}
