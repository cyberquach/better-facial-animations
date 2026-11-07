// Frozen Thin120 output against a full-sequence Python/ORT reference.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../src/plugin/better_facial_animations.cc"
#include "a2f/inference/ultra_performance_stats.h"

#define CHECK(x)                                      \
  do {                                                \
    if (!(x)) {                                       \
      fprintf(stderr, "line %d: %s\n", __LINE__, #x); \
      return 1;                                       \
    }                                                 \
  } while (0)

namespace {
constexpr int kChannels = a2f::kPerformanceChannels;
constexpr int kWindow = a2f::kUltraPerformanceWindowFrames;
constexpr int kContext = a2f::kUltraPerformanceContextFrames;

struct SlotNet : a2f::Net {
  int calls = 0;
  int Version() const override { return a2f::kNetUltraPerformance; }
  int Batch() const override { return 1; }
  bool Run(const float*, const float*, int, float*, std::string&) override {
    return false;
  }
  bool RunUltraPerformance(const float*, int64_t, const float*, float* motion,
                           std::string&) override {
    ++calls;
    std::fill_n(motion, kWindow * kChannels, 0.f);
    for (int slot = 0; slot < kWindow; ++slot)
      motion[slot * kChannels] = static_cast<float>(slot);
    return true;
  }
};

std::wstring FullPath(const wchar_t* path) {
  wchar_t full[32768];
  GetFullPathNameW(path, 32768, full, nullptr);
  return full;
}
}  // namespace

int main() {
  FILE* input = fopen("tests/ultra_performance.fixtures", "rb");
  CHECK(input);
  uint32_t samples = 0, frames = 0;
  CHECK(fread(&samples, sizeof samples, 1, input) == 1);
  CHECK(fread(&frames, sizeof frames, 1, input) == 1);
  CHECK(samples > 0 && samples <= 1000000 &&
        frames == (samples * 30 + 15999) / 16000);
  std::vector<float> pcm(samples), expected[2];
  CHECK(fread(pcm.data(), sizeof(float), samples, input) == samples);
  for (auto& ref : expected) {
    ref.resize(frames * kChannels);
    CHECK(fread(ref.data(), sizeof(float), ref.size(), input) == ref.size());
  }
  CHECK(fgetc(input) == EOF);
  fclose(input);
  const a2f::Emotion emotion{1, 100, 1, 0};
  a2f::Line line;
  std::string err, info;
  SlotNet fake;
  CHECK(a2f::GeneratePerformance(fake, pcm.data(), samples, emotion, true, line,
                                 err));
  CHECK(fake.calls == (frames + 89) / 90);
  for (size_t f = 0; f < frames; ++f) {
    const float value =
        float(f % 90 + kContext) * a2f::ultra_performance::kTargetStd[0] +
        a2f::ultra_performance::kTargetMean[0];
    CHECK(line.direct[f * kChannels] == value);
  }
  a2f::OrtOptions options;
  options.gpu = std::getenv("BFA_TEST_ULTRA_GPU") != nullptr;
  auto net = a2f::CreateUltraPerformanceNet(
      FullPath(L"build/mod/SKSE/Plugins/BetterFacialAnimations"),
      FullPath(L"dist/ultra-performance.onnx"), options, err, info);
  if (!net) fprintf(stderr, "%s\n", err.c_str());
  CHECK(net && net->Version() == a2f::kNetUltraPerformance);
  for (int run = 0; run < 4; ++run) {
    const int identity = run % 2;
    CHECK(a2f::GeneratePerformance(*net, pcm.data(), samples, emotion,
                                   identity == 0, line, err));
    CHECK(line.frames == frames && line.fps == 30 && line.eyes.empty() &&
          line.direct.size() == expected[identity].size());
    double error = 0;
    for (size_t i = 0; i < line.direct.size(); ++i) {
      CHECK(std::isfinite(line.direct[i]));
      error = std::max(
          error, double(std::abs(line.direct[i] - expected[identity][i])));
    }
    printf("Ultra identity %d native/full-sequence max error %.9g\n", identity,
           error);
    CHECK(error < .002);
  }
  CHECK(
      a2f::GeneratePerformance(*net, pcm.data(), 1, emotion, false, line, err));
  CHECK(line.frames == 1 && line.direct.size() == kChannels);
  int checks = 0;
  CHECK(!a2f::GeneratePerformance(*net, pcm.data(), samples, emotion, true,
                                  line, err, [&] { return ++checks < 4; }));
  CHECK(line.direct.empty() && err == "cancelled");
  CHECK(a2f::GeneratePerformance(*net, pcm.data(), samples, emotion, true, line,
                                 err));
  std::vector<double> times;
  for (int run = 0; run < 21; ++run) {
    const auto start = std::chrono::steady_clock::now();
    CHECK(a2f::GeneratePerformance(*net, pcm.data(), samples, emotion, true,
                                   line, err));
    times.push_back(std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - start)
                        .count());
  }
  std::sort(times.begin(), times.end());
  printf("%s; frontend + inference median %.3f ms for %.3fs audio\n",
         info.c_str(), times[10], samples / 16000.0);
  puts("Ultra context cropping, frozen model parity and cancellation: pass");
}
