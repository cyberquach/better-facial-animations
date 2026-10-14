// Performance model chunking and cancellation against a fake network; with
// fixtures, the shipped ONNX model against the Python reference.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "../src/plugin/better_facial_animations.cc"
#include "a2f/inference/performance_features.h"
#include "a2f/inference/performance_stats.h"

#define CHECK(x)                                          \
  do {                                                    \
    if (!(x)) {                                           \
      fprintf(stderr, "line %d: %s\n", __LINE__, #x);     \
      return 1;                                           \
    }                                                     \
  } while (0)

namespace {

constexpr int kWindow = a2f::kPerformanceWindowFrames;
constexpr int kContext = a2f::kPerformanceContextFrames;
constexpr int kHop = a2f::kPerformanceHopFrames;
constexpr int kChannels = a2f::kPerformanceChannels;
constexpr int kMel = a2f::kMelBins;
constexpr int kMod = a2f::kNumMod;

// Channel 0 of each output slot carries the slot's index, so the test can
// see which slots a frame is taken from; the rest stay at the mean.
struct FakeNet : a2f::Net {
  int calls = 0;
  bool reject = false, nonfinite = false;
  int64_t identity = -1;
  std::vector<std::vector<float>> mels, emotions;  // the windows seen
  int Version() const override { return a2f::kNetPerformance; }
  int Batch() const override { return 1; }
  bool Run(const float*, const float*, int, float*, std::string&) override {
    return false;
  }
  bool RunPerformance(const float* mel, int64_t id, const float* emotion,
                      float* motion, std::string&) override {
    ++calls;
    identity = id;
    if (reject) return false;
    mels.emplace_back(mel, mel + kWindow * kMel);
    emotions.emplace_back(emotion, emotion + kWindow * kMod);
    std::fill_n(motion, kWindow * kChannels, 0.f);
    for (int slot = 0; slot < kWindow; ++slot)
      motion[slot * kChannels] = static_cast<float>(slot);
    if (nonfinite) motion[kContext * kChannels + 1] = NAN;
    return std::isfinite(mel[0]) && std::isfinite(emotion[0]);
  }
};

bool Near(float a, float b) { return std::abs(a - b) <= 1e-5f * (1 + std::abs(b)); }

// Every window the fake saw against the frontend and emotion track computed
// here: frames outside the clip are silence, emotion holds its ends.
int CheckWindows(const FakeNet& fake, const std::vector<float>& pcm,
                 const a2f::Emotion& emotion, const a2f::Line& line) {
  std::vector<float> mel, track;
  CHECK(a2f::PerformanceFeatures(pcm.data(), pcm.size(), mel));
  const int frames = static_cast<int>(mel.size() / kMel);
  CHECK(frames == static_cast<int>(line.frames));
  a2f::EmotionTrack(pcm.data(), pcm.size(), frames, emotion, track);
  CHECK(fake.mels.size() == size_t((frames + kHop - 1) / kHop));
  for (size_t call = 0; call < fake.mels.size(); ++call)
    for (int slot = 0; slot < kWindow; ++slot) {
      const int frame = static_cast<int>(call) * kHop + slot - kContext;
      const int clamped = std::clamp(frame, 0, frames - 1);
      for (int bin = 0; bin < kMel; ++bin) {
        const float raw = frame >= 0 && frame < frames
                              ? mel[frame * kMel + bin]
                              : std::log(1e-6f);
        CHECK(Near(fake.mels[call][slot * kMel + bin],
                   (raw - a2f::performance::kMelMean[bin]) /
                       a2f::performance::kMelStd[bin]));
      }
      for (int mod = 0; mod < kMod; ++mod)
        CHECK(fake.emotions[call][slot * kMod + mod] ==
              track[clamped * kMod + mod]);
    }
  // Frame f comes from slot (f - start) + context, denormalized.
  for (int frame = 0; frame < frames; ++frame) {
    const float slot = static_cast<float>(frame % kHop + kContext);
    CHECK(Near(line.direct[frame * kChannels],
               slot * a2f::performance::kTargetStd[0] +
                   a2f::performance::kTargetMean[0]));
    CHECK(line.direct[frame * kChannels + 1] ==
          a2f::performance::kTargetMean[1]);
  }
  for (int mod = 0; mod < kMod; ++mod) {
    float peak = 0, mean = 0;
    for (int frame = 0; frame < frames; ++frame) {
      peak = std::max(peak, track[frame * kMod + mod]);
      mean += track[frame * kMod + mod] / frames;
    }
    CHECK(line.emotion_peak[mod] == peak && Near(line.emotion_mean[mod], mean));
  }
  return 0;
}

std::vector<float> ReadFloats(const char* file) {
  std::ifstream in(file, std::ios::binary);
  std::vector<char> raw((std::istreambuf_iterator<char>(in)), {});
  std::vector<float> out(raw.size() / sizeof(float));
  memcpy(out.data(), raw.data(), out.size() * sizeof(float));
  return out;
}

std::wstring Wide(const char* s) { return std::wstring(s, s + strlen(s)); }

}  // namespace

int main(int argc, char** argv) {
  FakeNet fake;
  std::string err;
  a2f::Line line;
  std::vector<float> pcm(64017, .01f);  // 4.001 s: two windows
  const a2f::Emotion emotion{1, 100, 1, .25};
  CHECK(a2f::GeneratePerformance(fake, pcm.data(), pcm.size(), emotion, true,
                                 line, err));
  CHECK(fake.calls == 2 && fake.identity == 0 && line.frames == 121);
  CHECK(line.fps == a2f::kPerformanceFps && line.direct_dim == kChannels &&
        line.eyes.empty());
  if (CheckWindows(fake, pcm, emotion, line)) return 1;
  // Not all zero: the frames were made from the audio.
  CHECK(std::any_of(fake.mels[0].begin(), fake.mels[0].end(),
                    [](float v) { return std::abs(v) > 1e-3f; }));
  fake.nonfinite = true;
  CHECK(!a2f::GeneratePerformance(fake, pcm.data(), pcm.size(), emotion, true,
                                  line, err));
  CHECK(err == "nonfinite performance output" && line.direct.empty());
  fake.nonfinite = false;
  fake.calls = 0;
  CHECK(!a2f::GeneratePerformance(fake, pcm.data(), pcm.size(), emotion, true,
                                  line, err, [] { return false; }));
  CHECK(fake.calls == 0 && line.direct.empty());
  CHECK(!a2f::GeneratePerformance(fake, pcm.data(), pcm.size(), emotion, false,
                                  line, err, [&] { return fake.calls < 1; }));
  CHECK(fake.calls == 1 && line.direct.empty());
  fake.reject = true;
  CHECK(!a2f::GeneratePerformance(fake, pcm.data(), pcm.size(), emotion, false,
                                  line, err));
  CHECK(line.direct.empty());
  fake.reject = false;
  CHECK(a2f::GeneratePerformance(fake, pcm.data(), 1, emotion, false, line,
                                 err));
  CHECK(line.frames == 1 && fake.identity == 1);
  fake.mels.clear();
  fake.emotions.clear();
  CHECK(a2f::GeneratePerformance(fake, pcm.data(), 1, emotion, false, line,
                                 err));
  if (CheckWindows(fake, std::vector<float>(pcm.begin(), pcm.begin() + 1),
                   emotion, line))
    return 1;
  CHECK(!a2f::GeneratePerformance(fake, pcm.data(), 0, emotion, false, line,
                                  err));
  if (argc == 3 || argc == 6) {
    a2f::OrtOptions options;
    options.gpu = false;
    std::string info;
    auto net = a2f::CreatePerformanceNet(Wide(argv[1]), Wide(argv[2]), options,
                                         err, info);
    if (!net) fprintf(stderr, "%s\n", err.c_str());
    CHECK(net && net->Version() == a2f::kNetPerformance);
    if (argc == 6) pcm = ReadFloats(argv[3]);
    CHECK(!pcm.empty());
    for (int identity = 0; identity < 2; ++identity) {
      CHECK(a2f::GeneratePerformance(*net, pcm.data(), pcm.size(), emotion,
                                     identity == 0, line, err));
      CHECK(!line.direct.empty());
      for (float value : line.direct) CHECK(std::isfinite(value));
      if (argc == 3) continue; // smoke test without optional parity fixtures
      const auto expected = ReadFloats(argv[4 + identity]);
      CHECK(line.direct.size() == expected.size());
      double max_error = 0;
      for (size_t i = 0; i < expected.size(); ++i)
        max_error = std::max(
            max_error, double(std::abs(expected[i] - line.direct[i])));
      printf("identity%d native/Python max error %.9g\n", identity, max_error);
      CHECK(max_error < .002);
    }
    std::vector<double> ms;
    for (int run = 0; run < 21; ++run) {
      const auto start = std::chrono::steady_clock::now();
      CHECK(a2f::GeneratePerformance(*net, pcm.data(), pcm.size(), emotion,
                                     true, line, err));
      ms.push_back(std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - start)
                       .count());
    }
    std::sort(ms.begin(), ms.end());
    printf("CPU frontend+inference median %.3f ms for %.3fs audio\n", ms[10],
           pcm.size() / double(a2f::kSampleRate));
    puts(info.c_str());
  }
  puts("performance generation, cancellation and output: pass");
}
