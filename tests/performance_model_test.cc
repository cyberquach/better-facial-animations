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
  int calls = 0, line_calls = 0;
  bool reject = false, nonfinite = false;
  int64_t identity = -1;
  std::vector<std::vector<float>> mels, emotions;  // the windows seen
  int Version() const override { return a2f::kNetPerformance; }
  int Batch() const override { return 1; }
  bool Run(const float*, const float*, int, float*, std::string&) override {
    return false;
  }
  bool RunPerformance(const float* mel, int64_t id, const float* emotion,
                      const a2f::PerformanceState& state, float* motion,
                      a2f::PerformanceState& next, std::string&) override {
    if (!state.frames) line_calls = 0;
    if (state.frames != std::min<int64_t>(a2f::kPerformanceMemoryFrames,
                                          line_calls * kHop)) return false;
    if (state.frames && state.keys[0] != float(line_calls)) return false;
    next.frames =
        std::min<int64_t>(a2f::kPerformanceMemoryFrames, state.frames + kHop);
    next.keys[0] = float(++line_calls);
    ++calls;
    identity = id;
    if (reject) return false;
    mels.emplace_back(mel, mel + kWindow * kMel);
    emotions.emplace_back(emotion, emotion + kWindow * kMod);
    std::fill_n(motion, kHop * kChannels, 0.f);
    for (int slot = 0; slot < kHop; ++slot)
      motion[slot * kChannels] = static_cast<float>(slot);
    if (nonfinite) motion[1] = NAN;
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
  // Output already excludes context: frame f comes from slot f - start.
  for (int frame = 0; frame < frames; ++frame) {
    const float slot = static_cast<float>(frame % kHop);
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

std::wstring FullPath(const char* path) {
  wchar_t full[32768];
  GetFullPathNameW(Wide(path).c_str(), 32768, full, nullptr);
  return full;
}

int OutputBuffers(a2f::Net& net) {
  std::vector<float> mel(kWindow * kMel), emotion(kWindow * kMod),
      motion(kHop * kChannels), expected_motion(motion.size());
  a2f::PerformanceState state, next, expected;
  std::string err;
  CHECK(net.RunPerformance(mel.data(), 0, emotion.data(), state,
                            expected_motion.data(), expected, err));
  CHECK(state.frames == 0);
  for (const auto* buffer : {&state.keys, &state.values})
    CHECK(std::all_of(buffer->begin(), buffer->end(),
                      [](float x) { return x == 0; }));

  // Same state object for input/output must still read the complete old state.
  CHECK(net.RunPerformance(mel.data(), 0, emotion.data(), state,
                            motion.data(), state, err));
  CHECK(motion == expected_motion && state.keys == expected.keys &&
        state.values == expected.values && state.frames == expected.frames);

  // Partial input overlap and output/output overlap retain the old copy order.
  state = a2f::PerformanceState();
  CHECK(net.RunPerformance(mel.data(), 0, emotion.data(), state,
                            state.keys.data() + 1, next, err));
  CHECK(std::equal(expected_motion.begin(), expected_motion.end(),
                   state.keys.begin() + 1));
  CHECK(next.keys == expected.keys && next.values == expected.values);
  state = a2f::PerformanceState();
  CHECK(net.RunPerformance(mel.data(), 0, emotion.data(), state,
                            next.keys.data() + 1, next, err));
  CHECK(next.keys == expected.keys && next.values == expected.values);
  CHECK(net.RunPerformance(mel.data(), 0, emotion.data(), state,
                            mel.data() + 1, next, err));
  CHECK(std::equal(expected_motion.begin(), expected_motion.end(),
                   mel.begin() + 1));
  CHECK(next.keys == expected.keys && next.values == expected.values);

  // A failed validation may change scratch outputs, but never the input state
  // or the frame counter. The same session must recover on the next valid run.
  std::fill(mel.begin(), mel.end(), 0.f);
  mel[0] = NAN;
  state = expected;
  next.frames = 17;
  CHECK(!net.RunPerformance(mel.data(), 0, emotion.data(), state,
                             motion.data(), next, err));
  CHECK(state.keys == expected.keys && state.values == expected.values &&
        state.frames == expected.frames && next.frames == 17 && !net.failed);
  CHECK(err.find("non-finite") != std::string::npos);
  mel[0] = 0;
  state = a2f::PerformanceState();
  CHECK(net.RunPerformance(mel.data(), 0, emotion.data(), state,
                            motion.data(), next, err));
  CHECK(motion == expected_motion && next.keys == expected.keys &&
        next.values == expected.values && next.frames == expected.frames);
  puts("performance output buffers, alias fallback and failure recovery: pass");
  return 0;
}

int PackagedParity() {
  a2f::OrtOptions options;
  options.gpu = false;
  std::string err, info;
  auto net = a2f::CreatePerformanceNet(
      FullPath("build/mod/SKSE/Plugins/BetterFacialAnimations"),
      FullPath("dist/performance.onnx"), options, err, info);
  if (!net) fprintf(stderr, "%s\n", err.c_str());
  CHECK(net);
  if (OutputBuffers(*net)) return 1;
  FILE* input = fopen("tests/performance.fixtures", "rb");
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
  // Repeat and change identities on the same session: no cache leakage.
  for (int run = 0; run < 4; ++run) {
    const int identity = run % 2;
    CHECK(a2f::GeneratePerformance(*net, pcm.data(), samples, emotion,
                                    identity == 0, line, err));
    CHECK(line.direct.size() == expected[identity].size());
    double worst = 0;
    for (size_t i = 0; i < line.direct.size(); ++i)
      worst = std::max(
          worst, double(std::abs(line.direct[i] - expected[identity][i])));
    printf("packaged identity%d run%d max error %.9g\n", identity, run, worst);
    CHECK(worst < .002);
  }
  int checks = 0;
  CHECK(!a2f::GeneratePerformance(*net, pcm.data(), samples, emotion, true,
                                  line, err,
                                  [&] { return ++checks < int(frames) + 2; }));
  CHECK(line.direct.empty());
  CHECK(a2f::GeneratePerformance(*net, pcm.data(), samples, emotion, true, line,
                                 err));
  for (size_t i = 0; i < line.direct.size(); ++i)
    CHECK(std::abs(line.direct[i] - expected[0][i]) < .002);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  FakeNet fake;
  std::string err;
  a2f::Line line;
  std::vector<float> pcm(160017, .01f);  // 10.001 s: four windows, rolls KV
  const a2f::Emotion emotion{1, 100, 1, .25};
  CHECK(a2f::GeneratePerformance(fake, pcm.data(), pcm.size(), emotion, true,
                                 line, err));
  CHECK(fake.calls == 4 && fake.identity == 0 && line.frames == 301);
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
  if (PackagedParity()) return 1;
  puts("performance generation, cancellation, state reset and packaged parity: pass");
}
