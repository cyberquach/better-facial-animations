// Voice audio -> Audio2Face network -> 30 fps A2F Direct coefficients.
#ifndef SRC_LIVE_CORE_H_
#define SRC_LIVE_CORE_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace a2f {

constexpr int kFps = 30;
constexpr int kSampleRate = 16000;
constexpr int kWindow = 8320;
constexpr int kWindowOffset = 4160;
constexpr int kNumPca = 272;
constexpr int kNumTongue = 10;  // after the skin's: tongue PCA coefficients
constexpr int kNumJaw = 15;     // then 5 jaw points' displacements
constexpr int kNumOut = 301;
constexpr int kNumEmo = 26;
constexpr int kNumImplicit = 16;
constexpr int kNumMod = 10;  // explicit emotions
// Skyrim emotion type: 0 Neutral 1 Anger 2 Disgust 3 Fear 4 Sad 5 Happy
// 6 Surprise 7 Puzzled

// The network: x (b, 8320) audio windows, emo (b, 26) -> out (b, 301): the
// skin's kNumPca PCA coefficients, the tongue's kNumTongue, kNumJaw jaw
// values and the eyes (not used).
// Implemented by OrtNet (ort_net.cc).
class Net {
 public:
  virtual ~Net() = default;
  // Preferred batch; Generate pads the last batch to it (one shape, one DML
  // plan).
  virtual int Batch() const = 0;
  virtual bool Run(const float* x, const float* emo, int b, float* out,
                   std::string& err) = 0;
  // Set when a run failed: the session is unusable (e.g. the GPU device was
  // lost) and has to be created again.
  bool failed = false;
};

struct Line {
  uint32_t frames = 0, direct_dim = 0;
  std::vector<float> direct;
  // log: the emotion the network was given, per explicit emotion (amazement
  // anger cheekiness disgust fear grief joy outofbreath pain sadness)
  float emotion_peak[kNumMod] = {}, emotion_mean[kNumMod] = {};
};

// A per-frame track (frames > 0) at fractional frame f, clamped to the track:
// frames i0 and i1 blended by a.
struct FrameSample {
  uint32_t i0, i1;
  float a;
};
inline FrameSample SampleFrame(float f, uint32_t frames) {
  const uint32_t last = frames - 1;
  f = std::clamp(f, 0.f, static_cast<float>(last));
  const uint32_t i0 = static_cast<uint32_t>(f);
  return {i0, std::min(i0 + 1, last), f - static_cast<float>(i0)};
}

// Smoothstep of x clamped to 0..1.
template <typename T>
T Smoothstep(T x) {
  T c = std::clamp(x, T(0), T(1));
  return c * c * (3 - 2 * c);
}

// Decode .fuz / .xwm / .wav bytes to 16 kHz mono float (same conversion as
// `ffmpeg -ac 1 -ar 16000` to s16).
bool DecodeAudio(const uint8_t* data, size_t n, std::vector<float>& pcm,
                 std::string& err);

// The emotion the network is given with a line, per frame:
//   gain x the game's x (1 + swell x loudness)
// for each explicit emotion, at most kMaxEmotion. The game's is the line's
// Skyrim emotion type (0..7) at value / 100, constant; loudness is the
// voice's level, 0..1 within the line.
constexpr double kMaxEmotion = 2;
struct Emotion {
  int type = 0, value = 0;
  float gain = 1, swell = 0;
};

// Per frame, with d = y - rest (rest: kNumOut), the k skin coefficients
// d[0:kNumPca] . proj[k] (proj: k x kNumPca), then
// d[kNumPca : kNumPca + kNumTongue + kNumJaw]. out is written anew.
bool Generate(Net& net, const float* pcm, size_t n, const Emotion& emotion,
              const float* proj, const float* rest, uint32_t k, Line& out,
              std::string& err);

// The emotion alone: frames x kNumMod.
void EmotionTrack(const float* pcm, size_t n, int frames,
                  const Emotion& emotion, std::vector<float>& track);

}  // namespace a2f

#endif  // SRC_LIVE_CORE_H_
