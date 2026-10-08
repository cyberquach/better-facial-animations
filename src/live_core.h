// Voice audio -> Audio2Face network -> A2F Direct coefficients: 30 fps with
// v2.3, 60 fps with v3.0.
#ifndef SRC_LIVE_CORE_H_
#define SRC_LIVE_CORE_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
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

// Audio2Face-3D v3.0, folded (tools/a2f3_fold.py): a 1 s window every 0.5 s,
// the GRU state chained between windows, 30 frames at 60 fps per window.
constexpr int kFps3 = 60;
constexpr int kWindow3 = 16000;
constexpr int kStride3 = 8000;
constexpr int kFrames3 = 30;  // frames kept per window
constexpr int kSkip3 = 45;    // frames emitted before the clip starts
constexpr int kLatent3 = 512;
constexpr int kJawEyes3 = kNumJaw + 4;  // the jaw's values, then the eyes
constexpr int kNumOut3 = 2 * kLatent3 + kJawEyes3;
constexpr int kState3 = 2 * 2 * 256;
constexpr int kNumIdentity3 = 3;  // 0 Claire, 1 James, 2 Mark

// The network. v2.3 (Run): x (b, 8320) audio windows, emo (b, 26) -> out
// (b, 301): the skin's kNumPca PCA coefficients, the tongue's kNumTongue,
// kNumJaw jaw values and the eyes (not used). v3.0 (Run3): one window
// (kWindow3), the identity one-hot (kNumIdentity3), the emotion (kFrames3 x
// kNumMod) and the GRU state (kState3) -> the skin and tongue latents
// (kFrames3 x kLatent3 each), the jaw and eyes (kFrames3 x kJawEyes3) and the
// next state.
// Implemented by OrtNet and OrtNet3 (ort_net.cc).
class Net {
 public:
  virtual ~Net() = default;
  // 3: the folded v3.0 network, run with Run3; else v2.3, run with Run.
  virtual int Version() const { return 2; }
  virtual bool Run3(const float* window, const float* identity,
                    const float* emotion, const float* state, float* skin,
                    float* tongue, float* jaw_eyes, float* next_state,
                    std::string& err) {
    err = "not a v3.0 network";
    return false;
  }
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
  float fps = kFps;  // of direct
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

// Conservative silence gate, not speech recognition: any 20 ms window
// with AC RMS above -70 dBFS is usable. Rejects digital silence, tiny codec
// noise and DC while retaining quiet/short speech anywhere in the clip.
bool HasAudibleAudio(const float* pcm, size_t n);

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
// Both generators check allowed before each network run, if supplied.
// Cancellation returns false; discard the partial output.
bool Generate(Net& net, const float* pcm, size_t n, const Emotion& emotion,
              const float* proj, const float* rest, uint32_t k, Line& out,
              std::string& err, const std::function<bool()>& allowed = {});

// v3.0: the same coefficients from the folded network's output, with d = y -
// rest (rest: kNumOut3): the skin's d[0:kLatent3] . proj[k], the tongue's
// d[kLatent3 : 2 kLatent3] . tongue_proj[j] (kNumTongue x kLatent3), then the
// jaw's d[2 kLatent3 : + kNumJaw].
struct Face3 {
  int identity;  // 0 Claire, 2 Mark
  const float* proj;
  const float* tongue_proj;
  const float* rest;
  uint32_t k;
};
bool Generate3(Net& net, const float* pcm, size_t n, const Emotion& emotion,
               const Face3& face, Line& out, std::string& err,
               const std::function<bool()>& allowed = {});

// The emotion alone: frames (at kFps) x kNumMod.
void EmotionTrack(const float* pcm, size_t n, int frames,
                  const Emotion& emotion, std::vector<float>& track);

}  // namespace a2f

#endif  // SRC_LIVE_CORE_H_
