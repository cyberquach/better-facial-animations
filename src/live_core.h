// Voice audio -> Audio2Face network -> 30 fps A2F Direct coefficients and the
// engine expression curve.
#ifndef SRC_LIVE_CORE_H_
#define SRC_LIVE_CORE_H_

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
constexpr int kNumTongue = 10;
constexpr int kNumRaw = 282;
constexpr int kNumOut = 301;
constexpr int kNumEmo = 26;
constexpr int kNumImplicit = 16;
constexpr int kNumMod = 10;  // explicit emotions
// Skyrim emotion type: 0 Neutral 1 Anger 2 Disgust 3 Fear 4 Sad 5 Happy
// 6 Surprise 7 Puzzled

// The network: x (b, 8320) audio windows, emo (b, 26) -> out (b, 301).
// Implemented by OrtNet (ort_net.cc).
class Net {
 public:
  virtual ~Net() = default;
  // Preferred batch; Generate pads the last batch to it (one shape, one DML
  // plan).
  virtual int Batch() const = 0;
  virtual bool Run(const float* x, const float* emo, int b, float* out,
                   std::string& err) = 0;
};

// One generated line: direct is frames x direct_dim A2F Direct coefficients;
// expr is the engine expression index (255 none) and expression its level
// per frame (empty when none).
struct Line {
  uint32_t frames = 0, expr = 255, direct_dim = 0;
  std::vector<float> direct, expression;
};

// A2F Direct coefficients from the network output y (kNumRaw per frame):
// k skin coefficients (y[0:272] - rest) . proj[k] followed by the 10 tongue
// coefficients y[272 + j] - rest[272 + j]. proj: k x kNumPca, rest: kNumRaw.
struct DirectProj {
  const float* proj;
  const float* rest;
  uint32_t k;
};

// Decode .fuz / .xwm / .wav bytes to 16 kHz mono float (same conversion as
// `ffmpeg -ac 1 -ar 16000` to s16).
bool DecodeAudio(const uint8_t* data, size_t n, std::vector<float>& pcm,
                 std::string& err);

struct Timing {
  double net = 0, post = 0;
};

// emotion: Skyrim emotion type (0..7) and value (0..100), fed to the network
// constant over the line. direct: the speaker's sex model projection.
bool Generate(Net& net, const float* pcm, size_t n, int emotion, int value,
              const DirectProj& direct, Line& out, std::string& err,
              Timing* timing = nullptr);

}  // namespace a2f

#endif  // SRC_LIVE_CORE_H_
