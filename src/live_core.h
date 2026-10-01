// Voice audio -> Audio2Face network -> SDK skin animator + folded retarget
// solve -> 30 fps uint8 morph curves.
#ifndef SRC_LIVE_CORE_H_
#define SRC_LIVE_CORE_H_

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace a2f {

constexpr int kNumChannels = 27;  // 26 solved + the expression curve
constexpr int kNumSolve = 26;
constexpr int kNumMouth = 16;
constexpr int kNumMod = 10;
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
constexpr int kNumFeat = 7;
extern const char* const kChannelNames[kNumSolve];
// Skyrim emotion type: 0 Neutral 1 Anger 2 Disgust 3 Fear 4 Sad 5 Happy
// 6 Surprise 7 Puzzled

// Everything adjustable at runtime; defaults come from A2FLive.dat.
struct Settings {
  // Audio2Face (SDK model config)
  float input_strength, upper_smoothing, lower_smoothing, upper_face_strength,
      lower_face_strength, skin_strength, tongue_strength;
  // retarget solve
  float strength, ridge, smooth, feature_weight;
  bool tongue_on;
  float tip_up_full, tip_up_min, th_fwd, tongue_share;
  // emotion -> A2F explicit emotion, expression morph
  float a2f_floor, a2f_gain, a2f_cap, expr_floor, expr_gain, onset, attack,
      release, emphasis;
  bool emotion_to_a2f = true, expression_on = true;
  // per channel
  float gain[kNumSolve], gamma[kNumSolve], offset[kNumSolve], max[kNumSolve],
      cost[kNumSolve];
  bool solve[kNumSolve];
};

// Row-major float matrix.
struct Array {
  uint32_t rows = 0, cols = 0;
  std::vector<float> values;
  const float* operator[](size_t r) const { return values.data() + r * cols; }
};

// The arrays in A2FLive.dat and the default Settings read from them.
struct Data {
  std::map<std::string, Array> arrays;
  Settings defaults{};
  bool Load(const std::wstring& path, std::string& err);
  // The named array if it has the given shape, else nullptr with err set.
  const Array* Get(const std::string& name, uint32_t rows, uint32_t cols,
                   std::string& err) const;
};

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

// Generated curves: frames x kNumChannels; expr is the engine expression
// index or 255. direct: frames x direct_dim A2F Direct coefficients (empty
// unless Generate was given a DirectProj).
struct Line {
  uint32_t frames = 0, expr = 255, direct_dim = 0;
  std::vector<uint8_t> data;
  std::vector<float> direct;
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
  double net = 0, solve = 0;
};

// emotion: Skyrim emotion type (0..7) and value (0..100). female selects the
// head the curves are solved for. direct, when given, also fills out.direct
// and holds the network's emotion constant over the line.
bool Generate(const Data& d, Net& net, const Settings& s, const float* pcm,
              size_t n, int emotion, int value, bool female, Line& out,
              std::string& err, Timing* timing = nullptr,
              const DirectProj* direct = nullptr);

}  // namespace a2f

#endif  // SRC_LIVE_CORE_H_
