// Voice audio -> Audio2Face network -> SDK skin animator + folded retarget solve -> 30 fps uint8 morph curves.
#pragma once
#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <map>

namespace a2f {

constexpr int NCH = 27, NSOLVE = 26, NMOUTH = 16, NMOD = 10, FPS = 30, SR = 16000, WIN = 8320, WIN_OFS = 4160;
constexpr int NPCA = 272, NTONGUE = 10, NRAW = 282, NOUT = 301, NEMO = 26, NIMPLICIT = 16, NFEAT = 7;
extern const char* const CHANNEL_NAMES[NSOLVE];
// Skyrim emotion type: 0 Neutral 1 Anger 2 Disgust 3 Fear 4 Sad 5 Happy 6 Surprise 7 Puzzled

// Everything adjustable at runtime; defaults come from A2FLive.dat.
struct Settings {
    // Audio2Face (SDK model config)
    float inputStrength, upperSmoothing, lowerSmoothing, upperFaceStrength, lowerFaceStrength, skinStrength, tongueStrength;
    // retarget solve
    float strength, ridge, smooth, featureWeight;
    bool tongueOn; float tipUpFull, tipUpMin, thFwd, tongueShare;
    // emotion -> A2F explicit emotion, expression morph
    float a2fFloor, a2fGain, a2fCap, exprFloor, exprGain, onset, attack, release, emphasis;
    bool emotionToA2F = true, expressionOn = true;
    // per channel
    float gain[NSOLVE], gamma[NSOLVE], offset[NSOLVE], max[NSOLVE], cost[NSOLVE]; bool solve[NSOLVE];
};

struct Array { uint32_t rows = 0, cols = 0; std::vector<float> v; const float* operator[](size_t r) const { return v.data() + r * cols; } };

struct Data {
    std::map<std::string, Array> a;
    Settings defaults{};
    bool Load(const std::wstring& path, std::string& err);
    const Array* Get(const std::string& name, uint32_t rows, uint32_t cols, std::string& err) const;
};

// The network: x (b, 8320) audio windows, emo (b, 26) -> out (b, 301). Implemented by OrtNet (ort_net.cpp).
struct Net {
    virtual ~Net() = default;
    virtual int Batch() const = 0;  // preferred batch; Generate pads the last batch to it (one shape, one DML plan)
    virtual bool Run(const float* x, const float* emo, int b, float* out, std::string& err) = 0;
};

struct Line { uint32_t frames = 0, expr = 255; std::vector<uint8_t> data; };  // frames x NCH; expr: engine index or 255

// Decode .fuz / .xwm / .wav bytes to 16 kHz mono float (same conversion as `ffmpeg -ac 1 -ar 16000` to s16).
bool DecodeAudio(const uint8_t* data, size_t n, std::vector<float>& pcm, std::string& err);

struct Timing { double net = 0, solve = 0; };
// emotion: Skyrim emotion type (0..7) and value (0..100). female selects the head the curves are solved for.
bool Generate(const Data& d, Net& net, const Settings& s, const float* pcm, size_t n, int emotion, int value, bool female,
              Line& out, std::string& err, Timing* timing = nullptr);

}  // namespace a2f
