#include "a2f/inference/performance_features.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <utility>

#include "a2f/inference/live_core.h"

namespace a2f {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr int kFftSize = 512;
constexpr int kFftBins = kFftSize / 2 + 1;
constexpr int kHopSamples = 160;     // 10 ms: 100 spectra per second
constexpr int kWindowSamples = 400;  // 25 ms Hann window, centred in the FFT
constexpr int kWindowStart = (kFftSize - kWindowSamples) / 2;
constexpr int kSpectraPerSecond = kSampleRate / kHopSamples;
constexpr float kLogFloor = 1e-6f;

template <int Bins>
using MelBank = std::array<float, Bins * kFftBins>;
template <int Bins>
using Spectrum = std::array<float, Bins>;

double MelToHz(double mel) { return 700 * (std::pow(10., mel) - 1); }

// Triangular filters with kMelBins + 2 edges spaced evenly on the mel scale
// from 0 Hz to Nyquist.
template <int Bins>
MelBank<Bins> MakeMelBank() {
  MelBank<Bins> bank{};
  double edges_hz[Bins + 2];
  const double top_mel = std::log10(1 + kSampleRate / 2. / 700);
  for (int edge = 0; edge < Bins + 2; ++edge)
    edges_hz[edge] = MelToHz(edge / (Bins + 1.) * top_mel);
  for (int bin = 0; bin < Bins; ++bin) {
    const double low = edges_hz[bin], centre = edges_hz[bin + 1],
                 high = edges_hz[bin + 2];
    for (int frequency = 0; frequency < kFftBins; ++frequency) {
      const double hz = frequency * static_cast<double>(kSampleRate) / kFftSize;
      const double rising = (hz - low) / (centre - low);
      const double falling = (high - hz) / (high - centre);
      bank[bin * kFftBins + frequency] =
          static_cast<float>(std::max(0., std::min(rising, falling)));
    }
  }
  return bank;
}

// In-place radix-2 Cooley-Tukey transform of kFftSize values.
void Fft(std::complex<double>* samples) {
  for (int i = 1, reversed = 0; i < kFftSize; ++i) {
    int bit = kFftSize / 2;
    for (; reversed & bit; bit >>= 1) reversed ^= bit;
    reversed ^= bit;
    if (i < reversed) std::swap(samples[i], samples[reversed]);
  }
  for (int length = 2; length <= kFftSize; length *= 2) {
    const std::complex<double> root = std::polar(1., -2 * kPi / length);
    for (int block = 0; block < kFftSize; block += length) {
      std::complex<double> twiddle = 1.;
      for (int pair = 0; pair < length / 2; ++pair) {
        const std::complex<double> even = samples[block + pair];
        const std::complex<double> odd =
            samples[block + pair + length / 2] * twiddle;
        samples[block + pair] = even + odd;
        samples[block + pair + length / 2] = even - odd;
        twiddle *= root;
      }
    }
  }
}

// The log-mel spectrum of the window centred on spectrum index `index`,
// padding past either end of the clip with silence.
template <int Bins>
void LogMelSpectrum(const float* pcm, size_t n, size_t index,
                    const MelBank<Bins>& bank, Spectrum<Bins>& out,
                    int hop = kHopSamples) {
  std::array<std::complex<double>, kFftSize> samples{};
  for (int i = 0; i < kWindowSamples; ++i) {
    const int64_t position =
        static_cast<int64_t>(index * hop) + kWindowStart + i - kFftSize / 2;
    if (position < 0 || static_cast<size_t>(position) >= n) continue;
    const float hann =
        .5f - .5f * static_cast<float>(std::cos(2 * kPi * i / kWindowSamples));
    samples[kWindowStart + i] = pcm[position] * hann;
  }
  Fft(samples.data());
  float power[kFftBins];
  for (int frequency = 0; frequency < kFftBins; ++frequency)
    power[frequency] = static_cast<float>(std::norm(samples[frequency]));
  for (int bin = 0; bin < Bins; ++bin) {
    float energy = 0;
    for (int frequency = 0; frequency < kFftBins; ++frequency)
      energy += bank[bin * kFftBins + frequency] * power[frequency];
    out[bin] = std::log(energy + kLogFloor);
  }
}

}  // namespace

// Each 30 fps frame interpolates the two 100 Hz spectra around its time, so
// only those two are kept however long the clip.
bool PerformanceFeatures(const float* pcm, size_t n, std::vector<float>& mel,
                         const std::function<bool()>& allowed) {
  static const auto bank = MakeMelBank<kMelBins>();
  const size_t frames = (n * kPerformanceFps + kSampleRate - 1) / kSampleRate;
  const size_t spectra = n / kHopSamples + 1;
  Spectrum<kMelBins> before{}, after{};
  mel.resize(frames * kMelBins);
  for (size_t frame = 0; frame < frames; ++frame) {
    if (allowed && !allowed()) return false;
    const double position =
        static_cast<double>(frame) * kSpectraPerSecond / kPerformanceFps;
    const size_t first = static_cast<size_t>(position);
    const size_t second = std::min(first + 1, spectra - 1);
    LogMelSpectrum<kMelBins>(pcm, n, first, bank, before);
    LogMelSpectrum<kMelBins>(pcm, n, second, bank, after);
    const float blend = static_cast<float>(position - first);
    for (int bin = 0; bin < kMelBins; ++bin)
      mel[frame * kMelBins + bin] =
          before[bin] * (1 - blend) + after[bin] * blend;
  }
  return true;
}

// Exact student frontend: normalize the complete padded 60000-sample window,
// then 48 HTK log-mels at 20 ms stride, mel-major [48,188].
void EmotionFeatures(const float* pcm, std::vector<float>& mel) {
  static const auto bank = MakeMelBank<48>();
  double mean = 0, variance = 0;
  for (int i = 0; i < kVoiceWindow; ++i) mean += pcm[i];
  mean /= kVoiceWindow;
  for (int i = 0; i < kVoiceWindow; ++i) {
    const double centered = pcm[i] - mean;
    variance += centered * centered;
  }
  const double scale = 1 / std::sqrt(variance / kVoiceWindow + 1e-7);
  std::vector<float> normalized(kVoiceWindow);
  for (int i = 0; i < kVoiceWindow; ++i)
    normalized[i] = static_cast<float>((pcm[i] - mean) * scale);
  mel.resize(48 * 188);
  Spectrum<48> spectrum{};
  for (int frame = 0; frame < 188; ++frame) {
    LogMelSpectrum<48>(normalized.data(), normalized.size(), frame, bank,
                       spectrum, 320);
    for (int bin = 0; bin < 48; ++bin) mel[bin * 188 + frame] = spectrum[bin];
  }
}

}  // namespace a2f
