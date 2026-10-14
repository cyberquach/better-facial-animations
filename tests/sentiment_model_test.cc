#include <windows.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "a2f/inference/ort_net.h"
std::wstring FullPath(const char* path) {
  wchar_t w[1024], full[1024];
  MultiByteToWideChar(CP_UTF8, 0, path, -1, w, 1024);
  GetFullPathNameW(w, 1024, full, nullptr);
  return full;
}
int main(int argc, char** argv) {
  if (argc < 3) return 2;
  a2f::OrtOptions options;
  options.gpu = argc > 4;
  options.independent_device = argc > 5;
  const auto dir = FullPath(argv[1]);
  // Match the production loader; ORT delay-loads DirectML by name.
  if (options.gpu && !LoadLibraryExW((dir + L"\\DirectML.dll").c_str(), nullptr,
                                    LOAD_WITH_ALTERED_SEARCH_PATH)) {
    puts("FAIL: bundled DirectML could not load");
    return 1;
  }
  std::vector<float> pcm(a2f::kVoiceWindow);
  constexpr int kCases = 5;
  float output[2][kCases][a2f::kNumVoiceEmo] = {};
  for (int model = 0; model < (argc > 3 ? 2 : 1); ++model) {
    std::string err;
    auto net =
        a2f::CreateOrtEmotionNet(dir, FullPath(argv[2 + model]), options, err);
    if (!net) {
      printf("FAIL: %s\n", err.c_str());
      return 1;
    }
    for (int sample = 0; sample < kCases; ++sample) {
      for (size_t i = 0; i < pcm.size(); ++i) {
        const float wave = std::sin(i * .0864);
        pcm[i] = sample == 0 ? 0.f : sample == 1 ? .1f :
                 sample == 2 ? .00001f * wave :
                 sample == 3 ? .2f * wave : .9f * wave;
      }
      if (!net->Run(pcm.data(), output[model][sample], err)) {
        printf("FAIL case %d: %s\n", sample, err.c_str());
        return 1;
      }
      printf("model %d case %d:", model, sample);
      for (float v : output[model][sample]) {
        if (!std::isfinite(v)) return 1;
        printf(" %.6f", v);
      }
      puts("");
    }
  }
  if (argc > 3) {
    for (int sample = 0; sample < kCases; ++sample)
      for (int i = 0; i < a2f::kNumVoiceEmo; ++i)
        if (std::fabs(output[0][sample][i] - output[1][sample][i]) > .05f) {
          puts("FAIL: reduced-precision logits differ from FP32 by > 0.05");
          return 1;
        }
  }
  puts(argc > 3 ? "Audio2Emotion load, inference and FP16 agreement: pass"
                : "Audio2Emotion load, inference and finite logits: pass");
}
