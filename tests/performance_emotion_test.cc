#include <windows.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "a2f/inference/ort_net.h"
#include "a2f/inference/performance_features.h"
#define CHECK(x)                                  \
  do {                                            \
    if (!(x)) {                                   \
      printf("FAIL line %d: %s\n", __LINE__, #x); \
      return 1;                                   \
    }                                             \
  } while (0)

std::wstring FullPath(const char* path) {
  wchar_t text[32768], full[32768];
  MultiByteToWideChar(CP_UTF8, 0, path, -1, text, 32768);
  GetFullPathNameW(text, 32768, full, nullptr);
  return full;
}
int main(int argc, char** argv) {
  a2f::OrtOptions options;
  options.gpu = false;
  std::string err;
  auto net = a2f::CreateOrtEmotionNet(
      FullPath("build/mod/SKSE/Plugins/BetterFacialAnimations"),
      FullPath("dist/emotion_performance.onnx"), options, err, true);
  if (!net) printf("%s\n", err.c_str());
  CHECK(net);
  FILE* input =
      fopen(argc > 1 ? argv[1] : "tests/emotion_performance.fixtures", "rb");
  CHECK(input);
  unsigned count = 0;
  CHECK(fread(&count, sizeof count, 1, input) == 1 && count > 0 &&
        count <= 128);
  float worst_mel = 0, worst_logit = 0;
  for (unsigned sample = 0; sample < count; ++sample) {
    std::vector<float> pcm(a2f::kVoiceWindow), reference(48 * 188), mel;
    float expected[a2f::kNumVoiceEmo], actual[a2f::kNumVoiceEmo];
    CHECK(fread(pcm.data(), sizeof(float), pcm.size(), input) == pcm.size());
    CHECK(fread(reference.data(), sizeof(float), reference.size(), input) ==
          reference.size());
    CHECK(fread(expected, sizeof(float), a2f::kNumVoiceEmo, input) ==
          a2f::kNumVoiceEmo);
    a2f::EmotionFeatures(pcm.data(), mel);
    CHECK(mel.size() == reference.size());
    for (size_t i = 0; i < mel.size(); ++i) {
      CHECK(std::isfinite(mel[i]));
      worst_mel = std::max(worst_mel, std::abs(mel[i] - reference[i]));
    }
    CHECK(net->Run(pcm.data(), actual, err));
    for (int i = 0; i < a2f::kNumVoiceEmo; ++i) {
      CHECK(std::isfinite(actual[i]));
      worst_logit = std::max(worst_logit, std::abs(actual[i] - expected[i]));
    }
  }
  CHECK(fgetc(input) == EOF);
  fclose(input);
  printf(
      "%u cases: frontend max error %.7f; native ONNX logits max error %.7f\n",
      count, worst_mel, worst_logit);
  CHECK(worst_mel < .025f && worst_logit < .01f);
  // Double-precision centering avoids float-reduction residue on constant DC.
  std::vector<float> dc(a2f::kVoiceWindow, .1f), silence(a2f::kVoiceWindow), a,
      b;
  a2f::EmotionFeatures(dc.data(), a);
  a2f::EmotionFeatures(silence.data(), b);
  CHECK(a == b);
  puts(
      "Performance Audio2Emotion frontend and packaged checkpoint parity: "
      "pass");
}
