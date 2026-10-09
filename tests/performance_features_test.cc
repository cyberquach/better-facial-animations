// Writes the native mel features of a float32 PCM file, for comparison with
// the training frontend. Build with src/inference/performance_features.cc.
#include "a2f/inference/performance_features.h"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <vector>

int main(int argc, char** argv) {
  if (argc != 3) return 2;
  std::ifstream in(argv[1], std::ios::binary);
  std::vector<char> raw((std::istreambuf_iterator<char>(in)), {});
  std::vector<float> pcm(raw.size() / sizeof(float)), mel;
  std::copy(raw.begin(), raw.end(), reinterpret_cast<char*>(pcm.data()));
  if (!a2f::PerformanceFeatures(pcm.data(), pcm.size(), mel)) return 1;
  std::ofstream out(argv[2], std::ios::binary);
  out.write(reinterpret_cast<const char*>(mel.data()),
            mel.size() * sizeof(float));
}
