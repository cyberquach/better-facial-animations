// The performance network's audio frontend: 16 kHz log-mel frames at 30 fps,
// matching the training frontend (tools/export_performance_stats.py).
#ifndef INCLUDE_A2F_INFERENCE_PERFORMANCE_FEATURES_H_
#define INCLUDE_A2F_INFERENCE_PERFORMANCE_FEATURES_H_

#include <cstddef>
#include <functional>
#include <vector>

namespace a2f {

// mel receives kMelBins values per frame, ceil(n / 16000 * 30) frames. False
// when allowed() returns false.
bool PerformanceFeatures(const float* pcm, size_t n, std::vector<float>& mel,
                         const std::function<bool()>& allowed = {});

}  // namespace a2f

#endif  // INCLUDE_A2F_INFERENCE_PERFORMANCE_FEATURES_H_
