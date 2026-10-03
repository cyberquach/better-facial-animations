// The CPU session (GPU=0 and the CPU fallback) loads and runs the shipped
// FP16 network, and agrees with the FP32 network.
// usage: ort_net_test <plugin dir> <model> [FP32 reference model]
#include "../src/ort_net.h"

#include <windows.h>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {
int failures = 0;
void Expect(bool ok, const char* label) {
  printf("%s %s\n", ok ? "ok" : "FAIL", label);
  if (!ok) ++failures;
}
std::wstring FullPath(const char* path) {
  wchar_t w[MAX_PATH], full[MAX_PATH];
  MultiByteToWideChar(CP_ACP, 0, path, -1, w, MAX_PATH);
  GetFullPathNameW(w, MAX_PATH, full, nullptr);
  return full;
}
constexpr int kBatch = 2;
// One batch of a 220 Hz tone with a neutral emotion; empty on failure.
std::vector<float> Run(const std::wstring& dir, const std::wstring& model,
                       const char* label) {
  a2f::OrtOptions o;
  o.gpu = false;
  o.batch = kBatch;
  std::string err, info;
  auto net = a2f::CreateOrtNet(dir, model, o, err, info);
  Expect(net != nullptr, label);
  if (!net) {
    printf("  %s\n", err.c_str());
    return {};
  }
  printf("  %s\n", info.c_str());
  std::vector<float> x(kBatch * a2f::kWindow), emo(kBatch * a2f::kNumEmo),
      y(kBatch * a2f::kNumOut);
  for (size_t i = 0; i < x.size(); ++i)
    x[i] = 0.3f * static_cast<float>(
                      std::sin(i * 2 * 3.14159265 * 220 / a2f::kSampleRate));
  if (!net->Run(x.data(), emo.data(), kBatch, y.data(), err) || net->failed) {
    printf("  %s\n", err.c_str());
    return {};
  }
  return y;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    printf("usage: ort_net_test <plugin dir> <model> [FP32 model]\n");
    return 2;
  }
  const std::wstring dir = FullPath(argv[1]);
  std::vector<float> y = Run(dir, FullPath(argv[2]), "CPU session loads");
  Expect(!y.empty(), "CPU session runs");
  bool finite = !y.empty();
  float lo = 0, hi = 0;
  for (float v : y) {
    finite = finite && std::isfinite(v);
    lo = std::fmin(lo, v);
    hi = std::fmax(hi, v);
  }
  Expect(finite && hi - lo > 1e-3f, "output is finite and not constant");
  if (argc > 3 && !y.empty()) {
    std::vector<float> ref =
        Run(dir, FullPath(argv[3]), "FP32 reference loads");
    double sum = 0, max = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
      double d = std::fabs(static_cast<double>(y[i]) - ref[i]);
      sum += d;
      max = std::fmax(max, d);
    }
    double mean = ref.empty() ? 1e9 : sum / ref.size();
    printf("  FP16 vs FP32: mean %.3g, max %.3g (range %.3g..%.3g)\n", mean,
           max, lo, hi);
    // A tone is harder on FP16 than speech (speech: mean 0.0006, max 0.05 on
    // outputs spanning +-50); this tone gives 0.07 % / 3.6 % of the range.
    Expect(mean < 0.003 * (hi - lo) && max < 0.08 * (hi - lo),
           "FP16 output matches FP32");
  }
  // Must not crash without a GPU; under Wine it may know nothing.
  a2f::GpuState g = a2f::QueryGpu();
  printf("  adapter 0: \"%s\", %.0f MB dedicated, %.0f / %.0f MB in use%s\n",
         g.adapter.c_str(), g.dedicated_mb, g.used_mb, g.budget_mb,
         g.known ? "" : " (unknown)");
  printf(failures ? "%d FAILED\n" : "all ok\n", failures);
  return failures != 0;
}
