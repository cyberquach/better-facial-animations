// Recovery inside an explicitly selected shared GPU route. No device probes.
#ifndef INCLUDE_A2F_INFERENCE_GPU_RETRY_H_
#define INCLUDE_A2F_INFERENCE_GPU_RETRY_H_
#include <functional>
#include <initializer_list>
#include <string>

namespace a2f {
inline bool RetryableGpuFailure(const std::string& err) {
  for (const char* permanent : {"GPU attempt failed", "independent device:",
       "cannot load onnxruntime.dll", "wrong version", "model file not found",
       "cannot load bundled DirectML.dll"})
    if (err.find(permanent) != std::string::npos) return false;
  return true;
}
inline bool LostDevice(const std::string& err) {
  for (const char* code : {"887A0005", "887A0006", "887A0007", "887A0020"})
    if (err.find(code) != std::string::npos) return true;
  return false;
}
struct SharedGpuSettings {
  bool metacommands = true;
  bool wrapper = false;
};
// Remember successful alternatives only. The callback owns creation/warmup and
// teardown protection; wrapped() reports wrappers observed during that work.
// Works for facial and sentiment sessions, without switching GPU routes.
template <typename Create, typename Wrapped>
auto CreateSharedGpu(const Create& create, SharedGpuSettings& settings,
                     const Wrapped& wrapped, std::string& err,
                     const std::function<void(const std::string&)>& log = {})
    -> decltype(create(true, false, err)) {
  auto say = [&](const std::string& text) { if (log) log(text); };
  auto on_device = [&](bool wrapper, std::string& error) {
    if (auto net = create(settings.metacommands, wrapper, error)) return net;
    if (!settings.metacommands || !RetryableGpuFailure(error))
      return decltype(create(true, false, err)){};
    say("GPU session with driver metacommands failed (" + error +
        "): trying without them");
    std::string off;
    if (auto net = create(false, wrapper, off)) {
      settings.metacommands = false;
      say("driver metacommands are off from now on: GPU session loaded");
      return net;
    }
    say("GPU session without driver metacommands failed too (" + off + ")");
    // Do not hide a terminal attempt failure behind the first error.
    if (!RetryableGpuFailure(off)) error = off;
    return decltype(create(true, false, err)){};
  };
  if (auto net = on_device(settings.wrapper, err)) return net;
  if (settings.wrapper || !RetryableGpuFailure(err) || !LostDevice(err) || !wrapped())
    return decltype(create(true, false, err)){};
  say("GPU session was lost on the device inside the wrapper (" + err +
      "): trying the wrapper itself");
  std::string wrapper_error;
  if (auto net = on_device(true, wrapper_error)) {
    settings.wrapper = true;
    say("the model runs on the wrapper from now on");
    return net;
  }
  say("the wrapper failed too (" + wrapper_error + ")");
  if (!RetryableGpuFailure(wrapper_error)) err = wrapper_error;
  return decltype(create(true, false, err)){};
}
}  // namespace a2f
#endif
