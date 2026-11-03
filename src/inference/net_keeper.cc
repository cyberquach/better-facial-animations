#include "a2f/inference/net_keeper.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace a2f {
// Problems that do not stop the model from loading, one line each.
std::vector<std::string> GpuWarnings(const GpuState& g) {
  std::vector<std::string> out;
  char b[400];
  if (!g.known) return out;
  if (!g.bigger_adapter.empty()) {
    snprintf(b, sizeof b,
             "The model runs on \"%s\" (%.0f MB); \"%s\" (%.0f MB) is also "
             "present. If Skyrim renders on the bigger one, set it as "
             "Skyrim's high-performance GPU in Windows Settings > Display > "
             "Graphics.",
             g.adapter.c_str(), g.dedicated_mb, g.bigger_adapter.c_str(),
             g.bigger_mb);
    out.push_back(b);
  }
  if (g.used_mb > g.budget_mb) {
    snprintf(b, sizeof b,
             "This process uses %.0f MB of GPU memory, %.0f MB over the "
             "%.0f MB budget of this adapter: expect stutter. Use smaller "
             "textures or close other programs that use the GPU.",
             g.used_mb, g.used_mb - g.budget_mb, g.budget_mb);
    out.push_back(b);
  }
  return out;
}

// What went wrong with a session that failed with err, for the settings
// window: the problem first, then what Windows says caused it and what to do.
std::vector<std::string> DiagnoseNet(bool gpu, const std::string& err,
                                     const GpuState& g, double need_mb) {
  auto has = [&err](const char* what) {
    return err.find(what) != std::string::npos;
  };
  std::vector<std::string> out;
  char b[400];
  if (gpu && (has("independent device:") || has("independent context:") ||
              has("GPU attempt failed"))) {
    out.push_back(
        "GPU model unavailable in Independent GPU mode. Vanilla lip-sync "
        "remains active.");
    if (has("runtime API unavailable:"))
      out.push_back(
          "The loaded DirectX runtime does not expose the Independent "
          "Devices API. Independent GPU mode needs Windows 10 22H2 or Windows "
          "11 with the August 2023 or later cumulative update (Proton: a "
          "current vkd3d-proton) and a current graphics driver; see "
          "GPU-SETUP.txt and the OS prerequisite line in the diagnostic "
          "report. Driver support was not tested.");
    else if (has("device creation: CreateDevice"))
      out.push_back(
          "The Independent Devices API is available, but device "
          "creation failed. See the diagnostic report for the native error; "
          "this is not evidence that the Windows API is missing. Update the "
          "graphics driver.");
    else if (has("independent context:") || LostDevice(err))
      out.push_back(
          "The independent device was lost. It is created again "
          "automatically, up to 3 times; further failures wait for Try GPU "
          "again.");
    out.push_back(
        "Choose Try GPU again, or enable Force CPU to use CPU. To leave "
        "Independent GPU mode, untick it in the installer "
        "(IndependentGpuDevice=0) and restart Skyrim; the shared route needs "
        "no special Windows or driver support.");
    return out;
  }
  if (has("model file not found")) {
    out.push_back("FAILED: the model file is missing (" + err + ").");
    out.push_back(
        "Reinstall the mod, or select another model in Settings / Model or "
        "BetterFacialAnimations.ini.");
    return out;
  }
  if (has("cannot load onnxruntime.dll")) {
    out.push_back("FAILED: " + err + ".");
    out.push_back(
        "onnxruntime.dll in SKSE/Plugins/BetterFacialAnimations is "
        "missing, blocked by antivirus, or its DirectML.dll is "
        "missing. Reinstall the mod.");
    return out;
  }
  if (has("wrong version")) {
    out.push_back(
        "FAILED: onnxruntime.dll is not the version this plugin "
        "was built for.");
    out.push_back("Files of two mod versions are mixed. Reinstall the mod.");
    return out;
  }
  if (!gpu) {
    out.push_back("FAILED to load the model on the CPU.");
    return out;
  }
  const double free_mb = std::max(0.0, g.budget_mb - g.used_mb);
  char vram[80] = "free VRAM unknown";
  if (g.known)
    snprintf(vram, sizeof vram, "%.0f MB free of %.0f MB", free_mb,
             g.budget_mb);
  // The reason Windows gives for the lost device (ort_net.cc appends it).
  std::string reason;
  const char kReason[] = "device removed reason ";
  size_t at = err.find(kReason);
  if (at != std::string::npos) reason = err.substr(at + sizeof kReason - 1, 8);
  // The reason DirectML gives when its own device stopped.
  std::string dml_reason;
  const char kDmlReason[] = "DirectML removed reason ";
  at = err.find(kDmlReason);
  if (at != std::string::npos)
    dml_reason = err.substr(at + sizeof kDmlReason - 1, 8);
  // E_OUTOFMEMORY; DXGI device removed / hung / reset / driver error
  const bool oom = has("8007000E");
  const bool pressure = g.known && free_mb < need_mb;
  const bool lost = LostDevice(err);
  if (!lost && !oom && (has("DML api") || has("DirectML:"))) {
    out.push_back("GPU FAILED: DirectML is not available.");
    out.push_back(
        "Keep the bundled runtime files together. The selected GPU route "
        "needs a supporting runtime and driver; IndependentGpuDevice=1 "
        "also requires the Independent Devices API. See GPU-SETUP.txt.");
    return out;
  }
  if (oom) {
    snprintf(b, sizeof b,
             "GPU FAILED: memory allocation failed (%s; the model usually "
             "uses about %.0f MB).",
             vram, need_mb);
    out.push_back(b);
    out.push_back(
        "Free VRAM (smaller textures, close other programs that "
        "use the GPU), then choose Try GPU again.");
  } else if (lost) {
    snprintf(b, sizeof b, "GPU FAILED: the graphics device was lost (%s now).",
             vram);
    out.push_back(b);
  } else {
    snprintf(b, sizeof b, "GPU FAILED: the model did not load (%s).", vram);
    out.push_back(b);
  }
  if (pressure && !oom)
    out.push_back(
        "Free VRAM is below what this model usually needs: memory "
        "pressure is a likely cause.");
  if (lost) {
    if (reason == "887A0006")
      out.push_back(
          "Windows: the graphics driver stopped responding (device "
          "hung). Usual causes: VRAM overcommitted, an unstable "
          "overclock, or a driver bug.");
    else if (reason == "887A0007")
      out.push_back(
          "Windows: the graphics driver was reset (device reset), by "
          "a driver crash or another program's bad commands.");
    else if (reason == "887A0020")
      out.push_back(
          "Windows: internal error in the graphics driver. Update "
          "the graphics driver.");
    else if (reason == "887A0005")
      out.push_back(
          "Windows: the graphics adapter was removed or disabled (a "
          "driver update, or the GPU was switched or unplugged).");
    else if (reason == "8007000E")
      out.push_back("Windows: the device ran out of memory.");
    else if (!reason.empty())
      out.push_back("Windows: device removed, reason " + reason + ".");
    else if (!dml_reason.empty())
      out.push_back("DirectML reported device failure reason " + dml_reason +
                    ".");
    else
      out.push_back(
          "Windows gave no device-removal reason: the failure happened inside "
          "DirectML, the graphics driver or a program that hooks Direct3D 12.");
    out.push_back(
        "Shared GPU retries automatically with backoff, independent GPU up "
        "to 3 times; Try GPU again starts a manual attempt. Attach "
        "BetterFacialAnimations.log to a bug report if it recurs.");
  }
  return out;
}

void NetKeeper::GpuFailed(const std::string& err) {
  gpu_error_ = error_ = err.empty() ? "GPU initialization failed" : err;
  if (independent_) {
    gpu_blocked_ = !RetryableIndependentFailure(gpu_error_) ||
                   independent_retries_ >= kIndependentRetries;
    if (!gpu_blocked_) ++independent_retries_;
  } else {
    gpu_blocked_ = !RetryableGpuFailure(gpu_error_);
  }
  if (gpu_blocked_) {
    retry_at_ = 0;
    Log(std::string(independent_ && independent_retries_ >= kIndependentRetries
                        ? "independent GPU failed " +
                              std::to_string(kIndependentRetries + 1) +
                              " times; choose Try GPU again or Force CPU: "
                        : "GPU attempt failed; choose Try GPU again or Force "
                          "CPU: ") +
        error_);
  } else {
    retry_wait_ =
        retry_wait_ ? std::min(retry_wait_ * 2, kRetryMax) : kRetryMin;
    retry_at_ = now_() + retry_wait_;
    Log(std::string(independent_ ? "independent" : "shared") +
        " GPU unavailable; retry in " +
        std::to_string(static_cast<int>(retry_wait_)) + " s: " + error_);
  }
}

Net* NetKeeper::Ensure() {
  if (Wanted()) return net_.get();
  if (want_gpu_ && !gpu_blocked_ && now_() >= retry_at_) {
    std::string err, info;
    if (auto n = create_(true, err, info)) {
      net_ = std::move(n);
      on_gpu_ = true;
      info_ = info;
      gpu_error_.clear();
      error_.clear();
      retry_at_ = retry_wait_ = 0;
      independent_retries_ = 0;
      Log("network loaded: " + info);
    } else {
      GpuFailed(err);
    }
  }
  if (!want_gpu_ && !net_ && now_() >= cpu_retry_at_) {
    std::string err, info;
    net_ = create_(false, err, info);
    on_gpu_ = false;
    if (net_) {
      info_ = info;
      cpu_retry_at_ = 0;
      error_.clear();
      Log("network loaded on CPU: " + info);
    } else {
      cpu_retry_at_ = now_() + kCpuRetry;
      error_ = err;
      Log("CPU unavailable; retry in " +
          std::to_string(static_cast<int>(kCpuRetry)) + " s: " + err);
    }
  }
  if (!net_) info_ = "not loaded";
  return net_.get();
}

}  // namespace a2f
