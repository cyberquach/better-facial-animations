#include "a2f/inference/net_keeper.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace a2f {
namespace {

// The device was removed, hung or reset or the driver failed.
bool LostDevice(const std::string& err) {
  // DXGI device removed / hung / reset / driver internal error
  for (const char* code : {"887A0005", "887A0006", "887A0007", "887A0020"})
    if (err.find(code) != std::string::npos) return true;
  return false;
}

}  // namespace

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
  if (gpu && has("GPU disabled until restart")) {
    out.push_back("GPU context unavailable; using CPU fallback if available.");
    out.push_back("Restart the game to retry the GPU. Switching models will "
                  "not recreate a failed graphics device.");
    return out;
  }
  if (gpu && has("independent device:")) {
    out.push_back(
        "Independent GPU device failed; using CPU fallback if available.");
    out.push_back("This mode requires a supporting D3D12 runtime and driver. "
                  "Shared-device and wrapper retries are disabled.");
    out.push_back("Update Windows and the graphics driver, then restart. "
                  "See the detailed error and log; ForceCPU=1 selects CPU "
                  "only.");
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
        "use the GPU), then restart the game to retry the GPU.");
  } else if (lost) {
    snprintf(b, sizeof b, "GPU FAILED: the graphics device was lost (%s now).",
             vram);
    out.push_back(b);
  } else {
    snprintf(b, sizeof b, "GPU FAILED: the model did not load (%s).", vram);
    out.push_back(b);
  }
  if (pressure && !oom)
    out.push_back("Free VRAM is below what this model usually needs: memory "
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
    out.push_back("Restart the game to retry the GPU; attach "
                  "BetterFacialAnimations.log to a bug report if it recurs.");
  }
  return out;
}

Net* NetKeeper::Ensure() {
  if (loaded()) return net_.get();
  if (want_gpu_ && gpu_error_.empty()) {
    std::string err, info;
    if (auto n = create_(true, err, info)) {
      net_ = std::move(n);
      on_gpu_ = true;
      info_ = info;
      error_.clear();
      Log("network loaded: " + info);
    } else {
      gpu_error_ = error_ = err.empty() ? "GPU initialization failed" : err;
      Log("GPU disabled until restart: " + error_);
    }
  }
  if (!net_ && now_() >= cpu_retry_at_) {
    std::string err, info;
    net_ = create_(false, err, info);
    on_gpu_ = false;
    if (net_) {
      info_ = info;
      retry_at_ = 0;
      error_ = want_gpu_ ? gpu_error_ : "";  // the GPU's failure stays shown
      Log("network loaded on CPU: " + info);
    } else {
      retry_at_ = cpu_retry_at_ = now_() + kCpuRetry;
      error_ = want_gpu_ ? gpu_error_ + "; CPU: " + err : err;
      Log("CPU unavailable; retry in " +
          std::to_string(static_cast<int>(kCpuRetry)) + " s: " + err);
    }
  }
  if (!net_) info_ = "not loaded";
  return net_.get();
}

}  // namespace a2f
