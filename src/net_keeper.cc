#include "net_keeper.h"

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
             "The game and the model run on \"%s\" (%.0f MB), not on \"%s\" "
             "(%.0f MB). Set Skyrim to the high-performance GPU in Windows "
             "Settings > Display > Graphics.",
             g.adapter.c_str(), g.dedicated_mb, g.bigger_adapter.c_str(),
             g.bigger_mb);
    out.push_back(b);
  }
  if (g.used_mb > g.budget_mb) {
    snprintf(b, sizeof b,
             "The game uses %.0f MB of VRAM, %.0f MB over the %.0f MB Windows "
             "grants it: textures are being moved to system memory (stutter) "
             "and the graphics device can be lost. Use smaller textures or "
             "close other programs that use the GPU.",
             g.used_mb, g.used_mb - g.budget_mb, g.budget_mb);
    out.push_back(b);
  }
  return out;
}

// What went wrong with a session that failed with err, for the settings
// window: the problem first, then what Windows says caused it and what to do.
// attempts: failures in a row.
std::vector<std::string> DiagnoseNet(bool gpu, const std::string& err,
                                     int attempts, const GpuState& g) {
  auto has = [&err](const char* what) {
    return err.find(what) != std::string::npos;
  };
  std::vector<std::string> out;
  char b[400];
  if (has("model file not found")) {
    out.push_back("FAILED: the model file is missing (" + err + ").");
    out.push_back(
        "Reinstall the mod, or correct Model= in "
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
  const bool oom = has("8007000E") || (g.known && free_mb < kNetVramMb);
  const bool lost =
      has("887A0005") || has("887A0006") || has("887A0007") || has("887A0020");
  if (!lost && !oom && (has("DML api") || has("DirectML:"))) {
    out.push_back("GPU FAILED: DirectML is not available.");
    out.push_back(
        "It needs Windows 10 version 1903 or later and a DirectX 12 "
        "graphics card with a current driver.");
    return out;
  }
  if (oom) {
    snprintf(b, sizeof b,
             "GPU FAILED: no VRAM left (%s; the model needs about %.0f MB).",
             vram, kNetVramMb);
    out.push_back(b);
    out.push_back(
        "Free VRAM (smaller textures, close other programs that "
        "use the GPU): the model loads at the next retry.");
  } else if (lost) {
    snprintf(b, sizeof b, "GPU FAILED: the graphics device was lost (%s now).",
             vram);
    out.push_back(b);
  } else {
    snprintf(b, sizeof b, "GPU FAILED: the model did not load (%s).", vram);
    out.push_back(b);
  }
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
      out.push_back(
          "Direct3D 12 reports its device as working: DirectML stopped by "
          "itself (reason " + dml_reason + "), on a step that failed inside "
          "DirectML or the graphics driver.");
    else
      out.push_back(
          "Direct3D 12 reports its device as working and DirectML gave no "
          "reason: a step failed inside DirectML or the graphics driver.");
    if (attempts >= 3 && !reason.empty()) {
      snprintf(b, sizeof b,
               "Still lost after %d attempts: restart the game to get the GPU "
               "back.",
               attempts);
      out.push_back(b);
    } else if (attempts >= 3) {
      snprintf(b, sizeof b,
               "Still failing after %d attempts, each on a new device: the "
               "cause is in this system's driver, DirectML or a program that "
               "hooks Direct3D 12, not in VRAM. Attach "
               "BetterFacialAnimations.log to a bug report.",
               attempts);
      out.push_back(b);
    }
  }
  return out;
}

std::string DescribeMetacommands(const Metacommands& m) {
  switch (m.state) {
    case Metacommands::kOn:
      return "on (the GPU session loaded with them)";
    case Metacommands::kOff:
      return "off: the GPU session failed with them and loaded without them";
    case Metacommands::kFailed:
      return "not the cause: the GPU session fails with them and without "
             "them";
    default:
      return "not tried yet";
  }
}

std::unique_ptr<Net> CreateGpuNet(
    const CreateGpu& create, Metacommands& meta, std::string& err,
    std::string& info, const std::function<void(const std::string&)>& log) {
  auto say = [&log](const std::string& s) {
    if (log) log(s);
  };
  if (meta.state != Metacommands::kOff) {
    if (auto net = create(true, err, info)) {
      if (meta.state != Metacommands::kOn)
        say("driver metacommands work: the GPU session loaded with them");
      meta.state = Metacommands::kOn;
      meta.error.clear();
      return net;
    }
    // the runtime itself did not load: the same without them
    for (const char* other :
         {"cannot load onnxruntime.dll", "onnxruntime.dll: wrong version"})
      if (err.find(other) != std::string::npos) return nullptr;
    say("GPU session with driver metacommands failed (" + err +
        "): trying without them");
    meta.error = err;
  }
  std::string err_off;
  if (auto net = create(false, err_off, info)) {
    if (meta.state != Metacommands::kOff)
      say("driver metacommands are off from now on: the GPU session loaded "
          "without them");
    meta.state = Metacommands::kOff;
    return net;
  }
  if (meta.state == Metacommands::kOff) {
    err = err_off;  // the only attempt
  } else {
    say("GPU session without driver metacommands failed too (" + err_off +
        ")");
    meta.state = Metacommands::kFailed;
  }
  return nullptr;
}

Net* NetKeeper::Ensure() {
  if (Wanted()) return net_.get();
  if (now_() >= retry_at_) {
    std::string err, info;
    if (auto n = create_(want_gpu_, err, info)) {
      net_ = std::move(n);
      on_gpu_ = want_gpu_;
      info_ = info;
      attempts_ = 0;
      retry_wait_ = 0;
      error_.clear();
      Log("network loaded: " + info);
    } else {
      ++attempts_;
      retry_wait_ =
          retry_wait_ ? std::min(retry_wait_ * 2, kRetryMax) : kRetryMin;
      retry_at_ = now_() + retry_wait_;
      error_ = err;
      Log("network not loaded (attempt " + std::to_string(attempts_) +
          ", next in " + std::to_string(static_cast<int>(retry_wait_)) +
          " s): " + err);
    }
  }
  if (!net_ && want_gpu_ && now_() >= cpu_retry_at_) {
    std::string err, info;
    net_ = create_(false, err, info);
    on_gpu_ = false;
    if (net_) {
      info_ = info;
      Log("network loaded as CPU fallback: " + info);
    } else {
      cpu_retry_at_ = now_() + kRetryMax;
      Log("CPU fallback failed: " + err);
    }
  }
  if (!net_) info_ = "not loaded";
  return net_.get();
}

}  // namespace a2f
