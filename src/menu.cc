// The settings page: SKSE Menu Framework (optional; its cimgui exports).
#include "menu.h"

#include <windows.h>

#include <algorithm>
#include <cstdio>
#include <string>

#include "head_cache.h"
#include "net_keeper.h"
#include "ort_net.h"
#include "plugin.h"
#include "scoped_lock.h"
#include "skse.h"

namespace bfa {

using skse::Log;

namespace {

struct ImVec2 {
  float x, y;
};
struct ImVec4 {
  float x, y, z, w;
};
struct ImGui {  // cimgui export names
  bool (*SliderFloat)(const char*, float*, float, float, const char*, int);
  bool (*Checkbox)(const char*, bool*);
  void (*Text)(const char*, ...);
  void (*SeparatorText)(const char*);
  bool (*Button)(const char*, ImVec2);
  void (*SetTooltip)(const char*, ...);
  bool (*IsItemHovered)(int);
  void (*SameLine)(float, float);
  // optional
  void (*TextColored)(ImVec4, const char*, ...);
  void (*PushTextWrapPos)(float);
  void (*PopTextWrapPos)();
} ig;

constexpr ImVec4 kRed{1, 0.3f, 0.25f, 1}, kYellow{1, 0.8f, 0.2f, 1},
    kGreen{0.4f, 0.9f, 0.4f, 1};
// One wrapped line of the Diagnostics section.
void Notice(ImVec4 color, const char* tag, const std::string& text) {
  if (ig.PushTextWrapPos) ig.PushTextWrapPos(0);
  if (ig.TextColored)
    ig.TextColored(color, "%s%s", tag, text.c_str());
  else
    ig.Text("%s%s", tag, text.c_str());
  if (ig.PopTextWrapPos) ig.PopTextWrapPos();
}

// Everything that is wrong or degraded, as exactly as it can be told: errors
// (red) stop lip-sync or the GPU session, warnings (yellow) make it worse.
void RenderDiagnostics(const Stats& st, size_t queued, const Tune& tune) {
  static a2f::GpuState& gpu = *new a2f::GpuState;
  static double gpu_at = -1e9;
  if (Now() - gpu_at > 1) {  // DXGI is too slow to ask every frame
    gpu = a2f::QueryGpu();
    gpu_at = Now();
  }
  ig.SeparatorText("Diagnostics");
  if (gpu.known)
    ig.Text(
        "GPU: %s; the game uses %.0f of the %.0f MB VRAM Windows grants "
        "it",
        gpu.adapter.c_str(), gpu.used_mb, gpu.budget_mb);
  else
    ig.Text("GPU: %s (VRAM use unknown)",
            gpu.adapter.empty() ? "not found" : gpu.adapter.c_str());
  int problems = 0;
  char b[400];
  if (!ready) {
    ++problems;
    Notice(kRed, "ERROR: ",
           st.startup_error.empty()
               ? "the plugin has not finished starting."
               : "not running: " + st.startup_error + ". Reinstall the mod.");
  }
  if (!st.net_error.empty()) {
    ++problems;
    for (size_t i = 0; i < st.net_error.size(); ++i)
      Notice(kRed, i ? "  " : "ERROR: ", st.net_error[i]);
    snprintf(b, sizeof b, "Tried %d time(s); next try in %.0f s. %s",
             st.net_attempts, std::max(0.0, st.net_retry_at - Now()),
             net_loaded ? "Lines are generated on the CPU meanwhile."
                        : "Vanilla lip-sync plays meanwhile.");
    Notice(kRed, "  ", b);
    Notice(kRed, "  Runtime message: ", st.net_detail);
  } else if (ready && tune.enabled && !net_loaded) {
    Notice(kYellow, "", "The model is loading.");
  }
  if (net_loaded && !st.net_gpu && config.gpu) {
    ++problems;
    Notice(kYellow, "WARNING: ",
           "running on the CPU fallback (about 0.4x realtime): long lines "
           "start late. It goes back to the GPU by itself.");
  }
  for (const auto& w : a2f::GpuWarnings(gpu)) {
    ++problems;
    Notice(kYellow, "WARNING: ", w);
  }
  const double speed = st.audio_sec / std::max(st.gen_sec, 1e-6);
  if (st.lines >= 5 && speed < 2 && (st.net_gpu || !config.gpu)) {
    ++problems;
    snprintf(b, sizeof b,
             "generation is slow (%.1fx realtime%s): lines start late.%s",
             speed, st.net_gpu ? " on the GPU" : " on the CPU",
             st.net_gpu ? " The GPU is busy or short of VRAM."
                        : " GPU=0 in the ini: set GPU=1 to use the GPU.");
    Notice(kYellow, "WARNING: ", b);
  }
  if (st.lines >= 8 && st.late * 4 > st.lines) {
    ++problems;
    snprintf(b, sizeof b,
             "%u of %u lines were ready after they started (blended in "
             "late).%s",
             st.late, st.lines, tune.prefetch ? "" : " Turn Prefetch on.");
    Notice(kYellow, "WARNING: ", b);
  }
  if (queued > 20) {
    ++problems;
    snprintf(b, sizeof b,
             "%zu lines are waiting to be generated: generation cannot keep "
             "up.",
             queued);
    Notice(kYellow, "WARNING: ", b);
  }
  if (!st.last_error.empty())
    Notice(kYellow, "Last skipped line: ", st.last_error);
  if (!tune.enabled)
    Notice(kYellow, "", "Disabled: vanilla lip-sync plays.");
  else if (!problems && net_loaded)
    Notice(kGreen, "", "No problems detected.");
  if (!st.events.empty()) {
    ig.Text("Recent events:");
    for (const auto& [at, text] : st.events) {
      snprintf(b, sizeof b, "  %.0f s ago: ", Now() - at);
      Notice(ImVec4{0.75f, 0.75f, 0.75f, 1}, b, text);
    }
  }
}

void RenderMenu() {
  size_t cached, queued;
  const Stats st = SnapshotStats(cached, queued);
  ig.SeparatorText("Status");
  ig.Text("Version: %s", kVersion);
  ig.Text("Model: %s", st.net_info.c_str());
  ig.Text("Lines generated: %u (%u failed / no audio), cached %zu, queued %zu",
          st.lines, st.failed, cached, queued);
  if (st.lines)
    ig.Text("Speed: %.0fx realtime, last line %.0f ms; started late: %u",
            st.audio_sec / std::max(st.gen_sec, 1e-6), st.last_ms, st.late);
  RenderDiagnostics(st, queued, SnapshotTune());
  static const char* const kHeaders[] = {
      nullptr, "Playback", "Local Animation Strength",
      "Emotion (from the next line spoken)", "Debug"};
  Sec cur = Sec::kGeneral;
  for (size_t i = 0; i < kNumParams; ++i) {
    const Param& p = kParams[i];
    if (p.sec != cur) {
      cur = p.sec;
      ig.SeparatorText(kHeaders[static_cast<int>(cur)]);
    }
    ScopedLock guard(settings_lock);
    if (p.flag ? ig.Checkbox(p.label, p.flag)
               : ig.SliderFloat(p.label, p.value, p.lo, p.hi, "%.3g", 0)) {
      if (p.flag)
        Log("setting %s = %d", p.key, *p.flag);
      else
        Log("setting %s = %g", p.key, *p.value);
    }
    guard.Unlock();
    if (ig.IsItemHovered(0)) ig.SetTooltip("%s", p.tip);
  }
  ApplyLogging();
  ig.SeparatorText("");
  if (ig.Button("Save to BetterFacialAnimations.ini", ImVec2{0, 0})) {
    SaveIni();
  }
  ig.SameLine(0, -1);
  if (ig.Button("Defaults", ImVec2{0, 0})) {
    ResetTune();
  }
  ig.SameLine(0, -1);
  if (ig.Button("Clear cache", ImVec2{0, 0})) {
    ClearLines();
    ClearHeads();
  }
}

// Resolves an export into fn; false when missing.
template <typename Fn>
bool Import(HMODULE m, const char* name, Fn& fn) {
  fn = reinterpret_cast<Fn>(reinterpret_cast<void*>(GetProcAddress(m, name)));
  return fn != nullptr;
}

}  // namespace

void RegisterMenu() {
  HMODULE m = GetModuleHandleA("SKSEMenuFramework.dll");
  if (!m) {
    Log("SKSE Menu Framework not installed: settings via "
        "BetterFacialAnimations.ini only");
    return;
  }
  void (*add)(const char*, void (*)());
  if (!Import(m, "AddSectionItem", add) ||
      !Import(m, "igSliderFloat", ig.SliderFloat) ||
      !Import(m, "igCheckbox", ig.Checkbox) || !Import(m, "igText", ig.Text) ||
      !Import(m, "igSeparatorText", ig.SeparatorText) ||
      !Import(m, "igButton", ig.Button) ||
      !Import(m, "igSetTooltip", ig.SetTooltip) ||
      !Import(m, "igIsItemHovered", ig.IsItemHovered) ||
      !Import(m, "igSameLine", ig.SameLine)) {
    Log("SKSE Menu Framework: missing exports, no menu page");
    return;
  }
  Import(m, "igTextColored", ig.TextColored);
  if (!Import(m, "igPushTextWrapPos", ig.PushTextWrapPos) ||
      !Import(m, "igPopTextWrapPos", ig.PopTextWrapPos)) {
    ig.PushTextWrapPos = nullptr;
    ig.PopTextWrapPos = nullptr;
  }
  add("Better Facial Animations/Settings", RenderMenu);
  Log("menu page registered (SKSE Menu Framework)");
}

}  // namespace bfa
