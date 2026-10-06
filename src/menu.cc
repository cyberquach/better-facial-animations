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

constexpr ImVec4 kRed{1, 0.3f, 0.25f, 1}, kYellow{1, 0.8f, 0.2f, 1};
// Wrapped text in a color.
void Notice(ImVec4 color, const std::string& text) {
  if (ig.PushTextWrapPos) ig.PushTextWrapPos(0);
  if (ig.TextColored)
    ig.TextColored(color, "%s", text.c_str());
  else
    ig.Text("%s", text.c_str());
  if (ig.PopTextWrapPos) ig.PopTextWrapPos();
}

// The details of the item just drawn (id: any address of its own), as a
// tooltip once the pointer has stayed on it.
void Tip(const void* id, const char* details) {
  constexpr double kRest = 0.4;  // s
  constexpr size_t kColumns = 60;
  static const void* over;
  static double since, seen;  // the pointer came to over, was last on it
  if (!ig.IsItemHovered(0)) return;
  const double now = Now();
  if (over != id || now - seen > 0.1) {
    over = id;
    since = now;
  }
  seen = now;
  if (now - since < kRest) return;
  std::string text = details;
  for (size_t i = 0, start = 0, space = 0; i < text.size(); ++i) {
    if (text[i] == '\n') {
      start = i + 1;
      continue;
    }
    if (text[i] == ' ') space = i;
    if (i - start >= kColumns && space > start) {
      text[space] = '\n';
      start = space + 1;
    }
  }
  ig.SetTooltip("%s", text.c_str());
}

constexpr char kNetTip = 0, kSlowTip = 0, kLinesTip = 0;  // Tip ids

// Everything that is wrong (red: no lip-sync or no GPU session) or degraded
// (yellow), one line each; nothing when all is well.
void RenderProblems(const Stats& st, size_t queued, const Tune& tune,
                    const a2f::GpuState& gpu) {
  char b[400];
  if (!ready)
    Notice(kRed, st.startup_error.empty()
                     ? "Starting."
                     : "Not running: " + st.startup_error +
                           ". Reinstall the mod.");
  if (!st.net_error.empty()) {
    // the problem, its cause and the advice; the rest on the last of them
    for (const auto& line : st.net_error) Notice(kRed, line);
    snprintf(b, sizeof b, "\nTried %d time(s); next try in %.0f s. %s",
             st.net_attempts, std::max(0.0, st.net_retry_at - Now()),
             net_loaded ? "Lines run on the CPU meanwhile."
                        : "Vanilla lip-sync plays meanwhile.");
    Tip(&kNetTip, ("Runtime message: " + st.net_detail + b).c_str());
  } else if (ready && tune.enabled && !net_loaded) {
    Notice(kYellow, "Loading the model.");
  }
  if (net_loaded && !st.net_gpu && config.gpu) {
    Notice(kYellow, "On the CPU until the GPU is back: long lines start late.");
  }
  for (const auto& w : a2f::GpuWarnings(gpu)) Notice(kYellow, w);
  const double speed = st.audio_sec / std::max(st.gen_sec, 1e-6);
  if (st.lines >= 5 && speed < 2 && (st.net_gpu || !config.gpu)) {
    snprintf(b, sizeof b, "Slow (%.1fx realtime): lines start late.", speed);
    Notice(kYellow, b);
    Tip(&kSlowTip, st.net_gpu ? "The GPU is busy or short of VRAM."
                              : "Force CPU is on: the GPU is much faster.");
  }
  if (st.lines >= 8 && st.late * 4 > st.lines) {
    snprintf(b, sizeof b, "%u of %u lines started late.%s", st.late, st.lines,
             tune.prefetch ? "" : " Turn Prefetch on.");
    Notice(kYellow, b);
  }
  if (queued > 20) {
    snprintf(b, sizeof b, "%zu lines queued: generation cannot keep up.",
             queued);
    Notice(kYellow, b);
  }
  if (!st.last_error.empty())
    Notice(kYellow, "Last skipped line: " + st.last_error);
}

void RenderMenu() {
  static a2f::GpuState& gpu = *new a2f::GpuState;
  static double gpu_at = -1e9;
  if (Now() - gpu_at > 1) {  // DXGI is too slow to ask every frame
    gpu = a2f::QueryGpu();
    gpu_at = Now();
  }
  size_t cached, queued;
  const Stats st = SnapshotStats(cached, queued);
  const Tune tune = SnapshotTune();
  char b[400];
  ig.SeparatorText("Status");
  ig.Text("%s | %s", kVersion,
          st.net_info.empty() ? "model not loaded" : st.net_info.c_str());
  if (gpu.known) {
    ig.Text("GPU: %s, %.0f / %.0f MB VRAM", gpu.adapter.c_str(), gpu.used_mb,
            gpu.budget_mb);
    Tip(&gpu, "The VRAM the game uses, of what Windows grants it.");
  } else {
    ig.Text("GPU: %s", gpu.adapter.empty() ? "not found" : gpu.adapter.c_str());
  }
  if (st.lines || st.failed) {
    ig.Text("Lines: %u, %.0fx realtime", st.lines,
            st.audio_sec / std::max(st.gen_sec, 1e-6));
    snprintf(b, sizeof b,
             "Failed or without audio: %u\nStarted late: %u\nCached: %zu\n"
             "Queued: %zu\nLast line: %.0f ms",
             st.failed, st.late, cached, queued, st.last_ms);
    Tip(&kLinesTip, b);
  }
  RenderProblems(st, queued, tune, gpu);
  ig.Text("Expressive Facial Animation: Men %s | Women %s",
          config.expressive_faces[0] ? "detected" : "not detected",
          config.expressive_faces[1] ? "detected" : "not detected");
  Tip(config.expressive_faces,
      "Mouth strength is multiplied by 0.85 for each detected edition; "
      "otherwise it uses 1.0. Checks active Male 1.21 and Female 1.7 assets "
      "at startup. Modified versions may not be recognized.");
  static const char* const kHeaders[] = {
      nullptr, "Model", "Playback", "Strength", "Emotion", "Debug"};
  Sec cur = Sec::kGeneral;
  bool changed = false;
  for (size_t i = 0; i < kNumParams; ++i) {
    const Param& p = kParams[i];
    if (p.sec != cur) {
      cur = p.sec;
      ig.SeparatorText(kHeaders[static_cast<int>(cur)]);
    }
    ScopedLock guard(settings_lock);
    if (p.flag ? ig.Checkbox(p.label, p.flag)
               : ig.SliderFloat(p.label, p.value, p.lo, p.hi, "%.3g", 0)) {
      changed = true;
      if (p.flag)
        Log("setting %s = %d", p.key, *p.flag);
      else
        Log("setting %s = %g", p.key, *p.value);
    }
    guard.Unlock();
    Tip(&p, p.tip);
  }
  ApplyLogging();
  ig.SeparatorText("");
  // the model and its device are those the game started with
  if (tune.old_model != (config.network == 2) || tune.force_cpu != !config.gpu)
    Notice(kYellow, "Model settings apply once the game is restarted.");
  static const char kDefaults[] = "Defaults", kClear[] = "Clear cache";
  if (ig.Button(kDefaults, ImVec2{0, 0})) {
    ResetTune();
    changed = true;
  }
  if (changed) SaveIni();
  Tip(kDefaults, "Resets the sliders and switches, except Model and Debug.");
  ig.SameLine(0, -1);
  if (ig.Button(kClear, ImVec2{0, 0})) {
    ClearLines();
    ClearHeads();
  }
  Tip(kClear, "Forgets the generated lines and the fitted heads.");
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
