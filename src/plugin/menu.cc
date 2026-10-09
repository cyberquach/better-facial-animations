// The settings page: SKSE Menu Framework (optional; its cimgui exports).
#include "a2f/plugin/menu.h"

#include <windows.h>

#include <algorithm>
#include <cstdio>
#include <string>

#include "a2f/animation/head_cache.h"
#include "a2f/inference/net_keeper.h"
#include "a2f/inference/ort_net.h"
#include "a2f/plugin/plugin.h"
#include "a2f/runtime/scoped_lock.h"
#include "a2f/runtime/skse.h"

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
  if (net_loaded && !st.net_gpu && st.want_gpu) {
    Notice(kYellow, "On the CPU until the GPU is back: long lines start late.");
  }
  for (const auto& w : a2f::GpuWarnings(gpu)) Notice(kYellow, w);
  const double speed = st.audio_sec / std::max(st.gen_sec, 1e-6);
  if (st.lines >= 5 && speed < 2 && (st.net_gpu || !st.want_gpu)) {
    snprintf(b, sizeof b, "Slow (%.1fx realtime): lines start late.", speed);
    Notice(kYellow, b);
    Tip(&kSlowTip, st.network == a2f::kNetPerformance && !st.net_gpu
                       ? "Performance model runs on the CPU; check "
                         "CPU load and enable Prefetch."
                   : st.net_gpu ? "The GPU is busy or short of VRAM."
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
  ig.Text("Expressive Facial Animation - Male Edition: %s",
          config.expressive_faces[0] ? "Installed mod detected" : "Not detected");
  ig.Text("Expressive Facial Animation - Female Edition: %s",
          config.expressive_faces[1] ? "Installed mod detected" : "Not detected");
  snprintf(b, sizeof b,
           "Mouth strength is multiplied by %g for each detected edition; "
           "otherwise it uses 1.0. Checks active Male 1.21 and Female 1.7 "
           "assets at startup. Modified versions may not be recognized.",
           kExpressiveMouthScale);
  Tip(config.expressive_faces, b);
  static const char* const kHeaders[] = {
      nullptr, "Model", "Playback", "Strength", "Emotion", "Debug"};
  Sec cur = Sec::kGeneral;
  bool changed = false;
  int profile = -1;
  for (size_t i = 0; i < kNumParams; ++i) {
    const Param& p = kParams[i];
    if (p.sec != cur || p.profile != profile) {
      cur = p.sec;
      profile = p.profile;
      snprintf(b, sizeof b, "%s%s",
               profile < 0 ? ""
               : profile == 2 ? "Player — "
               : profile   ? "Female NPC — "
                           : "Male NPC — ",
               kHeaders[static_cast<int>(cur)]);
      ig.SeparatorText(b);
    }
    snprintf(b, sizeof b, "%s##%zu", p.label, i);
    ScopedLock guard(settings_lock);
    if (p.flag ? ig.Checkbox(b, p.flag)
               : ig.SliderFloat(b, p.value, p.lo, p.hi, "%.3g", 0)) {
      changed = true;
      SelectModelCheckbox(p);
      if (p.flag)
        Log("setting %s = %d", p.key, *p.flag);
      else
        Log("setting %s%s = %g",
            p.profile < 0 ? ""
            : p.profile == 2 ? "Player "
            : p.profile   ? "Female "
                          : "Male ",
            p.key, *p.value);
    }
    guard.Unlock();
    Tip(&p, p.tip);
  }
  ApplyLogging();
  ig.SeparatorText("");
  // The worker publishes the applied model independently of the selection.
  if (!st.model_error.empty())
    Notice(kRed, st.model_error);
  else if (st.switching_model || tune.SelectedModel() != st.network ||
           (tune.force_cpu != !st.want_gpu))
    Notice(kYellow,
           "Switching model; new dialogue uses it when loading finishes.");
  static const char kDefaults[] = "Defaults", kClear[] = "Clear cache";
  if (ig.Button(kDefaults, ImVec2{0, 0})) {
    ResetTune();
    changed = true;
  }
  if (changed) {
    WakeWorker();
    ScheduleSaveIni();
  }
  Tip(kDefaults, "Resets the sliders and switches, except Model and Debug.");
  ig.SameLine(0, -1);
  if (ig.Button(kClear, ImVec2{0, 0})) {
    ClearLines();
    ClearHeads();
  }
  Tip(kClear, "Forgets the generated lines and the fitted heads.");
}

// A bounded list keeps large mod lists from overwhelming the settings page.
void RenderRaces() {
  auto races = SnapshotRaces();
  std::sort(races.begin(), races.end(),
            [](const RaceSetting& left, const RaceSetting& right) {
              return left.label == right.label ? left.key < right.key
                                               : left.label < right.label;
            });
  ig.SeparatorText("Apply facial animations by race");
  ig.Text("Unchecked races use vanilla animation. Changes save automatically.");
  ig.Text("Heads need compatible face morphs.");
  constexpr size_t kPageSize = 12;
  static size_t page = 0;
  const size_t pages = std::max(size_t{1}, (races.size() + kPageSize - 1) /
                                            kPageSize);
  page = std::min(page, pages - 1);
  if (ig.Button("Previous", ImVec2{0, 0}) && page > 0) --page;
  ig.SameLine(0, -1);
  ig.Text("%zu / %zu", page + 1, pages);
  ig.SameLine(0, -1);
  if (ig.Button("Next", ImVec2{0, 0}) && page + 1 < pages) ++page;
  for (size_t i = page * kPageSize;
       i < std::min(races.size(), (page + 1) * kPageSize); ++i) {
    RaceSetting& race = races[i];
    const std::string label = race.label + "##" + race.key;
    if (ig.Checkbox(label.c_str(), &race.enabled)) {
      SetRaceEnabled(race.id, race.enabled);
      ScheduleSaveIni();
    }
    if (ig.IsItemHovered(0)) ig.SetTooltip("%s", race.key.c_str());
  }
  if (races.empty()) ig.Text("No races detected.");
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
  add("Better Facial Animations/Races", RenderRaces);
  Log("menu page registered (SKSE Menu Framework)");
}

}  // namespace bfa
