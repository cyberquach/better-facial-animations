// The settings page: SKSE Menu Framework (optional; its cimgui exports).
#include "a2f/plugin/menu.h"

#include <windows.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>

#include "a2f/animation/head_cache.h"
#include "a2f/inference/net_keeper.h"
#include "a2f/inference/ort_net.h"
#include "a2f/plugin/plugin.h"
#include "a2f/runtime/engine.h"
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
  void (*BeginDisabled)(bool);
  void (*EndDisabled)();
  void (*TextColored)(ImVec4, const char*, ...);
  void (*PushTextWrapPos)(float);
  void (*PopTextWrapPos)();
  bool (*Begin)(const char*, bool*, int);
  void (*End)();
  void (*SetNextWindowSize)(ImVec2, int);
} ig;

// ImGuiHoveredFlags_AllowWhenDisabled (Menu Framework's cimgui ABI).
constexpr int kHoverDisabled = 1 << 10;

const char* ParamTip(const Param& p, bool npc = false) {
  if (p.needs == Param::kSentiment && !config.sentiment_available)
    return "Sentiment Analysis is unavailable because the optional sentiment "
           "model is not installed in this copy of Better Facial Animations. "
           "The download without sentiment does not include it.\n\n"
           "Download and install the release that includes the sentiment "
           "model. It must provide emotion_fp16.onnx in "
           "Data/SKSE/Plugins/BetterFacialAnimations. Restart Skyrim after "
           "installing it, then enable Sentiment Analysis.\n\n"
           "This model detects emotion from voices. It requires about 600 MB "
           "of additional VRAM on GPU and adds processing time. Force CPU "
           "uses system memory and additional CPU processing.";
  if (npc && !strcmp(p.key, "SentimentAnalysis"))
    return "Enable voice-detected emotion for this NPC. "
           "Requires global Sentiment Analysis. "
           "Disabling skips sentiment inference for this "
           "NPC and preserves their multiplier. Applies next line.";
  return p.tip;
}

// The group a parameter is shown in: its own, or its ini section's.
Group GroupFor(const Param& p) {
  static const Group by_section[] = {Group::kGeneral,  Group::kModel,
                                     Group::kPlayback, Group::kMovement,
                                     Group::kInputs,   Group::kDebug};
  return p.group != Group::kAuto ? p.group
                                 : by_section[static_cast<int>(p.sec)];
}
const char* GroupTitle(Group group) {
  static const char* const titles[] = {
      "General", "Animation model & processing", "Emotion input sources",
      "Speech timing", "Expression transitions", "Facial movement strength",
      "Emotion input amounts", "Maximum emotion inputs", "Diagnostics"};
  return titles[static_cast<int>(group)];
}
void HelpText(const char* text) {
  if (ig.PushTextWrapPos) ig.PushTextWrapPos(0);
  ig.Text("%s", text);
  if (ig.PopTextWrapPos) ig.PopTextWrapPos();
}
void GroupHelp(Group group) {
  switch (group) {
    case Group::kModel:
      HelpText("Choose one model. Force CPU runs the model on CPU. Changes "
               "apply live.");
      break;
    case Group::kSources:
      HelpText("Combine authored scene emotion and voice sentiment. Each "
               "source has its own toggle. Lip-sync remains active with both "
               "off.");
      break;
    case Group::kMovement:
      HelpText("Scales generated facial movement: 0 = none, 1 = normal. "
               "These are movement multipliers.");
      break;
    case Group::kInputs:
      HelpText("Adjust each source before the inputs are combined. The "
               "maximum emotion inputs below limit the combined result. "
               "Applies next line.");
      break;
    case Group::kLimits:
      HelpText("Maximum emotion inputs after all sources and multipliers are "
               "combined. 0 blocks an input, 1 is the default, 2 is the "
               "highest limit. Applies next line.");
      break;
    default:
      break;
  }
}
const std::vector<size_t>& MenuOrder() {
  static const std::vector<size_t> order = [] {
    std::vector<size_t> indices;
    for (size_t i = 0; i < kNumParams; ++i) indices.push_back(i);
    std::stable_sort(indices.begin(), indices.end(), [](size_t a, size_t b) {
      const Param& left = kParams[a];
      const Param& right = kParams[b];
      // Diagnostics follows all profiles rather than interrupting the
      // shared controls.
      const int lp = left.sec == Sec::kDebug ? 3 : left.profile;
      const int rp = right.sec == Sec::kDebug ? 3 : right.profile;
      return lp != rp ? lp < rp : GroupFor(left) < GroupFor(right);
    });
    return indices;
  }();
  return order;
}

// A parameter's checkbox or slider; true when it changed. A control whose
// need (Param::needs) is unmet is disabled: older frameworks without disabled
// widgets show an inert grey label instead.
bool ParamControl(const Param& p, const char* label, bool* flag, float* value,
                  bool sentiment_enabled = true, bool vanilla_enabled = true) {
  const bool scene = p.needs == Param::kScene;
  const bool disabled =
      (scene && !vanilla_enabled) ||
      (p.needs == Param::kSentiment &&
       (!config.sentiment_available || !sentiment_enabled));
  const char* reason = scene ? "enable Vanilla / scene emotion inputs first"
                       : config.sentiment_available
                           ? "enable Sentiment Analysis first"
                           : "model not installed";
  if (disabled && !ig.BeginDisabled) {
    if (ig.TextColored)
      ig.TextColored(ImVec4{.5f, .5f, .5f, 1}, "%s (%s)", p.label, reason);
    else
      ig.Text("%s (%s)", p.label, reason);
    return false;
  }
  if (disabled) ig.BeginDisabled(true);
  const bool changed =
      flag ? ig.Checkbox(label, flag)
           : ig.SliderFloat(label, value, p.lo, p.hi, "%.3g", 0);
  if (disabled) ig.EndDisabled();
  return !disabled && changed;
}

// Common prefix of Menu Framework 2/3's WindowInterface. Only Framework 3
// exports the input callback API; older versions retain page-based controls.
struct MenuWindow {
  std::atomic<bool> open{false};
};
MenuWindow *npc_window = nullptr, *main_window = nullptr;
bool (*blocking_window)() = nullptr;
const skse::SKSETaskInterface* menu_tasks = nullptr;
bool npc_input = false;
std::atomic<bool> bind_npc_key{false};
std::atomic<bool> overrides_drawn{false};
std::atomic<bool> consumed_keys[256]{};
SRWLOCK npc_menu_lock = SRWLOCK_INIT;
uint64_t menu_epoch = 0;
bool target_pending = false;
NpcSetting& selected_npc = *new NpcSetting;

uint8_t* Singleton(skse::Rel id) {
  const auto address = skse::Addr(id);
  return address ? *reinterpret_cast<uint8_t**>(address) : nullptr;
}

bool CanTargetNpc() {
  return !skse::IsVr() && menu_tasks && menu_tasks->AddTask &&
         skse::Addr(kIdCrosshairPick);
}

// Input callbacks run on the game's input thread. Do not steal keys from
// text entry, paused/item/application menus or other framework windows.
bool NpcHotkeyAllowed() {
  if (!CanTargetNpc() || !blocking_window || blocking_window()) return false;
  auto* ui = Singleton(kIdUI);
  auto* controls = Singleton(kIdControlMap);
  if (!ui || !controls) return false;
  // The top ControlMap context must be gameplay (0), including when a
  // mod's inventory/console menu deliberately does not pause the game.
  const auto* contexts = Field<uint32_t*>(controls, kControlMapContexts.Get());
  const auto count = Field<uint32_t>(controls, kControlMapNumContexts.Get());
  if (count && (!contexts || count > 64 || contexts[count - 1] != 0))
    return false;
  return !Field<uint32_t>(ui, kUiNumPausesGame) &&
         !Field<uint32_t>(ui, kUiNumItemMenus) &&
         !Field<uint32_t>(ui, kUiNumApplicationMenus) &&
         !Field<uint8_t>(ui, kUiModal) &&
         !Field<int8_t>(controls, kControlMapTextEntry.Get());
}

class NpcTargetTask final : public skse::TaskDelegate {
 public:
  explicit NpcTargetTask(uint64_t epoch) : epoch_(epoch) {}
  void Run() override {
    {
      ScopedLock guard(npc_menu_lock, ScopedLock::kShared);
      if (epoch_ != menu_epoch) return;
    }
    // No engine pointers escape this task. A reference owns the actor while
    // we copy its base identity, so despawn and load cannot leave a UI pointer.
    NpcSetting npc;
    if (auto* pick = Singleton(kIdCrosshairPick)) {
      ScopedReference ref(Field<uint32_t>(pick, kCrosshairTarget));
      npc = DescribeNpc(ref.get());
    }
    ScopedLock guard(npc_menu_lock);
    if (epoch_ != menu_epoch) return;
    selected_npc = std::move(npc);
    target_pending = false;
    if (npc_window) npc_window->open = true;
  }
  void Dispose() override { delete this; }

 private:
  uint64_t epoch_;
};

void RequestNpcTarget() {
  if (!CanTargetNpc() || !npc_window) return;
  uint64_t epoch;
  {
    ScopedLock guard(npc_menu_lock);
    if (target_pending) return;
    target_pending = true;
    epoch = menu_epoch;
  }
  menu_tasks->AddTask(new NpcTargetTask(epoch));
}

// RE::InputEvent / IDEvent / ButtonEvent ABI. Framework calls this once per
// event; true consumes that event. Keyboard only, no polling or extra hook.
bool NpcInput(uint8_t* event) {
  if (!event || Field<uint32_t>(event, 0x0C) != 0 ||
      Field<uint32_t>(event, 0x08) != 0)
    return false;
  const int key = Field<uint32_t>(event, 0x20);
  const float value = Field<float>(event, 0x28);
  const float held = Field<float>(event, 0x2C);
  if (key < 1 || key > 255) return false;
  if (consumed_keys[key]) {
    if (value == 0) consumed_keys[key] = false;
    if (value <= 0 || held != 0) return true;
    // The framework may hide a release while another widget owns input.
    consumed_keys[key] = false;  // a fresh down is a new press
  }
  if (value <= 0 || held != 0) return false;
  if (bind_npc_key) {
    if (main_window ? !main_window->open : !blocking_window()) {
      bind_npc_key = false;
      return false;
    }
    bind_npc_key = false;
    consumed_keys[key] = true;
    if (key != 1) {
      SetNpcHotkey(key);
      ScheduleSaveIni();
    }
    return true;  // Escape cancels rebinding
  }
  if (npc_window && npc_window->open) {
    if (key != 1) return false;
    npc_window->open = false;
    consumed_keys[key] = true;
    return true;
  }
  if (!key || key != NpcHotkey() || !NpcHotkeyAllowed()) return false;
  consumed_keys[key] = true;
  RequestNpcTarget();
  return true;
}

std::string NpcKeyName() {
  const int key = NpcHotkey();
  if (!key) return "Not assigned";
  char name[128];
  const LONG scan = ((key & 0x7F) << 16) | ((key & 0x80) ? (1 << 24) : 0);
  if (GetKeyNameTextA(scan, name, sizeof name)) return name;
  return "Key " + std::to_string(key);
}

void RenderNpcWindow() {
  bool open = true;
  if (ig.SetNextWindowSize) ig.SetNextWindowSize(ImVec2{650, 650}, 4);
  if (ig.Begin("NPC overrides##BFA", &open, 0)) {
    NpcSetting npc;
    {
      ScopedLock guard(npc_menu_lock, ScopedLock::kShared);
      npc = selected_npc;
    }
    if (npc.key.empty()) {
      ig.Text("Look at an NPC, then open this window again.");
      ig.Text(
          "The player and NPCs with temporary base records cannot be "
          "selected.");
    } else {
      for (const auto& entry : SnapshotOverrides())
        if (entry.key == npc.key) {
          npc = entry;
          break;
        }
      ig.Text("%s", npc.label.c_str());
      if (ig.IsItemHovered(0)) ig.SetTooltip("%s", npc.key.c_str());
      ig.Text("Applies to this NPC base, including copies, across all saves.");
      ig.Text(
          "Edits save automatically. Untouched sliders follow the global "
          "profile.");
      bool enabled = !npc.blocked;
      bool changed =
          ig.Checkbox("Enable facial animations for this NPC", &enabled);
      npc.blocked = !enabled;
      ig.Text("Global enable, race and Sentiment Analysis settings still apply.");
      const Tune defaults = SnapshotTune();
      FaceTune inherited = defaults.ForSex(npc.female);
      Group section = Group::kAuto;
      for (size_t index : MenuOrder()) {
        const Param& param = kParams[index];
        if (!NpcParamSupported(param)) continue;
        if (GroupFor(param) != section) {
          section = GroupFor(param);
          ig.SeparatorText(GroupTitle(section));
          GroupHelp(section);
        }
        auto value = npc.values.find(param.key);
        const float fallback =
            param.profile == 0                    ? *param.FaceValue(inherited)
            : !strcmp(param.key, "SentimentAnalysis") ? defaults.sentiment_analysis
            : !strcmp(param.key, "SentimentMultiplier")
                ? defaults.sentiment_multiplier
            : !strcmp(param.key, "GazeStrength")  ? defaults.gaze_strength
            : !strcmp(param.key, "DisableTongue") ? defaults.disable_tongue
                                                  : defaults.conversation_gaze;
        float current = value == npc.values.end() ? fallback : value->second;
        const std::string label =
            std::string(param.label) + "##npc_" + param.key;
        bool checked = current != 0;
        const auto sentiment = npc.values.find("SentimentAnalysis");
        const bool sentiment_enabled =
            defaults.sentiment_analysis &&
            (param.flag || sentiment == npc.values.end() || sentiment->second != 0);
        if (ParamControl(param, label.c_str(),
                         param.flag ? &checked : nullptr, &current,
                         sentiment_enabled, defaults.vanilla_emotions)) {
          npc.values[param.key] = param.flag ? float(checked) : current;
          changed = true;
        }
        if (ig.IsItemHovered(kHoverDisabled))
          ig.SetTooltip("%s", ParamTip(param, true));
        ig.SameLine(0, -1);
        if (npc.values.count(param.key)) {
          if (ig.Button((std::string("Use global##") + param.key).c_str(),
                        ImVec2{0, 0})) {
            npc.values.erase(param.key);
            changed = true;
          }
        } else {
          ig.Text("Global");
        }
      }
      if (ig.Button("Reset NPC to global settings", ImVec2{0, 0})) {
        npc.blocked = false;
        npc.values.clear();
        changed = true;
      }
      if (changed) {
        SetNpcOverride(npc);
        {
          ScopedLock guard(npc_menu_lock);
          selected_npc = npc;
        }
        ScheduleSaveIni();
      }
    }
    if (ig.Button("Close", ImVec2{0, 0})) open = false;
  }
  ig.End();
  if (!open) npc_window->open = false;
}

void RenderOverrides() {
  overrides_drawn = true;
  ig.SeparatorText("NPC overrides");
  ig.Text(
      "Edit playback, strength and emotion per NPC. Changes save "
      "automatically.");
  ig.Text("Choices apply across saves and survive load-order changes.");
  if (npc_input) {
    ig.Text("Target hotkey: %s", NpcKeyName().c_str());
    if (ig.Button(
            bind_npc_key ? "Press a key (Escape cancels)" : "Assign hotkey",
            ImVec2{0, 0}))
      bind_npc_key = !bind_npc_key;
    ig.SameLine(0, -1);
    if (ig.Button("Unbind", ImVec2{0, 0})) {
      bind_npc_key = false;
      SetNpcHotkey(0);
      ScheduleSaveIni();
    }
    ig.Text(
        "Choose an unused keyboard key. Aim at an NPC, press it, then edit "
        "their settings.");
  } else {
    ig.Text("Target hotkeys require SKSE Menu Framework 3 and Skyrim SE/AE.");
  }
  if (CanTargetNpc() && npc_window &&
      ig.Button("Select NPC under crosshair", ImVec2{0, 0}))
    RequestNpcTarget();
  auto entries = SnapshotOverrides();
  constexpr size_t kPageSize = 12;
  static size_t page = 0;
  const size_t pages =
      std::max(size_t{1}, (entries.size() + kPageSize - 1) / kPageSize);
  page = std::min(page, pages - 1);
  ig.SeparatorText("Saved NPCs");
  if (ig.Button("Previous##npc", ImVec2{0, 0}) && page) --page;
  ig.SameLine(0, -1);
  ig.Text("%zu / %zu", page + 1, pages);
  ig.SameLine(0, -1);
  if (ig.Button("Next##npc", ImVec2{0, 0}) && page + 1 < pages) ++page;
  for (size_t i = page * kPageSize;
       i < std::min(entries.size(), (page + 1) * kPageSize); ++i) {
    const auto& npc = entries[i];
    ig.Text("%s%s", npc.label.c_str(), npc.id ? "" : " (not loaded)");
    if (ig.IsItemHovered(0)) ig.SetTooltip("%s", npc.key.c_str());
    ig.SameLine(0, -1);
    if (npc_window && ig.Button(("Edit##" + npc.key).c_str(), ImVec2{0, 0})) {
      ScopedLock guard(npc_menu_lock);
      selected_npc = npc;
      npc_window->open = true;
    }
    ig.SameLine(0, -1);
    if (ig.Button(("Remove##" + npc.key).c_str(), ImVec2{0, 0})) {
      RemoveNpcOverride(npc);
      {
        ScopedLock guard(npc_menu_lock);
        if (selected_npc.key == npc.key) {
          selected_npc.blocked = false;
          selected_npc.values.clear();
        }
      }
      ScheduleSaveIni();
    }
  }
  if (entries.empty()) ig.Text("No saved NPC overrides.");
}

void NpcMenuEvent(int type) {
  // Framework's AfterRender event: changing pages or closing the menu ends
  // a pending rebind, so typing on another mod's page cannot assign a key.
  if (type == 4 && !overrides_drawn.exchange(false)) bind_npc_key = false;
}

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
void Tip(const void* id, const char* details, int hover_flags = 0) {
  constexpr double kRest = 0.4;  // s
  constexpr size_t kColumns = 60;
  static const void* over;
  static double since, seen;  // the pointer came to over, was last on it
  if (!ig.IsItemHovered(hover_flags)) return;
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
  if (!ready) {
    if (!st.startup_error.empty())
      Notice(kRed, "Not running: " + st.startup_error);
    else
      Notice(kYellow, st.waiting_for_game
                          ? "The model will load when you enter a game."
                          : "Starting.");
  }
  if (!st.net_error.empty()) {
    // the problem, its cause and the advice; the rest on the last of them
    for (const auto& line : st.net_error) Notice(kRed, line);
    if (st.net_retry_at > 0)
      snprintf(b, sizeof b, "\nInference retry in %.0f s. Vanilla lip-sync plays meanwhile.",
               std::max(0.0, st.net_retry_at - Now()));
    else
      snprintf(b, sizeof b, "\nChoose Try GPU again or select Force CPU. Vanilla lip-sync plays meanwhile.");
    Tip(&kNetTip, ("Runtime message: " + st.net_detail + b).c_str());
  } else if (ready && tune.enabled && !net_loaded) {
    Notice(kYellow, "Loading the model.");
  }
  if (ready && tune.enabled && !tune.force_cpu &&
      (!st.net_error.empty() ||
       (tune.sentiment_analysis && config.sentiment_available &&
        st.sentiment_info.find("unavailable") != std::string::npos))) {
    if (ig.Button("Try GPU again", {0, 0})) RequestGpuRetry();
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
  static bool details = false;
  if (ig.Button(details ? "Hide status details" : "Show status details",
                ImVec2{0, 0}))
    details = !details;
  if (details) {
    ig.Text("GPU route: %s (change installer/INI, then restart)",
            config.independent_device ? "Independent" : "4.0-style shared");
    if (gpu.known) {
      ig.Text("GPU: %s, %.0f / %.0f MB VRAM", gpu.adapter.c_str(), gpu.used_mb,
              gpu.budget_mb);
      Tip(&gpu, "The VRAM the game uses, of what Windows grants it.");
    } else {
      ig.Text("GPU: %s",
              gpu.adapter.empty() ? "not found" : gpu.adapter.c_str());
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
  }
  RenderProblems(st, queued, tune, gpu);
  // The worker publishes the applied model independently of the selection.
  if (!st.model_error.empty())
    Notice(kRed, st.model_error);
  else if (st.switching_model ||
           tune.SelectedModel() != st.network ||
           (tune.force_cpu != !st.want_gpu))
    Notice(kYellow,
           "Switching model; new dialogue uses it when loading finishes.");
  if (details) {
    ig.Text("Sentiment Analysis: %s", st.sentiment_info.c_str());
    for (int sex = 0; sex < 2; ++sex)
      ig.Text("Expressive Facial Animation - %s Edition: %s",
              sex ? "Female" : "Male",
              config.expressive_faces[sex] ? "Installed mod detected"
                                           : "Not detected");
    snprintf(b, sizeof b,
             "Mouth strength: %g with a detected edition; 1.0 by default. "
             "Detects original Male 1.21 and Female 1.7 assets at startup.",
             kExpressiveMouthScale);
    Tip(config.expressive_faces, b);
  }
  HelpText("Changes save automatically. Male and Female NPC profiles are "
           "separate; the Player profile applies to either sex. NPC overrides "
           "can replace profile values.");
  Group cur = Group::kAuto;
  static int selected_profile = 0;
  bool profiles_shown = false;
  bool changed = false;
  int profile = -2;  // no parameter's: the first one opens a group
  for (size_t i : MenuOrder()) {
    const Param& p = kParams[i];
    if (p.profile >= 0) {
      if (!profiles_shown) {
        profiles_shown = true;
        ig.SeparatorText("Character profile");
        static const char* const names[] = {"Male NPCs", "Female NPCs",
                                            "Player"};
        for (int tab = 0; tab < 3; ++tab) {
          if (tab) ig.SameLine(0, -1);
          if (ig.Button(names[tab], ImVec2{0, 0})) selected_profile = tab;
        }
        ig.Text("Editing: %s", names[selected_profile]);
        HelpText("Editing the selected profile. Each profile keeps its "
                 "settings. Player settings apply to either sex.");
      }
      if (p.profile != selected_profile) continue;
    }
    if (GroupFor(p) != cur || p.profile != profile) {
      cur = GroupFor(p);
      profile = p.profile;
      ig.SeparatorText(GroupTitle(cur));
      GroupHelp(cur);
    }
    snprintf(b, sizeof b, "%s##%zu", p.label, i);
    ScopedLock guard(settings_lock);
    if (ParamControl(p, b, p.flag, p.value, p.flag || tune.sentiment_analysis,
                     tune.vanilla_emotions)) {
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
    Tip(&p, ParamTip(p), kHoverDisabled);
  }
  ApplyLogging();
  ig.SeparatorText("Reset & cache");
  static const char kDefaults[] = "Reset settings to defaults";
  static const char kClear[] = "Clear generated animation cache";
  if (ig.Button(kDefaults, ImVec2{0, 0})) {
    ResetTune();
    changed = true;
  }
  if (changed) {
    WakeWorker();
    ScheduleSaveIni();
  }
  Tip(kDefaults,
      "Resets shared settings and ALL three profiles, including hidden "
      "profiles. Resets race choices too. Keeps model, CPU/GPU selection and "
      "logging. NPC overrides are managed separately.");
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

void InitMenu(const skse::SKSEInterface* iface) {
  menu_tasks = static_cast<const skse::SKSETaskInterface*>(
      iface->QueryInterface(skse::kInterfaceTask));
}

void MenuMessage(uint32_t type) {
  if (type != skse::kMessagePreLoadGame && type != skse::kMessagePostLoadGame &&
      type != skse::kMessageNewGame)
    return;
  ScopedLock guard(npc_menu_lock);
  ++menu_epoch;
  target_pending = false;
  selected_npc = {};
  bind_npc_key = false;
  for (auto& key : consumed_keys) key = false;
  if (npc_window) npc_window->open = false;
}

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
  if (!Import(m, "igBeginDisabled", ig.BeginDisabled) ||
      !Import(m, "igEndDisabled", ig.EndDisabled)) {
    ig.BeginDisabled = nullptr;
    ig.EndDisabled = nullptr;
  }
  Import(m, "igSetNextWindowSize", ig.SetNextWindowSize);
  Import(m, "igTextColored", ig.TextColored);
  if (!Import(m, "igPushTextWrapPos", ig.PushTextWrapPos) ||
      !Import(m, "igPopTextWrapPos", ig.PopTextWrapPos)) {
    ig.PushTextWrapPos = nullptr;
    ig.PopTextWrapPos = nullptr;
  }
  add("Better Facial Animations/Settings", RenderMenu);
  add("Better Facial Animations/Races", RenderRaces);
  add("Better Facial Animations/Overrides", RenderOverrides);
  MenuWindow* (*add_window)(void (*)());
  MenuWindow* (*get_main)();
  int64_t (*register_input)(bool (*)(uint8_t*));
  int64_t (*register_event)(void (*)(int), float);
  if (CanTargetNpc() && Import(m, "AddWindow", add_window) &&
      Import(m, "igBegin", ig.Begin) && Import(m, "igEnd", ig.End)) {
    npc_window = add_window(RenderNpcWindow);
    if (npc_window &&
        Import(m, "IsAnyBlockingWindowOpened", blocking_window) &&
        Import(m, "RegisterInpoutEvent", register_input) &&
        Import(m, "RegisterEventPriority", register_event) &&
        skse::Addr(kIdUI) && skse::Addr(kIdControlMap)) {
      // Older Framework 3 builds expose input callbacks but not GetMainWindow.
      if (Import(m, "GetMainWindow", get_main)) main_window = get_main();
      register_input(NpcInput);  // upstream export really is spelled Inpout
      register_event(NpcMenuEvent, 0);
      npc_input = true;
    }
  }
  Log("menu page registered (SKSE Menu Framework)");
}

}  // namespace bfa
