#include "a2f/inference/diagnostics.h"
// The settings page: SKSE Menu Framework (optional; its cimgui exports).
#include <windows.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <span>
#include <string>
#include <utility>

#include "a2f/animation/head_cache.h"
#include "a2f/inference/net_keeper.h"
#include "a2f/inference/ort_net.h"
#include "a2f/plugin/menu.h"
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
  bool (*BeginCombo)(const char*, const char*, int);
  void (*EndCombo)();
  bool (*Selectable)(const char*, bool, int, ImVec2);
  // optional
  void (*BeginDisabled)(bool);
  void (*EndDisabled)();
  void (*TextColored)(ImVec4, const char*, ...);
  void (*PushTextWrapPos)(float);
  void (*PopTextWrapPos)();
  bool (*Begin)(const char*, bool*, int);
  void (*End)();
  void (*SetNextWindowSize)(ImVec2, int);
  void (*Dummy)(ImVec2);
  void (*SetNextItemWidth)(float);
  float (*GetFontSize)();
  void (*SetWindowFontScale)(float);
  void (*GetContentRegionAvail)(ImVec2*);
  void (*BeginGroup)();
  void (*EndGroup)();
  void (*AlignTextToFramePadding)();
} ig;

// ImGuiHoveredFlags_AllowWhenDisabled (Menu Framework's cimgui ABI).
constexpr int kHoverDisabled = 1 << 10;
constexpr ImVec4 kHeadingPink{1.0f, .56f, .76f, 1};

// Scale relative to the user's font settings, and restore the host window.
class ScopedFontScale {
 public:
  explicit ScopedFontScale(float factor) {
    if (!ig.GetFontSize || !ig.SetWindowFontScale) return;
    original_size_ = ig.GetFontSize();
    ig.SetWindowFontScale(1);
    scale_ = original_size_ / ig.GetFontSize();
    ig.SetWindowFontScale(scale_ * factor);
  }
  ~ScopedFontScale() {
    if (scale_ <= 0) return;
    // Recent ImGui versions round the reported size. Correct that rounding
    // so repeated headings cannot slowly change the user's body text size.
    for (int attempt = 0; attempt < 4; ++attempt) {
      ig.SetWindowFontScale(scale_);
      const float restored = ig.GetFontSize();
      if (restored == original_size_) break;
      scale_ *= original_size_ / restored;
    }
  }
  ScopedFontScale(const ScopedFontScale&) = delete;
  ScopedFontScale& operator=(const ScopedFontScale&) = delete;

 private:
  float scale_ = 0;
  float original_size_ = 0;
};

float FontSize() { return ig.GetFontSize ? ig.GetFontSize() : 20; }

void Heading(const char* title, float scale) {
  ScopedFontScale font(scale);
  if (ig.PushTextWrapPos) ig.PushTextWrapPos(0);
  if (ig.TextColored)
    ig.TextColored(kHeadingPink, "%s", title);
  else
    ig.Text("%s", title);
  if (ig.PopTextWrapPos) ig.PopTextWrapPos();
}

const char* ParamTip(const Param& p, bool npc = false) {
  if (p.needs == Param::kSentiment && !SentimentAvailable(SnapshotTune()))
    return "Sentiment Analysis is unavailable because the optional sentiment "
           "model is not installed in this copy of Better Facial Animations. "
           "Both downloads include Performance; only the with-sentiment "
           "download "
           "includes Full.\n\n"
           "Download and install the release that includes the sentiment "
           "model. Full uses emotion_fp16.onnx; Performance uses "
           "emotion_performance.onnx in "
           "Data/SKSE/Plugins/BetterFacialAnimations. Restart Skyrim after "
           "installing it, then enable Sentiment Analysis.\n\n"
           "Full uses about 745 MB "
           "of VRAM on GPU and adds processing time. Force CPU "
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
                                     Group::kInputs,   Group::kDebug,
                                     Group::kGaze};
  static_assert(std::size(by_section) == static_cast<size_t>(Sec::kGaze) + 1);
  return p.group != Group::kAuto ? p.group
                                 : by_section[static_cast<int>(p.sec)];
}
const char* GroupTitle(Group group) {
  static const char* const titles[] = {"General",
                                       "Models & processing",
                                       "Emotion sources",
                                       "Speech timing",
                                       "Expression transitions",
                                       "Facial movement strength",
                                       "Gaze",
                                       "Emotion input amounts",
                                       "Maximum emotion inputs",
                                       "Diagnostics"};
  static_assert(std::size(titles) == static_cast<size_t>(Group::kAuto));
  return titles[static_cast<int>(group)];
}
void Space(float height = 6) {
  if (ig.Dummy) ig.Dummy({0, height});
}
void Section(const char* title) {
  Space(FontSize() * 1.25f);
  Heading(title, 1.25f);
  ig.SeparatorText("");
  Space(FontSize() * .35f);
}
void HelpText(const char* text) {
  if (ig.PushTextWrapPos) ig.PushTextWrapPos(0);
  ig.Text("%s", text);
  if (ig.PopTextWrapPos) ig.PopTextWrapPos();
}
void GroupHelp(Group group) {
  switch (group) {
    case Group::kModel:
      HelpText(
          "Each model has its own selection. Hover over a choice for its "
          "purpose, resource use and limitations.");
      break;
    case Group::kSources:
      HelpText(
          "Combine authored scene emotion with the selected Audio2Emotion "
          "model. Turning both sources off keeps audio-driven lip-sync.");
      break;
    case Group::kMovement:
      HelpText(
          "Scales generated facial movement: 0 = none, 1 = normal. "
          "These are movement multipliers.");
      break;
    case Group::kGaze:
      HelpText(
          "The eyes of the NPC you are talking to, and where they look. "
          "Each chance is how often a pause in a line with that emotion "
          "leaves them looking at you; mixed emotions blend. Afraid, "
          "surprised and disgusted are in the INI.");
      break;
    case Group::kInputs:
      HelpText(
          "Adjust each source before the inputs are combined. The "
          "maximum emotion inputs below limit the combined result. "
          "Applies next line.");
      break;
    case Group::kLimits:
      HelpText(
          "Maximum emotion inputs after all sources and multipliers are "
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
      const int lp = left.sec == Sec::kDebug ? kNumProfiles : left.profile;
      const int rp = right.sec == Sec::kDebug ? kNumProfiles : right.profile;
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
                  bool sentiment_enabled = true, bool vanilla_enabled = true,
                  bool performance_sentiment = false) {
  const bool available = performance_sentiment
                             ? config.performance_sentiment_available
                             : config.sentiment_available;
  const bool scene = p.needs == Param::kScene;
  const bool disabled =
      (scene && !vanilla_enabled) ||
      (p.needs == Param::kSentiment && (!available || !sentiment_enabled));
  const char* reason = scene ? "enable Vanilla / scene emotion inputs first"
                       : available ? "enable Sentiment Analysis first"
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
  return !skse::IsVr() && menu_tasks && skse::Addr(kIdCrosshairPick);
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
  const auto& stack = reinterpret_cast<RE::ControlMap*>(controls)
                          ->GetRuntimeData()
                          .contextPriorityStack;
  const auto* contexts = stack.data();
  const auto count = stack.size();
  if (count &&
      (!contexts || count > 64 ||
       contexts[count - 1] != RE::UserEvents::INPUT_CONTEXT_ID::kGameplay))
    return false;
  return !reinterpret_cast<RE::UI*>(ui)->numPausesGame &&
         !reinterpret_cast<RE::UI*>(ui)->numItemMenus &&
         !reinterpret_cast<RE::UI*>(ui)->numApplicationMenus &&
         !reinterpret_cast<RE::UI*>(ui)->modal &&
         !reinterpret_cast<RE::ControlMap*>(controls)
              ->GetRuntimeData()
              .textEntryCount;
}

void SelectNpcTarget(uint64_t epoch) {
  {
    ScopedLock guard(npc_menu_lock, ScopedLock::kShared);
    if (epoch != menu_epoch) return;
  }
  // No engine pointers escape this task. A reference owns the actor while
  // we copy its base identity, so despawn and load cannot leave a UI pointer.
  NpcSetting npc;
  if (auto* pick = Singleton(kIdCrosshairPick)) {
    ScopedReference ref(reinterpret_cast<RE::CrosshairPickData*>(pick)
                            ->GetActiveTarget()
                            .native_handle());
    npc = DescribeNpc(ref.get());
  }
  ScopedLock guard(npc_menu_lock);
  if (epoch != menu_epoch) return;
  selected_npc = std::move(npc);
  target_pending = false;
  if (npc_window) npc_window->open = true;
}

void RequestNpcTarget() {
  if (!CanTargetNpc() || !npc_window) return;
  uint64_t epoch;
  {
    ScopedLock guard(npc_menu_lock);
    if (target_pending) return;
    target_pending = true;
    epoch = menu_epoch;
  }
  menu_tasks->AddTask([epoch] { SelectNpcTarget(epoch); });
}

// RE::InputEvent / IDEvent / ButtonEvent ABI. Framework calls this once per
// event; true consumes that event. Keyboard only, no polling or extra hook.
bool NpcInput(uint8_t* event) {
  const auto* button = reinterpret_cast<const RE::ButtonEvent*>(event);
  if (!button || button->eventType != RE::INPUT_EVENT_TYPE::kButton ||
      button->device != RE::INPUT_DEVICE::kKeyboard)
    return false;
  const int key = button->GetIDCode();
  const float value = button->Value(), held = button->HeldDuration();
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
      SaveIni();
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
      ig.Text(
          "Global enable, race and Sentiment Analysis settings still apply.");
      const Tune defaults = SnapshotTune();
      FaceTune inherited = defaults.ForSex(npc.female);
      Group section = Group::kAuto;
      for (size_t index : MenuOrder()) {
        const Param& param = kParams[index];
        if (!NpcParamSupported(param) || !param.menu) continue;
        if (GroupFor(param) != section) {
          section = GroupFor(param);
          Section(GroupTitle(section));
          GroupHelp(section);
        }
        auto value = npc.values.find(param.key);
        const float fallback =
            param.profile == kProfileMale ? *param.FaceValue(inherited)
            : !strcmp(param.key, "SentimentAnalysis")
                ? defaults.sentiment_analysis
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
            (param.flag || sentiment == npc.values.end() ||
             sentiment->second != 0);
        if (ParamControl(param, label.c_str(), param.flag ? &checked : nullptr,
                         &current, sentiment_enabled, defaults.vanilla_emotions,
                         defaults.performance_sentiment)) {
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
  Section("NPC overrides");
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
      SaveIni();
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
  Section("Saved NPCs");
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
      SaveIni();
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
                          ? "The model will load after game entry and initial loading."
                          : "Starting.");
  }
  if (!st.net_error.empty()) {
    // the problem, its cause and the advice; the rest on the last of them
    for (const auto& line : st.net_error) Notice(kRed, line);
    if (st.net_retry_at > 0)
      snprintf(b, sizeof b,
               "\nInference retry in %.0f s. Vanilla lip-sync plays meanwhile.",
               std::max(0.0, st.net_retry_at - Now()));
    else
      snprintf(b, sizeof b,
               "\nChoose Try GPU again or select Force CPU. Vanilla lip-sync "
               "plays meanwhile.");
    Tip(&kNetTip, ("Runtime message: " + st.net_detail + b).c_str());
  } else if (ready && tune.enabled && !net_loaded) {
    Notice(kYellow, "Loading the model.");
  }
  if (ready && tune.enabled &&
      ((!tune.force_cpu && !st.net_error.empty()) ||
       (tune.EmotionUsesGpu() && tune.sentiment_analysis &&
        SentimentAvailable(tune) &&
        st.sentiment_info.find("unavailable") != std::string::npos))) {
    if (ig.Button("Try GPU again", {0, 0})) RequestGpuRetry();
  }
  for (const auto& w : a2f::GpuWarnings(gpu)) Notice(kYellow, w);
  const double speed = st.audio_sec / std::max(st.gen_sec, 1e-6);
  if (st.lines >= 5 && speed < 2 && (st.net_gpu || !st.want_gpu)) {
    snprintf(b, sizeof b, "Slow (%.1fx realtime): lines start late.", speed);
    Notice(kYellow, b);
    Tip(&kSlowTip, (st.network == a2f::kNetPerformance ||
                    st.network == a2f::kNetUltraPerformance) &&
                           !st.net_gpu
                       ? "The selected student model runs on the CPU; check "
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

std::string DiagnosticReport(const Stats& st, const Tune& tune,
                             const a2f::diagnostics::Snapshot& snapshot) {
  std::string report =
      std::string("Better Facial Animations ") + kVersion + "\n";
  report += std::string("Requested: ") + (tune.force_cpu ? "CPU" : "GPU") +
            "; adapter: " + std::to_string(tune.gpu_adapter) + "; GPU route: " +
            (config.independent_device ? "independent" : "shared") + "\n";
  report += "Applied adapter: " + std::to_string(st.gpu_adapter) + "\n";
  report += "Requested emotion device: " +
            (tune.emotion_device == kCpuDevice
                 ? std::string("CPU")
                 : "GPU " + std::to_string(tune.emotion_device)) +
            "\n";
  report +=
      "Current model: " + (st.net_info.empty() ? "not loaded" : st.net_info) +
      "\n";
  report += "Startup: " + st.startup_error + "\n";
  report += "Face runtime: " + st.net_detail +
            "\nSentiment: " + st.sentiment_info + "\n";
  report +=
      "Independent Devices runtime API: " + snapshot.independent_api + "\n";
  const auto now = GetTickCount64();
  for (size_t i = 0; i != snapshot.components.size(); ++i) {
    const auto& state = snapshot.components[i];
    report += i ? "Sentiment startup: " : "Face startup: ";
    if (!state.attempt) {
      report += "not attempted\n";
      continue;
    }
    report +=
        "attempt " + std::to_string(state.attempt) + ", " +
        (state.gpu ? (state.independent ? "independent GPU" : "shared GPU")
                   : "CPU") +
        ", " +
        (state.loading ? "in progress"
         : state.ready ? "last startup succeeded"
                       : "not ready") +
        ", stage: " + a2f::diagnostics::Name(state.stage) + ", elapsed " +
        std::to_string(
            ((state.loading ? now : state.finished) - state.started) / 1000) +
        " s\n";
    if (state.loading)
      report += "At this stage: " +
                std::to_string((now - state.stage_started) / 1000) +
                " s (completion unknown)\n";
    if (!state.error.empty()) report += "Error: " + state.error + "\n";
    if (state.last_failure.attempt)
      report += "Last failed attempt " +
                std::to_string(state.last_failure.attempt) + " (" +
                (state.last_failure.gpu
                     ? (state.last_failure.independent ? "independent GPU"
                                                       : "shared GPU")
                     : "CPU") +
                ") at " + a2f::diagnostics::Name(state.last_failure.stage) +
                ": " + state.last_failure.detail + "\n";
  }
  report += "System/modules snapshot (passive; no GPU capability probe):\n";
  for (const auto& line : snapshot.system) report += line + "\n";
  return report;
}

bool CopyReport(const std::string& report) {
  const int count =
      MultiByteToWideChar(CP_UTF8, 0, report.c_str(), -1, nullptr, 0);
  if (!count) return false;
  HGLOBAL memory = GlobalAlloc(GMEM_MOVEABLE, count * sizeof(wchar_t));
  if (!memory) return false;
  auto* data = static_cast<wchar_t*>(GlobalLock(memory));
  if (!data) {
    GlobalFree(memory);
    return false;
  }
  const bool converted =
      MultiByteToWideChar(CP_UTF8, 0, report.c_str(), -1, data, count) != 0;
  GlobalUnlock(memory);
  if (!converted || !OpenClipboard(GetForegroundWindow())) {
    GlobalFree(memory);
    return false;
  }
  const bool copied =
      EmptyClipboard() && SetClipboardData(CF_UNICODETEXT, memory);
  CloseClipboard();
  if (!copied) GlobalFree(memory);
  return copied;
}

void RenderDiagnostics(const Stats& st, const Tune& tune) {
  const auto snapshot = a2f::diagnostics::State().Read();
  for (size_t i = 0; i != snapshot.components.size(); ++i) {
    const auto& state = snapshot.components[i];
    if (!state.loading) continue;
    Notice(kYellow,
           std::string(i ? "Sentiment: " : "Face: ") +
               a2f::diagnostics::Name(state.stage) + " (" +
               std::to_string((GetTickCount64() - state.stage_started) / 1000) +
               " s)");
  }
  static bool show = false;
  static std::string copy_status;
  if (ig.Button(show ? "Hide diagnostic report" : "Show diagnostic report",
                {0, 0}))
    show = !show;
  if (ig.Button("Copy diagnostic report", {0, 0})) {
    copy_status = CopyReport(DiagnosticReport(st, tune, snapshot))
                      ? "Diagnostic report copied."
                      : "Clipboard unavailable. The report is shown below.";
    show = true;
  }
  if (!copy_status.empty()) HelpText(copy_status.c_str());
  if (show) HelpText(DiagnosticReport(st, tune, snapshot).c_str());
}

std::string GpuLabel(int index, const a2f::GpuAdapterList& list) {
  for (const auto& adapter : list.adapters)
    if (adapter.index == index)
      return "GPU " + std::to_string(index) + ": " + adapter.name;
  return "GPU " + std::to_string(index) +
         (list.complete ? " (unavailable)"
          : index == 0  ? " (default)"
                        : " (saved selection)");
}

bool RenderProcessingDevice(const Tune& tune, bool emotion = false) {
  const auto list = SnapshotGpuAdapters();
  // Resolve saved adapter names once after game entry, even while closed.
  // Discovery runs on the worker; rendering never queries the driver.
  if (!list.requested && ready) RequestGpuAdapters();
  const int device = emotion ? tune.emotion_device
                             : (tune.force_cpu ? kCpuDevice : tune.gpu_adapter);
  const std::string preview =
      device == kCpuDevice ? "CPU (Force CPU)" : GpuLabel(device, list);
  const char* title = emotion ? "Audio2Emotion processing device"
                              : "Audio2Face processing device";
  if (!ig.BeginCombo || !ig.EndCombo || !ig.Selectable) {
    // Older Menu Framework without combos: CPU or the saved GPU.
    bool cpu = device == kCpuDevice;
    const std::string label = std::string(title) + ": CPU (Force CPU)";
    if (ig.Checkbox(label.c_str(), &cpu)) {
      if (emotion)
        SetEmotionDevice(cpu ? kCpuDevice : std::max(0, tune.gpu_adapter));
      else
        SetProcessingDevice(cpu, tune.gpu_adapter);
      Log("%s: %s", title, cpu ? "CPU" : "GPU");
      return true;
    }
    HelpText(
        ("Unticked: " +
         GpuLabel(emotion ? std::max(0, tune.emotion_device) : tune.gpu_adapter,
                  list) +
         ". Update SKSE Menu Framework to pick a GPU here, or set "
         "GpuAdapter / EmotionDevice in the INI.")
            .c_str());
    return false;
  }
  bool changed = false;
  ig.Text("%s", title);
  if (ig.SetNextItemWidth) ig.SetNextItemWidth(-1);
  if (ig.BeginCombo(emotion ? "##EmotionDevice" : "##Processing device",
                    preview.c_str(), 0)) {
    const auto choose = [&](const std::string& label, int value) {
      const bool selected = device == value;
      const std::string id = label + (emotion ? "##EmotionDevice" : "");
      if (ig.Selectable(id.c_str(), selected, 0, {0, 0}) && !selected) {
        if (emotion)
          SetEmotionDevice(value);
        else
          SetProcessingDevice(value == kCpuDevice,
                              value == kCpuDevice ? tune.gpu_adapter : value);
        Log("%s: %s", title, label.c_str());
        changed = true;
      }
    };
    choose("CPU (Force CPU)", kCpuDevice);
    // Default remains selectable before discovery, for legacy INIs and
    // recovery.
    if (!list.complete &&
        std::none_of(list.adapters.begin(), list.adapters.end(),
                     [](const auto& adapter) { return adapter.index == 0; }))
      choose(GpuLabel(0, list), 0);
    for (const auto& adapter : list.adapters)
      choose(GpuLabel(adapter.index, list), adapter.index);
    if (!ready)
      ig.Text("GPU list becomes available after entering a game.");
    else if (!list.complete)
      ig.Text("Detecting GPUs...");
    if (!list.error.empty()) HelpText(list.error.c_str());
    if (ready && list.complete &&
        ig.Button(
            emotion ? "Refresh GPU list##EmotionDevice" : "Refresh GPU list",
            {0, 0}))
      RequestGpuAdapters();
    ig.EndCombo();
  }
  return changed;
}

struct ModelOption {  // one dropdown row
  const char* name;
  int id;
  const char* tip;
};
constexpr ModelOption kEmotionOptions[] = {
    {"Disabled", kEmotionDisabled,
     "No voice-driven emotion is added. The Audio2Emotion session is unloaded "
     "after any current inference finishes, freeing its model memory. "
     "Authored scene emotion and Audio2Face lip-sync remain available."},
    {"Full", kEmotionFull,
     "The full Audio2Emotion model analyzes each spoken line to supply emotion "
     "to Audio2Face. Use this for dialogue beyond the Performance training "
     "corpus, including unfamiliar quest mods. It does not guarantee correct "
     "emotion on every line.\n\n"
     "Supports compatible AMD and NVIDIA GPUs through DirectML; a NVIDIA GPU "
     "is not required. Audio2Emotion has its own processing device dropdown. "
     "Uses about 745 MB of VRAM on GPU, plus processing "
     "time. CPU uses system RAM and adds CPU work."},
    {"Performance", kEmotionPerformance,
     "Enabled by default. Experimental model designed to memorize the Full "
     "model's answers for "
     "the dialogue used in training. It is not trained to reliably handle "
     "dialogue outside that corpus. Results on unfamiliar lines can be hit "
     "or miss, especially in obscure quest mods. It does not perfectly "
     "reproduce every training answer either.\n\n"
     "About 31 MB of VRAM on GPU. Uses the same AMD/NVIDIA DirectML or CPU "
     "options as Full. The bundled checkpoint is still experimental. "
     "Choose Full when coverage of unfamiliar dialogue matters."}};
// Disabled, the default model (Tune), the fallback, then the others as
// FaceModels() orders them.
const std::vector<ModelOption> kFaceOptions = [] {
  std::vector<ModelOption> options = {
      {"Disabled", kFaceDisabled,
       "Use the game's own facial animation and lip-sync. BFA stops "
       "generating animations and unloads both inference sessions after "
       "current work returns. Your Audio2Emotion selection is remembered for "
       "when you enable Audio2Face again; it only supplies inputs to "
       "Audio2Face."}};
  const auto add = [&](const ModelInfo& info) {
    options.push_back({info.name, info.network, info.tip});
  };
  const ModelInfo& standard = ModelFor(Tune().SelectedModel());
  add(standard);
  if (&standard != &FallbackFaceModel()) add(FallbackFaceModel());
  for (const ModelInfo& info : FaceModels())
    if (info.selected && &info != &standard) add(info);
  return options;
}();

// Whether an Audio2Emotion row's model is installed.
bool EmotionOptionAvailable(const ModelOption& option) {
  return option.id == kEmotionDisabled ||
         (option.id == kEmotionPerformance
              ? config.performance_sentiment_available
              : config.sentiment_available);
}

// Every row of options is offered; the first is shown for a selection that
// none of them names.
bool RenderModelChoice(const char* label, std::span<const ModelOption> options,
                       int selected, bool emotion, float width = 300,
                       bool label_above = false) {
  const ModelOption* current = &options.front();
  for (const ModelOption& option : options)
    if (option.id == selected) current = &option;
  const auto help = [](const ModelOption& option) { return option.tip; };
  const auto installed = [&](const ModelOption& option) {
    return !emotion || EmotionOptionAvailable(option);
  };
  if (!ig.BeginCombo || !ig.EndCombo || !ig.Selectable) {
    // Older Menu Framework without combos: one checkbox per option.
    ig.Text("%s", label);
    for (const ModelOption& option : options) {
      const bool available = installed(option);
      bool on = selected == option.id;
      const std::string row = std::string(option.name) +
                              (available ? "" : " (not installed)") + "##" +
                              label;
      const bool clicked = ig.Checkbox(row.c_str(), &on);
      Tip(&option, help(option));
      if (!clicked || !on || !available || selected == option.id) continue;
      if (emotion)
        SelectEmotionOption(option.id);
      else
        SelectFaceOption(option.id);
      return true;
    }
    return false;
  }
  if (label_above) {
    if (ig.AlignTextToFramePadding) ig.AlignTextToFramePadding();
    ig.Text("%s", label);
  }
  if (ig.SetNextItemWidth) ig.SetNextItemWidth(width);
  const std::string widget_label =
      label_above ? std::string("##") + label : label;
  const bool open = ig.BeginCombo(widget_label.c_str(), current->name, 0);
  // BeginCombo leaves the selected field as the last item, including when open.
  Tip(options.data(), help(*current));
  bool changed = false;
  if (open) {
    for (const ModelOption& option : options) {
      const bool available = installed(option);
      const std::string row = std::string(option.name) +
                              (available ? "" : " (not installed)") + "##" +
                              label;
      if (!available && ig.BeginDisabled) ig.BeginDisabled(true);
      const bool clicked =
          ig.Selectable(row.c_str(), selected == option.id, 0, {0, 0});
      if (!available && ig.BeginDisabled) ig.EndDisabled();
      const std::string details =
          std::string(help(option)) +
          (available
               ? ""
               : "\n\nModel not installed. Install it, then restart Skyrim.");
      Tip(&option, details.c_str(), kHoverDisabled);
      if (clicked && available && selected != option.id) {
        if (emotion)
          SelectEmotionOption(option.id);
        else
          SelectFaceOption(option.id);
        changed = true;
      }
    }
    ig.EndCombo();
  }
  return changed;
}

bool RenderEmotionControl(const char* key, const char* label, float width = 0) {
  for (size_t i = 0; i < kNumParams; ++i) {
    const Param& p = kParams[i];
    if (p.profile != kProfileShared || strcmp(p.key, key)) continue;
    if (width && ig.SetNextItemWidth) ig.SetNextItemWidth(width);
    char id[128];
    snprintf(id, sizeof id, "%s##%zu", label, i);
    const Tune tune = SnapshotTune();
    ScopedLock guard(settings_lock);
    const bool changed =
        ParamControl(p, id, p.flag, p.value, tune.sentiment_analysis,
                     tune.vanilla_emotions, tune.performance_sentiment);
    guard.Unlock();
    Tip(&p, ParamTip(p), kHoverDisabled);
    return changed;
  }
  return false;
}

// A model's graph fusion checkbox, under its processing device: inert on
// CPU, where it has no effect.
bool RenderGraphFusion(const char* key, bool cpu) {
  for (size_t i = 0; i < kNumParams; ++i) {
    const Param& p = kParams[i];
    if (p.profile != kProfileShared || strcmp(p.key, key)) continue;
    char id[128];
    snprintf(id, sizeof id, "%s##%s", p.label, key);
    if (cpu && !ig.BeginDisabled) {
      ig.Text("%s (GPU only)", p.label);
      Tip(&p, p.tip, kHoverDisabled);
      return false;
    }
    if (cpu) ig.BeginDisabled(true);
    ScopedLock guard(settings_lock);
    const bool changed = ig.Checkbox(id, p.flag) && !cpu;
    const bool on = *p.flag;
    guard.Unlock();
    if (cpu) ig.EndDisabled();
    Tip(&p, p.tip, kHoverDisabled);
    if (changed) Log("setting %s = %d", p.key, on);
    return changed;
  }
  return false;
}

bool RenderModels(const Tune& tune) {
  Section("Animation models");
  HelpText(
      "Hover over a selected model or an option to compare resources and "
      "tradeoffs.");
  Space();
  bool changed = RenderModelChoice("Audio2Face", kFaceOptions,
                                   tune.FaceChoice(), false, -1, true);
  Space(12);
  changed |= RenderProcessingDevice(tune);
  changed |= RenderGraphFusion("GraphFusion", SnapshotTune().force_cpu);

  Section("Emotion & sentiment");
  HelpText(
      "Choose voice-driven emotion and how strongly it shapes expressions.");
  Space(10);
  ImVec2 available{0, 0};
  if (ig.GetContentRegionAvail) ig.GetContentRegionAvail(&available);
  const float gap = FontSize();
  const bool columns = ig.BeginGroup && ig.EndGroup && ig.SetNextItemWidth &&
                       available.x >= FontSize() * 30;
  const float width = columns ? (available.x - gap) * .5f
                              : (available.x > 0 ? available.x : 300);
  if (columns) ig.BeginGroup();
  changed |= RenderModelChoice(
      "Audio2Emotion", kEmotionOptions, tune.SelectedEmotion(), true, width,
      true);
  if (columns) {
    ig.EndGroup();
    ig.SameLine(0, gap);
    ig.BeginGroup();
  } else {
    Space(10);
  }
  if (ig.AlignTextToFramePadding) ig.AlignTextToFramePadding();
  ig.Text("Sentiment strength");
  changed |= RenderEmotionControl("SentimentMultiplier", "", width);
  if (columns) ig.EndGroup();
  Space(12);
  changed |= RenderEmotionControl("VanillaEmotions",
                                  "Pass through vanilla emotion cues");
  HelpText(
      "Adds scene and dialogue emotions alongside the selected voice model.");
  Space(12);
  changed |= RenderProcessingDevice(SnapshotTune(), true);
  changed |= RenderGraphFusion("EmotionGraphFusion",
                               !SnapshotTune().EmotionUsesGpu());
  const Tune current = SnapshotTune();
  if (!current.enabled)
    HelpText(
        "Audio2Face is disabled. Your emotion choice is saved for when you "
        "enable it.");
  else if (current.sentiment_analysis && !SentimentAvailable(current))
    Notice(kYellow, "The selected Audio2Emotion model is not installed.");
  return changed;
}

void RenderMenu() {
  size_t cached, queued;
  const Stats st = SnapshotStats(cached, queued);
  // The worker's view at its last publish: rendering never probes the driver.
  const a2f::GpuState& gpu = st.gpu;
  Tune tune = SnapshotTune();
  char b[400];
  Space(6);
  Heading("Better Facial Animations", 1.6f);
  Space(4);
  HelpText("Models, expression and character tuning");
  Section("Runtime status");
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
      Tip(&gpu,
          "The VRAM the game used, of what Windows grants it, when the model "
          "last loaded or failed.");
    } else {
      ig.Text("GPU: %s",
              gpu.adapter.empty()
                  ? "not queried (keeps diagnostics independent of the driver)"
                  : gpu.adapter.c_str());
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
  RenderDiagnostics(st, tune);
  // The worker publishes the applied model independently of the selection.
  if (!st.model_error.empty())
    Notice(kRed, st.model_error);
  else if (tune.enabled &&
           (st.switching_model || tune.SelectedModel() != st.network ||
            (tune.force_cpu != !st.want_gpu) ||
            (!tune.force_cpu && (tune.gpu_adapter != st.gpu_adapter ||
                                 tune.face_graph_fusion != st.graph_fusion))))
    Notice(kYellow,
           "Switching model; new dialogue uses it when loading finishes.");
  if (details) {
    ig.Text("Sentiment Analysis: %s", st.sentiment_info.c_str());
    for (size_t sex = 0; sex < std::size(config.expressive_faces); ++sex)
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
  HelpText(
      "Changes save automatically. Male and Female NPC profiles are "
      "separate; the Player profile applies to either sex. NPC overrides "
      "can replace profile values.");
  Group cur = Group::kAuto;
  static int selected_profile = kProfileMale;
  bool profiles_shown = false;
  bool changed = RenderModels(tune);
  tune = SnapshotTune();
  int profile = -2;  // no parameter's: the first one opens a group
  for (size_t i : MenuOrder()) {
    const Param& p = kParams[i];
    if (!p.menu || p.own_control) continue;
    if (p.profile != kProfileShared) {
      if (!profiles_shown) {
        profiles_shown = true;
        Section("Character profile");
        for (int tab = 0; tab < kNumProfiles; ++tab) {
          if (tab) ig.SameLine(0, -1);
          if (ig.Button(kProfiles[tab].tab, ImVec2{0, 0}))
            selected_profile = tab;
        }
        ig.Text("Editing: %s", kProfiles[selected_profile].tab);
        HelpText(
            "Editing the selected profile. Each profile keeps its "
            "settings. Player settings apply to either sex.");
      }
      if (p.profile != selected_profile) continue;
    }
    if (GroupFor(p) != cur || p.profile != profile) {
      cur = GroupFor(p);
      profile = p.profile;
      Section(GroupTitle(cur));
      GroupHelp(cur);
    }
    snprintf(b, sizeof b, "%s##%zu", p.label, i);
    ScopedLock guard(settings_lock);
    if (ParamControl(p, b, p.flag, p.value, p.flag || tune.sentiment_analysis,
                     tune.vanilla_emotions, tune.performance_sentiment)) {
      changed = true;
      if (p.flag)
        Log("setting %s = %d", p.key, *p.flag);
      else
        Log("setting %s%s%s = %g",
            p.profile == kProfileShared ? "" : kProfiles[p.profile].section,
            p.profile == kProfileShared ? "" : " ", p.key, *p.value);
    }
    guard.Unlock();
    Tip(&p, ParamTip(p), kHoverDisabled);
    Space(7);
  }
  ApplyLogging();
  Section("Reset & cache");
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
      "profiles. Resets race choices too. Keeps the Audio2Face and "
      "Audio2Emotion models, their processing devices and graph fusion, "
      "and logging. NPC overrides are managed separately.");
  Space();
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
  Section("Apply facial animations by race");
  ig.Text("Unchecked races use vanilla animation. Changes save automatically.");
  ig.Text("Heads need compatible face morphs.");
  constexpr size_t kPageSize = 12;
  static size_t page = 0;
  const size_t pages =
      std::max(size_t{1}, (races.size() + kPageSize - 1) / kPageSize);
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
      SaveIni();
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
  Import(m, "igBeginCombo", ig.BeginCombo);
  Import(m, "igEndCombo", ig.EndCombo);
  Import(m, "igSelectable_Bool", ig.Selectable);
  Import(m, "igSetNextWindowSize", ig.SetNextWindowSize);
  Import(m, "igDummy", ig.Dummy);
  Import(m, "igSetNextItemWidth", ig.SetNextItemWidth);
  Import(m, "igGetFontSize", ig.GetFontSize);
  Import(m, "igSetWindowFontScale", ig.SetWindowFontScale);
  Import(m, "igGetContentRegionAvail", ig.GetContentRegionAvail);
  Import(m, "igAlignTextToFramePadding", ig.AlignTextToFramePadding);
  if (!Import(m, "igBeginGroup", ig.BeginGroup) ||
      !Import(m, "igEndGroup", ig.EndGroup)) {
    ig.BeginGroup = nullptr;
    ig.EndGroup = nullptr;
  }
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
    if (npc_window && Import(m, "IsAnyBlockingWindowOpened", blocking_window) &&
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
