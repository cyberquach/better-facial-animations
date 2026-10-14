// Exercise production UI/input code with a fake framework and game dispatcher.
#include <array>
#include <cstdarg>
#include <cstdlib>
#include <cstring>

#include "../src/plugin/menu.cc"

#define CHECK(x)                                           \
  do {                                                     \
    if (!(x)) {                                            \
      std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); \
      std::exit(1);                                        \
    }                                                      \
  } while (0)
namespace bfa {
const char kVersion[] = "test";
std::atomic<bool> ready{false}, net_loaded{false};
double menu_time = 0;
double Now() { return menu_time; }
Stats SnapshotStats(size_t& cached, size_t& queued) {
  cached = queued = 0;
  return {};
}
void ClearLines() {}
void ClearHeads() {}
void WakeWorker() {}
}  // namespace bfa
namespace a2f {
GpuState QueryGpu() { return {}; }
std::vector<std::string> GpuWarnings(const GpuState&) { return {}; }
}  // namespace a2f
extern "C" {
skse::SKSEPluginVersionData SKSEPlugin_Version = {1, 0, "npc_menu_test"};
}
namespace {
std::array<uint8_t, 0x400> ui_data{}, control_data{}, pick_data{}, actor_data{},
    base_data{}, file_data{};
uint8_t *ui = ui_data.data(), *controls = control_data.data(),
        *pick = pick_data.data();
uint8_t* actor = actor_data.data();
bool blocked = false;
int saves = 0;
std::vector<skse::TaskDelegate*> tasks;
bool Blocking() { return blocked; }
void Queue(skse::TaskDelegate* task) { tasks.push_back(task); }
bool Lookup(const uint32_t& handle, uint8_t** out) {
  *out = handle == 42 ? actor : nullptr;
  if (*out) ++bfa::Field<LONG>(*out, bfa::kRefrRefCount);
  return *out != nullptr;
}
void RunTasks() {
  auto pending = std::move(tasks);
  tasks.clear();
  for (auto* task : pending) {
    task->Run();
    task->Dispose();
  }
}
bool widgets_disabled = false, last_item_disabled = false, hovering = false;
std::string tooltip;
void BeginDisabled(bool disabled) { widgets_disabled = disabled; }
void EndDisabled() { widgets_disabled = false; }
std::string click;
std::vector<std::string> drawn_sliders;
std::map<std::string, float> slider_values;
bool Button(const char* label, bfa::ImVec2) { return click == label; }
bool Checkbox(const char* label, bool* value) {
  last_item_disabled = widgets_disabled;
  if (widgets_disabled || click != label) return false;
  *value = !*value;
  return true;
}
bool Slider(const char* label, float* value, float, float, const char*, int) {
  last_item_disabled = widgets_disabled;
  drawn_sliders.emplace_back(label);
  slider_values[label] = *value;
  if (widgets_disabled || click != label) return false;
  *value = .4f;
  return true;
}
bool Begin(const char*, bool*, int) { return true; }
void End() {}
void Text(const char*, ...) { last_item_disabled = false; }
void Separator(const char*) {}
void SameLine(float, float) {}
bool Hovered(int flags) {
  return hovering && (!last_item_disabled || (flags & (1 << 10)));
}
void Tooltip(const char* format, ...) {
  char text[4096];
  va_list args;
  va_start(args, format);
  vsnprintf(text, sizeof(text), format, args);
  va_end(args);
  tooltip = text;
}
}  // namespace
extern "C" void __wrap__ZN3bfa15ScheduleSaveIniEv() { ++saves; }
extern "C" uint64_t __wrap__ZN4skse4AddrEy(uint64_t id) {
  using namespace bfa;
  if (id == kIdCrosshairPick.Get()) return reinterpret_cast<uint64_t>(&pick);
  if (id == kIdUI.Get()) return reinterpret_cast<uint64_t>(&ui);
  if (id == kIdControlMap.Get()) return reinterpret_cast<uint64_t>(&controls);
  if (id == kIdLookupReference.Get())
    return reinterpret_cast<uint64_t>(&Lookup);
  return 0;
}
int main() {
  using namespace bfa;
  skse::SetRuntime(skse::Version(1, 6, 1170));
  const skse::SKSETaskInterface task_api{2, Queue, nullptr};
  MenuWindow popup, main;
  menu_tasks = &task_api;
  npc_window = &popup;
  main_window = &main;
  blocking_window = Blocking;
  npc_input = true;
  ig.Checkbox = Checkbox;
  ig.SliderFloat = Slider;
  ig.Begin = Begin;
  ig.End = End;
  ig.Text = Text;
  ig.Button = Button;
  ig.SeparatorText = Separator;
  ig.SameLine = SameLine;
  ig.IsItemHovered = Hovered;
  ig.SetTooltip = Tooltip;
  for (const auto& p : kParams) {
    if (strcmp(p.key, "SentimentAnalysis") &&
        strcmp(p.key, "SentimentMultiplier")) continue;
    bool flag = false;
    float value = .8f;
    click = "optional-control";
    config.sentiment_available = false;
    CHECK(!ParamControl(p, click.c_str(), p.flag ? &flag : nullptr, &value));
    CHECK(!flag && value == .8f);
    ig.BeginDisabled = BeginDisabled;
    ig.EndDisabled = EndDisabled;
    CHECK(!ParamControl(p, click.c_str(), p.flag ? &flag : nullptr, &value));
    CHECK(!flag && value == .8f && !widgets_disabled);
    // A disabled checkbox/slider must still explain the missing download.
    hovering = true;
    tooltip.clear();
    Tip(&p, ParamTip(p), kHoverDisabled);
    for (int frame = 0; frame < 10; ++frame) {
      menu_time += .05;
      Tip(&p, ParamTip(p), kHoverDisabled);
    }
    CHECK(tooltip.find("not installed") != std::string::npos);
    CHECK(tooltip.find("emotion_fp16.onnx") != std::string::npos);
    CHECK(tooltip.find("Restart Skyrim") != std::string::npos);
    CHECK(!strcmp(ParamTip(p), ParamTip(p, true)));
    // Older frameworks draw an inert text label; it keeps the same help.
    ig.BeginDisabled = nullptr;
    ig.EndDisabled = nullptr;
    CHECK(!ParamControl(p, click.c_str(), p.flag ? &flag : nullptr, &value));
    CHECK(Hovered(kHoverDisabled));
    hovering = false;
    ig.BeginDisabled = BeginDisabled;
    ig.EndDisabled = EndDisabled;
    config.sentiment_available = true;
    CHECK(ParamTip(p) == p.tip);
    if (!strcmp(p.key, "SentimentMultiplier")) {
      CHECK(!ParamControl(p, click.c_str(), nullptr, &value, false));
      CHECK(value == .8f && !widgets_disabled);
    }
    CHECK(ParamControl(p, click.c_str(), p.flag ? &flag : nullptr, &value));
    CHECK(p.flag ? flag : value == .4f);
    ig.BeginDisabled = nullptr;
    ig.EndDisabled = nullptr;
  }
  click.clear();
  for (const auto& p : kParams) {
    if (strcmp(p.key, "VoiceEmotion") && strcmp(p.key, "VoiceSwell")) continue;
    float value = .8f;
    click = "scene-input";
    ig.BeginDisabled = BeginDisabled;
    ig.EndDisabled = EndDisabled;
    CHECK(!ParamControl(p, click.c_str(), nullptr, &value, true, false));
    CHECK(value == .8f);
    CHECK(ParamControl(p, click.c_str(), nullptr, &value, true, true));
    CHECK(value == .4f);
  }
  ig.BeginDisabled = BeginDisabled;
  ig.EndDisabled = EndDisabled;
  const char* profiles[] = {"Male NPCs", "Female NPCs", "Player"};
  for (int selected = 0; selected < 3; ++selected) {
    click = profiles[selected];
    drawn_sliders.clear();
    RenderMenu();
    for (size_t i = 0; i < kNumParams; ++i) {
      const auto& p = kParams[i];
      if (!p.value || p.profile < 0) continue;
      const std::string label = std::string(p.label) + "##" + std::to_string(i);
      const bool drawn = std::find(drawn_sliders.begin(), drawn_sliders.end(), label) != drawn_sliders.end();
      CHECK(drawn == (p.profile == selected));
    }
  }
  click.clear();
  // Independent ABI fixtures: 1.6.1130 added a ControlMap context pointer,
  // shifting linkedMappings and all following fields by eight bytes.
  for (const auto runtime : {skse::Version(1, 5, 97),
                             skse::Version(1, 6, 640),
                             skse::Version(1, 6, 659),
                             skse::Version(1, 6, 1130),
                             skse::Version(1, 6, 1170)}) {
    skse::SetRuntime(runtime);
    control_data.fill(0);
    const bool modern = runtime >= skse::Version(1, 6, 1130);
    const size_t stack = modern ? 0x108 : 0x100;
    const size_t text_entry = modern ? 0x128 : 0x120;
    uint32_t contexts[] = {3, 0};
    Field<uint32_t*>(controls, stack) = contexts;
    Field<uint32_t>(controls, stack + 8) = 10;  // capacity, not size
    Field<uint32_t>(controls, stack + 16) = 2;
    CHECK(NpcHotkeyAllowed());
    if (modern) {
      // Reported crash: linkedMappings.size == 0x43 was read as a pointer,
      // context capacity 10 as size, then contexts[9] read address 0x67.
      Field<uint64_t>(controls, 0x100) = 0x43;
      Field<uint32_t>(controls, 0x120) = 0xFFFFFFFF;  // enabledControls
      CHECK(NpcHotkeyAllowed());
    }
    contexts[1] = 2;  // console on top, even without pausing
    CHECK(!NpcHotkeyAllowed());
    contexts[1] = 0;
    controls[text_entry] = 1;
    CHECK(!NpcHotkeyAllowed());
    controls[text_entry] = 0;
    Field<uint32_t>(controls, stack + 16) = 65;
    CHECK(!NpcHotkeyAllowed());
    Field<uint32_t>(controls, stack + 16) = 1;
    Field<uint32_t*>(controls, stack) = nullptr;
    CHECK(!NpcHotkeyAllowed());
    Field<uint32_t>(controls, stack + 16) = 0;
    CHECK(NpcHotkeyAllowed());
  }
  skse::SetRuntime(skse::Version(1, 6, 1170));
  control_data.fill(0);
  std::array<uint8_t, 0x30> event{};
  auto key = [&](int code, float value = 1, float held = 0) {
    Field<uint32_t>(event.data(), 0x20) = code;
    Field<float>(event.data(), 0x28) = value;
    Field<float>(event.data(), 0x2C) = held;
    return NpcInput(event.data());
  };
  SetNpcHotkey(66);
  CHECK(!key(65));
  CHECK(!key(66, 1, .1f));
  CHECK(!key(66, 0, .1f));
  Field<uint32_t>(event.data(), 8) = 1;
  CHECK(!key(66));  // mouse
  Field<uint32_t>(event.data(), 8) = 0;
  Field<uint32_t>(event.data(), 12) = 2;
  CHECK(!key(66));  // character input
  Field<uint32_t>(event.data(), 12) = 0;
  for (int off : {0x160, 0x164, 0x178, 0x17C}) {
    ui[off] = 1;
    CHECK(!key(66));
    ui[off] = 0;
  }
  controls[0x128] = 1;
  CHECK(!key(66));
  controls[0x128] = 0;
  uint32_t context = 3;
  Field<uint32_t*>(controls, 0x108) = &context;
  Field<uint32_t>(controls, 0x118) = 1;
  CHECK(!key(66));
  context = 0;
  blocked = true;
  CHECK(!key(66));
  blocked = false;
  main.open = true;
  bind_npc_key = true;
  CHECK(key(67));
  CHECK(NpcHotkey() == 67 && !bind_npc_key && saves == 1);
  CHECK(key(67, 1, .3f));
  CHECK(key(67, 0, .3f));  // consume repeats and release
  bind_npc_key = true;
  CHECK(key(1));
  CHECK(NpcHotkey() == 67);
  CHECK(key(1, 0, .1f));
  bind_npc_key = true;
  main.open = false;
  CHECK(!key(67));
  CHECK(!bind_npc_key);  // closing settings cancels capture
  bind_npc_key = true;
  overrides_drawn = true;
  NpcMenuEvent(4);
  CHECK(bind_npc_key);
  NpcMenuEvent(4);
  CHECK(!bind_npc_key);  // navigating away cancels capture

  actor[kFormType] = kFtActor;
  Field<uint32_t>(actor, kFormId) = 0x123;
  Field<LONG>(actor, kRefrRefCount) = 1;
  Field<uint8_t*>(actor, kRefrBase) = base_data.data();
  base_data[kFormType] = kFtNpc;
  Field<uint32_t>(base_data.data(), kFormId) = 0x05001234;
  Field<const char*>(base_data.data(), kNpcName) = "Val Serano";
  uint8_t* files[] = {file_data.data()};
  struct {
    uint8_t** items;
    uint32_t count;
  } source{files, 1};
  Field<void*>(base_data.data(), kFormFiles) = &source;
  std::strcpy(reinterpret_cast<char*>(file_data.data() + kFileName),
              "Follower.esp");
  Field<uint32_t>(pick, 4) = 42;
  CHECK(key(67));
  CHECK(tasks.size() == 1 && !popup.open);
  RequestNpcTarget();
  CHECK(tasks.size() == 1);  // bounded request
  RunTasks();
  CHECK(popup.open && selected_npc.key == "follower.esp|001234");
  CHECK(Field<LONG>(actor, kRefrRefCount) == 1);
  Field<uint32_t>(pick, 4) =
      0;  // target changes after selection: confirmation stays pinned
  click = "Enable facial animations for this NPC";
  RenderNpcWindow();
  click.clear();
  CHECK(!NpcAllowed(actor) && popup.open && saves == 2);
  popup.open = true;
  click = "Enable facial animations for this NPC";
  RenderNpcWindow();
  click.clear();
  CHECK(NpcAllowed(actor) && popup.open && saves == 3);
  click = "Mouth##npc_DirectMouth";
  RenderNpcWindow();
  click.clear();
  CHECK(DescribeNpc(actor).values.at("DirectMouth") == .4f && saves == 4);
  click = "Close";
  RenderNpcWindow();
  click.clear();
  CHECK(!popup.open);
  click = "Edit##follower.esp|001234";
  RenderOverrides();
  click.clear();
  CHECK(popup.open && selected_npc.values.at("DirectMouth") == .4f);
  click = "Use global##DirectMouth";
  RenderNpcWindow();
  click.clear();
  CHECK(DescribeNpc(actor).values.empty() && saves == 5);
  const char* sentiment_label =
      "Sentiment addition multiplier##npc_SentimentMultiplier";
  const Param* sentiment_toggle = nullptr;
  for (const auto& p : kParams)
    if (!strcmp(p.key, "SentimentAnalysis")) sentiment_toggle = &p;
  CHECK(sentiment_toggle);
  *sentiment_toggle->flag = false;
  click = sentiment_label;
  RenderNpcWindow();
  CHECK(DescribeNpc(actor).values.empty());
  *sentiment_toggle->flag = true;
  config.sentiment_available = false;
  RenderNpcWindow();
  CHECK(DescribeNpc(actor).values.empty());
  config.sentiment_available = true;
  RenderNpcWindow();
  CHECK(slider_values.at(sentiment_label) == SnapshotTune().sentiment_multiplier);
  CHECK(DescribeNpc(actor).values.at("SentimentMultiplier") == .4f);
  click = "Close";
  RenderNpcWindow();
  click = "Edit##follower.esp|001234";
  RenderOverrides();
  click.clear();
  RenderNpcWindow();
  CHECK(slider_values.at(sentiment_label) == .4f);
  click = "Sentiment Analysis##npc_SentimentAnalysis";
  RenderNpcWindow();
  CHECK(DescribeNpc(actor).values.at("SentimentAnalysis") == 0);
  CHECK(DescribeNpc(actor).values.at("SentimentMultiplier") == .4f);
  const int saved_before_disabled = saves;
  click = sentiment_label;
  RenderNpcWindow();
  CHECK(saves == saved_before_disabled);
  click = "Sentiment Analysis##npc_SentimentAnalysis";
  RenderNpcWindow();
  CHECK(DescribeNpc(actor).values.at("SentimentAnalysis") == 1);
  *sentiment_toggle->flag = false;
  RenderNpcWindow();
  CHECK(DescribeNpc(actor).values.at("SentimentAnalysis") == 1);
  *sentiment_toggle->flag = true;
  click = "Use global##SentimentAnalysis";
  RenderNpcWindow();
  CHECK(!DescribeNpc(actor).values.count("SentimentAnalysis"));
  // Global changes apply after clearing the NPC's saved value.
  for (const auto& p : kParams)
    if (!strcmp(p.key, "SentimentMultiplier")) *p.value = 1.5f;
  click = "Use global##SentimentMultiplier";
  RenderNpcWindow();
  CHECK(DescribeNpc(actor).values.empty());
  click.clear();
  RenderNpcWindow();
  CHECK(slider_values.at(sentiment_label) == 1.5f);
  click = sentiment_label;
  RenderNpcWindow();
  CHECK(DescribeNpc(actor).values.at("SentimentMultiplier") == .4f);
  click = "Smile reaches eyes##npc_SmileEyes";
  RenderNpcWindow();
  click.clear();
  CHECK(DescribeNpc(actor).values.at("SmileEyes") == .4f);
  click = "Reset NPC to global settings";
  RenderNpcWindow();
  click.clear();
  CHECK(DescribeNpc(actor).values.empty() && SnapshotOverrides().empty());
  click = "Close";
  RenderNpcWindow();
  click.clear();
  CHECK(key(67, 0, .2f));
  CHECK(key(67));
  RunTasks();
  CHECK(popup.open && selected_npc.key.empty());
  click = "Enable facial animations for this NPC";
  RenderNpcWindow();
  click.clear();
  CHECK(SnapshotOverrides()
            .empty());  // invalid target cannot add stale previous NPC
  CHECK(key(1));
  CHECK(!popup.open);
  CHECK(key(1, 0, .2f));
  CHECK(key(67, 0, .2f));
  // A delayed selection cannot reopen a window after loading a different save.
  CHECK(key(67));
  CHECK(tasks.size() == 1);
  MenuMessage(skse::kMessagePreLoadGame);
  RunTasks();
  CHECK(!popup.open && selected_npc.key.empty() && !target_pending);
  // A disappeared actor also produces a harmless empty selection.
  Field<uint32_t>(pick, 4) = 99;
  RequestNpcTarget();
  RunTasks();
  CHECK(popup.open && selected_npc.key.empty());
  MenuMessage(skse::kMessageNewGame);
  skse::SetRuntime(skse::kVrRuntime);
  CHECK(!CanTargetNpc());
  CHECK(!key(67));
  // Framework 3 without GetMainWindow still binds and cancels on close.
  skse::SetRuntime(skse::Version(1, 6, 1170));
  popup.open = false;
  main_window = nullptr;
  blocked = true;
  bind_npc_key = true;
  CHECK(key(68));
  CHECK(NpcHotkey() == 68 && !bind_npc_key);
  CHECK(key(68, 0, .1f));
  blocked = false;
  bind_npc_key = true;
  CHECK(!key(69));
  CHECK(!bind_npc_key);
  std::puts(
      "NPC menu: binding, input guards, immutable target, confirm/remove, "
      "stale tasks passed");
}
