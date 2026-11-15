#include "runtime_fixture.h"
// Exercise production UI/input code with a fake framework and game dispatcher.
#include <array>
#include <cmath>
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
int wakes = 0, discoveries = 0;
a2f::GpuAdapterList adapter_list;
void WakeWorker() { ++wakes; }
a2f::GpuAdapterList SnapshotGpuAdapters() { return adapter_list; }
void RequestGpuAdapters() {
  ++discoveries;
  adapter_list.requested = true;
}
int retry_requests = 0;
void RequestGpuRetry() { ++retry_requests; }
}  // namespace bfa
namespace a2f {
GpuState QueryGpu() {
  CHECK(false);
  return {};
}
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
std::vector<SKSE::Impl::TaskDelegate*> tasks;
bool Blocking() { return blocked; }
void Queue(void* opaque) {
  auto* task = static_cast<SKSE::Impl::TaskDelegate*>(opaque);
  tasks.push_back(task);
}
bool Lookup(const uint32_t& handle, uint8_t** out) {
  *out = handle == 42 ? actor : nullptr;
  if (*out) ++bfa::Field<LONG>(*out, 0x28);
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
std::vector<std::string> control_order;
std::map<std::string, float> slider_values;
bool Button(const char* label, bfa::ImVec2) { return click == label; }
std::vector<std::string> checkboxes;
bool Checkbox(const char* label, bool* value) {
  control_order.emplace_back(label);
  checkboxes.emplace_back(label);
  last_item_disabled = widgets_disabled;
  if (widgets_disabled || click != label) return false;
  *value = !*value;
  return true;
}
bool Slider(const char* label, float* value, float, float, const char*, int) {
  control_order.emplace_back(label);
  last_item_disabled = widgets_disabled;
  drawn_sliders.emplace_back(label);
  slider_values[label] = *value;
  if (widgets_disabled || click != label) return false;
  *value = .4f;
  return true;
}
std::string device_preview, hover_target, last_label;
std::string face_preview;
std::string current_combo;
std::map<std::string, std::vector<std::string>> device_options;
bool combo_open = false;
bool BeginCombo(const char* label, const char* preview, int) {
  current_combo = label;
  control_order.emplace_back(label);
  last_label = label;
  last_item_disabled = false;
  if (!strcmp(label, "##Processing device")) device_preview = preview;
  if (!strcmp(label, "##Audio2Face")) face_preview = preview;
  return combo_open;
}
bool Selectable(const char* label, bool, int, bfa::ImVec2) {
  if (current_combo == "##Processing device" ||
      current_combo == "##EmotionDevice") {
    std::string visible = label;
    visible = visible.substr(0, visible.find("##"));
    device_options[current_combo].push_back(visible);
  }
  last_label = label;
  last_item_disabled = widgets_disabled;
  return !widgets_disabled && click == label;
}
bool Begin(const char*, bool*, int) { return true; }
void End() {}
void Text(const char*, ...) { last_item_disabled = false; }
void Separator(const char*) {}
void SameLine(float, float) {}
float font_scale = 1.3f, content_width = 900;
int groups = 0, group_depth = 0, columns = 0;
std::vector<float> item_widths, heading_sizes;
bool rounded_font = false;
float GetFontSize() {
  return rounded_font ? std::round(20.6f * font_scale) : 20 * font_scale;
}
void SetFontScale(float scale) { font_scale = scale; }
void ContentRegion(bfa::ImVec2* out) { *out = {content_width, 600}; }
void BeginGroup() {
  ++groups;
  ++group_depth;
}
void EndGroup() {
  CHECK(group_depth > 0);
  --group_depth;
}
void SameLineLayout(float, float) { ++columns; }
void ItemWidth(float width) { item_widths.push_back(width); }
void ColoredText(bfa::ImVec4 color, const char*, ...) {
  if (color.x == bfa::kHeadingPink.x && color.y == bfa::kHeadingPink.y &&
      color.z == bfa::kHeadingPink.z)
    heading_sizes.push_back(GetFontSize());
}
bool Hovered(int flags) {
  return hovering && (hover_target.empty() || hover_target == last_label) &&
         (!last_item_disabled || (flags & (1 << 10)));
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
extern "C" void TestSaveIni() { ++saves; }
extern "C" uint64_t TestAddress(uint64_t id) {
  using namespace bfa;
  if (id == kIdCrosshairPick.Get()) return reinterpret_cast<uint64_t>(&pick);
  if (id == kIdUI.Get()) return reinterpret_cast<uint64_t>(&ui);
  if (id == kIdControlMap.Get()) return reinterpret_cast<uint64_t>(&controls);
  if (id == kIdLookupReference.Get())
    return reinterpret_cast<uint64_t>(&Lookup);
  return 0;
}
int main() {
  bfa::testing::address = TestAddress;
  bfa::testing::save_ini = TestSaveIni;
  using namespace bfa;
  skse::SetRuntime(skse::Version(1, 6, 1170));
  const SKSE::Impl::SKSETaskInterface task_api{2, Queue, nullptr};
  MenuWindow popup, main;
  menu_tasks = reinterpret_cast<const skse::SKSETaskInterface*>(&task_api);
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
  ig.BeginCombo = BeginCombo;
  ig.EndCombo = End;
  ig.Selectable = Selectable;
  ready = false;
  combo_open = false;
  RenderProcessingDevice(SnapshotTune());
  CHECK(discoveries == 0);  // no graphics queries during startup
  ready = true;
  RenderMenu();
  CHECK(discoveries == 1);
  RenderMenu();
  CHECK(discoveries == 1);  // no repeated discovery while the worker is busy
  adapter_list = {{{0, "Intel(R) UHD Graphics 770", 0},
                   {2, "NVIDIA GeForce RTX 4070", 12288}},
                  "",
                  true,
                  true};
  RenderMenu();
  CHECK(device_preview == "GPU 0: Intel(R) UHD Graphics 770");
  combo_open = true;
  click = "GPU 2: NVIDIA GeForce RTX 4070";
  const int previous_saves = saves, previous_wakes = wakes;
  RenderMenu();
  CHECK(!SnapshotTune().force_cpu && SnapshotTune().gpu_adapter == 2);
  CHECK(saves == previous_saves + 1 && wakes == previous_wakes + 1);
  click.clear();
  combo_open = false;
  RenderMenu();
  CHECK(device_preview == "GPU 2: NVIDIA GeForce RTX 4070");
  combo_open = true;
  click = "CPU (Force CPU)";
  RenderMenu();
  CHECK(SnapshotTune().force_cpu && SnapshotTune().gpu_adapter == 2);
  click.clear();
  RenderMenu();
  CHECK(device_preview == "CPU (Force CPU)");
  for (const auto& label : checkboxes)
    CHECK(label.find("Force CPU") == std::string::npos);
  adapter_list.adapters.clear();
  adapter_list.error = "discovery failed";
  SetProcessingDevice(false, 2);
  RenderMenu();
  CHECK(device_preview == "GPU 2 (unavailable)" &&
        SnapshotTune().gpu_adapter == 2);
  click = "CPU (Force CPU)";
  RenderMenu();
  CHECK(
      SnapshotTune().force_cpu);  // CPU recovery works even without a GPU list
  SetProcessingDevice(false, 0);
  combo_open = false;
  saves = wakes = 0;
  click.clear();
  // Independent model dropdowns, missing files, cache wake and exact labels.
  combo_open = true;
  CHECK(SnapshotTune().sentiment_analysis &&
        SnapshotTune().performance_sentiment);
  SelectEmotionOption(kEmotionDisabled);
  config.performance_sentiment_available = false;
  click = "Performance (not installed)##Audio2Emotion";
  RenderMenu();
  CHECK(!SnapshotTune().sentiment_analysis);
  config.performance_sentiment_available = true;
  click = "Performance##Audio2Emotion";
  const int model_saves = saves, model_wakes = wakes;
  RenderMenu();
  CHECK(SnapshotTune().sentiment_analysis &&
        SnapshotTune().performance_sentiment);
  CHECK(saves == model_saves + 1 && wakes == model_wakes + 1);
  click = "Disabled##Audio2Face";
  RenderMenu();
  CHECK(!SnapshotTune().enabled && SnapshotTune().sentiment_analysis);
  click = "V3##Audio2Face";
  RenderMenu();
  CHECK(SnapshotTune().enabled && SnapshotTune().sentiment_analysis);
  // Every Audio2Face model is offered and is named when selected, however
  // it was chosen (6.1.1 listed four of five, and showed Legacy as Disabled).
  for (const ModelOption& option : kFaceOptions) {
    click = std::string(option.name) + "##Audio2Face";
    RenderMenu();
    CHECK(SnapshotTune().FaceChoice() == option.id);
    click.clear();
    RenderMenu();
    CHECK(face_preview == option.name);
  }
  // The dropdown and the ini switches both follow the one model table.
  CHECK(kFaceOptions.size() == FaceModels().size() + 1);
  // The default model (Performance) heads them, then the fallback (V3).
  CHECK(kFaceOptions[1].id == a2f::kNetPerformance &&
        kFaceOptions[1].id == Tune().SelectedModel() &&
        kFaceOptions[2].id == FallbackFaceModel().network);
  for (const ModelInfo& info : FaceModels()) {
    size_t switches = 0;
    for (const Param& p : kParams)
      switches += info.ini_key && !strcmp(p.key, info.ini_key) && p.flag &&
                  p.stored && p.own_control;
    CHECK(switches == (info.selected ? 1u : 0u));
  }
  SelectFaceOption(kFaceDisabled);
  SetSelectedModel(a2f::kNetV2);
  SelectFaceOption(a2f::kNetV2);
  RenderMenu();
  CHECK(face_preview == "Legacy");
  click = "V3##Audio2Face";
  RenderMenu();
  CHECK(SnapshotTune().SelectedModel() == a2f::kNetV3);
  config.sentiment_available = true;
  click = "Full##Audio2Emotion";
  RenderMenu();
  CHECK(SnapshotTune().sentiment_analysis &&
        !SnapshotTune().performance_sentiment);
  click = "Disabled##Audio2Emotion";
  RenderMenu();
  CHECK(!SnapshotTune().sentiment_analysis && SnapshotTune().enabled);
  CHECK(!strcmp(kEmotionOptions[0].name, "Disabled"));
  CHECK(!strcmp(kEmotionOptions[1].name, "Full"));
  CHECK(!strcmp(kEmotionOptions[2].name, "Performance"));
  CHECK(strstr(kEmotionOptions[1].tip, "AMD and NVIDIA"));
  CHECK(strstr(kEmotionOptions[2].tip, "memorize"));
  CHECK(strstr(kEmotionOptions[2].tip, "obscure quest mods"));
  click.clear();
  hovering = true;
  combo_open = false;
  hover_target = "Audio2Emotion";
  tooltip.clear();
  for (int frame = 0; frame < 12; ++frame) {
    menu_time += .05;
    RenderModelChoice("Audio2Emotion", kEmotionOptions, kEmotionPerformance,
                      true);
  }
  std::replace(tooltip.begin(), tooltip.end(), '\n', ' ');
  CHECK(tooltip.find("memorize") != std::string::npos);
  CHECK(tooltip.find("obscure quest mods") != std::string::npos);
  combo_open = true;
  hover_target = "Full##Audio2Emotion";
  tooltip.clear();
  for (int frame = 0; frame < 12; ++frame) {
    menu_time += .05;
    RenderModelChoice("Audio2Emotion", kEmotionOptions, kEmotionPerformance,
                      true);
  }
  std::replace(tooltip.begin(), tooltip.end(), '\n', ' ');
  CHECK(tooltip.find("AMD and NVIDIA") != std::string::npos);
  CHECK(tooltip.find("745 MB") != std::string::npos);
  hovering = false;
  hover_target.clear();
  combo_open = false;
  config.performance_sentiment_available = false;
  saves = wakes = 0;
  click.clear();
  // GPU errors offer an explicit retry; the action never changes Force CPU.
  ready = true;
  net_loaded = false;
  Tune retry_tune = SnapshotTune();
  retry_tune.enabled = true;
  retry_tune.force_cpu = false;
  Stats retry_stats;
  retry_stats.net_error = {"injected GPU error"};
  click = "Try GPU again";
  RenderProblems(retry_stats, 0, retry_tune, {});
  CHECK(retry_requests == 1 && !retry_tune.force_cpu);
  retry_tune.force_cpu = true;
  RenderProblems(retry_stats, 0, retry_tune, {});
  CHECK(retry_requests == 1);
  // The smaller package can retry Performance emotion without Full installed.
  retry_tune.force_cpu = false;
  retry_tune.sentiment_analysis = true;
  retry_tune.performance_sentiment = true;
  retry_stats.net_error.clear();
  retry_stats.sentiment_info = "GPU sentiment unavailable: injected error";
  net_loaded = true;
  config.sentiment_available = false;
  config.performance_sentiment_available = true;
  RenderProblems(retry_stats, 0, retry_tune, {});
  CHECK(retry_requests == 2);
  retry_tune.performance_sentiment = false;
  RenderProblems(retry_stats, 0, retry_tune, {});
  CHECK(retry_requests == 2);  // Missing Full must not enable the action.
  config.sentiment_available = true;
  RenderProblems(retry_stats, 0, retry_tune, {});
  CHECK(retry_requests == 3);
  retry_tune.sentiment_analysis = false;
  RenderProblems(retry_stats, 0, retry_tune, {});
  CHECK(retry_requests == 3);
  config.performance_sentiment_available = false;
  ready = false;
  click.clear();
  for (const auto& p : kParams) {
    if (strcmp(p.key, "SentimentAnalysis") &&
        strcmp(p.key, "SentimentMultiplier"))
      continue;
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
  // Emotion controls stay adjacent, render once, and preserve saved behavior.
  std::string strength, vanilla;
  for (size_t i = 0; i < kNumParams; ++i) {
    if (!strcmp(kParams[i].key, "SentimentMultiplier"))
      strength = "##" + std::to_string(i);
    if (!strcmp(kParams[i].key, "VanillaEmotions"))
      vanilla = "Pass through vanilla emotion cues##" + std::to_string(i);
  }
  ig.GetFontSize = GetFontSize;
  ig.SetWindowFontScale = SetFontScale;
  ig.GetContentRegionAvail = ContentRegion;
  ig.BeginGroup = BeginGroup;
  ig.EndGroup = EndGroup;
  ig.SetNextItemWidth = ItemWidth;
  ig.SameLine = SameLineLayout;
  ig.TextColored = ColoredText;
  config.performance_sentiment_available = true;
  SelectEmotionOption(2);
  control_order.clear();
  click = strength;
  int before_saves = saves, before_wakes = wakes;
  RenderMenu();
  CHECK(saves == before_saves + 1 && wakes == before_wakes + 1);
  CHECK(SnapshotTune().sentiment_multiplier == .4f);
  CHECK(groups == 2 && group_depth == 0 && columns > 0);
  CHECK(font_scale == 1.3f);
  rounded_font = true;
  const float original_font_size = GetFontSize();
  for (int frame = 0; frame < 100; ++frame) {
    {
      ScopedFontScale heading(1.6f);
    }
    CHECK(GetFontSize() == original_font_size);
  }
  rounded_font = false;
  font_scale = 1.3f;
  CHECK(heading_sizes.front() > 40 && heading_sizes[1] > 32);
  auto emotion =
      std::find(control_order.begin(), control_order.end(), "##Audio2Emotion");
  CHECK(emotion != control_order.end() && emotion + 2 < control_order.end());
  CHECK(emotion[1] == strength && emotion[2] == vanilla);
  CHECK(std::count(control_order.begin(), control_order.end(), strength) == 1);
  CHECK(std::count(control_order.begin(), control_order.end(), vanilla) == 1);
  CHECK(std::find(item_widths.begin(), item_widths.end(), 437) !=
        item_widths.end());
  groups = columns = 0;
  content_width = 360;
  item_widths.clear();
  const bool vanilla_before = SnapshotTune().vanilla_emotions;
  click = vanilla;
  RenderMenu();
  CHECK(SnapshotTune().vanilla_emotions != vanilla_before);
  CHECK(groups == 0 && group_depth == 0 && font_scale == 1.3f);
  CHECK(std::find(item_widths.begin(), item_widths.end(), 360) !=
        item_widths.end());
  combo_open = true;
  click = "Disabled##Audio2Emotion";
  RenderMenu();
  CHECK(!SnapshotTune().sentiment_analysis);
  click = strength;
  before_saves = saves;
  RenderMenu();
  CHECK(saves == before_saves && SnapshotTune().sentiment_multiplier == .4f);
  SelectEmotionOption(2);
  combo_open = false;
  ig.GetFontSize = nullptr;
  ig.SetWindowFontScale = nullptr;
  ig.GetContentRegionAvail = nullptr;
  ig.BeginGroup = nullptr;
  ig.EndGroup = nullptr;
  ig.SetNextItemWidth = nullptr;
  ig.SameLine = SameLine;
  ig.TextColored = nullptr;
  ready = true;
  combo_open = true;
  adapter_list = {
      {{0, "Integrated", 512}, {2, "Discrete", 8192}}, "", true, true};
  const Tune face_device = SnapshotTune();
  device_options.clear();
  click.clear();
  RenderMenu();
  CHECK(device_options["##Processing device"] ==
        device_options["##EmotionDevice"]);
  CHECK(device_options["##EmotionDevice"].size() ==
        3);  // CPU + two detected GPUs.
  click = "CPU (Force CPU)##EmotionDevice";
  RenderMenu();
  CHECK(SnapshotTune().emotion_device == kCpuDevice);
  CHECK(SnapshotTune().force_cpu == face_device.force_cpu &&
        SnapshotTune().gpu_adapter == face_device.gpu_adapter);
  click = "GPU 2: Discrete##EmotionDevice";
  before_saves = saves;
  RenderMenu();
  CHECK(SnapshotTune().emotion_device == 2 && saves == before_saves + 1);
  // Graph fusion: on GPU a checkbox toggles it; on CPU it is inert, a
  // disabled checkbox or, without BeginDisabled, a label. Reset keeps it.
  for (const char* key : {"GraphFusion", "EmotionGraphFusion"}) {
    const Param* fusion = nullptr;
    for (const Param& p : kParams)
      if (!strcmp(p.key, key)) fusion = &p;
    CHECK(fusion && !*fusion->flag);
    click = std::string(fusion->label) + "##" + key;
    CHECK(RenderGraphFusion(key, false) && *fusion->flag);
    CHECK(!RenderGraphFusion(key, true) && *fusion->flag && last_item_disabled);
    checkboxes.clear();
    ig.BeginDisabled = nullptr;
    CHECK(!RenderGraphFusion(key, true) && checkboxes.empty());
    ig.BeginDisabled = BeginDisabled;
  }
  ResetTune();
  CHECK(SnapshotTune().emotion_device == 2);
  CHECK(SnapshotTune().face_graph_fusion &&
        SnapshotTune().emotion_graph_fusion);
  for (const Param& p : kParams)
    if (strstr(p.key, "GraphFusion")) *p.flag = false;
  // Emotion GPU recovery is available even when the face model uses CPU.
  retry_tune.force_cpu = true;
  retry_tune.sentiment_analysis = true;
  retry_tune.emotion_device = 2;
  click = "Try GPU again";
  const int retry_before = retry_requests;
  RenderProblems(retry_stats, 0, retry_tune, {});
  CHECK(retry_requests == retry_before + 1);
  retry_tune.emotion_device = kCpuDevice;
  RenderProblems(retry_stats, 0, retry_tune, {});
  CHECK(retry_requests == retry_before + 1);
  click = "GPU 0: Integrated##EmotionDevice";
  RenderMenu();
  CHECK(SnapshotTune().emotion_device == 0);
  combo_open = false;
  ready = false;
  saves = wakes = 0;
  const char* profiles[] = {"Male NPCs", "Female NPCs", "Player"};
  for (int selected = 0; selected < 3; ++selected) {
    click = profiles[selected];
    drawn_sliders.clear();
    RenderMenu();
    for (size_t i = 0; i < kNumParams; ++i) {
      const auto& p = kParams[i];
      if (!p.value || p.profile < 0) continue;
      const std::string label = std::string(p.label) + "##" + std::to_string(i);
      const bool drawn = std::find(drawn_sliders.begin(), drawn_sliders.end(),
                                   label) != drawn_sliders.end();
      CHECK(drawn == (p.menu && p.profile == selected));
    }
  }
  click.clear();
  // Independent ABI fixtures: 1.6.1130 added a ControlMap context pointer,
  // shifting linkedMappings and all following fields by eight bytes.
  for (const auto runtime :
       {skse::Version(1, 5, 97), skse::Version(1, 6, 640),
        skse::Version(1, 6, 659), skse::Version(1, 6, 1130),
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

  actor[0x1A] = kFtActor;
  Field<uint32_t>(actor, 0x14) = 0x123;
  Field<LONG>(actor, 0x28) = 1;
  Field<uint8_t*>(actor, 0x40) = base_data.data();
  base_data[0x1A] = kFtNpc;
  Field<uint32_t>(base_data.data(), 0x14) = 0x05001234;
  Field<const char*>(base_data.data(), 0xE0) = FixtureString("Val Serano");
  uint8_t* files[] = {file_data.data()};
  struct {
    uint8_t** items;
    uint32_t count;
  } source{files, 1};
  Field<void*>(base_data.data(), 0x08) = &source;
  std::strcpy(reinterpret_cast<char*>(file_data.data() + 0x58), "Follower.esp");
  Field<uint32_t>(pick, 4) = 42;
  CHECK(key(67));
  CHECK(tasks.size() == 1 && !popup.open);
  RequestNpcTarget();
  CHECK(tasks.size() == 1);  // bounded request
  RunTasks();
  CHECK(popup.open && selected_npc.key == "follower.esp|001234");
  CHECK(Field<LONG>(actor, 0x28) == 1);
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
  SelectEmotionOption(
      1);  // These missing-model checks toggle Full availability.
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
  CHECK(slider_values.at(sentiment_label) ==
        SnapshotTune().sentiment_multiplier);
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
  // Rendering the report uses only cached diagnostics and retains native
  // detail.
  a2f::diagnostics::Snapshot snapshot;
  auto& failure = snapshot.components[0];
  failure.attempt = 7;
  failure.gpu = failure.independent = true;
  failure.stage = a2f::diagnostics::Stage::kDevice;
  failure.error = "CreateDevice failed [HRESULT 0x887A0004]";
  failure.last_failure = {7, failure.stage, failure.error, true, true};
  Tune cpu;
  cpu.force_cpu = true;
  const auto report = DiagnosticReport(Stats{}, cpu, snapshot);
  CHECK(report.find("Requested: CPU") != std::string::npos);
  CHECK(report.find("attempt 7, independent GPU") != std::string::npos);
  CHECK(report.find(failure.error) != std::string::npos);
  CHECK(
      report.find(
          "Last failed attempt 7 (independent GPU) at D3D12 device creation") !=
      std::string::npos);
  RenderMenu();  // Any call to QueryGpu is fatal above.
  std::puts(
      "NPC menu: binding, input guards, immutable target, confirm/remove, "
      "stale tasks passed");
}
