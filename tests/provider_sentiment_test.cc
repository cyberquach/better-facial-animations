#include "runtime_fixture.h"
// A provider line (include/a2f/speech/BFA_API.h) through the production
// worker: the selected Audio2Emotion model's reading is fed to the facial
// network with the caller's channels and lands in the animation the line
// plays. Engine lookups, the emotion model and the facial network are
// controlled fixtures; audio decoding and the emotion track are production.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "a2f/inference/ort_net.h"

namespace a2f {
int emotion_runs = 0;
bool performance_seen = false;
std::wstring path_seen;
std::function<void()> on_emotion_run;  // called as a reading begins
// Audio2Emotion that hears happiness (class 3 -> kJoy) in everything.
class HappyEmotion : public EmotionNet {
 public:
  bool Run(const float*, float* logits, std::string&) override {
    ++emotion_runs;
    if (on_emotion_run) on_emotion_run();
    std::fill_n(logits, kNumVoiceEmo, 0.f);
    logits[3] = 12;
    return true;
  }
};
std::unique_ptr<EmotionNet> CaptureEmotion(const std::wstring&,
                                           const std::wstring& path,
                                           const OrtOptions&, std::string&,
                                           bool performance = false) {
  performance_seen = performance;
  path_seen = path;
  return std::make_unique<HappyEmotion>();
}
}  // namespace a2f
#define CreateOrtEmotionNet CaptureEmotion
#include "../src/plugin/better_facial_animations.cc"
#undef CreateOrtEmotionNet
#include "a2f/speech/BFA_API.h"

extern "C" __declspec(dllexport) const BFA_Interface* BFA_QueryInterface(
    uint32_t wanted_version);

namespace {
#define CHECK(x) Check((x), #x, __LINE__)
void Check(bool ok, const char* expression, int line) {
  if (ok) return;
  fprintf(stderr, "%d: %s\n", line, expression);
  std::exit(1);
}

constexpr uint32_t kActorId = 0x1234;
std::array<uint8_t, 0x1000> actor_data{}, base{}, race{}, face_data{},
    manager{};
uint8_t* actor = actor_data.data();
uint8_t* fg = face_data.data();
uint8_t* mtm = manager.data();
uint8_t* GetFace(uint8_t*) { return fg; }
bool NotInFaction(const uint8_t*, const uint8_t*) { return false; }
void Handle(uint32_t& h, uint8_t*) { h = 42; }
bool Lookup(const uint32_t& h, uint8_t** out) {
  *out = h == 42 ? actor : nullptr;
  if (*out) ++bfa::Field<LONG>(*out, 0x28);
  return *out != nullptr;
}
uint8_t* LookupForm(uint32_t id) { return id == kActorId ? actor : nullptr; }
void Resource(skse::ResId* id, const char* path) {
  *id = {};
  id->file = static_cast<uint32_t>(std::hash<std::string>{}(path));
}

SRWLOCK events_lock = SRWLOCK_INIT;
std::vector<BFA_Event> got;
void OnEvent(const BFA_Event* e, void*) {
  bfa::ScopedLock guard(events_lock);
  got.push_back(*e);
}
bool Wait(size_t n) {
  for (int i = 0; i < 600; ++i) {
    {
      bfa::ScopedLock guard(events_lock);
      if (got.size() >= n) return true;
    }
    Sleep(5);
  }
  return false;
}
BFA_Event At(size_t i) {
  bfa::ScopedLock guard(events_lock);
  CHECK(i < got.size());
  return got[i];
}
size_t Events() {
  bfa::ScopedLock guard(events_lock);
  return got.size();
}

void SetFlag(const char* key, bool on) {
  bfa::ScopedLock guard(bfa::settings_lock);
  for (const bfa::Param& p : bfa::kParams)
    if (!strcmp(p.key, key)) *p.flag = on;
}
void SetValue(const char* key, float value) {
  bfa::ScopedLock guard(bfa::settings_lock);
  for (const bfa::Param& p : bfa::kParams)
    if (!strcmp(p.key, key) && p.value) *p.value = value;
}

// extra: samples beyond the whole seconds.
std::vector<uint8_t> Wave(int seconds = 1, int extra = 0) {
  constexpr int rate = 16000;
  std::vector<int16_t> pcm(rate * seconds + extra);
  for (size_t i = 0; i < pcm.size(); ++i)
    pcm[i] = static_cast<int16_t>(
        8000 * std::sin(i * 6.283185307179586 * 220 / rate));
  std::vector<uint8_t> wav(44 + pcm.size() * 2);
  auto u32 = [&](int at, uint32_t v) { memcpy(wav.data() + at, &v, 4); };
  auto u16 = [&](int at, uint16_t v) { memcpy(wav.data() + at, &v, 2); };
  memcpy(wav.data(), "RIFF", 4);
  u32(4, wav.size() - 8);
  memcpy(wav.data() + 8, "WAVEfmt ", 8);
  u32(16, 16);
  u16(20, 1);
  u16(22, 1);
  u32(24, rate);
  u32(28, rate * 2);
  u16(32, 2);
  u16(34, 16);
  memcpy(wav.data() + 36, "data", 4);
  u32(40, pcm.size() * 2);
  memcpy(wav.data() + 44, pcm.data(), pcm.size() * 2);
  return wav;
}

std::vector<SKSE::Impl::TaskDelegate*> pending_tasks;
void AddTask(void* opaque) {
  pending_tasks.push_back(static_cast<SKSE::Impl::TaskDelegate*>(opaque));
}
SKSE::Impl::SKSETaskInterface task_api{2, AddTask, nullptr};
void* Query(uint32_t id) {
  return id == skse::kInterfaceTask ? &task_api : nullptr;
}
uint32_t PluginHandle() { return 1; }
void RunTasks() {
  auto copy = std::move(pending_tasks);
  pending_tasks.clear();
  for (auto* task : copy) {
    task->Run();
    task->Dispose();
  }
}

// The facial network: records the emotion it is given and answers with it,
// so the animation (channel 0 of every frame) is the joy it was fed.
int face_runs = 0;
a2f::EmotionCaps fed_peak{};
struct PerformanceNet : a2f::Net {
  int Version() const override { return a2f::kNetPerformance; }
  int Batch() const override { return 1; }
  bool Run(const float*, const float*, int, float*, std::string&) override {
    return false;
  }
  bool RunPerformance(const float*, int64_t, const float* emotion,
                      const a2f::PerformanceState& state, float* motion,
                      a2f::PerformanceState& next, std::string&) override {
    next.frames = std::min<int64_t>(a2f::kPerformanceMemoryFrames,
                                    state.frames + a2f::kPerformanceHopFrames);
    ++face_runs;
    std::fill_n(motion,
                a2f::kPerformanceHopFrames * a2f::kPerformanceChannels, 0.f);
    for (int slot = 0; slot < a2f::kPerformanceHopFrames; ++slot) {
      for (int c = 0; c < a2f::kNumMod; ++c)
        fed_peak[c] = std::max(
            fed_peak[c],
            emotion[(slot + a2f::kPerformanceContextFrames) * a2f::kNumMod +
                    c]);
      motion[slot * a2f::kPerformanceChannels] =
          emotion[(slot + a2f::kPerformanceContextFrames) * a2f::kNumMod +
                  a2f::kJoy];
    }
    return true;
  }
};

// One provider line through intake and the worker; its generated animation.
struct Generated {
  uint64_t handle = 0;
  std::shared_ptr<bfa::Entry> entry;
  std::shared_ptr<const a2f::Line> line;
  float multiplier = 0;  // the sentiment multiplier the worker was given
};
Generated Speak(const BFA_Interface* api, BFA_Line& line,
                a2f::NetKeeper& keeper) {
  using namespace bfa;
  Generated g;
  BFA_Result result;
  line.utterance_id = api->NewUtteranceId();
  g.handle = api->Prepare(&line, &result);
  CHECK(g.handle && result == BFA_kResultPending);
  RunTasks();
  CHECK(queue.size() == 1);
  const Job job = queue.front();
  queue.pop_front();
  CHECK(job.audio && job.entry->watchers.size() == 1);
  g.multiplier = job.emotion.sentiment_multiplier;
  fed_peak = {};
  Process(job, keeper);
  g.entry = job.entry;
  g.line = job.entry->line;
  CHECK(g.entry->state == State::kReady && g.line && g.line->frames > 0);
  return g;
}
float Direct(const a2f::Line& line) {
  float peak = 0;
  for (uint32_t f = 0; f < line.frames; ++f)
    peak = std::max(peak, line.direct[f * line.direct_dim]);
  return peak;
}
void ReadyFace() {
  using namespace bfa;
  faces[fg] = Face();
  faces[fg].head_state = State::kReady;
  faces[fg].actor = actor;
  faces[fg].actor_handle = 42;
  Count();
}
}  // namespace

extern "C" uint64_t TestAddress(uint64_t id) {
  using namespace bfa;
  if (id == kIdLookupFormById.Get())
    return reinterpret_cast<uint64_t>(&LookupForm);
  if (id == kIdCreateHandle.Get()) return reinterpret_cast<uint64_t>(&Handle);
  if (id == kIdLookupReference.Get())
    return reinterpret_cast<uint64_t>(&Lookup);
  if (id == kIdGenerateFromPath.Get())
    return reinterpret_cast<uint64_t>(&Resource);
  if (id == kIdMenuTopicManager.Get()) return reinterpret_cast<uint64_t>(&mtm);
  return 0;
}

int main() {
  bfa::testing::address = TestAddress;
  using namespace bfa;
  skse::SetRuntime(skse::Version(1, 6, 1170));
  static void* vtable[0x100]{};
  vtable[0x63] = reinterpret_cast<void*>(&GetFace);
  vtable[0xF9] = reinterpret_cast<void*>(&NotInFaction);
  Field<void**>(actor, 0) = vtable;
  Field<uint8_t*>(actor, 0x40) = base.data();
  actor[0x1A] = kFtActor;
  base[0x1A] = kFtNpc;
  Field<uint8_t*>(actor, 0x1f8) = race.data();
  Field<uint32_t>(race.data(), 0x14) = 0x13746;  // Nord
  Field<uint32_t>(race.data(), 0x108) = 2;
  Field<uint32_t>(actor, 0x14) = kActorId;
  Field<LONG>(actor, 0x28) = 1;
  CHECK(SpeakerOk(actor));

  SKSE::Impl::SKSEInterface skse_api{};
  skse_api.QueryInterface = Query;
  skse_api.GetPluginHandle = PluginHandle;
  InitProvider(reinterpret_cast<const skse::SKSEInterface*>(&skse_api));
  const auto* api = BFA_QueryInterface(BFA_API_VERSION);
  CHECK(api);
  BFA_Result result;
  CHECK(api->RegisterListener(OnEvent, nullptr, &result) &&
        result == BFA_kResultOK);

  // The user's installation: both emotion models present, emotion on CPU,
  // the Performance facial model selected and loaded.
  ready = true;
  net_loaded = true;
  config.sentiment_available = config.performance_sentiment_available = true;
  config.independent_device = false;
  SetEmotionDevice(kCpuDevice);
  auto mapping = std::make_shared<direct::Model>();
  mapping->network = a2f::kNetV3;
  mapping->faces.resize(2);
  auto state = std::make_shared<ModelState>();
  state->network = a2f::kNetPerformance;
  state->gpu = false;
  state->mapping = mapping;
  std::atomic_store(&active_model, std::shared_ptr<const ModelState>(state));
  SetFlag("UsePerformanceModel", true);
  CHECK(SnapshotTune().SelectedModel() == a2f::kNetPerformance);
  a2f::NetKeeper keeper(
      false,
      [](bool, std::string&, std::string&) {
        return std::unique_ptr<a2f::Net>(new PerformanceNet);
      },
      [] { return Now(); });
  ReadyFace();

  auto wav = Wave();
  BFA_Line line{BFA_API_VERSION, sizeof(BFA_Line),    0, 0, kActorId,
                wav.data(),      (uint32_t)wav.size()};

  // Performance Audio2Emotion selected, no caller emotion: the model's joy
  // reaches the facial network and the line's animation.
  SelectEmotionOption(2);
  auto heard = Speak(api, line, keeper);
  CHECK(heard.multiplier == 1 && a2f::emotion_runs > 0 && face_runs > 0);
  CHECK(a2f::performance_seen &&
        a2f::path_seen.find(kPerformanceSentimentModel) != std::wstring::npos);
  CHECK(!heard.entry->degraded);
  CHECK(heard.line->emotion_peak[a2f::kJoy] > .9f &&
        fed_peak[a2f::kJoy] > .9f);
  for (int c = 0; c < a2f::kNumMod; ++c)
    if (c != a2f::kJoy) CHECK(fed_peak[c] < .05f);
  const float happy = Direct(*heard.line);

  // That animation is what the provider's line plays: Start puts it on the
  // face and the provider is told it began.
  CHECK(api->Start(heard.handle, 0) == BFA_kResultPending);
  RunTasks();
  CHECK(playing.size() == 1 && playing.at(fg).entry == heard.entry);
  CHECK(playing.at(fg).entry->line->emotion_peak[a2f::kJoy] > .9f);
  CHECK(Apply(fg, SnapshotTune(), false, actor, 42, a2f::kNoSound, -1) &
        kFaceChanged);
  CHECK(Wait(1) && At(0).event == BFA_kStarted && At(0).handle == heard.handle);
  CHECK(api->EndUtterance(line.utterance_id, BFA_kAborted) != BFA_kResultNotFound);
  ResetPlayback(true);
  ResetPlayback(false);
  ReadyFace();

  // The caller's channels add to the model's reading; both are fed.
  BFA_Emotion emotion{BFA_API_VERSION, sizeof(BFA_Emotion), {}};
  emotion.channels[BFA_kAnger] = 1;
  line.emotion = &emotion;
  auto both = Speak(api, line, keeper);
  CHECK(both.entry != heard.entry);  // its own cache entry
  CHECK(fed_peak[a2f::kJoy] > .9f && fed_peak[a2f::kAnger] > .9f);
  CHECK(both.line->emotion_peak[a2f::kAnger] > .9f);
  CHECK(api->EndUtterance(line.utterance_id, BFA_kAborted) == BFA_kResultOK);
  line.emotion = nullptr;

  // Full Audio2Emotion selected: the Full model is the one loaded and fed.
  SelectEmotionOption(1);
  auto full = Speak(api, line, keeper);
  CHECK(full.multiplier == 1 && !a2f::performance_seen &&
        a2f::path_seen.find(kSentimentModel) != std::wstring::npos);
  CHECK(full.line->emotion_peak[a2f::kJoy] > .9f && fed_peak[a2f::kJoy] > .9f);
  CHECK(api->EndUtterance(line.utterance_id, BFA_kAborted) == BFA_kResultOK);

  // The NPC's multiplier scales the model's reading.
  SelectEmotionOption(2);
  SetValue("SentimentMultiplier", .5f);
  auto half = Speak(api, line, keeper);
  CHECK(half.multiplier == .5f);
  CHECK(fed_peak[a2f::kJoy] > .4f && fed_peak[a2f::kJoy] < .6f);
  CHECK(half.line->emotion_peak[a2f::kJoy] < .6f);
  CHECK(api->EndUtterance(line.utterance_id, BFA_kAborted) == BFA_kResultOK);
  SetValue("SentimentMultiplier", 1);

  // Audio2Emotion disabled: no model runs, nothing is fed, and the animation
  // differs from the heard line's.
  SelectEmotionOption(0);
  const int emotion_runs_before = a2f::emotion_runs;
  auto deaf = Speak(api, line, keeper);
  CHECK(deaf.multiplier == 0 && a2f::emotion_runs == emotion_runs_before);
  CHECK(deaf.entry != heard.entry && !deaf.entry->degraded);
  for (int c = 0; c < a2f::kNumMod; ++c)
    CHECK(fed_peak[c] == 0 && deaf.line->emotion_peak[c] == 0);
  CHECK(Direct(*deaf.line) < happy);
  CHECK(api->EndUtterance(line.utterance_id, BFA_kAborted) == BFA_kResultOK);

  // Selected but not installed: the line goes without, not degraded.
  SelectEmotionOption(2);
  config.performance_sentiment_available = false;
  auto missing = Speak(api, line, keeper);
  CHECK(missing.multiplier == 0 && a2f::emotion_runs == emotion_runs_before);
  CHECK(missing.line->emotion_peak[a2f::kJoy] == 0 && !missing.entry->degraded);
  CHECK(api->EndUtterance(line.utterance_id, BFA_kAborted) == BFA_kResultOK);
  config.performance_sentiment_available = true;

  // A clip that does not end on a frame: the voice's joy is in the moods of
  // its gaze beats as it is in its animation.
  {
    auto odd_wav = Wave(1, 100);
    BFA_Line odd{BFA_API_VERSION, sizeof(BFA_Line),        0, 0, kActorId,
                 odd_wav.data(),  (uint32_t)odd_wav.size()};
    const auto spoken = Speak(api, odd, keeper);
    CHECK(spoken.line->emotion_peak[a2f::kJoy] > .9f);
    CHECK(spoken.entry->beats && !spoken.entry->beats->empty() &&
          spoken.entry->beats->front().moods[a2f::kGazeHappy] > .9f);
    CHECK(api->EndUtterance(odd.utterance_id, BFA_kAborted) == BFA_kResultOK);
  }

  // Audio2Emotion switched off while a line's sentiment is read: the line is
  // made without it, under the key that has it, and marked degraded, to be
  // generated again once sentiment would be tried again.
  {
    auto cut_wav = Wave(7);
    BFA_Line cut{BFA_API_VERSION, sizeof(BFA_Line),        0, 0, kActorId,
                 cut_wav.data(),  (uint32_t)cut_wav.size()};
    const int heard_before = a2f::emotion_runs;
    a2f::on_emotion_run = [&] {
      if (a2f::emotion_runs - heard_before == 3) SelectEmotionOption(0);
    };
    const auto made = Speak(api, cut, keeper);
    a2f::on_emotion_run = {};
    SelectEmotionOption(2);
    CHECK(made.multiplier == 1 && a2f::emotion_runs - heard_before == 3);
    CHECK(made.line->emotion_peak[a2f::kJoy] == 0);
    CHECK(made.entry->degraded);
    CHECK(api->EndUtterance(cut.utterance_id, BFA_kAborted) == BFA_kResultOK);
  }

  puts("provider sentiment: the selected Audio2Emotion model's reading, the "
       "caller's channels and the multiplier reach the API line's animation; "
       "Disabled and missing models feed nothing");
}
