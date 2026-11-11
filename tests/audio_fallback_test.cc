#include "runtime_fixture.h"
// Production audio and playback paths; engine access, head mapping and the
// inference network are mocked.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "../src/plugin/better_facial_animations.cc"

namespace {
#define CHECK(x) Check((x), #x, __LINE__)
void Check(bool ok, const char* expression, int line) {
  if (ok) return;
  fprintf(stderr, "%d: %s\n", line, expression);
  std::exit(1);
}

std::array<uint8_t, 0x1000> actor_data{}, base{}, race{}, face_data{};
std::array<uint8_t, 0x1000> manager{}, node{};
uint8_t* actor = actor_data.data();
uint8_t* fg = face_data.data();
uint8_t* mtm = manager.data();
uint8_t* GetFace(uint8_t*) { return fg; }
void Handle(uint32_t& h, uint8_t*) { h = 42; }
bool Lookup(const uint32_t& h, uint8_t** out) {
  *out = h == 42 ? actor : nullptr;
  if (*out) ++bfa::Field<LONG>(*out, 0x28);
  return *out != nullptr;
}
void Resource(skse::ResId* id, const char*) { *id = {}; }
std::unordered_map<std::string, std::vector<uint8_t>> files;
std::array<uint8_t, 16> stream{};
const std::vector<uint8_t>* opened = nullptr;
size_t offset = 0;
uint32_t Read(void*, void* out, uint32_t n, uint32_t*, uint32_t) {
  const auto count = std::min<size_t>(n, opened->size() - offset);
  memcpy(out, opened->data() + offset, count);
  offset += count;
  return static_cast<uint32_t>(count);
}
void* Open(void* ptr, const char* path, uint64_t, void*) {
  auto* s = static_cast<uint8_t*>(ptr);
  auto it = files.find(path);
  if (it != files.end()) {
    opened = &it->second;
    offset = 0;
    bfa::Field<uint32_t>(stream.data(), 8) = opened->size();
    bfa::Field<void*>(s, 0x20) = stream.data();
    bfa::Field<decltype(&Read)>(s, 0x10) = &Read;
  }
  return ptr;
}
void Close(void*) {}
std::vector<uint8_t> Wav(bool audible) {
  constexpr int samples = 3200;
  std::vector<uint8_t> bytes(44 + samples * 2);
  memcpy(bytes.data(), "RIFF", 4);
  auto u32 = [&](int at, uint32_t value) {
    memcpy(bytes.data() + at, &value, 4);
  };
  auto u16 = [&](int at, uint16_t value) {
    memcpy(bytes.data() + at, &value, 2);
  };
  u32(4, bytes.size() - 8);
  memcpy(bytes.data() + 8, "WAVEfmt ", 8);
  u32(16, 16);
  u16(20, 1);
  u16(22, 1);
  u32(24, 16000);
  u32(28, 32000);
  u16(32, 2);
  u16(34, 16);
  memcpy(bytes.data() + 36, "data", 4);
  u32(40, samples * 2);
  for (int i = 0; i < samples; ++i)
    u16(44 + i * 2, audible ? static_cast<int16_t>(1000 * sin(i * .1)) : 0);
  return bytes;
}
int runs = 0;
int map_calls = 0;
std::string head_key = "test head";
bfa::State map_state = bfa::State::kReady;
bool net_broken = false;  // the network fails every line
const direct::Model* mapped_with = nullptr;  // the last HeadKey's mapping
int perf_runs = 0;
int64_t perf_identity = -1;
bool perf_switch = false;  // the user selects another model mid-line
void SelectPerformance(bool on) {
  bfa::ScopedLock guard(bfa::settings_lock);
  for (const bfa::Param& p : bfa::kParams)
    if (!strcmp(p.key, "UsePerformanceModel")) *p.flag = on;
}
struct PerformanceNet : a2f::Net {
  int Version() const override { return a2f::kNetPerformance; }
  int Batch() const override { return 1; }
  bool Run(const float*, const float*, int, float*, std::string&) override {
    return false;
  }
  bool RunPerformance(const float*, int64_t id, const float*,
                      const a2f::PerformanceState& state, float* motion,
                      a2f::PerformanceState& next, std::string&) override {
    next.frames =
        std::min<int64_t>(256, state.frames + a2f::kPerformanceHopFrames);
    ++perf_runs;
    perf_identity = id;
    if (perf_switch) SelectPerformance(false);
    std::fill_n(motion,
                a2f::kPerformanceHopFrames * a2f::kPerformanceChannels, 0.f);
    return true;
  }
};
struct Net : a2f::Net {
  int Batch() const override { return 2; }
  bool Run(const float*, const float*, int b, float* out,
           std::string& err) override {
    ++runs;
    if (net_broken) {
      err = "broken";
      return false;
    }
    std::fill_n(out, b * a2f::kNumOut, 0.f);
    return true;
  }
};
void Drain(a2f::NetKeeper& keeper) {
  while (!bfa::queue.empty()) {
    auto job = bfa::queue.front();
    bfa::queue.pop_front();
    bfa::Process(job, keeper);
  }
}

}  // namespace

extern "C" uint64_t TestAddress(uint64_t id) {
  using namespace bfa;
  if (id == kIdStreamCtor.Get()) return reinterpret_cast<uint64_t>(&Open);
  if (id == kIdStreamDtor.Get()) return reinterpret_cast<uint64_t>(&Close);
  if (id == kIdMenuTopicManager.Get()) return reinterpret_cast<uint64_t>(&mtm);
  if (id == kIdCreateHandle.Get()) return reinterpret_cast<uint64_t>(&Handle);
  if (id == kIdLookupReference.Get())
    return reinterpret_cast<uint64_t>(&Lookup);
  if (id == kIdGenerateFromPath.Get())
    return reinterpret_cast<uint64_t>(&Resource);
  return 0;
}
extern "C" std::string
TestHeadKey(
    uint8_t*, bool, std::shared_ptr<const direct::Model> mapping) {
  mapped_with = mapping.get();
  return head_key;
}
extern "C" bfa::State MapHead(uint8_t*, const std::string&, bool,
                              std::shared_ptr<const direct::Head>& head,
                              std::shared_ptr<const direct::Model>) {
  ++map_calls;
  head.reset();
  if (map_state == bfa::State::kReady) {
    auto mapped = std::make_shared<direct::Head>();
    mapped->shapes.resize(1);
    mapped->shapes[0].part = direct::Part::kHead;
    head = mapped;
  }
  return map_state;
}

void TestHeadRebuild() {
  using namespace bfa;
  ResetPlayback(false);
  auto line = std::make_shared<a2f::Line>();
  line->frames = 3;
  line->direct_dim = direct::kNumMouthCoef;
  line->direct.assign(line->frames * line->direct_dim, .6f);
  auto entry = std::make_shared<Entry>();
  entry->state = State::kReady;
  entry->line = line;
  Playing p;
  p.actor = actor;
  p.actor_handle = 42;
  p.entry = entry;
  p.direct_frame = 1;
  entry->model_state = SnapshotModel();
  p.snap = true;
  playing[fg] = p;
  Face f;
  f.actor = actor;
  f.actor_handle = 42;
  f.until = play_clock + 100;
  f.level[2] = .8f;
  faces[fg] = f;
  Count();
  head_key = "rebuild-0";
  AddDirect(node.data(), true);
  CHECK(!playing.at(fg).snap);  // mapping became ready on this call
  CHECK(faces.at(fg).level[2] == .8f);
  const size_t dim = line->direct_dim + direct::kNumPhonemes + 1;
  auto retained = [&] {
    std::vector<float> pose(dim, 0.f);
    CHECK(faces.at(fg).transition.Blend(nullptr, Now(), 1, true, false, pose));
    return pose[0];
  };
  // Cached head changes must not restart onset, erase expressions, or use
  // the old geometry. Include repeated changes and an unchanged control.
  for (int i = 0; i < 12; ++i) {
    auto old_head = faces.at(fg).head;
    head_key = "rebuild-" + std::to_string(i + 1);
    const int before = map_calls;
    AddDirect(node.data(), true);
    CHECK(map_calls == before + 1);
    CHECK(faces.at(fg).head != old_head);
    CHECK(faces.at(fg).level[2] == .8f);
    CHECK(std::fabs(retained() - .6f) < .01f);
    AddDirect(node.data(), false);
    CHECK(map_calls == before + 1);
  }
  // Pending and failed replacements discard the old mapping, preserving
  // ownership bookkeeping. A later valid identity can recover.
  for (State state : {State::kPending, State::kFailed}) {
    map_state = state;
    head_key += "-unavailable";
    AddDirect(node.data(), true);
    CHECK(faces.at(fg).head_state == state && !faces.at(fg).head);
    CHECK(faces.at(fg).level[2] == .8f);
  }
  map_state = State::kReady;
  head_key += "-recovered";
  AddDirect(node.data(), true);
  CHECK(std::fabs(retained() - .6f) < .01f);
  // Even equal-sized coefficient vectors from different model objects
  // must not carry a pose into an incompatible basis.
  auto replacement = std::make_shared<ModelState>(*entry->model_state);
  auto mapping = std::make_shared<direct::Model>(*replacement->mapping);
  mapping->faces.resize(2);
  replacement->mapping = mapping;
  entry->model_state = replacement;
  AddDirect(node.data(), true);
  CHECK(retained() < .01f);
  playing.at(fg).snap = true;
  AddDirect(node.data(), true);
  CHECK(std::fabs(retained() - .6f) < .01f);
  // The two faces of v3 have different coefficient bases too.
  playing.at(fg).female = !playing.at(fg).female;
  AddDirect(node.data(), true);
  CHECK(retained() < .01f);
  head_key = "test head";
  ResetPlayback(false);
}
void TestSilence() {
  // Quiet, short audio after silence must still pass; noise and DC must not.
  std::vector<float> pcm(a2f::kSampleRate * 3, 0.f);
  CHECK(!a2f::HasAudibleAudio(nullptr, 0));
  CHECK(!a2f::HasAudibleAudio(pcm.data(), pcm.size()));
  for (size_t i = 0; i < pcm.size(); ++i)
    pcm[i] = i % 2 ? 1.f / 32768 : -1.f / 32768;
  CHECK(!a2f::HasAudibleAudio(pcm.data(), pcm.size()));
  std::fill(pcm.begin(), pcm.end(), .2f);
  CHECK(!a2f::HasAudibleAudio(pcm.data(), pcm.size()));
  std::fill(pcm.begin(), pcm.end(), 0.f);
  for (size_t i = pcm.size() - 160; i < pcm.size(); ++i)
    pcm[i] = .001f * std::sin(i * .1);
  CHECK(a2f::HasAudibleAudio(pcm.data(), pcm.size()));
  CHECK(a2f::HasAudibleAudio(pcm.data() + pcm.size() - 160, 160));
}

void TestReplacement(a2f::NetKeeper& keeper, const bfa::Tune& t) {
  using namespace bfa;
  // Both arbitrary silent filenames and Fuz paths are decided by content.
  for (const char* path : {"Sound/Voice/Other.esp/silent",
                           "Sound/Voice/Fuz Ro Doh/Stock_1",
                           "Sound/Voice/missing"}) {
    files.clear();
    cache.clear();
    queue.clear();
    const bool missing = std::string(path) == "Sound/Voice/missing";
    if (!missing) files[std::string(path) + ".wav"] = Wav(false);
    const int before = runs;
    const auto failures = stats.failed;
    auto entry = Request(path, 0, 0, false, false, t, 42, world_epoch);
    Drain(keeper);
    CHECK(entry->state == State::kFailed && !entry->line && runs == before);
    // Silence is not a failure; a missing file is reported as one.
    CHECK(cache.empty() && stats.failed == failures + (missing ? 1 : 0));
    // A later TTS file at this same path must be re-read and generated.
    files[std::string(path) + ".wav"] = Wav(true);
    auto retry = Request(path, 0, 0, false, true, t, 42, world_epoch);
    CHECK(retry != entry);
    Drain(keeper);
    CHECK(retry->state == State::kReady && retry->line && runs > before);
  }
}

void TestPendingMapping(const bfa::Tune& t, a2f::NetKeeper& keeper,
                        const float* phon) {
  using namespace bfa;
  // A fresh face must begin mapping before inference completes.
  faces.clear();
  Count();
  const int before_mapping = map_calls;
  Apply(fg, t, false, actor, 42, 11, 1);
  Expressions(fg, .016f, t, false, actor, 42);
  CHECK(faces.count(fg));
  AddDirect(node.data(), false);
  CHECK(map_calls == before_mapping + 1);
  CHECK(faces.at(fg).head_state == State::kReady);
  // Another pending update preserves that mapping and a retained pose.
  auto old_line = std::make_shared<a2f::Line>();
  std::vector<float> pose(1, 1.f);
  faces.at(fg).transition.Blend(old_line, Now(), 0, true, true, pose);
  Expressions(fg, .016f, t, false, actor, 42);
  CHECK(faces.at(fg).head_state == State::kReady);
  CHECK(faces.at(fg).head_key == "test head");
  pose[0] = 0;
  CHECK(faces.at(fg).transition.Blend(nullptr, Now(), 1, true, false, pose));
  CHECK(pose[0] > .99f);
  for (int i = 0; i < direct::kNumPhonemes; ++i) CHECK(phon[i] == .4f);
  Drain(keeper);
  CHECK(!FailedSpeech(fg) && playing.at(fg).entry->state == State::kReady);
  // Use the mapping prepared by the actual expression/morph flow.
  Expressions(fg, .016f, t, false, actor, 42);
  CHECK(Apply(fg, t, false, actor, 42, 11, 1) & kFaceChanged);
  CHECK(playing.at(fg).direct_frame >= 0);
}

void SeedOwnedFace(float* expr, float* phon) {
  using namespace bfa;
  Face f;
  f.actor = actor;
  f.actor_handle = 42;
  f.ownership_blend = f.speech_blend = 1;
  f.head_state = State::kReady;
  std::fill_n(f.src, direct::kNumExpressions, .6f);
  auto line = std::make_shared<a2f::Line>();
  std::vector<float> pose(
      1 + direct::kNumMouthCoef + direct::kNumPhonemes, 1);
  f.transition.Blend(line, Now(), 0, true, true, pose);
  faces[fg] = f;
  std::fill_n(expr, direct::kNumExpressions, 0);
  std::fill_n(phon, direct::kNumPhonemes, .4f);
  Count();
}

void TestHandoff(const bfa::Tune& t, a2f::NetKeeper& keeper,
                 const std::string& stem, float* expr, float* phon) {
  using namespace bfa;
  BeginSpeech(actor, "Sound/Voice/Skyrim.esm/normal", 8, 3, 0, 0, t,
              world_epoch);
  CHECK(!FailedSpeech(fg) && queue.size() == 1);
  Drain(keeper);
  CHECK(!FailedSpeech(fg));
  SeedOwnedFace(expr, phon);
  BeginSpeech(actor, stem, 9, 10, 0, 0, t, world_epoch);
  CHECK(!FailedSpeech(fg) && queue.size() == 1);
  Drain(keeper);
  CHECK(FailedSpeech(fg));
  // Morph call before the animation update must release the old pose.
  AddDirect(node.data(), false);
  CHECK(faces.empty() && FailedSpeech(fg) && fg[0x218]);
  for (int i = 0; i < direct::kNumExpressions; ++i)
    CHECK(expr[i] == (direct::PlaysExpression(i) ? .6f : 0.f));
  for (int i = 0; i < direct::kNumPhonemes; ++i) CHECK(phon[i] == .4f);
  Drain(keeper);
  CHECK(FailedSpeech(fg) && playing.at(fg).entry->state == State::kFailed);
  SeedOwnedFace(expr, phon);
  RestoreFaceKeyframes(fg, world_epoch);
  Apply(fg, t, false, actor, 42, 9, 1);
  CHECK(playing.count(fg));  // vanilla until its sound ends
  CHECK(Expressions(fg, .1f, t, false, actor, 42) & kFaceChanged);
  CHECK(faces.empty() && FailedSpeech(fg));
  for (int i = 0; i < direct::kNumExpressions; ++i)
    CHECK(expr[i] == (direct::PlaysExpression(i) ? .6f : 0.f));
  for (int i = 0; i < direct::kNumPhonemes; ++i) CHECK(phon[i] == .4f);
  CHECK(Expressions(fg, .1f, t, false, actor, 42) == kNoUpdate);
  // Sound completion allows normal expression ownership again.
  playing.at(fg).voice.requested_at = Now() - 2;
  Apply(fg, t, false, actor, 42, 9, 0);
  CHECK(!FailedSpeech(fg) && playing.empty());
  AddDirect(node.data(), false);  // the morph hook finds the partner again
  Expressions(fg, .1f, t, false, actor, 42);
  CHECK(faces.count(fg));
}

// A line the network fails is a failure, not silence: counted, cached, and
// vanilla until its sound ends, like a silent line.
void TestFailedGeneration(const bfa::Tune& t, a2f::NetKeeper& keeper,
                          float* expr, float* phon) {
  using namespace bfa;
  const auto failures = stats.failed;
  net_broken = true;
  BeginSpeech(actor, "Sound/Voice/Skyrim.esm/broken", 13, 3, 0, 0, t,
              world_epoch);
  Drain(keeper);
  net_broken = false;
  CHECK(FailedSpeech(fg) && stats.failed == failures + 1);
  CHECK(std::any_of(cache.begin(), cache.end(), [](const auto& e) {
    return e.second->state == State::kFailed;
  }));
  SeedOwnedFace(expr, phon);
  Apply(fg, t, false, actor, 42, 13, 1);
  CHECK(playing.count(fg));
  CHECK(Expressions(fg, .1f, t, false, actor, 42) & kFaceChanged);
  CHECK(faces.empty());
  playing.at(fg).voice.requested_at = Now() - 2;
  Apply(fg, t, false, actor, 42, 13, 0);
  CHECK(playing.empty());
}

// While a line is generated the face's expressions fade back to the
// engine's over kOwnOut instead of snapping, and fade in again when ready.
void TestPendingOwnership(const bfa::Tune& t, a2f::NetKeeper& keeper,
                          float* expr, float* phon) {
  using namespace bfa;
  SeedOwnedFace(expr, phon);
  BeginSpeech(actor, "Sound/Voice/Skyrim.esm/owned", 14, 3, 0, 0, t,
              world_epoch);
  CHECK(queue.size() == 1);
  Expressions(fg, .016f, t, false, actor, 42);
  CHECK(faces.count(fg) && faces.at(fg).speech_blend < 1);
  CHECK(faces.at(fg).ownership_blend > .9f &&
        faces.at(fg).ownership_blend < 1);
  for (int i = 0; i < 4; ++i) Expressions(fg, .5f, t, false, actor, 42);
  CHECK(faces.count(fg) && faces.at(fg).ownership_blend == 0);
  Drain(keeper);
  Expressions(fg, .016f, t, false, actor, 42);
  CHECK(faces.at(fg).ownership_blend > 0);
  playing.erase(fg);
  Count();
}

// The sound is over: a played line also runs its forced-reset tail.
void EndLine(const bfa::Tune& t, uint32_t sound) {
  using namespace bfa;
  playing.at(fg).voice.requested_at = Now() - 2;
  for (int i = 0; i <= kDirectTail && !playing.empty(); ++i)
    Apply(fg, t, false, actor, 42, sound, 0);
  CHECK(playing.empty());
}

// The performance network through Process: the identity is the speaker's
// sex, the line 30 fps and as wide as the v3 heads. After the settings
// switch models a playing line keeps the mapping it was made for; a line in
// the network when they switch is cancelled, neither failed nor cached.
void TestPerformance(a2f::NetKeeper& keeper, const bfa::Tune& t) {
  using namespace bfa;
  a2f::NetKeeper perf_keeper(
      false,
      [](bool, std::string&, std::string&) {
        return std::unique_ptr<a2f::Net>(new PerformanceNet);
      },
      [] { return Now(); });
  for (const char* name : {"v3", "perf", "cancelled"})
    files[std::string("Sound/Voice/Skyrim.esm/") + name + ".wav"] = Wav(true);
  cache.clear();
  queue.clear();
  playing.clear();
  faces.clear();
  Count();
  BeginSpeech(actor, "Sound/Voice/Skyrim.esm/v3", 20, 3, 0, 0, t, world_epoch);
  Drain(keeper);
  CHECK(playing.at(fg).entry->state == State::kReady);
  Apply(fg, t, false, actor, 42, 20, 1);
  Expressions(fg, .016f, t, false, actor, 42);
  mapped_with = nullptr;
  AddDirect(node.data(), false);
  CHECK(mapped_with == &model);
  // The worker switched to the performance model with its own mapping.
  auto perf_mapping = std::make_shared<direct::Model>();
  perf_mapping->network = a2f::kNetV3;
  perf_mapping->faces.resize(2);
  auto perf_state = std::make_shared<ModelState>();
  perf_state->network = a2f::kNetPerformance;
  perf_state->gpu = false;
  perf_state->mapping = perf_mapping;
  std::atomic_store(&active_model,
                    std::shared_ptr<const ModelState>(perf_state));
  SelectPerformance(true);
  const Tune pt = SnapshotTune();
  CHECK(pt.SelectedModel() == a2f::kNetPerformance);
  mapped_with = nullptr;
  AddDirect(node.data(), false);
  CHECK(mapped_with == &model);  // the playing line keeps its mapping
  EndLine(pt, 20);
  const int before = runs, perf_before = perf_runs;
  BeginSpeech(actor, "Sound/Voice/Skyrim.esm/perf", 21, 3, 0, 0, pt,
              world_epoch);
  CHECK(queue.size() == 1 && queue.front().entry->model_state == perf_state);
  Drain(perf_keeper);
  CHECK(runs == before && perf_runs == perf_before + 1 && perf_identity == 1);
  const auto entry = playing.at(fg).entry;
  CHECK(entry->state == State::kReady && entry->line);
  CHECK(entry->line->fps == a2f::kPerformanceFps &&
        entry->line->direct_dim == a2f::kPerformanceChannels &&
        entry->line->frames == 6 && entry->line->eyes.empty());
  Apply(fg, pt, false, actor, 42, 21, 1);
  Expressions(fg, .016f, pt, false, actor, 42);
  mapped_with = nullptr;
  AddDirect(node.data(), false);
  CHECK(mapped_with == perf_mapping.get());
  EndLine(pt, 21);
  const auto failures = stats.failed, lines = stats.lines;
  perf_switch = true;
  BeginSpeech(actor, "Sound/Voice/Skyrim.esm/cancelled", 22, 3, 0, 0, pt,
              world_epoch);
  const auto cancelled = queue.front().entry;
  Drain(perf_keeper);
  perf_switch = false;
  CHECK(perf_runs == perf_before + 2);
  CHECK(cancelled->state == State::kFailed && !cancelled->line);
  CHECK(stats.failed == failures && stats.lines == lines);
  CHECK(std::none_of(cache.begin(), cache.end(), [&](const auto& e) {
    return e.second == cancelled;
  }));
  CHECK(SnapshotTune().SelectedModel() == a2f::kNetV3);
  std::atomic_store(&active_model, std::shared_ptr<const ModelState>());
  playing.clear();
  faces.clear();
  Count();
}

void TestSpeechTiming() {
  using namespace bfa;
  std::array<uint8_t, 0x20> resource{}, replacement{};
  Field<uint32_t>(resource.data(), 0x0C) = 0x30000000;
  Field<uint32_t>(replacement.data(), 0x0C) = 0x30000000;
  Field<uint32_t>(replacement.data(), 0) = 1;
  Playing p;
  bool lip;
  const char* why;
  auto reset = [&] {
    Field<uint8_t*>(fg, 0x228) = resource.data();
    p = Playing();
    p.entry = std::make_shared<Entry>();
    p.entry->state = State::kReady;
    auto line = std::make_shared<a2f::Line>();
    line->frames = 360;
    line->fps = 60;
    p.entry->line = line;
    p.voice = {12, 100, 6};
    p.lip_key = skse::KeyOf(*skse::ActiveLine(fg));
  };
  auto sample = [&](double elapsed, float timer, uint32_t sound = 12,
                    int state = 1) {
    Field<float>(fg, 0x12C) = timer;
    return LineClock(fg, p, 100 + elapsed, lip, why, sound, state);
  };
  for (bool female : {false, true}) {
    reset();
    p.female = female;
    CHECK(sample(0, 0) == 0 && !why);
    float last = 0;
    for (int frame = 1; frame <= 180; ++frame) {
      const float t = frame / 60.f;
      const float shown = sample(t, 2 * t);
      CHECK(!why && shown >= last);
      if (t >= 1) CHECK(std::fabs(shown - t) < 0.02f);
      last = shown;
    }
    CHECK(p.speech_clock.corrected);
    Field<uint8_t*>(fg, 0x228) = nullptr;
    for (int frame = 181; frame <= 330; ++frame) {
      const float t = frame / 60.f;
      CHECK(std::fabs(sample(t, 6) - t) < 0.02f && !why);
    }
    sample(5.6, 6, 13);
    CHECK(why);  // another sound must not sustain the old face
  }
  // Transient sound observations must not switch back to the accelerated
  // native timer, or bypass the stop grace after the lip resource expires.
  reset();
  sample(0, 0);
  for (int frame = 1; frame <= 120; ++frame) sample(frame / 60., frame / 30.f);
  CHECK(p.speech_clock.corrected && !why);
  CHECK(std::fabs(sample(121 / 60., 121 / 30.f, 12, 0) - 121 / 60.f) < .02f);
  CHECK(!why);
  for (int frame = 122; frame <= 180; ++frame)
    CHECK(std::fabs(sample(frame / 60., frame / 30.f) - frame / 60.f) < .02f);
  Field<uint8_t*>(fg, 0x228) = nullptr;
  CHECK(std::fabs(sample(181 / 60., 6, 12, 0) - 181 / 60.f) < .02f);
  CHECK(!why);
  CHECK(std::fabs(sample(182 / 60., 6) - 182 / 60.f) < .02f && !why);
  sample(3.1, 6, 12, 0);
  CHECK(!why);
  sample(3.4, 6, 12, 0);
  CHECK(why);  // a sustained stop still ends the corrected line
  reset();
  CHECK(sample(0, 0) == 0);
  for (int tick = 1; tick <= 600; ++tick) {
    const float timer = (tick / 2) / 60.f;
    CHECK(std::fabs(sample(tick / 120., timer) - timer) < 1e-4f);
    CHECK(!why && !p.speech_clock.corrected);
  }
  CHECK(sample(7, 5) == 5 && !why);
  CHECK(sample(8, 6) == 6 && !why);
  reset();
  CHECK(sample(0, 0) == 0);
  CHECK(sample(4, 5) == 5 && !p.speech_clock.corrected);
  Field<uint8_t*>(fg, 0x228) = replacement.data();
  sample(4.1, 5);
  CHECK(why);  // a different native lip resource still supersedes the line
  reset();
  sample(0, 0);
  sample(1, 1, 12, 0);
  CHECK(!why);  // one stopped frame is not the end of a lip-synced line
  sample(1.05, 1.05);
  CHECK(!why);
  sample(1.5, 1.5, 12, 0);
  CHECK(!why);  // the stopped interval restarts after the sound played again
  sample(1.8, 1.8, 12, 0);
  CHECK(why);  // a sound stopped this long wins over its lip resource
  Field<uint8_t*>(fg, 0x228) = nullptr;
}

void TestPlayback(uint32_t runtime, a2f::NetKeeper& keeper) {
  using namespace bfa;
  skse::SetRuntime(runtime);
  TestSpeechTiming();
  ResetPlayback(false);
  cache.clear();
  queue.clear();
  void* vtable[0x100]{};
  vtable[0x63] = reinterpret_cast<void*>(&GetFace);
  Field<void**>(actor, 0) = vtable;
  Field<uint8_t*>(actor, 0x40) = base.data();
  actor[0x1A] = kFtActor;
  base[0x1A] = kFtNpc;
  // Conversation admission requires a real supported FaceGen race.
  Field<uint8_t*>(actor, (skse::Runtime() >= skse::Version(1, 6, 629) ? 0x1f8 : 0x1f0)) = race.data();
  Field<uint32_t>(race.data(), 0x14) = 0x13746;  // Nord
  Field<uint32_t>(race.data(), 0x108) = 2;
  CHECK(SpeakerOk(actor));
  Field<uint32_t>(actor, 0x14) = 0x123;
  Field<LONG>(actor, 0x28) = 1;
  Field<uint32_t>(mtm, 0x68) = 42;
  Field<uint8_t*>(node.data(), (skse::IsVr() ? 0x178 : 0x150)) = fg;
  Field<uint8_t*>(node.data(), (skse::IsVr() ? 0x110 : 0xF8)) = actor;
  Tune t;
  std::string stem;
  CHECK(VoiceStem("Data\\Sound\\Voice\\Fuz Ro Doh\\Stock_10.xwm", stem));
  TestReplacement(keeper, t);
  files.clear();
  cache.clear();
  queue.clear();
  files["Sound/Voice/Skyrim.esm/normal.wav"] = Wav(true);
  files["Sound/Voice/Skyrim.esm/next.wav"] = Wav(true);
  files["Sound/Voice/Skyrim.esm/broken.wav"] = Wav(true);
  files["Sound/Voice/Skyrim.esm/owned.wav"] = Wav(true);
  files[stem + ".wav"] = Wav(false);
  float expr[direct::kNumExpressions], phon[direct::kNumPhonemes];
  auto& ex = FixtureKeyframe(fg, 0xC0);
  ex.values = expr;
  ex.count = direct::kNumExpressions;
  auto& ph = FixtureKeyframe(fg, 0x140);
  ph.values = phon;
  ph.count = direct::kNumPhonemes;
  TestHandoff(t, keeper, stem, expr, phon);
  TestFailedGeneration(t, keeper, expr, phon);
  TestPendingOwnership(t, keeper, expr, phon);
  BeginSpeech(actor, stem, 10, 2, 0, 0, t, world_epoch);
  CHECK(!FailedSpeech(fg));
  BeginSpeech(actor, "Sound/Voice/Skyrim.esm/next", 11, 3, 0, 0, t,
              world_epoch);
  CHECK(!FailedSpeech(fg) && queue.size() == 2);
  TestPendingMapping(t, keeper, phon);
  TestPerformance(keeper, t);
  TestHeadRebuild();
  // An exhausted lip resource cannot end a voice that is still playing.
  BeginSpeech(actor, stem, 12, 2, 0, 0, t, world_epoch);
  std::array<uint8_t, 0x20> lip_resource{};
  Field<uint32_t>(lip_resource.data(), 0x0C) = 0x30000000;
  Field<uint8_t*>(fg, 0x228) = lip_resource.data();
  Apply(fg, t, false, actor, 42, 12, 1);
  CHECK(playing.at(fg).lip_seen);
  Field<uint8_t*>(fg, 0x228) = nullptr;
  Apply(fg, t, false, actor, 42, 12, 1);
  CHECK(playing.count(fg));
  playing.at(fg).voice.requested_at -= 1;
  Apply(fg, t, false, actor, 42, 12, 0);
  CHECK(playing.count(fg));  // retain the face during a transient stopped report
  playing.at(fg).voice.stopped_at -= a2f::VoiceStart::kLipStop + .01;
  Apply(fg, t, false, actor, 42, 12, 0);
  CHECK(playing.empty());
  // A skipped line must clear both the cached native mouth and the saved
  // BFA pose, for player and NPC, even when the lip resource remains alive.
  // The release fades into what the console/script keyframes hold (held),
  // read each frame; with none set it is the fade to neutral.
  for (int round = 0; round < 8; ++round) {
    {
      const bool player = round & 1, external = round & 2, held = round & 4;
      t.male.face_out = external ? 0.f : .8f;
      t.player_face.face_out = external ? 0.f : .4f;
      Field<uint32_t>(actor, 0x14) = player ? kPlayerFormId : 0x123;
      BeginSpeech(actor, stem, 12, 2, 0, 0, t, world_epoch);
      auto& p = playing.at(fg);
      p.direct_frames = 12;
      p.direct_tail = kDirectTail;
      p.voice.seen = true;
      p.voice.requested_at = Now() - 2;
      p.voice.stopped_at = Now() - 1;
      if (external) {
        p.external = std::make_shared<PlayerSpeech>();
        p.external->stopped = true;
      }
      float source[direct::kNumPhonemes];
      std::fill_n(source, direct::kNumPhonemes, .8f);
      auto& source_phon = FixtureKeyframe(fg, 0x120);
      source_phon.values = source;
      source_phon.count = direct::kNumPhonemes;
      // Console/script channels (mfg, MFG Fix, ZaZ) belong to other writers;
      // the expression one is reset only without SetExpressionOverride.
      constexpr uint32_t kModifiers = 14;  // engine modifier channels
      const bool expr_override = player != external;
      Field<bool>(fg, 0x21E) = expr_override;
      float console_expr[direct::kNumExpressions];
      float console_mod[kModifiers];
      float console_phon[direct::kNumPhonemes];
      std::fill_n(console_expr, direct::kNumExpressions, held ? .6f : 0.f);
      std::fill_n(console_mod, kModifiers, held ? .7f : 0.f);
      std::fill_n(console_phon, direct::kNumPhonemes, held ? .9f : 0.f);
      console_mod[2] = 0;  // not set: stays with the cleared final
      float mods[kModifiers];
      std::fill_n(mods, kModifiers, .2f);
      auto& final_mod = FixtureKeyframe(fg, 0x100);
      final_mod.values = mods;
      final_mod.count = kModifiers;
      // What the finals show of the console values at a point of the fade.
      const auto shown = [&](float release) {
        std::vector<float> all;
        for (float value : console_expr)
          all.push_back(expr_override ? value * release : 0.f);
        for (float value : console_mod) all.push_back(value * release);
        for (float value : console_phon) all.push_back(value * release);
        return all;
      };
      const auto finals = [&] {
        std::vector<float> all(expr, expr + direct::kNumExpressions);
        all.insert(all.end(), mods, mods + kModifiers);
        all.insert(all.end(), phon, phon + direct::kNumPhonemes);
        return all;
      };
      const std::vector<float> cleared(finals().size(), 0.f);
      const auto same = [](const std::vector<float>& a,
                           const std::vector<float>& b) {
        return !memcmp(a.data(), b.data(), a.size() * sizeof(float));
      };
      const std::pair<size_t, std::pair<float*, uint32_t>> console[] = {
          {0x20, {console_expr, direct::kNumExpressions}},
          {0x60, {console_mod, kModifiers}},
          {0x80, {console_phon, direct::kNumPhonemes}}};
      for (const auto& [offset, array] : console) {
        FixtureKeyframe(fg, offset).values = array.first;
        FixtureKeyframe(fg, offset).count = array.second;
      }
      std::fill_n(phon, direct::kNumPhonemes, .8f);
      std::fill_n(expr, direct::kNumExpressions, .5f);
      auto& face = faces[fg];
      float tongue[direct::kNumPhonemes];
      face.phonemes.Capture(ph, 1, tongue);
      std::vector<float> pose(1, 1.f);
      face.transition.Blend(std::make_shared<a2f::Line>(), Now(), 0, true,
                            true, pose);
      Count();
      const int update = Apply(fg, t, false, actor, 42, 12, 0);
      CHECK(update & kFaceChanged);
      CHECK(update & kKeepUpdating);
      CHECK(playing.empty());
      const float expected_fade = player ? t.player_face.face_out : t.male.face_out;
      if (expected_fade > 0) {
        CHECK(faces.count(fg));
        const double fade_start = faces.at(fg).neutral_started;
        const float fade = faces.at(fg).neutral_duration;
        CHECK(fade == expected_fade);
        // The actual stored speech pose eases halfway to zero, rather than
        // snapping or fading back to the engine's stale open-mouth phonemes.
        pose[0] = 0;
        CHECK(faces.at(fg).transition.Blend(nullptr, fade_start + fade * .5,
                                          fade, true, false, pose));
        CHECK(std::fabs(pose[0] - .5f) < .001f);
        // The line's last frame is all the plugin's pose: nothing of the
        // console values yet.
        CHECK(same(finals(), cleared));
        std::vector<float> before = finals();
        for (int step = 1; step <= 3; ++step) {
          // A script changes and clears its values during the fade.
          if (step == 3 && held) {
            console_phon[0] = .3f;
            console_mod[1] = 0;
          }
          testing::clock_offset = testing::clock_offset.load() + fade * .25;
          const double from = Now();
          CHECK(Expressions(fg, fade * .25f, t, false, actor, 42) &
                kKeepUpdating);
          const double to = Now();
          CHECK(faces.count(fg));
          const std::vector<float> now = finals();
          const std::vector<float>
              low = shown(a2f::Smoothstep(
                  static_cast<float>(from - fade_start) / fade)),
              high = shown(a2f::Smoothstep(
                  static_cast<float>(to - fade_start) / fade));
          const size_t changed_phon = direct::kNumExpressions + kModifiers;
          const size_t cleared_mod = direct::kNumExpressions + 1;
          for (size_t i = 0; i < now.size(); ++i) {
            CHECK(now[i] >= low[i] && now[i] <= high[i]);
            // toward the console values, never back or past them
            if (step < 3 || (i != changed_phon && i != cleared_mod))
              CHECK(now[i] >= before[i]);
          }
          CHECK((now[direct::kNumExpressions + kModifiers + 1] > 0) == held);
          if (!held) CHECK(same(now, cleared));
          before = now;
        }
        testing::clock_offset = testing::clock_offset.load() + fade * .25 + .01;
        CHECK(Expressions(fg, fade * .25f, t, false, actor, 42) & kFaceChanged);
      }
      CHECK(!faces.count(fg));
      RestoreFaceKeyframes(fg, world_epoch);
      for (float value : source) CHECK(value == 0);
      // Released exactly on the console values as they are now; with none
      // set, on the cleared finals as before.
      CHECK(same(finals(), shown(1)));
      if (!held) CHECK(same(finals(), cleared));
      CHECK(console_phon[0] == (!held ? 0.f : expected_fade > 0 ? .3f : .9f));
      for (int i = 1; i < direct::kNumPhonemes; ++i)
        CHECK(console_phon[i] == (held ? .9f : 0.f));
      for (float value : console_expr)
        CHECK(value == (expr_override && held ? .6f : 0.f));
      for (uint32_t i = 0; i < kModifiers; ++i)
        CHECK(console_mod[i] ==
              (!held || i == 2 || (i == 1 && expected_fade > 0) ? 0.f : .7f));
      for (const auto& entry : console) {
        FixtureKeyframe(fg, entry.first).values = nullptr;
        FixtureKeyframe(fg, entry.first).count = 0;
      }
      final_mod.values = nullptr;
      final_mod.count = 0;
      std::fill_n(phon, direct::kNumPhonemes, 0.f);
      std::fill_n(expr, direct::kNumExpressions, 0.f);
      Field<bool>(fg, 0x21E) = false;
      // Another writer can set the face afterward; no delayed restoration.
      phon[0] = .25f;
      expr[0] = .3f;
      RestoreFaceKeyframes(fg, world_epoch);
      CHECK(phon[0] == .25f && expr[0] == .3f);
      source_phon.values = nullptr;
      source_phon.count = 0;
    }
  }
  t = Tune();
  Field<uint32_t>(actor, 0x14) = 0x123;
  BeginSpeech(actor, stem, 12, 2, 0, 0, t, world_epoch);
  CHECK(playing.count(fg));
  ResetPlayback(true);
  CHECK(!FailedSpeech(fg) && faces.empty());
  ResetPlayback(false);
}

int main() {
  bfa::testing::head_key = TestHeadKey;
  bfa::testing::find_head = MapHead;
  bfa::testing::address = TestAddress;
  using namespace bfa;
  model.faces.resize(1);
  model.faces[0].rest.resize(a2f::kNumOut);
  a2f::NetKeeper keeper(
      false,
      [](bool, std::string&, std::string&) { return std::make_unique<Net>(); },
      [] { return Now(); });
  TestSilence();
  for (uint32_t runtime : {skse::Version(1, 5, 97),
                           skse::Version(1, 6, 1170), skse::kVrRuntime}) {
    TestPlayback(runtime, keeper);
  }
  puts("Audio fallback contracts passed: silence, missing audio, TTS retry, "
       "handoff, failed generation, ownership fade while generating, mapping "
       "during inference, transitions, late readiness, completion, load "
       "cleanup and performance lines (SE/AE/VR)");
}
