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

#include "../src/better_facial_animations.cc"

namespace {
#define CHECK(x) Check((x), #x, __LINE__)
void Check(bool ok, const char* expression, int line) {
  if (ok) return;
  fprintf(stderr, "%d: %s\n", line, expression);
  std::exit(1);
}

std::array<uint8_t, 0x1000> actor_data{}, base{}, face_data{};
std::array<uint8_t, 0x1000> manager{}, node{};
uint8_t* actor = actor_data.data();
uint8_t* fg = face_data.data();
uint8_t* mtm = manager.data();
uint8_t* GetFace(uint8_t*) { return fg; }
void Handle(uint32_t& h, uint8_t*) { h = 42; }
bool Lookup(const uint32_t& h, uint8_t** out) {
  *out = h == 42 ? actor : nullptr;
  if (*out) ++bfa::Field<LONG>(*out, bfa::kRefrRefCount);
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
bool net_broken = false;  // the network fails every line
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

extern "C" uint64_t __wrap__ZN4skse4AddrEy(uint64_t id) {
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
extern "C" std::string __wrap__ZN3bfa7HeadKeyB5cxx11EPhb(uint8_t*, bool) {
  return "test head";
}
extern "C" bfa::State MapHead(
    uint8_t*, const std::string&, bool,
    std::shared_ptr<const direct::Head>&)
    asm("__wrap__ZN3bfa8FindHeadEPhRKNSt7__cxx1112basic_stringIcSt11char_traits"
        "IcESaIcEEEbRSt10shared_ptrIKN6direct4HeadEE");
extern "C" bfa::State MapHead(
    uint8_t*, const std::string&, bool,
    std::shared_ptr<const direct::Head>& head) {
  ++map_calls;
  head = std::make_shared<direct::Head>();
  return bfa::State::kReady;
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
  CHECK(faces.empty() && FailedSpeech(fg) && fg[kFgDirty]);
  for (int i = 0; i < direct::kNumExpressions; ++i)
    CHECK(expr[i] == (direct::PlaysExpression(i) ? .6f : 0.f));
  for (int i = 0; i < direct::kNumPhonemes; ++i) CHECK(phon[i] == .4f);
  Drain(keeper);
  CHECK(FailedSpeech(fg) && playing.at(fg).entry->state == State::kFailed);
  SeedOwnedFace(expr, phon);
  RestoreExpressions(fg, world_epoch);
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

void TestPlayback(uint32_t runtime, a2f::NetKeeper& keeper) {
  using namespace bfa;
  skse::SetRuntime(runtime);
  ResetPlayback(false);
  cache.clear();
  queue.clear();
  void* vtable[0x100]{};
  vtable[kVfGetFaceGenAnimationData] = reinterpret_cast<void*>(&GetFace);
  Field<void**>(actor, 0) = vtable;
  Field<uint8_t*>(actor, kRefrBase) = base.data();
  actor[kFormType] = kFtActor;
  base[kFormType] = kFtNpc;
  Field<uint32_t>(actor, kFormId) = 0x123;
  Field<LONG>(actor, kRefrRefCount) = 1;
  Field<uint32_t>(mtm, kMtmSpeaker) = 42;
  Field<uint8_t*>(node.data(), kNodeFg.Get()) = fg;
  Field<uint8_t*>(node.data(), kAvUserData.Get()) = actor;
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
  auto& ex = Field<skse::KeyframeMultiple>(fg, skse::kFgExpr3);
  ex.values = expr;
  ex.count = direct::kNumExpressions;
  auto& ph = Field<skse::KeyframeMultiple>(fg, skse::kFgPhon3);
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
  // A matched lip resource keeps vanilla ownership, then its removal ends it.
  BeginSpeech(actor, stem, 12, 2, 0, 0, t, world_epoch);
  std::array<uint8_t, 0x20> lip_resource{};
  Field<uint32_t>(lip_resource.data(), 0x0C) = 0x30000000;
  Field<uint8_t*>(fg, skse::kFgDialogue) = lip_resource.data();
  Apply(fg, t, false, actor, 42, 12, 1);
  CHECK(playing.at(fg).lip_seen);
  Field<uint8_t*>(fg, skse::kFgDialogue) = nullptr;
  Apply(fg, t, false, actor, 42, 12, 1);
  CHECK(playing.empty());
  BeginSpeech(actor, stem, 12, 2, 0, 0, t, world_epoch);
  CHECK(playing.count(fg));
  ResetPlayback(true);
  CHECK(!FailedSpeech(fg) && faces.empty());
  ResetPlayback(false);
}

int main() {
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
       "during inference, transitions, late readiness, completion and load "
       "cleanup (SE/AE/VR)");
}
