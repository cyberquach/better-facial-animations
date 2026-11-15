#include "runtime_fixture.h"
// The provider interface (include/a2f/speech/BFA_API.h) over the production
// plugin: deferred admission, lifecycle events, playback timing and reply
// cleanup. Engine tasks/lookups and generated animation are controlled
// fixtures.
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "../src/plugin/better_facial_animations.cc"
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

// The events, as the listener thread delivers them.
SRWLOCK events_lock = SRWLOCK_INIT;
std::vector<BFA_Event> got;
void OnEvent(const BFA_Event* e, void* context) {
  CHECK(context == &got && e->version == BFA_API_VERSION &&
        e->size == sizeof(BFA_Event));
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

std::vector<int16_t> Sine(int rate, double seconds, double hz = 220) {
  std::vector<int16_t> pcm(static_cast<size_t>(rate * seconds));
  for (size_t i = 0; i < pcm.size(); ++i)
    pcm[i] = static_cast<int16_t>(8000 *
                                  std::sin(i * 6.283185307179586 * hz / rate));
  return pcm;
}

std::vector<uint8_t> Wave() {
  auto pcm = Sine(16000, .5);
  std::vector<uint8_t> wav(44 + pcm.size() * 2);
  auto u32 = [&](int at, uint32_t v) { memcpy(wav.data() + at, &v, 4); };
  auto u16 = [&](int at, uint16_t v) { memcpy(wav.data() + at, &v, 2); };
  memcpy(wav.data(), "RIFF", 4);
  u32(4, wav.size() - 8);
  memcpy(wav.data() + 8, "WAVEfmt ", 8);
  u32(16, 16);
  u16(20, 1);
  u16(22, 1);
  u32(24, 16000);
  u32(28, 32000);
  u16(32, 2);
  u16(34, 16);
  memcpy(wav.data() + 36, "data", 4);
  u32(40, pcm.size() * 2);
  memcpy(wav.data() + 44, pcm.data(), pcm.size() * 2);
  return wav;
}
std::vector<SKSE::Impl::TaskDelegate*> pending_tasks;
constexpr int kDispatchBudgetMs = 3000;  // bounded: a hang is a failure
bool dispatch_probe = false;
uint64_t cancel_on_dispatch = 0;
std::atomic<bool> dispatch_done{false};
void AddTask(void* opaque) {
  if (dispatch_probe) {
    // Model a dispatcher holding the SKSE queue lock: enqueue cannot proceed
    // until the already queued production task returns.
    dispatch_probe = false;
    CHECK(!pending_tasks.empty());
    auto* queued = pending_tasks.back();
    pending_tasks.pop_back();
    dispatch_done = false;
    std::thread dispatcher([queued] {
      if (cancel_on_dispatch) {
        const auto* api = BFA_QueryInterface(BFA_API_VERSION);
        CHECK(api->EndUtterance(cancel_on_dispatch, BFA_kAborted) ==
              BFA_kResultOK);
      }
      queued->Run();
      queued->Dispose();
      dispatch_done = true;
    });
    for (int waited = 0; waited < kDispatchBudgetMs && !dispatch_done;
         waited += 5)
      Sleep(5);
    CHECK(dispatch_done && "provider calls/dispatch queue lock inversion");
    dispatcher.join();
  }
  auto* task = static_cast<SKSE::Impl::TaskDelegate*>(opaque);
  pending_tasks.push_back(task);
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
void Generate() {
  bfa::ScopedLock guard(bfa::lock);
  for (auto& job : bfa::queue) {
    auto output = std::make_shared<a2f::Line>();
    output->frames = 15;
    output->fps = 30;
    output->direct_dim = 1;
    output->direct.assign(15, 0);
    job.entry->state = bfa::State::kReady;
    job.entry->line = output;
    bfa::NotifyWatchers(*job.entry);
  }
  bfa::queue.clear();
}
// Face updates until the line has left and the face has handed back.
void Settle() {
  for (int i = 0; i < 80; ++i) {
    const auto tune = bfa::SnapshotTune();
    bfa::Apply(fg, tune, false, actor, 42, a2f::kNoSound, -1);
    bfa::Expressions(fg, 0.1f, tune, false, actor, 42);
    Sleep(20);
  }
}
struct StopRace {
  const BFA_Interface* api;
  bfa::PlayerSpeechPtr line;
  HANDLE start;
  BFA_Result result = BFA_kResultInternalError;
};
DWORD WINAPI StopFromProvider(void* context) {
  auto& race = *static_cast<StopRace*>(context);
  WaitForSingleObject(race.start, INFINITE);
  race.result = race.api->Stop(race.line->handle, BFA_kSkipped);
  return 0;
}
DWORD WINAPI CompleteFromRenderer(void* context) {
  auto& race = *static_cast<StopRace*>(context);
  WaitForSingleObject(race.start, INFINITE);
  bfa::ScopedLock guard(bfa::lock);
  bfa::ProviderLineStopping(race.line, BFA_kCompleted);
  return 0;
}
void Clear() {
  bfa::ScopedLock guard(bfa::lock);
  bfa::queue.clear();
  bfa::cache.clear();
  bfa::playing.clear();
  bfa::Count();
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
  return 0;  // Unexpected engine calls must not silently succeed.
}

int main() {
  bfa::testing::address = TestAddress;
  using namespace bfa;
  SetSelectedModel(a2f::kNetV3);  // the fixture's network (config.network)
  skse::SetRuntime(skse::Version(1, 6, 1170));
  static void* vtable[0x100]{};
  vtable[0x63] = reinterpret_cast<void*>(&GetFace);
  vtable[(skse::IsVr() ? 0xFB : 0xF9)] = reinterpret_cast<void*>(&NotInFaction);
  Field<void**>(actor, 0) = vtable;
  Field<uint8_t*>(actor, 0x40) = base.data();
  actor[0x1A] = kFtActor;
  base[0x1A] = kFtNpc;
  Field<uint8_t*>(
      actor, (skse::Runtime() >= skse::Version(1, 6, 629) ? 0x1f8 : 0x1f0)) =
      race.data();
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
  CHECK(api && api->size == sizeof(*api) && !BFA_QueryInterface(0));
  BFA_Result result;
  CHECK(!api->RegisterListener(nullptr, nullptr, &result) &&
        result == BFA_kResultInvalidArgument);
  auto listener = api->RegisterListener(OnEvent, &got, &result);
  CHECK(listener && result == BFA_kResultOK);
  auto wav = Wave();
  // Reply ids come from the plugin; invented ones are refused.
  uint64_t ids[17] = {};
  for (int i = 1; i < 17; ++i) {
    ids[i] = api->NewUtteranceId();
    CHECK(ids[i] == (uint64_t)i);
  }
  BFA_Line line{BFA_API_VERSION, sizeof(BFA_Line),    ids[7], 0, kActorId,
                wav.data(),      (uint32_t)wav.size()};
  auto foreign = line;
  foreign.utterance_id = 1000;
  CHECK(!api->Prepare(&foreign, &result) && result == BFA_kResultNotFound);
  CHECK(!api->Prepare(nullptr, &result) &&
        result == BFA_kResultInvalidArgument);
  CHECK(!api->Prepare(&line, nullptr));
  auto bad = line;
  bad.audio_bytes = 4;
  CHECK(!api->Prepare(&bad, &result) && result == BFA_kResultInvalidArgument);
  bad.audio_bytes = BFA_MAX_AUDIO_BYTES + 1;
  CHECK(!api->Prepare(&bad, &result) && result == BFA_kResultInvalidArgument);
  // Engine work is deferred; admission failure arrives as an event.
  auto rejected = api->Prepare(&line, &result);
  CHECK(rejected && result == BFA_kResultPending && queue.empty());
  RunTasks();
  CHECK(Wait(1) && At(0).event == BFA_kDeclined &&
        At(0).reason == BFA_kResultModelNotLoaded);
  CHECK(api->EndUtterance(ids[7], BFA_kAborted) == BFA_kResultOK);
  ready = true;
  net_loaded = true;
  faces[fg] = Face();
  faces[fg].head_state = State::kReady;
  faces[fg].actor = actor;
  faces[fg].actor_handle = 42;
  Count();
  auto h = api->Prepare(&line, &result);
  CHECK(h && result == BFA_kResultPending && queue.empty());
  CHECK(api->Pause(h) == BFA_kResultInvalidState);
  CHECK(api->Start(h, NAN) == BFA_kResultInvalidArgument);
  RunTasks();
  CHECK(queue.size() == 1);
  auto entry = queue.front().entry;
  std::weak_ptr<const std::vector<uint8_t>> audio = queue.front().audio;
  CHECK(api->Start(h, 0) == BFA_kResultPending);
  RunTasks();
  CHECK(playing.size() == 1 && Events() == 1);  // no premature Started
  const auto session = playing.at(fg).external;
  Generate();
  CHECK(Events() == 1);  // generation alone is not animation
  const auto tune = SnapshotTune();
  CHECK(Apply(fg, tune, false, actor, 42, a2f::kNoSound, -1) & kFaceChanged);
  CHECK(Wait(2) && At(1).event == BFA_kStarted && At(1).handle == h);
  CHECK(api->Start(h, .3) == BFA_kResultOK &&
        std::fabs(session->Position(Now()) - .3) < 1e-2);
  CHECK(api->Pause(h) == BFA_kResultOK && api->Pause(h) == BFA_kResultNoChange);
  CHECK(std::fabs(session->Position(Now()) - .3) < 1e-2 && session->Paused());
  CHECK(api->Seek(h, -1) == BFA_kResultInvalidArgument && session->Paused());
  CHECK(api->Seek(h, .2) == BFA_kResultOK &&
        !session->Paused());  // Seek unpauses
  CHECK(std::fabs(session->Position(Now()) - .2) < 1e-2);
  CHECK(ProviderClaims(kActorId));
  CHECK(BeginSpeech(actor, "Sound\\Voice\\bark", 5, 1, 0, 0, tune,
                    world_epoch) == BFA_kResultTakenByOtherLine);
  CHECK(api->Seek(h, 5) == BFA_kResultOK);
  for (int i = 0; i < 8 && !playing.empty(); ++i)
    Apply(fg, tune, false, actor, 42, a2f::kNoSound, -1);  // past its audio
  CHECK(playing.empty() && ProviderClaims(kActorId));
  Sleep(50);
  CHECK(Events() == 2);  // the reply keeps the face: told at its hand-back
  CHECK(api->Seek(h, 0) == BFA_kResultInvalidState);
  // During a sentence gap the held reply suppresses the native smile, but
  // its outgoing speech pose must keep ticking even without native lips.
  {
    Face& face = faces.at(fg);
    face.ownership_blend = face.speech_blend = 1;
    face.until = play_clock + kFaceGrace;
    std::vector<float> pose(1, 1.f);
    face.transition.Blend(entry->line, Now(), 1, true, true, pose);
    pose[0] = 0;
    face.transition.Blend(nullptr, Now(), 1, true, false, pose);
    CHECK(Expressions(fg, .1f, tune, false, actor, 42) & kKeepUpdating);
    CHECK(face.speech_blend == 1 && Events() == 2);
    face.transition = a2f::PoseTransition();
    CHECK(!(Expressions(fg, .1f, tune, false, actor, 42) & kKeepUpdating));
  }
  // The reply's next segment takes the face without a hand-back in between.
  line.segment = 1;
  auto h2 = api->Prepare(&line, &result);
  RunTasks();
  Generate();
  CHECK(api->Start(h2, 0) == BFA_kResultPending);
  RunTasks();
  CHECK(Apply(fg, tune, false, actor, 42, a2f::kNoSound, -1) & kFaceChanged);
  CHECK(Wait(3) && At(2).handle == h2 && At(2).event == BFA_kStarted);
  Expressions(fg, 0.1f, tune, false, actor,
              42);  // the face is wanted (kept after the line)
  CHECK(api->Seek(h2, 5) == BFA_kResultOK);
  for (int i = 0; i < 8 && !playing.empty(); ++i)
    Apply(fg, tune, false, actor, 42, a2f::kNoSound, -1);
  CHECK(playing.empty() && faces.at(fg).leaving.size() == 2);
  CHECK(api->EndUtterance(ids[7], BFA_kCompleted) == BFA_kResultOK);
  CHECK(!ProviderClaims(kActorId) && cache.empty() && !entry->line);
  Settle();  // the face fades to the engine's expressions, then both are told
  CHECK(Wait(5) && At(3).handle == h && At(3).event == BFA_kEnded &&
        At(3).reason == BFA_kCompleted);
  CHECK(At(4).handle == h2 && At(4).event == BFA_kEnded &&
        faces.at(fg).leaving.empty());
  CHECK(api->Seek(h, 0) == BFA_kResultInvalidHandle);
  session->audio.reset();
  CHECK(audio.expired());

  // Prepared ahead, queued, cached and in-flight work is discarded per reply.
  line.utterance_id = ids[8];
  auto a = api->Prepare(&line, &result);
  RunTasks();
  auto cancelled =
      queue.front();  // simulate an inference worker retaining the job
  auto cancelled_line = cancelled.entry->watchers.front();
  line.segment = 1;
  auto b = api->Prepare(&line, &result);  // still in SKSE task queue
  CHECK(a && b && api->EndUtterance(ids[8], BFA_kSkipped) == BFA_kResultOK);
  CHECK(queue.empty() && cache.empty() && cancelled.entry->cancelled &&
        cancelled.entry->watchers.empty());
  CHECK(!cancelled_line->audio && !JobAllowed(cancelled));
  RunTasks();
  CHECK(queue.empty());
  CHECK(Wait(7));
  auto n = Events();
  ProviderLineStarted(cancelled_line);
  CHECK(Events() == n && api->Start(a, 0) == BFA_kResultInvalidHandle);
  CHECK(api->EndUtterance(ids[8], BFA_kCompleted) == BFA_kResultNotFound);

  // The line's emotion is validated and copied at Prepare; null is neutral.
  BFA_Emotion emotion{BFA_API_VERSION, sizeof(BFA_Emotion), {}};
  emotion.channels[BFA_kJoy] = 1;
  line.utterance_id = ids[9];
  line.emotion = &emotion;
  auto m = api->Prepare(&line, &result);
  CHECK(m && result == BFA_kResultPending);
  emotion.channels[BFA_kJoy] = 2;  // the caller's copy is theirs again
  RunTasks();
  CHECK(queue.front().emotion.channels[a2f::kJoy] == 1);
  CHECK(queue.front().emotion.type == a2f::kExplicitEmotion);
  CHECK(api->Stop(m, BFA_kSkipped) == BFA_kResultOK);
  CHECK(api->EndUtterance(ids[9], BFA_kSkipped) == BFA_kResultOK);
  emotion.channels[0] = NAN;
  CHECK(!api->Prepare(&line, &result) && result == BFA_kResultInvalidArgument);
  emotion.channels[0] = 0;
  emotion.version = 2;
  CHECK(!api->Prepare(&line, &result) && result == BFA_kResultInvalidArgument);
  line.emotion = nullptr;
  CHECK(Wait(8));

  // Failure events retire handles, no status records or release calls needed.
  line.utterance_id = ids[10];
  auto f = api->Prepare(&line, &result);
  RunTasks();
  auto failed = queue.front().entry->watchers.front();
  ProviderLineFailed(failed, BFA_kResultInferenceFailed);
  CHECK(Wait(9) && At(8).event == BFA_kFailed &&
        At(8).reason == BFA_kResultInferenceFailed);
  CHECK(api->Start(f, 0) == BFA_kResultInvalidHandle);
  CHECK(api->EndUtterance(ids[10], BFA_kAborted) == BFA_kResultOK);

  // Loading a world invalidates even tasks not yet admitted.
  line.utterance_id = ids[11];
  auto reset = api->Prepare(&line, &result);
  ResetPlayback(true);
  RunTasks();
  CHECK(Wait(10) && At(9).handle == reset && At(9).event == BFA_kEnded &&
        queue.empty());
  ResetPlayback(false);
  line.utterance_id = ids[12];
  auto expired = api->Prepare(&line, &result);
  bfa::testing::clock_offset = 121;
  CHECK(Wait(11) && At(10).handle == expired && At(10).event == BFA_kExpired);
  bfa::testing::clock_offset = 0;
  RunTasks();
  CHECK(queue.empty());
  // A playing line without a playback call for 120 s expires from the face.
  faces[fg] = Face();
  faces[fg].head_state = State::kReady;
  faces[fg].actor = actor;
  faces[fg].actor_handle = 42;
  Count();
  line.utterance_id = ids[16];
  auto idle = api->Prepare(&line, &result);
  RunTasks();
  Generate();
  CHECK(api->Start(idle, 0) == BFA_kResultPending);
  RunTasks();
  CHECK(Apply(fg, SnapshotTune(), false, actor, 42, a2f::kNoSound, -1) &
        kFaceChanged);
  CHECK(Wait(12) && At(11).event == BFA_kStarted && At(11).handle == idle);
  const auto idle_line = playing.at(fg).external;
  bfa::testing::clock_offset = 121;
  for (int i = 0; i < 600 && !idle_line->stopped; ++i) Sleep(5);
  CHECK(idle_line->expired &&
        Events() == 12);  // the renderer reports after its last write
  Settle();
  CHECK(playing.empty());
  CHECK(Wait(13) && At(12).handle == idle && At(12).event == BFA_kExpired &&
        At(12).reason == BFA_kResultExpired);
  bfa::testing::clock_offset = 0;  // its reply hold expired with it
  CHECK(api->EndUtterance(ids[16], BFA_kAborted) == BFA_kResultNotFound);
  RunTasks();
  Clear();
  line.utterance_id = ids[13];
  auto first = api->Prepare(&line, &result);
  RunTasks();
  auto firstEntry = queue.front().entry;
  line.utterance_id = ids[14];
  auto other = api->Prepare(&line, &result);
  RunTasks();
  CHECK(first && other && queue.size() == 2 && cache.size() == 2);
  CHECK(api->EndUtterance(ids[13], BFA_kAborted) == BFA_kResultOK);
  CHECK(queue.size() == 1 && cache.size() == 1 && firstEntry->cancelled);
  auto otherEntry = queue.front().entry;
  Generate();
  CHECK(api->Start(other, 0) == BFA_kResultPending);
  RunTasks();
  auto otherLine = playing.at(fg).external;
  faces[fg] = Face();
  faces[fg].head_state = State::kReady;
  faces[fg].actor = actor;
  faces[fg].actor_handle = 42;
  Count();
  // Shown for a while, then ended by its provider: the final interrupted
  // utterance fades to neutral and releases ownership for later writers.
  for (int i = 0; i < 5; ++i) {
    Apply(fg, SnapshotTune(), false, actor, 42, a2f::kNoSound, -1);
    Expressions(fg, 0.1f, SnapshotTune(), false, actor, 42);
  }
  CHECK(faces.at(fg).speech_blend > 0.5f);
  CHECK(Wait(15) && At(13).handle == first && At(13).event == BFA_kEnded &&
        At(14).handle == other && At(14).event == BFA_kStarted);
  Sleep(50);
  const size_t before = Events();
  CHECK(before == 15);
  CHECK(api->EndUtterance(ids[14], BFA_kAborted) == BFA_kResultPending);
  CHECK(!otherLine->audio && otherEntry->line && queue.empty() &&
        cache.empty());  // retained until the next face update
  for (int i = 0; i < 8 && !playing.empty(); ++i) {
    Apply(fg, SnapshotTune(), false, actor, 42, a2f::kNoSound, -1);
    Expressions(fg, 0.016f, SnapshotTune(), false, actor, 42);
  }
  CHECK(playing.empty());
  CHECK(faces.count(fg));
  CHECK(Events() == before);
  Settle();
  CHECK(!faces.count(fg));
  CHECK(Wait(before + 1) && At(before).handle == other &&
        At(before).event == BFA_kEnded && At(before).reason == BFA_kAborted);
  CHECK(faces.count(fg) == 0 || faces.at(fg).leaving.empty());

  // Start must watch replacement work after a prepared entry is evicted.
  {
    const size_t before = Events();
    line.utterance_id = api->NewUtteranceId();
    const auto handle = api->Prepare(&line, &result);
    RunTasks();
    Generate();
    ClearLines();
    CHECK(api->Start(handle, 0) == BFA_kResultPending);
    RunTasks();
    CHECK(queue.size() == 1 && queue.front().entry->watchers.size() == 1);
    {
      ScopedLock guard(lock);
      CancelJob(queue.front());
      queue.clear();
    }
    Apply(fg, SnapshotTune(), false, actor, 42, a2f::kNoSound, -1);
    CHECK(Wait(before + 1));
    CHECK(At(before).handle == handle && At(before).event == BFA_kFailed &&
          At(before).reason == BFA_kResultCancelled);
    CHECK(api->EndUtterance(line.utterance_id, BFA_kAborted) == BFA_kResultOK);
  }

  // Disabling an animated face must finish its fade before notifying callers.
  {
    const size_t before = Events();
    faces[fg] = Face();
    faces[fg].head_state = State::kReady;
    faces[fg].actor = actor;
    faces[fg].actor_handle = 42;
    Count();
    line.utterance_id = api->NewUtteranceId();
    const auto handle = api->Prepare(&line, &result);
    RunTasks();
    CHECK(api->Start(handle, 0) == BFA_kResultPending);
    RunTasks();
    CHECK(queue.front().entry->watchers.size() == 1);  // no duplicate watcher
    Generate();
    const auto session = playing.at(fg).external;
    for (int i = 0; i < 5; ++i) {
      Apply(fg, SnapshotTune(), false, actor, 42, a2f::kNoSound, -1);
      Expressions(fg, .1f, SnapshotTune(), false, actor, 42);
    }
    CHECK(Wait(before + 1) && At(before).event == BFA_kStarted);
    auto disabled = SnapshotTune();
    disabled.enabled = false;
    Apply(fg, disabled, false, actor, 42, a2f::kNoSound, -1);
    Expressions(fg, .016f, disabled, false, actor, 42);
    CHECK(faces.at(fg).speech_blend > 0 && !session->reported);
    CHECK(api->Stop(handle, BFA_kAborted) == BFA_kResultNoChange);
    for (int i = 0; i < 100; ++i)
      Expressions(fg, .016f, disabled, false, actor, 42);
    CHECK(Wait(before + 2));
    CHECK(At(before + 1).handle == handle &&
          At(before + 1).event == BFA_kDeclined &&
          At(before + 1).reason == BFA_kResultDisabled);
    CHECK(api->EndUtterance(line.utterance_id, BFA_kAborted) == BFA_kResultOK);
  }

  // Both serial orders and simultaneous Stop/completion preserve the first
  // cause and emit exactly one terminal event, only after renderer release.
  for (int attempt = 0; attempt < 66; ++attempt) {
    const size_t before = Events();
    line.utterance_id = api->NewUtteranceId();
    const auto handle = api->Prepare(&line, &result);
    RunTasks();
    auto session = queue.front().entry->watchers.front();
    session->on_face = true;  // renderer hand-back is controlled below
    BFA_Result stop;
    if (attempt == 0) {
      stop = api->Stop(handle, BFA_kSkipped);
      ProviderLineStopping(session, BFA_kCompleted);
      CHECK(stop == BFA_kResultPending);
    } else if (attempt == 1) {
      ProviderLineStopping(session, BFA_kCompleted);
      stop = api->Stop(handle, BFA_kSkipped);
      CHECK(stop == BFA_kResultNoChange);
    } else {
      StopRace race{api, session, CreateEventW(nullptr, TRUE, FALSE, nullptr)};
      CHECK(race.start);
      HANDLE threads[] = {
          CreateThread(nullptr, 0, StopFromProvider, &race, 0, nullptr),
          CreateThread(nullptr, 0, CompleteFromRenderer, &race, 0, nullptr)};
      CHECK(threads[0] && threads[1]);
      SetEvent(race.start);
      CHECK(WaitForMultipleObjects(2, threads, TRUE, 5000) == WAIT_OBJECT_0);
      for (HANDLE thread : threads) CloseHandle(thread);
      CloseHandle(race.start);
      stop = race.result;
    }
    CHECK(!session->reported && Events() == before);
    CHECK(stop == BFA_kResultPending || stop == BFA_kResultNoChange);
    ProviderLineLeft(session, BFA_kCompleted);
    ProviderLineLeft(session, BFA_kAborted);  // duplicate/late completion
    CHECK(Wait(before + 1));
    CHECK(Events() == before + 1 && At(before).handle == handle &&
          At(before).event == BFA_kEnded);
    CHECK(At(before).reason ==
          (stop == BFA_kResultPending ? BFA_kSkipped : BFA_kCompleted));
    CHECK(api->EndUtterance(line.utterance_id, BFA_kAborted) == BFA_kResultOK);
  }

  // World reset overrides a pending stop and retires it without a fade.
  {
    const size_t before = Events();
    line.utterance_id = api->NewUtteranceId();
    const auto handle = api->Prepare(&line, &result);
    RunTasks();
    Generate();
    CHECK(api->Start(handle, 0) == BFA_kResultPending);
    RunTasks();
    CHECK(api->Stop(handle, BFA_kSkipped) == BFA_kResultPending);
    ResetPlayback(true);
    CHECK(Wait(before + 1) && At(before).handle == handle &&
          At(before).event == BFA_kEnded && At(before).reason == BFA_kAborted);
    ResetPlayback(false);
    Clear();
  }

  // The new capacity bound rejects intake without leaking a handle or task.
  line.utterance_id = ids[15];
  for (int i = 0; i < 256; ++i) {
    line.segment = i;
    CHECK(api->Prepare(&line, &result));
  }
  CHECK(!api->Prepare(&line, &result) && result == BFA_kResultCapacity);
  CHECK(api->EndUtterance(ids[15], BFA_kAborted) == BFA_kResultOK);
  RunTasks();
  CHECK(queue.empty() && cache.empty());
  CHECK(api->UnregisterListener(listener) == BFA_kResultOK);
  CHECK(api->UnregisterListener(listener) == BFA_kResultNotFound);
  // Exercise both enqueue paths against an already queued, cancelled task.
  // Even a cancelled task must acquire calls before checking its stop flag.
  // Also end the reply while enqueue is blocked: late tasks must not revive it.
  for (int scenario = 0; scenario < 4; ++scenario) {
    const bool start = scenario % 2 != 0;
    const bool cancel = scenario >= 2;
    line.utterance_id = api->NewUtteranceId();
    auto target = api->Prepare(&line, &result);
    CHECK(target && result == BFA_kResultPending);
    auto cancelled = api->Prepare(&line, &result);
    CHECK(cancelled && api->Stop(cancelled, BFA_kAborted) == BFA_kResultOK);
    cancel_on_dispatch = cancel ? line.utterance_id : 0;
    dispatch_probe = true;
    if (start) {
      CHECK(api->Start(target, 0) == BFA_kResultPending);
    } else {
      CHECK(api->Prepare(&line, &result) && result == BFA_kResultPending);
    }
    CHECK(!dispatch_probe && dispatch_done);
    CHECK(api->EndUtterance(line.utterance_id, BFA_kAborted) ==
          (cancel ? BFA_kResultNotFound : BFA_kResultOK));
    RunTasks();
    CHECK(queue.empty() && cache.empty());
    CHECK(api->Start(target, 0) == BFA_kResultInvalidHandle);
  }
  puts(
      "provider: event lifecycle, playback clock, reply purge, cancellation, "
      "emotion, reset and expiry passed");
}
