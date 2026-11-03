#include <array>
#include <cstdlib>

#include "../src/plugin/better_facial_animations.cc"
#include "runtime_fixture.h"
#define CHECK(x)                                      \
  do {                                                \
    if (!(x)) {                                       \
      fprintf(stderr, "line %d: %s\n", __LINE__, #x); \
      std::exit(1);                                   \
    }                                                 \
  } while (0)
namespace {
std::array<uint8_t, 0x100> manager{};
uint8_t* mtm = manager.data();
std::array<uint8_t, 0x400> player_actor{}, player_base{}, player_race{},
    player_fg{};
uint8_t* player = player_actor.data();
uint8_t* PlayerFace(uint8_t*) { return player_fg.data(); }
void CreateHandle(uint32_t& handle, uint8_t* actor) {
  handle = actor == player ? 7 : 43;
}
bool LookupReference(const uint32_t& handle, uint8_t** ref) {
  if (handle != 7) return false;
  *ref = player;
  InterlockedIncrement(&bfa::Field<LONG>(player, 0x28));
  return true;
}
void ResourceId(skse::ResId* id, const char*) { *id = {}; }
float NativeReturn(uint8_t*, const char*, void*, uint64_t, uint64_t, uint64_t,
                   uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
                   uint64_t, uint64_t) {
  return 3.5f;
}
int runs = 0, destroyed = 0;
struct FakeNet : a2f::Net {
  ~FakeNet() override { ++destroyed; }
  int Batch() const override { return 2; }
  bool Run(const float*, const float*, int b, float* out,
           std::string&) override {
    std::fill_n(out, b * a2f::kNumOut, 0.f);
    ++runs;
    bfa::Field<uint32_t>(mtm, 0x68) = 0;
    return true;
  }
  bool Run3(const float*, const float*, const float*, const float*, float* skin,
            float* tongue, float* jaw, float* state, std::string&) override {
    std::fill_n(skin, a2f::kFrames3 * a2f::kLatent3, 0.f);
    std::fill_n(tongue, a2f::kFrames3 * a2f::kLatent3, 0.f);
    std::fill_n(jaw, a2f::kFrames3 * a2f::kJawEyes3, 0.f);
    std::fill_n(state, a2f::kState3, 0.f);
    ++runs;
    bfa::Field<uint32_t>(mtm, 0x68) = 0;
    return true;
  }
};
}  // namespace
extern "C" uint64_t TestAddress(uint64_t id) {
  if (id == kIdMenuTopicManager.Get()) return reinterpret_cast<uint64_t>(&mtm);
  if (id == kIdLookupReference.Get())
    return reinterpret_cast<uint64_t>(&LookupReference);
  if (id == kIdPlayer.Get()) return reinterpret_cast<uint64_t>(&player);
  if (id == kIdCreateHandle.Get())
    return reinterpret_cast<uint64_t>(&CreateHandle);
  if (id == kIdGenerateFromPath.Get())
    return reinterpret_cast<uint64_t>(&ResourceId);
  return 0;
}
int main() {
  bfa::testing::address = TestAddress;
  using namespace bfa;
  skse::SetRuntime(skse::Version(1, 6, 1170));
  Tune t;
  CHECK(!t.potato_mode);
  CHECK(t.player);
  for (const Param& p : kParams) CHECK(strcmp(p.key, "NativePlayerSpeech"));
  t.potato_mode = true;
  for (size_t i = 0; i < kNumParams; ++i)
    if (!strcmp(kParams[i].key, "PotatoMode")) {
      ScopedLock guard(settings_lock);
      CHECK(!*kParams[i].flag);
      *kParams[i].flag = true;
      CHECK(!strcmp(kParams[i].label, "Potato Mode"));
      CHECK(!strcmp(kParams[i].tip,
                    "Only animate the player and NPC in an active dialogue "
                    "window with one another."));
    }
  CHECK(!DialogueAllows(t, 42));
  // lastSpeaker persists after close but must never admit work.
  Field<uint32_t>(mtm, 0x68 + 4) = 42;
  CHECK(!DialogueAllows(t, 42));
  CHECK(!Request("ambient", 0, 0, false, true, t, 42, world_epoch));
  CHECK(queue.empty());
  Field<uint32_t>(mtm, 0x68) = 42;
  CHECK(DialogueAllows(t, 42) && !DialogueAllows(t, 43));
  auto e = Request("line", 0, 0, false, false, t, 42, world_epoch);
  CHECK(e && queue.size() == 1 && JobAllowed(queue.front()));
  Job old = queue.front();
  queue.pop_front();  // also cover a cancelled line already in flight
  Field<uint32_t>(mtm, 0x68) = 43;
  CHECK(!JobAllowed(old));
  auto next = Request("line", 0, 0, false, true, t, 43, world_epoch);
  CHECK(next && next != e && e->state == State::kFailed);
  CancelJob(old);  // cannot erase a newer request of the same line
  CHECK(cache.at(old.cache_key) == next);
  // Global Disabled still leaves the speaker-scope policy unchanged.
  t.enabled = false;
  CHECK(DialogueAllows(t, 43) && !EnabledFor(t, false));
  t.enabled = true;
  t.potato_mode = false;
  CHECK(DialogueAllows(t, 0));
  t.potato_mode = true;
  ResetPlayback(true);
  CHECK(!JobAllowed(queue.front()));
  ResetPlayback(false);
  queue.clear();
  cache.clear();
  // Cached content may be shared, but cancellation belongs to every requester.
  // Exercise either arrival order, queued and already-in-flight work, and a
  // mode change after two NPCs requested the same audio.
  for (bool player_first : {false, true}) {
    for (bool in_flight : {false, true}) {
      Field<uint32_t>(mtm, 0x68) = 42;
      const Tune shared = SnapshotTune();
      auto first = Request("shared_voice", 0, 0, false, true, shared,
                           player_first ? 7 : 42, world_epoch, player_first);
      Job shared_job = queue.front();
      if (in_flight) queue.pop_front();
      auto second = Request("shared_voice", 0, 0, false, true, shared,
                            player_first ? 42 : 7, world_epoch, !player_first);
      CHECK(first && first == second);
      CHECK(queue.size() == (in_flight ? 0 : 1));
      Field<uint32_t>(mtm, 0x68) = 0;
      CHECK(JobAllowed(shared_job));  // player still needs this pending line
      for (const auto& p : kParams)
        if (!strcmp(p.key, "PlayerFacialAnimations")) *p.flag = false;
      // The switches pause playback (A/B); they do not cancel shared work.
      CHECK(JobAllowed(shared_job));
      for (const auto& p : kParams)
        if (!strcmp(p.key, "PlayerFacialAnimations")) *p.flag = true;
      ResetPlayback(true);  // a new world: nobody remains eligible
      CHECK(!JobAllowed(shared_job));
      CancelJob(shared_job);
      CHECK(first->state == State::kFailed && cache.empty());
      ResetPlayback(false);
      Field<uint32_t>(mtm, 0x68) = 0;
      CHECK(!JobAllowed(shared_job));  // cancellation cannot revive old work
      auto retry = Request("shared_voice", 0, 0, false, true, SnapshotTune(), 7,
                           world_epoch, true);
      CHECK(retry && retry != first);
      ResetPlayback(true);
      CHECK(!JobAllowed(queue.back()));
      ResetPlayback(false);
      queue.clear();
      cache.clear();
    }
  }
  for (const auto& p : kParams)
    if (!strcmp(p.key, "PotatoMode")) *p.flag = false;
  auto npc_first = Request("shared_npc", 0, 0, false, false, SnapshotTune(), 42,
                           world_epoch);
  auto npc_second =
      Request("shared_npc", 0, 0, false, true, SnapshotTune(), 43, world_epoch);
  CHECK(npc_first && npc_first == npc_second && queue.size() == 1);
  for (const auto& p : kParams)
    if (!strcmp(p.key, "PotatoMode")) *p.flag = true;
  Field<uint32_t>(mtm, 0x68) = 43;
  CHECK(JobAllowed(queue.front()));  // mode switch keeps the current partner
  Field<uint32_t>(mtm, 0x68) = 44;
  CHECK(!JobAllowed(queue.front()));
  queue.clear();
  cache.clear();
  // No model creation when the request was withdrawn; release is idempotent.
  int creates = 0;
  a2f::NetKeeper keeper(
      true,
      [&](bool, std::string&, std::string&) {
        ++creates;
        return std::make_unique<FakeNet>();
      },
      [] { return 0.; });
  std::string err;
  bool no_net = false;
  CHECK(!keeper.Run([](a2f::Net&, std::string&) { return true; }, err, no_net,
                    [] { return false; }));
  CHECK(creates == 0);
  CHECK(keeper.Ensure() && creates == 1);
  keeper.Unload();
  keeper.Unload();
  CHECK(destroyed == 1 && !keeper.loaded() && !keeper.on_gpu());
  // A long line ends after its first submitted run, on both networks.
  FakeNet net;
  std::vector<float> pcm(a2f::kSampleRate * 4, 0.f), rest(a2f::kNumOut3, 0.f);
  std::vector<float> proj(a2f::kNumPca, 0.f),
      tongue(a2f::kNumTongue * a2f::kLatent3, 0.f);
  a2f::Line line;
  auto allowed = [&] { return DialogueAllows(t, 42); };
  Field<uint32_t>(mtm, 0x68) = 42;
  runs = 0;
  CHECK(!a2f::Generate(net, pcm.data(), pcm.size(), {}, proj.data(),
                       rest.data(), 1, line, err, allowed));
  CHECK(runs == 1 && err == "generation cancelled");
  proj.resize(a2f::kLatent3);
  Field<uint32_t>(mtm, 0x68) = 42;
  runs = 0;
  CHECK(!a2f::Generate3(net, pcm.data(), pcm.size(), {},
                        {2, proj.data(), tongue.data(), rest.data(), 1}, line,
                        err, allowed));
  CHECK(runs == 1);
  // Unowned ambient faces must not be dirtied by the rejection gate.
  std::array<uint8_t, 0x400> idle_fg{}, idle_node{};
  Field<uint8_t*>(idle_node.data(), (skse::IsVr() ? 0x178 : 0x150)) =
      idle_fg.data();
  AddDirect(idle_node.data(), true);
  CHECK(!idle_fg[0x218]);
  CHECK(!FaceNeedsUpdate(idle_fg.data()));
  // Direct-only handoff restores the engine's expressions with no extra pose.
  std::array<uint8_t, 0x400> fg{}, node{};
  float values[direct::kNumExpressions]{};
  auto& ex = FixtureKeyframe(fg.data(), 0xC0);
  ex.values = values;
  ex.count = direct::kNumExpressions;
  Face f;
  f.actor_handle = 42;
  f.until = 999;
  std::fill_n(f.src, direct::kNumExpressions, 0.4f);
  faces[fg.data()] = f;
  Count();
  // Retained cleanup is admitted, but an unrelated NPC is still skipped.
  CHECK(FaceNeedsUpdate(fg.data()));
  CHECK(!FaceNeedsUpdate(idle_fg.data()));
  Field<uint8_t*>(node.data(), (skse::IsVr() ? 0x178 : 0x150)) = fg.data();
  AddDirect(node.data(), true);
  CHECK(faces.empty() && fg[0x218]);
  CHECK(!FaceNeedsUpdate(fg.data()));
  for (int i = 0; i < direct::kNumExpressions; ++i)
    CHECK(values[i] == (direct::PlaysExpression(i) ? 0.4f : 0.f));
  faces[fg.data()] = f;
  Count();
  CHECK(Expressions(fg.data(), 0, t, false, nullptr, 42) & kFaceChanged);
  CHECK(faces.empty());
  queue.clear();
  cache.clear();
  auto flag = [](const char* key, bool value) {
    ScopedLock guard(settings_lock);
    for (const Param& p : kParams)
      if (p.flag && !strcmp(p.key, key)) *p.flag = value;
  };
  flag("PotatoMode", true);
  flag("Enabled", true);
  Field<uint32_t>(mtm, 0x68) = 0;
  CHECK(WorkerStep(keeper, false) == INFINITE && keeper.loaded());
  const int resident_creates = creates, resident_destroyed = destroyed;
  CHECK(keeper.Ensure());
  CHECK(WorkerStep(keeper, false) == INFINITE && keeper.loaded());
  CHECK(creates == resident_creates && destroyed == resident_destroyed);
  flag("PotatoMode", false);
  CHECK(WorkerStep(keeper, false) == INFINITE && keeper.loaded());
  flag("Enabled", false);
  CHECK(WorkerStep(keeper, false) == INFINITE);
  const auto revision = settings_revision;
  WakeWorker();
  CHECK(settings_revision == revision + 1);
  flag("Enabled", true);
  a2f::NetKeeper failed(
      false,
      [](bool, std::string&, std::string&) -> std::unique_ptr<a2f::Net> {
        return nullptr;
      },
      [] { return 0.; });
  CHECK(WorkerStep(failed, false) == 1000);
  // A settings change while creating a session must not be lost to an
  // indefinite wait based on the old snapshot.
  a2f::NetKeeper changed(
      false,
      [&](bool, std::string&, std::string&) -> std::unique_ptr<a2f::Net> {
        flag("PotatoMode", true);
        WakeWorker();
        return std::make_unique<FakeNet>();
      },
      [] { return 0.; });
  CHECK(WorkerStep(changed, false) == 0);
  CHECK(WorkerStep(changed, false) == INFINITE && changed.loaded());
  {
    double time = 0;
    bool recovered = false;
    int attempts = 0;
    a2f::NetKeeper retrying(
        true,
        [&](bool gpu, std::string& error,
            std::string&) -> std::unique_ptr<a2f::Net> {
          if (gpu && !recovered) {
            ++attempts;
            error = "shared failure";
            return nullptr;
          }
          return std::make_unique<FakeNet>();
        },
        [&] { return time; }, nullptr, true);
    CHECK(WorkerStep(retrying, false) == 1000 && !retrying.loaded() &&
          !retrying.on_gpu());
    time = 1;
    CHECK(WorkerStep(retrying, false) == 1000 && attempts == 1);
    time = 2;
    recovered = true;
    CHECK(WorkerStep(retrying, false) == INFINITE && retrying.on_gpu());
  }
  // An independent runtime-API failure waits for user input; no automatic
  // CPU or polling.
  int gpu_creates = 0, cpu_creates = 0;
  double clock = 0;
  a2f::NetKeeper fallback(
      true,
      [&](bool gpu, std::string& error,
          std::string&) -> std::unique_ptr<a2f::Net> {
        if (gpu) {
          ++gpu_creates;
          error = "independent device: runtime API unavailable";
          return nullptr;
        }
        ++cpu_creates;
        return std::make_unique<FakeNet>();
      },
      [&] { return clock; }, nullptr, true);
  flag("PotatoMode", false);
  CHECK(WorkerStep(fallback, false) == INFINITE);
  CHECK(!fallback.Wanted() && !fallback.loaded() && !fallback.on_gpu());
  CHECK(gpu_creates == 1 && cpu_creates == 0 && !stats.net_gpu &&
        stats.net_retry_at == 0);
  for (int conversation = 0; conversation < 3; ++conversation) {
    flag("PotatoMode", true);
    Field<uint32_t>(mtm, 0x68) = 0;
    // Changing conversation state cannot re-arm GPU initialization.
    CHECK(WorkerStep(fallback, false) == INFINITE && !fallback.loaded());
    CHECK(gpu_creates == 1 && fallback.error() == "independent device: runtime API unavailable" &&
          stats.net_detail == fallback.error());
    // A new admitted line cannot select CPU implicitly.
    Field<uint32_t>(mtm, 0x68) = 42;
    clock += 100000;
    CHECK(!fallback.Run([](a2f::Net&, std::string&) { return true; }, err,
                        no_net,
                        [&] { return DialogueAllows(SnapshotTune(), 42); }));
    CHECK(gpu_creates == 1 && cpu_creates == 0);
    flag("PotatoMode", false);
    CHECK(WorkerStep(fallback, false) == INFINITE);
  }
  // Both real player entry paths must reach generation without a dialogue
  // partner. NPCs remain restricted and player/master settings still apply.
  flag("PotatoMode", true);
  flag("PlayerFacialAnimations", true);
  Field<uint32_t>(mtm, 0x68) = 0;
  void* player_vtable[0x100]{};
  player_vtable[0x63] = reinterpret_cast<void*>(&PlayerFace);
  Field<void**>(player, 0) = player_vtable;
  player[0x1A] = kFtActor;
  Field<uint32_t>(player, 0x14) = kPlayerFormId;
  Field<LONG>(player, 0x28) = 1;
  Field<uint8_t*>(player, 0x40) = player_base.data();
  player_base[0x1A] = kFtNpc;
  Field<uint8_t*>(
      player, (skse::Runtime() >= skse::Version(1, 6, 629) ? 0x1f8 : 0x1f0)) =
      player_race.data();
  Field<uint32_t>(player_race.data(), 0x108) = 2;
  ready = true;
  native_speech_original = NativeReturn;
  auto native = [&](const char* path = "Voice\\dbvo\\player.wav") {
    return NativeSpeakHook(player, path, nullptr, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0);
  };
  for (bool dbrev : {false, true}) {
    queue.clear();
    cache.clear();
    playing.clear();
    faces.clear();
    Count();
    auto speech = std::make_shared<PlayerSpeech>();
    speech->path = "Voice/dbrev/player.wav";
    speech->duration = 3;
    speech->started_at = Now();
    auto start = [&] {
      if (dbrev)
        StartPlayerSpeech(speech);
      else
        CHECK(native() == 3.5f);
    };
    start();
    CHECK(queue.size() == 1);
    CHECK(JobAllowed(queue.front()));
    CHECK(playing.at(player_fg.data()).player);
    CHECK(bool(playing.at(player_fg.data()).external) == dbrev);
    // If both routes observe the same playback, keep one job and retain
    // the provider's cancellation whichever notification arrives first.
    if (dbrev) {
      CHECK(native("Voice\\dbrev\\player.wav") == 3.5f);
    } else {
      speech->path = "Voice/dbvo/player.wav";
      StartPlayerSpeech(speech);
    }
    CHECK(queue.size() == 1 && playing.size() == 1);
    CHECK(playing.at(player_fg.data()).external == speech);
    Field<uint32_t>(mtm, 0x68) = 42;
    CHECK(JobAllowed(queue.front()));
    CHECK(!Request("ambient", 0, 0, false, true, SnapshotTune(), 43,
                   world_epoch));
    Field<uint32_t>(mtm, 0x68) = 0;
    Apply(player_fg.data(), SnapshotTune(), false, player, 7, a2f::kNoSound,
          -1);
    CHECK(playing.count(player_fg.data()) == 1);
    // A morph update before expression ownership must not drop player speech.
    std::array<uint8_t, 0x400> player_node{};
    Field<uint8_t*>(player_node.data(), (skse::IsVr() ? 0x178 : 0x150)) =
        player_fg.data();
    AddDirect(player_node.data(), true);
    CHECK(playing.count(player_fg.data()) == 1);
    CHECK(Field<LONG>(player, 0x28) == 1);
    // The expression update must retain the player's pending speech.
    Expressions(player_fg.data(), .016f, SnapshotTune(), false, player, 7);
    CHECK(playing.count(player_fg.data()) == 1);
    // The player-only switch keeps shared work; global Disabled cancels it.
    flag("PlayerFacialAnimations", false);
    CHECK(JobAllowed(queue.front()));
    playing.clear();
    start();
    CHECK(playing.empty());
    flag("PlayerFacialAnimations", true);
    flag("Enabled", false);
    CHECK(!JobAllowed(queue.front()));  // Disabled cancels inference work.
    start();
    CHECK(playing.empty());
    flag("Enabled", true);
  }
  queue.clear();
  cache.clear();
  faces.clear();
  Count();
  flag("PlayerFacialAnimations", false);
  CHECK(native() == 3.5f && queue.empty());
  // CPU creation retries also run while Potato mode is idle.
  CHECK(WorkerStep(failed, false) == 1000);
  // Cancellation after GPU loss never creates an unrequested CPU session.
  gpu_creates = cpu_creates = 0;
  a2f::NetKeeper lost(
      true,
      [&](bool gpu, std::string&, std::string&) -> std::unique_ptr<a2f::Net> {
        gpu ? ++gpu_creates : ++cpu_creates;
        return std::make_unique<FakeNet>();
      },
      [&] { return clock; });
  bool admitted = true;
  CHECK(!lost.Run(
      [&](a2f::Net& network, std::string& error) {
        network.failed = true;
        error = "GPU run failure";
        admitted = false;
        return false;
      },
      err, no_net, [&] { return admitted; }));
  CHECK(!lost.loaded() && gpu_creates == 1 && cpu_creates == 0);
  CHECK(!lost.Ensure() && !lost.on_gpu() && gpu_creates == 1 &&
        cpu_creates == 0);
  puts("potato mode, explicit providers and worker lifecycle contracts passed");
}
