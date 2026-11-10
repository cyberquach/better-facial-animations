#include "a2f/inference/diagnostics.h"
// Better Facial Animations: Audio2Face lip-sync generated in the game for every
// voiced line of every NPC with a FaceGen head. The speak / response hooks
// queue a line; a worker thread reads the voice file through the engine's
// resource system, decodes it and runs the network (live_core.cc); the face
// hook plays the line and the morph hook adds its A2F Direct vertex deltas
// (direct.cc) to the head's face meshes, through the head's mapping
// (head_cache.cc). Settings: settings.cc and menu.cc.
#include <windows.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "a2f/animation/direct.h"
#include "a2f/animation/gaze.h"
#include "a2f/animation/head_cache.h"
#include "a2f/animation/pose.h"
#include "a2f/inference/live_core.h"
#include "a2f/inference/net_keeper.h"
#include "a2f/inference/ort_net.h"
#include "a2f/plugin/menu.h"
#include "a2f/plugin/plugin.h"
#include "a2f/runtime/call_sites.h"
#include "a2f/runtime/engine.h"
#include "a2f/runtime/face_update.h"
#include "a2f/runtime/ids.h"
#include "a2f/runtime/scoped_lock.h"
#include "a2f/runtime/skse.h"
#include "a2f/speech/BFA_API.h"
#include "a2f/speech/native_speech_hook.h"
#include "a2f/speech/player_speech.h"
#include "a2f/speech/provider.h"
#include "a2f/speech/voice_sync.h"

// From VERSION, with a feature-build suffix for releases, via build.sh.
#ifndef BFA_VERSION
#error "build with build.sh, which defines BFA_VERSION from VERSION"
#endif

namespace {

// "major.minor.patch[.revision]" -> SKSE's packed plugin version.
constexpr uint32_t PackVersion(const char* s) {
  uint32_t part[4] = {};
  for (int i = 0; *s && i < 4; ++s) {
    if (*s == '.') {
      ++i;
    } else {
      part[i] = part[i] * 10 + (*s - '0');
    }
  }
  if (part[0] > 255 || part[1] > 255 || part[2] > 4095 || part[3] > 15)
    return 0;
  return skse::Version(part[0], part[1], part[2]) | part[3];
}

static_assert(PackVersion("3.0.1") == 0x03000010);
static_assert(PackVersion("3.0.1.1") == 0x03000011);
static_assert(PackVersion("3.2.0.15") == 0x0302000F);
static_assert(PackVersion("3.2.0.16") == 0);
static_assert(PackVersion(BFA_VERSION) != 0,
              "version cannot be encoded by SKSE");

}  // namespace

extern "C"
    __declspec(dllexport) skse::SKSEPluginVersionData SKSEPlugin_Version = {
        1,
        PackVersion(BFA_VERSION),
        "BetterFacialAnimations",
        "Cyberquach",
        "",
        0,
        0,
        {skse::Version(1, 6, 353), skse::Version(1, 6, 640),
         skse::Version(1, 6, 659, skse::kGog), skse::Version(1, 6, 1170),
         skse::Version(1, 6, 1179, skse::kGog), skse::Version(1, 7, 99),
         skse::Version(1, 7, 104), 0},
        0};

namespace bfa {

using skse::Addr;
using skse::Log;

extern const char kVersion[] = BFA_VERSION;
std::atomic<bool> net_loaded{false};
std::atomic<bool> ready{false};
direct::Model& model = *new direct::Model;
namespace {
auto& active_model = *new std::shared_ptr<const ModelState>;
}
std::shared_ptr<const ModelState> SnapshotModel() {
  if (auto state = std::atomic_load(&active_model)) return state;
  // Before worker startup (also used by standalone contract fixtures).
  return std::make_shared<const ModelState>(ModelState{
      config.network, config.gpu, config.model,
      std::shared_ptr<const direct::Model>(&model, [](const direct::Model*) {}),
      config.gpu_adapter});
}

namespace {

// A line ready later than this after its start is blended in.
constexpr float kLateStart = 0.1f;
// An unmapped head's expressions play on the engine's keyframes, which have
// no regions: at the mean of the regions' scales.
constexpr float MeanExpressionScale() {
  float m = 0;
  for (float x : direct::kExpressionScale) m += x;
  return m / direct::kNumSkinRegions;
}

// ---------------- generated lines ----------------
struct Entry {
  bool cancelled = false;  // EndUtterance discarded this provider entry
  uint32_t failure_reason = BFA_kResultAudioFailed;
  State state = State::kPending;
  std::shared_ptr<const a2f::Line> line;
  uint64_t last_use = 0;
  // Generated without voice sentiment while that was unavailable: generated
  // again once the worker would try sentiment again (sentiment_retry_at).
  bool degraded = false;
  bool performance_sentiment = false;
  struct Requester {
    uint32_t handle;
    bool player;
  };
  // Pending content is reusable, but each requester has its own eligibility.
  // Accessed under lock, including from inference cancellation callbacks.
  std::vector<Requester> requesters;
  uint64_t epoch = 0;
  std::shared_ptr<const ModelState> model_state;
  // provider lines (BFA_API.h) told when it is generated or fails
  std::vector<PlayerSpeechPtr> watchers;
};
struct Job {
  std::string cache_key, path;
  // the emotion and its settings when it was requested
  a2f::Emotion emotion;
  // the speaker's sex: with v3.0 the identity that speaks it
  bool female = false;
  std::shared_ptr<Entry> entry;
  // a provider's audio bytes (path is then its name); null: read the file
  std::shared_ptr<const std::vector<uint8_t>> audio;
};
enum FaceUpdate { kNoUpdate = 0, kFaceChanged = 1, kKeepUpdating = 2 };
constexpr int kDirectTail = 3;  // forced resets after the last Direct frame
struct Playing {
  std::shared_ptr<Entry> entry;
  std::string name;
  uint64_t lip_key = 0;
  uint8_t* actor = nullptr;
  // the actor's handle (HandleOf), which tells when it is gone: the actor is
  // only used while a ScopedReference of the handle holds it
  uint32_t actor_handle = 0;
  bool female = false;  // the actor's sex: the face its head is mapped onto
  double seen = 0;      // last update (play_clock)
  bool player = false;
  PlayerSpeechPtr external;
  a2f::VoiceStart voice;
  a2f::LipClock lip_clock;
  a2f::SpeechClock speech_clock;
  float ready_at = -1;
  bool lip_seen = false, logged = false;
  // Direct: the frame (fractional, -1 none) for the morph hook, and whether
  // the line was ready late (it is caught up with over BlendIn)
  float direct_frame = -1;
  float tongue_phonemes[direct::kNumPhonemes] = {};
  bool late = false;
  // Enabled was off during the line: its next frame is shown at once
  bool snap = false;
  int direct_tail = 0;
  // log: last clock, Direct frames played / deformed, peak delta per part
  float last_t = 0;
  uint32_t direct_frames = 0, direct_adds = 0;
  float peak[3] = {};  // by direct::Part
  // log, while the line's frames play: the engine's largest expression
  // (index, value) and the most of one the plugin showed after the fade
  int engine_expr = -1;
  float engine_expr_v = 0, shown = 0;
};
// A provider interface line (BFA_API.h): it owns the face while it plays.
bool ProviderLine(const Playing& p) {
  return p.external && p.external->handle != 0;
}
// Final phonemes are cached by the engine when the source LIP sample is
// unchanged. Restore our last suppression before that update, including on
// line end/disable, so neither tongue capture nor vanilla reads our zeros.
struct PhonemeRestore {
  float* values = nullptr;
  uint32_t count = 0;
  float source[direct::kNumPhonemes] = {};
  float written[direct::kNumPhonemes] = {};

  void Capture(skse::KeyframeMultiple& phon, float mix, float* tongue) {
    *this = {};
    std::fill_n(tongue, direct::kNumPhonemes, 0.f);
    if (!phon.values || phon.count < direct::kNumPhonemes) return;
    values = phon.values;
    count = phon.count;
    for (int i = 0; i < direct::kNumPhonemes; ++i) {
      source[i] = values[i];
      tongue[i] = source[i] * mix;
      written[i] = mix >= 1.f ? 0.f : source[i] + (0.f - source[i]) * mix;
      values[i] = written[i];
    }
    phon.isUpdated = true;
  }

  bool Restore(skse::KeyframeMultiple& phon) {
    // Never restore into replaced storage or overwrite another writer's pose.
    const bool ours =
        values && phon.values == values && phon.count == count &&
        std::equal(std::begin(written), std::end(written), phon.values);
    values = nullptr;  // consume even when the engine/another mod took over
    if (!ours) return false;
    std::copy_n(source, direct::kNumPhonemes, phon.values);
    phon.isUpdated = true;
    return true;
  }
};
// A face the plugin animates (Expressions, AddDirect): anyone speaking a
// line and the player's conversation partner, for a while after
// (kFaceGrace). It holds the head's mapping onto Mark.
struct Face {
  PhonemeRestore phonemes;
  // the engine's own expression keyframe, put back before its update (so
  // the plugin's writes are never read back as the engine's), and what the
  // plugin wrote in its place
  float src[direct::kNumExpressions] = {};
  // src as shown: following it smoothly (kSrcFollow), so a script's or a
  // provider's one-frame rewrite of the engine's expressions (an override)
  // never snaps the face
  float shown[direct::kNumExpressions] = {};
  bool shown_set = false;  // shown starts at src
  float written[direct::kNumExpressions] = {};
  float level[direct::kNumExpressions] = {};  // added to the meshes
  // 0..1: from the engine's expressions to the plugin's.
  float ownership_blend = 0;
  float speech_blend = 0;  // 0..1: the expressions faded out for a line
  double until = 0;  // had until (kFaceGrace; then fades back to the engine's)
  double seen = 0;   // last update
  // Not updated for kFaceStale (unloaded, or out of sight) with the plugin's
  // values still in the keyframes: keep their restoration state to put the
  // engine's back when the face is next updated.
  bool parked = false;
  // failed: not mappable (vanilla lip-sync)
  State head_state = State::kPending;
  std::shared_ptr<const direct::Head> head;
  std::string head_key;  // geometry, morph identity and sex used by head
  // whose face it is and its handle, to tell when the actor joins a scene
  // (FaceActor); null: not known. The pointer is only used while a
  // ScopedReference of the handle holds it.
  uint8_t* actor = nullptr;
  uint32_t actor_handle = 0;
  // its sex, known from its line or as the conversation partner
  bool female = false;
  bool player = false;
  a2f::PoseTransition transition;
  double neutral_started = -1;
  float neutral_duration = 0;
  float neutral_levels[direct::kNumExpressions] = {};
  std::shared_ptr<const direct::Model> pose_model;
  int pose_face = 0;
  // Provider lines that left the face, told (their terminal event) once it
  // is handed back: their utterance no longer holds it and the engine's
  // expressions have faded back in, or another line took it. The provider's
  // own writes then never fight the fade.
  std::vector<PlayerSpeechPtr> leaving;
};
// Times on play_clock:
// s a face is kept after its line or conversation before it is handed back
// (kOwnOut): over the gaps between a speaker's lines -- the engine's pause
// between the responses of one dialogue line, DBReV's post-line delay
// (PlayerLineStart::totalSeconds) and the next line's inference -- so the
// face carries its expressions from one line into the next without dipping
// to the engine's, but not for long after the last.
constexpr double kFaceGrace = 3;
constexpr double kFaceParked = 300;  // s a parked face keeps its restoration
constexpr double kFaceStale = 30;    // s without an update: unloaded
// s without an update: the line's head is gone (unloaded or rebuilt, as on
// leaving the race menu) and the line is dropped
constexpr double kPlayingStale = 30;
constexpr double kPlayStep = 0.25;  // the most play_clock advances at once
// s for the plugin to take a face's expressions over from the engine and to
// give them back
constexpr float kOwnIn = 0.25f, kOwnOut = 1;
// Maximum seconds for a full-scale (0..1) native expression change.
constexpr float kSrcFollow = 0.3f;
constexpr size_t kMaxFaces = 8;

SRWLOCK lock = SRWLOCK_INIT;  // guards everything below
CONDITION_VARIABLE queue_cv = CONDITION_VARIABLE_INIT;
std::atomic<bool> gpu_retry_requested{false};
std::atomic<bool> gpu_adapters_requested{false};
a2f::GpuAdapterList gpu_adapters;  // protected by lock
uint64_t settings_revision = 0;    // lock held; prevents a lost settings wakeup
auto& cache = *new std::unordered_map<std::string, std::shared_ptr<Entry>>;
auto& queue = *new std::deque<Job>;
// by FaceGen animation data
auto& playing = *new std::unordered_map<uint8_t*, Playing>;
auto& faces = *new std::unordered_map<uint8_t*, Face>;
// playing.size() and faces.size(), so the per-frame hooks can skip the lock
std::atomic<int> num_playing{0}, num_faces{0};
// the conversation partner (TESObjectREFR*) whose face partner_fg is, found
// by the morph hook (TrackPartner); written together under the lock
std::atomic<uint8_t*> partner_ref{nullptr}, partner_fg{nullptr};
std::atomic<uint32_t> partner_handle{0};
std::atomic<bool> partner_female{false};
// The time face updates have been running: the wall clock without the gaps
// between them (a paused game, a menu), so a pause does not make a line or
// a face look abandoned.
double play_clock, play_wall;
uint64_t use_clock;
// When the worker next tries the sentiment model after a failure (Now()); 0:
// no failure pending. Worker-written, read by Request.
std::atomic<double> sentiment_retry_at{0};
// The world the game is in, counted up whenever it is replaced (a save
// loaded, a new game), and whether a load is under way. A hook notes the
// epoch when it starts and gives up once CurrentWorld says it has passed:
// the actors, faces and keyframes it knew are the old world's.
std::atomic<uint64_t> world_epoch{0};
std::atomic<bool> loading_game{false};
Stats& stats = *new Stats;

// Lock held, after playing or faces changed.
void Count() {
  num_playing = static_cast<int>(playing.size());
  num_faces = static_cast<int>(faces.size());
}

// The world is replaced (loading: a load begins, else it is over or a new
// game starts): forgets the lines and faces without touching the old world's
// (nothing is restored). Cached audio and head mappings are copies and stay.
void ResetPlayback(bool loading) {
  ScopedLock guard(lock);
  loading_game = loading;
  ++world_epoch;
  playing.clear();
  ProviderReset();
  faces.clear();
  partner_fg = nullptr;
  partner_ref = nullptr;
  partner_handle = 0;
  partner_female = false;
  Count();
  // play_clock carries on (the sweeps go by it), without the load's time
  play_wall = Now();
}

bool CurrentWorld(uint64_t epoch) {
  return !loading_game && world_epoch == epoch;
}

// Lock held: advances play_clock to now and returns it.
double Tick(double now) {
  play_clock += std::clamp(now - play_wall, 0., kPlayStep);
  play_wall = now;
  return play_clock;
}

// Lock held: the face of this animation data, unless none or parked.
Face* LiveFace(uint8_t* fg) {
  auto it = faces.find(fg);
  return it == faces.end() || it->second.parked ? nullptr : &it->second;
}

// Lock held: tells the provider lines whose hand-back the face has finished
// (or will not do), except those of the utterance keep (0: all).
void ReleaseLeaving(Face& f, uint64_t keep = 0) {
  for (auto it = f.leaving.begin(); it != f.leaving.end();) {
    if (keep && (*it)->utterance_id == keep) {
      ++it;
      continue;
    }
    ProviderLineLeft(*it, BFA_kCompleted);
    it = f.leaving.erase(it);
  }
}

// Lock held: the line is leaving the face (or never got there) for the
// reason (BFA_StopReason). A provider's is told at once when nothing of it
// was shown or another utterance holds the face; else once the face is
// handed back (Expressions, Face::leaving): between the segments of its
// utterance the face keeps its expression state for the next one.
void LineLeft(uint8_t* fg, Playing& p, uint32_t reason) {
  if (!ProviderLine(p)) return;
  ProviderLineStopping(p.external, reason);
  Face* face = p.direct_frames != 0 ? LiveFace(fg) : nullptr;
  const uint64_t holder = face ? ProviderHolder(p.external->form_id) : 0;
  if (!face || (holder && holder != p.external->utterance_id)) {
    ProviderLineLeft(p.external, reason);
    return;
  }
  face->leaving.push_back(p.external);
}

}  // namespace

double Now() {
  static const LARGE_INTEGER f = [] {
    LARGE_INTEGER frequency;
    QueryPerformanceFrequency(&frequency);
    return frequency;
  }();
  LARGE_INTEGER c;
  QueryPerformanceCounter(&c);
  return
#ifdef ENABLE_COMMONLIBSSE_TESTING
      testing::clock_offset.load() +
#endif
      static_cast<double>(c.QuadPart) / f.QuadPart;
}

Stats SnapshotStats(size_t& cached, size_t& queued) {
  ScopedLock guard(lock, ScopedLock::kShared);
  Stats st = stats;
  cached = cache.size();
  queued = queue.size();
  return st;
}

void WakeWorker() {
  ScopedLock guard(lock);
  ++settings_revision;
  WakeConditionVariable(&queue_cv);
}

a2f::GpuAdapterList SnapshotGpuAdapters() {
  ScopedLock guard(lock, ScopedLock::kShared);
  return gpu_adapters;
}
void RequestGpuAdapters() {
  {
    ScopedLock guard(lock);
    if (gpu_adapters.requested && !gpu_adapters.complete) return;
    gpu_adapters.requested = true;
    gpu_adapters.complete = false;
    gpu_adapters.error.clear();
  }
  gpu_adapters_requested = true;
  WakeWorker();
}

void RequestGpuRetry() {
  gpu_retry_requested = true;
  WakeWorker();
}

void ClearLines() {
  ScopedLock guard(lock);
  for (auto it = cache.begin(); it != cache.end();)
    it = it->second->state == State::kPending ? std::next(it) : cache.erase(it);
}

namespace {

// "Data\Sound\Voice\Skyrim.esm\MaleNord\X.wav" ->
// "Sound\Voice\Skyrim.esm\MaleNord\X" (engine resource path, no extension).
bool VoiceStem(const char* voice, std::string& stem) {
  const char* p = voice;
  for (; *p; ++p)
    if (!_strnicmp(p, "sound\\", 6) || !_strnicmp(p, "sound/", 6)) break;
  if (!*p) return false;
  stem = p;
  size_t dot = stem.find_last_of('.'), sl = stem.find_last_of("\\/");
  if (dot != std::string::npos && (sl == std::string::npos || dot > sl))
    stem.resize(dot);
  return !stem.empty();
}

// Confirmed silent, missing or unusable audio releases the speech pose.
// Pending inference still prepares the head and eases out the previous pose.
bool FailedSpeech(uint8_t* fg) {
  ScopedLock guard(lock, ScopedLock::kShared);
  auto it = playing.find(fg);
  return it != playing.end() && it->second.entry->state == State::kFailed;
}

// Whether the actor (its base an NPC) is female.
bool IsFemale(uint8_t* ref) {
  uint8_t* base = reinterpret_cast<uint8_t*>(
      reinterpret_cast<RE::TESObjectREFR*>(ref)->data.objectReference);
  return base &&
         reinterpret_cast<RE::TESForm*>(base)->formType.underlying() ==
             kFtNpc &&
         (reinterpret_cast<RE::TESNPC*>(base)
              ->actorData.actorBaseFlags.underlying() &
          1);
}

// Exact base-game race IDs also cover overridden records and custom NPCs
// using those races. Shared head geometry alone must not enable this limit.
bool LimitLowerLip(uint8_t* actor) {
  uint8_t* race =
      actor
          ? reinterpret_cast<uint8_t*>(
                reinterpret_cast<RE::Actor*>(actor)->GetActorRuntimeData().race)
          : nullptr;
  if (!race) return false;
  switch (reinterpret_cast<RE::TESForm*>(race)->formID) {
    case 0x13745:  // Khajiit
    case 0x88845:  // Khajiit vampire
    case 0x13740:  // Argonian
    case 0x8883A:  // Argonian vampire
      return true;
    default:
      return false;
  }
}

bool RaceAllowed(uint8_t* actor) {
  uint8_t* race =
      actor
          ? reinterpret_cast<uint8_t*>(
                reinterpret_cast<RE::Actor*>(actor)->GetActorRuntimeData().race)
          : nullptr;
  return !race || RaceEnabled(reinterpret_cast<RE::TESForm*>(race)->formID);
}

// Actor with a FaceGen head and an enabled race.
bool SpeakerOk(uint8_t* ref) {
  if (!ref ||
      reinterpret_cast<RE::TESForm*>(ref)->formType.underlying() != kFtActor)
    return false;
  uint8_t* base = reinterpret_cast<uint8_t*>(
      reinterpret_cast<RE::TESObjectREFR*>(ref)->data.objectReference);
  uint8_t* race = reinterpret_cast<uint8_t*>(
      reinterpret_cast<RE::Actor*>(ref)->GetActorRuntimeData().race);
  if (!base ||
      reinterpret_cast<RE::TESForm*>(base)->formType.underlying() != kFtNpc ||
      !race ||
      !(reinterpret_cast<RE::TESRace*>(race)->data.flags.underlying() & 2))
    return false;
  return RaceAllowed(ref) && NpcAllowed(ref);
}

bool IsPlayer(uint8_t* actor) {
  return reinterpret_cast<RE::TESForm*>(actor)->formID == kPlayerFormId;
}

uint32_t FormIdOf(const uint8_t* actor) {
  return actor ? reinterpret_cast<RE::TESForm*>(const_cast<uint8_t*>(actor))
                     ->formID
               : 0;
}

// The plugin animates this face: Enabled and, for the player's, the player
// setting.
bool EnabledFor(const Tune& tune, bool player) {
  return tune.enabled && (!player || tune.player);
}

// The factions SexLab (all versions, P+ too) and OStim (NG, Standalone) keep
// an actor in for as long as it is in one of their scenes. Those scenes
// shape the face themselves (open mouths, expressions) and other mods read
// it, while their moans and comments are voiced lines: the plugin leaves
// such an actor's face alone.
struct SceneFaction {
  const char* file;
  uint32_t id;  // in the file
};
constexpr SceneFaction kSceneFactions[] = {
    {"SexLab.esm", 0xE50F},  // SexLabAnimatingFaction
    {"OStim.esp", 0xECA},    // OStimActorCountFaction
    {"OStim.esp", 0xD93},    // OStimExcitementFaction
};
// The factions of kSceneFactions that are loaded: the first
// num_scene_factions; fixed after LoadSceneFactions.
const uint8_t* scene_factions[std::size(kSceneFactions)];
size_t num_scene_factions;

// At kMessageDataLoaded.
void LoadSceneFactions() {
  num_scene_factions = 0;
  const auto* factions = FormArray(kFtFaction);
  if (!factions) {
    Log("scene factions: no form array");
    return;
  }
  for (uint32_t i = 0; i < factions->size(); ++i) {
    const auto* form = factions->data()[i];
    if (!form || form->formType.underlying() != kFtFaction) continue;
    const uint32_t id = form->formID;
    auto* f = reinterpret_cast<uint8_t*>(factions->data()[i]);
    for (const SceneFaction& sf : kSceneFactions) {
      if (LocalFormId(id) != sf.id) continue;
      const char* name = SourceFileName(f);
      if (!name || _stricmp(name, sf.file) ||
          num_scene_factions == std::size(scene_factions)) {
        continue;
      }
      scene_factions[num_scene_factions++] = f;
      Log("scene faction %s %X found (%08X)", sf.file, sf.id, id);
    }
  }
  if (!num_scene_factions) Log("scene factions: none (no SexLab or OStim)");
}

// The actor is in a SexLab or OStim scene. Asks the engine: not to be called
// with the plugin's or a face's lock held.
bool InScene(const uint8_t* actor) {
  if (!num_scene_factions || !actor ||
      reinterpret_cast<const RE::TESForm*>(actor)->formType.underlying() !=
          kFtActor)
    return false;
  for (size_t i = 0; i < num_scene_factions; ++i) {
    if (reinterpret_cast<const RE::Actor*>(actor)->IsInFaction(
            reinterpret_cast<const RE::TESFaction*>(scene_factions[i])))
      return true;
  }
  return false;
}

// ZaZ's zbfBondageShell documents gag and mouth-pose keywords as the
// inter-mod equipment contract. Resolve optional keywords after plugins load;
// items from dependent mods count too. Never infer restraints from item names.
std::vector<const RE::BGSKeyword*> mouth_gear_keywords;

bool MouthGearKeyword(const char* name) {
  if (!name) return false;
  for (const char* wanted : {"zbfWornGag", "zbfEffectOpenMouth",
                            "zbfAnimMouthOpen", "zbfAnimMouthClose",
                            "zbfAnimMouthKhajiitPlug", "zbfAnimMouthShout"})
    if (!_stricmp(name, wanted)) return true;
  return false;
}

void LoadMouthGearKeywords() {
  mouth_gear_keywords.clear();
  const auto* forms = FormArray(static_cast<uint8_t>(RE::FormType::Keyword));
  if (!forms) return;
  for (auto* form : *forms) {
    if (!form || form->formType != RE::FormType::Keyword) continue;
    auto* keyword = static_cast<RE::BGSKeyword*>(form);
    const char* file = SourceFileName(reinterpret_cast<uint8_t*>(form));
    if (file && !_stricmp(file, "ZaZAnimationPack.esm") &&
        MouthGearKeyword(keyword->formEditorID.c_str()))
      mouth_gear_keywords.push_back(keyword);
  }
  Log("mouth gear: %zu ZaZ keywords loaded", mouth_gear_keywords.size());
}

bool MouthGearItem(const RE::TESForm* item) {
  if (!item || item->formType != RE::FormType::Armor) return false;
  const auto* armor = static_cast<const RE::TESObjectARMO*>(item);
  for (auto* keyword : armor->GetKeywords())
    for (auto* blocked : mouth_gear_keywords)
      if (keyword == blocked) return true;
  return false;
}

// Read the actor's third-person equipped slots, including for the player in
// first person. Bounded slot traversal, no inventory initialization or polling
// cache. A missing biped has no face to animate; recheck when it is rebuilt.
// Like InScene, call outside the plugin/face locks.
bool WearingMouthGear(const uint8_t* actor) {
  if (mouth_gear_keywords.empty() || !actor ||
      reinterpret_cast<const RE::TESForm*>(actor)->formType != RE::FormType::ActorCharacter)
    return false;
  const auto& biped = reinterpret_cast<const RE::Actor*>(actor)->GetBiped(false);
  if (!biped) return false;
  for (const auto& slot : biped->objects)
    if (MouthGearItem(slot.item)) return true;
  return false;
}

bool FaceExternallyOwned(const uint8_t* actor) {
  return InScene(actor) || WearingMouthGear(actor);
}

// speaker is cleared by the engine when dialogue closes; lastSpeaker (0x6C)
// deliberately is not used. The same singleton already drives TrackPartner.
uint32_t PartnerHandle() {
  static auto mtm = reinterpret_cast<uint8_t**>(Addr(kIdMenuTopicManager));
  return !loading_game && mtm && *mtm
             ? reinterpret_cast<RE::MenuTopicManager*>(*mtm)
                   ->speaker.native_handle()
             : 0;
}

// Whether the actor is one the plugin animates at all: everyone, or in Potato
// mode the player and the dialogue partner. Enabled is not part of it: that
// restores vanilla animation and stops inference; this only filters speakers.
bool DialogueAllows(const Tune& tune, uint32_t handle, bool player = false) {
  return !tune.potato_mode || player || (handle && handle == PartnerHandle());
}

// Lock held: keep shared work while at least one requester still needs it.
bool EntryAllowed(const Entry& entry, const Tune& tune) {
  return tune.enabled && !entry.cancelled && entry.state == State::kPending &&
         entry.performance_sentiment == tune.performance_sentiment &&
         CurrentWorld(entry.epoch) &&
         (!entry.model_state ||
          entry.model_state->network == tune.SelectedModel()) &&
         std::any_of(entry.requesters.begin(), entry.requesters.end(),
                     [&](const Entry::Requester& requester) {
                       return DialogueAllows(tune, requester.handle,
                                             requester.player);
                     });
}

bool JobAllowed(const Job& j) {
  ScopedLock guard(lock, ScopedLock::kShared);
  return EntryAllowed(*j.entry, SnapshotTune());
}

// Lock held, after the entry's state is final: its provider lines are told
// of a failure (success shows as kStarted once the line is on the face).
void NotifyWatchers(Entry& e) {
  std::vector<PlayerSpeechPtr> watchers = std::move(e.watchers);
  e.watchers.clear();
  if (e.state == State::kReady) return;
  for (const PlayerSpeechPtr& w : watchers)
    ProviderLineFailed(w, e.failure_reason);
}

// Lock held. Prepare and Start may resolve different cache entries.
void WatchEntry(Entry& entry, const PlayerSpeechPtr& line) {
  if (entry.state == State::kPending) {
    if (std::find(entry.watchers.begin(), entry.watchers.end(), line) ==
        entry.watchers.end())
      entry.watchers.push_back(line);
  } else if (entry.state == State::kFailed) {
    ProviderLineFailed(line, entry.failure_reason);
  }
}

// Lock held. Never leave cancelled entries pending or poison future requests.
void CancelJob(const Job& j) {
  j.entry->state = State::kFailed;
  j.entry->failure_reason = BFA_kResultCancelled;
  NotifyWatchers(*j.entry);
  auto it = cache.find(j.cache_key);
  if (it != cache.end() && it->second == j.entry) cache.erase(it);
}

// Queue a line unless it is cached or in flight (with the emotion settings
// as now: after a change the next line spoken is generated anew). Returns
// its entry. female: the speaker's sex; with a face per sex (v3.0) a line is
// generated for each. given: instead of the game's emotion, a provider's
// (a Skyrim type and value, or explicit channels; its gain a factor on the
// setting). audio: a provider's bytes, whose stem names them by content.
std::shared_ptr<Entry> Request(
    const std::string& stem, int emo, int val, bool female, bool urgent,
    const Tune& tune, uint32_t speaker, uint64_t epoch, bool player = false,
    const a2f::Emotion* given = nullptr,
    std::shared_ptr<const std::vector<uint8_t>> audio = {},
    uint8_t* actor = nullptr) {
  std::string key = Lower(stem.c_str());
  const FaceTune settings = NpcFaceTune(tune, actor, female, player);
  const bool channels = given && given->type == a2f::kExplicitEmotion;
  if (given && !channels) {
    emo = given->type;
    val = given->value;
  }
  const float factor = tune.vanilla_emotions ? (given ? given->gain : 1) : 0;
  // neutral: no emotion to feed, or none wanted
  if (settings.voice_emotion <= 0 || factor <= 0 ||
      (!channels && (emo < 1 || emo > 7))) {
    emo = 0;
    val = 0;
  }
  a2f::Emotion em;
  em.type = channels && factor > 0 && settings.voice_emotion > 0
                ? a2f::kExplicitEmotion
                : emo;
  em.value = val;
  em.gain = em.type ? settings.voice_emotion * factor : 0;
  em.swell = em.gain > 0 ? settings.voice_swell : 0;
  em.maximum = settings.max_emotion;
  if (em.type == a2f::kExplicitEmotion) em.channels = given->channels;
  const Tune shared = NpcSharedTune(tune, player ? nullptr : actor);
  em.sentiment_multiplier =
      shared.sentiment_analysis && SentimentAvailable(shared)
          ? shared.sentiment_multiplier
          : 0;
  auto part = [&key](double v) {
    key += '|';
    key += std::to_string(std::lround(v));
  };
  for (double v : {1. * em.type, 1. * val, em.gain * 100., em.swell * 100.})
    part(v);
  if (em.type == a2f::kExplicitEmotion)
    for (float level : em.channels) part(level * 1000.);
  // Preserve every stored cap value: even a small edit must bypass old work.
  const auto cap_part = [&](float cap) {
    uint32_t cap_bits;
    std::memcpy(&cap_bits, &cap, sizeof cap_bits);
    key += "|cap=" + std::to_string(cap_bits);
  };
  for (float cap : em.maximum) cap_part(cap);
  cap_part(em.sentiment_multiplier);
  part(tune.performance_sentiment);
  const auto selection = SnapshotModel();
  part(selection->network);
  if (selection->mapping->faces.size() > 1) key += female ? "|f" : "|m";
  ScopedLock guard(lock);
  if (!CurrentWorld(epoch) || !DialogueAllows(tune, speaker, player)) return {};
  // Include in-flight work: it may already have left the queue.
  auto previous = cache.find(key);
  if (previous != cache.end() && previous->second->state == State::kPending &&
      !EntryAllowed(*previous->second, tune)) {
    previous->second->state = State::kFailed;
    cache.erase(previous);
  }
  auto& e = cache[key];
  if (e && e->degraded && e->state != State::kPending &&
      Now() >= sentiment_retry_at)
    e.reset();  // a playing line keeps its own reference
  if (!e) {
    e = std::make_shared<Entry>();
    e->epoch = epoch;
    e->performance_sentiment = tune.performance_sentiment;
    e->model_state = selection;
    Job j{key, stem, em, female, e, std::move(audio)};
    if (urgent) {
      queue.push_front(j);
    } else {
      queue.push_back(j);
    }
    WakeConditionVariable(&queue_cv);
  } else if (urgent && e->state == State::kPending) {
    // prefetched but not started: move to the front
    for (auto it = queue.begin(); it != queue.end(); ++it)
      if (it->cache_key == key) {
        Job j = *it;
        queue.erase(it);
        queue.push_front(j);
        break;
      }
  }
  if (e->state == State::kPending &&
      std::none_of(e->requesters.begin(), e->requesters.end(),
                   [&](const Entry::Requester& requester) {
                     return requester.handle == speaker &&
                            requester.player == player;
                   })) {
    e->requesters.push_back({speaker, player});
  }
  e->last_use = ++use_clock;
  return e;
}

// Read a file through BSResourceNiBinaryStream (BSAs + loose files, MO2 VFS);
// path relative to Data.
bool ReadGameFile(const std::string& path, std::vector<uint8_t>& out) {
  using Ctor = void* (*)(void*, const char*, uint64_t, void*);
  using Dtor = void (*)(void*);
  // CommonLib's private ctor/dtor lack VR VariantIDs. Keep only these entry
  // points local; storage, resource size and partial reads use its real types.
  alignas(RE::BSResourceNiBinaryStream)
      std::byte storage[sizeof(RE::BSResourceNiBinaryStream)]{};
  auto& stream = *reinterpret_cast<RE::BSResourceNiBinaryStream*>(storage);
  Engine<Ctor, kIdStreamCtor>()(&stream, path.c_str(), 0, nullptr);
  bool ok = false;
  if (stream.stream) {
    const uint32_t n = stream.stream->totalSize;
    out.resize(n);
    uint32_t got = 0;
    while (got < n) {
      const auto before = stream.RE::NiBinaryStream::tell();
      stream.read(out.data() + got, n - got);
      const auto count = stream.RE::NiBinaryStream::tell() - before;
      if (!count) break;
      got += count;
    }
    ok = n > 0 && got == n;
  }
  Engine<Dtor, kIdStreamDtor>()(&stream);
  return ok;
}

// EFA has no ESP. Identify the winning resources, including BSA/MO2 overrides,
// rather than guessing from a filename shared with vanilla. Unknown versions
// or modified assets retain the uncalibrated mouth strength.
struct ExpressiveHead {
  const char* path;
  size_t size;
  uint64_t hash;
};
constexpr ExpressiveHead kExpressiveHeads[] = {
    {"meshes\\actors\\character\\character assets\\MaleHead.tri", 301905,
     0x78d43dd9b2f22ba8ull},  // Male Edition 1.21
    {"meshes\\actors\\character\\character assets\\FemaleHead.tri", 329989,
     0x890ed4834ecadef6ull},  // Female Edition 1.7
};

bool IsExpressiveHead(const std::vector<uint8_t>& bytes, bool female) {
  const auto& expected = kExpressiveHeads[female];
  if (bytes.size() != expected.size) return false;
  uint64_t hash = 14695981039346656037ull;
  for (uint8_t byte : bytes) hash = (hash ^ byte) * 1099511628211ull;
  return hash == expected.hash;
}

void LoadExpressiveFaces() {
  std::vector<uint8_t> bytes;
  for (bool female : {false, true}) {
    config.expressive_faces[female] =
        ReadGameFile(kExpressiveHeads[female].path, bytes) &&
        IsExpressiveHead(bytes, female);
  }
  Log("Expressive Facial Animation: men %s, women %s (mouth x%g if detected)",
      config.expressive_faces[0] ? "detected" : "not detected",
      config.expressive_faces[1] ? "detected" : "not detected",
      kExpressiveMouthScale);
}

// ---------------- worker ----------------
// Network events go to the log.
void NetEvent(const std::string& text) { skse::Report("%s", text.c_str()); }

// Worker-owned successful shared variants, separately for each facial model
// and sentiment (slot 0). Choosing these never creates/probes another device.
a2f::SharedGpuSettings shared_gpu_settings[a2f::kNetUltraPerformance + 1];

void UnloadSentiment();

std::unique_ptr<a2f::Net> CreateNet(bool gpu, std::string& err,
                                    std::string& info) {
  a2f::diagnostics::Attempt diagnostic(a2f::diagnostics::Component::kFace, gpu,
                                       config.independent_device, err);
  const auto finish = [&](std::unique_ptr<a2f::Net> net) {
    if (net) diagnostic.Ready();
    return net;
  };
  const auto selection = SnapshotModel();
  const std::wstring dir = PluginDir(), file = dir + L"\\" + selection->file;
  a2f::OrtOptions o;
  o.gpu = gpu;
  o.gpu_adapter = selection->gpu_adapter;
  o.batch = config.batch;
  o.independent_device = config.independent_device;
  o.force_bundled_agility = config.force_bundled_agility;
  if (selection->mapping->network !=
      ModelFor(selection->network).mapping_network) {
    err = "the A2FDirect bin is of the other network";
    return nullptr;
  }
  if (GetFileAttributesW(file.c_str()) == INVALID_FILE_ATTRIBUTES) {
    char name[260] = "";
    WideCharToMultiByte(CP_UTF8, 0, selection->file.c_str(), -1, name,
                        sizeof name, nullptr, nullptr);
    err = std::string("model file not found: ") + name;
    return nullptr;
  }
  const auto create = [&](std::string& e, std::string& i) {
    switch (selection->network) {
      case a2f::kNetUltraPerformance:
        return a2f::CreateUltraPerformanceNet(dir, file, o, e, i);
      case a2f::kNetPerformance:
        return a2f::CreatePerformanceNet(dir, file, o, e, i);
      case a2f::kNetV3:
        return a2f::CreateOrtNet3(dir, file, o, e, i);
      default:
        return a2f::CreateOrtNet(dir, file, o, e, i);
    }
  };
  if (!gpu) return finish(create(err, info));
  if (config.independent_device) {
    // A lost device latches the context (RunFailed); the keeper's bounded
    // retries arrive here. Both sessions sat on that device: drop the
    // sentiment one too, then create the context again. A missing API or a
    // failed device creation stays latched until the user's retry.
    std::string latched;
    if (!a2f::GpuAllowed(latched)) {
      if (!a2f::LostDevice(latched)) {
        err = latched;
        return nullptr;
      }
      UnloadSentiment();
      a2f::RetryGpu();
      skse::Report("independent GPU device lost (%s): creating it again",
                   latched.c_str());
    }
    return finish(create(err, info));
  }
  return finish(a2f::CreateSharedGpu(
      [&](bool metacommands, bool wrapper, std::string& error) {
        o.metacommands = metacommands;
        a2f::SetUseDeviceWrapper(wrapper);
        const double started = Now();
        auto net = create(error, info);
        skse::Report("GPU session with metacommands %s on %s: %s after %.0f ms",
                     metacommands ? "on" : "off",
                     wrapper ? "the wrapper" : "the device",
                     net ? "loaded" : "failed", (Now() - started) * 1e3);
        return net;
      },
      shared_gpu_settings[selection->network], a2f::DeviceWrapped, err,
      NetEvent));
}

// Copies the keeper's state to the settings window.
void PublishNet(const a2f::NetKeeper& keeper) {
  const auto selection = SnapshotModel();
  static std::string diagnosed;  // the error the diagnosis was made for
  static auto& why = *new std::vector<std::string>;
  if (keeper.error().empty()) {
    diagnosed.clear();
    why.clear();
  } else if (keeper.error() != diagnosed) {
    diagnosed = keeper.error();
    // DXGI only (adapter and VRAM budget); no Direct3D device is created.
    why = a2f::DiagnoseNet(selection->gpu, keeper.error(),
                           selection->gpu ? a2f::QueryGpu() : a2f::GpuState{},
                           ModelFor(selection->network).vram_mb);
    for (const auto& line : why) skse::Report("%s", line.c_str());
    // the system as it is at the first failure (the report at startup is
    // from the main menu)
    static bool reported;
    if (selection->gpu && !reported) {
      reported = true;
      skse::Report("-- at the first failure");
      auto report = a2f::GpuReport(false);
      a2f::diagnostics::State().System(report);
      for (const auto& line : report) skse::Report("%s", line.c_str());
    }
  }
  net_loaded = keeper.loaded();
  // The menu never asks the driver itself: this is its VRAM and adapter view.
  const a2f::GpuState gpu = selection->gpu ? a2f::QueryGpu() : a2f::GpuState{};
  ScopedLock guard(lock);
  stats.gpu = gpu;
  stats.network = keeper.loaded() ? keeper.version() : selection->network;
  stats.want_gpu = selection->gpu;
  stats.gpu_adapter = selection->gpu_adapter;
  stats.net_info = keeper.info();
  stats.net_error = why;
  stats.net_detail = keeper.error();
  stats.net_retry_at = keeper.retry_at();
  stats.net_gpu = keeper.on_gpu();
}

// Worker-owned: never accessed by the UI. Disabling Sentiment Analysis releases
// the session on the worker after an already submitted inference returns.
std::unique_ptr<a2f::EmotionNet> sentiment_net;
bool sentiment_gpu = false;
bool sentiment_performance = false;
int sentiment_device = 0;

void SentimentStatus(const std::string& message) {
  ScopedLock guard(lock);
  stats.sentiment_info = message;
}

std::string IdleSentimentStatus() {
  return SentimentAvailable(SnapshotTune())
             ? "off (sentiment model unloaded)"
             : "unavailable (optional sentiment model not installed)";
}

void UnloadSentiment() {
  sentiment_net.reset();
  a2f::diagnostics::State().Unload(a2f::diagnostics::Component::kSentiment);
  sentiment_retry_at = 0;
  SentimentStatus(IdleSentimentStatus());
}

// Worker only: changing emotion settings must not rebuild the facial session.
void ConfigureSentiment(const Tune& tune) {
  if (sentiment_performance == tune.performance_sentiment &&
      sentiment_device == tune.emotion_device)
    return;
  UnloadSentiment();
  if (sentiment_device != tune.emotion_device) {
    shared_gpu_settings[0] = {};
    if (tune.EmotionUsesGpu()) a2f::RetryGpu();
  }
  sentiment_performance = tune.performance_sentiment;
  sentiment_device = tune.emotion_device;
}

enum class Sentiment {
  kReady,        // in emotion.sentiment (empty: none wanted)
  kCancelled,    // the line is no longer wanted
  kUnavailable,  // the model failed or is in its retry wait: dialogue only
};

// The line's voice sentiment. Unavailable leaves dialogue usable: the line
// is generated without it and marked degraded (Entry). A cancelled line
// keeps a healthy session for the next one.
Sentiment PrepareSentiment(const Job& job, const std::vector<float>& pcm,
                           a2f::Emotion& emotion) {
  // 0 without the model (Request)
  if (emotion.sentiment_multiplier <= 0) {
    emotion.sentiment.clear();
    return Sentiment::kReady;
  }
  const auto allowed = [&] {
    return JobAllowed(job) && SnapshotTune().sentiment_analysis;
  };
  if (!allowed()) return Sentiment::kCancelled;
  const Tune tune = SnapshotTune();
  ConfigureSentiment(tune);
  const bool performance = tune.performance_sentiment;
  std::string err;
  if (sentiment_gpu && !a2f::GpuAllowed(err)) sentiment_net.reset();
  if (!sentiment_net) {
    if (Now() < sentiment_retry_at) return Sentiment::kUnavailable;
    a2f::OrtOptions options;
    options.gpu = tune.EmotionUsesGpu();
    options.gpu_adapter = std::max(0, tune.emotion_device);
    options.independent_device = config.independent_device;
    options.force_bundled_agility = config.force_bundled_agility;
    a2f::diagnostics::Attempt diagnostic(
        a2f::diagnostics::Component::kSentiment, options.gpu,
        options.independent_device, err);
    sentiment_gpu = options.gpu;
    const std::wstring dir = PluginDir();
    const auto create = [&](std::string& error) {
      return a2f::CreateOrtEmotionNet(
          dir,
          dir + L"\\" +
              (performance ? kPerformanceSentimentModel : kSentimentModel),
          options, error, performance);
    };
    if (options.gpu && !options.independent_device) {
      sentiment_net = a2f::CreateSharedGpu(
          [&](bool metacommands, bool wrapper, std::string& error) {
            options.metacommands = metacommands;
            a2f::SetUseDeviceWrapper(wrapper);
            return create(error);
          },
          shared_gpu_settings[0], a2f::DeviceWrapped, err, NetEvent);
    } else
      sentiment_net = create(err);
    if (sentiment_net) {
      diagnostic.Ready();
      SentimentStatus(
          std::string(performance ? "Performance (experimental): " : "Full: ") +
          (options.gpu ? "sentiment ready (GPU " +
                             std::to_string(options.gpu_adapter) + ")"
                       : "sentiment ready (CPU)"));
    }
  }
  if (sentiment_net &&
      a2f::AnalyzeSentiment(*sentiment_net, pcm.data(), pcm.size(),
                            emotion.sentiment, err, allowed))
    return Sentiment::kReady;
  if (!allowed()) return Sentiment::kCancelled;
  if (sentiment_net)
    a2f::diagnostics::State().RuntimeFailure(
        a2f::diagnostics::Component::kSentiment, err);
  sentiment_net.reset();
  sentiment_retry_at = Now() + (sentiment_gpu ? 30 : 60);
  SentimentStatus(std::string("dialogue only; ") +
                  (sentiment_gpu ? "GPU sentiment unavailable: "
                                 : "CPU sentiment unavailable: ") +
                  err);
  skse::Report("Audio2Emotion unavailable: %s", err.c_str());
  return Sentiment::kUnavailable;
}

// Generates one queued line and stores the result in the cache.
void Process(const Job& j, a2f::NetKeeper& keeper) {
  const auto allowed = [&] { return JobAllowed(j); };
  if (!allowed()) {
    ScopedLock guard(lock);
    CancelJob(j);
    return;
  }
  static std::atomic<int> logged;
  std::vector<uint8_t> bytes;
  const std::vector<uint8_t>* source = &bytes;
  std::string err, file;
  if (j.audio) {
    source = j.audio.get();
    file = j.path;
  } else {
    for (const char* ext : {".fuz", ".xwm", ".wav"})
      if (ReadGameFile(j.path + ext, bytes)) {
        file = j.path + ext;
        break;
      }
  }
  std::shared_ptr<a2f::Line> line;
  double t = Now(), audio_sec = 0;
  bool inference_attempted = false;
  int generated_network = j.entry->model_state->network;
  bool no_net = false;  // not this line's fault: generate it again next time
  Sentiment sentiment = Sentiment::kReady;
  bool silent = false;  // nothing to animate: vanilla, not a failure
  if (file.empty()) {
    err = "no voice file";
  } else {
    std::vector<float> pcm;
    if (a2f::DecodeAudio(source->data(), source->size(), pcm, err)) {
      audio_sec = pcm.size() / static_cast<double>(a2f::kSampleRate);
      silent = !a2f::HasAudibleAudio(pcm.data(), pcm.size());
      if (silent) {
        err = "silent or unusable audio; keeping vanilla facial animation";
      } else {
        a2f::Emotion emotion = j.emotion;
        sentiment = PrepareSentiment(j, pcm, emotion);
        line = std::make_shared<a2f::Line>();
        inference_attempted = true;
        if (!keeper.Run(
                [&](a2f::Net& net, std::string& e) {
                  generated_network = net.Version();
                  const direct::Mark& face =
                      j.entry->model_state->mapping->Face(j.female);
                  bool generated;
                  if (net.Version() == a2f::kNetPerformance ||
                      net.Version() == a2f::kNetUltraPerformance)
                    generated = a2f::GeneratePerformance(
                        net, pcm.data(), pcm.size(), emotion, j.female, *line,
                        e, allowed);
                  else if (net.Version() != a2f::kNetV3)
                    generated = a2f::Generate(
                        net, pcm.data(), pcm.size(), emotion, face.proj.data(),
                        face.rest.data(), face.k, *line, e, allowed);
                  else {
                    // Claire speaks the women's lines, Mark the men's.
                    const a2f::Face3 f{j.female ? 0 : 2, face.proj.data(),
                                       face.tongue_proj.data(),
                                       face.rest.data(), face.k};
                    generated = a2f::Generate3(net, pcm.data(), pcm.size(),
                                               emotion, f, *line, e, allowed);
                  }
                  if (!generated && net.failed && allowed())
                    a2f::diagnostics::State().RuntimeFailure(
                        a2f::diagnostics::Component::kFace, e);
                  return generated;
                },
                err, no_net, allowed))
          line.reset();
        PublishNet(keeper);
      }
    }
  }
  double ms = (Now() - t) * 1e3;
  ScopedLock guard(lock);
  if (!EntryAllowed(*j.entry, SnapshotTune())) {
    CancelJob(j);
    return;
  }
  auto it = cache.find(j.cache_key);
  if (it != cache.end() && it->second == j.entry) {
    it->second->state = line ? State::kReady : State::kFailed;
    it->second->line = line;
    // Re-read silent/missing assets on the next request: a TTS provider may
    // replace them later. The active line retains its fallback entry.
    if (!line && (no_net || silent || file.empty())) cache.erase(it);
  }
  j.entry->failure_reason = no_net                ? BFA_kResultModelNotLoaded
                            : inference_attempted ? BFA_kResultInferenceFailed
                                                  : BFA_kResultAudioFailed;
  NotifyWatchers(*j.entry);
  if (line) {
    ++stats.lines;
    stats.audio_sec += audio_sec;
    stats.gen_sec += ms / 1e3;
    stats.last_ms = ms;
  } else if (!silent) {
    ++stats.failed;
    stats.last_error = file.empty() ? j.path + ": " + err : err;
  }
  if (cache.size() > static_cast<size_t>(config.cache_lines)) {
    // drop the least recently used finished lines
    std::vector<std::pair<uint64_t, std::string>> v;
    for (auto& [k, e] : cache)
      if (e->state != State::kPending) v.emplace_back(e->last_use, k);
    std::sort(v.begin(), v.end());
    for (size_t i = 0;
         i < v.size() &&
         cache.size() > static_cast<size_t>(config.cache_lines) * 3 / 4;
         ++i)
      cache.erase(v[i].second);
  }
  j.entry->degraded = sentiment == Sentiment::kUnavailable;
  guard.Unlock();
  if (LogMore(logged)) {
    if (line) {
      // what the network was given: every emotion with a mean of 0.02 up
      static const char* const kNames[a2f::kNumMod] = {
          "amazement", "anger", "cheekiness",  "disgust", "fear",
          "grief",     "joy",   "outofbreath", "pain",    "sadness"};
      std::string fed;
      for (int c = 0; c < a2f::kNumMod; ++c)
        if (line->emotion_mean[c] >= 0.02f) {
          char b[64];
          snprintf(b, sizeof b, " %s %.2f (peak %.2f)", kNames[c],
                   line->emotion_mean[c], line->emotion_peak[c]);
          fed += b;
        }
      Log("gen %s (%s) emo %d/%d: %.2f s audio, %u frames in %.0f ms; fed%s",
          file.c_str(),
          generated_network == a2f::kNetUltraPerformance ? "Ultra Performance"
          : generated_network == a2f::kNetPerformance    ? "Performance"
          : generated_network == a2f::kNetV2             ? "Mark"
          : j.female                                     ? "Claire"
                                                         : "Mark 3.0",
          j.emotion.type, j.emotion.value, audio_sec, line->frames, ms,
          fed.empty() ? " no emotion" : fed.c_str());
    } else {
      Log("skip %s: %s", j.path.c_str(), err.c_str());
    }
  }
}

// Worker only, between inference calls: applies the settings' model and
// device. Playing lines keep their mapping; pending work is cancelled. When
// the new mapping cannot be loaded the selection goes back to the model that
// is running, so new lines keep being generated.
void SwitchModel(a2f::NetKeeper& keeper, const Tune& tune) {
  const auto old = SnapshotModel();
  const int network = tune.SelectedModel();
  const bool gpu = !tune.force_cpu;
  const bool retry = gpu_retry_requested.exchange(false);
  const bool adapter_changed = old->gpu_adapter != tune.gpu_adapter;
  if (old->network == network && old->gpu == gpu && !adapter_changed && !retry)
    return;
  if (retry && !gpu && old->network == network && !old->gpu &&
      !adapter_changed) {
    UnloadSentiment();
    if (tune.EmotionUsesGpu()) a2f::RetryGpu();
    return;
  }
  UnloadSentiment();
  {
    ScopedLock guard(lock);
    stats.switching_model = true;
    stats.model_error.clear();
  }
  skse::Report("model switch requested: %d -> %d, GPU=%d", old->network,
               network, gpu);
  const ModelInfo& files = ModelFor(network);
  auto next = std::make_shared<ModelState>();
  next->network = network;
  next->gpu = gpu;
  next->gpu_adapter = tune.gpu_adapter;
  next->file = files.model_file;
  next->mapping = old->mapping;
  if (old->mapping->network != files.mapping_network) {
    auto mapping = std::make_shared<direct::Model>();
    std::string err;
    if (!mapping->Load(PluginDir() + L"\\" + files.mapping_file, err)) {
      SetSelectedModel(old->network);
      SaveIni();
      ScopedLock guard(lock);
      stats.switching_model = false;
      stats.model_error =
          "Model switch failed; previous model retained. Fix the file and "
          "select again: " +
          err;
      skse::Report("model switch failed: %s", err.c_str());
      return;
    }
    next->mapping = std::move(mapping);
  }
  keeper.Reconfigure(gpu);
  if (adapter_changed)
    for (auto& settings : shared_gpu_settings) settings = {};
  if (gpu || tune.EmotionUsesGpu()) a2f::RetryGpu();
  {
    ScopedLock guard(lock);
    for (const auto& job : queue) CancelJob(job);
    queue.clear();
    cache.clear();
    std::atomic_store(&active_model, std::shared_ptr<const ModelState>(next));
    // Per-model timings must not mix across switches.
    stats = Stats{};
    stats.switching_model = true;
  }
  PublishNet(keeper);  // clear stale diagnostics before the first new attempt
  if (tune.enabled) keeper.Ensure();
  PublishNet(keeper);
  {
    ScopedLock guard(lock);
    stats.switching_model = false;
  }
  skse::Report("model switch applied: %d, %s", network, keeper.info().c_str());
}

// One lifecycle/queue iteration. Tests can inspect the wait policy without
// blocking; production waits under lock so jobs and settings cannot be lost.
DWORD WorkerStep(a2f::NetKeeper& keeper, bool wait = true) {
  uint64_t revision;
  {
    ScopedLock guard(lock);
    revision = settings_revision;
  }
  if (gpu_adapters_requested.exchange(false)) {
    auto discovered = a2f::EnumerateGpuAdapters();
    ScopedLock guard(lock);
    gpu_adapters = std::move(discovered);
  }
  const Tune tune = SnapshotTune();
  SwitchModel(keeper, tune);
  ConfigureSentiment(tune);
  if (!tune.enabled && keeper.loaded()) {
    keeper.Unload();
    PublishNet(keeper);
  }
  if (!tune.sentiment_analysis || !tune.enabled || !SentimentAvailable(tune)) {
    UnloadSentiment();
  } else if (!sentiment_net && sentiment_retry_at == 0)
    SentimentStatus("waiting for speech");
  // Keep the session ready in both modes; Potato mode restricts speakers,
  // not residency. Shared GPU recovery wakes while vanilla lip-sync is used.
  if (tune.enabled && keeper.needs_poll()) {
    keeper.Ensure();
    PublishNet(keeper);
  }
  ScopedLock worker_guard(lock);
  for (auto it = queue.begin(); it != queue.end();) {
    if (!EntryAllowed(*it->entry, SnapshotTune())) {
      CancelJob(*it);
      it = queue.erase(it);
    } else {
      ++it;
    }
  }
  if (queue.empty()) {
    const DWORD timeout = tune.enabled && keeper.needs_poll() ? 1000 : INFINITE;
    if (revision != settings_revision) return 0;
    if (wait) SleepConditionVariableSRW(&queue_cv, &lock, timeout, 0);
    return timeout;
  }
  Job j = queue.front();
  queue.pop_front();
  worker_guard.Unlock();
  Process(j, keeper);
  return 0;
}

DWORD WINAPI Worker(void*) {
  std::string err;
  if (!model.Load(PluginDir() + L"\\" + config.direct, err)) {
    char name[64] = "";
    WideCharToMultiByte(CP_UTF8, 0, config.direct.c_str(), -1, name,
                        sizeof name, nullptr, nullptr);
    skse::Report("%s: %s", name, err.c_str());
    ScopedLock guard(lock);
    stats.startup_error = std::string(name) + ": " + err;
    return 1;
  }
  std::atomic_store(&active_model, SnapshotModel());
  {
    ScopedLock guard(lock);
    stats.network = config.network;
    stats.want_gpu = config.gpu;
    stats.gpu_adapter = config.gpu_adapter;
  }
  ready = true;
  skse::Report("settings: GPU=%d, batch %d, model %ls", config.gpu,
               config.batch, config.model.c_str());
  skse::Report(
      "GPU route: %s; CPU only when selected, no other GPU route is tried",
      config.independent_device ? "independent device, private compute queue"
                                : "4.0-style shared device");
  a2f::diagnostics::State().System(a2f::GpuReport(false));
  a2f::SetOrtLog([](const std::string& s) { skse::Report("%s", s.c_str()); });
  if (config.independent_device) {
    // Advisory: the API itself is queried at the first attempt.
    skse::Report("%s", a2f::IndependentPrerequisite().c_str());
    skse::Report(
        "independent GPU recovery: a lost device is created again, %d "
        "retries at 2/4/8-second intervals; a missing runtime API or failed "
        "device creation waits for Try GPU again",
        a2f::NetKeeper::kIndependentRetries);
  } else {
    skse::Report(
        "shared GPU recovery: metacommands off, conditional wrapper, "
        "then 2/4/8/16/30-second retries; no Auto device probe");
  }
  a2f::NetKeeper keeper(config.gpu, CreateNet, Now, NetEvent,
                        config.independent_device);
  for (;;) WorkerStep(keeper);
}

// ---------------- call-site hooks ----------------
// Every `call rel32` to the callee inside the caller functions (HookCalls) or
// in the whole code section (HookAllCalls) is redirected through a thunk
// (include/a2f/runtime/call_sites.h). Each site keeps its own original target
// (looked up by return address), so a plugin that patches a site after us is
// chained.
template <typename Fn>
Fn SiteOrig(void* ret, skse::Rel callee_id) {
  return call_sites::Orig<Fn>(ret, reinterpret_cast<void*>(Addr(callee_id)));
}
IMAGE_NT_HEADERS* NtHeaders(uint64_t base) {
  return reinterpret_cast<IMAGE_NT_HEADERS*>(
      base + reinterpret_cast<IMAGE_DOS_HEADER*>(base)->e_lfanew);
}
// Hooks the calls to `callee` in [from, to); `caller_id` is 0 for a scan. An
// E8 whose rel32 only happens to reach the callee (data or the tail of another
// instruction) would be hooked too; at one chance in 2^32 per E8 byte that is
// not worth a disassembler, and the hook calls the site's own original anyway.
int HookCallsIn(const char* what, uint64_t callee, uint64_t from, uint64_t to,
                void* hook, uint64_t caller_id) {
  const uint64_t base = skse::ImageBase();
  const uint64_t end = base + NtHeaders(base)->OptionalHeader.SizeOfImage;
  to = std::min(to, end);
  int n = 0;
  for (uint64_t at = from; at + 5 <= to; ++at) {
    if (*reinterpret_cast<uint8_t*>(at) != 0xE8 ||
        call_sites::CallTarget(at) != callee)
      continue;
    if (call_sites::num_sites >= call_sites::kMaxSites) {
      Log("%s: call at rva %llx not hooked (%d sites in use)", what, at - base,
          call_sites::num_sites);
      break;
    }
    if (!call_sites::Hook(at, hook, what, end)) {
      Log("%s: call at rva %llx not hooked (no thunk in reach)", what,
          at - base);
      continue;
    }
    if (caller_id)
      Log("%s: hooked call at rva %llx (caller id %llu)", what, at - base,
          caller_id);
    else
      Log("%s: hooked call at rva %llx", what, at - base);
    ++n;
  }
  return n;
}
template <size_t N>
int HookCalls(const char* what, skse::Rel callee_id,
              const skse::Rel (&callers)[N], void* hook) {
  uint64_t callee = Addr(callee_id);
  if (!callee) {
    Log("%s: address id %llu not found", what, callee_id.Get());
    return 0;
  }
  int n = 0;
  for (skse::Rel caller : callers) {
    uint64_t f = Addr(caller);
    if (f) n += HookCallsIn(what, callee, f, f + 0x2000, hook, caller.Get());
  }
  if (!n)
    Log("%s: no call sites found (another plugin may have replaced them)",
        what);
  return n;
}
// Every call to the callee in the executable's code sections, for callees
// whose callers are not all known (and differ between runtimes). Runs once at
// load: a linear pass over the ~30 MB of code takes a few milliseconds.
int HookAllCalls(const char* what, skse::Rel callee_id, void* hook) {
  const uint64_t base = skse::ImageBase();
  uint64_t callee = Addr(callee_id);
  if (!callee) {
    Log("%s: address id %llu not found", what, callee_id.Get());
    return 0;
  }
  IMAGE_NT_HEADERS* nt = NtHeaders(base);
  int n = 0;
  IMAGE_SECTION_HEADER* section = IMAGE_FIRST_SECTION(nt);
  for (int i = 0; i < nt->FileHeader.NumberOfSections; ++i, ++section) {
    if (!(section->Characteristics & IMAGE_SCN_MEM_EXECUTE)) continue;
    const uint64_t from = base + section->VirtualAddress;
    n += HookCallsIn(what, callee, from, from + section->Misc.VirtualSize,
                     hook, 0);
  }
  if (n)
    Log("%s: %d call sites hooked", what, n);
  else
    Log("%s: no call sites found (another plugin may have replaced them)",
        what);
  return n;
}
// After all plugins loaded: chain in front of anyone who patched our sites
// since.
void RetakeSites() {
  const uint64_t base = skse::ImageBase();
  for (int i = 0; i < call_sites::num_sites; ++i) {
    call_sites::Site& s = call_sites::sites[i];
    switch (call_sites::Retake(s, base)) {
      case call_sites::Retaken::kKept:
        break;
      case call_sites::Retaken::kChained:
        Log("%s: rva %llx was re-hooked by another plugin; chained", s.what,
            s.at - base);
        break;
      case call_sites::Retaken::kLost:
        Log("%s: rva %llx overwritten by another plugin; this site is not "
            "hooked",
            s.what, s.at - base);
        break;
    }
  }
}

// ---------------- hooks ----------------
uint32_t SoundId(const void* handle) {
  return handle ? *static_cast<const uint32_t*>(handle) : a2f::kNoSound;
}

// The reference's handle (made by the engine if it has none yet), or 0. ref
// is one an engine callback passed or a ScopedReference holds.
uint32_t HandleOf(uint8_t* ref) {
  uint32_t handle = 0;
  auto create = Engine<void (*)(uint32_t&, uint8_t*), kIdCreateHandle>();
  if (ref && create) create(handle, ref);
  return handle;
}

// The node is in the reference's 3D.
bool OfRef(uint8_t* node, const uint8_t* ref) {
  auto* object = reinterpret_cast<RE::NiAVObject*>(node);
  const auto* owner = reinterpret_cast<const RE::TESObjectREFR*>(ref);
  // GetUserData() walks parents itself. Read only this node's field so our
  // traversal remains bounded, including malformed cycles. These are the
  // SE/AE and VR offsets used by CommonLib's GetUserData implementation.
  const auto owner_offset = REL::Relocate(0xF8, 0xF8, 0x110);
  for (int i = 0; object && i < 32; object = object->parent, ++i) {
    if (REL::RelocateMember<RE::TESObjectREFR*>(object, owner_offset) == owner)
      return true;
  }
  return false;
}

// The actor's voice sound ID and whether it is playing: 1, 0, or -1 if
// unknown (no high process).
int ActorVoice(uint8_t* actor, uint32_t& sound) {
  auto is_playing = Engine<bool (*)(void*), kIdSoundIsPlaying>();
  uint8_t* proc = reinterpret_cast<uint8_t*>(reinterpret_cast<RE::Actor*>(actor)
                                                 ->GetActorRuntimeData()
                                                 .currentProcess);
  uint8_t* high = proc ? reinterpret_cast<uint8_t*>(
                             reinterpret_cast<RE::AIProcess*>(proc)->high)
                       : nullptr;
  uint8_t* handle =
      high ? reinterpret_cast<uint8_t*>(
                 &reinterpret_cast<RE::HighProcessData*>(high)->soundHandles[0])
           : nullptr;
  sound = SoundId(handle);
  return !handle || !is_playing ? -1 : is_playing(handle);
}

// Shared entry for engine speech and provider lines (external: a DBReV /
// DBVO notification, or a provider interface line); the one place a
// scene's or a suppressed actor is turned away. Returns BFA_kResultOK once
// the line is in playing, else the BFA_ResultCode.
uint32_t BeginSpeech(uint8_t* actor, const std::string& stem, uint32_t sound_id,
                     float dur, int emo_type, int emo_value, const Tune& tune,
                     uint64_t epoch, PlayerSpeechPtr external = {}) {
  if (!CurrentWorld(epoch)) return BFA_kResultNoFaceGenHead;
  if (!DialogueAllows(tune, HandleOf(actor), IsPlayer(actor)))
    return BFA_kResultNotInDialogue;
  if (!RaceAllowed(actor)) return BFA_kResultRaceDisabled;
  if (!NpcAllowed(actor)) return BFA_kResultDisabled;
  const uint32_t form_id = FormIdOf(actor);
  const bool provider = external && external->handle != 0;
  static std::atomic<int> logged;
  if (FaceExternallyOwned(actor)) {
    if (LogMore(logged)) Log("skip %s: its actor is in a scene", stem.c_str());
    return BFA_kResultInScene;
  }
  if (!external && ProviderClaims(form_id)) {
    if (LogMore(logged))
      Log("skip %s: a provider holds its actor's face", stem.c_str());
    return BFA_kResultTakenByOtherLine;
  }
  uint8_t* fg = reinterpret_cast<uint8_t*>(
      reinterpret_cast<RE::Actor*>(actor)->GetFaceGenAnimationData());
  if (!fg) return BFA_kResultNoFaceGenHead;
  skse::ResId r{};
  Engine<GenerateFromPathFn, kIdGenerateFromPath>()(&r,
                                                    (stem + ".lip").c_str());
  const uint64_t lip_key = skse::KeyOf(r);
  if (external && !provider) {
    // The engine hook may already have seen this same playback (the lip
    // resource tells, whatever the path's case and separators): its line
    // keeps its clock and pose and takes on the provider's cancellation.
    ScopedLock guard(lock);
    auto old = playing.find(fg);
    if (old != playing.end() && !old->second.external &&
        old->second.lip_key == lip_key) {
      old->second.external = std::move(external);
      return BFA_kResultOK;
    }
  }
  if (!external) {
    // A provider interface line owns the face while it plays: the engine's
    // own (idle chatter, a bark) waits for it, and is not generated.
    ScopedLock guard(lock, ScopedLock::kShared);
    auto old = playing.find(fg);
    if (old != playing.end() && ProviderLine(old->second) &&
        !old->second.external->stopped) {
      if (LogMore(logged))
        Log("skip %s: a provider's line is playing", stem.c_str());
      return BFA_kResultTakenByOtherLine;
    }
  }
  // without a handle the actor could not be told to be still there later
  const uint32_t handle = HandleOf(actor);
  if (!handle) {
    static std::atomic<int> no_handle;
    if (LogMore(no_handle))
      Log("skip %s: the actor has no handle", stem.c_str());
    return BFA_kResultNoFaceGenHead;
  }
  const bool female = IsFemale(actor);
  const a2f::Emotion* given = provider ? &external->emotion : nullptr;
  auto e = Request(stem, emo_type, emo_value, female, true, tune, handle, epoch,
                   IsPlayer(actor), given, provider ? external->audio : nullptr,
                   actor);
  if (!e) return BFA_kResultNotInDialogue;
  Playing p;
  p.entry = e;
  p.name = stem;
  p.lip_key = lip_key;
  p.actor = actor;
  p.actor_handle = handle;
  p.female = female;
  p.player = IsPlayer(actor);
  p.external = std::move(external);
  p.voice = {sound_id, Now(), dur};
  if (p.external) {
    // The provider's line started when it said so. Its sound is not known
    // (the actor may still be playing the line before), so none is followed.
    p.voice.requested_at = p.voice.at = p.external->started_at;
    p.voice.seen = true;
  } else {
    uint32_t sound;
    int playing_now = ActorVoice(actor, sound);
    p.voice.Observe(sound, playing_now, p.voice.requested_at);
  }
  ScopedLock guard(lock);
  if (!CurrentWorld(epoch)) return BFA_kResultNoFaceGenHead;
  if (provider) WatchEntry(*e, p.external);
  if (p.external && p.external->stopped) return BFA_kResultCancelled;
  auto old = playing.find(fg);
  if (old != playing.end()) {
    // a provider's line started meanwhile
    if (!p.external && ProviderLine(old->second) &&
        !old->second.external->stopped)
      return BFA_kResultTakenByOtherLine;
    // The other way around: the engine's line replaces DBReV's or DBVO's
    // and keeps its cancellation.
    if (!p.external && old->second.lip_key == lip_key && old->second.external &&
        !ProviderLine(old->second) && !old->second.external->stopped)
      p.external = old->second.external;
    LineLeft(fg, old->second, BFA_kSuperseded);
  }
  p.seen = play_clock;
  playing[fg] = std::move(p);
  Count();
  return BFA_kResultOK;
}

// Shared observation runs only once when a call-site hook enters the native
// entry hook. Native plugin calls have no call-site wrapper.
thread_local unsigned speech_depth = 0;
struct SpeechCallScope {
  SpeechCallScope() { ++speech_depth; }
  ~SpeechCallScope() { --speech_depth; }
};
SpeakFn native_speech_original = nullptr;

float ObserveSpeech(uint8_t* actor, const char* voice, void* handle,
                    uint64_t emo_type, uint64_t emo_value, float dur,
                    uint64_t epoch) {
  if (!CurrentWorld(epoch)) return dur;
  // dialogue first calls this without a sound, then again with it
  static std::atomic<int> logged;
  auto note = [&](const char* what) {
    if (voice && LogMore(logged))
      Log("speak %s: %s (duration %.2f, sound %u, emotion %d/%d)", voice, what,
          dur, SoundId(handle), static_cast<int>(emo_type),
          static_cast<int>(emo_value));
  };
  if (dur <= 0 && SoundId(handle) == a2f::kNoSound) {
    note("no sound yet");
    return dur;
  }
  const Tune tune = SnapshotTune();
  std::string stem;
  if (!tune.enabled || !voice || !ready || !SpeakerOk(actor) ||
      !VoiceStem(voice, stem) || (!tune.player && IsPlayer(actor))) {
    note("not for the plugin");
    return dur;
  }
  note("taken");
  BeginSpeech(actor, stem, SoundId(handle), dur, static_cast<int>(emo_type),
              static_cast<int>(emo_value), tune, epoch);
  return dur;
}

__declspec(noinline) float SpeakHook(uint8_t* actor, const char* voice,
                                     void* handle, uint64_t emo_type,
                                     uint64_t emo_value, uint64_t a6,
                                     uint64_t a7, uint64_t a8, uint64_t a9,
                                     uint64_t a10, uint64_t a11, uint64_t a12,
                                     uint64_t a13, uint64_t a14) {
  auto orig = SiteOrig<SpeakFn>(_ReturnAddress(), kIdSpeakVoiceLine);
  const uint64_t epoch = world_epoch;
  SpeechCallScope scope;
  float dur = orig(actor, voice, handle, emo_type, emo_value, a6, a7, a8, a9,
                   a10, a11, a12, a13, a14);
  return ObserveSpeech(actor, voice, handle, emo_type, emo_value, dur, epoch);
}

__declspec(noinline) float NativeSpeakHook(
    uint8_t* actor, const char* voice, void* handle, uint64_t emo_type,
    uint64_t emo_value, uint64_t a6, uint64_t a7, uint64_t a8, uint64_t a9,
    uint64_t a10, uint64_t a11, uint64_t a12, uint64_t a13, uint64_t a14) {
  const uint64_t epoch = world_epoch;
  const bool nested = speech_depth != 0;
  SpeechCallScope scope;
  float dur = native_speech_original(actor, voice, handle, emo_type, emo_value,
                                     a6, a7, a8, a9, a10, a11, a12, a13, a14);
  if (nested || !CurrentWorld(epoch) || !actor || !IsPlayer(actor) ||
      !EnabledFor(SnapshotTune(), true))
    return dur;
  // Native SpeakSound accepts paths relative to Data/Sound; DBVO 2 uses
  // that form. Keep the actual path passed to the engine unchanged.
  std::string stem, absolute;
  if (voice && *voice && !VoiceStem(voice, stem)) {
    absolute = std::string("Sound\\") + voice;
    voice = absolute.c_str();
  }
  return ObserveSpeech(actor, voice, handle, emo_type, emo_value, dur, epoch);
}

__declspec(noinline) uint8_t* ResponseHook(uint8_t* self, void* topic,
                                           void* info, uint8_t* speaker,
                                           void* response) {
  auto orig =
      SiteOrig<ResponseCtorFn>(_ReturnAddress(), kIdDialogueResponseCtor);
  const uint64_t epoch = world_epoch;
  uint8_t* r = orig(self, topic, info, speaker, response);
  if (!CurrentWorld(epoch)) return r;
  std::string stem;
  const Tune tune = SnapshotTune();
  const char* voice =
      r ? reinterpret_cast<RE::DialogueResponse*>(r)->voice.c_str() : nullptr;
  if (tune.enabled && tune.prefetch && voice && ready && SpeakerOk(speaker) &&
      DialogueAllows(tune, HandleOf(speaker), IsPlayer(speaker)) &&
      !ProviderClaims(FormIdOf(speaker)) && !FaceExternallyOwned(speaker) &&
      VoiceStem(voice, stem)) {
    const int emo = static_cast<int>(reinterpret_cast<RE::DialogueResponse*>(r)
                                         ->animFaceArchType.underlying());
    Request(
        stem, emo,
        static_cast<int>(reinterpret_cast<RE::DialogueResponse*>(r)->percent),
        IsFemale(speaker), false, tune, HandleOf(speaker), epoch,
        IsPlayer(speaker), nullptr, {}, speaker);
  }
  return r;
}

// Engine modifiers the face mesh takes over (brows, squints); the blinks
// (0, 1) stay with the engine.
constexpr int kModZeroed[] = {2, 3, 4, 5, 6, 7, 12, 13};
inline float Mix(float engine, float ours, float mix) {
  return mix >= 1.f ? ours : engine + (ours - engine) * mix;
}

// The mesh carries the mouth, brows and eyes: zeroes the final phoneme and
// modifier keyframes (but the blinks); mix < 1 blends from the engine's
// values. The expressions are Expressions'.
void PlayFrame(uint8_t* fg, float mix, float* tongue_phonemes,
               PhonemeRestore& preserved) {
  auto& phon = (reinterpret_cast<RE::BSFaceGenAnimationData*>(fg)->phoneme3);
  auto& mod = (reinterpret_cast<RE::BSFaceGenAnimationData*>(fg)->modifier3);
  preserved.Capture(phon, mix, tongue_phonemes);
  if (mod.values && mod.count >= 14) {
    for (int e : kModZeroed) mod.values[e] = Mix(mod.values[e], 0.f, mix);
    mod.isUpdated = true;
  }
}

std::atomic<int> logged_plays, logged_ends;

// Lock held: forgets the lines of heads no longer updated, now and then, but
// keep's. Such a line would never end, and a new head at the same address
// would play it and read its actor.
void SweepPlaying(uint8_t* keep) {
  static double swept = 0;
  if (play_clock - swept <= 1) return;
  swept = play_clock;
  for (auto it = playing.begin(); it != playing.end();) {
    if (it->first != keep && play_clock - it->second.seen > kPlayingStale) {
      if (LogMore(logged_ends))
        Log("drop %s: its head is gone", it->second.name.c_str());
      LineLeft(it->first, it->second, BFA_kAborted);
      it = playing.erase(it);
    } else {
      ++it;
    }
  }
  Count();
}

// Logs the end of a line: what it played, or why it never did.
void LogEnd(const Playing& p, State head_state, const char* why, float t) {
  if (!LogMore(logged_ends)) return;
  if (!p.logged) {
    Log("drop %s: never played, %s at %.2f s (line %s, head %s, lip %s)",
        p.name.c_str(), why, t,
        p.entry->state == State::kReady     ? "ready"
        : p.entry->state == State::kPending ? "still generating"
                                            : "failed",
        head_state == State::kReady     ? "mapped"
        : head_state == State::kPending ? "not mapped yet"
                                        : "not mappable",
        p.lip_seen ? "seen" : "not seen");
    return;
  }
  Log("end %s: at %.2f s (engine lip timer %.2f s), %u Direct frames, "
      "deformed %u, peak head %.3f mouth %.3f eyes %.3f; engine expression "
      "%d at %.2f (shown %.2f)",
      p.name.c_str(), p.last_t, p.lip_clock.last_lip, p.direct_frames,
      p.direct_adds, p.peak[0], p.peak[1], p.peak[2], p.engine_expr,
      p.engine_expr_v, p.shown);
}

// Lock held: the line's clock for this frame (0 while its sound is awaited)
// and, in why, the reason the line is over (null: it is not).
float LineClock(uint8_t* fg, Playing& p, double now, bool& lip,
                const char*& why, uint32_t sound, int playing_now) {
  const skse::ResId* act = skse::ActiveLine(fg);
  int voice_state = p.voice.Observe(sound, playing_now, now);
  bool hold = p.voice.Holding(voice_state);  // until the line's sound plays
  // A provider interface line's clock is the provider's (Start, Pause,
  // Seek); provider.cc stops it when the provider falls silent.
  const bool provider = ProviderLine(p);
  float t = hold ? 0
                 : static_cast<float>(provider ? p.external->Position(now)
                                               : p.voice.Elapsed(now));
  why = !provider && now - p.voice.requested_at > 120 ? "timed out" : nullptr;
  if (p.external && p.external->stopped)
    why = "the provider ended or replaced the line";
  // A provider's own line (BeginSpeech) has no sound to follow: it ends with
  // the provider's notification or audio length, never the delay after it.
  const bool followed = !p.external || p.voice.sound_id != a2f::kNoSound;
  const bool native_voice = !p.external && p.voice.sound_id != a2f::kNoSound;
  const bool audible = native_voice && p.voice.seen && voice_state == 1;
  // A stopped observation of the same sound may be transient. Once the
  // speech clock is established, retain its phase until the stop grace ends.
  // A different sound cannot sustain an expired lip resource.
  const bool same_voice =
      native_voice && p.voice.seen && sound == p.voice.sound_id;
  lip = act && skse::KeyOf(*act) == p.lip_key;
  if (lip) {  // engine lip clock
    t = p.lip_clock.Update(
        reinterpret_cast<RE::BSFaceGenAnimationData*>(fg)->phoneme1.timer, now,
        p.voice, hold, p.direct_frames != 0);
    if (audible || (same_voice && p.speech_clock.initialized))
      t = p.speech_clock.Update(t, now);
    p.lip_seen = true;
  } else if (p.lip_seen) {
    if (!act && same_voice && p.speech_clock.initialized)
      t = p.speech_clock.Continue(now);
    else
      why = "the engine ended or replaced its lip line";
  } else if (followed && p.voice.Ended(voice_state, now)) {
    why = "the voice ended";  // no .lip: wall clock while the voice plays
  }
  if (native_voice &&
      p.voice.EndedFor(voice_state, now, a2f::VoiceStart::kLipStop))
    why = "the voice ended";
  if (!why && p.external && p.external->duration > 0 &&
      t >= p.external->duration)
    why = "the player audio ended";
  if (!why && p.entry->state == State::kReady &&
      t > p.entry->line->frames / p.entry->line->fps + 0.3f)
    why = "the line is over";
  return t;
}

// The face's actor and its handle: its line's, else the one its Face was
// made for, else the conversation partner whose face it is. The caller
// resolves the handle (ScopedReference) with no lock held and uses the actor
// only if that gives the same one.
struct FaceActor {
  uint8_t* actor = nullptr;
  uint32_t handle = 0;
};
// Include retained faces even after dialogue closes: they still need their
// final restoration. Global counts alone admit every nearby NPC while one
// face is active. The idle case needs no lock or actor/settings lookup.
bool FaceNeedsUpdate(uint8_t* fg) {
  if (!fg) return false;
  if (fg == partner_fg) return true;
  if (!num_faces && !num_playing) return false;
  ScopedLock guard(lock, ScopedLock::kShared);
  return faces.count(fg) || playing.count(fg);
}

FaceActor FaceActorOf(uint8_t* fg) {
  ScopedLock guard(lock, ScopedLock::kShared);
  const auto line = playing.find(fg);
  if (line != playing.end())
    return {line->second.actor, line->second.actor_handle};
  const auto face = faces.find(fg);
  if (face != faces.end() && face->second.actor)
    return {face->second.actor, face->second.actor_handle};
  if (fg == partner_fg) return {partner_ref, partner_handle};
  return {};
}

// Plays the current line, unless its actor is in a scene (in_scene: the line
// is dropped). kFaceChanged requests a mesh reset; kKeepUpdating schedules
// another face update, including the final resets. actor: the face's, held
// by the caller, whose handle is actor_handle (null: gone); sound and
// playing_now: its voice (ActorVoice).
void ResetEndedFace(uint8_t* fg);
void ClearEndedKeyframes(uint8_t* fg);

int Apply(uint8_t* fg, const Tune& tune, bool in_scene, uint8_t* actor,
          uint32_t actor_handle, uint32_t sound, int playing_now) {
  double now = Now();
  ScopedLock guard(lock);
  Tick(now);
  SweepPlaying(fg);
  auto it = playing.find(fg);
  if (it == playing.end()) {
    return kNoUpdate;
  }
  Playing& p = it->second;
  const FaceTune settings = NpcFaceTune(tune, actor, p.female, p.player);
  p.seen = play_clock;
  // the line began after the caller looked its actor up: from the next update
  if (actor_handle != p.actor_handle) return kKeepUpdating;
  // the actor is gone and this is another's head at the same address
  if (actor != p.actor) {
    if (LogMore(logged_ends)) Log("drop %s: its actor is gone", p.name.c_str());
    LineLeft(fg, p, BFA_kAborted);
    playing.erase(it);
    Count();
    return kNoUpdate;
  }
  // The actor joined a scene: drop the line. Expressions releases its
  // Face immediately and requests an engine mesh rebuild.
  const bool provider = ProviderLine(p);
  if ((provider && !EnabledFor(tune, p.player)) || in_scene ||
      !RaceAllowed(actor) || !NpcAllowed(actor) ||
      !DialogueAllows(tune, actor_handle, p.player)) {
    if (LogMore(logged_ends))
      Log("drop %s: its actor is no longer eligible", p.name.c_str());
    const bool played = p.direct_frames != 0;
    if (provider) {
      uint32_t why = BFA_kResultNotInDialogue;
      if (!tune.enabled) {
        why = BFA_kResultDisabled;
      } else if (!EnabledFor(tune, p.player)) {
        why = BFA_kResultPlayerDisabled;
      } else if (in_scene) {
        why = BFA_kResultInScene;
      } else if (!NpcAllowed(actor)) {
        why = BFA_kResultDisabled;
      } else if (!RaceAllowed(actor)) {
        why = BFA_kResultRaceDisabled;
      }
      ProviderLineDeclined(p.external, why);
    }
    LineLeft(fg, p, BFA_kAborted);
    playing.erase(it);
    Count();
    return played ? kFaceChanged | kKeepUpdating : kNoUpdate;
  }
  bool lip;
  const char* why;
  const float t = LineClock(fg, p, now, lip, why, sound, playing_now);
  const bool done = why != nullptr;
  // A completed or interrupted spoken line fades to neutral using FaceFadeOut.
  // Clear native speech caches now; retain only our pose for the release fade.
  // A provider explicitly holding an utterance still owns its segment gaps.
  if (done && p.direct_frames && !ProviderHolder(FormIdOf(actor))) {
    LogEnd(p, LiveFace(fg) ? LiveFace(fg)->head_state : State::kPending, why, t);
    LineLeft(fg, p, BFA_kCompleted);
    playing.erase(it);
    Face* ending = LiveFace(fg);
    if (ending && settings.face_out > 0) {
      ending->neutral_started = now;
      ending->neutral_duration = settings.face_out;
      std::copy_n(ending->level, direct::kNumExpressions, ending->neutral_levels);
      ending->transition.FadeToNeutral(now, settings.face_out);
      ending->phonemes = {};
      std::fill_n(ending->src, direct::kNumExpressions, 0.f);
      std::fill_n(ending->written, direct::kNumExpressions, 0.f);
      ClearEndedKeyframes(fg);
    } else {
      ResetEndedFace(fg);
    }
    Count();
    return kFaceChanged | kKeepUpdating;
  }
  const bool enabled = EnabledFor(tune, p.player);
  Face* face = LiveFace(fg);
  const State head_state = face ? face->head_state : State::kPending;
  const bool generated = p.entry->state == State::kReady;
  // A provider's line whose head cannot be mapped, or whose inference
  // failed, is declined: the provider keeps its own lips for it.
  if (ProviderLine(p) && !done &&
      (head_state == State::kFailed || p.entry->state == State::kFailed)) {
    if (head_state == State::kFailed)
      ProviderLineDeclined(p.external, BFA_kResultHeadMapFailed);
    LogEnd(p, head_state, "declined", t);
    const bool played = p.direct_frames != 0;
    LineLeft(fg, p, BFA_kAborted);
    playing.erase(it);
    Count();
    return played ? kFaceChanged | kKeepUpdating : kNoUpdate;
  }
  // holding (t = 0) too: the engine's .lip already runs before the sound
  if (done || !generated || !enabled || head_state != State::kReady) {
    p.direct_frame = -1;
    if (!enabled) p.snap = true;
    bool tail = p.direct_tail > 0;
    if (tail) --p.direct_tail;
    // the mapping is found in the morph hook: keep it coming
    if (!done && enabled && head_state == State::kPending) tail = true;
    if (!done && generated && !enabled && !p.logged && tune.logging) {
      p.logged = true;
      Log("vanilla %s: %s is off", p.name.c_str(),
          tune.enabled ? "PlayerFacialAnimations" : "Enabled");
    }
    if (done && !tail) {
      LogEnd(p, head_state, why, t);
      LineLeft(fg, p, BFA_kCompleted);
      playing.erase(it);
      Count();
    }
    // disabled (A/B) or still generating: vanilla lip-sync meanwhile
    return tail ? kKeepUpdating : kNoUpdate;
  }
  if (p.ready_at < 0) {
    p.ready_at = t;
    if (t > kLateStart) ++stats.late;
  }
  p.late = p.ready_at > kLateStart;
  const float mix =
      p.late ? std::clamp((t - p.ready_at) / std::max(settings.blend_in, 1e-3f),
                          0.f, 1.f)
             : 1.f;
  p.last_t = t;
  ++p.direct_frames;
  p.direct_frame =
      std::fmax(0.f, (t + settings.time_offset) * p.entry->line->fps);
  p.direct_tail = kDirectTail;
  const bool first = !p.logged;
  p.logged = true;
  const std::string name = first ? p.name : std::string();
  const float ready_at = p.ready_at;
  const uint32_t frames = p.entry->line->frames;
  if (ProviderLine(p)) ProviderLineStarted(p.external);
  PlayFrame(fg, mix, p.tongue_phonemes, face->phonemes);
  guard.Unlock();
  if (first && LogMore(logged_plays))
    Log("play %s: %u frames, %s clock, ready at %.2f s", name.c_str(), frames,
        lip ? "lip" : "wall", ready_at);
  return kFaceChanged | kKeepUpdating;
}

// Drops a face (lock held).
void DropFace(uint8_t* fg) {
  auto it = faces.find(fg);
  if (it != faces.end()) ReleaseLeaving(it->second);
  if (fg == partner_fg) {
    partner_fg = nullptr;
    partner_ref = nullptr;
    partner_handle = 0;
  }
  faces.erase(fg);
  Count();
}

// Both locks held. The neutral release must not restore cached dialogue
// phonemes/expressions. Once the fade ends, DropFace releases later writers.
void ClearEndedKeyframes(uint8_t* fg) {
  auto& data = *reinterpret_cast<RE::BSFaceGenAnimationData*>(fg);
  for (auto* key : {&data.expressionKeyFrame, &data.expression3,
                    &data.modifierKeyFrame, &data.modifier1, &data.modifier3,
                    &data.phenomeKeyFrame, &data.phoneme1, &data.phoneme3}) {
    if (key->values) std::fill_n(key->values, key->count, 0.f);
    key->isUpdated = true;
  }
  data.unk218 = 1;
}

void ResetEndedFace(uint8_t* fg) {
  ClearEndedKeyframes(fg);
  DropFace(fg);
}

// Face's lock held: puts the engine's expression values (src) back into the
// keyframe the plugin overwrote. With written, only while the keyframe is
// still as the plugin left it (else the engine, a scene or a new head at
// this address wrote it since). Returns whether it wrote.
bool RestoreKeyframe(skse::KeyframeMultiple& ex, const float* src,
                     const float* written) {
  if (!ex.values || ex.count < direct::kNumExpressions) return false;
  for (int i = 0; written && i < direct::kNumExpressions; ++i)
    if (direct::PlaysExpression(i) && ex.values[i] != written[i]) return false;
  for (int i = 0; i < direct::kNumExpressions; ++i)
    if (direct::PlaysExpression(i)) ex.values[i] = src[i];
  return true;
}

// Both locks held: restores the engine keyframes and drops the face.
// Returns whether there was one.
bool RestoreAndDropFace(uint8_t* fg) {
  auto face = faces.find(fg);
  if (face == faces.end()) return false;
  face->second.phonemes.Restore(
      (reinterpret_cast<RE::BSFaceGenAnimationData*>(fg)->phoneme3));
  auto& ex = (reinterpret_cast<RE::BSFaceGenAnimationData*>(fg)->expression3);
  if (RestoreKeyframe(ex, face->second.src, face->second.written))
    ex.isUpdated = true;
  DropFace(fg);
  return true;
}

// Before the engine update, restore its speech and expression keyframes.
// Phonemes and parked expressions are restored only if still as we left them.
void RestoreFaceKeyframes(uint8_t* fg, uint64_t epoch) {
  float src[direct::kNumExpressions], written[direct::kNumExpressions];
  PhonemeRestore phonemes;
  ScopedLock guard(lock);
  auto it = faces.find(fg);
  const bool found = it != faces.end();
  const bool parked = found && it->second.parked;
  if (found) {
    phonemes = it->second.phonemes;
    it->second.phonemes = {};
    std::copy_n(it->second.src, direct::kNumExpressions, src);
    std::copy_n(it->second.written, direct::kNumExpressions, written);
  }
  guard.Unlock();
  if (!found) return;
  // the face's lock is waited for without the plugin's; a load may begin
  // meanwhile, and then the keyframe is no longer written
  skse::ScopedFaceLock face_guard(
      reinterpret_cast<RE::BSFaceGenAnimationData*>(fg)->lock);
  if (!CurrentWorld(epoch)) return;
  phonemes.Restore(
      (reinterpret_cast<RE::BSFaceGenAnimationData*>(fg)->phoneme3));
  RestoreKeyframe(
      (reinterpret_cast<RE::BSFaceGenAnimationData*>(fg)->expression3), src,
      parked ? written : nullptr);
}

// Lock held: the face of this animation data; made when wanted and missing,
// with the longest idle of a full set sent off (it fades out and drops
// itself).
Face* FaceFor(uint8_t* fg, bool wanted, double clock) {
  auto it = faces.find(fg);
  if (it != faces.end()) return &it->second;
  if (!wanted) return nullptr;
  size_t live = 0;
  Face* idle = nullptr;
  for (auto& [key, f] : faces) {
    if (f.parked) continue;
    ++live;
    if (f.until > clock && (!idle || f.until < idle->until)) idle = &f;
  }
  if (live >= kMaxFaces && idle) idle->until = 0;
  Face* f = &faces[fg];
  Count();
  return f;
}

// Lock held: now and then, the faces no longer updated (unloaded), but
// keep's. One that left the plugin's values in the engine's keyframe is
// parked until it is updated again (RestoreFaceKeyframes) or kFaceParked
// passes; the others are forgotten.
void SweepFaces(uint8_t* keep, double clock) {
  static double swept = 0;
  if (clock - swept <= 1) return;
  swept = clock;
  for (auto it = faces.begin(); it != faces.end();) {
    Face& f = it->second;
    uint8_t* fg = it->first;
    ++it;  // DropFace erases
    if (fg == keep || clock - f.seen <= (f.parked ? kFaceParked : kFaceStale))
      continue;
    if (f.parked ||
        (!f.phonemes.values && std::equal(std::begin(f.src), std::end(f.src),
                                          std::begin(f.written)))) {
      DropFace(fg);
      continue;
    }
    Face parked;
    parked.phonemes = f.phonemes;
    std::copy_n(f.src, direct::kNumExpressions, parked.src);
    std::copy_n(f.written, direct::kNumExpressions, parked.written);
    parked.parked = true;
    parked.seen = clock;
    ReleaseLeaving(f);
    f = std::move(parked);
  }
}

// Puts a face's expression levels (lv, of the engine's scale) where they
// play: on a mapped head on the meshes (Face::level, for AddDirect) with the
// keyframes zeroed; elsewhere (head not mappable, or still mapping) on the
// keyframes (ex; null: the head has none), at the regions' mean scale.
// Returns like Apply.
int ShowExpressions(Face& f, const float* lv, skse::KeyframeMultiple* ex) {
  const bool mapped = f.head_state == State::kReady;
  int updates = kNoUpdate;
  for (int i = 0; i < direct::kNumExpressions; ++i) {
    if (!direct::PlaysExpression(i)) continue;
    const float level = mapped ? f.ownership_blend * lv[i] : 0;
    if (std::fabs(level - f.level[i]) > 1e-4f)
      updates = kFaceChanged;  // re-apply the meshes
    f.level[i] = level;
    f.written[i] =
        (1 - f.ownership_blend) * f.shown[i] +
        (mapped ? 0 : f.ownership_blend * lv[i] * MeanExpressionScale());
    if (ex) ex->values[i] = f.written[i];
  }
  if (ex) ex->isUpdated = true;
  // the mapping is found in the morph hook: keep it coming
  if (f.head_state == State::kPending) updates = kFaceChanged | kKeepUpdating;
  return updates;
}

// The face of anyone speaking a line and of the player's conversation
// partner, after Apply (face lock held): keeps the Face, and plays its
// emotion expressions (direct::PlaysExpression: dialogue, mood, combat)
// itself, scaled per region; while a line plays they are all faded out (the
// network carries the emotion). Fades in and out (Face::ownership_blend).
// A face whose actor is in a scene (in_scene) goes back to the engine.
// Returns like Apply.
int Expressions(uint8_t* fg, float dt, const Tune& tune, bool in_scene,
                uint8_t* actor, uint32_t actor_handle) {
  // Exclusive scenes own the face immediately, including paused frames.
  // The original update has already produced their keyframes. Do not fade
  // our expressions or speech pose over those values.
  if (!DialogueAllows(tune, actor_handle, actor && IsPlayer(actor)) ||
      in_scene || !RaceAllowed(actor) || !NpcAllowed(actor)) {
    ScopedLock guard(lock);
    bool changed = faces.count(fg) != 0;
    auto line = playing.find(fg);
    if (line != playing.end()) {
      LineLeft(fg, line->second, BFA_kAborted);
      playing.erase(line);
      changed = true;
    }
    DropFace(fg);
    return changed ? kFaceChanged : kNoUpdate;
  }
  // Also for callers other than the hook (which checked FaceNeedsUpdate): a
  // face of a world that is gone must not be read.
  if (fg != partner_fg && !num_faces && !num_playing) return kNoUpdate;
  auto& ex = (reinterpret_cast<RE::BSFaceGenAnimationData*>(fg)->expression3);
  const bool has_ex = ex.values && ex.count >= direct::kNumExpressions;
  const double now = Now();
  ScopedLock guard(lock);
  const double clock = Tick(now);
  const bool partner = fg == partner_fg && actor && actor == partner_ref &&
                       actor_handle == partner_handle;
  auto pit = playing.find(fg);
  if (pit != playing.end() && pit->second.entry->state == State::kFailed) {
    // The original update has already restored the engine keyframes.
    // Discard the previous speech pose, including its release fade.
    const bool changed = faces.count(fg) != 0;
    DropFace(fg);
    return changed ? kFaceChanged : kNoUpdate;
  }
  const bool in_line = pit != playing.end();
  auto existing = faces.find(fg);
  const bool player = in_line
                          ? pit->second.player
                          : existing != faces.end() && existing->second.player;
  const bool enabled = EnabledFor(tune, player);
  const bool wanted = enabled && !in_scene && (in_line || (partner && has_ex));
  Face* found = FaceFor(fg, wanted, clock);
  if (!found) {
    return kNoUpdate;
  }
  Face& f = *found;
  SweepFaces(fg, clock);
  if (f.parked) f = Face();  // back: its keyframe was restored
  f.seen = clock;
  if (f.neutral_started >= 0) {
    if (in_line || (actor && ProviderHolder(FormIdOf(actor)))) {
      f.neutral_started = -1;  // new speech takes over the current pose
    } else {
      const float elapsed = static_cast<float>(now - f.neutral_started);
      if (elapsed >= f.neutral_duration) {
        ResetEndedFace(fg);
      } else {
        ClearEndedKeyframes(fg);
        const float weight = 1 - a2f::Smoothstep(elapsed / f.neutral_duration);
        for (int i = 0; i < direct::kNumExpressions; ++i)
          f.level[i] = f.neutral_levels[i] * weight;
      }
      return kFaceChanged | kKeepUpdating;
    }
  }
  // Pass native gradual changes through; bound abrupt overrides without an
  // asymptotic tail that keeps rebuilding an otherwise idle face for seconds.
  const float step = std::max(dt, 0.f) / kSrcFollow;
  bool following = false;
  for (int i = 0; i < direct::kNumExpressions; ++i) {
    f.src[i] = has_ex && direct::PlaysExpression(i) ? ex.values[i] : 0.f;
    const float before = f.shown[i];
    const float delta = f.src[i] - before;
    f.shown[i] = f.shown_set && std::fabs(delta) > step
                     ? before + std::copysign(step, delta)
                     : f.src[i];
    following |= before != f.shown[i] || f.shown[i] != f.src[i];
  }
  f.shown_set = true;
  if (actor && (actor != f.actor || !f.actor_handle)) {
    f.actor = actor;
    f.player = IsPlayer(actor);
    f.female = IsFemale(actor);
    f.actor_handle = actor_handle;
  }
  const bool speaking = in_line && enabled && pit->second.direct_frame >= 0;
  // the utterance holding the face between its segments (ProviderHolder)
  const uint64_t holder =
      enabled && !in_scene && actor ? ProviderHolder(FormIdOf(actor)) : 0;
  const bool held = !speaking && holder != 0;
  // While inference runs the head is prepared (the Face is kept) but the
  // expressions go back to the engine's, fading like after a line, until a
  // generated animation is ready -- unless an utterance holds the face, or
  // the face is still had from the line before (kFaceGrace): the next line
  // of a speaker then starts without a dip to the engine's expressions.
  const bool pending = in_line && pit->second.entry->state == State::kPending;
  // whose the face is now: the partner's, a line's, a holding utterance's
  const bool had = (wanted && (partner || !pending)) || holder;
  if (had) {
    f.until = clock + kFaceGrace;
  } else if (in_scene) {
    f.until = 0;  // back to the engine's
  }
  const bool owns_expressions = enabled && clock < f.until;
  f.ownership_blend = std::clamp(
      f.ownership_blend + (owns_expressions ? dt / kOwnIn : -dt / kOwnOut), 0.f,
      1.f);
  // the levels to show: the engine's, faded out while a line plays (the
  // network alone shapes the speaking face; the game's emotion reaches it
  // as the network's input) and back in after it
  const FaceTune settings = NpcFaceTune(tune, actor, f.female, f.player);
  const float previous_speech_blend = f.speech_blend;
  if (!held)
    f.speech_blend = std::clamp(
        f.speech_blend + (speaking ? dt / std::max(settings.face_in, 1e-3f)
                                   : -dt / std::max(settings.face_out, 1e-3f)),
        0.f, 1.f);
  float lv[direct::kNumExpressions];
  for (int i = 0; i < direct::kNumExpressions; ++i)
    lv[i] = f.shown[i] * (1 - f.speech_blend);
  if (has_ex && speaking && pit->second.direct_frame >= 0) {  // log
    Playing& p = pit->second;
    for (uint32_t i = 0; i < ex.count; ++i)
      if (ex.values[i] > p.engine_expr_v) {
        p.engine_expr_v = ex.values[i];
        p.engine_expr = static_cast<int>(i);
      }
    constexpr uint32_t kPastFade = 15;  // frames
    for (int i = 0; p.direct_frames > kPastFade && i < direct::kNumExpressions;
         ++i)
      p.shown = std::fmax(p.shown, f.ownership_blend * lv[i]);
  }
  int updates = ShowExpressions(f, lv, has_ex ? &ex : nullptr);
  // the speech pose is still easing out (AddDirect)
  const bool easing = f.transition.moving() && !f.transition.stale(now);
  // CHIM's audio does not keep the engine's native lip animation ticking.
  // Request the next update throughout every fade, including its final mesh
  // reset, or a quiet face can freeze mid-blend then jump on its next update.
  if (easing || following || previous_speech_blend != f.speech_blend ||
      (!held && f.speech_blend > 0 && f.speech_blend < 1))
    updates |= kFaceChanged | kKeepUpdating;
  // The lines that left are told once the face is handed back: another
  // line or utterance has it, or the engine's expressions are back.
  if (!f.leaving.empty()) {
    const uint64_t owner = in_line && ProviderLine(pit->second)
                               ? pit->second.external->utterance_id
                               : holder;
    if (in_line || holder) {
      ReleaseLeaving(f, owner);
    } else if (f.speech_blend <= 0 && !easing) {
      ReleaseLeaving(f);
    }
  }
  if (f.ownership_blend > 0 && f.ownership_blend < 1)
    updates |= kFaceChanged | kKeepUpdating;  // fading
  // In the grace after its line or conversation a quiet face is still
  // updated, so that its hand-back starts on time.
  if (owns_expressions && !had) updates |= kKeepUpdating;
  if (!owns_expressions && f.ownership_blend <= 0) {
    if (easing) {
      updates |= kKeepUpdating;  // dropped once the pose has eased out
    } else if (!pending) {
      DropFace(fg);  // back to the engine's
    }
  }
  return updates;
}

using FaceNodeUpdateFn = void (*)(uint8_t*, NiUpdateData*, uint32_t);
FaceNodeUpdateFn face_node_update_original = nullptr;

void FaceNodeUpdateHook(uint8_t* node, NiUpdateData* data, uint32_t flags) {
  auto* fg =
      reinterpret_cast<uint8_t*>(reinterpret_cast<RE::BSFaceGenNiNode*>(node)
                                     ->GetRuntimeData()
                                     .animationData.get());
  if (!FaceNeedsUpdate(fg)) {
    face_node_update_original(node, data, flags);
    return;
  }
  const ScopedFaceUpdate update(fg, data->time);
  face_node_update_original(node, data, flags);
}

bool InstallFaceNodeUpdateHook() {
  const auto address = Addr(kIdFaceNodeVtable);
  if (!address) return false;
  face_node_update_original = reinterpret_cast<FaceNodeUpdateFn>(
      REL::Relocation<uintptr_t>(address).write_vfunc(
          kVfUpdateDownwardPass.Get(), FaceNodeUpdateHook));
  return true;
}

// Runs where the head's morphs are applied (ID 26998), around the engine's
// recompute of the face keyframes (ID 26598) -- writing earlier would be
// overwritten. Returns "changed" so the morphs are re-applied.
__declspec(noinline) bool AnimUpdateHook(uint8_t* fg, float dt, uint64_t flag) {
  auto orig = SiteOrig<AnimUpdateFn>(_ReturnAddress(), kIdFaceAnimUpdate);
  if (!FaceNeedsUpdate(fg)) return orig(fg, dt, flag);
  // Skyrim uses the global frame delta even for zero-time pose refreshes
  // (for example Fluid Motion's second pass). Still recompute the keyframes.
  dt = ScopedFaceUpdate::Delta(fg, dt);
  const uint64_t epoch = world_epoch;
  if (num_faces && ready && CurrentWorld(epoch))
    RestoreFaceKeyframes(fg, epoch);
  bool changed = orig(fg, dt, flag);
  if (ready && CurrentWorld(epoch)) {
    // The actor is held until the face's lock is released: letting go of
    // the last reference destroys it.
    const FaceActor known = FaceActorOf(fg);
    ScopedReference reference(known.handle);
    uint8_t* actor = reference.get() == known.actor ? known.actor : nullptr;
    const bool in_scene = FaceExternallyOwned(actor);
    uint32_t sound = a2f::kNoSound;
    const int playing_now =
        actor && num_playing ? ActorVoice(actor, sound) : -1;
    const Tune tune = SnapshotTune();
    int updates = kNoUpdate;
    {
      skse::ScopedFaceLock face_guard(
          reinterpret_cast<RE::BSFaceGenAnimationData*>(fg)->lock);
      if (CurrentWorld(epoch)) {
        if (num_playing)
          updates = Apply(fg, tune, in_scene, actor, known.handle, sound,
                          playing_now);
        updates |= Expressions(fg, dt, tune, in_scene, actor, known.handle);
      }
    }
    changed |= updates != 0;
    // "changed" makes the node reset its meshes; keep the update coming
    if (updates & kKeepUpdating)
      reinterpret_cast<RE::BSFaceGenAnimationData*>(fg)->unk218 = 1;
  }
  return changed;
}

// ---------------- Direct: vertex deltas on the dynamic face meshes ---------
// The player's conversation partner's face, once its node comes by: its
// expressions are played.
void TrackPartner(uint8_t* node, uint8_t* fg) {
  const uint32_t handle = PartnerHandle();
  ScopedReference reference(handle);
  uint8_t* ref = reference.get();
  if (!SpeakerOk(ref)) ref = nullptr;
  const bool ended = !ref && partner_ref;
  const bool found =
      ref && (ref != partner_ref || fg != partner_fg) && OfRef(node, ref);
  if (!ended && !found) return;
  ScopedLock guard(lock);
  partner_fg = found ? fg : nullptr;
  partner_ref = ref;
  partner_handle = ref ? handle : 0;
  if (found) partner_female = IsFemale(ref);
}

// Adds the deltas (by head.shapes) to the node's dynamic face shapes, noting
// each part's peak. Returns whether any shape received deltas.
bool AddDeltas(uint8_t* node, const direct::Head& head,
               const std::vector<std::vector<float>>& deltas, float* peak) {
  bool added = false;
  ForFaceShapes(node, [&](uint8_t* shape, const char* name, direct::Part part,
                          bool, uint32_t nv, const float*) {
    size_t si = 0;
    while (si < head.shapes.size() &&
           (head.shapes[si].nv != nv || head.shapes[si].name != name))
      ++si;
    if (si == head.shapes.size()) return;
    const auto& d = deltas[si];
    // pinned pieces count with the head's followers
    const direct::Part mapped_part = head.shapes[si].part;
    const int pi = mapped_part == direct::Part::kPinned
                       ? 0
                       : static_cast<int>(mapped_part);
    thread_local std::vector<float> xyz;
    xyz.resize(nv * 3);
    float* data = Engine<LockDynamicFn, kIdLockDynamic>()(shape);
    if (!data) {
      Engine<UnlockDynamicFn, kIdUnlockDynamic>()(shape);
      return;
    }
    added = true;
    for (uint32_t v = 0; v < nv; ++v)
      for (int c = 0; c < 3; ++c) {
        const size_t j = v * 3 + c;
        const float dv = d[j];
        peak[pi] = std::fmax(peak[pi], std::fabs(dv));
        xyz[j] = data[v * 4 + c] + dv;
      }
    Engine<UpdateDynamicFn, kIdUpdateDynamic>()(shape, xyz.data(), false);
    Engine<UnlockDynamicFn, kIdUnlockDynamic>()(shape);
  });
  return added;
}

// Apply asset calibration at playback, independently of the model's sex. In
// particular, v2.3 uses Mark for both sexes and may share a cached mapping.
void SpeechScales(const direct::Head& head, const FaceTune& settings,
                  bool female, float* scales) {
  for (int r = 0; r < direct::kNumRegions; ++r)
    scales[r] = settings.direct_scale[r] * head.auto_scale[r];
  if (config.expressive_faces[female])
    scales[direct::kMouth] *= kExpressiveMouthScale;
}

// Lock held: the line the face is playing, or null.
std::shared_ptr<Entry> PlayingEntry(uint8_t* fg) {
  const auto it = playing.find(fg);
  return it == playing.end() ? nullptr : it->second.entry;
}

// Finds the face's mapping, then (after the engine rebuilt the shapes from
// the base and the morphs: reset) adds the line's Direct frame and the
// expressions to every dynamic face shape of the node. The line's model
// selects the mapping; the face is left alone if its line changes meanwhile.
void AddDirect(uint8_t* node, bool reset) {
  uint8_t* fg =
      reinterpret_cast<uint8_t*>(reinterpret_cast<RE::BSFaceGenNiNode*>(node)
                                     ->GetRuntimeData()
                                     .animationData.get());
  if (!fg) return;
  // Morph calls can arrive without an intervening animation update. Query
  // the retained actor outside our locks before reading or applying a pose.
  TrackPartner(node, fg);
  if (!FaceNeedsUpdate(fg)) return;
  const FaceActor known = FaceActorOf(fg);
  ScopedReference reference(known.handle);
  // The retained actor, when it is still the face's; null otherwise.
  uint8_t* actor = reference.get() == known.actor ? reference.get() : nullptr;
  const Tune base_tune = SnapshotTune();
  if (!DialogueAllows(base_tune, known.handle, actor && IsPlayer(actor)) ||
      (actor &&
       (FaceExternallyOwned(actor) || !RaceAllowed(actor) || !NpcAllowed(actor)))) {
    {
      ScopedLock guard(lock, ScopedLock::kShared);
      if (!faces.count(fg) && !playing.count(fg)) return;
    }
    // Request an engine rebuild on the next update if this call did not
    // reset the mesh; never add even the old pose's fade-out to a scene.
    skse::ScopedFaceLock face_guard(
        reinterpret_cast<RE::BSFaceGenAnimationData*>(fg)->lock);
    ScopedLock guard(lock);
    RestoreAndDropFace(fg);
    auto line = playing.find(fg);
    if (line != playing.end()) {
      LineLeft(fg, line->second, BFA_kAborted);
      playing.erase(line);
      Count();
    }
    reinterpret_cast<RE::BSFaceGenAnimationData*>(fg)->unk218 = 1;
    return;
  }
  const Tune tune = NpcSharedTune(base_tune, actor);
  if (FailedSpeech(fg)) {
    skse::ScopedFaceLock face_guard(
        reinterpret_cast<RE::BSFaceGenAnimationData*>(fg)->lock);
    ScopedLock guard(lock);
    // unless the line changed while waiting for the face lock
    auto line = playing.find(fg);
    if (line != playing.end() && line->second.entry->state == State::kFailed &&
        RestoreAndDropFace(fg))
      reinterpret_cast<RE::BSFaceGenAnimationData*>(fg)->unk218 = 1;
    return;
  }
  bool enabled = false;
  // the face's sex: its line's speaker's or the partner's, else as last known
  bool female = false;
  bool player = false;
  auto selection = SnapshotModel();
  std::shared_ptr<Entry> expected_entry;
  {
    ScopedLock guard(lock);
    Face* face = LiveFace(fg);
    if (!face) return;
    auto it = playing.find(fg);
    if (it != playing.end())
      face->female = it->second.female;
    else if (fg == partner_fg)
      face->female = partner_female;
    female = face->female;
    player = it != playing.end() ? it->second.player : face->player;
    expected_entry = PlayingEntry(fg);
    if (expected_entry && expected_entry->model_state)
      selection = expected_entry->model_state;
  }
  const FaceTune settings = NpcFaceTune(tune, actor, female, player);
  const std::string key = HeadKey(node, female, selection->mapping);
  std::shared_ptr<const a2f::Line> line;
  std::shared_ptr<const direct::Head> head;
  float frame = -1;
  float levels[direct::kNumExpressions];
  float tongue_phonemes[direct::kNumPhonemes] = {};
  bool late = false;
  State state;
  {
    ScopedLock guard(lock);
    Face* face = LiveFace(fg);
    if (!face || PlayingEntry(fg) != expected_entry) return;
    enabled = EnabledFor(tune, face->player);
    if (face->head_key != key) {
      face->head_key = key;
      face->head_state = State::kPending;
      face->head.reset();
      // Geometry changes invalidate the mapping, not the model-space pose
      // or expression ownership. Never apply the old mapping to new vertices.
    }
    state = face->head_state;
    head = face->head;
    std::copy_n(face->level, direct::kNumExpressions, levels);
  }
  if (state == State::kPending) {
    state = FindHead(node, key, female, head, selection->mapping);
    if (state != State::kPending) {
      ScopedLock guard(lock);
      Face* face = LiveFace(fg);
      if (!face || face->head_key != key || PlayingEntry(fg) != expected_entry)
        return;
      face->head_state = state;
      face->head = head;
    }
  }
  if (!reset || state != State::kReady || !head) return;
  {
    ScopedLock guard(lock);
    Face* face = LiveFace(fg);
    if (!face || face->head_key != key || PlayingEntry(fg) != expected_entry)
      return;
    // A cached replacement can become ready in this very call. Sample its
    // current speech now rather than submitting a spurious end-of-line pose.
    auto it = playing.find(fg);
    if (enabled && it != playing.end() && it->second.direct_frame >= 0) {
      line = it->second.entry->line;
      frame = it->second.direct_frame;
      late = it->second.late;
      std::copy_n(it->second.tongue_phonemes, direct::kNumPhonemes,
                  tongue_phonemes);
    }
  }
  // at its last frame the line is over: the pose eases out from there
  if (line && frame >= static_cast<float>(line->frames - 1)) line.reset();
  thread_local std::vector<float> coefficients;
  direct::Coefficients(*head, line.get(), frame, coefficients);
  if (!line) std::fill_n(tongue_phonemes, direct::kNumPhonemes, 0.f);
  // Fade authored tongue weights with the same line lifetime as the face.
  const size_t tongue_start = coefficients.size();
  coefficients.insert(coefficients.end(), tongue_phonemes,
                      tongue_phonemes + direct::kNumPhonemes);
  // Carry joy through the same transition as speech, including line endings.
  const size_t joy_index = coefficients.size();
  coefficients.push_back(
      line ? std::clamp(line->emotion_mean[a2f::kJoy], 0.f, 1.f) : 0.f);
  bool speaking = false;
  {
    ScopedLock guard(lock);
    Face* face = LiveFace(fg);
    if (!face || face->head_key != key || PlayingEntry(fg) != expected_entry)
      return;
    auto it = playing.find(fg);
    const bool snap = line && it != playing.end() && it->second.snap;
    if (snap) it->second.snap = false;
    const int model_face = selection->mapping->FaceIndex(female);
    if (face->pose_model != selection->mapping ||
        face->pose_face != model_face) {
      face->transition = a2f::PoseTransition();
      face->pose_model = selection->mapping;
      face->pose_face = model_face;
    }
    const float fade = !line  ? settings.face_out
                       : late ? settings.blend_in
                              : settings.face_in;
    speaking =
        face->transition.Blend(line, Now(), fade, enabled, snap, coefficients);
  }
  // a head made for smaller facial motion takes the speech scaled down
  float scales[direct::kNumRegions];
  SpeechScales(*head, settings, female, scales);
  thread_local std::vector<std::vector<float>> deltas;
  if (!direct::Evaluate(*head, speaking ? coefficients.data() : nullptr, scales,
                        levels, coefficients.data() + tongue_start,
                        tune.disable_tongue ? direct::TongueMode::kModel
                                            : direct::TongueMode::kAuthored,
                        deltas, settings.upper_lip, settings.lower_lip,
                        settings.smile_eyes, coefficients[joy_index],
                        LimitLowerLip(actor)))
    return;
  float peak[3] = {};
  const bool added = AddDeltas(node, *head, deltas, peak);
  ScopedLock guard(lock);
  auto it = playing.find(fg);
  if (added && it != playing.end()) {
    ++it->second.direct_adds;
    for (int i = 0; i < 3; ++i)
      it->second.peak[i] = std::fmax(it->second.peak[i], peak[i]);
  }
}

// Offsets the look morphs of the NPC speaking to the player while the
// original morph routine consumes them, then puts them back: offsets never
// accumulate and nothing has to be restored when the actor goes away.
uint64_t GazeMorphs(ApplyMorphsFn orig, uint8_t* node, uint8_t force,
                    bool& reset, uint64_t epoch) {
  auto* fg =
      reinterpret_cast<uint8_t*>(reinterpret_cast<RE::BSFaceGenNiNode*>(node)
                                     ->GetRuntimeData()
                                     .animationData.get());
  if (!fg || !PartnerHandle()) return orig(node, force);
  Tune tune = SnapshotTune();
  if (!CurrentWorld(epoch) || !ready || !tune.enabled || skse::IsVr())
    return orig(node, force);
  // Globally off, and no NPC override can turn it on: skip the handle work.
  const bool gaze_off = !tune.conversation_gaze || tune.gaze_strength <= 0;
  if (gaze_off && !NpcOverridesActive()) return orig(node, force);
  const FaceActor known = FaceActorOf(fg);
  if (!known.handle || known.handle != PartnerHandle())
    return orig(node, force);
  ScopedReference actor(known.handle);
  if (!actor.get() || actor.get() != known.actor || IsPlayer(actor.get()) ||
      FaceExternallyOwned(actor.get()) || !NpcAllowed(actor.get()))
    return orig(node, force);
  tune = NpcSharedTune(std::move(tune), actor.get());
  if (!tune.conversation_gaze || tune.gaze_strength <= 0)
    return orig(node, force);
  // Use the priority-resolved target: scripts/packages can override dialogue.
  auto* process =
      reinterpret_cast<uint8_t*>(reinterpret_cast<RE::Actor*>(actor.get())
                                     ->GetActorRuntimeData()
                                     .currentProcess);
  auto get_target =
      Engine<uint32_t* (*)(uint8_t*, uint32_t*), kIdHeadtrackTarget>();
  if (!process || !get_target) return orig(node, force);
  uint32_t target_handle = 0;
  get_target(process, &target_handle);
  ScopedReference target(target_handle);
  if (!target.get() || !IsPlayer(target.get())) return orig(node, force);
  skse::ScopedFaceLock face_guard(
      reinterpret_cast<RE::BSFaceGenAnimationData*>(fg)->lock);
  float offset[2] = {};
  {
    ScopedLock guard(lock, ScopedLock::kShared);
    auto it = playing.find(fg);
    if (CurrentWorld(epoch) && it != playing.end()) {
      const Playing& p = it->second;
      if (p.actor == actor.get() && p.actor_handle == known.handle &&
          !p.player && p.entry->state == State::kReady && p.entry->line &&
          p.direct_frame >= 0 && p.ready_at >= 0 &&
          !(p.external && p.external->stopped)) {
        const a2f::Line& line = *p.entry->line;
        // The sampled frame includes playback sync; fade in on that same
        // clock so late starts do not jump when the sync slider is nonzero.
        const float ready_frame =
            (p.ready_at +
             NpcFaceTune(tune, actor.get(), p.female, false).time_offset) *
            line.fps;
        a2f::GazeOffset(line, p.direct_frame, ready_frame, tune.gaze_strength,
                        offset);
      }
    }
  }
  auto& mods = (reinterpret_cast<RE::BSFaceGenAnimationData*>(fg)->modifier3);
  constexpr uint32_t kNeededMods = a2f::kFirstGazeMorph + a2f::kNumGazeMorphs;
  if ((!offset[0] && !offset[1]) || !mods.values || mods.count < kNeededMods)
    return orig(node, force);
  float saved[a2f::kNumGazeMorphs], adjusted[a2f::kNumGazeMorphs];
  float* const gaze = mods.values + a2f::kFirstGazeMorph;
  std::copy_n(gaze, a2f::kNumGazeMorphs, saved);
  std::copy_n(saved, a2f::kNumGazeMorphs, adjusted);
  if (!a2f::OffsetGazeMorphs(adjusted, offset)) return orig(node, force);
  float* const values = mods.values;
  const bool updated = mods.isUpdated;
  std::copy_n(adjusted, a2f::kNumGazeMorphs, gaze);
  mods.isUpdated = true;
  reset = true;
  const uint64_t result = orig(node, 1);
  // Restore only values no other hook changed meanwhile.
  if (mods.values == values && mods.count >= kNeededMods) {
    bool untouched = true;
    for (int morph = 0; morph < a2f::kNumGazeMorphs; ++morph) {
      if (gaze[morph] == adjusted[morph])
        gaze[morph] = saved[morph];
      else
        untouched = false;
    }
    if (untouched) mods.isUpdated = updated;
  }
  // Even if this was the last eligible frame, the next pass must remove its
  // baked eye offset. No animation value or target is left overridden.
  reinterpret_cast<RE::BSFaceGenNiNode*>(node)->GetRuntimeData().flags |= 1;
  reinterpret_cast<RE::BSFaceGenAnimationData*>(fg)->unk218 = 1;
  return result;
}

// The render thread copies each dynamic shape to the GPU under the shape's
// lock. A copy between the engine's reset (in the original) and AddDeltas
// draws that frame at base + morphs without the pose: a random one-frame
// flicker while speaking. Hold every face shape's lock from before the
// original until the deltas are in. BSSpinLock is re-entrant for its owning
// thread, so the engine's own locking inside ApplyMorphs and AddDeltas nests.
class ScopedShapeLocks {
 public:
  explicit ScopedShapeLocks(uint8_t* node) {
    ForFaceShapes(node, [this](uint8_t* shape, const char*, direct::Part, bool,
                               uint32_t, const float*) {
      Engine<LockDynamicFn, kIdLockDynamic>()(shape);
      shapes_.push_back(shape);
    });
  }
  ~ScopedShapeLocks() {
    for (auto it = shapes_.rbegin(); it != shapes_.rend(); ++it)
      Engine<UnlockDynamicFn, kIdUnlockDynamic>()(*it);
  }
  ScopedShapeLocks(const ScopedShapeLocks&) = delete;
  ScopedShapeLocks& operator=(const ScopedShapeLocks&) = delete;

 private:
  std::vector<uint8_t*> shapes_;
};

// Deltas are only ever added right after a reset to the base vertices (force,
// or the node's reset flag, which the original clears). For a face the plugin
// plays, the face's lock (first, as GazeMorphs and AddDirect take it inside)
// and the shape locks span the original and the deltas.
__declspec(noinline) uint64_t ApplyMorphsHook(uint8_t* node, uint8_t force) {
  auto orig = SiteOrig<ApplyMorphsFn>(_ReturnAddress(), kIdApplyMorphs);
  const uint64_t epoch = world_epoch;
  if (!CurrentWorld(epoch)) return orig(node, force);
  const bool direct =
      ready && (num_playing || num_faces || PartnerHandle() != partner_handle);
  auto* fg = reinterpret_cast<RE::BSFaceGenAnimationData*>(
      reinterpret_cast<RE::BSFaceGenNiNode*>(node)
          ->GetRuntimeData()
          .animationData.get());
  std::optional<skse::ScopedFaceLock> face_guard;
  std::optional<ScopedShapeLocks> shape_locks;
  if (direct && fg && FaceNeedsUpdate(reinterpret_cast<uint8_t*>(fg))) {
    face_guard.emplace(fg->lock);
    shape_locks.emplace(node);
  }
  bool reset =
      force ||
      (reinterpret_cast<RE::BSFaceGenNiNode*>(node)->GetRuntimeData().flags &
       1);
  uint64_t r = GazeMorphs(orig, node, force, reset, epoch);
  if (direct && CurrentWorld(epoch)) AddDirect(node, reset);
  return r;
}

// ---------------- SKSE ----------------
std::atomic<bool> startup_data_loaded{false}, worker_started{false};

// A NewGame message comes from the quest initialization hook, not from
// loading-screen completion. Keep the October 13 startup-scan prerequisite,
// then defer only new-game startup to a game task after Main/Loading menus
// close. Existing-save startup and the worker's lifetime stay unchanged.
skse::SKSETaskInterface* startup_tasks = nullptr;
std::atomic<uint64_t> new_game_start_epoch{0}, new_game_task_epoch{0};

bool StartupMenusClosed() {
#ifdef ENABLE_COMMONLIBSSE_TESTING
  return testing::startup_menus_closed && testing::startup_menus_closed();
#else
  auto* ui = RE::UI::GetSingleton();
  return ui && !ui->IsMenuOpen(RE::MainMenu::MENU_NAME) &&
         !ui->IsMenuOpen(RE::LoadingMenu::MENU_NAME);
#endif
}

void StartInferenceWorker() {
  if (!startup_data_loaded || worker_started.exchange(true)) return;
  skse::Report("game entered: starting inference after startup module scans");
  HANDLE h = CreateThread(nullptr, 0, Worker, nullptr, 0, nullptr);
  const DWORD error = h ? 0 : GetLastError();
  if (h) {
    SetThreadPriority(h, THREAD_PRIORITY_BELOW_NORMAL);
    CloseHandle(h);
  } else {
    worker_started = false;
    skse::Report("cannot start inference worker (error %lu)", error);
  }
  ScopedLock guard(lock);
  stats.waiting_for_game = !h;
  stats.startup_error =
      h ? ""
        : "cannot start inference worker (error " + std::to_string(error) + ")";
}

// Enqueue outside BFA locks: SKSE VR may dispatch with its task lock held.
// Do not poll or enqueue from the task itself; a menu close supplies the next
// attempt. Epoch checks cancel a task when another world starts loading.
void QueueNewGameStartup() {
  const uint64_t epoch = new_game_start_epoch.load();
  if (!epoch || !startup_data_loaded || worker_started) return;
  if (!startup_tasks) {
    ScopedLock guard(lock);
    stats.startup_error = "New-game startup needs the SKSE task interface.";
    return;
  }
  uint64_t expected = 0;
  if (!new_game_task_epoch.compare_exchange_strong(expected, epoch)) return;
  startup_tasks->AddTask([epoch] {
    uint64_t queued = epoch;
    new_game_task_epoch.compare_exchange_strong(queued, 0);
    if (new_game_start_epoch != epoch || !CurrentWorld(epoch) ||
        !StartupMenusClosed())
      return;
    StartInferenceWorker();
    if (worker_started) {
      uint64_t pending = epoch;
      new_game_start_epoch.compare_exchange_strong(pending, 0);
    }
  });
}

class StartupMenuSink : public RE::BSTEventSink<RE::MenuOpenCloseEvent> {
 public:
  RE::BSEventNotifyControl ProcessEvent(
      const RE::MenuOpenCloseEvent* event,
      RE::BSTEventSource<RE::MenuOpenCloseEvent>*) override {
    if (event && !event->opening && event->menuName.c_str()) {
      const std::string_view name = event->menuName.c_str();
      if (name == RE::MainMenu::MENU_NAME || name == RE::LoadingMenu::MENU_NAME)
        QueueNewGameStartup();
    }
    return RE::BSEventNotifyControl::kContinue;
  }
};

void RegisterStartupMenuSink() {
  static StartupMenuSink sink;
  static bool registered = false;
  if (registered) return;
#ifdef ENABLE_COMMONLIBSSE_TESTING
  if (testing::startup_register_menu_sink) {
    testing::startup_register_menu_sink(&sink);
    registered = true;
  }
#else
  if (auto* ui = RE::UI::GetSingleton()) {
    ui->AddEventSink<RE::MenuOpenCloseEvent>(&sink);
    registered = true;
  } else {
    skse::Report("new-game startup: menu event source unavailable");
  }
#endif
}

void StartWorkerAfterStartup(const skse::SKSEMessage& message) {
  if (!startup_data_loaded || worker_started) return;
  // Neither menu events, speech/provider requests nor GPU controls can arm
  // startup before game entry. This preserves startup module-scan exclusion.
  if (message.type == skse::kMessageNewGame) {
    new_game_start_epoch = world_epoch.load();
    skse::Report(
        "new game: deferring inference until main/loading menus close");
    QueueNewGameStartup();
  } else if (message.type == skse::kMessagePostLoadGame && message.data) {
    StartInferenceWorker();
  }
}

void OnMessage(skse::SKSEMessage* m) {
  if (!m) return;
  if (m->type == skse::kMessagePreLoadGame ||
      m->type == skse::kMessagePostLoadGame ||
      m->type == skse::kMessageNewGame) {
    // PostLoadGame is sent for a failed load too: the hooks resume either
    // way, with nothing of the old world kept.
    new_game_start_epoch = 0;
    new_game_task_epoch = 0;
    const bool loading = m->type == skse::kMessagePreLoadGame;
    ResetPlayback(loading);
    Log("playback reset: %s (epoch %llu)",
        loading                            ? "pre-load"
        : m->type == skse::kMessageNewGame ? "new game"
                                           : "post-load",
        static_cast<unsigned long long>(world_epoch.load()));
  }
  MenuMessage(m->type);
  PlayerSpeechMessage(m->type);
  ProviderMessage(m->type);
  StartWorkerAfterStartup(*m);
  if (m->type == skse::kMessagePostLoad && DbvoInstalled() &&
      !native_speech_original) {
    if (NativeSpeechRuntimeSupported(skse::Runtime()))
      native_speech_original = reinterpret_cast<SpeakFn>(
          HookNativeSpeech(reinterpret_cast<uint8_t*>(Addr(kIdSpeakVoiceLine)),
                           reinterpret_cast<void*>(&NativeSpeakHook)));
    Log("native player speech: %s",
        native_speech_original ? "entry observer installed (experimental)"
                               : "unavailable: unsupported runtime or modified "
                                 "entry; native lips retained");
  }
  if (m->type != skse::kMessageDataLoaded) return;
  LoadExpressiveFaces();
  LoadHeadPartTypes();  // before the worker, which turns Direct on
  LoadSceneFactions();
  LoadMouthGearKeywords();
  LoadRaceSettings();
  LoadNpcSettings();
  RetakeSites();
  RegisterMenu();
  RegisterStartupMenuSink();
  {
    ScopedLock guard(lock);
    stats.waiting_for_game = true;
    stats.sentiment_info = IdleSentimentStatus();
  }
  startup_data_loaded = true;
  skse::Report("inference waits for a successful save load or new game");
}

bool Load(const skse::SKSEInterface* iface) {
  SKSE::Init(iface, false);
  skse::SetRuntime(iface->RuntimeVersion().pack());
  InitMenu(iface);
  InitPlayerSpeech(iface);
  InitProvider(iface);
  startup_tasks = static_cast<skse::SKSETaskInterface*>(
      iface->QueryInterface(skse::kInterfaceTask));
  LoadIni();  // turns the log on when Logging=1
  skse::Report("Better Facial Animations %s, game runtime %08X", kVersion,
               iface->RuntimeVersion().pack());
  if (!skse::LoadAddressLibrary()) {
    skse::Report("address library not loaded");
    return false;
  }
  if (!Addr(kIdGenerateFromPath) || !Addr(kIdStreamCtor) ||
      !Addr(kIdStreamDtor)) {
    skse::Report("address ids not found");
    return false;
  }
  if (!HookCalls("speak", kIdSpeakVoiceLine, kSpeakCallers,
                 reinterpret_cast<void*>(&SpeakHook)))
    return false;
  HookCalls("response", kIdDialogueResponseCtor, kResponseCallers,
            reinterpret_cast<void*>(&ResponseHook));
  if (!HookCalls("face", kIdFaceAnimUpdate, kFaceCallers,
                 reinterpret_cast<void*>(&AnimUpdateHook)))
    return false;
  if (!InstallFaceNodeUpdateHook()) return false;
  if (!Addr(kIdMorphData) || !Addr(kIdApplyMorph) || !Addr(kIdLockDynamic) ||
      !Addr(kIdUnlockDynamic) || !Addr(kIdUpdateDynamic) ||
      !HookAllCalls("morphs", kIdApplyMorphs,
                    reinterpret_cast<void*>(&ApplyMorphsHook)))
    return false;
  auto* msg = static_cast<skse::SKSEMessagingInterface*>(
      iface->QueryInterface(skse::kInterfaceMessaging));
  if (!msg || !msg->RegisterListener("SKSE", OnMessage)) {
    Log("no messaging interface");
    return false;
  }
  return true;
}

}  // namespace

void StartPlayerSpeech(PlayerSpeechPtr speech) {
  const uint64_t epoch = world_epoch;
  if (!speech || speech->stopped || !CurrentWorld(epoch)) return;
  const Tune tune = SnapshotTune();
  if (!ready || !tune.enabled || !tune.player) return;
  const auto address = Addr(kIdPlayer);
  auto* actor = address ? *reinterpret_cast<uint8_t**>(address) : nullptr;
  if (!SpeakerOk(actor)) return;
  std::string stem;
  if (!VoiceStem(("Sound/" + speech->path).c_str(), stem)) return;
  // as the engine's own paths, so both share one generated line
  std::replace(stem.begin(), stem.end(), '/', '\\');
  Log("player speech: %s (audio %.3f s)", stem.c_str(), speech->duration);
  const float duration = speech->duration;
  BeginSpeech(actor, stem, a2f::kNoSound, duration, 0, 0, tune, epoch,
              std::move(speech));
}
}  // namespace bfa

// SE 1.5.97's and VR's SKSE ask this instead of reading SKSEPlugin_Version.
extern "C" __declspec(dllexport) bool SKSEPlugin_Query(
    const skse::SKSEInterface* iface, skse::PluginInfo* info) {
  info->infoVersion = 1;
  info->name = SKSEPlugin_Version.pluginName;
  info->version = SKSEPlugin_Version.pluginVersion;
  return !iface->IsEditor() &&
         (iface->RuntimeVersion().pack() == skse::Version(1, 5, 97) ||
          iface->RuntimeVersion().pack() == skse::kVrRuntime);
}

extern "C" __declspec(dllexport) bool SKSEPlugin_Load(
    const skse::SKSEInterface* iface) {
  return bfa::Load(iface);
}

BOOL WINAPI DllMain(HINSTANCE, DWORD, LPVOID) { return TRUE; }

namespace bfa {
// ---------------- the provider interface's plugin side ----------------
uint32_t ResolveProviderActor(uint32_t form_id, ProviderActor& out) {
  const Tune tune = SnapshotTune();
  if (!tune.enabled) return BFA_kResultDisabled;
  if (!ready || !net_loaded) return BFA_kResultModelNotLoaded;
  if (loading_game) return BFA_kResultNoFaceGenHead;
  const auto lookup = Engine<uint8_t* (*)(uint32_t), kIdLookupFormById>();
  if (!lookup) return BFA_kResultUnsupportedRuntime;
  uint8_t* actor = lookup(form_id);
  if (!actor ||
      reinterpret_cast<const RE::TESForm*>(actor)->formType.underlying() !=
          kFtActor)
    return BFA_kResultNoFaceGenHead;
  const bool player = IsPlayer(actor);
  if (player && !tune.player) return BFA_kResultPlayerDisabled;
  uint8_t* base = reinterpret_cast<uint8_t*>(
      reinterpret_cast<RE::TESObjectREFR*>(actor)->data.objectReference);
  uint8_t* race = reinterpret_cast<uint8_t*>(
      reinterpret_cast<RE::Actor*>(actor)->GetActorRuntimeData().race);
  if (!base ||
      reinterpret_cast<RE::TESForm*>(base)->formType.underlying() != kFtNpc ||
      !race ||
      !(reinterpret_cast<RE::TESRace*>(race)->data.flags.underlying() & 2))
    return BFA_kResultNoFaceGenHead;
  if (!RaceAllowed(actor)) return BFA_kResultRaceDisabled;
  if (!NpcAllowed(actor)) return BFA_kResultDisabled;
  if (FaceExternallyOwned(actor)) return BFA_kResultInScene;
  const uint32_t handle = HandleOf(actor);
  if (!handle) return BFA_kResultNoFaceGenHead;
  if (!DialogueAllows(tune, handle, player)) return BFA_kResultNotInDialogue;
  uint8_t* fg = reinterpret_cast<uint8_t*>(
      reinterpret_cast<RE::Actor*>(actor)->GetFaceGenAnimationData());
  if (!fg) return BFA_kResultNoFaceGenHead;
  out = ProviderActor{actor, handle, fg, IsFemale(actor), player};
  ScopedLock guard(lock, ScopedLock::kShared);
  auto face = faces.find(fg);
  if (face != faces.end()) out.head_state = face->second.head_state;
  return BFA_kResultOK;
}

uint32_t ProviderPrepare(const PlayerSpeechPtr& line,
                         const ProviderActor& who) {
  auto e =
      Request(line->stem, 0, 0, who.female, true, SnapshotTune(), who.handle,
              world_epoch, who.player, &line->emotion, line->audio, who.actor);
  if (!e) return BFA_kResultNotInDialogue;
  ScopedLock guard(lock);
  WatchEntry(*e, line);
  return BFA_kResultOK;
}

uint32_t ProviderStart(const PlayerSpeechPtr& line) {
  const uint64_t epoch = world_epoch;
  ProviderActor who;
  const uint32_t r = ResolveProviderActor(line->form_id, who);
  if (r != BFA_kResultOK) return r;
  return BeginSpeech(who.actor, line->stem, a2f::kNoSound, line->duration, 0, 0,
                     SnapshotTune(), epoch, line);
}

void ProviderDiscardUtterance(uint64_t utterance) {
  char prefix[40];
  snprintf(prefix, sizeof prefix,
           "provider:%016llx:", static_cast<unsigned long long>(utterance));
  const std::string key(prefix);
  const auto of_utterance = [&](const std::string& cache_key) {
    return cache_key.compare(0, key.size(), key) == 0;
  };
  ScopedLock guard(lock);
  // A line shown on a face leaves it the regular way (stopped: LineClock),
  // its animation kept for the hand-back fade; one not shown is failed so
  // that it leaves at once.
  std::vector<std::shared_ptr<Entry>> keep;
  for (auto& [fg, p] : playing) {
    if (!ProviderLine(p) || p.external->utterance_id != utterance) continue;
    if (p.direct_frames != 0 && p.external->stopped) {
      keep.push_back(p.entry);
      continue;
    }
    p.direct_frame = -1;
    p.entry->cancelled = true;
    p.entry->line.reset();
    p.entry->state = State::kFailed;
  }
  for (auto it = cache.begin(); it != cache.end();) {
    if (!of_utterance(it->first)) {
      ++it;
      continue;
    }
    it->second->watchers.clear();
    if (std::find(keep.begin(), keep.end(), it->second) == keep.end()) {
      it->second->cancelled = true;
      it->second->line.reset();
      it->second->state = State::kFailed;
    }
    it = cache.erase(it);
  }
  queue.erase(
      std::remove_if(queue.begin(), queue.end(),
                     [&](const Job& j) { return of_utterance(j.cache_key); }),
      queue.end());
}
}  // namespace bfa
