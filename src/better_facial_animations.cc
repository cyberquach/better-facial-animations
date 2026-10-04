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
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "call_sites.h"
#include "direct.h"
#include "engine.h"
#include "head_cache.h"
#include "ids.h"
#include "live_core.h"
#include "menu.h"
#include "net_keeper.h"
#include "ort_net.h"
#include "plugin.h"
#include "pose.h"
#include "scoped_lock.h"
#include "skse.h"
#include "voice_sync.h"

// From the VERSION file, passed in by build.sh.
#ifndef BFA_VERSION
#error "build with build.sh, which defines BFA_VERSION from VERSION"
#endif

namespace {

// "major.minor.patch" -> SKSE's packed plugin version.
constexpr uint32_t PackVersion(const char* s) {
  uint32_t part[3] = {};
  for (int i = 0; *s; ++s) {
    if (*s == '.') {
      ++i;
    } else {
      part[i] = part[i] * 10 + (*s - '0');
    }
  }
  return skse::Version(part[0], part[1], part[2]);
}

}  // namespace

__declspec(dllexport) skse::SKSEPluginVersionData SKSEPlugin_Version = {
    1,
    PackVersion(BFA_VERSION),
    "BetterFacialAnimations",
    "Cyberquach",
    "",
    0,
    0,
    {skse::Version(1, 6, 640), skse::Version(1, 6, 1170),
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

constexpr wchar_t kPluginDir[] = L"Data\\SKSE\\Plugins\\BetterFacialAnimations";
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
  State state = State::kPending;
  std::shared_ptr<const a2f::Line> line;
  uint64_t last_use = 0;
};
struct Job {
  std::string cache_key, path;
  // the emotion and its settings when it was requested
  a2f::Emotion emotion;
};
enum FaceUpdate { kNoUpdate = 0, kFaceChanged = 1, kKeepUpdating = 2 };
constexpr int kDirectTail = 3;  // forced resets after the last Direct frame
struct Playing {
  std::shared_ptr<Entry> entry;
  std::string name;
  uint64_t lip_key = 0;
  uint8_t* actor = nullptr;
  // the actor's handle (HandleOf), which tells when it is gone; 0: it had
  // none, the pointer is trusted
  uint32_t actor_handle = 0;
  double seen = 0;  // last update (play_clock)
  a2f::VoiceStart voice;
  a2f::LipClock lip_clock;
  float ready_at = -1;
  bool lip_seen = false, logged = false;
  // Direct: the frame (fractional, -1 none) for the morph hook, and whether
  // the line was ready late (it is caught up with over BlendIn)
  float direct_frame = -1;
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
// A face the plugin animates (Expressions, AddDirect): anyone speaking a
// line and the player's conversation partner, for a while after
// (kFaceLinger). It holds the head's mapping onto Mark.
struct Face {
  // the engine's own expression keyframe, put back before its update (so
  // the plugin's writes are never read back as the engine's), and what the
  // plugin wrote in its place
  float src[direct::kNumExpressions] = {};
  float written[direct::kNumExpressions] = {};
  float level[direct::kNumExpressions] = {};  // added to the meshes
  // 0..1: from the engine's expressions to the plugin's.
  float ownership_blend = 0;
  float speech_blend = 0;  // 0..1: the expressions faded out for a line
  double until = 0;  // the plugin's until (then fades back to the engine's)
  double seen = 0;   // last update
  // Not updated for kFaceStale (unloaded, or out of sight) with the plugin's
  // values still in the keyframe: only src and written are kept, to put the
  // engine's back when the face is next updated.
  bool parked = false;
  // failed: not mappable (vanilla lip-sync)
  State head_state = State::kPending;
  std::shared_ptr<const direct::Head> head;
  std::string head_key;  // geometry and morph identity used by head
  a2f::PoseTransition transition;
};
// Times on play_clock:
constexpr double kFaceLinger = 300;  // s after the conversation or line
constexpr double kFaceStale = 30;    // s without an update: unloaded
// s without an update: the line's head is gone (unloaded or rebuilt, as on
// leaving the race menu) and the line is dropped
constexpr double kPlayingStale = 30;
constexpr double kPlayStep = 0.25;  // the most play_clock advances at once
// s for the plugin to take a face's expressions over from the engine and to
// give them back
constexpr float kOwnIn = 0.25f, kOwnOut = 1;
constexpr size_t kMaxFaces = 8;
constexpr size_t kMaxEvents = 8;

SRWLOCK lock = SRWLOCK_INIT;  // guards everything below
CONDITION_VARIABLE queue_cv = CONDITION_VARIABLE_INIT;
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
// The time face updates have been running: the wall clock without the gaps
// between them (a paused game, a menu), so a pause does not make a line or
// a face look abandoned.
double play_clock, play_wall;
uint64_t use_clock;
Stats& stats = *new Stats;

// Lock held, after playing or faces changed.
void Count() {
  num_playing = static_cast<int>(playing.size());
  num_faces = static_cast<int>(faces.size());
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

}  // namespace

double Now() {
  static const LARGE_INTEGER f = [] {
    LARGE_INTEGER frequency;
    QueryPerformanceFrequency(&frequency);
    return frequency;
  }();
  LARGE_INTEGER c;
  QueryPerformanceCounter(&c);
  return static_cast<double>(c.QuadPart) / f.QuadPart;
}

std::string Lower(const char* s) {
  std::string n = s;
  for (auto& c : n) c = static_cast<char>(tolower(static_cast<uint8_t>(c)));
  return n;
}

Stats SnapshotStats(size_t& cached, size_t& queued) {
  ScopedLock guard(lock, ScopedLock::kShared);
  Stats st = stats;
  cached = cache.size();
  queued = queue.size();
  return st;
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

// Actor with a FaceGen head (people; animals and creatures have none),
// except Argonians.
bool SpeakerOk(uint8_t* ref) {
  if (!ref || ref[kFormType] != kFtActor) return false;
  uint8_t* base = Field<uint8_t*>(ref, kRefrBase);
  uint8_t* race = Field<uint8_t*>(ref, kActorRace.Get());
  if (!base || base[kFormType] != kFtNpc || !race ||
      !(Field<uint32_t>(race, kRaceFlags) & 2))
    return false;
  uint32_t race_id = Field<uint32_t>(race, kFormId);
  return race_id != kArgonianRace && race_id != kArgonianRaceVampire;
}

// Queue a line unless it is cached or in flight (with the emotion settings
// as now: after a change the next line spoken is generated anew). Returns
// its entry.
std::shared_ptr<Entry> Request(const std::string& stem, int emo, int val,
                               bool urgent, const Tune& tune) {
  std::string key = Lower(stem.c_str());
  // neutral: no emotion to feed, or none wanted
  if (tune.voice_emotion <= 0 || emo < 1 || emo > 7) {
    emo = 0;
    val = 0;
  }
  a2f::Emotion em;
  em.type = emo;
  em.value = val;
  em.gain = emo ? tune.voice_emotion : 0;
  em.swell = em.gain > 0 ? tune.voice_swell : 0;
  auto part = [&key](double v) {
    key += '|';
    key += std::to_string(std::lround(v));
  };
  for (double v : {1. * emo, 1. * val, em.gain * 100., em.swell * 100.})
    part(v);
  ScopedLock guard(lock);
  auto& e = cache[key];
  if (!e) {
    e = std::make_shared<Entry>();
    Job j{key, stem, em};
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
  e->last_use = ++use_clock;
  return e;
}

// Read a file through BSResourceNiBinaryStream (BSAs + loose files, MO2 VFS);
// path relative to Data.
bool ReadGameFile(const std::string& path, std::vector<uint8_t>& out) {
  using Ctor = void* (*)(void*, const char*, uint64_t, void*);
  using Dtor = void (*)(void*);
  using ReadFn = uint32_t (*)(void*, void*, uint32_t, uint32_t*, uint32_t);
  alignas(16) uint8_t s[0x40] = {};
  Engine<Ctor, kIdStreamCtor>()(s, path.c_str(), 0, nullptr);
  // BSResource::Stream*, null when not found
  void* st = Field<void*>(s, 0x20);
  bool ok = false;
  if (st) {
    uint32_t n = Field<uint32_t>(st, 8);
    out.resize(n);
    auto rd = Field<ReadFn>(s, 0x10);
    uint32_t got = 0;
    while (got < n) {
      uint32_t k = rd(s, out.data() + got, n - got, nullptr, 1);
      if (!k) break;
      got += k;
    }
    ok = n > 0 && got == n;
  }
  Engine<Dtor, kIdStreamDtor>()(s);
  return ok;
}

// ---------------- worker ----------------
std::wstring PluginDir() {
  wchar_t full[MAX_PATH];
  GetFullPathNameW(kPluginDir, MAX_PATH, full, nullptr);
  return full;
}

void NetEvent(const std::string& text);

// Worker thread only: what the GPU sessions found out about the driver's
// metacommands.
a2f::Metacommands metacommands;

std::unique_ptr<a2f::Net> CreateNet(bool gpu, std::string& err,
                                    std::string& info) {
  const std::wstring dir = PluginDir(), file = dir + L"\\" + config.model;
  static bool dml_loaded;
  if (gpu && !dml_loaded) {
    // onnxruntime delay-loads DirectML.dll by name: load ours first so that
    // is the one it gets
    dml_loaded = LoadLibraryExW((dir + L"\\DirectML.dll").c_str(), nullptr,
                                LOAD_WITH_ALTERED_SEARCH_PATH) != nullptr;
    if (!dml_loaded)
      skse::Report("DirectML.dll not loaded from %ls (error %lu)", dir.c_str(),
                   GetLastError());
  }
  a2f::OrtOptions o;
  o.gpu = gpu;
  o.batch = config.batch;
  if (GetFileAttributesW(file.c_str()) == INVALID_FILE_ATTRIBUTES) {
    char name[260] = "";
    WideCharToMultiByte(CP_UTF8, 0, config.model.c_str(), -1, name, sizeof name,
                        nullptr, nullptr);
    err = std::string("model file not found: ") + name;
    return nullptr;
  }
  if (!gpu) return a2f::CreateOrtNet(dir, file, o, err, info);
  // with the driver's metacommands, and without them when that fails
  return a2f::CreateGpuNet(
      [&](bool with, std::string& e, std::string& i) {
        o.metacommands = with;
        const double t = Now();
        auto net = a2f::CreateOrtNet(dir, file, o, e, i);
        skse::Report("GPU session with metacommands %s: %s after %.0f ms",
                     with ? "on" : "off", net ? "loaded" : "failed",
                     (Now() - t) * 1e3);
        return net;
      },
      metacommands, err, info, NetEvent);
}

// Worker thread only; created by the worker once the ini is read. Loads the
// network at the main menu (little VRAM in use) and never unloads it.
a2f::NetKeeper* keeper;

// Network events go to the log and to the settings window.
void NetEvent(const std::string& text) {
  skse::Report("%s", text.c_str());
  ScopedLock guard(lock);
  stats.events.emplace_back(Now(), text);
  if (stats.events.size() > kMaxEvents) stats.events.pop_front();
}

// Copies the keeper's state to the settings window.
void PublishNet() {
  static int diagnosed;  // attempts the diagnosis was made for
  static auto& why = *new std::vector<std::string>;
  if (keeper->error().empty()) {
    diagnosed = 0;
    why.clear();
  } else if (keeper->attempts() != diagnosed) {
    diagnosed = keeper->attempts();
    why = a2f::DiagnoseNet(config.gpu, keeper->error(), diagnosed,
                           a2f::QueryGpu());
    for (const auto& line : why) skse::Report("%s", line.c_str());
    // the system as it is at the first failure (the report at startup is
    // from the main menu)
    static bool reported;
    if (config.gpu && !reported) {
      reported = true;
      skse::Report("-- at the first failure");
      for (const auto& line : a2f::GpuReport())
        skse::Report("%s", line.c_str());
    }
  }
  net_loaded = keeper->loaded();
  ScopedLock guard(lock);
  stats.net_info = keeper->info();
  stats.net_error = why;
  stats.net_detail = keeper->error();
  stats.net_attempts = keeper->attempts();
  stats.net_retry_at = keeper->retry_at();
  stats.net_gpu = keeper->on_gpu();
  stats.net_metacommands = config.gpu
                               ? a2f::DescribeMetacommands(metacommands)
                               : "not used (GPU=0: the CPU runs the model)";
}

// Generates one queued line and stores the result in the cache.
void Process(const Job& j) {
  static std::atomic<int> logged;
  std::vector<uint8_t> bytes;
  std::string err, file;
  for (const char* ext : {".fuz", ".xwm", ".wav"})
    if (ReadGameFile(j.path + ext, bytes)) {
      file = j.path + ext;
      break;
    }
  std::shared_ptr<a2f::Line> line;
  double t = Now(), audio_sec = 0;
  bool no_net = false;  // not this line's fault: generate it again next time
  if (file.empty()) {
    err = "no voice file";
  } else {
    std::vector<float> pcm;
    if (a2f::DecodeAudio(bytes.data(), bytes.size(), pcm, err)) {
      audio_sec = pcm.size() / static_cast<double>(a2f::kSampleRate);
      line = std::make_shared<a2f::Line>();
      if (!keeper->Run(
              [&](a2f::Net& net, std::string& e) {
                return a2f::Generate(net, pcm.data(), pcm.size(), j.emotion,
                                     model.mark.proj.data(), model.rest.data(),
                                     model.mark.k, *line, e);
              },
              err, no_net))
        line.reset();
      PublishNet();
    }
  }
  double ms = (Now() - t) * 1e3;
  ScopedLock guard(lock);
  auto it = cache.find(j.cache_key);
  if (it != cache.end()) {
    it->second->state = line ? State::kReady : State::kFailed;
    it->second->line = line;
    if (!line && no_net) cache.erase(it);
  }
  if (line) {
    ++stats.lines;
    stats.audio_sec += audio_sec;
    stats.gen_sec += ms / 1e3;
    stats.last_ms = ms;
  } else {
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
      Log("gen %s emo %d/%d: %.2f s audio, %u frames in %.0f ms; fed%s",
          file.c_str(), j.emotion.type, j.emotion.value, audio_sec,
          line->frames, ms, fed.empty() ? " no emotion" : fed.c_str());
    } else {
      Log("skip %s: %s", j.path.c_str(), err.c_str());
    }
  }
}

DWORD WINAPI Worker(void*) {
  std::string err;
  if (!model.Load(PluginDir() + L"\\A2FDirect.bin", err)) {
    skse::Report("A2FDirect.bin: %s", err.c_str());
    ScopedLock guard(lock);
    stats.startup_error = "A2FDirect.bin: " + err;
    return 1;
  }
  ready = true;
  skse::Report("settings: GPU=%d, batch %d, model %ls", config.gpu,
               config.batch, config.model.c_str());
  for (const auto& line : a2f::GpuReport()) skse::Report("%s", line.c_str());
  a2f::SetOrtLog([](const std::string& s) { skse::Report("%s", s.c_str()); });
  keeper = new a2f::NetKeeper(config.gpu, CreateNet, Now, NetEvent);
  auto load = [] {
    keeper->Ensure();
    PublishNet();
  };
  if (SnapshotTune().enabled) load();
  for (;;) {
    ScopedLock worker_guard(lock);
    while (queue.empty()) {
      // Without the wanted session: wake up to try it again.
      SleepConditionVariableSRW(&queue_cv, &lock,
                                keeper->Wanted() ? INFINITE : 1000, 0);
      if (queue.empty() && SnapshotTune().enabled && !keeper->Wanted()) {
        worker_guard.Unlock();
        load();
        worker_guard.Lock();
      }
    }
    Job j = queue.front();
    queue.pop_front();
    worker_guard.Unlock();
    Process(j);
  }
}

// ---------------- call-site hooks ----------------
// Every `call rel32` to the callee inside the caller functions is redirected
// through a thunk (src/call_sites.h). Each site keeps its own original target
// (looked up by return address), so a plugin that patches a site after us is
// chained.
template <typename Fn>
Fn SiteOrig(void* ret, skse::Rel callee_id) {
  return call_sites::Orig<Fn>(ret, reinterpret_cast<void*>(Addr(callee_id)));
}
template <size_t N>
int HookCalls(const char* what, skse::Rel callee_id,
              const skse::Rel (&callers)[N], void* hook) {
  const uint64_t base = skse::ImageBase();
  uint64_t callee = Addr(callee_id);
  auto* nt = reinterpret_cast<IMAGE_NT_HEADERS*>(
      base + reinterpret_cast<IMAGE_DOS_HEADER*>(base)->e_lfanew);
  uint64_t end = base + nt->OptionalHeader.SizeOfImage;
  if (!callee) {
    Log("%s: address id %llu not found", what, callee_id.Get());
    return 0;
  }
  int n = 0;
  for (skse::Rel caller : callers) {
    uint64_t f = Addr(caller);
    if (!f) continue;
    for (uint64_t at = f; at < f + 0x2000 && at + 5 < end; ++at) {
      if (*reinterpret_cast<uint8_t*>(at) != 0xE8 ||
          call_sites::CallTarget(at) != callee)
        continue;
      if (!call_sites::Hook(at, hook, what, base)) {
        Log("%s: call at rva %llx not hooked (no thunk in reach)", what,
            at - base);
        continue;
      }
      Log("%s: hooked call at rva %llx (caller id %llu)", what, at - base,
          caller.Get());
      ++n;
    }
  }
  if (!n)
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

// The reference handle table.
uint8_t* HandleEntries() {
  static const auto entries =
      reinterpret_cast<uint8_t*>(Addr(kIdHandleEntries));
  return entries;
}

// The reference a handle is of, or null once it is gone.
uint8_t* RefOf(uint32_t h) {
  const uint8_t* entries = HandleEntries();
  if (!h || !entries) return nullptr;
  const uint8_t* e = entries + (h & kHandleIndexMask) * kHandleEntrySize;
  uint32_t bits = *reinterpret_cast<const uint32_t*>(e);
  uint8_t* obj = *reinterpret_cast<uint8_t* const*>(e + 8);
  if (!(bits & kHandleInUse) || ((bits ^ h) & kHandleAgeMask) || !obj)
    return nullptr;
  return obj - kRefrHandleObj;
}

// The reference's handle, or 0 if it has none (or its entry is not the
// reference's after all).
uint32_t HandleOf(uint8_t* ref) {
  const uint8_t* entries = HandleEntries();
  uint32_t count = Field<uint32_t>(ref, kRefrRefCount);
  if (!entries || !(count & kRefCountHasHandle)) return 0;
  uint32_t index = count >> kRefCountIndexShift;
  if (!index || index > kHandleIndexMask) return 0;
  uint32_t bits =
      *reinterpret_cast<const uint32_t*>(entries + index * kHandleEntrySize);
  uint32_t h = index | (bits & kHandleAgeMask);
  return RefOf(h) == ref ? h : 0;
}

// The reference the player is in a conversation with, or null.
uint8_t* PartnerRef() {
  static auto mtm = reinterpret_cast<uint8_t**>(Addr(kIdMenuTopicManager));
  if (!mtm || !*mtm) return nullptr;
  return RefOf(Field<uint32_t>(*mtm, kMtmSpeaker));
}

// The node is in the reference's 3D.
bool OfRef(uint8_t* node, const uint8_t* ref) {
  for (int i = 0; node && i < 32; node = Field<uint8_t*>(node, kAvParent), ++i)
    if (Field<uint8_t*>(node, kAvUserData) == ref) return true;
  return false;
}

// The actor's voice sound ID and whether it is playing: 1, 0, or -1 if
// unknown (no high process).
int ActorVoice(uint8_t* actor, uint32_t& sound) {
  auto is_playing = Engine<bool (*)(void*), kIdSoundIsPlaying>();
  uint8_t* proc = Field<uint8_t*>(actor, kActorProcess.Get());
  uint8_t* high = proc ? Field<uint8_t*>(proc, kProcessHigh) : nullptr;
  uint8_t* handle = high ? high + kHighVoiceHandle : nullptr;
  sound = SoundId(handle);
  return !handle || !is_playing ? -1 : is_playing(handle);
}

__attribute__((noinline)) float SpeakHook(
    uint8_t* actor, const char* voice, void* handle, uint64_t emo_type,
    uint64_t emo_value, uint64_t a6, uint64_t a7, uint64_t a8, uint64_t a9,
    uint64_t a10, uint64_t a11, uint64_t a12, uint64_t a13, uint64_t a14) {
  auto orig = SiteOrig<SpeakFn>(__builtin_return_address(0), kIdSpeakVoiceLine);
  float dur = orig(actor, voice, handle, emo_type, emo_value, a6, a7, a8, a9,
                   a10, a11, a12, a13, a14);
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
  if (!(tune.enabled || net_loaded) || !voice || !ready || !SpeakerOk(actor) ||
      !VoiceStem(voice, stem)) {
    note("not for the plugin");
    return dur;
  }
  note("taken");
  auto get_fg = reinterpret_cast<uint8_t* (*)(uint8_t*)>(
      Field<void**>(actor, 0)[kVfGetFaceGenAnimationData]);
  uint8_t* fg = get_fg(actor);
  if (!fg) return dur;
  auto e =
      Request(stem, static_cast<int>(static_cast<uint32_t>(emo_type)),
              static_cast<int>(static_cast<uint32_t>(emo_value)), true, tune);
  skse::ResId r{};
  Engine<GenerateFromPathFn, kIdGenerateFromPath>()(&r,
                                                    (stem + ".lip").c_str());
  Playing p;
  p.entry = e;
  p.name = stem;
  p.lip_key = skse::KeyOf(r);
  p.actor = actor;
  p.actor_handle = HandleOf(actor);
  static std::atomic<int> no_handle;
  if (!p.actor_handle && LogMore(no_handle))
    Log("speak %s: the actor has no handle", voice);
  p.voice = {SoundId(handle), Now(), dur};
  uint32_t sound;
  int playing_now = ActorVoice(actor, sound);
  p.voice.Observe(sound, playing_now, p.voice.requested_at);
  ScopedLock guard(lock);
  p.seen = play_clock;
  playing[fg] = std::move(p);
  Count();
  return dur;
}

__attribute__((noinline)) uint8_t* ResponseHook(uint8_t* self, void* topic,
                                                void* info, uint8_t* speaker,
                                                void* response) {
  auto orig = SiteOrig<ResponseCtorFn>(__builtin_return_address(0),
                                       kIdDialogueResponseCtor);
  uint8_t* r = orig(self, topic, info, speaker, response);
  std::string stem;
  const Tune tune = SnapshotTune();
  const char* voice = r ? Field<const char*>(r, kRespVoice) : nullptr;
  if ((tune.enabled || net_loaded) && tune.prefetch && voice && ready &&
      SpeakerOk(speaker) && VoiceStem(voice, stem))
    Request(stem, static_cast<int>(Field<uint32_t>(r, kRespEmoType)),
            static_cast<int>(Field<uint32_t>(r, kRespEmoValue)), false, tune);
  return r;
}

// Engine modifiers the face mesh takes over (brows, squints); the blinks
// (0, 1) stay with the engine.
constexpr int kModZeroed[] = {2, 3, 4, 5, 6, 7, 12, 13};
constexpr uint32_t kNumPhonemes = 16;
inline float Mix(float engine, float ours, float mix) {
  return mix >= 1.f ? ours : engine + (ours - engine) * mix;
}

// The mesh carries the mouth, brows and eyes: zeroes the final phoneme and
// modifier keyframes (but the blinks); mix < 1 blends from the engine's
// values. The expressions are Expressions'.
void PlayFrame(uint8_t* fg, float mix) {
  auto& phon = Field<skse::KeyframeMultiple>(fg, skse::kFgPhon3);
  auto& mod = Field<skse::KeyframeMultiple>(fg, skse::kFgMod3);
  if (phon.values && phon.count >= kNumPhonemes) {
    for (uint32_t i = 0; i < kNumPhonemes; ++i)
      phon.values[i] = Mix(phon.values[i], 0.f, mix);
    phon.is_updated = true;
  }
  if (mod.values && mod.count >= 14) {
    for (int e : kModZeroed) mod.values[e] = Mix(mod.values[e], 0.f, mix);
    mod.is_updated = true;
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
                const char*& why) {
  const skse::ResId* act = skse::ActiveLine(fg);
  uint32_t sound;
  int playing_now = ActorVoice(p.actor, sound);
  int voice_state = p.voice.Observe(sound, playing_now, now);
  bool hold = p.voice.Holding(voice_state);  // until the line's sound plays
  float t = hold ? 0 : static_cast<float>(p.voice.Elapsed(now));
  why = now - p.voice.requested_at > 120 ? "timed out" : nullptr;
  lip = act && skse::KeyOf(*act) == p.lip_key;
  if (lip) {  // engine lip clock
    t = p.lip_clock.Update(Field<float>(fg, skse::kFgPhon1Timer), now, p.voice,
                           hold, p.direct_frames != 0);
    p.lip_seen = true;
  } else if (p.lip_seen) {
    why = "the engine ended or replaced its lip line";
  } else if (p.voice.Ended(voice_state, now)) {
    why = "the voice ended";  // no .lip: wall clock while the voice plays
  }
  if (!why && p.entry->state == State::kFailed)
    why = "the line could not be generated";
  if (!why && p.entry->state == State::kReady &&
      t > p.entry->line->frames / static_cast<float>(a2f::kFps) + 0.3f)
    why = "the line is over";
  return t;
}

// Plays the current line. kFaceChanged requests a mesh reset;
// kKeepUpdating schedules another face update, including the final resets.
int Apply(uint8_t* fg, const Tune& tune) {
  double now = Now();
  ScopedLock guard(lock);
  Tick(now);
  SweepPlaying(fg);
  auto it = playing.find(fg);
  if (it == playing.end()) {
    return kNoUpdate;
  }
  Playing& p = it->second;
  p.seen = play_clock;
  // the actor is gone and this is another's head at the same address
  if (p.actor_handle && RefOf(p.actor_handle) != p.actor) {
    if (LogMore(logged_ends)) Log("drop %s: its actor is gone", p.name.c_str());
    playing.erase(it);
    Count();
    return kNoUpdate;
  }
  bool lip;
  const char* why;
  const float t = LineClock(fg, p, now, lip, why);
  const bool done = why != nullptr;
  const Face* face = LiveFace(fg);
  const State head_state = face ? face->head_state : State::kPending;
  const bool generated = p.entry->state == State::kReady;
  // holding (t = 0) too: the engine's .lip already runs before the sound
  if (done || !generated || !tune.enabled || head_state != State::kReady) {
    p.direct_frame = -1;
    if (!tune.enabled) p.snap = true;
    bool tail = p.direct_tail > 0;
    if (tail) --p.direct_tail;
    // the mapping is found in the morph hook: keep it coming
    if (!done && tune.enabled && head_state == State::kPending) tail = true;
    if (!done && generated && !tune.enabled && !p.logged && tune.logging) {
      p.logged = true;
      Log("vanilla %s: Enabled is off", p.name.c_str());
    }
    if (done && !tail) {
      LogEnd(p, head_state, why, t);
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
      p.late ? std::clamp((t - p.ready_at) / std::max(tune.blend_in, 1e-3f),
                          0.f, 1.f)
             : 1.f;
  p.last_t = t;
  ++p.direct_frames;
  p.direct_frame =
      std::fmax(0.f, (t + tune.time_offset) * static_cast<float>(a2f::kFps));
  p.direct_tail = kDirectTail;
  const bool first = !p.logged;
  p.logged = true;
  const std::string name = first ? p.name : std::string();
  const float ready_at = p.ready_at;
  const uint32_t frames = p.entry->line->frames;
  guard.Unlock();
  if (first && LogMore(logged_plays))
    Log("play %s: %u frames, %s clock, ready at %.2f s", name.c_str(), frames,
        lip ? "lip" : "wall", ready_at);
  PlayFrame(fg, mix);
  return kFaceChanged | kKeepUpdating;
}

// Drops a face (lock held).
void DropFace(uint8_t* fg) {
  if (fg == partner_fg) {
    partner_fg = nullptr;
    partner_ref = nullptr;
  }
  faces.erase(fg);
  Count();
}

// Before the engine's face update: puts back its own expression keyframe,
// which Expressions overwrote, so the engine carries on from it. A parked
// face's only while the keyframe is still as the plugin left it (else the
// engine, or a new head at this address, wrote it since).
void RestoreExpressions(uint8_t* fg) {
  auto& ex = Field<skse::KeyframeMultiple>(fg, skse::kFgExpr3);
  if (!ex.values || ex.count < direct::kNumExpressions) return;
  float src[direct::kNumExpressions], written[direct::kNumExpressions];
  ScopedLock guard(lock, ScopedLock::kShared);
  auto it = faces.find(fg);
  const bool found = it != faces.end();
  const bool parked = found && it->second.parked;
  if (found) {
    std::copy_n(it->second.src, direct::kNumExpressions, src);
    std::copy_n(it->second.written, direct::kNumExpressions, written);
  }
  guard.Unlock();
  if (!found) return;
  auto& face_lock = Field<skse::SpinLock>(fg, skse::kFgLock);
  face_lock.Lock();
  bool ours = true;
  for (int i = 0; parked && i < direct::kNumExpressions; ++i)
    if (direct::PlaysExpression(i) && ex.values[i] != written[i]) ours = false;
  for (int i = 0; ours && i < direct::kNumExpressions; ++i)
    if (direct::PlaysExpression(i)) ex.values[i] = src[i];
  face_lock.Unlock();
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
// parked until it is updated again (RestoreExpressions) or kFaceLinger
// passes; the others are forgotten.
void SweepFaces(uint8_t* keep, double clock) {
  static double swept = 0;
  if (clock - swept <= 1) return;
  swept = clock;
  for (auto it = faces.begin(); it != faces.end();) {
    Face& f = it->second;
    uint8_t* fg = it->first;
    ++it;  // DropFace erases
    if (fg == keep || clock - f.seen <= (f.parked ? kFaceLinger : kFaceStale))
      continue;
    if (f.parked ||
        std::equal(std::begin(f.src), std::end(f.src), std::begin(f.written))) {
      DropFace(fg);
      continue;
    }
    Face parked;
    std::copy_n(f.src, direct::kNumExpressions, parked.src);
    std::copy_n(f.written, direct::kNumExpressions, parked.written);
    parked.parked = true;
    parked.seen = clock;
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
        (1 - f.ownership_blend) * f.src[i] +
        (mapped ? 0 : f.ownership_blend * lv[i] * MeanExpressionScale());
    if (ex) ex->values[i] = f.written[i];
  }
  if (ex) ex->is_updated = true;
  // the mapping is found in the morph hook: keep it coming
  if (f.head_state == State::kPending) updates = kFaceChanged | kKeepUpdating;
  return updates;
}

// The face of anyone speaking a line and of the player's conversation
// partner, after Apply (face lock held): keeps the Face, and plays its
// emotion expressions (direct::PlaysExpression: dialogue, mood, combat)
// itself, scaled per region; while a line plays they are all faded out (the
// network carries the emotion). Fades in and out (Face::ownership_blend).
// Returns like Apply.
int Expressions(uint8_t* fg, float dt, const Tune& tune) {
  auto& ex = Field<skse::KeyframeMultiple>(fg, skse::kFgExpr3);
  const bool has_ex = ex.values && ex.count >= direct::kNumExpressions;
  if (fg != partner_fg && !num_faces && !num_playing) return kNoUpdate;
  const double now = Now();
  ScopedLock guard(lock);
  const double clock = Tick(now);
  const bool partner =
      fg == partner_fg && partner_ref && PartnerRef() == partner_ref;
  auto pit = playing.find(fg);
  const bool in_line = pit != playing.end();
  const bool wanted = tune.enabled && (in_line || (partner && has_ex));
  Face* found = FaceFor(fg, wanted, clock);
  if (!found) {
    return kNoUpdate;
  }
  Face& f = *found;
  SweepFaces(fg, clock);
  if (f.parked) f = Face();  // back: its keyframe was restored
  f.seen = clock;
  for (int i = 0; i < direct::kNumExpressions; ++i)
    f.src[i] = has_ex && direct::PlaysExpression(i) ? ex.values[i] : 0.f;
  if (wanted) f.until = clock + kFaceLinger;
  const bool owns_expressions = tune.enabled && clock < f.until;
  f.ownership_blend = std::clamp(
      f.ownership_blend + (owns_expressions ? dt / kOwnIn : -dt / kOwnOut), 0.f,
      1.f);
  // the levels to show: the engine's, faded out while a line plays (the
  // network alone shapes the speaking face; the game's emotion reaches it
  // as the network's input) and back in after it; from the moment it is
  // spoken, while it is still being generated too (not on a head that keeps
  // vanilla lip-sync)
  const bool speaking = in_line && tune.enabled &&
                        (pit->second.direct_frame >= 0 ||
                         (pit->second.entry->state == State::kPending &&
                          f.head_state != State::kFailed));
  f.speech_blend = std::clamp(
      f.speech_blend + (speaking ? dt / std::max(tune.face_in, 1e-3f)
                                 : -dt / std::max(tune.face_out, 1e-3f)),
      0.f, 1.f);
  float lv[direct::kNumExpressions];
  for (int i = 0; i < direct::kNumExpressions; ++i)
    lv[i] = f.src[i] * (1 - f.speech_blend);
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
  if (f.transition.moving() && !f.transition.stale(now))
    updates |= kFaceChanged;
  if (f.ownership_blend > 0 && f.ownership_blend < 1)
    updates |= kFaceChanged | kKeepUpdating;  // fading
  if (!owns_expressions && f.ownership_blend <= 0)
    DropFace(fg);  // back to the engine's
  return updates;
}

// Runs where the head's morphs are applied (ID 26998), around the engine's
// recompute of the face keyframes (ID 26598) -- writing earlier would be
// overwritten. Returns "changed" so the morphs are re-applied.
__attribute__((noinline)) bool AnimUpdateHook(uint8_t* fg, float dt,
                                              uint64_t flag) {
  auto orig =
      SiteOrig<AnimUpdateFn>(__builtin_return_address(0), kIdFaceAnimUpdate);
  if (fg && num_faces && ready) RestoreExpressions(fg);
  bool changed = orig(fg, dt, flag);
  if (fg && ready) {
    auto& face_lock = Field<skse::SpinLock>(fg, skse::kFgLock);
    face_lock.Lock();
    const Tune tune = SnapshotTune();
    int updates = num_playing ? Apply(fg, tune) : kNoUpdate;
    updates |= Expressions(fg, dt, tune);
    face_lock.Unlock();
    changed |= updates != 0;
    // "changed" makes the node reset its meshes; keep the update coming
    if (updates & kKeepUpdating) fg[kFgDirty] = 1;
  }
  return changed;
}

// ---------------- Direct: vertex deltas on the dynamic face meshes ---------
// The player's conversation partner's face, once its node comes by: its
// expressions are played.
void TrackPartner(uint8_t* node, uint8_t* fg) {
  uint8_t* ref = PartnerRef();
  const bool ended = !ref && partner_ref;
  const bool found =
      ref && (ref != partner_ref || fg != partner_fg) && OfRef(node, ref);
  if (!ended && !found) return;
  ScopedLock guard(lock);
  partner_fg = found ? fg : nullptr;
  partner_ref = ref;
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
    added = true;
    const std::vector<float>& d = deltas[si];
    int pi = static_cast<int>(part);
    thread_local std::vector<float> xyz;
    xyz.resize(nv * 3);
    float* data = Engine<LockDynamicFn, kIdLockDynamic>()(shape);
    for (uint32_t v = 0; v < nv; ++v)
      for (int c = 0; c < 3; ++c) {
        float dv = d[v * 3 + c];
        peak[pi] = std::fmax(peak[pi], std::fabs(dv));
        xyz[v * 3 + c] = data[v * 4 + c] + dv;
      }
    Engine<UpdateDynamicFn, kIdUpdateDynamic>()(shape, xyz.data(), false);
    Engine<UnlockDynamicFn, kIdUnlockDynamic>()(shape);
  });
  return added;
}

// Finds the face's mapping, then (after the engine rebuilt the shapes from
// the base and the morphs: reset) adds the line's Direct frame and the
// expressions to every dynamic face shape of the node.
void AddDirect(uint8_t* node, bool reset) {
  uint8_t* fg = Field<uint8_t*>(node, kNodeFg);
  if (!fg) return;
  TrackPartner(node, fg);
  {
    ScopedLock guard(lock, ScopedLock::kShared);
    if (!LiveFace(fg)) return;
  }
  const Tune tune = SnapshotTune();
  const std::string key = HeadKey(node);
  std::shared_ptr<const a2f::Line> line;
  std::shared_ptr<const direct::Head> head;
  float frame = -1;
  float levels[direct::kNumExpressions];
  bool late = false;
  State state;
  {
    ScopedLock guard(lock);
    Face* face = LiveFace(fg);
    if (!face) return;
    if (face->head_key != key) {
      face->head_key = key;
      face->head_state = State::kPending;
      face->head.reset();
      face->transition = a2f::PoseTransition();
      std::fill_n(face->level, direct::kNumExpressions, 0.f);
    }
    state = face->head_state;
    head = face->head;
    std::copy_n(face->level, direct::kNumExpressions, levels);
    auto it = playing.find(fg);
    if (tune.enabled && state == State::kReady && it != playing.end() &&
        it->second.direct_frame >= 0) {
      line = it->second.entry->line;
      frame = it->second.direct_frame;
      late = it->second.late;
    }
  }
  if (state == State::kPending) {
    state = FindHead(node, key, head);
    if (state != State::kPending) {
      ScopedLock guard(lock);
      Face* face = LiveFace(fg);
      if (!face || face->head_key != key) return;
      face->head_state = state;
      face->head = head;
    }
  }
  if (!reset || state != State::kReady || !head) return;
  // at its last frame the line is over: the pose eases out from there
  if (line && frame >= static_cast<float>(line->frames - 1)) line.reset();
  thread_local std::vector<float> coefficients;
  direct::Coefficients(*head, line.get(), frame, coefficients);
  bool speaking = false;
  {
    ScopedLock guard(lock);
    Face* face = LiveFace(fg);
    if (!face || face->head_key != key) return;
    auto it = playing.find(fg);
    const bool snap = line && it != playing.end() && it->second.snap;
    if (snap) it->second.snap = false;
    const float fade = !line  ? tune.face_out
                       : late ? tune.blend_in
                              : tune.face_in;
    speaking = face->transition.Blend(line, Now(), fade, tune.enabled, snap,
                                      coefficients);
  }
  // a head made for smaller facial motion takes the speech scaled down
  float scales[direct::kNumRegions];
  for (int r = 0; r < direct::kNumRegions; ++r)
    scales[r] = tune.direct_scale[r] * head->auto_scale[r];
  thread_local std::vector<std::vector<float>> deltas;
  if (!direct::Evaluate(*head, speaking ? coefficients.data() : nullptr, scales,
                        levels, deltas))
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

// Deltas are only ever added right after a reset to the base vertices (force,
// or the node's reset flag, which the original clears).
__attribute__((noinline)) uint64_t ApplyMorphsHook(uint8_t* node,
                                                   uint8_t force) {
  auto orig =
      SiteOrig<ApplyMorphsFn>(__builtin_return_address(0), kIdApplyMorphs);
  bool reset = force || (Field<uint16_t>(node, kNodeFlags) & 1);
  uint64_t r = orig(node, force);
  if (ready && (num_playing || num_faces || PartnerRef() != partner_ref))
    AddDirect(node, reset);
  return r;
}

// ---------------- SKSE ----------------
void OnMessage(skse::SKSEMessage* m) {
  if (m->type != skse::kMessageDataLoaded) return;
  LoadHeadPartTypes();  // before the worker, which turns Direct on
  RetakeSites();
  RegisterMenu();
  HANDLE h = CreateThread(nullptr, 0, Worker, nullptr, 0, nullptr);
  if (h) {
    SetThreadPriority(h, THREAD_PRIORITY_BELOW_NORMAL);
    CloseHandle(h);
  }
}

bool Load(const skse::SKSEInterface* iface) {
  skse::SetRuntime(iface->runtimeVersion);
  LoadIni();  // turns the log on when Logging=1
  skse::Report("Better Facial Animations %s, game runtime %08X", kVersion,
               iface->runtimeVersion);
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
  if (!Addr(kIdMorphData) || !Addr(kIdApplyMorph) || !Addr(kIdLockDynamic) ||
      !Addr(kIdUnlockDynamic) || !Addr(kIdUpdateDynamic) ||
      !HookCalls("morphs", kIdApplyMorphs, kMorphCallers,
                 reinterpret_cast<void*>(&ApplyMorphsHook)))
    return false;
  auto* msg = static_cast<skse::SKSEMessagingInterface*>(
      iface->QueryInterface(skse::kInterfaceMessaging));
  if (!msg ||
      !msg->RegisterListener(iface->GetPluginHandle(), "SKSE", OnMessage)) {
    Log("no messaging interface");
    return false;
  }
  return true;
}

}  // namespace
}  // namespace bfa

// SE 1.5.97's SKSE asks this instead of reading SKSEPlugin_Version.
extern "C" __declspec(dllexport) bool SKSEPlugin_Query(
    const skse::SKSEInterface* iface, skse::PluginInfo* info) {
  info->infoVersion = 1;
  info->name = SKSEPlugin_Version.name;
  info->version = SKSEPlugin_Version.pluginVersion;
  return !iface->isEditor && iface->runtimeVersion == skse::Version(1, 5, 97);
}

extern "C" __declspec(dllexport) bool SKSEPlugin_Load(
    const skse::SKSEInterface* iface) {
  return bfa::Load(iface);
}

BOOL WINAPI DllMain(HINSTANCE, DWORD, LPVOID) { return TRUE; }
