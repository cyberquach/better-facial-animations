// Better Facial Animations: Audio2Face lip-sync generated in the game for every
// voiced line of every NPC with a FaceGen head. The speak / response hooks
// queue a line; a worker thread reads the voice file through the engine's
// resource system, decodes it and runs the network (live_core.cc); the face
// hook plays the line and the morph hook adds its A2F Direct vertex deltas
// (direct.cc) to the head's face meshes.
#include <windows.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "direct.h"
#include "live_core.h"
#include "ort_net.h"
#include "skse.h"

using skse::Addr;
using skse::Log;

namespace {

constexpr char kVersion[] = "1.2.2";

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
  return (part[0] << 24) | (part[1] << 16) | (part[2] << 4);
}

}  // namespace

extern "C" {
__declspec(dllexport) extern skse::SKSEPluginVersionData SKSEPlugin_Version;
}
skse::SKSEPluginVersionData SKSEPlugin_Version = {
    1,
    PackVersion(kVersion),
    "BetterFacialAnimations",
    "Cyberquach",
    "",
    0,
    0,
    {0x01064920 /* 1.6.1170 */, 0},
    0};

namespace {

// ---------------- engine (AE 1.6.1170) ----------------
constexpr uint64_t kIdSpeakVoiceLine = 37542;
constexpr uint64_t kIdDialogueResponseCtor = 35249;
// play-response, UpdateInDialogue, 44673, console SpeakSound
constexpr uint64_t kSpeakCallers[] = {39772, 37541, 44673, 22864};
// DialogueItem ctor, MenuTopicManager x2
constexpr uint64_t kResponseCallers[] = {35220, 35254, 35256};
constexpr uint64_t kIdStreamCtor = 71014;  // BSResourceNiBinaryStream
constexpr uint64_t kIdStreamDtor = 71016;
// BSFaceGenAnimationData::Update(fg, dt, bool)
constexpr uint64_t kIdFaceAnimUpdate = 26598;
// BSFaceGenNiNode: update keyframes, apply morphs
constexpr uint64_t kFaceCallers[] = {26998};
// BSFaceGenNiNode::ApplyMorphs(node, force): reset to base + morphs, run by
// the "Face morphing" job (26999) and the player's face update (36564)
constexpr uint64_t kIdApplyMorphs = 26988;
constexpr uint64_t kMorphCallers[] = {26999, 36564};
constexpr uint64_t kIdMorphData = 27020;    // shape -> BSFaceGenMorphDataHead
constexpr uint64_t kIdApplyMorph = 26926;   // add one morph to a shape
constexpr uint64_t kIdLockDynamic = 70950;  // BSDynamicTriShape data lock
constexpr uint64_t kIdUnlockDynamic = 70952;
// BSDynamicTriShape::UpdateDynamicData(float3*, bool): what the engine's morph
// kernel writes through; renderers that mirror face meshes (Creation Engine
// Raytracing) hook it, so Direct writes through it too
constexpr uint64_t kIdUpdateDynamic = 70954;
constexpr uint64_t kIdSoundIsPlaying = 67620;    // bool(BSSoundHandle*)
constexpr uint64_t kIdGenerateFromPath = 69979;  // ResId from "Sound\...\X.lip"
// BSFaceGenNiNode / BSDynamicTriShape / NiObjectNET
constexpr uint32_t kNodeChildren = 0x118;
constexpr uint32_t kNodeNumChildren = 0x122;
constexpr uint32_t kNodeFg = 0x150;
constexpr uint32_t kNodeFlags = 0x160;  // u16, bit0 = reset needed
constexpr uint32_t kObjName = 0x10;
constexpr uint32_t kObjExtra = 0x20;
constexpr uint32_t kObjNumExtra = 0x28;
constexpr uint32_t kGeomType = 0x150;  // 4 = BSDynamicTriShape
constexpr uint32_t kDynNumVerts = 0x15A;
constexpr uint32_t kDynDataSize = 0x170;
constexpr uint32_t kBaseVerts = 0x18;  // BSFaceGenBaseMorphExtraData ("FOD")
constexpr uint32_t kBaseNumVerts = 0x24;
constexpr uint32_t kFgDirty = 0x218;  // byte: run the face update next frame
constexpr uint32_t kActorProcess = 0xF8;
constexpr uint32_t kProcessHigh = 0x10;
constexpr uint32_t kHighVoiceHandle = 0x2FC;
constexpr uint32_t kFormId = 0x14;
constexpr uint32_t kFormType = 0x1A;
constexpr uint32_t kRefrBase = 0x40;
constexpr uint32_t kNpcFlags = 0x38;
constexpr uint32_t kActorRace = 0x1F8;
constexpr uint32_t kRaceFlags = 0x108;
// Skyrim.esm ArgonianRace / ArgonianRaceVampire: their snout doesn't match the
// human-trained model, so they keep vanilla lip-sync.
constexpr uint32_t kArgonianRace = 0x00013740;
constexpr uint32_t kArgonianRaceVampire = 0x0008883A;
constexpr uint8_t kFtNpc = 0x2B;
constexpr uint8_t kFtActor = 0x3E;
constexpr uint32_t kVfGetFaceGenAnimationData = 0x63;
// DialogueResponse
constexpr uint32_t kRespEmoType = 0x10;
constexpr uint32_t kRespEmoValue = 0x14;
constexpr uint32_t kRespVoice = 0x18;

using SpeakFn = float (*)(uint8_t* actor, const char* voice, void* handle,
                          uint64_t emo_type, uint64_t emo_value, uint64_t,
                          uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
                          uint64_t, uint64_t, uint64_t);
using ResponseCtorFn = uint8_t* (*)(uint8_t* self, void* topic, void* info,
                                    uint8_t* speaker, void* response);
using AnimUpdateFn = bool (*)(uint8_t* fg, float dt, uint64_t flag);
using ApplyMorphsFn = uint64_t (*)(uint8_t* node, uint8_t force);
using ApplyMorphFn = void (*)(void* morph_data, int type, int index,
                              uint8_t* shape, float value);
using GenerateFromPathFn = void (*)(skse::ResId*, const char*);

constexpr char kIni[] = "Data\\SKSE\\Plugins\\BetterFacialAnimations.ini";
constexpr wchar_t kIniW[] = L"Data\\SKSE\\Plugins\\BetterFacialAnimations.ini";
constexpr wchar_t kPluginDir[] = L"Data\\SKSE\\Plugins\\BetterFacialAnimations";
constexpr char kAddressLibrary[] =
    "Data\\SKSE\\Plugins\\versionlib-1-6-1170-0.bin";

// ---------------- settings ----------------
struct Config {  // fixed after load
  bool gpu = true;
  int batch = 2, unload_after = 0, log_lines = 40, cache_lines = 300;
  std::wstring model;
};
struct Tune {  // menu
  bool enabled = true, prefetch = true, direct_expression = true,
       logging = false;
  float time_offset = 0, blend_in = 0.15f;
  float direct_scale[direct::kNumRegions] = {1, 1, 1, 1, 1, 1};
};
// A line ready later than this after its start is blended in.
constexpr float kLateStart = 0.1f;
Config config;
Tune tune;                 // scalars; read by the playback hooks without a lock
volatile bool net_loaded;  // lines keep generating while disabled (instant A/B)
SRWLOCK settings_lock = SRWLOCK_INIT;  // menu edits vs. saving
direct::Model model;
volatile bool ready;  // A2FDirect.bin loaded and the morph hook in place

// Every ini key / menu item: a slider (value, lo..hi) or, with flag set, a
// checkbox, in an ini section / menu group. Table order = menu order.
enum class Sec { kGeneral, kPlayback, kDirect, kDebug };
struct Param {
  const char* key;
  float* value;
  float lo, hi;
  const char* label;
  const char* tip;
  Sec sec;
  bool* flag;
};
const Param kParams[] = {
    {"Enabled", nullptr, 0, 0, "Enabled",
     "Play Audio2Face lip-sync. Untick to see the game's own lip-sync "
     "instantly (A/B).",
     Sec::kGeneral, &tune.enabled},
    {"Prefetch", nullptr, 0, 0, "Prefetch",
     "Start generating when a dialogue response is created, before it is "
     "spoken.",
     Sec::kGeneral, &tune.prefetch},
    {"TimeOffset", &tune.time_offset, -0.3f, 0.3f, "Sync offset (s)",
     "Shifts the animation against the voice. Positive = earlier mouth shapes.",
     Sec::kPlayback},
    {"BlendIn", &tune.blend_in, 0, 0.5f, "Blend-in (s)",
     "Crossfade from vanilla lip-sync when a line finishes generating after it "
     "started.",
     Sec::kPlayback},
    {"DirectBrows", &tune.direct_scale[direct::kBrows], 0, 2, "Brows",
     "Brow deformation.", Sec::kDirect},
    {"DirectEyes", &tune.direct_scale[direct::kEyes], 0, 2, "Eyes",
     "Eyelid / eye region deformation.", Sec::kDirect},
    {"DirectCheeks", &tune.direct_scale[direct::kCheeks], 0, 2, "Cheeks & nose",
     "Cheek and nose deformation.", Sec::kDirect},
    {"DirectMouth", &tune.direct_scale[direct::kMouth], 0, 2, "Mouth",
     "Lip and mouth skin deformation.", Sec::kDirect},
    {"DirectJaw", &tune.direct_scale[direct::kJaw], 0, 2, "Jaw & teeth",
     "Jaw, teeth and gums (Skyrim's own mouth shapes fitted to the "
     "lips and chin).",
     Sec::kDirect},
    {"DirectTongue", &tune.direct_scale[direct::kTongue], 0, 2, "Tongue",
     "Tongue deformation.", Sec::kDirect},
    {"DirectExpression", nullptr, 0, 0, "Engine expression",
     "Also play the dialogue emotion expression morph on top.", Sec::kDirect,
     &tune.direct_expression},
    {"Logging", nullptr, 0, 0, "Logging",
     "Write BetterFacialAnimations.log (My Games/Skyrim Special "
     "Edition/SKSE). Turn on to report a problem; startup is only logged "
     "when it is on at launch.",
     Sec::kDebug, &tune.logging},
};
const char* Section(const Param& p) {
  static const char* const kNames[] = {"General", "Playback", "Direct",
                                       "Debug"};
  return kNames[static_cast<int>(p.sec)];
}
// Menu defaults; the Direct scales come from A2FDirect.bin (loaded).
void ResetTune() {
  bool logging = tune.logging;  // a diagnostic switch, not a tuning value
  tune = Tune();
  tune.logging = logging;
  std::copy_n(model.default_scale, direct::kNumRegions, tune.direct_scale);
}

// Turns the log on / off as tune.logging says; a new log starts with the
// version.
void ApplyLogging() {
  if (skse::SetLogging(tune.logging))
    Log("Better Facial Animations %s", kVersion);
}

// Called at load (General) and again by the worker once the Direct scales'
// defaults are known (the ini overlays them).
void LoadIni() {
  config.gpu = GetPrivateProfileIntA("General", "GPU", 1, kIni) != 0;
  config.batch = std::max(
      1, static_cast<int>(GetPrivateProfileIntA("General", "Batch", 2, kIni)));
  config.unload_after =
      GetPrivateProfileIntA("General", "UnloadAfterIdleSeconds", 0, kIni);
  config.cache_lines = std::max(8, static_cast<int>(GetPrivateProfileIntA(
                                       "General", "CacheLines", 300, kIni)));
  config.log_lines = GetPrivateProfileIntA("Debug", "LogLines", 40, kIni);
  wchar_t m[260];
  GetPrivateProfileStringW(L"General", L"Model", L"network_fp16.onnx", m, 260,
                           kIniW);
  config.model = m;
  char b[128], d[32];
  for (const auto& p : kParams) {
    if (p.flag) {
      *p.flag = GetPrivateProfileIntA(Section(p), p.key, *p.flag, kIni) != 0;
      continue;
    }
    snprintf(d, sizeof d, "%g", *p.value);
    GetPrivateProfileStringA(Section(p), p.key, d, b, sizeof b, kIni);
    *p.value = static_cast<float>(atof(b));
  }
  ApplyLogging();
}
void SaveIni() {
  char b[64];
  for (const auto& p : kParams) {
    if (p.value) snprintf(b, sizeof b, "%.4g", *p.value);
    WritePrivateProfileStringA(Section(p), p.key,
                               p.value   ? b
                               : *p.flag ? "1"
                                         : "0",
                               kIni);
  }
}

// ---------------- generated lines ----------------
enum class State { kPending, kReady, kFailed };
struct Entry {
  State state = State::kPending;
  std::shared_ptr<const a2f::Line> line;
  uint64_t last_use = 0;
  std::string error;
};
struct Job {
  std::string cache_key, path;
  int emo, val;
  bool female;
};
constexpr int kDirectTail = 3;  // forced resets after the last Direct frame
struct Playing {
  std::shared_ptr<Entry> entry;
  std::string name;
  uint64_t lip_key;
  uint8_t* actor;
  double start;
  bool female;
  float ready_at = -1;
  bool lip_seen = false, logged = false;
  // Direct: the frame (fractional, -1 none) and blend for the morph hook
  float direct_frame = -1, direct_mix = 1;
  int direct_tail = 0;
  // log: last clock, Direct frames played / deformed, peak delta per part
  float last_t = 0;
  uint32_t direct_frames = 0, direct_adds = 0;
  float peak[direct::kNumParts] = {};
};
struct Stats {
  uint32_t lines = 0, failed = 0, late = 0;
  double audio_sec = 0, gen_sec = 0, last_ms = 0;
  std::string net_info, last_error;
};

SRWLOCK lock = SRWLOCK_INIT;  // guards everything below
CONDITION_VARIABLE queue_cv = CONDITION_VARIABLE_INIT;
std::unordered_map<std::string, std::shared_ptr<Entry>> cache;
std::deque<Job> queue;
std::unordered_map<uint8_t*, Playing> playing;  // by FaceGen animation data
volatile LONG num_playing;  // playing.size(), so the per-frame hook can skip
                            // the lock
uint64_t use_clock;
Stats stats;

double Now() {
  static LARGE_INTEGER f;
  if (!f.QuadPart) QueryPerformanceFrequency(&f);
  LARGE_INTEGER c;
  QueryPerformanceCounter(&c);
  return static_cast<double>(c.QuadPart) / f.QuadPart;
}

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
// except Argonians -> sex.
bool SpeakerOk(uint8_t* ref, bool& female) {
  if (!ref || ref[kFormType] != kFtActor) return false;
  uint8_t* base = *reinterpret_cast<uint8_t**>(ref + kRefrBase);
  uint8_t* race = *reinterpret_cast<uint8_t**>(ref + kActorRace);
  if (!base || base[kFormType] != kFtNpc || !race ||
      !(*reinterpret_cast<uint32_t*>(race + kRaceFlags) & 2))
    return false;
  uint32_t race_id = *reinterpret_cast<uint32_t*>(race + kFormId);
  if (race_id == kArgonianRace || race_id == kArgonianRaceVampire) return false;
  female = *reinterpret_cast<uint32_t*>(base + kNpcFlags) & 1;
  return true;
}

// Queue a line unless it is cached or in flight. Returns its entry.
std::shared_ptr<Entry> Request(const std::string& stem, int emo, int val,
                               bool female, bool urgent) {
  std::string key = stem;
  for (auto& c : key) c = static_cast<char>(tolower(static_cast<uint8_t>(c)));
  char sfx[32];
  snprintf(sfx, sizeof sfx, "|%c%d|%d", female ? 'f' : 'm', emo, val);
  key += sfx;
  AcquireSRWLockExclusive(&lock);
  auto& e = cache[key];
  if (!e) {
    e = std::make_shared<Entry>();
    Job j{key, stem, emo, val, female};
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
  auto r = e;
  ReleaseSRWLockExclusive(&lock);
  return r;
}

// Read a file through BSResourceNiBinaryStream (BSAs + loose files, MO2 VFS);
// path relative to Data.
bool ReadGameFile(const std::string& path, std::vector<uint8_t>& out) {
  using Ctor = void* (*)(void*, const char*, uint64_t, void*);
  using Dtor = void (*)(void*);
  using ReadFn = uint32_t (*)(void*, void*, uint32_t, uint32_t*, uint32_t);
  static Ctor ctor = reinterpret_cast<Ctor>(Addr(kIdStreamCtor));
  static Dtor dtor = reinterpret_cast<Dtor>(Addr(kIdStreamDtor));
  alignas(16) uint8_t s[0x40] = {};
  ctor(s, path.c_str(), 0, nullptr);
  // BSResource::Stream*, null when not found
  void* st = *reinterpret_cast<void**>(s + 0x20);
  bool ok = false;
  if (st) {
    uint32_t n = *reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(st) + 8);
    out.resize(n);
    auto rd = *reinterpret_cast<ReadFn*>(s + 0x10);
    uint32_t got = 0;
    while (got < n) {
      uint32_t k = rd(s, out.data() + got, n - got, nullptr, 1);
      if (!k) break;
      got += k;
    }
    ok = n > 0 && got == n;
  }
  dtor(s);
  return ok;
}

// ---------------- worker ----------------
std::unique_ptr<a2f::Net> net;  // worker thread only

std::wstring PluginDir() {
  wchar_t full[MAX_PATH];
  GetFullPathNameW(kPluginDir, MAX_PATH, full, nullptr);
  return full;
}

// Loads the network on first use; falls back to CPU when the GPU session
// fails.
bool EnsureNet() {
  if (net) return true;
  const std::wstring dir = PluginDir(), model = dir + L"\\" + config.model;
  static bool dml_loaded;
  if (!dml_loaded) {
    // onnxruntime delay-loads DirectML.dll by name: load ours first so that
    // is the one it gets
    dml_loaded = LoadLibraryExW((dir + L"\\DirectML.dll").c_str(), nullptr,
                                LOAD_WITH_ALTERED_SEARCH_PATH) != nullptr;
    if (!dml_loaded)
      Log("DirectML.dll not loaded from %ls (error %lu)", dir.c_str(),
          GetLastError());
  }
  a2f::OrtOptions o;
  o.gpu = config.gpu;
  o.batch = config.batch;
  std::string err, info;
  double t = Now();
  net = a2f::CreateOrtNet(dir, model, o, err, info);
  if (!net && o.gpu) {
    Log("GPU session failed (%s); falling back to CPU", err.c_str());
    o.gpu = false;
    net = a2f::CreateOrtNet(dir, model, o, err, info);
  }
  AcquireSRWLockExclusive(&lock);
  stats.net_info = net ? info : "failed: " + err;
  ReleaseSRWLockExclusive(&lock);
  if (!net) {
    Log("network not loaded: %s", err.c_str());
    return false;
  }
  net_loaded = true;
  Log("network loaded: %s, %ls (%.0f ms)", info.c_str(), config.model.c_str(),
      (Now() - t) * 1e3);
  return true;
}

// Generates one queued line and stores the result in the cache.
void Process(const Job& j) {
  static int logged;
  std::vector<uint8_t> bytes;
  std::string err, file;
  for (const char* ext : {".fuz", ".xwm", ".wav"})
    if (ReadGameFile(j.path + ext, bytes)) {
      file = j.path + ext;
      break;
    }
  std::shared_ptr<a2f::Line> line;
  double t = Now(), audio_sec = 0;
  a2f::Timing tm;
  if (file.empty()) {
    err = "no voice file";
  } else {
    std::vector<float> pcm;
    if (a2f::DecodeAudio(bytes.data(), bytes.size(), pcm, err) && EnsureNet()) {
      line = std::make_shared<a2f::Line>();
      audio_sec = pcm.size() / static_cast<double>(a2f::kSampleRate);
      if (!a2f::Generate(*net, pcm.data(), pcm.size(), j.emo, j.val,
                         model.Proj(j.female), *line, err, &tm))
        line.reset();
    } else if (err.empty()) {
      err = "network unavailable";
    }
  }
  double ms = (Now() - t) * 1e3;
  AcquireSRWLockExclusive(&lock);
  auto it = cache.find(j.cache_key);
  if (it != cache.end()) {
    it->second->state = line ? State::kReady : State::kFailed;
    it->second->line = line;
    it->second->error = err;
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
  ReleaseSRWLockExclusive(&lock);
  if (tune.logging && ++logged <= config.log_lines) {
    if (line) {
      Log("gen %s %c emo %d/%d: %.2f s audio, %u frames in %.0f ms (net "
          "%.0f, post %.0f)",
          file.c_str(), j.female ? 'F' : 'M', j.emo, j.val, audio_sec,
          line->frames, ms, tm.net * 1e3, tm.post * 1e3);
    } else {
      Log("skip %s: %s", j.path.c_str(), err.c_str());
    }
  }
}

DWORD WINAPI Worker(void*) {
  std::string err;
  if (!model.Load(PluginDir() + L"\\A2FDirect.bin", err)) {
    Log("A2FDirect.bin: %s", err.c_str());
    return 1;
  }
  AcquireSRWLockExclusive(&settings_lock);
  ResetTune();
  LoadIni();
  ReleaseSRWLockExclusive(&settings_lock);
  ready = true;
  if (tune.enabled) EnsureNet();  // resident from the start
  double idle_since = Now();
  for (;;) {
    AcquireSRWLockExclusive(&lock);
    while (queue.empty()) {
      SleepConditionVariableSRW(
          &queue_cv, &lock, config.unload_after > 0 && net ? 1000 : INFINITE,
          0);
      if (queue.empty() && net && config.unload_after > 0 &&
          Now() - idle_since > config.unload_after) {
        ReleaseSRWLockExclusive(&lock);
        net.reset();
        net_loaded = false;
        Log("network unloaded after %d s idle", config.unload_after);
        AcquireSRWLockExclusive(&lock);
        stats.net_info = "unloaded (idle)";
      }
    }
    Job j = queue.front();
    queue.pop_front();
    ReleaseSRWLockExclusive(&lock);
    Process(j);
    idle_since = Now();
  }
}

// ---------------- call-site hooks ----------------
// Every `call rel32` to the callee inside the caller functions is redirected
// through a thunk. Each site keeps its own original target (looked up by
// return address), so a plugin that patches a site after us is chained.
struct Site {
  uint64_t at;
  void* orig;
  void* thunk;
  const char* what;
};
Site sites[16];
int num_sites;
uint8_t* thunks;
size_t thunk_used;

// jmp [rip+0] -> fn, within rel32 reach of the exe.
void* Thunk(void* fn) {
  if (!thunks) {
    const uint64_t base = skse::ImageBase();
    for (uint64_t a = (base - 0x10000) & ~0xFFFFull;
         a > base - 0x70000000 && !thunks; a -= 0x10000)
      thunks = static_cast<uint8_t*>(
          VirtualAlloc(reinterpret_cast<void*>(a), 0x1000,
                       MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE));
    if (!thunks) return nullptr;
  }
  if (thunk_used + 16 > 0x1000) return nullptr;
  uint8_t* t = thunks + thunk_used;
  thunk_used += 16;
  t[0] = 0xFF;
  t[1] = 0x25;
  *reinterpret_cast<uint32_t*>(t + 2) = 0;
  *reinterpret_cast<void**>(t + 6) = fn;
  return t;
}
uint64_t CallTarget(uint64_t at) {
  return at + 5 + *reinterpret_cast<int32_t*>(at + 1);
}
bool PatchCall(uint64_t at, void* to) {
  int64_t rel = static_cast<int64_t>(reinterpret_cast<uint64_t>(to) - (at + 5));
  if (rel != static_cast<int32_t>(rel)) return false;
  DWORD old;
  VirtualProtect(reinterpret_cast<void*>(at + 1), 4, PAGE_EXECUTE_READWRITE,
                 &old);
  *reinterpret_cast<int32_t*>(at + 1) = static_cast<int32_t>(rel);
  VirtualProtect(reinterpret_cast<void*>(at + 1), 4, old, &old);
  return true;
}
// Original target of the site that called a hook; the callee itself for
// unknown callers.
void* SiteOrig(void* ret, uint64_t callee_id) {
  for (int i = 0; i < num_sites; ++i)
    if (sites[i].at + 5 == reinterpret_cast<uint64_t>(ret))
      return sites[i].orig;
  return reinterpret_cast<void*>(Addr(callee_id));
}
template <size_t N>
int HookCalls(const char* what, uint64_t callee_id,
              const uint64_t (&callers)[N], void* hook) {
  const uint64_t base = skse::ImageBase();
  uint64_t callee = Addr(callee_id);
  auto* nt = reinterpret_cast<IMAGE_NT_HEADERS*>(
      base + reinterpret_cast<IMAGE_DOS_HEADER*>(base)->e_lfanew);
  uint64_t end = base + nt->OptionalHeader.SizeOfImage;
  if (!callee) {
    Log("%s: address id %llu not found", what, callee_id);
    return 0;
  }
  void* thunk = Thunk(hook);
  int n = 0;
  if (!thunk) {
    Log("%s: no thunk memory", what);
    return 0;
  }
  for (uint64_t caller : callers) {
    uint64_t f = Addr(caller);
    if (!f) continue;
    for (uint64_t at = f; at < f + 0x2000 && at + 5 < end && num_sites < 16;
         ++at) {
      if (*reinterpret_cast<uint8_t*>(at) != 0xE8 || CallTarget(at) != callee ||
          !PatchCall(at, thunk))
        continue;
      sites[num_sites++] = {at, reinterpret_cast<void*>(callee), thunk, what};
      Log("%s: hooked call at rva %llx (caller id %llu)", what, at - base,
          caller);
      ++n;
    }
  }
  if (!n)
    Log("%s: no call sites found (another plugin may have replaced them)",
        what);
  return n;
}
// After all plugins loaded: chain behind anyone who patched our sites since.
void RetakeSites() {
  const uint64_t base = skse::ImageBase();
  for (int i = 0; i < num_sites; ++i) {
    Site& s = sites[i];
    bool call = *reinterpret_cast<uint8_t*>(s.at) == 0xE8;
    if (call && CallTarget(s.at) == reinterpret_cast<uint64_t>(s.thunk))
      continue;
    void* other = call ? reinterpret_cast<void*>(CallTarget(s.at)) : nullptr;
    if (call && PatchCall(s.at, s.thunk)) {
      s.orig = other;
      Log("%s: rva %llx was re-hooked by another plugin; chained", s.what,
          s.at - base);
    } else {
      Log("%s: rva %llx overwritten by another plugin; this site is not hooked",
          s.what, s.at - base);
    }
  }
}

// ---------------- hooks ----------------
__attribute__((noinline)) float SpeakHook(
    uint8_t* actor, const char* voice, void* handle, uint64_t emo_type,
    uint64_t emo_value, uint64_t a6, uint64_t a7, uint64_t a8, uint64_t a9,
    uint64_t a10, uint64_t a11, uint64_t a12, uint64_t a13, uint64_t a14) {
  auto orig = reinterpret_cast<SpeakFn>(
      SiteOrig(__builtin_return_address(0), kIdSpeakVoiceLine));
  float dur = orig(actor, voice, handle, emo_type, emo_value, a6, a7, a8, a9,
                   a10, a11, a12, a13, a14);
  bool female;
  std::string stem;
  if (!(tune.enabled || net_loaded) || !voice || !ready ||
      !SpeakerOk(actor, female) || !VoiceStem(voice, stem))
    return dur;
  auto get_fg = reinterpret_cast<uint8_t* (*)(uint8_t*)>(
      (*reinterpret_cast<void***>(actor))[kVfGetFaceGenAnimationData]);
  uint8_t* fg = get_fg(actor);
  if (!fg) return dur;
  auto e =
      Request(stem, static_cast<int>(static_cast<uint32_t>(emo_type)),
              static_cast<int>(static_cast<uint32_t>(emo_value)), female, true);
  static auto gen_path =
      reinterpret_cast<GenerateFromPathFn>(Addr(kIdGenerateFromPath));
  skse::ResId r{};
  gen_path(&r, (stem + ".lip").c_str());
  Playing p{e, stem, skse::KeyOf(r), actor, Now(), female};
  AcquireSRWLockExclusive(&lock);
  playing[fg] = p;
  num_playing = static_cast<LONG>(playing.size());
  ReleaseSRWLockExclusive(&lock);
  return dur;
}

__attribute__((noinline)) uint8_t* ResponseHook(uint8_t* self, void* topic,
                                                void* info, uint8_t* speaker,
                                                void* response) {
  auto orig = reinterpret_cast<ResponseCtorFn>(
      SiteOrig(__builtin_return_address(0), kIdDialogueResponseCtor));
  uint8_t* r = orig(self, topic, info, speaker, response);
  bool female;
  std::string stem;
  const char* voice =
      r ? *reinterpret_cast<const char**>(r + kRespVoice) : nullptr;
  if ((tune.enabled || net_loaded) && tune.prefetch && voice && ready &&
      SpeakerOk(speaker, female) && VoiceStem(voice, stem))
    Request(stem,
            static_cast<int>(*reinterpret_cast<uint32_t*>(r + kRespEmoType)),
            static_cast<int>(*reinterpret_cast<uint32_t*>(r + kRespEmoValue)),
            female, false);
  return r;
}

// The actor's voice sound is still playing (lines without .lip data have no
// engine clock to watch).
bool VoicePlaying(uint8_t* actor) {
  static auto is_playing =
      reinterpret_cast<bool (*)(void*)>(Addr(kIdSoundIsPlaying));
  uint8_t* proc = *reinterpret_cast<uint8_t**>(actor + kActorProcess);
  uint8_t* high =
      proc ? *reinterpret_cast<uint8_t**>(proc + kProcessHigh) : nullptr;
  return !high || !is_playing || is_playing(high + kHighVoiceHandle);
}

// Engine modifiers the face mesh takes over (brows, squints); the blinks
// (0, 1) stay with the engine.
constexpr int kModZeroed[] = {2, 3, 4, 5, 6, 7, 12, 13};
constexpr uint32_t kNumPhonemes = 16, kNumExpressions = 7;
inline float Mix(float engine, float ours, float mix) {
  return mix >= 1.f ? ours : engine + (ours - engine) * mix;
}

// The mesh carries the mouth, brows and eyes: zeroes the final phoneme and
// modifier keyframes (but the blinks) and writes the line's expression at
// lip time t (s) when "Engine expression" is on; mix < 1 blends from the
// engine's values.
void PlayFrame(uint8_t* fg, const a2f::Line& line, float t, float mix) {
  auto* phon = reinterpret_cast<skse::KeyframeMultiple*>(fg + skse::kFgPhon3);
  auto* mod = reinterpret_cast<skse::KeyframeMultiple*>(fg + skse::kFgMod3);
  auto* ex = reinterpret_cast<skse::KeyframeMultiple*>(fg + skse::kFgExpr3);
  if (phon->values && phon->count >= kNumPhonemes) {
    for (uint32_t i = 0; i < kNumPhonemes; ++i)
      phon->values[i] = Mix(phon->values[i], 0.f, mix);
    phon->is_updated = true;
  }
  if (mod->values && mod->count >= 14) {
    for (int e : kModZeroed) mod->values[e] = Mix(mod->values[e], 0.f, mix);
    mod->is_updated = true;
  }
  if (ex->values && ex->count >= kNumExpressions) {
    float level = 0;
    if (tune.direct_expression && !line.expression.empty()) {
      float f = std::fmax(0.f, (t + tune.time_offset) * a2f::kFps);
      uint32_t i0 = static_cast<uint32_t>(f), last = line.frames - 1;
      float a = f - static_cast<float>(i0);
      // past the end: hold the last frame, easing out over 0.1 s
      level =
          i0 >= last
              ? line.expression[last] * std::fmax(0.f, 1.f - (f - last) / 3.f)
              : (1 - a) * line.expression[i0] + a * line.expression[i0 + 1];
    }
    for (uint32_t i = 0; i < kNumExpressions; ++i)
      ex->values[i] = Mix(ex->values[i], i == line.expr ? level : 0.f, mix);
    ex->is_updated = true;
  }
}

volatile LONG logged_plays, logged_ends;

// Plays the head's current line for this frame. Returns bit 0: keyframes
// written, bit 1: the meshes must be rebuilt (Direct deltas added or, for a
// few frames after them, cleared).
int Apply(uint8_t* fg) {
  AcquireSRWLockExclusive(&lock);
  auto it = playing.find(fg);
  if (it == playing.end()) {
    ReleaseSRWLockExclusive(&lock);
    return 0;
  }
  Playing& p = it->second;
  const skse::ResId* act = skse::ActiveLine(fg);
  float t = static_cast<float>(Now() - p.start);
  bool done = t > 120;
  if (act && skse::KeyOf(*act) == p.lip_key) {  // engine lip clock
    t = *reinterpret_cast<float*>(fg + skse::kFgPhon1Timer);
    p.lip_seen = true;
  } else if (p.lip_seen) {  // the engine ended or replaced the line
    done = true;
  } else if (t > 0.5f && !VoicePlaying(p.actor)) {
    done = true;  // no .lip: wall clock while the voice plays
  }
  std::shared_ptr<const a2f::Line> line;
  if (!done) {
    if (p.entry->state == State::kFailed) {
      done = true;
    } else if (p.entry->state == State::kReady) {
      line = p.entry->line;
      if (t > line->frames / static_cast<float>(a2f::kFps) + 0.3f) done = true;
    }
  }
  bool direct = !done && line && tune.enabled;
  if (!direct) {
    p.direct_frame = -1;
    bool tail = p.direct_tail > 0;
    if (tail) --p.direct_tail;
    if (done && !tail) {
      if (p.logged && tune.logging &&
          InterlockedIncrement(&logged_ends) <= config.log_lines)
        Log("end %s: at %.2f s, %u Direct frames, deformed %u, peak head %.3f "
            "mouth %.3f eyes %.3f",
            p.name.c_str(), p.last_t, p.direct_frames, p.direct_adds, p.peak[0],
            p.peak[1], p.peak[2]);
      playing.erase(it);
      num_playing = static_cast<LONG>(playing.size());
    }
    // disabled (A/B) or still generating: vanilla lip-sync meanwhile
    ReleaseSRWLockExclusive(&lock);
    return tail ? 2 : 0;
  }
  if (p.ready_at < 0) {
    p.ready_at = t;
    if (t > kLateStart) ++stats.late;
  }
  float mix =
      p.ready_at <= kLateStart
          ? 1.f
          : std::clamp((t - p.ready_at) / std::max(tune.blend_in, 1e-3f), 0.f,
                       1.f);
  p.last_t = t;
  ++p.direct_frames;
  p.direct_frame =
      std::fmax(0.f, (t + tune.time_offset) * static_cast<float>(a2f::kFps));
  p.direct_mix = mix;
  p.direct_tail = kDirectTail;
  bool first = !p.logged;
  p.logged = true;
  std::string name = first ? p.name : std::string();
  float ready_at = p.ready_at;
  ReleaseSRWLockExclusive(&lock);
  if (first && tune.logging &&
      InterlockedIncrement(&logged_plays) <= config.log_lines)
    Log("play %s: %u frames, %s clock, ready at %.2f s", name.c_str(),
        line->frames, act ? "lip" : "wall", ready_at);
  PlayFrame(fg, *line, t, mix);
  return 3;
}

// Runs where the head's morphs are applied (ID 26998), right after the engine
// recomputes the face keyframes (ID 26598) -- writing earlier would be
// overwritten. Returns "changed" so the morphs are re-applied.
__attribute__((noinline)) bool AnimUpdateHook(uint8_t* fg, float dt,
                                              uint64_t flag) {
  auto orig = reinterpret_cast<AnimUpdateFn>(
      SiteOrig(__builtin_return_address(0), kIdFaceAnimUpdate));
  bool changed = orig(fg, dt, flag);
  if (fg && num_playing) {
    auto* face_lock = reinterpret_cast<skse::SpinLock*>(fg + skse::kFgLock);
    face_lock->Lock();
    int r = Apply(fg);
    face_lock->Unlock();
    changed |= r != 0;
    // "changed" makes the node reset its meshes; keep the update coming
    if (r & 2) fg[kFgDirty] = 1;
  }
  return changed;
}

// ---------------- Direct: vertex deltas on the dynamic face meshes ---------
// One mapping per distinct mesh (sex, part, vertex count, hash of the base
// vertices), built on a thread of its own; until it exists the mesh is left
// alone.
struct MapJob {
  std::string key;
  bool female;
  direct::Part part;
  uint32_t nv;
  bool lips;                        // the head skin: align the lips
  std::vector<float> base, bigaah;  // nv x 3; bigaah may be empty
};
SRWLOCK map_lock = SRWLOCK_INIT;
CONDITION_VARIABLE map_cv = CONDITION_VARIABLE_INIT;
std::unordered_map<std::string, std::shared_ptr<const direct::Mapping>>
    mappings;  // null while building
std::deque<MapJob> map_queue;
bool mapper_started;

direct::Part Classify(const char* name) {
  std::string n = name;
  for (auto& c : n) c = static_cast<char>(tolower(static_cast<uint8_t>(c)));
  if (n.find("mouth") != std::string::npos) return direct::Part::kMouth;
  if (n.find("eyes") != std::string::npos) return direct::Part::kEyes;
  for (const char* w : {"head", "brows", "beard"})
    if (n.find(w) != std::string::npos) return direct::Part::kHead;
  return direct::Part::kNone;
}

// The head skin itself (not brows / beard), whose lips the mapping aligns.
bool IsHeadSkin(const char* name) {
  std::string n = name;
  for (auto& c : n) c = static_cast<char>(tolower(static_cast<uint8_t>(c)));
  return n.find("head") != std::string::npos &&
         n.find("brow") == std::string::npos &&
         n.find("beard") == std::string::npos;
}

// Base (pre-morph) vertices of a face shape: its "FOD" extra data.
const float* BaseVerts(uint8_t* shape, uint32_t& nv) {
  uint8_t** extra = *reinterpret_cast<uint8_t***>(shape + kObjExtra);
  uint16_t n = *reinterpret_cast<uint16_t*>(shape + kObjNumExtra);
  for (uint16_t i = 0; extra && i < n; ++i) {
    uint8_t* e = extra[i];
    const char* name =
        e ? *reinterpret_cast<const char**>(e + kObjName) : nullptr;
    if (!name || strcmp(name, "FOD")) continue;
    nv = *reinterpret_cast<uint32_t*>(e + kBaseNumVerts);
    return *reinterpret_cast<const float**>(e + kBaseVerts);
  }
  return nullptr;
}

// The shape's own BigAah morph (engine morph type 0 = phoneme, index 1), read
// by applying it at 1.0 and undoing it; empty when the shape has no morphs.
void BigAah(uint8_t* shape, uint32_t nv, std::vector<float>& out) {
  static auto morph_data =
      reinterpret_cast<void* (*)(uint8_t*)>(Addr(kIdMorphData));
  static auto apply = reinterpret_cast<ApplyMorphFn>(Addr(kIdApplyMorph));
  static auto lock_dyn =
      reinterpret_cast<float* (*)(uint8_t*)>(Addr(kIdLockDynamic));
  static auto unlock_dyn =
      reinterpret_cast<void (*)(uint8_t*)>(Addr(kIdUnlockDynamic));
  void* md = morph_data(shape);
  if (!md) return;
  std::vector<float> before(static_cast<size_t>(nv) * 4);
  float* data = lock_dyn(shape);
  memcpy(before.data(), data, before.size() * sizeof(float));
  unlock_dyn(shape);
  apply(md, 0, 1, shape, 1.f);
  out.resize(static_cast<size_t>(nv) * 3);
  data = lock_dyn(shape);
  for (uint32_t i = 0; i < nv; ++i)
    for (int c = 0; c < 3; ++c)
      out[i * 3 + c] = data[i * 4 + c] - before[i * 4 + c];
  memcpy(data, before.data(), before.size() * sizeof(float));
  unlock_dyn(shape);
}

DWORD WINAPI Mapper(void*) {
  for (;;) {
    AcquireSRWLockExclusive(&map_lock);
    while (map_queue.empty())
      SleepConditionVariableSRW(&map_cv, &map_lock, INFINITE, 0);
    MapJob j = std::move(map_queue.front());
    map_queue.pop_front();
    ReleaseSRWLockExclusive(&map_lock);
    const direct::SexModel& sm = model.sex[j.female ? 0 : 1];
    const direct::RefMesh& ref = j.part == direct::Part::kHead    ? sm.head
                                 : j.part == direct::Part::kMouth ? sm.mouth
                                                                  : sm.eyes;
    float frame[3];
    direct::FrameOffset(ref, j.base.data(), j.nv, frame);
    for (uint32_t i = 0; i < j.nv; ++i)
      for (int c = 0; c < 3; ++c) j.base[i * 3 + c] += frame[c];
    std::vector<uint8_t> labels;
    std::vector<float> moved;
    float off[3] = {};
    if (!j.bigaah.empty()) {
      direct::LipLabels(ref, j.base.data(), j.bigaah.data(), j.nv, j.lips,
                        labels, moved, off);
    } else if (j.part == direct::Part::kMouth) {  // teeth / tongue: nearest
      labels.resize(j.nv);
      for (uint32_t i = 0; i < j.nv; ++i) {
        float best = INFINITY;
        for (uint32_t v = 0; v < ref.nv; ++v) {
          float d = 0;
          for (int c = 0; c < 3; ++c) {
            float e = j.base[i * 3 + c] - ref.verts[v * 3 + c];
            d += e * e;
          }
          if (d < best) best = d, labels[i] = ref.labels[v];
        }
      }
    }
    double t = Now();
    auto m = std::make_shared<direct::Mapping>();
    direct::BuildMapping(ref, j.base.data(), j.nv,
                         labels.empty() ? nullptr : labels.data(), *m,
                         moved.empty() ? nullptr : moved.data());
    Log("direct map %s: %u verts, %s, dist mean %.3f max %.3f, %u snapped, "
        "frame %.2f %.2f %.2f, upper lip %.2f %.2f %.2f, %s labels, %.1f ms",
        j.key.c_str(), j.nv, m->identity ? "identity" : "resampled",
        m->mean_dist, m->max_dist, m->snapped, frame[0], frame[1], frame[2],
        off[0], off[1], off[2],
        labels.empty()     ? "no"
        : j.bigaah.empty() ? "nearest"
                           : "BigAah",
        (Now() - t) * 1e3);
    AcquireSRWLockExclusive(&map_lock);
    mappings[j.key] = m;
    ReleaseSRWLockExclusive(&map_lock);
  }
}

// The shape's mapping, or null (queued to the mapper on first sight).
std::shared_ptr<const direct::Mapping> FindMapping(uint8_t* shape, bool female,
                                                   direct::Part part,
                                                   uint32_t nv, bool lips) {
  uint32_t nb = 0;
  const float* base = BaseVerts(shape, nb);
  if (!base || nb != nv) return nullptr;
  uint64_t h = 1469598103934665603ull;  // FNV-1a over every 4th vertex
  for (uint32_t i = 0; i < nv; i += 4)
    for (int c = 0; c < 3; ++c) {
      uint32_t b;
      memcpy(&b, &base[i * 3 + c], 4);
      h = (h ^ b) * 1099511628211ull;
    }
  char key[48];
  snprintf(key, sizeof key, "%c/%s/%u/%016llx", female ? 'f' : 'm',
           part == direct::Part::kHead    ? "head"
           : part == direct::Part::kMouth ? "mouth"
                                          : "eyes",
           nv, static_cast<unsigned long long>(h));
  AcquireSRWLockShared(&map_lock);
  auto it = mappings.find(key);
  bool known = it != mappings.end();
  std::shared_ptr<const direct::Mapping> m = known ? it->second : nullptr;
  ReleaseSRWLockShared(&map_lock);
  if (known) return m;
  MapJob j{key, female, part, nv, lips,
           std::vector<float>(base, base + nv * 3), {}};
  if (part == direct::Part::kHead) BigAah(shape, nv, j.bigaah);
  AcquireSRWLockExclusive(&map_lock);
  if (mappings.emplace(key, nullptr).second) {
    map_queue.push_back(std::move(j));
    if (!mapper_started) {
      HANDLE th = CreateThread(nullptr, 0, Mapper, nullptr, 0, nullptr);
      mapper_started = th != nullptr;
      if (th) CloseHandle(th);
    }
    WakeConditionVariable(&map_cv);
  }
  ReleaseSRWLockExclusive(&map_lock);
  return nullptr;
}

// Adds the head's Direct frame to every dynamic face shape of the node (after
// the engine rebuilt them from the base and the morphs).
void AddDirect(uint8_t* node) {
  static auto lock_dyn =
      reinterpret_cast<float* (*)(uint8_t*)>(Addr(kIdLockDynamic));
  static auto unlock_dyn =
      reinterpret_cast<void (*)(uint8_t*)>(Addr(kIdUnlockDynamic));
  static auto update_dyn = reinterpret_cast<void (*)(uint8_t*, float*, bool)>(
      Addr(kIdUpdateDynamic));
  uint8_t* fg = *reinterpret_cast<uint8_t**>(node + kNodeFg);
  if (!fg) return;
  std::shared_ptr<const a2f::Line> line;
  float frame = -1, mix = 1;
  bool female = false;
  AcquireSRWLockShared(&lock);
  auto it = playing.find(fg);
  if (it != playing.end() && it->second.direct_frame >= 0) {
    const Playing& p = it->second;
    line = p.entry->line;
    frame = p.direct_frame;
    mix = p.direct_mix;
    female = p.female;
  }
  ReleaseSRWLockShared(&lock);
  if (!line || frame < 0) return;
  const direct::SexModel& sm = model.sex[female ? 0 : 1];
  direct::Track track{line->direct.data(), line->frames, line->direct_dim, &sm};
  thread_local direct::Deltas deltas;
  bool have = false;
  float peak[direct::kNumParts] = {};
  uint8_t** kids = *reinterpret_cast<uint8_t***>(node + kNodeChildren);
  uint16_t n = *reinterpret_cast<uint16_t*>(node + kNodeNumChildren);
  for (uint16_t i = 0; kids && i < n; ++i) {
    uint8_t* shape = kids[i];
    if (!shape || shape[kGeomType] != 4) continue;
    const char* name = *reinterpret_cast<const char**>(shape + kObjName);
    direct::Part part = name ? Classify(name) : direct::Part::kNone;
    if (part == direct::Part::kNone) continue;
    uint32_t nv = *reinterpret_cast<uint16_t*>(shape + kDynNumVerts);
    if (*reinterpret_cast<uint32_t*>(shape + kDynDataSize) != nv * 16) continue;
    auto m = FindMapping(shape, female, part, nv, IsHeadSkin(name));
    if (!m) continue;
    if (!have) {
      direct::Evaluate(track, frame, tune.direct_scale, deltas);
      if (mix < 1)
        for (auto& d : deltas.part)
          for (float& v : d) v *= mix;
      have = true;
    }
    const auto& d = deltas.part[static_cast<int>(part)];
    for (float v : d)
      peak[static_cast<int>(part)] =
          std::fmax(peak[static_cast<int>(part)], std::fabs(v));
    thread_local std::vector<float> xyzw, xyz;
    xyzw.resize(nv * 4);
    xyz.resize(nv * 3);
    float* data = lock_dyn(shape);
    std::memcpy(xyzw.data(), data, nv * 16);
    direct::AddDeltas(*m, d, xyzw.data());
    for (uint32_t v = 0; v < nv; ++v)
      std::memcpy(&xyz[v * 3], &xyzw[v * 4], 12);
    update_dyn(shape, xyz.data(), false);
    unlock_dyn(shape);
  }
  if (!have) return;
  AcquireSRWLockExclusive(&lock);
  it = playing.find(fg);
  if (it != playing.end()) {
    ++it->second.direct_adds;
    for (int i = 0; i < direct::kNumParts; ++i)
      it->second.peak[i] = std::fmax(it->second.peak[i], peak[i]);
  }
  ReleaseSRWLockExclusive(&lock);
}

// Deltas are only ever added right after a reset to the base vertices (force,
// or the node's reset flag, which the original clears).
__attribute__((noinline)) uint64_t ApplyMorphsHook(uint8_t* node,
                                                   uint8_t force) {
  auto orig = reinterpret_cast<ApplyMorphsFn>(
      SiteOrig(__builtin_return_address(0), kIdApplyMorphs));
  bool reset = force || (*reinterpret_cast<uint16_t*>(node + kNodeFlags) & 1);
  uint64_t r = orig(node, force);
  if (reset && num_playing && ready) AddDirect(node);
  return r;
}

// ---------------- SKSE Menu Framework page (optional; cimgui exports) -------
struct ImVec2 {
  float x, y;
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
} ig;

void RenderMenu() {
  AcquireSRWLockShared(&lock);
  Stats st = stats;
  size_t cached = cache.size(), queued = queue.size();
  ReleaseSRWLockShared(&lock);
  ig.SeparatorText("Status");
  if (!ready)
    ig.Text(
        "Not running: turn on Logging, restart the game and see "
        "BetterFacialAnimations.log (A2FDirect.bin missing?)");
  ig.Text("Model: %s", st.net_info.c_str());
  ig.Text("Lines generated: %u (%u failed / no audio), cached %zu, queued %zu",
          st.lines, st.failed, cached, queued);
  if (st.lines)
    ig.Text("Speed: %.0fx realtime, last line %.0f ms; started late: %u",
            st.audio_sec / std::max(st.gen_sec, 1e-6), st.last_ms, st.late);
  if (!st.last_error.empty())
    ig.Text("Last skipped: %s", st.last_error.c_str());
  static const char* const kHeaders[] = {nullptr, "Playback", "Direct",
                                         "Debug"};
  Sec cur = Sec::kGeneral;
  for (const auto& p : kParams) {
    if (p.sec != cur) ig.SeparatorText(kHeaders[static_cast<int>(cur = p.sec)]);
    AcquireSRWLockExclusive(&settings_lock);
    p.flag ? ig.Checkbox(p.label, p.flag)
           : ig.SliderFloat(p.label, p.value, p.lo, p.hi, "%.3g", 0);
    ReleaseSRWLockExclusive(&settings_lock);
    if (ig.IsItemHovered(0)) ig.SetTooltip("%s", p.tip);
  }
  ApplyLogging();
  ig.SeparatorText("");
  if (ig.Button("Save to BetterFacialAnimations.ini", ImVec2{0, 0})) {
    AcquireSRWLockShared(&settings_lock);
    SaveIni();
    ReleaseSRWLockShared(&settings_lock);
  }
  ig.SameLine(0, -1);
  if (ig.Button("Defaults", ImVec2{0, 0}) && ready) {
    AcquireSRWLockExclusive(&settings_lock);
    ResetTune();
    ReleaseSRWLockExclusive(&settings_lock);
  }
  ig.SameLine(0, -1);
  if (ig.Button("Clear cache", ImVec2{0, 0})) {
    AcquireSRWLockExclusive(&lock);
    for (auto it = cache.begin(); it != cache.end();)
      it = it->second->state == State::kPending ? std::next(it)
                                                : cache.erase(it);
    ReleaseSRWLockExclusive(&lock);
    AcquireSRWLockExclusive(&map_lock);
    for (auto it = mappings.begin(); it != mappings.end();)
      it = it->second ? mappings.erase(it) : std::next(it);
    ReleaseSRWLockExclusive(&map_lock);
  }
}

// Resolves an export into fn; false when missing.
template <typename Fn>
bool Import(HMODULE m, const char* name, Fn& fn) {
  fn = reinterpret_cast<Fn>(GetProcAddress(m, name));
  return fn != nullptr;
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
  add("Better Facial Animations/Settings", RenderMenu);
  Log("menu page registered (SKSE Menu Framework)");
}

// ---------------- SKSE ----------------
void OnMessage(skse::SKSEMessage* m) {
  if (m->type != skse::kMessageDataLoaded) return;
  RetakeSites();
  RegisterMenu();
  HANDLE h = CreateThread(nullptr, 0, Worker, nullptr, 0, nullptr);
  if (h) {
    SetThreadPriority(h, THREAD_PRIORITY_BELOW_NORMAL);
    CloseHandle(h);
  }
}

}  // namespace

extern "C" __declspec(dllexport) bool SKSEPlugin_Load(
    const skse::SKSEInterface* iface) {
  LoadIni();  // turns the log on when Logging=1
  Log("runtime %08X", iface->runtimeVersion);
  if (!skse::LoadAddressLibrary(kAddressLibrary)) {
    Log("address library missing/unreadable");
    return false;
  }
  if (!Addr(kIdGenerateFromPath) || !Addr(kIdStreamCtor) ||
      !Addr(kIdStreamDtor)) {
    Log("address ids not found");
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

BOOL WINAPI DllMain(HINSTANCE, DWORD, LPVOID) { return TRUE; }
