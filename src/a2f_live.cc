// A2FLive: Audio2Face lip-sync generated in the game for every voiced line of
// every NPC with a FaceGen head. The speak / response hooks queue a line; a
// worker thread reads the voice file through the engine's resource system,
// decodes it and runs the network + solve (live_core.cc); the face hook plays
// the curves in the head's FaceGen update.
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

#include "live_core.h"
#include "ort_net.h"
#include "skse.h"

using skse::Addr;
using skse::Log;

namespace {

constexpr char kVersion[] = "1.0.1";

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
    1, PackVersion(kVersion),          "A2FLive", "Cyberquach", "", 0,
    0, {0x01064920 /* 1.6.1170 */, 0}, 0};

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
constexpr uint64_t kIdSoundIsPlaying = 67620;    // bool(BSSoundHandle*)
constexpr uint64_t kIdGenerateFromPath = 69979;  // ResId from "Sound\...\X.lip"
constexpr uint32_t kActorProcess = 0xF8;
constexpr uint32_t kProcessHigh = 0x10;
constexpr uint32_t kHighVoiceHandle = 0x2FC;
constexpr uint32_t kFormType = 0x1A;
constexpr uint32_t kRefrBase = 0x40;
constexpr uint32_t kNpcFlags = 0x38;
constexpr uint32_t kActorRace = 0x1F8;
constexpr uint32_t kRaceFlags = 0x108;
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
using GenerateFromPathFn = void (*)(skse::ResId*, const char*);

constexpr char kIni[] = "Data\\SKSE\\Plugins\\A2FLive.ini";
constexpr wchar_t kIniW[] = L"Data\\SKSE\\Plugins\\A2FLive.ini";
constexpr wchar_t kPluginDir[] = L"Data\\SKSE\\Plugins\\A2FLive";
constexpr char kAddressLibrary[] =
    "Data\\SKSE\\Plugins\\versionlib-1-6-1170-0.bin";

// ---------------- settings ----------------
struct Config {  // fixed after load
  bool gpu = true;
  int batch = 2, unload_after = 0, log_lines = 40, cache_lines = 300;
  std::wstring model;
};
struct Tune {  // menu
  bool enabled = true, prefetch = true;
  float time_offset = 0, blend_in = 0.15f, mouth = 1, brows = 1, expression = 1;
};
// A line ready later than this after its start is blended in.
constexpr float kLateStart = 0.1f;
Config config;
Tune tune;                 // scalars; read by the playback hooks without a lock
volatile bool net_loaded;  // lines keep generating while disabled (instant A/B)
a2f::Settings gen;         // generation settings; the worker snapshots them
volatile LONG gen_version = 1;  // bumped when gen changes: lines regenerate
SRWLOCK settings_lock = SRWLOCK_INIT;  // guards gen
a2f::Data data;
volatile bool ready;  // data file loaded

// Every ini key / menu item: a slider (value, lo..hi) or, with flag set, a
// checkbox. gen: applies to newly generated lines (ini section Generation;
// otherwise Playback for sliders, General for checkboxes). Table order = menu
// order.
struct Param {
  const char* key;
  float* value;
  float lo, hi;
  const char* label;
  const char* tip;
  bool gen;
  bool* flag;
};
const Param kParams[] = {
    {"Enabled", nullptr, 0, 0, "Enabled",
     "Play Audio2Face lip-sync. Untick to see the game's own lip-sync "
     "instantly "
     "(A/B).",
     false, &tune.enabled},
    {"Prefetch", nullptr, 0, 0, "Prefetch",
     "Start generating when a dialogue response is created, before it is "
     "spoken.",
     false, &tune.prefetch},
    {"EmotionToA2F", nullptr, 0, 0, "Emotion to Audio2Face",
     "Feed the line's dialogue emotion to the network.", true,
     &gen.emotion_to_a2f},
    {"Expression", nullptr, 0, 0, "Expression morph",
     "Drive the Dialogue<Emotion> expression morph.", true, &gen.expression_on},
    {"Tongue", nullptr, 0, 0, "Tongue layer",
     "Use Audio2Face's tongue to pick DST / Th shapes.", true, &gen.tongue_on},
    {"MouthStrength", &tune.mouth, 0, 2, "Mouth",
     "Scales the phoneme (mouth) curves at playback.", false},
    {"BrowStrength", &tune.brows, 0, 2, "Brows / blinks",
     "Scales brow, blink and squint curves at playback.", false},
    {"ExpressionStrength", &tune.expression, 0, 2, "Expression",
     "Scales the dialogue emotion expression at playback.", false},
    {"TimeOffset", &tune.time_offset, -0.3f, 0.3f, "Sync offset (s)",
     "Shifts the animation against the voice. Positive = earlier mouth shapes.",
     false},
    {"BlendIn", &tune.blend_in, 0, 0.5f, "Blend-in (s)",
     "Crossfade from vanilla lip-sync when a line finishes generating after it "
     "started.",
     false},
    {"Strength", &gen.strength, 0.3f, 2.5f, "Solve strength",
     "How far the Skyrim morphs follow the Audio2Face face.", true},
    {"Smooth", &gen.smooth, 0, 0.8f, "Smoothing",
     "Temporal smoothing of the morph curves (zero phase).", true},
    {"Ridge", &gen.ridge, 0, 0.2f, "Regularisation",
     "Penalty on morph weights; higher = fewer, softer shapes.", true},
    {"FeatureWeight", &gen.feature_weight, 0, 100, "Lip feature weight",
     "Weight of lip opening / width / protrusion matching.", true},
    {"InputStrength", &gen.input_strength, 0.5f, 2, "Audio gain",
     "Audio level fed to the network (Audio2Face input strength).", true},
    {"SkinStrength", &gen.skin_strength, 0.3f, 2, "A2F skin strength",
     "Audio2Face skin animation strength.", true},
    {"LowerFaceStrength", &gen.lower_face_strength, 0.3f, 2.5f,
     "A2F lower face", "Audio2Face mouth / jaw strength.", true},
    {"UpperFaceStrength", &gen.upper_face_strength, 0, 2, "A2F upper face",
     "Audio2Face brow / eye strength.", true},
    {"LowerFaceSmoothing", &gen.lower_smoothing, 0, 0.05f,
     "A2F lower smoothing", "Audio2Face lower face smoothing half-life (s).",
     true},
    {"UpperFaceSmoothing", &gen.upper_smoothing, 0, 0.05f,
     "A2F upper smoothing", "Audio2Face upper face smoothing half-life (s).",
     true},
    {"TongueStrength", &gen.tongue_strength, 0, 3, "A2F tongue",
     "Audio2Face tongue strength (drives the tongue phonemes).", true},
    {"TongueShare", &gen.tongue_share, 0, 1, "Tongue share",
     "How much jaw opening moves to tongue shapes (DST / Th).", true},
    {"EmotionFloor", &gen.a2f_floor, 0, 1, "Emotion floor",
     "Minimum emotion intensity sent to Audio2Face for emotional lines.", true},
    {"EmotionGain", &gen.a2f_gain, 0, 3, "Emotion gain",
     "Dialogue emotion -> Audio2Face emotion intensity.", true},
    {"EmotionCap", &gen.a2f_cap, 0, 2, "Emotion cap",
     "Maximum Audio2Face emotion.", true},
    {"ExpressionFloor", &gen.expr_floor, 0, 1, "Expression floor",
     "Minimum expression morph level for emotional lines.", true},
    {"ExpressionGain", &gen.expr_gain, 0, 3, "Expression gain",
     "Dialogue emotion -> expression morph level.", true},
    {"Emphasis", &gen.emphasis, 0, 1, "Expression emphasis",
     "How much the expression follows loudness.", true},
    {"EmotionOnset", &gen.onset, 0, 1, "Emotion onset",
     "Emotion level at the start / end of a line.", true},
    {"EmotionAttack", &gen.attack, 0.01f, 2, "Emotion attack (s)",
     "Ease-in time of the emotion.", true},
    {"EmotionRelease", &gen.release, 0.01f, 2, "Emotion release (s)",
     "Ease-out time of the emotion.", true},
};
const char* Section(const Param& p) {
  return p.gen ? "Generation" : p.flag ? "General" : "Playback";
}

// Called at load (General) and again by the worker once the data file's
// defaults are in gen (overlay the ini on them).
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
  // Optional per-channel overrides: Name=gain offset gamma max cost solve
  for (int j = 0; j < a2f::kNumSolve; ++j) {
    if (!GetPrivateProfileStringA("Channels", a2f::kChannelNames[j], "", b,
                                  sizeof b, kIni))
      continue;
    int solve = gen.solve[j];
    sscanf(b, "%f %f %f %f %f %d", &gen.gain[j], &gen.offset[j], &gen.gamma[j],
           &gen.max[j], &gen.cost[j], &solve);
    gen.solve[j] = solve != 0;
  }
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
  LONG version = 0;
  std::shared_ptr<const a2f::Line> line;
  uint64_t last_use = 0;
  std::string error;
};
struct Job {
  std::string cache_key, path;
  int emo, val;
  bool female;
};
struct Playing {
  std::shared_ptr<Entry> entry;
  std::string name;
  uint64_t lip_key;
  uint8_t* actor;
  double start;
  float ready_at = -1;
  bool lip_seen = false, logged = false;
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

// Actor with a FaceGen head (people; animals and creatures have none) -> sex.
bool SpeakerOk(uint8_t* ref, bool& female) {
  if (!ref || ref[kFormType] != kFtActor) return false;
  uint8_t* base = *reinterpret_cast<uint8_t**>(ref + kRefrBase);
  uint8_t* race = *reinterpret_cast<uint8_t**>(ref + kActorRace);
  if (!base || base[kFormType] != kFtNpc || !race ||
      !(*reinterpret_cast<uint32_t*>(race + kRaceFlags) & 2))
    return false;
  female = *reinterpret_cast<uint32_t*>(base + kNpcFlags) & 1;
  return true;
}

// Queue a line unless it is cached or in flight with the current settings.
// Returns its entry.
std::shared_ptr<Entry> Request(const std::string& stem, int emo, int val,
                               bool female, bool urgent) {
  std::string key = stem;
  for (auto& c : key) c = static_cast<char>(tolower(static_cast<uint8_t>(c)));
  char sfx[32];
  snprintf(sfx, sizeof sfx, "|%c%d|%d", female ? 'f' : 'm', emo, val);
  key += sfx;
  LONG ver = gen_version;
  AcquireSRWLockExclusive(&lock);
  auto& e = cache[key];
  if (!e || (e->version != ver && e->state != State::kPending)) {
    e = std::make_shared<Entry>();
    e->version = ver;
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
      AcquireSRWLockShared(&settings_lock);
      a2f::Settings s = gen;
      ReleaseSRWLockShared(&settings_lock);
      line = std::make_shared<a2f::Line>();
      audio_sec = pcm.size() / static_cast<double>(a2f::kSampleRate);
      if (!a2f::Generate(data, *net, s, pcm.data(), pcm.size(), j.emo, j.val,
                         j.female, *line, err, &tm))
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
  if (++logged <= config.log_lines) {
    if (line) {
      Log("gen %s %c emo %d/%d: %.2f s audio, %u frames in %.0f ms (net %.0f, "
          "solve %.0f)",
          file.c_str(), j.female ? 'F' : 'M', j.emo, j.val, audio_sec,
          line->frames, ms, tm.net * 1e3, tm.solve * 1e3);
    } else {
      Log("skip %s: %s", j.path.c_str(), err.c_str());
    }
  }
}

DWORD WINAPI Worker(void*) {
  std::string err;
  if (!data.Load(PluginDir() + L"\\A2FLive.dat", err)) {
    Log("A2FLive.dat: %s", err.c_str());
    return 1;
  }
  AcquireSRWLockExclusive(&settings_lock);
  gen = data.defaults;
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
  Playing p{e, stem, skse::KeyOf(r), actor, Now()};
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

// Line modifier order (BlinkL/R, BrowDownL/R, BrowInL/R, BrowUpL/R,
// SquintL/R) -> engine modifier index.
constexpr int kModEngine[a2f::kNumMod] = {0, 1, 2, 3, 4, 5, 6, 7, 12, 13};
inline float Mix(float engine, float ours, float mix) {
  return mix >= 1.f ? ours : engine + (ours - engine) * mix;
}

// Write the line's frame at lip time t (s) into the final phoneme / modifier /
// expression keyframes; mix < 1 blends from the engine's values.
void PlayFrame(uint8_t* fg, const a2f::Line& line, float t, float mix) {
  auto* phon = reinterpret_cast<skse::KeyframeMultiple*>(fg + skse::kFgPhon3);
  auto* mod = reinterpret_cast<skse::KeyframeMultiple*>(fg + skse::kFgMod3);
  auto* ex = reinterpret_cast<skse::KeyframeMultiple*>(fg + skse::kFgExpr3);
  float f = (t + tune.time_offset) * static_cast<float>(a2f::kFps);
  if (f < 0) f = 0;
  uint32_t i0 = static_cast<uint32_t>(f);
  float a = f - static_cast<float>(i0);
  bool past = i0 + 1 >= line.frames;
  const uint8_t* fr0 =
      line.data.data() +
      static_cast<size_t>(past ? line.frames - 1 : i0) * a2f::kNumChannels;
  const uint8_t* fr1 = past ? fr0 : fr0 + a2f::kNumChannels;
  // ease out over 0.1 s
  float fade =
      past ? std::fmax(0.f,
                       1.f - (f - static_cast<float>(line.frames - 1)) / 3.f)
           : 1.f;
  auto sample = [&](int ch, float scale) {
    return std::fmin(1.f, ((1 - a) * fr0[ch] + a * fr1[ch]) * (1.0f / 255.0f) *
                              scale * fade);
  };

  if (phon->values && phon->count >= a2f::kNumMouth) {
    for (int i = 0; i < a2f::kNumMouth; ++i)
      phon->values[i] = Mix(phon->values[i], sample(i, tune.mouth), mix);
    phon->is_updated = true;
  }
  if (mod->values && mod->count >= 14) {
    for (int i = 0; i < a2f::kNumMod; ++i) {
      float v = sample(a2f::kNumMouth + i, tune.brows);
      int e = kModEngine[i];
      // keep the engine's procedural blink
      mod->values[e] =
          (e <= 1) ? std::fmax(mod->values[e], v) : Mix(mod->values[e], v, mix);
    }
    mod->is_updated = true;
  }
  if (ex->values && ex->count >= 7) {
    for (uint32_t i = 0; i < 7; ++i)
      ex->values[i] = Mix(
          ex->values[i],
          i == line.expr ? sample(a2f::kNumSolve, tune.expression) : 0.f, mix);
    ex->is_updated = true;
  }
}

volatile LONG logged_plays;

// Plays the head's current line for this frame; returns whether keyframes
// were written.
bool Apply(uint8_t* fg) {
  AcquireSRWLockExclusive(&lock);
  auto it = playing.find(fg);
  if (it == playing.end()) {
    ReleaseSRWLockExclusive(&lock);
    return false;
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
  if (done) {
    playing.erase(it);
    num_playing = static_cast<LONG>(playing.size());
    ReleaseSRWLockExclusive(&lock);
    return false;
  }
  if (!line) {  // still generating: vanilla lip-sync meanwhile
    ReleaseSRWLockExclusive(&lock);
    return false;
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
  bool first = !p.logged;
  p.logged = true;
  std::string name = first ? p.name : std::string();
  float ready_at = p.ready_at;
  ReleaseSRWLockExclusive(&lock);
  if (first && InterlockedIncrement(&logged_plays) <= config.log_lines)
    Log("play %s: %u frames, %s clock, ready at %.2f s", name.c_str(),
        line->frames, act ? "lip" : "wall", ready_at);
  PlayFrame(fg, *line, t, mix);
  return true;
}

// Runs where the head's morphs are applied (ID 26998), right after the engine
// recomputes the face keyframes (ID 26598) -- writing earlier would be
// overwritten. Returns "changed" so the morphs are re-applied.
__attribute__((noinline)) bool AnimUpdateHook(uint8_t* fg, float dt,
                                              uint64_t flag) {
  auto orig = reinterpret_cast<AnimUpdateFn>(
      SiteOrig(__builtin_return_address(0), kIdFaceAnimUpdate));
  bool changed = orig(fg, dt, flag);
  if (fg && num_playing && tune.enabled) {
    auto* face_lock = reinterpret_cast<skse::SpinLock*>(fg + skse::kFgLock);
    face_lock->Lock();
    changed |= Apply(fg);
    face_lock->Unlock();
  }
  return changed;
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
  ig.Text("Model: %s", st.net_info.c_str());
  ig.Text("Lines generated: %u (%u failed / no audio), cached %zu, queued %zu",
          st.lines, st.failed, cached, queued);
  if (st.lines)
    ig.Text("Speed: %.0fx realtime, last line %.0f ms; started late: %u",
            st.audio_sec / std::max(st.gen_sec, 1e-6), st.last_ms, st.late);
  if (!st.last_error.empty())
    ig.Text("Last skipped: %s", st.last_error.c_str());
  // checkboxes, then sliders by section
  static const char* const kHeaders[] = {nullptr, "Playback (instant)",
                                         "Generation (new lines)"};
  int cur = 0;
  for (const auto& p : kParams) {
    int sec = p.flag ? 0 : 1 + p.gen;
    if (sec != cur) ig.SeparatorText(kHeaders[cur = sec]);
    AcquireSRWLockExclusive(&settings_lock);
    bool c = p.flag ? ig.Checkbox(p.label, p.flag)
                    : ig.SliderFloat(p.label, p.value, p.lo, p.hi, "%.3g", 0);
    ReleaseSRWLockExclusive(&settings_lock);
    if (ig.IsItemHovered(0)) ig.SetTooltip("%s", p.tip);
    if (c && p.gen) InterlockedIncrement(&gen_version);
  }
  ig.SeparatorText("");
  if (ig.Button("Save to A2FLive.ini", ImVec2{0, 0})) {
    AcquireSRWLockShared(&settings_lock);
    SaveIni();
    ReleaseSRWLockShared(&settings_lock);
  }
  ig.SameLine(0, -1);
  if (ig.Button("Defaults", ImVec2{0, 0}) && ready) {
    AcquireSRWLockExclusive(&settings_lock);
    gen = data.defaults;
    tune = Tune();
    ReleaseSRWLockExclusive(&settings_lock);
    InterlockedIncrement(&gen_version);
  }
  ig.SameLine(0, -1);
  if (ig.Button("Clear cache", ImVec2{0, 0})) {
    AcquireSRWLockExclusive(&lock);
    for (auto it = cache.begin(); it != cache.end();)
      it = it->second->state == State::kPending ? std::next(it)
                                                : cache.erase(it);
    ReleaseSRWLockExclusive(&lock);
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
    Log("SKSE Menu Framework not installed: settings via A2FLive.ini only");
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
  add("A2F Live/Settings", RenderMenu);
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
  skse::OpenLog();
  Log("A2FLive %s: runtime %08X", kVersion, iface->runtimeVersion);
  if (!skse::LoadAddressLibrary(kAddressLibrary)) {
    Log("address library missing/unreadable");
    return false;
  }
  if (!Addr(kIdGenerateFromPath) || !Addr(kIdStreamCtor) ||
      !Addr(kIdStreamDtor)) {
    Log("address ids not found");
    return false;
  }
  LoadIni();
  if (!HookCalls("speak", kIdSpeakVoiceLine, kSpeakCallers,
                 reinterpret_cast<void*>(&SpeakHook)))
    return false;
  HookCalls("response", kIdDialogueResponseCtor, kResponseCallers,
            reinterpret_cast<void*>(&ResponseHook));
  if (!HookCalls("face", kIdFaceAnimUpdate, kFaceCallers,
                 reinterpret_cast<void*>(&AnimUpdateHook)))
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
