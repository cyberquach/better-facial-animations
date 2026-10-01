// A2FLive: Audio2Face lip-sync generated in the game for every voiced line of every NPC with a FaceGen head.
// The speak / response hooks queue a line; a worker thread reads the voice file through the engine's resource system,
// decodes it and runs the network + solve (live_core.cpp); the face hook plays the curves in the head's FaceGen update.
#include "skse.h"
#include "live_core.h"
#include "ort_net.h"
#include <cmath>
#include <string>
#include <deque>
#include <memory>
#include <unordered_map>

extern "C" { __declspec(dllexport) extern SKSEPluginVersionData SKSEPlugin_Version; }
#define A2FLIVE_VERSION "1.0.1"
SKSEPluginVersionData SKSEPlugin_Version = {
    1, 0x01000010 /* 1.0.1 */, "A2FLive", "Cyberquach", "", 0, 0, { 0x01064920 /* 1.6.1170 */, 0 }, 0 };

// ---------------- engine (AE 1.6.1170) ----------------
constexpr u64 ID_SpeakVoiceLine = 37542, ID_DialogueResponseCtor = 35249;
constexpr u64 SPEAK_CALLERS[] = { 39772, 37541, 44673, 22864 };     // play-response, UpdateInDialogue, 44673, console SpeakSound
constexpr u64 RESPONSE_CALLERS[] = { 35220, 35254, 35256 };         // DialogueItem ctor, MenuTopicManager x2
constexpr u64 ID_StreamCtor = 71014, ID_StreamDtor = 71016;          // BSResourceNiBinaryStream
constexpr u64 ID_FaceAnimUpdate = 26598;                            // BSFaceGenAnimationData::Update(fg, dt, bool)
constexpr u64 FACE_CALLERS[] = { 26998 };                            // BSFaceGenNiNode: update keyframes, apply morphs
constexpr u64 ID_SoundIsPlaying = 67620;                             // bool(BSSoundHandle*)
constexpr u64 ID_GenerateFromPath = 69979;                           // ResID from "Sound\...\X.lip"
constexpr u32 ACTOR_PROCESS = 0xF8, PROCESS_HIGH = 0x10, HIGH_VOICE_HANDLE = 0x2FC;
constexpr u32 FORM_TYPE = 0x1A, REFR_BASE = 0x40, NPC_FLAGS = 0x38, ACTOR_RACE = 0x1F8, RACE_FLAGS = 0x108;
constexpr u8 FT_NPC = 0x2B, FT_ACTOR = 0x3E;
constexpr u32 VF_GetFaceGenAnimationData = 0x63;
constexpr u32 RESP_EMO_TYPE = 0x10, RESP_EMO_VALUE = 0x14, RESP_VOICE = 0x18;  // DialogueResponse

using Speak_t = float (*)(u8* actor, const char* voice, void* handle, u64 emoType, u64 emoValue, u64, u64, u64, u64, u64, u64, u64, u64, u64);
using ResponseCtor_t = u8* (*)(u8* self, void* topic, void* info, u8* speaker, void* response);
using AnimUpdate_t = bool (*)(u8* fg, float dt, u64 flag);
using GenerateFromPath_t = void (*)(ResID*, const char*);

#define PLUGINS "Data\\SKSE\\Plugins\\"
static const char* INI = PLUGINS "A2FLive.ini";
static const wchar_t* INI_W = L"" PLUGINS "A2FLive.ini";

// ---------------- settings ----------------
struct Cfg { bool gpu = true; int batch = 2, unloadAfter = 0, logLines = 40, cacheLines = 300; std::wstring model; };  // fixed after load
struct Tune { bool enabled = true, prefetch = true; float timeOffset = 0, blendIn = 0.15f, mouth = 1, brows = 1, expression = 1; };  // menu
constexpr float LATE_START = 0.1f;      // a line ready later than this after its start is blended in
static Cfg g_cfg;
static Tune g_tune;                     // scalars; read by the playback hooks without a lock
static volatile bool g_netLoaded;       // lets lines keep generating while disabled, for instant A/B
static a2f::Settings g_gen;             // generation settings; the worker snapshots them under g_setLock
static volatile LONG g_genVersion = 1;  // bumped when g_gen changes: cached lines are regenerated
static SRWLOCK g_setLock = SRWLOCK_INIT;
static a2f::Data g_data;
static volatile bool g_ready;           // data file loaded

// Every ini key / menu item: a slider (v, lo..hi) or, with b set, a checkbox. gen: applies to newly generated lines
// (ini section Generation; otherwise Playback for sliders, General for checkboxes). Table order = menu order.
struct Param { const char* key; float* v; float lo, hi; const char* label; const char* tip; bool gen; bool* b; };
static Param g_params[] = {
    { "Enabled", nullptr, 0, 0, "Enabled", "Play Audio2Face lip-sync. Untick to see the game's own lip-sync instantly (A/B).", false, &g_tune.enabled },
    { "Prefetch", nullptr, 0, 0, "Prefetch", "Start generating when a dialogue response is created, before it is spoken.", false, &g_tune.prefetch },
    { "EmotionToA2F", nullptr, 0, 0, "Emotion to Audio2Face", "Feed the line's dialogue emotion to the network.", true, &g_gen.emotionToA2F },
    { "Expression", nullptr, 0, 0, "Expression morph", "Drive the Dialogue<Emotion> expression morph.", true, &g_gen.expressionOn },
    { "Tongue", nullptr, 0, 0, "Tongue layer", "Use Audio2Face's tongue to pick DST / Th shapes.", true, &g_gen.tongueOn },
    { "MouthStrength", &g_tune.mouth, 0, 2, "Mouth", "Scales the phoneme (mouth) curves at playback.", false },
    { "BrowStrength", &g_tune.brows, 0, 2, "Brows / blinks", "Scales brow, blink and squint curves at playback.", false },
    { "ExpressionStrength", &g_tune.expression, 0, 2, "Expression", "Scales the dialogue emotion expression at playback.", false },
    { "TimeOffset", &g_tune.timeOffset, -0.3f, 0.3f, "Sync offset (s)", "Shifts the animation against the voice. Positive = earlier mouth shapes.", false },
    { "BlendIn", &g_tune.blendIn, 0, 0.5f, "Blend-in (s)", "Crossfade from vanilla lip-sync when a line finishes generating after it started.", false },
    { "Strength", &g_gen.strength, 0.3f, 2.5f, "Solve strength", "How far the Skyrim morphs follow the Audio2Face face.", true },
    { "Smooth", &g_gen.smooth, 0, 0.8f, "Smoothing", "Temporal smoothing of the morph curves (zero phase).", true },
    { "Ridge", &g_gen.ridge, 0, 0.2f, "Regularisation", "Penalty on morph weights; higher = fewer, softer shapes.", true },
    { "FeatureWeight", &g_gen.featureWeight, 0, 100, "Lip feature weight", "Weight of lip opening / width / protrusion matching.", true },
    { "InputStrength", &g_gen.inputStrength, 0.5f, 2, "Audio gain", "Audio level fed to the network (Audio2Face input strength).", true },
    { "SkinStrength", &g_gen.skinStrength, 0.3f, 2, "A2F skin strength", "Audio2Face skin animation strength.", true },
    { "LowerFaceStrength", &g_gen.lowerFaceStrength, 0.3f, 2.5f, "A2F lower face", "Audio2Face mouth / jaw strength.", true },
    { "UpperFaceStrength", &g_gen.upperFaceStrength, 0, 2, "A2F upper face", "Audio2Face brow / eye strength.", true },
    { "LowerFaceSmoothing", &g_gen.lowerSmoothing, 0, 0.05f, "A2F lower smoothing", "Audio2Face lower face smoothing half-life (s).", true },
    { "UpperFaceSmoothing", &g_gen.upperSmoothing, 0, 0.05f, "A2F upper smoothing", "Audio2Face upper face smoothing half-life (s).", true },
    { "TongueStrength", &g_gen.tongueStrength, 0, 3, "A2F tongue", "Audio2Face tongue strength (drives the tongue phonemes).", true },
    { "TongueShare", &g_gen.tongueShare, 0, 1, "Tongue share", "How much jaw opening moves to tongue shapes (DST / Th).", true },
    { "EmotionFloor", &g_gen.a2fFloor, 0, 1, "Emotion floor", "Minimum emotion intensity sent to Audio2Face for emotional lines.", true },
    { "EmotionGain", &g_gen.a2fGain, 0, 3, "Emotion gain", "Dialogue emotion -> Audio2Face emotion intensity.", true },
    { "EmotionCap", &g_gen.a2fCap, 0, 2, "Emotion cap", "Maximum Audio2Face emotion.", true },
    { "ExpressionFloor", &g_gen.exprFloor, 0, 1, "Expression floor", "Minimum expression morph level for emotional lines.", true },
    { "ExpressionGain", &g_gen.exprGain, 0, 3, "Expression gain", "Dialogue emotion -> expression morph level.", true },
    { "Emphasis", &g_gen.emphasis, 0, 1, "Expression emphasis", "How much the expression follows loudness.", true },
    { "EmotionOnset", &g_gen.onset, 0, 1, "Emotion onset", "Emotion level at the start / end of a line.", true },
    { "EmotionAttack", &g_gen.attack, 0.01f, 2, "Emotion attack (s)", "Ease-in time of the emotion.", true },
    { "EmotionRelease", &g_gen.release, 0.01f, 2, "Emotion release (s)", "Ease-out time of the emotion.", true },
};
static const char* Section(const Param& p) { return p.gen ? "Generation" : p.b ? "General" : "Playback"; }

// Called at load (General) and again by the worker once the data file's defaults are in g_gen (overlay the ini on them).
static void LoadIni() {
    g_cfg.gpu = GetPrivateProfileIntA("General", "GPU", 1, INI) != 0;
    g_cfg.batch = std::max(1, (int)GetPrivateProfileIntA("General", "Batch", 2, INI));
    g_cfg.unloadAfter = GetPrivateProfileIntA("General", "UnloadAfterIdleSeconds", 0, INI);
    g_cfg.cacheLines = std::max(8, (int)GetPrivateProfileIntA("General", "CacheLines", 300, INI));
    g_cfg.logLines = GetPrivateProfileIntA("Debug", "LogLines", 40, INI);
    wchar_t m[260]; GetPrivateProfileStringW(L"General", L"Model", L"network_fp16.onnx", m, 260, INI_W); g_cfg.model = m;
    char b[128], d[32];
    for (auto& p : g_params) {
        if (p.b) { *p.b = GetPrivateProfileIntA(Section(p), p.key, *p.b, INI) != 0; continue; }
        snprintf(d, sizeof d, "%g", *p.v); GetPrivateProfileStringA(Section(p), p.key, d, b, sizeof b, INI); *p.v = (float)atof(b);
    }
    for (int j = 0; j < a2f::NSOLVE; ++j) {  // optional per-channel overrides: Name=gain offset gamma max cost solve
        if (!GetPrivateProfileStringA("Channels", a2f::CHANNEL_NAMES[j], "", b, sizeof b, INI)) continue;
        int solve = g_gen.solve[j];
        sscanf(b, "%f %f %f %f %f %d", &g_gen.gain[j], &g_gen.offset[j], &g_gen.gamma[j], &g_gen.max[j], &g_gen.cost[j], &solve);
        g_gen.solve[j] = solve != 0;
    }
}
static void SaveIni() {
    char b[64];
    for (auto& p : g_params) {
        if (p.v) snprintf(b, sizeof b, "%.4g", *p.v);
        WritePrivateProfileStringA(Section(p), p.key, p.v ? b : *p.b ? "1" : "0", INI);
    }
}

// ---------------- generated lines ----------------
enum { PENDING, READY, FAILED };
struct Entry { int state = PENDING; LONG version = 0; std::shared_ptr<const a2f::Line> line; u64 lastUse = 0; std::string error; };
struct Job { std::string cacheKey, path; int emo, val; bool female; };
struct Playing { std::shared_ptr<Entry> e; std::string name; u64 rkey; u8* actor; double t0; float readyAt = -1; bool lipSeen = false, logged = false; };
struct Stats { u32 lines = 0, failed = 0, late = 0; double audioSec = 0, genSec = 0, lastMs = 0; std::string netInfo, lastError; };

static SRWLOCK g_lock = SRWLOCK_INIT;  // guards everything below
static CONDITION_VARIABLE g_cv = CONDITION_VARIABLE_INIT;
static std::unordered_map<std::string, std::shared_ptr<Entry>> g_cache;
static std::deque<Job> g_queue;
static std::unordered_map<u8*, Playing> g_playing;  // by FaceGen animation data
static volatile LONG g_nPlaying;                    // g_playing.size(), so the per-frame hook can skip the lock
static u64 g_useClock;
static Stats g_stats;

static double Now() { static LARGE_INTEGER f; if (!f.QuadPart) QueryPerformanceFrequency(&f); LARGE_INTEGER c; QueryPerformanceCounter(&c); return (double)c.QuadPart / f.QuadPart; }

// "Data\Sound\Voice\Skyrim.esm\MaleNord\X.wav" -> "Sound\Voice\Skyrim.esm\MaleNord\X" (engine resource path without extension)
static bool VoiceStem(const char* voice, std::string& stem) {
    const char* p = voice;
    for (; *p; ++p) if (!_strnicmp(p, "sound\\", 6) || !_strnicmp(p, "sound/", 6)) break;
    if (!*p) return false;
    stem = p;
    size_t dot = stem.find_last_of('.'), sl = stem.find_last_of("\\/");
    if (dot != std::string::npos && (sl == std::string::npos || dot > sl)) stem.resize(dot);
    return !stem.empty();
}

// Actor with a FaceGen head (people; animals and creatures have none) -> sex
static bool SpeakerOk(u8* ref, bool& female) {
    if (!ref || ref[FORM_TYPE] != FT_ACTOR) return false;
    u8* base = *(u8**)(ref + REFR_BASE);
    u8* race = *(u8**)(ref + ACTOR_RACE);
    if (!base || base[FORM_TYPE] != FT_NPC || !race || !(*(u32*)(race + RACE_FLAGS) & 2)) return false;
    female = *(u32*)(base + NPC_FLAGS) & 1;
    return true;
}

// Queue a line unless it is cached or in flight with the current settings. -> its entry
static std::shared_ptr<Entry> Request(const std::string& stem, int emo, int val, bool female, bool urgent) {
    std::string key = stem; for (auto& c : key) c = (char)tolower((u8)c);
    char sfx[32]; snprintf(sfx, sizeof sfx, "|%c%d|%d", female ? 'f' : 'm', emo, val); key += sfx;
    LONG ver = g_genVersion;
    AcquireSRWLockExclusive(&g_lock);
    auto& e = g_cache[key];
    if (!e || (e->version != ver && e->state != PENDING)) {
        e = std::make_shared<Entry>(); e->version = ver;
        Job j{ key, stem, emo, val, female };
        if (urgent) g_queue.push_front(j); else g_queue.push_back(j);
        WakeConditionVariable(&g_cv);
    } else if (urgent && e->state == PENDING) {  // prefetched but not started: move to the front
        for (auto it = g_queue.begin(); it != g_queue.end(); ++it)
            if (it->cacheKey == key) { Job j = *it; g_queue.erase(it); g_queue.push_front(j); break; }
    }
    e->lastUse = ++g_useClock;
    auto r = e;
    ReleaseSRWLockExclusive(&g_lock);
    return r;
}

// Read a file through BSResourceNiBinaryStream (BSAs + loose files, MO2 VFS); path relative to Data
static bool ReadGameFile(const std::string& path, std::vector<u8>& out) {
    using Ctor = void* (*)(void*, const char*, u64, void*);
    using Dtor = void (*)(void*);
    using ReadFn = u32 (*)(void*, void*, u32, u32*, u32);
    static Ctor ctor = (Ctor)Addr(ID_StreamCtor); static Dtor dtor = (Dtor)Addr(ID_StreamDtor);
    alignas(16) u8 s[0x40] = {};
    ctor(s, path.c_str(), 0, nullptr);
    void* st = *(void**)(s + 0x20);  // BSResource::Stream*, null when not found
    bool ok = false;
    if (st) {
        u32 n = *(u32*)((u8*)st + 8);
        out.resize(n);
        auto rd = *(ReadFn*)(s + 0x10);
        u32 got = 0;
        while (got < n) { u32 k = rd(s, out.data() + got, n - got, nullptr, 1); if (!k) break; got += k; }
        ok = n > 0 && got == n;
    }
    dtor(s);
    return ok;
}

// ---------------- worker ----------------
static std::unique_ptr<a2f::Net> g_net;  // worker thread only

static std::wstring PluginDir() {
    wchar_t full[MAX_PATH]; GetFullPathNameW(L"" PLUGINS "A2FLive", MAX_PATH, full, nullptr); return full;
}

static bool EnsureNet() {
    if (g_net) return true;
    const std::wstring dir = PluginDir(), model = dir + L"\\" + g_cfg.model;
    static bool dmlLoaded;
    if (!dmlLoaded) {  // onnxruntime delay-loads DirectML.dll by name: load ours first so that is the one it gets
        dmlLoaded = LoadLibraryExW((dir + L"\\DirectML.dll").c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH) != nullptr;
        if (!dmlLoaded) Log("DirectML.dll not loaded from %ls (error %lu)", dir.c_str(), GetLastError());
    }
    a2f::OrtOptions o; o.gpu = g_cfg.gpu; o.batch = g_cfg.batch;
    std::string err, info; double t = Now();
    g_net = a2f::CreateOrtNet(dir, model, o, err, info);
    if (!g_net && o.gpu) {
        Log("GPU session failed (%s); falling back to CPU", err.c_str());
        o.gpu = false; g_net = a2f::CreateOrtNet(dir, model, o, err, info);
    }
    AcquireSRWLockExclusive(&g_lock);
    g_stats.netInfo = g_net ? info : "failed: " + err;
    ReleaseSRWLockExclusive(&g_lock);
    if (!g_net) { Log("network not loaded: %s", err.c_str()); return false; }
    g_netLoaded = true;
    Log("network loaded: %s, %ls (%.0f ms)", info.c_str(), g_cfg.model.c_str(), (Now() - t) * 1e3);
    return true;
}

static void Process(const Job& j) {
    static int logged;
    std::vector<u8> bytes;
    std::string err, file;
    for (const char* ext : { ".fuz", ".xwm", ".wav" })
        if (ReadGameFile(j.path + ext, bytes)) { file = j.path + ext; break; }
    std::shared_ptr<a2f::Line> line;
    double t = Now(), audioSec = 0; a2f::Timing tm;
    if (file.empty()) err = "no voice file";
    else {
        std::vector<float> pcm;
        if (a2f::DecodeAudio(bytes.data(), bytes.size(), pcm, err) && EnsureNet()) {
            AcquireSRWLockShared(&g_setLock); a2f::Settings s = g_gen; ReleaseSRWLockShared(&g_setLock);
            line = std::make_shared<a2f::Line>();
            audioSec = pcm.size() / (double)a2f::SR;
            if (!a2f::Generate(g_data, *g_net, s, pcm.data(), pcm.size(), j.emo, j.val, j.female, *line, err, &tm)) line.reset();
        } else if (err.empty()) err = "network unavailable";
    }
    double ms = (Now() - t) * 1e3;
    AcquireSRWLockExclusive(&g_lock);
    auto it = g_cache.find(j.cacheKey);
    if (it != g_cache.end()) { it->second->state = line ? READY : FAILED; it->second->line = line; it->second->error = err; }
    if (line) { ++g_stats.lines; g_stats.audioSec += audioSec; g_stats.genSec += ms / 1e3; g_stats.lastMs = ms; }
    else { ++g_stats.failed; g_stats.lastError = file.empty() ? j.path + ": " + err : err; }
    if (g_cache.size() > (size_t)g_cfg.cacheLines) {  // drop the least recently used finished lines
        std::vector<std::pair<u64, std::string>> v;
        for (auto& [k, e] : g_cache) if (e->state != PENDING) v.emplace_back(e->lastUse, k);
        std::sort(v.begin(), v.end());
        for (size_t i = 0; i < v.size() && g_cache.size() > (size_t)g_cfg.cacheLines * 3 / 4; ++i) g_cache.erase(v[i].second);
    }
    ReleaseSRWLockExclusive(&g_lock);
    if (++logged <= g_cfg.logLines) {
        if (line) Log("gen %s %c emo %d/%d: %.2f s audio, %u frames in %.0f ms (net %.0f, solve %.0f)", file.c_str(), j.female ? 'F' : 'M', j.emo, j.val,
                      audioSec, line->frames, ms, tm.net * 1e3, tm.solve * 1e3);
        else Log("skip %s: %s", j.path.c_str(), err.c_str());
    }
}

static DWORD WINAPI Worker(void*) {
    std::string err;
    if (!g_data.Load(PluginDir() + L"\\A2FLive.dat", err)) { Log("A2FLive.dat: %s", err.c_str()); return 1; }
    AcquireSRWLockExclusive(&g_setLock);
    g_gen = g_data.defaults; LoadIni();
    ReleaseSRWLockExclusive(&g_setLock);
    g_ready = true;
    if (g_tune.enabled) EnsureNet();  // resident from the start
    double idleSince = Now();
    for (;;) {
        AcquireSRWLockExclusive(&g_lock);
        while (g_queue.empty()) {
            SleepConditionVariableSRW(&g_cv, &g_lock, g_cfg.unloadAfter > 0 && g_net ? 1000 : INFINITE, 0);
            if (g_queue.empty() && g_net && g_cfg.unloadAfter > 0 && Now() - idleSince > g_cfg.unloadAfter) {
                ReleaseSRWLockExclusive(&g_lock);
                g_net.reset(); g_netLoaded = false; Log("network unloaded after %d s idle", g_cfg.unloadAfter);
                AcquireSRWLockExclusive(&g_lock); g_stats.netInfo = "unloaded (idle)";
            }
        }
        Job j = g_queue.front(); g_queue.pop_front();
        ReleaseSRWLockExclusive(&g_lock);
        Process(j);
        idleSince = Now();
    }
}

// ---------------- call-site hooks ----------------
// Every `call rel32` to the callee inside the caller functions is redirected through a thunk. Each site keeps its own
// original target (looked up by return address), so a plugin that patches a site after us is chained.
struct Site { u64 at; void* orig; void* thunk; const char* what; };
static Site g_sites[16]; static int g_nSites;
static u8* g_thunks; static size_t g_thunkUsed;
static void* Thunk(void* fn) {  // jmp [rip+0] -> fn, within rel32 reach of the exe
    if (!g_thunks) {
        for (u64 a = (g_base - 0x10000) & ~0xFFFFull; a > g_base - 0x70000000 && !g_thunks; a -= 0x10000)
            g_thunks = (u8*)VirtualAlloc((void*)a, 0x1000, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE);
        if (!g_thunks) return nullptr;
    }
    if (g_thunkUsed + 16 > 0x1000) return nullptr;
    u8* t = g_thunks + g_thunkUsed; g_thunkUsed += 16;
    t[0] = 0xFF; t[1] = 0x25; *(u32*)(t + 2) = 0; *(void**)(t + 6) = fn;
    return t;
}
static u64 CallTarget(u64 at) { return at + 5 + *(i32*)(at + 1); }
static bool PatchCall(u64 at, void* to) {
    int64_t rel = (int64_t)((u64)to - (at + 5));
    if (rel != (i32)rel) return false;
    DWORD old; VirtualProtect((void*)(at + 1), 4, PAGE_EXECUTE_READWRITE, &old);
    *(i32*)(at + 1) = (i32)rel;
    VirtualProtect((void*)(at + 1), 4, old, &old);
    return true;
}
static void* SiteOrig(void* ret, u64 calleeId) {  // original target of the site that called a hook; the callee itself for unknown callers
    for (int i = 0; i < g_nSites; ++i) if (g_sites[i].at + 5 == (u64)ret) return g_sites[i].orig;
    return (void*)Addr(calleeId);
}
template <size_t N> static int HookCalls(const char* what, u64 calleeId, const u64 (&callers)[N], void* hook) {
    u64 callee = Addr(calleeId);
    auto* nt = (IMAGE_NT_HEADERS*)(g_base + ((IMAGE_DOS_HEADER*)g_base)->e_lfanew);
    u64 end = g_base + nt->OptionalHeader.SizeOfImage;
    if (!callee) { Log("%s: address id %llu not found", what, calleeId); return 0; }
    void* thunk = Thunk(hook); int n = 0;
    if (!thunk) { Log("%s: no thunk memory", what); return 0; }
    for (u64 caller : callers) {
        u64 f = Addr(caller); if (!f) continue;
        for (u64 at = f; at < f + 0x2000 && at + 5 < end && g_nSites < 16; ++at) {
            if (*(u8*)at != 0xE8 || CallTarget(at) != callee || !PatchCall(at, thunk)) continue;
            g_sites[g_nSites++] = { at, (void*)callee, thunk, what };
            Log("%s: hooked call at rva %llx (caller id %llu)", what, at - g_base, caller); ++n;
        }
    }
    if (!n) Log("%s: no call sites found (another plugin may have replaced them)", what);
    return n;
}
static void RetakeSites() {  // after all plugins loaded: chain behind anyone who patched our sites since
    for (int i = 0; i < g_nSites; ++i) {
        Site& s = g_sites[i];
        bool call = *(u8*)s.at == 0xE8;
        if (call && CallTarget(s.at) == (u64)s.thunk) continue;
        void* other = call ? (void*)CallTarget(s.at) : nullptr;
        if (call && PatchCall(s.at, s.thunk)) { s.orig = other; Log("%s: rva %llx was re-hooked by another plugin; chained", s.what, s.at - g_base); }
        else Log("%s: rva %llx overwritten by another plugin; this site is not hooked", s.what, s.at - g_base);
    }
}

// ---------------- hooks ----------------
__attribute__((noinline)) static float SpeakHook(u8* actor, const char* voice, void* handle, u64 emoType, u64 emoValue, u64 a6, u64 a7, u64 a8, u64 a9, u64 a10,
                       u64 a11, u64 a12, u64 a13, u64 a14) {
    auto orig = (Speak_t)SiteOrig(__builtin_return_address(0), ID_SpeakVoiceLine);
    float dur = orig(actor, voice, handle, emoType, emoValue, a6, a7, a8, a9, a10, a11, a12, a13, a14);
    bool female; std::string stem;
    if (!(g_tune.enabled || g_netLoaded) || !voice || !g_ready || !SpeakerOk(actor, female) || !VoiceStem(voice, stem)) return dur;
    auto getFg = (u8* (*)(u8*))(*(void***)actor)[VF_GetFaceGenAnimationData];
    u8* fg = getFg(actor);
    if (!fg) return dur;
    auto e = Request(stem, (int)(u32)emoType, (int)(u32)emoValue, female, true);
    static auto genPath = (GenerateFromPath_t)Addr(ID_GenerateFromPath);
    ResID r{}; genPath(&r, (stem + ".lip").c_str());
    Playing p{ e, stem, KeyOf(r), actor, Now() };
    AcquireSRWLockExclusive(&g_lock);
    g_playing[fg] = p; g_nPlaying = (LONG)g_playing.size();
    ReleaseSRWLockExclusive(&g_lock);
    return dur;
}

__attribute__((noinline)) static u8* ResponseHook(u8* self, void* topic, void* info, u8* speaker, void* response) {
    auto orig = (ResponseCtor_t)SiteOrig(__builtin_return_address(0), ID_DialogueResponseCtor);
    u8* r = orig(self, topic, info, speaker, response);
    bool female; std::string stem;
    const char* voice = r ? *(const char**)(r + RESP_VOICE) : nullptr;
    if ((g_tune.enabled || g_netLoaded) && g_tune.prefetch && voice && g_ready && SpeakerOk(speaker, female) && VoiceStem(voice, stem))
        Request(stem, (int)*(u32*)(r + RESP_EMO_TYPE), (int)*(u32*)(r + RESP_EMO_VALUE), female, false);
    return r;
}

// The actor's voice sound is still playing (lines without .lip data have no engine clock to watch)
static bool VoicePlaying(u8* actor) {
    static auto isPlaying = (bool (*)(void*))Addr(ID_SoundIsPlaying);
    u8* proc = *(u8**)(actor + ACTOR_PROCESS);
    u8* high = proc ? *(u8**)(proc + PROCESS_HIGH) : nullptr;
    return !high || !isPlaying || isPlaying(high + HIGH_VOICE_HANDLE);
}

// line modifier order (BlinkL/R, BrowDownL/R, BrowInL/R, BrowUpL/R, SquintL/R) -> engine modifier index
constexpr int MOD_ENGINE[a2f::NMOD] = { 0, 1, 2, 3, 4, 5, 6, 7, 12, 13 };
static inline float Mix(float engine, float ours, float mix) { return mix >= 1.f ? ours : engine + (ours - engine) * mix; }

// Write the line's frame at lip time t (s) into the final phoneme / modifier / expression keyframes; mix < 1 blends from the engine's values.
static void PlayFrame(u8* fg, const a2f::Line& line, float t, float mix) {
    auto* phon = (KeyframeMultiple*)(fg + FG_PHON3);
    auto* mod = (KeyframeMultiple*)(fg + FG_MOD3);
    auto* ex = (KeyframeMultiple*)(fg + FG_EXPR3);
    float f = (t + g_tune.timeOffset) * (float)a2f::FPS; if (f < 0) f = 0;
    u32 i0 = (u32)f; float a = f - (float)i0;
    bool past = i0 + 1 >= line.frames;
    const u8* fr0 = line.data.data() + (size_t)(past ? line.frames - 1 : i0) * a2f::NCH;
    const u8* fr1 = past ? fr0 : fr0 + a2f::NCH;
    float fade = past ? std::fmax(0.f, 1.f - (f - (float)(line.frames - 1)) / 3.f) : 1.f;  // ease out over 0.1 s
    auto sample = [&](int ch, float scale) { return std::fmin(1.f, ((1 - a) * fr0[ch] + a * fr1[ch]) * (1.0f / 255.0f) * scale * fade); };

    if (phon->values && phon->count >= a2f::NMOUTH) {
        for (int i = 0; i < a2f::NMOUTH; ++i) phon->values[i] = Mix(phon->values[i], sample(i, g_tune.mouth), mix);
        phon->isUpdated = true;
    }
    if (mod->values && mod->count >= 14) {
        for (int i = 0; i < a2f::NMOD; ++i) {
            float v = sample(a2f::NMOUTH + i, g_tune.brows); int e = MOD_ENGINE[i];
            mod->values[e] = (e <= 1) ? std::fmax(mod->values[e], v) : Mix(mod->values[e], v, mix);  // keep the engine's procedural blink
        }
        mod->isUpdated = true;
    }
    if (ex->values && ex->count >= 7) {
        for (u32 i = 0; i < 7; ++i) ex->values[i] = Mix(ex->values[i], i == line.expr ? sample(a2f::NSOLVE, g_tune.expression) : 0.f, mix);
        ex->isUpdated = true;
    }
}

static volatile LONG g_loggedPlays;

static bool Apply(u8* fg) {
    AcquireSRWLockExclusive(&g_lock);
    auto it = g_playing.find(fg);
    if (it == g_playing.end()) { ReleaseSRWLockExclusive(&g_lock); return false; }
    Playing& p = it->second;
    const ResID* act = ActiveLine(fg);
    float t = (float)(Now() - p.t0); bool done = t > 120;
    if (act && KeyOf(*act) == p.rkey) { t = *(float*)(fg + FG_PHON1_TIMER); p.lipSeen = true; }  // engine lip clock
    else if (p.lipSeen) done = true;                           // the engine ended or replaced the line
    else if (t > 0.5f && !VoicePlaying(p.actor)) done = true;  // no .lip: wall clock while the voice plays
    std::shared_ptr<const a2f::Line> line;
    if (!done) {
        if (p.e->state == FAILED) done = true;
        else if (p.e->state == READY) { line = p.e->line; if (t > line->frames / (float)a2f::FPS + 0.3f) done = true; }
    }
    if (done) { g_playing.erase(it); g_nPlaying = (LONG)g_playing.size(); ReleaseSRWLockExclusive(&g_lock); return false; }
    if (!line) { ReleaseSRWLockExclusive(&g_lock); return false; }  // still generating: vanilla lip-sync meanwhile
    if (p.readyAt < 0) { p.readyAt = t; if (t > LATE_START) ++g_stats.late; }
    float mix = p.readyAt <= LATE_START ? 1.f : std::clamp((t - p.readyAt) / std::max(g_tune.blendIn, 1e-3f), 0.f, 1.f);
    bool first = !p.logged; p.logged = true;
    std::string name = first ? p.name : std::string();
    float ready = p.readyAt;
    ReleaseSRWLockExclusive(&g_lock);
    if (first && InterlockedIncrement(&g_loggedPlays) <= g_cfg.logLines)
        Log("play %s: %u frames, %s clock, ready at %.2f s", name.c_str(), line->frames, act ? "lip" : "wall", ready);
    PlayFrame(fg, *line, t, mix);
    return true;
}

// Runs where the head's morphs are applied (ID 26998), right after the engine recomputes the face keyframes
// (ID 26598) -- writing earlier would be overwritten. Returns "changed" so the morphs are re-applied.
__attribute__((noinline)) static bool AnimUpdateHook(u8* fg, float dt, u64 flag) {
    auto orig = (AnimUpdate_t)SiteOrig(__builtin_return_address(0), ID_FaceAnimUpdate);
    bool changed = orig(fg, dt, flag);
    if (fg && g_nPlaying && g_tune.enabled) {
        auto* lock = (SpinLock*)(fg + FG_LOCK);
        lock->lock(); changed |= Apply(fg); lock->unlock();
    }
    return changed;
}

// ---------------- SKSE Menu Framework page (optional; cimgui exports) ----------------
struct ImVec2 { float x, y; };
static struct {
    bool (*SliderFloat)(const char*, float*, float, float, const char*, int);
    bool (*Checkbox)(const char*, bool*);
    void (*Text)(const char*, ...);
    void (*SeparatorText)(const char*);
    bool (*Button)(const char*, ImVec2);
    void (*SetTooltip)(const char*, ...);
    bool (*IsItemHovered)(int);
    void (*SameLine)(float, float);
} ig;

static void RenderMenu() {
    AcquireSRWLockShared(&g_lock); Stats st = g_stats; size_t cached = g_cache.size(), queued = g_queue.size(); ReleaseSRWLockShared(&g_lock);
    ig.SeparatorText("Status");
    ig.Text("Model: %s", st.netInfo.c_str());
    ig.Text("Lines generated: %u (%u failed / no audio), cached %zu, queued %zu", st.lines, st.failed, cached, queued);
    if (st.lines) ig.Text("Speed: %.0fx realtime, last line %.0f ms; started late: %u", st.audioSec / std::max(st.genSec, 1e-6), st.lastMs, st.late);
    if (!st.lastError.empty()) ig.Text("Last skipped: %s", st.lastError.c_str());
    static const char* HDR[] = { nullptr, "Playback (instant)", "Generation (new lines)" };  // checkboxes, then sliders by section
    int cur = 0;
    for (auto& p : g_params) {
        int sec = p.b ? 0 : 1 + p.gen;
        if (sec != cur) ig.SeparatorText(HDR[cur = sec]);
        AcquireSRWLockExclusive(&g_setLock);
        bool c = p.b ? ig.Checkbox(p.label, p.b) : ig.SliderFloat(p.label, p.v, p.lo, p.hi, "%.3g", 0);
        ReleaseSRWLockExclusive(&g_setLock);
        if (ig.IsItemHovered(0)) ig.SetTooltip("%s", p.tip);
        if (c && p.gen) InterlockedIncrement(&g_genVersion);
    }
    ig.SeparatorText("");
    if (ig.Button("Save to A2FLive.ini", ImVec2{ 0, 0 })) { AcquireSRWLockShared(&g_setLock); SaveIni(); ReleaseSRWLockShared(&g_setLock); }
    ig.SameLine(0, -1);
    if (ig.Button("Defaults", ImVec2{ 0, 0 }) && g_ready) {
        AcquireSRWLockExclusive(&g_setLock); g_gen = g_data.defaults; g_tune = Tune(); ReleaseSRWLockExclusive(&g_setLock);
        InterlockedIncrement(&g_genVersion);
    }
    ig.SameLine(0, -1);
    if (ig.Button("Clear cache", ImVec2{ 0, 0 })) {
        AcquireSRWLockExclusive(&g_lock);
        for (auto it = g_cache.begin(); it != g_cache.end();) it = it->second->state == PENDING ? std::next(it) : g_cache.erase(it);
        ReleaseSRWLockExclusive(&g_lock);
    }
}

static void RegisterMenu() {
    HMODULE m = GetModuleHandleA("SKSEMenuFramework.dll");
    if (!m) { Log("SKSE Menu Framework not installed: settings via A2FLive.ini only"); return; }
    auto add = (void (*)(const char*, void (*)()))GetProcAddress(m, "AddSectionItem");
#define IG(n) (ig.n = (decltype(ig.n))GetProcAddress(m, "ig" #n))
    if (!add || !IG(SliderFloat) || !IG(Checkbox) || !IG(Text) || !IG(SeparatorText) || !IG(Button) || !IG(SetTooltip) || !IG(IsItemHovered) || !IG(SameLine)) {
        Log("SKSE Menu Framework: missing exports, no menu page"); return;
    }
    add("A2F Live/Settings", RenderMenu);
    Log("menu page registered (SKSE Menu Framework)");
}

// ---------------- SKSE ----------------
static void OnMessage(SKSEMessage* m) {
    if (m->type != kMessage_DataLoaded) return;
    RetakeSites();
    RegisterMenu();
    HANDLE h = CreateThread(nullptr, 0, Worker, nullptr, 0, nullptr);
    if (h) { SetThreadPriority(h, THREAD_PRIORITY_BELOW_NORMAL); CloseHandle(h); }
}

extern "C" __declspec(dllexport) bool SKSEPlugin_Load(const SKSEInterface* skse) {
    OpenLog();
    g_base = (u64)GetModuleHandleA(nullptr);
    Log("A2FLive " A2FLIVE_VERSION ": runtime %08X", skse->runtimeVersion);
    if (!LoadAddressLibrary(PLUGINS "versionlib-1-6-1170-0.bin")) { Log("address library missing/unreadable"); return false; }
    if (!Addr(ID_GenerateFromPath) || !Addr(ID_StreamCtor) || !Addr(ID_StreamDtor)) { Log("address ids not found"); return false; }
    LoadIni();
    if (!HookCalls("speak", ID_SpeakVoiceLine, SPEAK_CALLERS, (void*)&SpeakHook)) return false;
    HookCalls("response", ID_DialogueResponseCtor, RESPONSE_CALLERS, (void*)&ResponseHook);
    if (!HookCalls("face", ID_FaceAnimUpdate, FACE_CALLERS, (void*)&AnimUpdateHook)) return false;
    auto* msg = (SKSEMessagingInterface*)skse->QueryInterface(kInterface_Messaging);
    if (!msg || !msg->RegisterListener(skse->GetPluginHandle(), "SKSE", OnMessage)) { Log("no messaging interface"); return false; }
    return true;
}

BOOL WINAPI DllMain(HINSTANCE, DWORD, LPVOID) { return TRUE; }
