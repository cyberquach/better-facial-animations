// SKSE ABI, log, Address Library, engine layout (AE 1.6.1170) and writing a frame of morph curves into a FaceGen head.
#pragma once
#include <windows.h>
#include <shlobj.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdarg>
#include <cmath>
#include <vector>
#include <algorithm>

using u8 = uint8_t; using u32 = uint32_t; using i32 = int32_t; using u64 = uint64_t;

// ---------------- SKSE ABI (skse64 PluginAPI.h) ----------------
struct SKSEPluginVersionData {
    u32 dataVersion; u32 pluginVersion; char name[256]; char author[256]; char supportEmail[252];
    u32 versionIndependenceEx; u32 versionIndependence; u32 compatibleVersions[16]; u32 seVersionRequired;
};
static_assert(sizeof(SKSEPluginVersionData) == 0x350, "SKSEPluginVersionData");
struct SKSEInterface {
    u32 skseVersion, runtimeVersion, editorVersion, isEditor;
    void* (*QueryInterface)(u32); u32 (*GetPluginHandle)(); u32 (*GetReleaseIndex)(); const void* (*GetPluginInfo)(const char*);
};
struct SKSEMessage { const char* sender; u32 type; u32 dataLen; void* data; };
struct SKSEMessagingInterface {
    u32 interfaceVersion;
    bool (*RegisterListener)(u32 handle, const char* sender, void (*cb)(SKSEMessage*));
};
enum { kInterface_Messaging = 5, kMessage_DataLoaded = 8 };

// ---------------- log ----------------
static FILE* g_log;
static void Log(const char* fmt, ...) {
    if (!g_log) return;
    va_list a; va_start(a, fmt); vfprintf(g_log, fmt, a); va_end(a); fputc('\n', g_log); fflush(g_log);
}
static void OpenLog(const wchar_t* name) {
    wchar_t* docs = nullptr; wchar_t path[MAX_PATH];
    if (SHGetKnownFolderPath(FOLDERID_Documents, 0, nullptr, &docs) != S_OK) return;
    _snwprintf(path, MAX_PATH, L"%ls\\My Games\\Skyrim Special Edition\\SKSE\\%ls", docs, name);
    CoTaskMemFree(docs); g_log = _wfopen(path, L"w");
}

// ---------------- Address Library (versionlib format 2, CommonLibSSE-NG REL/ID.h) ----------------
static u64 g_base;
static std::vector<std::pair<u64, u64>> g_ids;  // id -> offset, sorted by id
static bool LoadAddressLibrary(const char* path) {
    FILE* f = fopen(path, "rb"); if (!f) return false;
    std::vector<u8> d; fseek(f, 0, SEEK_END); d.resize(ftell(f)); fseek(f, 0, SEEK_SET);
    fread(d.data(), 1, d.size(), f); fclose(f);
    size_t p = 0;
    auto rd = [&](size_t n) -> u64 { u64 v = 0; memcpy(&v, &d[p], n); p += n; return v; };
    if ((i32)rd(4) != 2) return false;
    p += 16; i32 nameLen = (i32)rd(4); p += nameLen;
    i32 ptrSize = (i32)rd(4), count = (i32)rd(4);
    g_ids.reserve(count);
    u64 pid = 0, poff = 0;
    for (i32 i = 0; i < count; ++i) {
        u8 t = (u8)rd(1); u8 lo = t & 0xF, hi = t >> 4; u64 id, off;
        switch (lo) {
            case 0: id = rd(8); break; case 1: id = pid + 1; break; case 2: id = pid + rd(1); break;
            case 3: id = pid - rd(1); break; case 4: id = pid + rd(2); break; case 5: id = pid - rd(2); break;
            case 6: id = rd(2); break; case 7: id = rd(4); break; default: return false;
        }
        u64 base = (hi & 8) ? poff / ptrSize : poff;
        switch (hi & 7) {
            case 0: off = rd(8); break; case 1: off = base + 1; break; case 2: off = base + rd(1); break;
            case 3: off = base - rd(1); break; case 4: off = base + rd(2); break; case 5: off = base - rd(2); break;
            case 6: off = rd(2); break; case 7: off = rd(4); break; default: return false;
        }
        if (hi & 8) off *= ptrSize;
        g_ids.emplace_back(id, off); pid = id; poff = off;
    }
    std::sort(g_ids.begin(), g_ids.end());
    return true;
}
static u64 Addr(u64 id) {
    auto it = std::lower_bound(g_ids.begin(), g_ids.end(), std::make_pair(id, (u64)0));
    return (it != g_ids.end() && it->first == id) ? g_base + it->second : 0;
}

// ---------------- engine layout (AE 1.6.1170) ----------------
constexpr u64 ID_GenerateFromPath = 69979;
constexpr u32 FG_EXPR3 = 0xC0, FG_MOD3 = 0x100, FG_PHON3 = 0x140, FG_PHON1_TIMER = 0x12C, FG_LOCK = 0x220, FG_DIALOGUE = 0x228;
struct KeyframeMultiple { void* vtbl; u32 type; float timer; float* values; u32 count; bool isUpdated; };
struct ResID { u32 file; u32 ext; u32 dir; };
static inline u64 KeyOf(const ResID& r) { return ((u64)r.dir << 32) | r.file; }
using GenerateFromPath_t = void (*)(ResID*, const char*);

struct SpinLock {  // RE::BSSpinLock
    volatile LONG owner, count;
    void lock() {
        LONG me = (LONG)GetCurrentThreadId();
        if (owner == me) { InterlockedIncrement(&count); return; }
        for (u32 spins = 0; InterlockedCompareExchange(&count, 1, 0) != 0; ++spins) {
            if (spins < 10000) YieldProcessor(); else Sleep(0);
        }
        owner = me;
    }
    void unlock() {
        if (count == 1) { owner = 0; InterlockedCompareExchange(&count, 0, 1); } else InterlockedDecrement(&count);
    }
};

// The head's active dialogue line: lip resource id, or nullptr when no line / lip resource not loaded yet.
static const ResID* ActiveLine(u8* fg) {
    void* entry = *(void**)(fg + FG_DIALOGUE);
    if (!entry) return nullptr;
    u32 ctrl = *(u32*)((u8*)entry + 0x0C);
    if ((((ctrl & 0x70000000u) + 0xD0000000u) & 0xEFFFFFFFu) != 0) return nullptr;  // lip resource not loaded
    return (const ResID*)entry;
}

// ---------------- frame playback ----------------
// line channels: 16 phonemes, 10 modifiers, 1 expression
constexpr int CH_PHON = 0, N_PHON = 16, CH_MOD = 16, CH_EXPR = 26;
// line modifier order -> engine modifier index (BlinkL/R, BrowDownL/R, BrowInL/R, BrowUpL/R, SquintL/R)
constexpr int MOD_ENGINE[10] = { 0, 1, 2, 3, 4, 5, 6, 7, 12, 13 };

static inline float Sample(const u8* fr0, const u8* fr1, float a, int ch) {
    return ((1 - a) * fr0[ch] + a * fr1[ch]) * (1.0f / 255.0f);
}
static inline float Mix(float engine, float ours, float mix) { return mix >= 1.f ? ours : engine + (ours - engine) * mix; }

// Write a line's frame at lip time t (s) into the final phoneme / modifier / expression keyframes.
// data: uint8[frames][nch]; expr: engine expression index (>= 7: none); mix < 1 blends from the engine's values.
static void PlayFrame(u8* fg, const u8* data, u32 frames, u32 nch, u32 fps, u32 expr, float t, float sMouth, float sMod, float sExpr, float mix) {
    auto* phon = (KeyframeMultiple*)(fg + FG_PHON3);
    auto* mod = (KeyframeMultiple*)(fg + FG_MOD3);
    auto* ex = (KeyframeMultiple*)(fg + FG_EXPR3);
    float f = t * (float)fps; if (f < 0) f = 0;
    u32 i0 = (u32)f; float a = f - (float)i0;
    bool past = i0 + 1 >= frames;
    const u8* fr0 = data + (size_t)(past ? frames - 1 : i0) * nch;
    const u8* fr1 = past ? fr0 : fr0 + nch;
    float fade = past ? std::fmax(0.f, 1.f - (f - (float)(frames - 1)) / 3.f) : 1.f;  // ease out over 0.1 s

    if (phon->values && phon->count >= N_PHON) {
        for (int i = 0; i < N_PHON; ++i) phon->values[i] = Mix(phon->values[i], std::fmin(1.f, Sample(fr0, fr1, a, CH_PHON + i) * sMouth * fade), mix);
        phon->isUpdated = true;
    }
    if (mod->values && mod->count >= 14) {
        for (int i = 0; i < 10; ++i) {
            float v = std::fmin(1.f, Sample(fr0, fr1, a, CH_MOD + i) * sMod * fade);
            int e = MOD_ENGINE[i];
            mod->values[e] = (e <= 1) ? std::fmax(mod->values[e], v) : Mix(mod->values[e], v, mix);  // keep the engine's procedural blink
        }
        mod->isUpdated = true;
    }
    if (ex->values && ex->count >= 7) {
        for (int i = 0; i < 7; ++i) {
            float v = (u32)i == expr ? std::fmin(1.f, Sample(fr0, fr1, a, CH_EXPR) * sExpr * fade) : 0.f;
            ex->values[i] = Mix(ex->values[i], v, mix);
        }
        ex->isUpdated = true;
    }
}
