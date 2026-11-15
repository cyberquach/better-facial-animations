#include "runtime_fixture.h"
// What the render thread can copy from a speaking face, frame by frame. The
// real AnimUpdateHook (keyframe decisions) and ApplyMorphsHook run against an
// engine fake whose ApplyMorphs resets each listed shape, under that shape's
// lock, to base + morphs of the keyframes it finds (phonemes and modifiers on
// x and y), and a hook chained behind the plugin's (it can skip the engine
// or write an unlisted shape after it). BFA's deltas are on z only. Whenever
// a shape's lock depth returns to 0 the render observer checks it, and
// whenever all are unlocked it checks them together: every shape shows the
// engine's pose of the keyframes it reset from, plus at most one head's
// frame at one weight, the same on every shape of that head; the vanilla lips
// left in those keyframes plus that weight make one mouth (sum 1, at most 1
// while a pose eases); and every write of the plugin falls inside the lock
// span that holds the shape from its reset to its deltas.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "../src/plugin/better_facial_animations.cc"

namespace ff {
using namespace bfa;

// ---------------- failures ----------------
std::string case_name;
int frame_no = 0;
int case_failures = 0, total_failures = 0;
std::vector<std::string> failed_cases;
void Fail(const char* format, ...) {
  ++case_failures;
  ++total_failures;
  if (case_failures > 12) return;
  char text[512];
  va_list args;
  va_start(args, format);
  vsnprintf(text, sizeof text, format, args);
  va_end(args);
  fprintf(stderr, "FAIL face-frame %s frame %d: %s\n", case_name.c_str(),
          frame_no, text);
}
#define REQUIRE(x)                          \
  do {                                      \
    if (!(x)) Fail("%s (line %d)", #x, __LINE__); \
  } while (0)

// ---------------- fixture ----------------
enum { kHeadShape, kBrows, kBrows2, kLashes, kNumShapes };
struct ShapeFix {
  const char* name;
  uint32_t nv;
  bool listed;  // has base vertices ("FOD"): the engine resets it
  alignas(16) std::array<uint8_t, 0x200> object{};
  alignas(16) std::array<uint8_t, 0x40> fod{};
  uint8_t* extra = nullptr;
  std::vector<float> base;        // nv x 3: listed base, else the true rest
  std::vector<float> buffer;      // nv x 4: the dynamic vertices
  std::vector<float> engine_out;  // nv x 3: the engine's last reset
  int depth = 0;
  // the lock span: reset by the engine (or, unlisted, first written by the
  // plugin) in this hook call, and let go (depth 0) since
  bool reset_in_call = false, released_since_reset = false;
};
ShapeFix shapes[kNumShapes] = {{"FemaleHead", 3, true},
                               {"FemaleBrows", 2, true},
                               {"FemaleBrows2", 2, true},
                               {"FemaleLashes", 2, false}};
alignas(16) std::array<uint8_t, 0x500> node{}, face{};
alignas(16) std::array<uint8_t, 0x1000> actor_data{}, npc_base{}, race{};
uint8_t* const fg = face.data();
uint8_t* const actor = actor_data.data();
uint8_t* children[kNumShapes] = {};
float phon[direct::kNumPhonemes], mods[14], expr[17];

int Index(uint8_t* s) {
  for (int i = 0; i < kNumShapes; ++i)
    if (s == shapes[i].object.data()) return i;
  Fail("a lock or write on a shape that is not the face's");
  std::exit(1);
}
bool IsChild(int s) {
  const uint16_t n = Field<uint16_t>(node.data(), 0x122);
  for (uint16_t i = 0; i < n; ++i)
    if (children[i] == shapes[s].object.data()) return true;
  return false;
}
void SetChildren(std::initializer_list<int> list) {
  uint16_t n = 0;
  for (int s : list) children[n++] = shapes[s].object.data();
  Field<uint16_t>(node.data(), 0x122) = n;
}

// ---------------- heads (the mapping) ----------------
// Two heads with distinct per-vertex patterns: A with FemaleBrows, B (after
// the brows are swapped) with FemaleBrows2. One coefficient, z only.
struct HeadFix {
  std::string key;
  int follower;  // kBrows or kBrows2
  float skin[3];
  std::shared_ptr<direct::Head> head;
  std::vector<float> pattern[kNumShapes];  // z per vertex for c = 1
};
HeadFix heads[2] = {{"head A", kBrows, {1, 2, 3}}, {"head B", kBrows2, {2, 3, 1}}};

void BuildHeads() {
  for (HeadFix& h : heads) {
    auto head = std::make_shared<direct::Head>();
    head->k = 1;
    head->nv = 3;
    head->row = {0, 1, 2};
    head->basis.assign(3 * 1 * 3, 0.f);
    head->region_w.assign(3 * direct::kNumSkinRegions, 0.f);
    for (int v = 0; v < 3; ++v) {
      head->basis[v * 3 + 2] = h.skin[v];
      head->region_w[v * direct::kNumSkinRegions + direct::kMouth] = 1;
    }
    for (int i = 0; i < a2f::kNumJaw; ++i)
      head->jaw_points[i] = static_cast<float>((i * 7) % 5 + i / 3);
    head->jaw_motion[0] = head->jaw_motion[4] = head->jaw_motion[8] = 1;
    // followers: brows v0 <- skin 0, v1 <- skin 2 at 0.5; lashes v0 <- skin
    // 1, v1 <- skin 2
    auto follow = [&](int s, uint32_t a, float wa, uint32_t b, float wb) {
      direct::Head::Shape shape;
      shape.name = shapes[s].name;
      shape.part = direct::Part::kHead;
      shape.nv = 2;
      shape.idx.assign(2 * direct::kFollow, 0);
      shape.w.assign(2 * direct::kFollow, 0.f);
      shape.idx[0] = a;
      shape.w[0] = wa;
      shape.idx[direct::kFollow] = b;
      shape.w[direct::kFollow] = wb;
      head->shapes.push_back(shape);
      h.pattern[s] = {wa * h.skin[a], wb * h.skin[b]};
    };
    direct::Head::Shape skin;
    skin.name = shapes[kHeadShape].name;
    skin.part = direct::Part::kHead;
    skin.nv = 3;
    head->shapes.push_back(skin);
    head->skin = 0;
    h.pattern[kHeadShape] = {h.skin[0], h.skin[1], h.skin[2]};
    follow(h.follower, 0, 1, 2, 0.5f);
    follow(kLashes, 1, 1, 2, 1);
    h.head = head;
  }
}

// HeadKey: from the node's shapes, as the real one is. FindHead: pending for
// the key's first pending_calls calls, then ready.
std::map<std::string, int> pending_calls;
int find_calls = 0;
extern "C" std::string StubHeadKey(uint8_t*, bool,
                                   std::shared_ptr<const direct::Model>) {
  return IsChild(kBrows2) ? heads[1].key : heads[0].key;
}
extern "C" State StubFindHead(uint8_t*, const std::string& key, bool,
                              std::shared_ptr<const direct::Head>& out,
                              std::shared_ptr<const direct::Model>) {
  ++find_calls;
  out.reset();
  int& left = pending_calls[key];
  if (left > 0) {
    --left;
    return State::kPending;
  }
  for (HeadFix& h : heads)
    if (h.key == key) out = h.head;
  return out ? State::kReady : State::kFailed;
}

// ---------------- engine fakes ----------------
// The engine's lip sample (vanilla phonemes): copied into phoneme3 only when
// it changes (every 2 frames), as the engine does; modifiers are recomputed
// every update.
bool vanilla = false;
int sample = -1, written_sample = -2;
float Vanilla(int i) { return vanilla ? 0.1f + 0.05f * ((i + sample) % 4) : 0.f; }
float VanillaMod(int i) {
  return vanilla ? (i == 2 ? 0.3f : i == 12 ? 0.2f : 0.f) : 0.f;
}
bool EngineUpdate(uint8_t* f, float, uint64_t) {
  if (f != fg) Fail("the engine updated another face");
  const int now_sample = vanilla ? frame_no / 2 : -1;
  if (now_sample != written_sample) {
    sample = now_sample;
    written_sample = now_sample;
    for (int i = 0; i < direct::kNumPhonemes; ++i) phon[i] = Vanilla(i);
  }
  for (int i = 0; i < 14; ++i) mods[i] = VanillaMod(i);
  return true;
}

// Keyframes the last engine reset found (K) and the vanilla it would show
// without the plugin (V): phonemes plus the modifiers the plugin takes over.
double shown_k = 0, shown_v = 0;
double LipSum(const float* p, const float* m) {
  double sum = 0;
  for (int i = 0; i < direct::kNumPhonemes; ++i) sum += p[i];
  for (int i : kModZeroed) sum += m[i];
  return sum;
}

int bfa_writes = 0, foreign_writes = 0, engine_resets = 0;
bool in_hook = false, orig_running = false, foreign_writing = false;
void ObserveShape(int s);
void ObserveAll(const char* when);
float* ShapeLock(uint8_t* p) {
  ShapeFix& s = shapes[Index(p)];
  ++s.depth;
  return s.buffer.data();
}
void ShapeUnlock(uint8_t* p) {
  const int i = Index(p);
  ShapeFix& s = shapes[i];
  if (s.depth <= 0) {
    Fail("%s unlocked more than locked", s.name);
    return;
  }
  if (--s.depth) return;
  if (in_hook && s.reset_in_call) s.released_since_reset = true;
  ObserveShape(i);
  bool all = true;
  for (const ShapeFix& o : shapes) all &= o.depth == 0;
  if (all) ObserveAll("all unlocked");
}
void ShapeUpdate(uint8_t* p, const float* xyz, bool) {
  const int i = Index(p);
  ShapeFix& s = shapes[i];
  if (foreign_writing) {
    ++foreign_writes;
  } else {
    // coordinator case (2): the 2026-11-03 span
    ++bfa_writes;
    if (s.depth <= 0) Fail("the plugin wrote %s without its lock", s.name);
    if (in_hook && s.released_since_reset)
      Fail("the plugin wrote %s after the hook let it go since its reset",
           s.name);
    if (in_hook && !s.listed && !s.reset_in_call) s.reset_in_call = true;
  }
  for (uint32_t v = 0; v < s.nv; ++v)
    std::copy_n(xyz + v * 3, 3, s.buffer.data() + v * 4);
}
void* NoMorphData(uint8_t*) { return nullptr; }
// The engine's ApplyMorphs: a reset (force or the node's flag) rebuilds every
// listed child, each under its own (nested) lock, then clears the flag.
uint64_t EngineMorphs(uint8_t* n, uint8_t force) {
  if (n != node.data()) Fail("the engine morphed another node");
  if (!force && !(Field<uint16_t>(n, 0x160) & 1)) return 7;
  ++engine_resets;
  shown_k = LipSum(phon, mods);
  float v_phon[direct::kNumPhonemes], v_mods[14];
  for (int i = 0; i < direct::kNumPhonemes; ++i) v_phon[i] = Vanilla(i);
  for (int i = 0; i < 14; ++i) v_mods[i] = VanillaMod(i);
  shown_v = LipSum(v_phon, v_mods);
  double all_phon = 0, all_mods = 0;
  for (float p : phon) all_phon += p;
  for (float m : mods) all_mods += m;
  for (int i = 0; i < kNumShapes; ++i) {
    ShapeFix& s = shapes[i];
    if (!s.listed || !IsChild(i)) continue;
    float* data = ShapeLock(s.object.data());
    for (uint32_t v = 0; v < s.nv; ++v) {
      float out[3] = {s.base[v * 3], s.base[v * 3 + 1], s.base[v * 3 + 2]};
      out[0] += static_cast<float>(all_phon * 0.1 * (v + 1) * (i + 1));
      out[1] += static_cast<float>(all_mods * 0.05 * (v + 1));
      std::copy_n(out, 3, data + v * 4);
      std::copy_n(out, 3, s.engine_out.data() + v * 3);
    }
    if (in_hook) {
      s.reset_in_call = true;
      s.released_since_reset = false;
    }
    ShapeUnlock(s.object.data());
  }
  Field<uint16_t>(n, 0x160) &= ~1;
  return 7;
}
// The hook behind the plugin's: normally the engine; it can skip the
// engine's pass, or write an unlisted shape after it (another mod).
bool chain_skip = false, chain_foreign = false;
constexpr float kForeignEps = 2e-3f;
uint64_t Chain(uint8_t* n, uint8_t force) {
  orig_running = true;
  uint64_t r = 7;
  if (!chain_skip) r = EngineMorphs(n, force);
  if (chain_foreign && IsChild(kLashes)) {
    ShapeFix& s = shapes[kLashes];
    float* data = ShapeLock(s.object.data());
    std::vector<float> xyz(s.nv * 3);
    for (uint32_t v = 0; v < s.nv; ++v)
      for (int c = 0; c < 3; ++c)
        xyz[v * 3 + c] = data[v * 4 + c] + (v == 0 && c == 0 ? kForeignEps : 0);
    foreign_writing = true;
    ShapeUpdate(s.object.data(), xyz.data(), false);
    foreign_writing = false;
    ShapeUnlock(s.object.data());
  }
  orig_running = false;
  return r;
}
bool Lookup(const uint32_t& h, uint8_t** out) {
  *out = h == 42 ? actor : nullptr;
  if (*out) ++Field<LONG>(*out, 0x28);
  return *out != nullptr;
}
std::vector<std::string> reports;
void Report(const char* format, ...) {
  char text[1024];
  va_list args;
  va_start(args, format);
  vsnprintf(text, sizeof text, format, args);
  va_end(args);
  reports.push_back(text);
}
int Reported(const char* what) {
  int n = 0;
  for (const std::string& r : reports) n += r.find(what) != std::string::npos;
  return n;
}
}  // namespace ff

extern "C" uint64_t TestAddress(uint64_t id) {
  using namespace bfa;
  if (id == kIdFaceAnimUpdate.Get())
    return reinterpret_cast<uint64_t>(&ff::EngineUpdate);
  if (id == kIdApplyMorphs.Get()) return reinterpret_cast<uint64_t>(&ff::Chain);
  if (id == kIdLockDynamic.Get())
    return reinterpret_cast<uint64_t>(&ff::ShapeLock);
  if (id == kIdUnlockDynamic.Get())
    return reinterpret_cast<uint64_t>(&ff::ShapeUnlock);
  if (id == kIdUpdateDynamic.Get())
    return reinterpret_cast<uint64_t>(&ff::ShapeUpdate);
  if (id == kIdMorphData.Get())
    return reinterpret_cast<uint64_t>(&ff::NoMorphData);
  if (id == kIdLookupReference.Get())
    return reinterpret_cast<uint64_t>(&ff::Lookup);
  return 0;
}

namespace ff {
// ---------------- the render observer ----------------
// One shape as the render thread would copy it: the residual over the
// engine's last reset (listed) or the true rest (unlisted) is zero on x/y
// (a foreign writer's small x on unlisted shapes aside), and on z either
// zero or one head's pattern at a single coefficient.
struct Fit {
  int head = -1;  // -1: the engine's pose alone
  float c = 0;
  bool ok = true;
};
Fit FitShape(int i) {
  const ShapeFix& s = shapes[i];
  const std::vector<float>& under = s.listed ? s.engine_out : s.base;
  const float xy_tol = s.listed ? 1e-4f : kForeignEps + 1e-4f;
  std::vector<float> rz(s.nv);
  float peak = 0;
  Fit fit;
  for (uint32_t v = 0; v < s.nv; ++v) {
    for (int c = 0; c < 2; ++c)
      if (std::fabs(s.buffer[v * 4 + c] - under[v * 3 + c]) > xy_tol) {
        Fail("%s vertex %u axis %d is %g, the engine's %g: not the engine's "
             "pose of its keyframes",
             s.name, v, c, s.buffer[v * 4 + c], under[v * 3 + c]);
        fit.ok = false;
      }
    rz[v] = s.buffer[v * 4 + 2] - under[v * 3 + 2];
    peak = std::max(peak, std::fabs(rz[v]));
  }
  if (peak < 1e-4f) return fit;
  for (int h = 0; h < 2; ++h) {
    const std::vector<float>& p = heads[h].pattern[i];
    if (p.empty()) continue;
    double dot = 0, pp = 0;
    for (uint32_t v = 0; v < s.nv; ++v) {
      dot += rz[v] * p[v];
      pp += p[v] * p[v];
    }
    const float c = static_cast<float>(dot / pp);
    float resid = 0;
    for (uint32_t v = 0; v < s.nv; ++v)
      resid = std::max(resid, std::fabs(rz[v] - c * p[v]));
    if (resid <= 1e-4f + 1e-3f * std::fabs(c)) {
      fit.head = h;
      fit.c = c;
      return fit;
    }
  }
  char z[96];
  snprintf(z, sizeof z, "%g %g %g", rz[0], rz[1], s.nv > 2 ? rz[2] : 0.f);
  Fail("%s shows deltas of no single head (z residual %s)", s.name, z);
  fit.ok = false;
  return fit;
}

// The full speech pose's coefficient on the meshes (line coefficient times
// the mouth's speech scale); the fade phases the test declares.
float c_line = 0.2f, c_full = 0;
bool easing_phase = false;
int joint_checks = 0, posed_checks = 0;
float max_sum = 0;
constexpr float kTolSum = 0.03f;

void ObserveShape(int i) {
  if (!IsChild(i)) return;
  const Fit fit = FitShape(i);
  if (fit.head >= 0 && c_full > 0 && fit.c > c_full * (1 + kTolSum))
    Fail("%s shows %s at %.3f of the full pose: deltas added twice",
         shapes[i].name, heads[fit.head].key.c_str(), fit.c / c_full);
}

void ObserveAll(const char* when) {
  ++joint_checks;
  Fit fits[kNumShapes];
  int head = -1;
  for (int i = 0; i < kNumShapes; ++i) {
    if (!IsChild(i)) continue;
    fits[i] = FitShape(i);
    if (fits[i].head < 0) continue;
    if (head >= 0 && fits[i].head != head)
      Fail("%s: shapes show %s and %s at once", when, heads[head].key.c_str(),
           heads[fits[i].head].key.c_str());
    head = fits[i].head;
  }
  float c = 0;
  if (head >= 0) {
    // every shape of that head, at one weight
    c = fits[kHeadShape].c;
    for (int i = 0; i < kNumShapes; ++i) {
      if (!IsChild(i) || heads[head].pattern[i].empty()) continue;
      if (std::fabs(fits[i].c - c) > 1e-3f * std::max(c_full, std::fabs(c)) + 1e-5f)
        Fail("%s: %s shows %s at %.4f, %s at %.4f: a partial head", when,
             shapes[i].name, heads[head].key.c_str(), fits[i].c,
             shapes[kHeadShape].name, c);
    }
  }
  if (c_full <= 0) return;
  const float pose = c / c_full;
  if (pose > 1 + kTolSum)
    Fail("%s: the pose is at %.3f of full: deltas added twice", when, pose);
  if (pose > 0.01f) ++posed_checks;
  if (shown_v <= 1e-6) return;
  // the vanilla lips the keyframes still carry, and the pose: one mouth
  const float lips = static_cast<float>(shown_k / shown_v);
  const float sum = lips + pose;
  max_sum = std::max(max_sum, sum);
  if (sum > 1 + kTolSum)
    Fail("%s: vanilla lips at %.3f plus the pose at %.3f: a doubled mouth",
         when, lips, pose);
  else if (!easing_phase && sum < 1 - kTolSum)
    Fail("%s: vanilla lips at %.3f plus the pose at %.3f: a neutral mouth",
         when, lips, pose);
}

// ---------------- driving ----------------
std::shared_ptr<a2f::Line> MakeLine(uint32_t frames) {
  auto line = std::make_shared<a2f::Line>();
  line->frames = frames;
  line->fps = 30;
  line->direct_dim = 1 + direct::kNumMouthCoef;
  line->direct.assign(size_t{frames} * line->direct_dim, 0.f);
  for (uint32_t f = 0; f < frames; ++f) line->direct[f * line->direct_dim] = c_line;
  return line;
}
void SetLineCoefficient(a2f::Line& line, float c) {
  for (uint32_t f = 0; f < line.frames; ++f) line.direct[f * line.direct_dim] = c;
}

std::shared_ptr<Entry> StartLine(uint32_t frames, bool generated) {
  auto entry = std::make_shared<Entry>();
  entry->state = generated ? State::kReady : State::kPending;
  if (generated) entry->line = MakeLine(frames);
  Playing p;
  p.name = "line";
  p.actor = actor;
  p.actor_handle = 42;
  p.female = true;
  p.entry = entry;
  p.voice.requested_at = p.voice.at = Now();
  p.voice.seen = true;
  p.begun = Now();
  ScopedLock guard(lock);
  playing[fg] = p;
  Count();
  return entry;
}
void Generated(const std::shared_ptr<Entry>& entry, uint32_t frames) {
  ScopedLock guard(lock);
  entry->line = MakeLine(frames);
  entry->state = State::kReady;
}
bool HasFace() {
  ScopedLock guard(lock, ScopedLock::kShared);
  return faces.count(fg) != 0;
}
bool HasLine() {
  ScopedLock guard(lock, ScopedLock::kShared);
  return playing.count(fg) != 0;
}

void MorphCall(uint8_t force) {
  for (ShapeFix& s : shapes) s.reset_in_call = s.released_since_reset = false;
  in_hook = true;
  ApplyMorphsHook(node.data(), force);
  in_hook = false;
  for (const ShapeFix& s : shapes)
    if (s.depth) Fail("%s still locked after the morph call", s.name);
  if (Field<uint32_t>(fg, 0x224)) Fail("the face's lock is still held");
}

// One game frame: the face's animation update (the real hook, the engine's
// keyframes behind it), the node's reset flag if it changed, then 1-2 morph
// passes; extra: none, a second without reset (0) or with force (1).
void Frame(int extra = -1) {
  ++frame_no;
  testing::clock_offset = testing::clock_offset + 1.0 / 60;
  const bool changed = AnimUpdateHook(fg, 1.f / 60, 0);
  if (changed) Field<uint16_t>(node.data(), 0x160) |= 1;
  MorphCall(0);
  if (extra >= 0) MorphCall(static_cast<uint8_t>(extra));
  ObserveAll("after the frame");
}
void Frames(int n, int extra = -1) {
  for (int i = 0; i < n; ++i) Frame(extra);
}
// Alternating second passes: without a reset, then forced.
void MixedFrames(int n) {
  for (int i = 0; i < n; ++i) Frame(i % 3 == 0 ? -1 : i % 3 == 1 ? 0 : 1);
}

void SetPotato(bool on) {
  ScopedLock guard(settings_lock);
  for (const Param& p : kParams)
    if (!strcmp(p.key, "PotatoMode")) *p.flag = on;
}

void Begin(const char* name) {
  ResetPlayback(false);
  case_name = name;
  case_failures = 0;
  frame_no = 0;
  testing::clock_offset = testing::clock_offset + 10;
  for (int i = 0; i < kNumShapes; ++i) {
    ShapeFix& s = shapes[i];
    for (uint32_t v = 0; v < s.nv; ++v)
      for (int c = 0; c < 3; ++c) {
        s.base[v * 3 + c] = static_cast<float>(i * 10 + v * 3 + c + 1);
        s.engine_out[v * 3 + c] = s.base[v * 3 + c];
        s.buffer[v * 4 + c] = s.base[v * 3 + c];
      }
    s.depth = 0;
  }
  SetChildren({kHeadShape, kBrows, kLashes});
  Field<uint16_t>(node.data(), 0x160) = 0;
  pending_calls.clear();
  find_calls = 0;
  vanilla = false;
  written_sample = -2;
  std::fill(std::begin(phon), std::end(phon), 0.f);
  std::fill(std::begin(mods), std::end(mods), 0.f);
  std::fill(std::begin(expr), std::end(expr), 0.f);
  shown_k = shown_v = 0;
  chain_skip = chain_foreign = false;
  bfa_writes = foreign_writes = engine_resets = 0;
  joint_checks = posed_checks = 0;
  max_sum = 0;
  easing_phase = false;
  c_line = 0.2f;
  reports.clear();
  SetPotato(false);
}
void End() {
  printf("%s: %d frames, %d engine resets, %d plugin writes, %d snapshots "
         "(%d posed), max lips+pose %.3f%s\n",
         case_name.c_str(), frame_no, engine_resets, bfa_writes, joint_checks,
         posed_checks, max_sum, case_failures ? ": FAILED" : "");
  if (case_failures) failed_cases.push_back(case_name);
  SetPotato(false);
}

// ---------------- cases ----------------
// A line whose head is mapped 3 morph calls in (FindHead turns ready inside
// the morph call), played with reset, no-reset and forced second passes.
void StartReadyInCall() {
  Begin("start-ready-in-call");
  pending_calls[heads[0].key] = 3;
  vanilla = true;
  StartLine(150, true);
  Frames(6);
  MixedFrames(40);
  REQUIRE(bfa_writes > 0 && posed_checks > 20);
  End();
}
// The brows are swapped mid-line (a new head key): the old head is never
// drawn on the new shapes; the new one is mapped k calls later (k frames,
// under 0.5 s) or at once.
void KeyChange(int k) {
  Begin(k ? "key-change-pending" : "key-change-ready");
  vanilla = true;
  StartLine(150, true);
  Frames(20, 1);
  pending_calls[heads[1].key] = k;
  SetChildren({kHeadShape, kBrows2, kLashes});
  Frames(k + 25, 1);
  REQUIRE(posed_checks > 20);
  End();
}
// A frame with an unsafe delta goes back to the engine whole.
void UnsafeDelta() {
  Begin("unsafe-delta");
  vanilla = true;
  auto entry = StartLine(150, true);
  Frames(15);
  a2f::Line& line = const_cast<a2f::Line&>(*entry->line);
  SetLineCoefficient(line, 100);
  Frames(6, 1);
  SetLineCoefficient(line, c_line);
  Frames(15);
  REQUIRE(Reported("rejected frame") > 0);
  End();
}
// A hook behind the plugin's skips the engine's reset for some frames.
void SkippedReset() {
  Begin("skipped-reset");
  vanilla = true;
  StartLine(150, true);
  Frames(15);
  chain_skip = true;
  Frames(4, 1);
  chain_skip = false;
  Frames(15);
  REQUIRE(Reported("face reset not done") > 0);
  End();
}
// Another mod writes the unlisted lashes after the engine's pass.
void ForeignUnlisted() {
  Begin("foreign-unlisted");
  vanilla = true;
  StartLine(150, true);
  Frames(10);
  chain_foreign = true;
  MixedFrames(30);
  REQUIRE(foreign_writes > 0);
  End();
}
// The line ends (vanilla lips stop with its audio), the pose eases out over
// FaceFadeOut and the face goes back to the engine; then engine frames.
void LineEndHandBack() {
  Begin("line-end-hand-back");
  vanilla = true;
  StartLine(60, true);  // 2 s; over at 2.3 s
  Frames(118);
  vanilla = false;
  easing_phase = true;
  for (int i = 0; i < 180 && HasFace(); ++i) Frame(1);
  REQUIRE(!HasFace() && !HasLine());
  easing_phase = false;
  const int writes = bfa_writes;
  Frames(10, 1);
  REQUIRE(bfa_writes == writes);
  End();
}
// Potato mode mid-line drops the face at once: the hand-back resets every
// shape, unlisted too, in one span.
void AbruptHandBack() {
  Begin("abrupt-hand-back");
  vanilla = true;
  StartLine(150, true);
  Frames(30);
  SetPotato(true);
  Frames(1);
  REQUIRE(!HasFace());
  Frames(10, 1);
  End();
}
// Coordinator case (1): line A ends and its pose eases out while line B of
// the same speaker is still generating and the engine plays its vanilla
// lips; then B is ready (late, blended in from the share A's pose had left).
void NextLinePending() {
  Begin("next-line-pending");
  vanilla = true;
  StartLine(60, true);
  Frames(118);
  vanilla = false;
  Frames(25);  // past 2.3 s: A has ended, its pose eases out
  REQUIRE(!HasLine() && HasFace());
  easing_phase = true;
  auto b = StartLine(150, false);
  vanilla = true;
  Frames(36);  // 0.6 s
  REQUIRE(HasLine() && posed_checks > 0);
  Generated(b, 150);
  Frames(30);
  End();
}

// A line generated 0.3 s into its speech (late) starts from rest, blended in
// over BlendIn while vanilla lips play: the lips give way by the same curve.
void LateStart() {
  Begin("late-start");
  vanilla = true;
  auto entry = StartLine(150, false);
  Frames(18);
  Generated(entry, 150);
  Frames(30);
  End();
}

void Seed() {
  for (int i = 0; i < kNumShapes; ++i) {
    ShapeFix& s = shapes[i];
    uint8_t* o = s.object.data();
    o[0x150] = 4;  // BSDynamicTriShape
    Field<const char*>(o, 0x10) = FixtureString(s.name);
    Field<uint16_t>(o, 0x15A) = static_cast<uint16_t>(s.nv);
    Field<uint32_t>(o, 0x170) = s.nv * 16;
    s.base.resize(s.nv * 3);
    s.engine_out.resize(s.nv * 3);
    s.buffer.assign(s.nv * 4, 0.f);
    if (s.listed) {
      s.extra = s.fod.data();
      Field<uint8_t**>(o, 0x20) = &s.extra;
      Field<uint16_t>(o, 0x28) = 1;
      Field<const char*>(s.fod.data(), 0x10) = FixtureString("FOD");
      Field<const float*>(s.fod.data(), 0x18) = s.base.data();
      Field<uint32_t>(s.fod.data(), 0x24) = s.nv;
    }
  }
  Field<uint8_t**>(node.data(), 0x118) = children;
  Field<uint8_t*>(node.data(), 0x150) = fg;
  Field<uint8_t*>(node.data(), 0xF8) = actor;
  auto& ph = FixtureKeyframe(fg, 0x140);
  ph.values = phon;
  ph.count = direct::kNumPhonemes;
  auto& md = FixtureKeyframe(fg, 0x100);
  md.values = mods;
  md.count = 14;
  auto& ex = FixtureKeyframe(fg, 0xC0);
  ex.values = expr;
  ex.count = 17;
  Field<uint8_t*>(actor, 0x40) = npc_base.data();
  actor[0x1A] = kFtActor;
  npc_base[0x1A] = kFtNpc;
  Field<uint8_t*>(actor, 0x1f8) = race.data();
  Field<uint32_t>(race.data(), 0x14) = 0x13746;  // Nord
  Field<uint32_t>(race.data(), 0x108) = 2;
  Field<uint32_t>(actor, 0x14) = 0x123;
  Field<LONG>(actor, 0x28) = 1;
  ready = true;
  loading_game = false;
  BuildHeads();
  float scales[direct::kNumRegions];
  SpeechScales(*heads[0].head, NpcFaceTune(SnapshotTune(), actor, true, false),
               true, scales);
  c_full = c_line * scales[direct::kMouth];
}
}  // namespace ff

int main(int argc, char** argv) {
  using namespace ff;
  bfa::testing::address = TestAddress;
  bfa::testing::head_key = StubHeadKey;
  bfa::testing::find_head = StubFindHead;
  bfa::testing::report = Report;
  skse::SetRuntime(skse::Version(1, 6, 1170));
  Seed();
  const std::pair<const char*, void (*)()> cases[] = {
      {"start-ready-in-call", StartReadyInCall},
      {"key-change-pending", [] { KeyChange(12); }},
      {"key-change-ready", [] { KeyChange(0); }},
      {"unsafe-delta", UnsafeDelta},
      {"skipped-reset", SkippedReset},
      {"foreign-unlisted", ForeignUnlisted},
      {"line-end-hand-back", LineEndHandBack},
      {"abrupt-hand-back", AbruptHandBack},
      {"late-start", LateStart},
      {"next-line-pending", NextLinePending},
  };
  int run = 0;
  for (const auto& [name, fn] : cases) {
    if (argc > 1 && strcmp(argv[1], name)) continue;
    fn();
    ++run;
  }
  if (!run) {
    fprintf(stderr, "no case %s\n", argc > 1 ? argv[1] : "");
    return 1;
  }
  if (total_failures) {
    fprintf(stderr, "face-frame: %d failures in", total_failures);
    for (const std::string& c : failed_cases) fprintf(stderr, " %s", c.c_str());
    fprintf(stderr, "\n");
    return 1;
  }
  puts("ok: every face frame is the engine's or one head's whole, one mouth");
}
