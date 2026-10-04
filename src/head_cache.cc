// The head mappings declared in head_cache.h.
#include "head_cache.h"

#include <windows.h>

#include <cstdio>
#include <cstring>
#include <deque>
#include <iterator>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ids.h"
#include "scoped_lock.h"
#include "skse.h"

namespace bfa {

using skse::Addr;
using skse::Log;

namespace {

struct HeadEntry {
  State state = State::kPending;  // failed: not mappable
  std::shared_ptr<const direct::Head> head;
  size_t bytes = 0;
  uint64_t last_use = 0;
};
struct MapJob {
  std::string key;
  std::vector<direct::ShapeIn> shapes;
};
constexpr size_t kHeadCacheBytes = 128u << 20;
SRWLOCK map_lock = SRWLOCK_INIT;  // guards the five below
CONDITION_VARIABLE map_cv = CONDITION_VARIABLE_INIT;
auto& heads = *new std::unordered_map<std::string, HeadEntry>;
auto& map_queue = *new std::deque<MapJob>;
uint64_t head_clock;
bool mapper_started;

// Head part type by lowercase editor id, which FaceGen names its shapes
// after; filled once at data loaded, empty if the forms didn't look right.
auto& head_part_types = *new std::unordered_map<std::string, uint8_t>;

}  // namespace

// Extra parts (long beards' _1bit meshes, Khajiit mustache alphas, lashes,
// hairlines, blind eyes) are mostly Misc: they take their parent's type.
// Types past kEyebrows, which some mods use, are left to the shape name.
void LoadHeadPartTypes() {
  const BSTArray* parts = FormArray(kFtHeadPart);
  if (!parts) {
    Log("head parts: no form array, using shape names");
    return;
  }
  auto type_of = [](uint8_t* hp) { return Field<uint32_t>(hp, kHeadPartType); };
  auto name_of = [](uint8_t* hp) {
    return Field<const char*>(hp, kHeadPartName);
  };
  std::unordered_map<std::string, uint8_t> types;
  for (uint32_t i = 0; i < parts->n; ++i) {
    uint8_t* hp = parts->items[i];
    if (!hp || hp[kFormType] != kFtHeadPart) {
      Log("head parts: unexpected form %u of %u, using shape names", i,
          parts->n);
      return;
    }
    uint32_t type = type_of(hp);
    const char* name = name_of(hp);
    if (type < kHpNumTypes && name && *name)
      types[Lower(name)] = static_cast<uint8_t>(type);
  }
  uint32_t inherited = 0;
  for (uint32_t i = 0; i < parts->n; ++i) {
    uint32_t type = type_of(parts->items[i]);
    const auto& extras = Field<BSTArray>(parts->items[i], kHeadPartExtras);
    if (type == kHpMisc || type >= kHpNumTypes || !extras.Ok()) continue;
    for (uint32_t j = 0; j < extras.n; ++j) {
      uint8_t* x = extras.items[j];
      const char* name =
          x && x[kFormType] == kFtHeadPart ? name_of(x) : nullptr;
      if (!name || !*name || type_of(x) != kHpMisc) continue;
      types[Lower(name)] = static_cast<uint8_t>(type);
      ++inherited;
    }
  }
  head_part_types = std::move(types);
  Log("head parts: %u, %zu named, %u extras typed by their parent", parts->n,
      head_part_types.size(), inherited);
}

// The head part type of a shape, or kHpNumTypes when unknown.
static uint8_t HeadPartTypeOf(const std::string& n) {
  auto it = head_part_types.find(n);
  return it == head_part_types.end() ? uint8_t{kHpNumTypes} : it->second;
}

direct::Part Classify(const char* name, bool& skin) {
  std::string n = Lower(name);
  uint8_t type = HeadPartTypeOf(n);
  skin = type != kHpNumTypes ? type == kHpFace
                             : n.find("head") != std::string::npos &&
                                   n.find("brow") == std::string::npos &&
                                   n.find("beard") == std::string::npos;
  switch (type) {
    // Scar is also used for antlers and the like, which map to the scalp,
    // where the face doesn't move
    case kHpFace:
    case kHpFacialHair:
    case kHpScar:
    case kHpEyebrows:
      return direct::Part::kHead;
    case kHpEyes:
      return direct::Part::kEyes;
    case kHpHair:
      return direct::Part::kNone;
  }
  // Misc (mouths, stray parts) or unknown: by name
  if (n.find("mouth") != std::string::npos) return direct::Part::kMouth;
  if (n.find("eyes") != std::string::npos) return direct::Part::kEyes;
  for (const char* w : {"head", "brow", "beard", "mustache", "moustache",
                        "goatee", "stubble", "sideburn", "scar"})
    if (n.find(w) != std::string::npos) return direct::Part::kHead;
  return direct::Part::kNone;
}

const float* BaseVerts(uint8_t* shape, uint32_t& nv) {
  uint8_t** extra = Field<uint8_t**>(shape, kObjExtra);
  uint16_t n = Field<uint16_t>(shape, kObjNumExtra);
  for (uint16_t i = 0; extra && i < n; ++i) {
    uint8_t* e = extra[i];
    const char* name = e ? Field<const char*>(e, kObjName) : nullptr;
    if (!name || strcmp(name, "FOD")) continue;
    nv = Field<uint32_t>(e, kBaseNumVerts);
    return Field<const float*>(e, kBaseVerts);
  }
  return nullptr;
}

namespace {

// The shape's morphs it Reads (direct::Reads), each read by applying it at
// 1.0 and putting the vertices back; all zero where the shape lacks one,
// empty when the shape has no morphs at all.
void ReadMorphs(uint8_t* shape, direct::ShapeIn& s) {
  const auto lock_dyn = Engine<LockDynamicFn, kIdLockDynamic>();
  const auto unlock_dyn = Engine<UnlockDynamicFn, kIdUnlockDynamic>();
  const auto apply = Engine<ApplyMorphFn, kIdApplyMorph>();
  s.morphs.resize(direct::kNumMorphs);
  void* md = Engine<MorphDataFn, kIdMorphData>()(shape);
  if (!md) return;
  std::vector<float> before(static_cast<size_t>(s.nv) * 4);
  const size_t bytes = before.size() * sizeof(float);
  float* data = lock_dyn(shape);
  memcpy(before.data(), data, bytes);
  unlock_dyn(shape);
  for (int m = 0; m < direct::kNumMorphs; ++m) {
    if (!direct::Reads(s.skin, s.part, m)) continue;
    apply(md, direct::kMorphIds[m].type, direct::kMorphIds[m].index, shape,
          1.f);
    std::vector<float>& out = s.morphs[m];
    out.resize(static_cast<size_t>(s.nv) * 3);
    data = lock_dyn(shape);
    for (uint32_t i = 0; i < s.nv; ++i)
      for (int c = 0; c < 3; ++c)
        out[i * 3 + c] = data[i * 4 + c] - before[i * 4 + c];
    memcpy(data, before.data(), bytes);
    unlock_dyn(shape);
  }
}

// The memory a head mapping holds.
size_t HeadBytes(const direct::Head& h) {
  auto n = [](const auto& v) { return v.size() * sizeof(v[0]); };
  size_t b = n(h.row) + n(h.basis) + n(h.region_w) + n(h.shapes);
  for (const auto& s : h.shapes) {
    b += n(s.name) + n(s.idx) + n(s.w) + n(s.pos) + n(s.jaw) + n(s.tongue);
    for (const auto& e : s.expr) b += n(e);
  }
  return b;
}

void LogHead(const std::string& key, const direct::Head& h, double ms) {
  const float* r = h.residual;
  Log("head %s: %u skin verts, %zu shapes, Mark x%.3f, landmarks off by "
      "blink %.2f %.2f brow %.2f %.2f jaw %.2f lips %.2f smile %.2f %.2f, "
      "lips %u upper %u lower (to Mark's %.2f), mouth part %u vertices "
      "on the jaw, %u tongue, strength brows %.2f cheeks %.2f mouth "
      "%.2f, %.1f ms",
      key.c_str(), h.nv, h.shapes.size(), h.scale, r[0], r[1], r[2], r[3], r[4],
      r[5], r[6], r[7], h.lips[0], h.lips[1], h.lip_dist, h.jaw_verts,
      h.tongue_verts, h.auto_scale[direct::kBrows],
      h.auto_scale[direct::kCheeks], h.auto_scale[direct::kMouth], ms);
}

// map_lock held: drops the least recently used heads past the budget, but
// keep (a head in use is kept alive by its face).
void EvictHeads(const std::string& keep) {
  size_t total = 0;
  for (auto& [k, v] : heads) total += v.bytes;
  while (total > kHeadCacheBytes) {
    auto old = heads.end();
    for (auto it = heads.begin(); it != heads.end(); ++it)
      if (it->second.state == State::kReady &&
          (old == heads.end() || it->second.last_use < old->second.last_use))
        old = it;
    if (old == heads.end() || old->first == keep) break;
    total -= old->second.bytes;
    heads.erase(old);
  }
}

DWORD WINAPI Mapper(void*) {
  for (;;) {
    ScopedLock guard(map_lock);
    while (map_queue.empty())
      SleepConditionVariableSRW(&map_cv, &map_lock, INFINITE, 0);
    MapJob j = std::move(map_queue.front());
    map_queue.pop_front();
    guard.Unlock();
    double t = Now();
    auto h = std::make_shared<direct::Head>();
    std::string why;
    bool ok = direct::BuildHead(model, j.shapes, *h, why);
    if (ok) {
      LogHead(j.key, *h, (Now() - t) * 1e3);
    } else {
      Log("head %s: not animated, %s", j.key.c_str(), why.c_str());
    }
    guard.Lock();
    HeadEntry& e = heads[j.key];
    e.state = ok ? State::kReady : State::kFailed;
    e.bytes = ok ? HeadBytes(*h) : 0;
    if (ok) e.head = std::move(h);
    e.last_use = ++head_clock;
    EvictHeads(j.key);
  }
}

}  // namespace

// What tells one head from another: the number of shapes, the skin's
// vertices and a hash (FNV-1a) of each shape's name, type, all base vertices
// and morph resource. Empty: the node has no face shapes.
std::string HeadKey(uint8_t* node) {
  uint64_t h = 14695981039346656037ull;
  auto mix = [&h](const void* p, size_t n) {
    for (size_t i = 0; i < n; ++i)
      h = (h ^ static_cast<const uint8_t*>(p)[i]) * 1099511628211ull;
  };
  uint32_t shapes = 0, skin_nv = 0;
  ForFaceShapes(node, [&](uint8_t* shape, const char* name, direct::Part part,
                          bool skin, uint32_t nv, const float* base) {
    mix(name, strlen(name) + 1);
    mix(&nv, sizeof nv);
    mix(&part, sizeof part);
    mix(&skin, sizeof skin);
    mix(base, static_cast<size_t>(nv) * 3 * sizeof(float));
    const void* morphs = Engine<MorphDataFn, kIdMorphData>()(shape);
    mix(&morphs, sizeof morphs);
    ++shapes;
    if (skin && !skin_nv) skin_nv = nv;
  });
  if (!shapes) return "";
  char key[48];
  snprintf(key, sizeof key, "%u/%u/%016llx", shapes, skin_nv,
           static_cast<unsigned long long>(h));
  return key;
}

State FindHead(uint8_t* node, const std::string& key,
               std::shared_ptr<const direct::Head>& out) {
  if (key.empty()) return State::kFailed;
  ScopedLock guard(map_lock);
  auto it = heads.find(key);
  if (it != heads.end()) {
    it->second.last_use = ++head_clock;
    out = it->second.head;
    State state = it->second.state;
    return state;
  }
  heads.emplace(key, HeadEntry());
  guard.Unlock();
  MapJob j{key, {}};
  ForFaceShapes(node, [&](uint8_t* shape, const char* name, direct::Part part,
                          bool skin, uint32_t nv, const float* base) {
    direct::ShapeIn s;
    s.name = name;
    s.part = part;
    s.skin = skin && part == direct::Part::kHead;
    s.nv = nv;
    s.verts.assign(base, base + nv * 3);
    ReadMorphs(shape, s);
    j.shapes.push_back(std::move(s));
  });
  guard.Lock();
  map_queue.push_back(std::move(j));
  if (!mapper_started) {
    HANDLE th = CreateThread(nullptr, 0, Mapper, nullptr, 0, nullptr);
    mapper_started = th != nullptr;
    if (th) CloseHandle(th);
  }
  WakeConditionVariable(&map_cv);
  return State::kPending;
}

void ClearHeads() {
  ScopedLock guard(map_lock);
  for (auto it = heads.begin(); it != heads.end();)
    it = it->second.state != State::kPending ? heads.erase(it) : std::next(it);
}

}  // namespace bfa
