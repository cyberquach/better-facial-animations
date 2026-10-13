// The head mappings declared in head_cache.h.
#include "a2f/animation/head_cache.h"

#include <windows.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <deque>
#include <iterator>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "a2f/runtime/ids.h"
#include "a2f/runtime/scoped_lock.h"
#include "a2f/runtime/skse.h"

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
  bool female = false;  // the face it is mapped onto (direct::Model::Face)
  std::vector<direct::ShapeIn> shapes;
  std::shared_ptr<const direct::Model> mapping;
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
auto& head_part_types = *new std::unordered_map<std::string, uint32_t>;
auto& mouth_extras = *new std::unordered_set<std::string>;
auto& hair_parts = *new std::unordered_set<std::string>;

bool FacialHairName(const std::string& name) {
  for (const char* word :
       {"facialhair", "facial_hair", "facial hair", "beard", "mustache",
        "moustache", "goatee", "stubble", "sideburn"})
    if (name.find(word) != std::string::npos) return true;
  return false;
}

// Fallback for unregistered shapes and Misc/custom hair slots. Explicit
// facial-hair names remain facial followers, not scalp hair.
bool HairName(const std::string& name) {
  // Invisible SMP support meshes must also stay with their hair.
  if (name.find("virtualhead") != std::string::npos ||
      name.find("virtualbody") != std::string::npos)
    return true;
  if (name.find("scalp") != std::string::npos) return true;
  if (name.find("hair") == std::string::npos) return false;
  return !FacialHairName(name);
}

bool HairFallbackType(uint32_t type) {
  return type == kHpMisc || type > kHpEyebrows;
}

bool JewelryName(const std::string& name) {
  for (const char* word : {"pierc", "ring", "stud", "pendant", "necklace"})
    if (name.find(word) != std::string::npos) return true;
  return false;
}

bool MouthName(const std::string& name) {
  if (JewelryName(name)) return false;
  for (const char* word : {"mouth", "teeth", "tooth", "tongue", "gums",
                           "dentition"})
    if (name.find(word) != std::string::npos) return true;
  return false;
}

}  // namespace

// Extra parts (long beards' _1bit meshes, Khajiit mustache alphas, lashes,
// hairlines, blind eyes) are mostly Misc: they take their parent's type.
// Hair and all linked extras are excluded, including nested/shared extras.
// Custom slots (e.g. piercings) keep their full 32-bit type.
void LoadHeadPartTypes() {
  head_part_types.clear();
  mouth_extras.clear();
  hair_parts.clear();
  const BSTArray* parts = FormArray(kFtHeadPart);
  if (!parts) {
    Log("head parts: no form array, using shape names");
    return;
  }
  auto type_of = [](uint8_t* hp) { return Field<uint32_t>(hp, kHeadPartType); };
  auto name_of = [](uint8_t* hp) {
    return Field<const char*>(hp, kHeadPartName);
  };
  std::unordered_map<std::string, uint32_t> types;
  for (uint32_t i = 0; i < parts->n; ++i) {
    uint8_t* hp = parts->items[i];
    if (!hp || hp[kFormType] != kFtHeadPart) {
      Log("head parts: unexpected form %u of %u, using shape names", i,
          parts->n);
      return;
    }
    uint32_t type = type_of(hp);
    const char* name = name_of(hp);
    if (name && *name) types[Lower(name)] = type;
  }
  // Exclusion takes precedence over every inherited type and mouth hint.
  // Follow the complete graph, including unnamed intermediates, custom slots,
  // shared extras and cycles. A hair-owned accessory must not move separately
  // from the primary hair, even when another part also references it.
  std::vector<uint8_t*> hair_pending;
  std::unordered_set<uint8_t*> hair_visited;
  for (uint32_t i = 0; i < parts->n; ++i) {
    uint8_t* hp = parts->items[i];
    const char* name = name_of(hp);
    if (type_of(hp) == kHpHair ||
        (HairFallbackType(type_of(hp)) && name && HairName(Lower(name))))
      hair_pending.push_back(hp);
  }
  while (!hair_pending.empty()) {
    uint8_t* hp = hair_pending.back();
    hair_pending.pop_back();
    if (!hair_visited.insert(hp).second) continue;
    const char* name = name_of(hp);
    if (name && *name) hair_parts.insert(Lower(name));
    const auto& extras = Field<BSTArray>(hp, kHeadPartExtras);
    if (!extras.Ok()) continue;
    for (uint32_t j = 0; j < extras.n; ++j) {
      uint8_t* extra = extras.items[j];
      if (extra && extra[kFormType] == kFtHeadPart)
        hair_pending.push_back(extra);
    }
  }
  // Mouths have no dedicated engine slot. Propagate a candidate hint along
  // their Misc extras, not a type: geometry must still confirm the hint.
  // A worklist handles nested extras and cycles without depending on order.
  std::vector<uint8_t*> pending;
  std::unordered_set<uint8_t*> visited;
  for (uint32_t i = 0; i < parts->n; ++i) {
    uint8_t* hp = parts->items[i];
    const char* name = name_of(hp);
    if (type_of(hp) == kHpMisc && name && MouthName(Lower(name)))
      pending.push_back(hp);
  }
  while (!pending.empty()) {
    uint8_t* hp = pending.back();
    pending.pop_back();
    if (!visited.insert(hp).second) continue;
    const auto& extras = Field<BSTArray>(hp, kHeadPartExtras);
    if (!extras.Ok()) continue;
    for (uint32_t j = 0; j < extras.n; ++j) {
      uint8_t* extra = extras.items[j];
      if (!extra || extra[kFormType] != kFtHeadPart ||
          type_of(extra) != kHpMisc) continue;
      const char* name = name_of(extra);
      if (name && JewelryName(Lower(name))) continue;
      if (name && *name) mouth_extras.insert(Lower(name));
      pending.push_back(extra);
    }
  }
  uint32_t inherited = 0;
  for (uint32_t i = 0; i < parts->n; ++i) {
    uint32_t type = type_of(parts->items[i]);
    const auto& extras = Field<BSTArray>(parts->items[i], kHeadPartExtras);
    if (type == kHpMisc || type == kHpHair || !extras.Ok()) continue;
    for (uint32_t j = 0; j < extras.n; ++j) {
      uint8_t* x = extras.items[j];
      const char* name =
          x && x[kFormType] == kFtHeadPart ? name_of(x) : nullptr;
      if (!name || !*name || type_of(x) != kHpMisc) continue;
      // Oral Misc parts retain their identity even when owned/shared by a
      // face, beard or eye part. Ambiguous names still need geometry approval.
      const std::string n = Lower(name);
      if (MouthName(n) || mouth_extras.count(n)) continue;
      types[n] = type;
      ++inherited;
    }
  }
  head_part_types = std::move(types);
  Log("head parts: %u, %zu named, %u extras typed by their parent", parts->n,
      head_part_types.size(), inherited);
}

// The head part type of a shape, or kHpUnknown.
constexpr uint32_t kHpUnknown = UINT32_MAX;
static uint32_t HeadPartTypeOf(const std::string& n) {
  auto it = head_part_types.find(n);
  return it == head_part_types.end() ? kHpUnknown : it->second;
}

bool ExcludedHair(const std::string& n, uint32_t type) {
  return type == kHpHair || hair_parts.count(n) != 0 ||
         (HairFallbackType(type) && HairName(n));
}

// MouthCandidate for an already lowered name and its registered type.
bool OralCandidate(const std::string& n, uint32_t type) {
  if (ExcludedHair(n, type)) return false;
  if (type != kHpUnknown && type != kHpMisc) return false;
  // Explicit jewelry names win even when attached as mouth extras.
  if (JewelryName(n)) return false;
  return MouthName(n) || mouth_extras.count(n) != 0;
}

bool MouthCandidate(const char* name) {
  const std::string n = Lower(name);
  return OralCandidate(n, HeadPartTypeOf(n));
}

direct::Part Classify(const char* name, bool& skin) {
  std::string n = Lower(name);
  uint32_t type = HeadPartTypeOf(n);
  skin = false;
  if (ExcludedHair(n, type)) return direct::Part::kNone;
  skin = type != kHpUnknown ? type == kHpFace
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
  // a mod's own slot (piercings): rigid pieces pinned to the skin
  if (type != kHpUnknown && type > kHpEyebrows) return direct::Part::kPinned;
  // Explicit jewelry stays attached to skin, even with "mouth" in its name.
  if (JewelryName(n))
    return type != kHpUnknown ? direct::Part::kPinned : direct::Part::kNone;
  // Misc (mouths, stray parts) or unknown: by name
  if (n.find("mouth") != std::string::npos) {
    skin = false;
    return direct::Part::kMouth;
  }
  // Teeth/tongue names and unnamed oral descendants can contain head/eye
  // words too. Keep them out of skin selection until geometry confirms them.
  if (OralCandidate(n, type)) {
    skin = false;
    return type != kHpUnknown ? direct::Part::kPinned : direct::Part::kNone;
  }
  if (n.find("eyes") != std::string::npos) return direct::Part::kEyes;
  for (const char* w : {"head", "brow", "beard", "mustache", "moustache",
                        "goatee", "stubble", "sideburn", "scar", "lash"})
    if (n.find(w) != std::string::npos) return direct::Part::kHead;
  // A registered Misc part named like none of those is an accessory, pinned
  // like the piercings; geometry that is no head part at all is left alone.
  return type != kHpUnknown ? direct::Part::kPinned : direct::Part::kNone;
}

bool HairGeometry(uint8_t* shape, const char* name) {
  uint8_t* property = Field<uint8_t*>(shape, kGeomShaderProperty.Get());
  if (!property || !(Field<uint64_t>(property, kShaderFlags) & kShaderHairTint))
    return false;
  // NIF shape names can differ from HDPT editor IDs (e.g. KS Scarlet).
  // The shader flag closes that gap without changing materials or rendering.
  const std::string n = Lower(name);
  const uint32_t type = HeadPartTypeOf(n);
  if (ExcludedHair(n, type)) return true;
  // The flag alone only excludes slots that can hold scalp hair (Misc,
  // custom, unregistered): tinted faces, scars, eyes and lashes, facial hair
  // and brows keep their controls.
  if (type != kHpUnknown && !HairFallbackType(type)) return false;
  if (FacialHairName(n)) return false;
  for (const char* word : {"brow", "lash"})
    if (n.find(word) != std::string::npos) return false;
  return true;
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
    if (!direct::Reads(s.skin, s.part, m, s.mouth_candidate)) continue;
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
  size_t b = n(h.row) + n(h.basis) + n(h.region_w) + n(h.lip_w) + n(h.shapes);
  for (const auto& s : h.shapes) {
    b += n(s.name) + n(s.idx) + n(s.w) + n(s.pos) + n(s.jaw) +
         n(s.tongue_weight) + n(s.tongue) + n(s.model_jaw) + n(s.model_tongue);
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
      "%.2f, calibration ignored %u, %.1f ms",
      key.c_str(), h.nv, h.shapes.size(), h.scale, r[0], r[1], r[2], r[3], r[4],
      r[5], r[6], r[7], h.lips[0], h.lips[1], h.lip_dist, h.jaw_verts,
      h.tongue_verts, h.auto_scale[direct::kBrows],
      h.auto_scale[direct::kCheeks], h.auto_scale[direct::kMouth],
      h.calibration_ignored, ms);
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
    bool ok = direct::BuildHead(j.mapping->Face(j.female), j.shapes, *h, why);
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
// and morph resource, independent of child order. With a face per sex, the
// sex selects the model face. Empty: the node has no face shapes.
std::string HeadKey(uint8_t* node, bool female,
                    std::shared_ptr<const direct::Model> mapping) {
  if (!mapping) mapping = SnapshotModel()->mapping;
  uint64_t h = 14695981039346656037ull;
  auto mix = [&h](const void* p, size_t n) {
    for (size_t i = 0; i < n; ++i)
      h = (h ^ static_cast<const uint8_t*>(p)[i]) * 1099511628211ull;
  };
  std::vector<uint64_t> shape_hashes;
  uint32_t skin_nv = 0;
  ForFaceShapes(node, [&](uint8_t* shape, const char* name, direct::Part part,
                          bool skin, uint32_t nv, const float* base) {
    h = 14695981039346656037ull;
    mix(name, strlen(name) + 1);
    mix(&nv, sizeof nv);
    mix(&part, sizeof part);
    mix(&skin, sizeof skin);
    const bool mouth_candidate = MouthCandidate(name);
    mix(&mouth_candidate, sizeof mouth_candidate);
    mix(base, static_cast<size_t>(nv) * 3 * sizeof(float));
    const void* morphs = Engine<MorphDataFn, kIdMorphData>()(shape);
    mix(&morphs, sizeof morphs);
    shape_hashes.push_back(h);
    if (skin) skin_nv = std::max(skin_nv, nv);
  });
  if (shape_hashes.empty()) return "";
  std::sort(shape_hashes.begin(), shape_hashes.end());
  h = 14695981039346656037ull;
  mix(shape_hashes.data(), shape_hashes.size() * sizeof(uint64_t));
  char key[64];
  snprintf(key, sizeof key, "%d/%zu/%u/%016llx%s", mapping->network,
           shape_hashes.size(), skin_nv, static_cast<unsigned long long>(h),
           mapping->faces.size() < 2 ? ""
           : female                  ? "/f"
                                     : "/m");
  return key;
}

State FindHead(uint8_t* node, const std::string& key, bool female,
               std::shared_ptr<const direct::Head>& out,
               std::shared_ptr<const direct::Model> mapping) {
  if (!mapping) mapping = SnapshotModel()->mapping;
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
  MapJob j{key, female, {}, std::move(mapping)};
  ForFaceShapes(node, [&](uint8_t* shape, const char* name, direct::Part part,
                          bool skin, uint32_t nv, const float* base) {
    direct::ShapeIn s;
    s.name = name;
    s.part = part;
    s.mouth_candidate = MouthCandidate(name);
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
