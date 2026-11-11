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

#include "a2f/animation/rest_store.h"
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
  for (const char* word :
       {"mouth", "teeth", "tooth", "tongue", "gums", "dentition"})
    if (name.find(word) != std::string::npos) return true;
  return false;
}

std::string ExpressionPath(const RE::BGSHeadPart& part) {
  const char* path =
      part.morphs[RE::BGSHeadPart::MorphIndex::kDefaultMorph].model.c_str();
  if (!path) return {};
  std::string normalized = Lower(path);
  std::replace(normalized.begin(), normalized.end(), '/', '\\');
  return normalized;
}

}  // namespace

// Extra parts (long beards' _1bit meshes, Khajiit mustache alphas, lashes,
// hairlines, blind eyes) are mostly Misc: they take their parent's type.
// Hair extras stay excluded unless an authored beard resource identifies them.
// Custom slots (e.g. piercings) keep their full 32-bit type.
void LoadHeadPartTypes() {
  head_part_types.clear();
  mouth_extras.clear();
  hair_parts.clear();
  const auto* forms = FormArray(kFtHeadPart);
  if (!forms) {
    Log("head parts: no form array, using shape names");
    return;
  }
  std::unordered_map<std::string, uint32_t> types;
  std::unordered_set<std::string> duplicate_names;
  for (uint32_t i = 0; i < forms->size(); ++i) {
    auto* form = forms->data()[i];
    if (!form || form->formType != RE::FormType::HeadPart) {
      Log("head parts: unexpected form %u of %u, using shape names", i,
          forms->size());
      return;
    }
    auto* part = static_cast<RE::BGSHeadPart*>(form);
    const char* name = part->formEditorID.c_str();
    if (name && *name) {
      auto [it, inserted] =
          types.insert_or_assign(Lower(name), part->type.underlying());
      if (!inserted) duplicate_names.insert(it->first);
    }
  }
  // Exclusion takes precedence over inherited types and mouth hints.
  // Follow the complete graph, including unnamed intermediates, custom slots,
  // shared extras and cycles. A hair-owned accessory must not move separately
  // from the primary hair, even when another part also references it.
  std::vector<RE::BGSHeadPart*> hair_pending;
  std::unordered_set<RE::BGSHeadPart*> hair_visited;
  for (auto* form : *forms) {
    auto* hp = static_cast<RE::BGSHeadPart*>(form);
    const char* name = hp->formEditorID.c_str();
    if (hp->type.underlying() == kHpHair ||
        (HairFallbackType(hp->type.underlying()) && name &&
         HairName(Lower(name))))
      hair_pending.push_back(hp);
  }
  while (!hair_pending.empty()) {
    auto* hp = hair_pending.back();
    hair_pending.pop_back();
    if (!hair_visited.insert(hp).second) continue;
    const char* name = hp->formEditorID.c_str();
    if (name && *name) hair_parts.insert(Lower(name));
    const auto& extras = hp->extraParts;
    if (!ValidArray(extras)) continue;
    for (auto* extra : extras) {
      if (extra && extra->formType.underlying() == kFtHeadPart)
        hair_pending.push_back(extra);
    }
  }
  // Mouths have no dedicated engine slot. Propagate a candidate hint along
  // their Misc extras, not a type: geometry must still confirm the hint.
  // A worklist handles nested extras and cycles without depending on order.
  std::vector<RE::BGSHeadPart*> pending;
  std::unordered_set<RE::BGSHeadPart*> visited;
  for (auto* form : *forms) {
    auto* hp = static_cast<RE::BGSHeadPart*>(form);
    const char* name = hp->formEditorID.c_str();
    if (hp->type.underlying() == kHpMisc && name && MouthName(Lower(name)))
      pending.push_back(hp);
  }
  while (!pending.empty()) {
    auto* hp = pending.back();
    pending.pop_back();
    if (!visited.insert(hp).second) continue;
    const auto& extras = hp->extraParts;
    if (!ValidArray(extras)) continue;
    for (auto* extra : extras) {
      if (!extra || extra->formType.underlying() != kFtHeadPart ||
          extra->type.underlying() != kHpMisc)
        continue;
      const char* name = extra->formEditorID.c_str();
      if (name && JewelryName(Lower(name))) continue;
      if (name && *name) mouth_extras.insert(Lower(name));
      pending.push_back(extra);
    }
  }
  uint32_t inherited = 0;
  for (auto* form : *forms) {
    auto* hp = static_cast<RE::BGSHeadPart*>(form);
    uint32_t type = hp->type.underlying();
    const auto& extras = hp->extraParts;
    if (type == kHpMisc || type == kHpHair || !ValidArray(extras)) continue;
    for (auto* extra : extras) {
      const char* name = extra && extra->formType == RE::FormType::HeadPart
                             ? extra->formEditorID.c_str()
                             : nullptr;
      if (!name || !*name || extra->type.underlying() != kHpMisc) continue;
      // Oral Misc parts retain their identity even when owned/shared by a
      // face, beard or eye part. Ambiguous names still need geometry approval.
      const std::string n = Lower(name);
      if (MouthName(n) || mouth_extras.count(n)) continue;
      types[n] = type;
      ++inherited;
    }
  }
  // Some replacers package a second beard layer under Hair. A facial name
  // alone is insufficient: require the expression TRI of an independent,
  // explicitly typed beard. This changes registry entries once, not playback.
  std::unordered_set<std::string> beard_morphs;
  for (auto* form : *forms) {
    auto* part = static_cast<RE::BGSHeadPart*>(form);
    const char* name = part->formEditorID.c_str();
    if (part->type.underlying() != kHpFacialHair || !name ||
        hair_visited.count(part) || hair_parts.count(Lower(name)))
      continue;
    std::string path = ExpressionPath(*part);
    if (!path.empty()) beard_morphs.insert(std::move(path));
  }
  for (auto* form : *forms) {
    const auto* part = static_cast<RE::BGSHeadPart*>(form);
    const char* name = part->formEditorID.c_str();
    if (part->type.underlying() != kHpMisc || !name) continue;
    const std::string n = Lower(name);
    // A name-wide exception must not admit another form with the same name.
    if (!hair_parts.count(n) || duplicate_names.count(n) ||
        !FacialHairName(n) || HairName(n) || JewelryName(n) ||
        !beard_morphs.count(ExpressionPath(*part)))
      continue;
    hair_parts.erase(n);
    types[n] = kHpFacialHair;
  }
  head_part_types = std::move(types);
  Log("head parts: %u, %zu named, %u extras typed by their parent",
      forms->size(), head_part_types.size(), inherited);
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
  uint8_t* property =
      reinterpret_cast<uint8_t*>(reinterpret_cast<RE::BSGeometry*>(shape)
                                     ->GetGeometryRuntimeData()
                                     .shaderProperty.get());
  if (!property ||
      !(reinterpret_cast<RE::BSShaderProperty*>(property)->flags.underlying() &
        kShaderHairTint))
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
  uint8_t** extra = reinterpret_cast<uint8_t**>(
      reinterpret_cast<RE::NiObjectNET*>(shape)->extra);
  uint16_t n = reinterpret_cast<RE::NiObjectNET*>(shape)->extraDataSize;
  for (uint16_t i = 0; extra && i < n; ++i) {
    uint8_t* e = extra[i];
    const char* name =
        e ? reinterpret_cast<RE::NiExtraData*>(e)->name.c_str() : nullptr;
    if (!name || strcmp(name, "FOD")) continue;
    nv = reinterpret_cast<RE::BSFaceGenBaseMorphExtraData*>(e)->vertexCount;
    return reinterpret_cast<const float*>(
        reinterpret_cast<RE::BSFaceGenBaseMorphExtraData*>(e)->vertexData);
  }
  return nullptr;
}

static RestStore unlisted_rest;

std::shared_ptr<const std::vector<float>> UnlistedRest(uint8_t* shape,
                                                       uint32_t nv) {
  std::shared_ptr<const std::vector<float>> rest;
  const float* data = Engine<LockDynamicFn, kIdLockDynamic>()(shape);
  if (data) rest = unlisted_rest.Rest(shape, data, nv);
  Engine<UnlockDynamicFn, kIdUnlockDynamic>()(shape);
  return rest;
}

void NoteUnlistedWrite(uint8_t* shape, const float* xyz, uint32_t nv) {
  unlisted_rest.Wrote(shape, xyz, nv);
}

void ResetUnlisted(uint8_t* node) {
  if (unlisted_rest.empty()) return;
  const auto& children = reinterpret_cast<RE::NiNode*>(node)->GetChildren();
  const auto* kids = children.begin();
  const uint16_t n = children.free_idx();
  std::vector<float> xyz;
  for (uint16_t i = 0; kids && i < n; ++i) {
    auto* geometry = reinterpret_cast<RE::BSGeometry*>(kids[i].get());
    if (!geometry || geometry->GetType().underlying() != 4) continue;
    auto* dynamic = static_cast<RE::BSDynamicTriShape*>(geometry);
    auto* shape = reinterpret_cast<uint8_t*>(dynamic);
    const uint32_t nv = dynamic->GetTrishapeRuntimeData().vertexCount;
    uint32_t nb = 0;
    if (dynamic->GetDynamicTrishapeRuntimeData().dataSize != nv * 16 ||
        BaseVerts(shape, nb))
      continue;
    const float* data = Engine<LockDynamicFn, kIdLockDynamic>()(shape);
    if (data && unlisted_rest.Displaced(shape, data, nv, xyz))
      Engine<UpdateDynamicFn, kIdUpdateDynamic>()(shape, xyz.data(), false);
    Engine<UnlockDynamicFn, kIdUnlockDynamic>()(shape);
  }
}

const char* StaticMorphPart(uint8_t* node) {
  const auto& children = reinterpret_cast<RE::NiNode*>(node)->GetChildren();
  const auto* kids = children.begin();
  const uint16_t n = children.free_idx();
  for (uint16_t i = 0; kids && i < n; ++i) {
    auto* shape = reinterpret_cast<uint8_t*>(kids[i].get());
    uint32_t nv = 0;
    // Base vertices first: only face geometry has them, so the type is not
    // read from a child that is no geometry.
    if (!shape || !BaseVerts(shape, nv) ||
        reinterpret_cast<RE::BSGeometry*>(shape)->GetType().underlying() == 4 ||
        !Engine<MorphDataFn, kIdMorphData>()(shape))
      continue;
    const char* name = reinterpret_cast<RE::NiObjectNET*>(shape)->name.c_str();
    return name ? name : "";
  }
  return nullptr;
}

namespace {

// A follower shape's triangles (global vertex indices), which tell its hair
// cards apart: from the renderer's index copy, else, as on a skinned FaceGen
// shape whose block has none, from its skin partitions. Empty when neither
// gives a complete, in-range list.
void ReadFollowerTriangles(uint8_t* shape, direct::ShapeIn& s) {
  s.triangles.clear();
  if (s.skin || s.part != direct::Part::kHead) return;
  auto* geometry = reinterpret_cast<RE::BSTriShape*>(shape);
  const auto& runtime = geometry->GetGeometryRuntimeData();
  const auto* renderer = runtime.rendererData;
  const auto& mesh = geometry->GetTrishapeRuntimeData();
  if (mesh.vertexCount != s.nv) return;
  auto in_range = [&](const uint16_t* indices, size_t count, uint32_t limit) {
    return std::all_of(indices, indices + count,
                       [&](uint16_t index) { return index < limit; });
  };
  if (renderer && renderer->rawIndexData && mesh.triangleCount) {
    const auto* indices = renderer->rawIndexData;
    const size_t count = size_t{mesh.triangleCount} * 3;
    if (in_range(indices, count, s.nv))
      s.triangles.assign(indices, indices + count);
    return;
  }
  const auto* skin = runtime.skinInstance.get();
  const auto* partition = skin ? skin->skinPartition.get() : nullptr;
  // The array's own pointer: its size() reads an allocation header.
  const auto* parts =
      partition
          ? *reinterpret_cast<const RE::NiSkinPartition::Partition* const*>(
                &partition->partitions)
          : nullptr;
  constexpr uint32_t kMaxPartitions = 64;
  if (!parts || !partition->numPartitions ||
      partition->numPartitions > kMaxPartitions)
    return;
  std::vector<uint16_t> triangles;
  std::vector<uint8_t> owned(s.nv);
  for (uint32_t k = 0; k < partition->numPartitions; ++k) {
    const auto& part = parts[k];
    const uint16_t* indices = part.buffData && part.buffData->rawIndexData
                                  ? part.buffData->rawIndexData
                                  : part.triList;
    const size_t count = size_t{part.triangles} * 3;
    if (!count) continue;
    if (!indices) return;
    // Indices are the shape's own, or the partition's (through its vertex
    // map) when they do not all name vertices the partition holds.
    const uint16_t* map = part.vertexMap;
    bool global = in_range(indices, count, s.nv);
    if (map) {
      if (!in_range(map, part.vertices, s.nv)) return;
      std::fill(owned.begin(), owned.end(), 0);
      for (uint16_t i = 0; i < part.vertices; ++i) owned[map[i]] = 1;
      global = global && std::all_of(indices, indices + count,
                                     [&](uint16_t i) { return owned[i] != 0; });
    }
    if (global) {
      triangles.insert(triangles.end(), indices, indices + count);
    } else if (map && in_range(indices, count, part.vertices)) {
      for (size_t i = 0; i < count; ++i) triangles.push_back(map[indices[i]]);
    } else {
      return;
    }
  }
  s.triangles = std::move(triangles);
}

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

// Retained storage, including spare capacity; excludes allocator metadata.
size_t HeadBytes(const direct::Head& h) {
  auto n = [](const auto& v) { return v.capacity() * sizeof(v[0]); };
  size_t b = sizeof(h) + n(h.lip_jaw) + n(h.row) + n(h.basis) + n(h.region_w) +
             n(h.lip_w) + n(h.shapes) + n(h.calibrated_lips);
  for (const auto& s : h.shapes) {
    b += n(s.name) + n(s.idx) + n(s.w) + n(s.pos) + n(s.jaw) +
         n(s.tongue_weight) + n(s.tongue) + n(s.model_jaw) + n(s.model_tongue) +
         n(s.walls) + n(s.seam);
    for (const auto& e : s.expr) b += n(e);
  }
  return b;
}

void LogHead(const std::string& key, const direct::Head& h, double ms,
             uint32_t followers, uint32_t with_triangles) {
  const float* r = h.residual;
  Log("head %s: %u skin verts, %zu shapes, Mark x%.3f, landmarks off by "
      "blink %.2f %.2f brow %.2f %.2f jaw %.2f lips %.2f smile %.2f %.2f, "
      "lips %u upper %u lower (to Mark's %.2f), mouth part %u vertices "
      "on the jaw, %u tongue, strength brows %.2f cheeks %.2f mouth "
      "%.2f, calibration ignored %u, lip calibration %zu, triangles for %u "
      "of %u followers, %.1f ms",
      key.c_str(), h.nv, h.shapes.size(), h.scale, r[0], r[1], r[2], r[3], r[4],
      r[5], r[6], r[7], h.lips[0], h.lips[1], h.lip_dist, h.jaw_verts,
      h.tongue_verts, h.auto_scale[direct::kBrows],
      h.auto_scale[direct::kCheeks], h.auto_scale[direct::kMouth],
      h.calibration_ignored, h.calibrated_lips.size(), with_triangles,
      followers, ms);
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
      uint32_t followers = 0, with_triangles = 0;
      for (const auto& s : j.shapes) {
        if (s.skin || s.part != direct::Part::kHead) continue;
        ++followers;
        with_triangles += !s.triangles.empty();
      }
      LogHead(j.key, *h, (Now() - t) * 1e3, followers, with_triangles);
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
// vertices and a hash (FNV-1a) of each shape's name, type, all base vertices,
// morph resource and whether it is unlisted, independent of child order. With
// a face per sex, the sex selects the model face. Empty: the node has no face
// shapes.
std::string HeadKey(uint8_t* node, bool female,
                    std::shared_ptr<const direct::Model> mapping) {
#ifdef ENABLE_COMMONLIBSSE_TESTING
  if (testing::head_key) return testing::head_key(node, female, mapping);
#endif
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
    uint32_t nb = 0;
    const bool unlisted = !BaseVerts(shape, nb);
    mix(&unlisted, sizeof unlisted);
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
#ifdef ENABLE_COMMONLIBSSE_TESTING
  if (testing::find_head)
    return testing::find_head(node, key, female, out, mapping);
#endif
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
    uint32_t nb = 0;
    s.unlisted = !BaseVerts(shape, nb);
    ReadMorphs(shape, s);
    ReadFollowerTriangles(shape, s);
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
