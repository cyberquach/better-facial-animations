// One mapping per distinct head (every face shape's name, vertex count and
// base vertices) and sex onto Mark (the model's face for that sex), built on a
// thread of its own from the head's own morphs and kept while memory allows.
// Until it exists the head keeps vanilla lip-sync; a head whose skin lacks a
// morph the mapping needs keeps it for good.
#ifndef INCLUDE_A2F_ANIMATION_HEAD_CACHE_H_
#define INCLUDE_A2F_ANIMATION_HEAD_CACHE_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "a2f/animation/direct.h"
#include "a2f/plugin/plugin.h"
#include "a2f/runtime/engine.h"

namespace bfa {

// Reads the head parts' types, which tell a shape's part; once, at data
// loaded.
void LoadHeadPartTypes();

// The part a dynamic face shape follows: by its head part's type where that
// is unambiguous, else by its name. skin: it is the head skin itself (not
// brows / beard), whose lips the mapping aligns.
direct::Part Classify(const char* name, bool& skin);

// A name or Misc mouth-extra relationship suggests oral anatomy. This is
// only a hint; BuildHead checks placement and morph data before promotion.
bool MouthCandidate(const char* name);

// Hair-tinted geometry is excluded unless it is identifiable facial hair.
bool HairGeometry(uint8_t* shape, const char* name);

// Base (pre-morph) vertices of a face shape: its "FOD" extra data.
const float* BaseVerts(uint8_t* shape, uint32_t& nv);

// An unlisted follower shape (no base vertices: its head part is not among
// the NPC's, so the engine neither morphs nor rebuilds it) stands on the
// vertices it had before the plugin first wrote to it. UnlistedRest reads
// them (nv x 3; null when the shape's vertices cannot be read),
// NoteUnlistedWrite records a write to any face shape, and ResetUnlisted puts
// the node's unlisted shapes back as an engine rebuild does the others.
std::shared_ptr<const std::vector<float>> UnlistedRest(uint8_t* shape,
                                                       uint32_t nv);
void NoteUnlistedWrite(uint8_t* shape, const float* xyz, uint32_t nv);
void ResetUnlisted(uint8_t* node);

// The name ("" if it has none) of a child of the face node that carries base
// and morph vertices but is not a BSDynamicTriShape, or null. The engine's
// morph kernel takes the dynamic-data lock of such a shape without checking
// that it is dynamic: a null dereference as soon as one of its morphs has
// weight.
const char* StaticMorphPart(uint8_t* node);

// For its lifetime the node's static morph parts (StaticMorphPart) show no
// extra data, so the engine's ApplyMorphs passes them over (it looks their
// base and morph vertices up there) and morphs the rest of the head. The
// engine could not have morphed them anyway. complete(): every such part is
// hidden; otherwise the engine must not be called.
class ScopedStaticParts {
 public:
  explicit ScopedStaticParts(uint8_t* node);
  ~ScopedStaticParts();
  ScopedStaticParts(const ScopedStaticParts&) = delete;
  ScopedStaticParts& operator=(const ScopedStaticParts&) = delete;
  bool complete() const { return complete_; }

 private:
  static constexpr int kMax = 8;
  uint8_t* shapes_[kMax];
  uint16_t sizes_[kMax];
  int count_ = 0;
  bool complete_ = true;
};

// The node's head mapping and its state (failed: not mappable). On first
// sight reads the shapes and their morphs here, where the engine has the
// node, and queues the build.
// Recomputed before using a mapping, including for a previously failed head.
// female: the head's sex, which tells the face it is mapped onto when the
// model has one per sex.
std::string HeadKey(uint8_t* node, bool female,
                    std::shared_ptr<const direct::Model> mapping = {});
State FindHead(uint8_t* node, const std::string& key, bool female,
               std::shared_ptr<const direct::Head>& out,
               std::shared_ptr<const direct::Model> mapping = {});

#ifdef ENABLE_COMMONLIBSSE_TESTING
namespace testing {
inline decltype(&HeadKey) head_key;
inline decltype(&FindHead) find_head;
}  // namespace testing
#endif

// Forgets the mappings (not those still being built).
void ClearHeads();

// Implementation of the shape traversal shared by capture and playback.
// The node's dynamic face shapes the mapping covers.
template <typename Fn>
void ForFaceShapes(uint8_t* node, Fn fn) {
  const auto& children = reinterpret_cast<RE::NiNode*>(node)->GetChildren();
  const auto* kids = children.begin();
  uint16_t n = children.free_idx();
  for (uint16_t i = 0; kids && i < n; ++i) {
    auto* geometry = reinterpret_cast<RE::BSGeometry*>(kids[i].get());
    if (!geometry || geometry->GetType().underlying() != 4) continue;
    auto* dynamic = static_cast<RE::BSDynamicTriShape*>(geometry);
    auto* shape = reinterpret_cast<uint8_t*>(dynamic);
    const char* name = geometry->name.c_str();
    bool skin = false;
    direct::Part part = name ? Classify(name, skin) : direct::Part::kNone;
    if (part == direct::Part::kNone && (!name || !MouthCandidate(name)))
      continue;
    if (HairGeometry(shape, name)) continue;
    uint32_t nv = dynamic->GetTrishapeRuntimeData().vertexCount;
    if (dynamic->GetDynamicTrishapeRuntimeData().dataSize != nv * 16) continue;
    uint32_t nb = 0;
    const float* base = BaseVerts(shape, nb);
    std::shared_ptr<const std::vector<float>> rest;
    if (!base && !skin &&
        (part == direct::Part::kHead || part == direct::Part::kPinned)) {
      rest = UnlistedRest(shape, nv);
      if (!rest) continue;
      base = rest->data();
      nb = nv;
    }
    if (!base || nb != nv) continue;
    fn(shape, name, part, skin, nv, base);
  }
}

}  // namespace bfa

#endif  // INCLUDE_A2F_ANIMATION_HEAD_CACHE_H_
