// One mapping per distinct head (every face shape's name, vertex count and
// base vertices) onto Mark, built on a thread of its own from the head's own
// morphs and kept while memory allows. Until it exists the head keeps vanilla
// lip-sync; a head whose skin lacks a morph the mapping needs keeps it for
// good.
#ifndef SRC_HEAD_CACHE_H_
#define SRC_HEAD_CACHE_H_

#include <cstdint>
#include <memory>
#include <string>

#include "direct.h"
#include "engine.h"
#include "plugin.h"

namespace bfa {

// Reads the head parts' types, which tell a shape's part; once, at data
// loaded.
void LoadHeadPartTypes();

// The part a dynamic face shape follows: by its head part's type where that
// is unambiguous, else by its name. skin: it is the head skin itself (not
// brows / beard), whose lips the mapping aligns.
direct::Part Classify(const char* name, bool& skin);

// Base (pre-morph) vertices of a face shape: its "FOD" extra data.
const float* BaseVerts(uint8_t* shape, uint32_t& nv);

// The node's head mapping and its state (failed: not mappable). On first
// sight reads the shapes and their morphs here, where the engine has the
// node, and queues the build.
// Recomputed before using a mapping, including for a previously failed head.
std::string HeadKey(uint8_t* node);
State FindHead(uint8_t* node, const std::string& key,
               std::shared_ptr<const direct::Head>& out);

// Forgets the mappings (not those still being built).
void ClearHeads();

// Implementation of the shape traversal shared by capture and playback.
// The node's dynamic face shapes the mapping covers.
template <typename Fn>
void ForFaceShapes(uint8_t* node, Fn fn) {
  uint8_t** kids = Field<uint8_t**>(node, kNodeChildren.Get());
  uint16_t n = Field<uint16_t>(node, kNodeNumChildren.Get());
  for (uint16_t i = 0; kids && i < n; ++i) {
    uint8_t* shape = kids[i];
    if (!shape || shape[kGeomType.Get()] != 4) continue;
    const char* name = Field<const char*>(shape, kObjName);
    bool skin = false;
    direct::Part part = name ? Classify(name, skin) : direct::Part::kNone;
    if (part == direct::Part::kNone) continue;
    uint32_t nv = Field<uint16_t>(shape, kDynNumVerts.Get());
    if (Field<uint32_t>(shape, kDynDataSize.Get()) != nv * 16) continue;
    uint32_t nb = 0;
    const float* base = BaseVerts(shape, nb);
    if (!base || nb != nv) continue;
    fn(shape, name, part, skin, nv, base);
  }
}

}  // namespace bfa

#endif  // SRC_HEAD_CACHE_H_
