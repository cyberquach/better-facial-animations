// A2F Direct: per-vertex deltas for the engine's dynamic head / mouth shapes,
// evaluated on reference meshes (DIRECT_SPEC.md, tools/direct_export.py) and
// resampled onto any topology (High Poly Head). Shape-local space, the same
// as the .tri files.
#ifndef SRC_DIRECT_H_
#define SRC_DIRECT_H_

#include <cstdint>
#include <string>
#include <vector>

#include "live_core.h"

namespace direct {

enum class Part { kHead, kMouth, kEyes, kNone };
constexpr int kNumParts = 3;

// Deformation scales: the head skin by smooth per-vertex region weights
// (kBrows..kMouth), the jaw motion and the tongue by one factor each.
enum Region { kBrows, kEyes, kCheeks, kMouth, kJaw, kTongue, kNumRegions };

// Reference mesh of one part with region labels: head 0 skin, 1 upper lip,
// 2 lower lip; mouth 0 gums, 1 upper teeth, 2 lower teeth, 3 tongue.
struct RefMesh {
  uint32_t nv = 0;
  std::vector<float> verts;     // nv x 3
  std::vector<uint32_t> tris;   // nt x 3
  std::vector<uint8_t> labels;  // nv
};

struct SexModel {
  RefMesh head, mouth, eyes;  // eyes: no labels
  // skin: delta_head[midx[i]] = g sum_k coef[k] basis[k][i]
  uint32_t k = 0, nm = 0;
  std::vector<uint16_t> midx;
  std::vector<float> basis;     // k x nm x 3
  std::vector<float> proj;      // k x kNumPca
  std::vector<float> region_w;  // nm x 4: kBrows..kMouth weights
  // tongue: delta_mouth[tongue_idx[i]] = g sum_j c[j] tongue_basis[j][i]
  std::vector<uint16_t> tongue_idx;
  std::vector<float> tongue_basis;  // kNumTongue x n x 3
  // mouth (teeth, gums, tongue base): the race's own phoneme morphs with
  // w = argmin |A w - delta_head[rows]|, 0 <= w <= 1, fitted to the unscaled
  // skin; delta_mouth = scale_jaw sum_p w[p] rig_mouth[p]
  uint32_t rig_p = 0;
  std::vector<float> rig_ata;    // p x p: A^T A
  std::vector<float> rig_atb;    // p x k: A^T basis on the rows
  std::vector<float> rig_mouth;  // p x mouth nv x 3
  // eyes (lashes / lid line) follow the scaled head delta:
  // delta_eyes[v] = sum_j eye_w[v][j] delta_head[eye_idx[v][j]]
  uint32_t eye_k = 0;
  std::vector<uint16_t> eye_idx;  // eyes nv x eye_k
  std::vector<float> eye_w;
};

// A2FDirect.bin.
struct Model {
  SexModel sex[2];          // female, male
  std::vector<float> rest;  // kNumRaw
  float default_scale[kNumRegions];
  bool Load(const std::wstring& path, std::string& err);
  a2f::DirectProj Proj(bool female) const;
};

// One line's coefficients (a2f::Line::direct) and the model they index.
struct Track {
  const float* coeffs;
  uint32_t frames, dim;
  const SexModel* model;
};

// Deltas (float3 per reference vertex) of every part at a fractional frame,
// interpolated and eased out past the end like the morph curves, scaled per
// region (scales[kNumRegions]). The eyes follow the scaled head delta; the
// mouth rig is fitted to the unscaled one and scaled by kJaw.
struct Deltas {
  std::vector<float> part[kNumParts];  // by Part
};
void Evaluate(const Track& track, float frame, const float* scales,
              Deltas& out);

// Lip labels (0 skin, 1 upper lip, 2 lower lip) of a head mesh from its own
// BigAah morph deltas, the rule the reference labels were made with.
void LipLabels(const float* verts, const float* bigaah, uint32_t nv,
               std::vector<uint8_t>& labels);

// Target vertex -> reference triangle corners + barycentric weights, or the
// identity for the reference topology itself. A target vertex with a label
// only maps to triangles of its own region (nearest such vertex if none is
// nearby); target_labels may be null.
struct Mapping {
  bool identity = false;
  uint32_t nv = 0;
  std::vector<uint32_t> idx;  // 3 per vertex
  std::vector<float> w;       // 3 per vertex
  float max_dist = 0, mean_dist = 0;
  uint32_t snapped = 0;  // vertices with no same-region triangle nearby
};
void BuildMapping(const RefMesh& ref, const float* target, uint32_t nv,
                  const uint8_t* target_labels, Mapping& out);

// Adds the mapped deltas to xyz of float4 vertices.
void AddDeltas(const Mapping& m, const std::vector<float>& ref_deltas,
               float* xyzw);

}  // namespace direct

#endif  // SRC_DIRECT_H_
