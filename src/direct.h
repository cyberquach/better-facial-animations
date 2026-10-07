// A2F Direct: per-vertex deltas for the engine's dynamic face shapes. The
// skin basis lives on Mark, the face the network animates (A2FDirect.bin,
// tools/mark_export.py); with Audio2Face-3D v3.0 on Claire for women and on
// its Mark for men (A2FDirect3.bin, tools/mark_export3.py), and "Mark" below
// is whichever of them the head takes. Every head is mapped straight onto
// Mark from its own
// expression morphs (BuildHead); its brows, beard and eyes follow its skin,
// a piercing moves as one piece with the skin it is pinned to,
// and its lower teeth and gums turn with Mark's jaw. The tongue takes its
// softened authored tongue cues plus partial jaw follow. Shape-local
// space, the same as the .tri files.
#ifndef SRC_DIRECT_H_
#define SRC_DIRECT_H_

#include <cstdint>
#include <string>
#include <vector>

#include "live_core.h"

namespace direct {

// kPinned: rigid pieces on the skin (piercings), which are not to bend with it
enum class Part { kHead, kMouth, kEyes, kPinned, kNone };

// Deformation scales: the head skin by smooth per-vertex region weights
// (the kNumSkinRegions kBrows..kMouth), the mouth part by how far the jaw
// moves (lower teeth and gums) and how far the vanilla tongue moves.
enum Region { kBrows, kEyes, kCheeks, kMouth, kJaw, kTongue, kNumRegions };
constexpr int kNumSkinRegions = kJaw;

// The phonemes that tell a mouth part's tongue from its teeth: those that
// move the tongue against the jaw.
constexpr int kNumTonguePhonemes = 5;
constexpr int kNumPhonemes = 16;
// A line's coefficients per frame: the skin's k, then the network's tongue
// and jaw outputs less their rest.
constexpr uint32_t kNumMouthCoef = a2f::kNumTongue + a2f::kNumJaw;
constexpr int kNumJawPoints = a2f::kNumJaw / 3;
// The engine expression morphs the plugin plays itself, so that they can be
// scaled per region: 0..6 Dialogue*, 8..14 Mood*, 15 CombatAnger. 7
// MoodNeutral and 16 CombatShout are left to the engine.
constexpr int kNumExpressions = 16;
constexpr bool PlaysExpression(int i) { return i != 7; }
// Their scale per skin region: the mouth at half, where the expression
// competes with the lip-sync (the mouth part takes kMouth's).
constexpr float kExpressionScale[kNumSkinRegions] = {1, 1, 1, .5f};
constexpr int kNumLandmarks = 8;

// The engine morphs a head's shapes are read with (ShapeIn::morphs), by
// ApplyMorph (type, index): type 0 phoneme, 1 expression, 2 modifier. The
// skin reads the mapping's (kBlinkLeft..kOohQ and kHappy, which a head must
// have to be animated) and the expressions; the mouth part BigAah (how far
// each vertex follows the jaw), all speech phonemes and the expressions; the
// eyes the two blinks.
enum Morph {
  kBlinkLeft,
  kBlinkRight,
  kBrowUpLeft,
  kBrowUpRight,
  kBigAah,
  kOohQ,
  kTonguePhoneme,  // kNumTonguePhonemes
  kDst = kTonguePhoneme,
  kN,
  kK,
  kTh,
  kR,
  kExpression,                // kNumExpressions
  kHappy = kExpression + 10,  // MoodHappy
  kAah = kExpression + kNumExpressions,
  kBmp,
  kChJSh,
  kEee,
  kEh,
  kFv,
  kI,
  kOh,
  kW,
  kNumMorphs
};
// Engine phoneme order, sharing the mapping and classifier morphs above.
constexpr Morph kSpeechMorphs[kNumPhonemes] = {
    kAah, kBigAah, kBmp, kChJSh, kDst,  kEee, kEh, kFv,
    kI,   kK,      kN,   kOh,    kOohQ, kR,   kTh, kW};

struct MorphId {
  uint8_t type, index;
  const char* name;
};
extern const MorphId kMorphIds[kNumMorphs];

// Mark: neutral skin, jaw and tongue in Skyrim axes and Mark's units.
struct Mark {
  uint32_t nv = 0, k = 0;
  std::vector<float> verts;  // nv x 3
  std::vector<float> jaw;    // nv: z of jawOpen (how far it follows the jaw)
  float landmarks[kNumLandmarks * 3];
  std::vector<int32_t> row;  // nv: basis row, -1 for a vertex that is still
  std::vector<float> basis;  // rows x k x 3
  std::vector<float> proj;   // k x Model::n_skin
  std::vector<float> rest;   // Model::n_out: the network's output for silence
  // the jaw: its points at rest; they move by jaw_strength times the
  // network's jaw output less its rest, and the lower teeth with them rigidly
  float jaw_points[a2f::kNumJaw], jaw_strength = 0;
  // the tongue at rest (riding the jaw), its tip, and its vertices' motion
  // per tongue coefficient
  uint32_t tongue_nv = 0;
  std::vector<float> tongue_verts;  // tongue_nv x 3
  float tongue_tip[3];
  std::vector<float> tongue_basis;  // tongue_nv x kNumTongue x 3
  // v3.0: the tongue's coefficients from its latent, kNumTongue x kLatent3
  std::vector<float> tongue_proj;
};

// A2FDirect.bin (version 8: Mark, for the v2.3 network) or A2FDirect3.bin
// (version 9: Claire and Mark, for the folded v3.0 network).
struct Model {
  int network = 2;          // 2: v2.3, 3: v3.0
  std::vector<Mark> faces;  // v2.3: Mark; v3.0: female (Claire), male (Mark)
  // the network's output per frame and the skin's share of it
  uint32_t n_out = a2f::kNumOut, n_skin = a2f::kNumPca;
  // The face a head of that sex is mapped onto.
  int FaceIndex(bool female) const { return faces.size() > 1 && !female; }
  const Mark& Face(bool female) const { return faces[FaceIndex(female)]; }
  bool Load(const std::wstring& path, std::string& err);
};

// Whether the shape reads morph m (see Morph).
constexpr bool Reads(bool skin, Part part, int m, bool mouth_candidate = false) {
  if (mouth_candidate && (part == Part::kPinned || part == Part::kNone))
    part = Part::kMouth;
  if (skin) return m < kTonguePhoneme || (m >= kExpression && m < kAah);
  if (part == Part::kMouth) return m >= kBigAah;
  return part == Part::kEyes && m <= kBlinkRight;
}

// One dynamic face shape as the engine has it.
struct ShapeIn {
  std::string name;
  Part part = Part::kNone;
  bool skin = false;  // the head skin itself
  bool mouth_candidate = false;  // ambiguous name/relationship; validate spatially
  uint32_t nv = 0;
  std::vector<float> verts;  // nv x 3, base (pre-morph)
  // by Morph, nv x 3 each: those it Reads (zero when the shape lacks it),
  // else empty
  std::vector<std::vector<float>> morphs;
};

// One head mapped onto Mark.
struct Head {
  uint32_t k = 0, nv = 0;       // skin
  std::vector<int32_t> row;     // nv: basis row, -1 where the speech is still
  std::vector<float> basis;     // rows x k x 3 (Mark's, through the mapping)
  std::vector<float> region_w;  // nv x kNumSkinRegions
  // Upper/lower lip influence per skin vertex; fades out at the lip box edge.
  std::vector<float> lip_w;  // nv x 2
  // Mark's jaw points at rest on this head, and the matrix that takes a
  // point's network output (less rest) to its displacement here
  float jaw_points[a2f::kNumJaw] = {}, jaw_motion[9] = {};
  struct Shape {
    std::string name;
    Part part;
    uint32_t nv;
    // brows, beards, eyes: delta = sum_j w[j] delta_skin[idx[j]], kFollow
    // per vertex (pinned: the same for all the vertices of a piece)
    std::vector<uint32_t> idx;
    std::vector<float> w;
    // Mouth part: positions (nv x 3), jaw weights (nv), tongue mask (nv),
    // and masked authored phoneme deltas (nv x kNumPhonemes x 3).
    // The tongue combines softened vanilla cues with partial jaw follow.
    std::vector<float> pos, jaw, tongue_weight, tongue;
    // Pre-feature jaw weights and network tongue basis for the opt-out.
    std::vector<float> model_jaw, model_tongue;
    // skin and mouth part: the expression morphs, nv x 3 (empty: none)
    std::vector<float> expr[kNumExpressions];
  };
  std::vector<Shape> shapes;
  uint32_t skin = 0;  // the skin's index in shapes
  // How far this head's own morphs move next to a standard head's (their
  // peak over Mark's scale, over kAutoRef), kAutoMin..1: a head made for
  // smaller motion takes the speech scaled down by it. Brows from BrowUp,
  // cheeks from MoodHappy, mouth and jaw from BigAah; eyes and tongue 1.
  float auto_scale[kNumRegions] = {1, 1, 1, 1, 1, 1};
  // log: Mark's scale, landmark residuals after the similarity, skin
  // vertices on the upper and lower lip, mean distance of the lips to Mark's,
  // mouth part vertices that follow the jaw and that are tongue
  float scale = 0, residual[kNumLandmarks] = {}, lip_dist = 0;
  uint32_t lips[2] = {}, jaw_verts = 0, tongue_verts = 0;
};
constexpr int kFollow = 4;
// The standard heads' morph peaks over Mark's scale (the least of vanilla,
// Expressive Facial Animation and High Poly Head, female and male).
constexpr float kAutoBrowUp = 0.53f, kAutoHappy = 1.02f, kAutoBigAah = 2.5f;
constexpr float kAutoMin = 0.4f;

// Maps one head (all its dynamic face shapes) onto Mark. False, with why,
// when it has no skin shape or the skin lacks a morph of kMorphIds: such a
// head is not animated.
bool BuildHead(const Mark& mark, const std::vector<ShapeIn>& shapes,
               Head& out, std::string& why);

enum class TongueMode { kAuthored, kModel };

// The lips' speech motion at the sliders' neutral 1: fixed calibration the
// user multipliers apply on top of (LoadIni converts older multipliers).
constexpr float kUpperLipCalibration = 0.8f;
constexpr float kLowerLipCalibration = 1.2f;

// Per-shape float3 deltas from final speech coefficients (head.k +
// kNumMouthCoef; null for no speech), scaled per region, plus expression
// levels (kNumExpressions). Followers inherit the skin's deltas. False when
// neither speech nor expressions contribute. Optional vanilla phoneme weights
// (kNumPhonemes) supply softened tongue cues over half-strength jaw follow.
// Without weights the tongue only follows the jaw. TongueMode::kModel
// restores the pre-feature mapping and network tongue scale. Lip multipliers
// (times k{Upper,Lower}LipCalibration) affect speech skin motion only;
// followers inherit it, jaw/tongue and engine expressions retain their
// existing controls.
bool Evaluate(const Head& head, const float* coefficients, const float* scales,
              const float* expressions, const float* phonemes,
              TongueMode tongue_mode, std::vector<std::vector<float>>& out,
              float upper_lip = 1, float lower_lip = 1);

// The line's coefficients (head.k + kNumMouthCoef) at a fractional frame (of
// the line's own rate), as
// Evaluate takes them (past the end: the last frame's); zeros and false
// without a line.
bool Coefficients(const Head& head, const a2f::Line* line, float frame,
                  std::vector<float>& c);

// Shift that takes a mesh stored in skeleton space (custom heads such as
// High Poly Head: no shape transform, vertices ~120 units up) into
// shape-local space: the vanilla head transform, or zero for one already
// there.
void FrameOffset(const float* verts, uint32_t nv, float offset[3]);

}  // namespace direct

#endif  // SRC_DIRECT_H_
