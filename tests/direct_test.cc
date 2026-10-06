// Checks direct::FrameOffset on parts placed like real FaceGen meshes: a part
// other than the head skin is centred off the head, and that must not move it;
// and the mouth part on Mark's jaw (a made-up Mark and head): the upper teeth
// stay, the lower teeth move with the jaw's points as one piece.
// ./build.sh test builds and runs it.
#include "../src/direct.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace direct {
namespace {

int failures;

// A box of 8 corners around centre with half extents h.
std::vector<float> Box(float cx, float cy, float cz, float hx, float hy,
                       float hz) {
  std::vector<float> v;
  for (int i = 0; i < 8; ++i) {
    v.push_back(cx + (i & 1 ? hx : -hx));
    v.push_back(cy + (i & 2 ? hy : -hy));
    v.push_back(cz + (i & 4 ? hz : -hz));
  }
  return v;
}

void Expect(const char* what, const std::vector<float>& v, float y, float z) {
  float off[3];
  FrameOffset(v.data(), static_cast<uint32_t>(v.size() / 3), off);
  bool ok = std::fabs(off[0]) < 1e-4f && std::fabs(off[1] - y) < 1e-4f &&
            std::fabs(off[2] - z) < 1e-4f;
  if (!ok) ++failures;
  printf("%s %s: offset %.2f %.2f %.2f, want 0 %.2f %.2f\n", ok ? "ok" : "FAIL",
         what, off[0], off[1], off[2], y, z);
}


constexpr float kScales[kNumRegions] = {1, 1, 1, 1, 1, 1};
constexpr float kNoExpressions[kNumExpressions] = {};
enum class TongueMotion { kNone, kWithJaw, kStill, kPartialJaw };
struct MouthFixture {
  Head head;
  a2f::Line line;
};

void Check(bool ok, const char* label) {
  printf("%s %s\n", ok ? "ok" : "FAIL", label);
  if (!ok) ++failures;
}

// Mark himself, with upper/lower teeth, optional tongue and two pinned rings.
bool BuildFixture(TongueMotion mode, MouthFixture& fixture) {
  Mark mk;
  // the landmarks' places, and one skin vertex at each (plus filler)
  const float lm[kNumLandmarks][3] = {
      {-3, 8, 4}, {3, 8, 4},  {-3, 8, 6},  {3, 8, 6},
      {0, 8, -5}, {0, 9, -2}, {-2, 8, -1}, {2, 8, -1}};
  for (int i = 0; i < kNumLandmarks * 3; ++i)
    mk.landmarks[i] = lm[i / 3][i % 3];
  std::vector<float> v(&lm[0][0], &lm[0][0] + kNumLandmarks * 3);
  for (int i = 0; i < 64; ++i)
    for (float c : {(i % 4) * 2.f - 3, 6.f + i % 3, (i / 4) - 8.f})
      v.push_back(c);
  mk.nv = static_cast<uint32_t>(v.size() / 3);
  mk.k = 1;
  mk.verts = v;
  mk.jaw.assign(mk.nv, 0.f);
  mk.jaw[4] = -1;
  mk.row.assign(mk.nv, -1);
  mk.proj.assign(a2f::kNumPca, 0.f);
  const float jaw[a2f::kNumJaw] = {
      -2, 2, -3, 2, 2, -3, -1, 7, -3, 1, 7, -3, 0, 8, -4};
  std::copy(jaw, jaw + a2f::kNumJaw, mk.jaw_points);
  mk.jaw_strength = 2;
  mk.rest.assign(a2f::kNumOut, 0.f);
  ShapeIn skin;
  skin.part = Part::kHead;
  skin.skin = true;
  skin.nv = mk.nv;
  skin.verts = v;
  skin.morphs.assign(kNumMorphs, std::vector<float>(v.size(), 0.f));
  const int at[kNumLandmarks] = {
      kBlinkLeft, kBlinkRight, kBrowUpLeft, kBrowUpRight,
      kBigAah,    kOohQ,       kHappy,      kHappy};
  for (int i = 0; i < kNumLandmarks; ++i) skin.morphs[at[i]][i * 3 + 2] = -1;
  ShapeIn mouth;
  mouth.part = Part::kMouth;
  mouth.nv = 16;  // 8 upper teeth (z = -2), 8 lower (z = -3)
  for (int i = 0; i < 16; ++i)
    for (float c : {(i % 4) - 1.5f, 6.f + (i / 4) % 2, i < 8 ? -2.f : -3.f})
      mouth.verts.push_back(c);
  mouth.morphs.assign(kNumMorphs, {});
  mouth.morphs[kBigAah].assign(48, 0.f);
  for (int i = 8; i < 16; ++i)  // about the hinge y = 0: the drop grows with y
    mouth.morphs[kBigAah][i * 3 + 2] = -0.1f * mouth.verts[i * 3 + 1];
  // Full and absent BigAah motion exercise authored tongue detection.
  // Partial motion keeps the tongue outside the legacy teeth sample.
  if (mode != TongueMotion::kNone) {
    mk.tongue_nv = 4;
    mk.tongue_verts = {-.5f, 7, -3, .5f, 7, -3,
                      -.5f, 8, -3, .5f, 8, -3};
    mk.tongue_tip[0] = 0;
    mk.tongue_tip[1] = 8;
    mk.tongue_tip[2] = -3;
    mk.tongue_basis.assign(4 * a2f::kNumTongue * 3, 0.f);
    for (int i = 0; i < 4; ++i)
      mk.tongue_basis[i * a2f::kNumTongue * 3 + 2] = 1;
    mouth.nv += 4;
    mouth.verts.insert(mouth.verts.end(), mk.tongue_verts.begin(),
                       mk.tongue_verts.end());
    mouth.morphs[kBigAah].resize(mouth.nv * 3, 0.f);
    for (int i = 16; i < 20; ++i) {
      if (mode == TongueMotion::kWithJaw)
        mouth.morphs[kBigAah][i * 3 + 2] = -0.1f * mouth.verts[i * 3 + 1];
      if (mode == TongueMotion::kPartialJaw)
        mouth.morphs[kBigAah][i * 3 + 2] = -0.025f * mouth.verts[i * 3 + 1];
    }
    // One shape per engine phoneme, shared by classification and animation.
    // Teeth follow the jaw in each shape; the tongue has independent motion.
    for (int ph = 0; ph < kNumPhonemes; ++ph) {
      const int morph = kSpeechMorphs[ph];
      if (morph == kBigAah) continue;
      auto& delta = mouth.morphs[morph];
      delta.assign(mouth.nv * 3, 0.f);
      for (int i = 0; i < 20; ++i) {
        const float motion = 0.1f * (ph + 1);
        delta[i * 3 + 2] = motion * mouth.morphs[kBigAah][i * 3 + 2];
        if (i >= 16) delta[i * 3] = motion;
      }
    }
  }
  // two rings in one shape, each across the lip line (z = -2)
  ShapeIn rings;
  rings.part = Part::kPinned;
  rings.nv = 24;
  for (int i = 0; i < 24; ++i) {
    const float a = (i % 12) * 0.5236f;
    for (float c : {i < 12 ? -1.f : 1.f, 8.3f + 0.4f * std::cos(a),
                    -2.2f + 0.4f * std::sin(a)})
      rings.verts.push_back(c);
  }
  rings.morphs.assign(kNumMorphs, {});
  Head& h = fixture.head;
  std::string why;
  const bool built = BuildHead(mk, {skin, mouth, rings}, h, why);
  const bool ok = built && h.shapes.size() == 3 &&
                  h.jaw_verts == ((mode == TongueMotion::kWithJaw ||
                                   mode == TongueMotion::kPartialJaw)
                                      ? 12u
                                      : 8u) &&
                  h.tongue_verts == (mode == TongueMotion::kNone ? 0u : 4u);
  Check(ok, "mouth fixture mapping");
  if (!ok) printf("  jaw %u tongue %u\n", h.jaw_verts, h.tongue_verts);
  if (!why.empty()) printf("  %s\n", why.c_str());
  // The model jaw points all move down 0.25, scaled to 0.5 on this head.
  auto& line = fixture.line;
  line.frames = 1;
  line.direct_dim = mk.k + kNumMouthCoef;
  line.direct.assign(line.direct_dim, 0.f);
  for (int i = 0; i < kNumJawPoints; ++i)
    line.direct[mk.k + a2f::kNumTongue + i * 3 + 1] = -0.25f;
  return ok;
}

void PinnedRings(const Head& h) {
  bool ok = h.shapes[2].idx.size() == 24u * kFollow;
  float weight = 0;
  for (int i = 0; ok && i < 24 * kFollow; ++i) {
    const int first =
        i < 12 * kFollow ? i % kFollow : 12 * kFollow + i % kFollow;
    ok = h.shapes[2].idx[i] == h.shapes[2].idx[first] &&
         h.shapes[2].w[i] == h.shapes[2].w[first];
    if (i < kFollow) weight += h.shapes[2].w[i];
  }
  Check(ok && std::fabs(weight - 1) < 1e-4f &&
            h.shapes[2].idx[0] != h.shapes[2].idx[12 * kFollow],
        "pinned rings follow separate skin patches");
}

void JawAndCarriedPose(const MouthFixture& fixture) {
  const Head& h = fixture.head;
  std::vector<float> pose;
  std::vector<std::vector<float>> first, carried;
  bool ok = Coefficients(h, &fixture.line, 0, pose) &&
            Evaluate(h, pose.data(), kScales, kNoExpressions, nullptr,
                     TongueMode::kAuthored, first);
  for (int i = 0; ok && i < 16; ++i) {
    for (int axis = 0; axis < 3; ++axis) {
      const float want = i >= 8 && axis == 2 ? -0.5f : 0;
      ok = ok && std::fabs(first[1][i * 3 + axis] - want) < 1e-3f;
    }
  }
  Check(ok, "upper teeth stay; lower teeth follow jaw");
  ok = ok && Evaluate(h, pose.data(), kScales, kNoExpressions, nullptr,
                      TongueMode::kAuthored, carried);
  Check(ok && first == carried &&
            !Evaluate(h, nullptr, kScales, kNoExpressions, nullptr,
                      TongueMode::kAuthored, carried),
        "carried coefficients preserve pose; no speech clears it");
}

float AuthoredTongueDelta(TongueMotion mode, int ph, int vertex, int axis) {
  if (axis == 0 && ph != 1) return 0.1f * (ph + 1);
  if (axis != 2 || mode == TongueMotion::kStill) return 0;
  const float jaw = vertex < 18 ? -0.7f : -0.8f;
  return ph == 1 ? jaw : jaw * 0.1f * (ph + 1);
}

void TongueCues(const MouthFixture& fixture, TongueMotion mode) {
  const Head& h = fixture.head;
  std::vector<float> pose = fixture.line.direct;
  pose[h.k] = 100;  // The model tongue must not drive authored articulation.
  const float muted[kNumRegions] = {1, 1, 1, 1, 1, 0};
  for (int ph = 0; ph < kNumPhonemes; ++ph) {
    float weights[kNumPhonemes] = {};
    weights[ph] = 0.5f;
    std::vector<std::vector<float>> bent, neutral;
    bool ok = Evaluate(h, pose.data(), kScales, kNoExpressions, weights,
                       TongueMode::kAuthored, bent) &&
              Evaluate(h, pose.data(), muted, kNoExpressions, weights,
                       TongueMode::kAuthored, neutral);
    for (int i = 0; ok && i < 20; ++i) {
      for (int axis = 0; axis < 3; ++axis) {
        float want = 0, quiet = 0;
        if (i >= 16) {
          quiet = axis == 2 ? -0.25f : 0;
          want = quiet + 1.5f * 0.35f * 0.5f *
                             AuthoredTongueDelta(mode, ph, i, axis);
        } else if (axis == 2 && i >= 8) {
          want = quiet = -0.5f;
        }
        ok = ok && std::fabs(bent[1][i * 3 + axis] - want) < 1e-3f &&
             std::fabs(neutral[1][i * 3 + axis] - quiet) < 1e-3f;
      }
    }
    printf("phoneme %d: ", ph);
    Check(ok, "authored articulation and muted jaw follow preserve teeth");
  }
}

void OverlapAndToggle(const MouthFixture& fixture, TongueMotion mode) {
  const Head& h = fixture.head;
  std::vector<float> pose = fixture.line.direct;
  pose[h.k] = 100;
  float overlap[kNumPhonemes] = {};
  overlap[0] = overlap[14] = 1;
  std::vector<std::vector<float>> bent, old_pose, resumed;
  bool ok = Evaluate(h, pose.data(), kScales, kNoExpressions, overlap,
                     TongueMode::kAuthored, bent);
  for (int i = 16; ok && i < 20; ++i) {
    for (int axis = 0; axis < 3; ++axis) {
      const float rest = axis == 2 ? -0.25f : 0;
      const float average = (AuthoredTongueDelta(mode, 0, i, axis) +
                             AuthoredTongueDelta(mode, 14, i, axis)) * 0.5f;
      ok = ok && std::fabs(bent[1][i * 3 + axis] -
                           (rest + 1.5f * 0.35f * average)) < 1e-3f;
    }
  }
  Check(ok, "overlapping tongue cues average instead of stacking");
  // The legacy classifier intentionally excludes both synthetic tongue cases.
  ok = ok &&
       Evaluate(h, pose.data(), kScales, kNoExpressions, overlap,
                TongueMode::kModel, old_pose) &&
       Evaluate(h, pose.data(), kScales, kNoExpressions, overlap,
                TongueMode::kAuthored, resumed);
  for (int i = 16; ok && i < 20; ++i) {
    const float old_z = mode == TongueMotion::kWithJaw ? -0.5f : 0;
    ok = std::fabs(old_pose[1][i * 3 + 2] - old_z) < 1e-3f;
  }
  Check(ok && bent == resumed, "legacy toggle restores and resumes tongue");
}

// Exercise actual legacy classification and model deformation, not only jaw
// follow on vertices that the legacy classifier deliberately excludes.
void LegacyTongue() {
  MouthFixture fixture;
  if (!BuildFixture(TongueMotion::kPartialJaw, fixture)) return;
  const Head& h = fixture.head;
  const auto& mouth = h.shapes[1];
  Check(!mouth.model_tongue.empty(), "legacy classifier finds tongue basis");
  std::vector<float> pose = fixture.line.direct;
  std::vector<std::vector<float>> neutral, bent, muted, authored, restored;
  float scales[kNumRegions] = {1, 1, 1, 1, 1, 0};
  float cues[kNumPhonemes] = {};
  cues[14] = 1;
  bool ok = Evaluate(h, pose.data(), kScales, kNoExpressions, cues,
                     TongueMode::kModel, neutral);
  pose[h.k] = 0.5f;
  ok = Evaluate(h, pose.data(), kScales, kNoExpressions, cues,
                TongueMode::kModel, bent) &&
       ok;
  ok = Evaluate(h, pose.data(), scales, kNoExpressions, cues,
                TongueMode::kModel, muted) &&
       ok;
  ok = Evaluate(h, pose.data(), kScales, kNoExpressions, cues,
                TongueMode::kAuthored, authored) &&
       ok;
  ok = Evaluate(h, pose.data(), kScales, kNoExpressions, cues,
                TongueMode::kModel, restored) &&
       ok;
  for (int i = 0; ok && i < 20; ++i) {
    float motion_squared = 0;
    for (int axis = 0; axis < 3; ++axis) {
      const int index = i * 3 + axis;
      const float delta = bent[1][index] - neutral[1][index];
      motion_squared += delta * delta;
      ok = ok && std::fabs(muted[1][index] - neutral[1][index]) < 1e-5f;
    }
    // Every tongue vertex articulates; neither set of teeth moves with it.
    ok = ok && (i >= 16 ? motion_squared > 0.01f : motion_squared < 1e-10f);
  }
  Check(ok, "legacy tongue articulates and mutes without moving teeth");
  Check(ok && bent == restored && bent != authored,
        "legacy tongue pose returns after switching modes");
}

void MouthScaling(Head h) {
  h.row[0] = 0;
  h.basis.assign(3, 0.f);
  h.basis[2] = 1;
  h.region_w.assign(h.nv * kNumSkinRegions, 0.f);
  h.region_w[kMouth] = 1;
  std::vector<float> pose(h.k + kNumMouthCoef, 0.f);
  pose[0] = 1;
  float scales[kNumRegions] = {1, 1, 1, 1, 1, 1};
  std::vector<std::vector<float>> deltas;
  bool ok = true;
  for (float value : {0.f, 0.5f, 1.f, 2.f}) {
    scales[kMouth] = value;
    ok = Evaluate(h, pose.data(), scales, kNoExpressions, nullptr,
                  TongueMode::kAuthored, deltas) &&
         ok;
    ok = ok && std::fabs(deltas[h.skin][2] - value) < 1e-5f;
  }
  h.region_w[kMouth] = 0;
  h.region_w[kBrows] = 1;
  ok = ok &&
       Evaluate(h, pose.data(), scales, kNoExpressions, nullptr,
                TongueMode::kAuthored, deltas) &&
       std::fabs(deltas[h.skin][2] - 1) < 1e-5f;
  Check(ok, "mouth scale is applied once; brows unchanged");
}

void MorphSelection() {
  // Runtime capture must visit each (type, index) once per shape.
  for (Part part : {Part::kHead, Part::kMouth, Part::kEyes, Part::kPinned}) {
    bool seen[3][16] = {};
    int count = 0;
    bool ok = true;
    for (int m = 0; m < kNumMorphs; ++m) {
      if (!Reads(part == Part::kHead, part, m)) continue;
      const auto& id = kMorphIds[m];
      ok = ok && !seen[id.type][id.index];
      seen[id.type][id.index] = true;
      ++count;
    }
    int expected = 0;
    if (part == Part::kHead) expected = 22;
    if (part == Part::kMouth) expected = 32;
    if (part == Part::kEyes) expected = 2;
    Check(ok && count == expected, "unique morph capture for shape role");
  }
  bool ok = true;
  for (int ph = 0; ph < kNumPhonemes; ++ph) {
    const auto& id = kMorphIds[kSpeechMorphs[ph]];
    ok = ok && id.type == 0 && id.index == ph;
  }
  Check(ok, "speech channels select the matching engine morph");
}

int RunTests() {
  // shape-local, the head and parts around it (a long beard hangs ~8 below
  // and ~7 in front of the head's centre)
  Expect("local head", Box(0, 0, 0, 6, 8, 11), 0, 0);
  Expect("local brows", Box(0, 7, 4, 4, 1, 1), 0, 0);
  Expect("local long beard", Box(0, 6.6f, -7.8f, 3, 2, 5), 0, 0);
  // skeleton space (High Poly Head replacers): the vanilla head transform
  // (0, -1.55, 120.34) back, whatever the part's own box
  const float sy = -1.55f, sz = 120.34f;
  Expect("skeleton head", Box(0, sy, sz, 6, 8, 11), -sy, -sz);
  Expect("skeleton beard", Box(0, sy + 2.8f, sz - 1.8f, 4, 3, 4), -sy, -sz);
  Expect("skeleton brows", Box(0, sy + 5.3f, sz + 4, 4, 1, 1), -sy, -sz);
  Expect("skeleton eyes", Box(0, sy + 5.3f, sz + 3, 3, 1, 1), -sy, -sz);
  MorphSelection();
  LegacyTongue();
  for (TongueMotion mode : {TongueMotion::kNone, TongueMotion::kWithJaw,
                           TongueMotion::kStill}) {
    MouthFixture fixture;
    if (!BuildFixture(mode, fixture)) continue;
    PinnedRings(fixture.head);
    JawAndCarriedPose(fixture);
    if (mode == TongueMotion::kNone) {
      MouthScaling(fixture.head);
    } else {
      TongueCues(fixture, mode);
      OverlapAndToggle(fixture, mode);
    }
  }
  printf("%s: %d failures\n", failures ? "FAIL" : "ok", failures);
  return failures ? 1 : 0;
}

}  // namespace
}  // namespace direct

int main() { return direct::RunTests(); }
