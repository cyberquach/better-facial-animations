// Checks direct::FrameOffset on parts placed like real FaceGen meshes: a part
// other than the head skin is centred off the head, and that must not move it;
// and the mouth part on Mark's jaw (a made-up Mark and head): the upper teeth
// stay, the lower teeth move with the jaw's points as one piece.
// ./build.sh test builds and runs it.
#include "a2f/animation/direct.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
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

void CheckLipJawMotion(const Head& h, const ShapeIn& skin,
                       uint32_t first_skin) {
  std::vector<float> c(h.k + kNumMouthCoef, 0.f);
  // A shut jaw still permits upward lip closure, but no downward gap.
  c[0] = 1;
  for (float direction : {-1.f, 1.f}) {
    Head vertical = h;
    vertical.basis = {0.f, 0.f, direction};
    std::vector<std::vector<float>> result;
    Evaluate(vertical, c.data(), kScales, kNoExpressions, nullptr,
             TongueMode::kAuthored, result, 1, 1, 0, 0, true);
    bool ok = true;
    for (uint32_t i = 4; i < 8; ++i) {
      const float z = result[0][(first_skin + i) * 3 + 2];
      ok &= direction < 0 ? std::fabs(z) < 1e-5f : z > 0;
    }
    Check(ok, "closed jaw limits downward motion and preserves upward closure");
  }
  // A rotating jaw has different vertical travel at each lip point.
  Head rotated = h;
  rotated.basis = {0.f, 0.f, -10.f};
  std::fill_n(rotated.jaw_motion, 9, 0.f);
  rotated.jaw_motion[0] = rotated.jaw_motion[4] = rotated.jaw_motion[8] = 1.f;
  const float angle = -0.15f, cs = std::cos(angle), sn = std::sin(angle);
  for (int i = 0; i < kNumJawPoints; ++i) {
    const float y = h.jaw_points[i * 3 + 1], z = h.jaw_points[i * 3 + 2];
    c[h.k + a2f::kNumTongue + i * 3 + 1] = cs * y - sn * z - y;
    c[h.k + a2f::kNumTongue + i * 3 + 2] = sn * y + cs * z - z;
  }
  std::vector<std::vector<float>> rotation;
  Evaluate(rotated, c.data(), kScales, kNoExpressions, nullptr,
           TongueMode::kAuthored, rotation, 1, 1, 0, 0, true);
  bool follows_rotation = true;
  for (uint32_t i = 4; i < 8; ++i) {
    const uint32_t v = first_skin + i;
    const float y = skin.verts[v * 3 + 1], z = skin.verts[v * 3 + 2];
    follows_rotation &=
        std::fabs(rotation[0][v * 3 + 2] - (sn * y + cs * z - z)) < 1e-4f;
  }
  Check(follows_rotation, "lower lip limit follows local jaw rotation");
}

// Use the same cached head with the limit enabled and disabled. A caller's
// race choice must not change the shared mapping or other mouth behavior.
void CheckLipJawFade(const Head& head, uint32_t partial, uint32_t outside) {
  Head h = head;
  h.basis = {0.3f, 0.2f, -1.f};
  std::vector<float> c(h.k + kNumMouthCoef, 0.f);
  c[0] = 1;
  const float weight = h.lip_w[partial * 2 + 1];
  Check(weight > 0.f && weight < 1.f && h.lip_w[outside * 2 + 1] == 0.f,
        "fixture has partial and zero-weight neighboring lower lip vertices");
  Head legacy = h;
  legacy.lip_jaw.clear();
  for (auto mode : {TongueMode::kAuthored, TongueMode::kModel}) {
    std::vector<std::vector<float>> enabled, disabled, original, again;
    Evaluate(h, c.data(), kScales, kNoExpressions, nullptr, mode, enabled, 1, 1,
             0, 0, true);
    Evaluate(h, c.data(), kScales, kNoExpressions, nullptr, mode, disabled);
    Evaluate(legacy, c.data(), kScales, kNoExpressions, nullptr, mode, original,
             1, 1, 0, 0, true);
    Evaluate(h, c.data(), kScales, kNoExpressions, nullptr, mode, again, 1, 1,
             0, 0, true);
    Check(disabled == original && enabled == again,
          "race gate preserves legacy output and does not mutate cached heads");
    const float before = disabled[0][partial * 3 + 2];
    const float after = enabled[0][partial * 3 + 2];
    Check(before < after && after < 0.f &&
              std::fabs(after - before * (1.f - weight)) < 1e-5f,
          "partial lip weight fades the correction instead of fully clamping");
    bool unchanged = true;
    for (int j = 0; j < 3; ++j)
      unchanged &= enabled[0][outside * 3 + j] == disabled[0][outside * 3 + j];
    for (uint32_t v : {partial, outside})
      for (int j = 0; j < 2; ++j)
        unchanged &= enabled[0][v * 3 + j] == disabled[0][v * 3 + j];
    Check(unchanged, "fade preserves the outer neighbor and horizontal motion");
  }
}

// An oral border duplicates skin positions and native jaw motion, as on
// Khajiit. Nearby teeth alone must not become skin followers.
void SharedMouthBorder(const Mark& mk, ShapeIn skin, ShapeIn mouth) {
  const uint32_t first_skin = skin.nv, first_mouth = mouth.nv;
  for (uint32_t i = 0; i < 8; ++i) {
    const float x = (i % 4) * 0.5f - 0.75f;
    const float z = i < 4 ? -1.8f : -2.2f;
    skin.verts.insert(skin.verts.end(), {x, 9.f, z});
    mouth.verts.insert(mouth.verts.end(), {x + 0.001f, 9.f, z});
  }
  skin.nv += 8;
  mouth.nv += 8;
  for (auto* shape : {&skin, &mouth}) {
    for (auto& morph : shape->morphs)
      if (!morph.empty()) morph.resize(shape->nv * 3);
    for (uint32_t v = shape->nv - 8; v < shape->nv; ++v)
      shape->morphs[kBigAah][v * 3 + 2] = v < shape->nv - 4 ? 0 : -0.4f;
  }
  // Neighbors beyond the full-strength border exercise the skin fade.
  skin.verts.insert(skin.verts.end(), {2.3f, 9.f, -2.2f, 2.8f, 9.f, -2.2f});
  skin.nv += 2;
  for (auto& morph : skin.morphs)
    if (!morph.empty()) morph.resize(skin.nv * 3);
  for (uint32_t v = skin.nv - 2; v < skin.nv; ++v)
    skin.morphs[kBigAah][v * 3 + 2] = -0.4f;
  std::string why;
  for (float factor : {0.5f, 1.f, 2.f}) {
    for (bool skeleton_mouth : {false, true}) {
      ShapeIn s = skin, m = mouth;
      for (auto* shape : {&s, &m}) {
        for (float& v : shape->verts) v *= factor;
        for (auto& morph : shape->morphs)
          for (float& v : morph) v *= factor;
      }
      if (skeleton_mouth)
        for (uint32_t v = 0; v < m.nv; ++v) {
          m.verts[v * 3 + 1] -= 1.55f;
          m.verts[v * 3 + 2] += 120.34f;
        }
      Head h;
      bool ok = BuildHead(mk, {s, m}, h, why);
      Check(ok && h.shapes[1].seam.size() == 8,
            "shared oral border recognized across scale and coordinate frames");
      if (!ok || h.shapes[1].seam.size() != 8) continue;
      // Independent speech channels intentionally pull skin and teeth apart.
      h.row.assign(h.nv, 0);
      h.basis = {0.3f, 0.2f, -0.8f};
      h.shapes[1].expr[10].assign(m.nv * 3, 0.1f);
      Head detached = h;
      detached.shapes[1].seam.clear();
      detached.lip_jaw.clear();
      std::vector<float> c(h.k + kNumMouthCoef, 0);
      c[0] = 1;
      for (int i = 0; i < kNumJawPoints; ++i)
        c[h.k + a2f::kNumTongue + i * 3 + 1] = -0.25f;
      for (auto mode : {TongueMode::kAuthored, TongueMode::kModel}) {
        for (float lip : {0.f, 1.f, 2.f}) {
          float scales[kNumRegions] = {1, 1, 1, 1, 2 - lip, 1};
          std::vector<std::vector<float>> actual, original;
          Evaluate(h, c.data(), scales, kNoExpressions, nullptr, mode, actual,
                   lip, 2 - lip, 0, 0, true);
          Evaluate(detached, c.data(), scales, kNoExpressions, nullptr, mode,
                   original, lip, 2 - lip, 0, 0, true);
          bool closed = true;
          for (uint32_t i = 0; i < 8; ++i)
            for (int j = 0; j < 3; ++j)
              closed &= actual[1][(first_mouth + i) * 3 + j] ==
                        actual[0][(first_skin + i) * 3 + j];
          Check(closed, "speech preserves the authored oral seam offset");
          bool limited = true, upper_unchanged = true;
          const float jaw_z = -0.25f * h.jaw_motion[7] * scales[kJaw];
          for (uint32_t i = 0; i < 8; ++i) {
            const uint32_t v = first_skin + i;
            if (i >= 4)
              limited &= actual[0][v * 3 + 2] >= jaw_z - 1e-5f;
            else
              for (int j = 0; j < 3; ++j)
                upper_unchanged &=
                    actual[0][v * 3 + j] == original[0][v * 3 + j];
            for (int j = 0; j < 2; ++j)
              limited &= actual[0][v * 3 + j] == original[0][v * 3 + j];
          }
          Check(limited, "lower lip cannot drop below rigid jaw displacement");
          Check(upper_unchanged, "upper lip retains independent articulation");
          Check(
              std::equal(actual[1].begin(), actual[1].begin() + first_mouth * 3,
                         original[1].begin()),
              "lip lock preserves rigid interior teeth");
        }
      }
      CheckLipJawMotion(h, s, first_skin);
      CheckLipJawFade(h, skin.nv - 2, skin.nv - 1);
      float expr[kNumExpressions] = {};
      expr[10] = 1;
      std::vector<std::vector<float>> actual, original;
      Evaluate(h, nullptr, kScales, expr, nullptr, TongueMode::kAuthored,
               actual, 1, 1, 0, 0, true);
      Evaluate(detached, nullptr, kScales, expr, nullptr, TongueMode::kAuthored,
               original, 1, 1, 0, 0, true);
      Check(actual == original, "expression-only playback is unchanged");
      const auto expression = actual;
      std::vector<std::vector<float>> speech;
      Evaluate(h, c.data(), kScales, kNoExpressions, nullptr,
               TongueMode::kAuthored, speech, 1, 1, 0, 0, true);
      Evaluate(h, c.data(), kScales, expr, nullptr, TongueMode::kAuthored,
               actual, 1, 1, 0, 0, true);
      bool overlay = true;
      for (size_t si = 0; si < actual.size(); ++si)
        for (size_t i = 0; i < actual[si].size(); ++i)
          overlay &= std::fabs(actual[si][i] - speech[si][i] -
                               expression[si][i]) < 1e-5f;
      Check(overlay, "speech seam correction preserves authored expressions");
      std::fill(c.begin(), c.end(), 0);
      Evaluate(h, c.data(), kScales, kNoExpressions, nullptr,
               TongueMode::kAuthored, actual, 1, 1, 0, 0, true);
      Check(std::all_of(actual[1].begin(), actual[1].end(),
                        [](float v) { return std::fabs(v) < 1e-5f; }),
            "zero speech leaves the mouth at rest");
    }
  }
  for (int control = 0; control < 3; ++control) {
    ShapeIn m = mouth;
    for (uint32_t v = first_mouth; v < m.nv; ++v) {
      if (control == 0 || (control == 2 && v == first_mouth))
        m.verts[v * 3 + 1] += 0.1f;
      if (control == 1) m.morphs[kBigAah][v * 3 + 2] += 1;
    }
    Head h;
    Check(BuildHead(mk, {skin, m}, h, why) && h.shapes[1].seam.empty() &&
              h.lip_jaw.empty(),
          "nearby teeth, conflicting morphs and sparse contacts are not seams");
  }
}

// Mark himself, with upper/lower teeth, optional tongue and two pinned rings.
bool BuildFixture(TongueMotion mode, MouthFixture& fixture) {
  Mark mk;
  // the landmarks' places, and one skin vertex at each (plus filler)
  const float lm[kNumLandmarks][3] = {{-3, 8, 4},  {3, 8, 4},  {-3, 8, 6},
                                      {3, 8, 6},   {0, 8, -5}, {0, 9, -2},
                                      {-2, 8, -1}, {2, 8, -1}};
  for (int i = 0; i < kNumLandmarks * 3; ++i)
    mk.landmarks[i] = lm[i / 3][i % 3];
  std::vector<float> v(&lm[0][0], &lm[0][0] + kNumLandmarks * 3);
  for (int i = 0; i < 64; ++i)
    for (float c : {(i % 4) * 2.f - 3, 6.f + i % 3, (i / 4) - 8.f})
      v.push_back(c);
  // Boundary samples exercise both lip controls under anatomical scaling.
  v.insert(v.end(), {2.3f, 9.f, -2.f, -2.3f, 9.f, -2.f});
  mk.nv = static_cast<uint32_t>(v.size() / 3);
  mk.k = 1;
  mk.verts = v;
  mk.jaw.assign(mk.nv, 0.f);
  mk.jaw[4] = -1;
  mk.row.assign(mk.nv, -1);
  mk.proj.assign(a2f::kNumPca, 0.f);
  const float jaw[a2f::kNumJaw] = {-2, 2, -3, 2,  2, -3, -1, 7,
                                   -3, 1, 7,  -3, 0, 8,  -4};
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
  skin.morphs[kBigAah][v.size() - 4] = -0.1f;
  mk.jaw[mk.nv - 2] = -0.1f;
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
  if (ok && mode == TongueMotion::kStill) {
    // Splitting a stationary-in-BigAah tongue off its teeth must preserve
    // authored motion, while the legacy opt-out keeps the old behavior.
    ShapeIn teeth = mouth, tongue = mouth;
    teeth.nv = 16;
    teeth.verts.resize(48);
    tongue.nv = 4;
    tongue.verts.erase(tongue.verts.begin(), tongue.verts.begin() + 48);
    for (int m = 0; m < kNumMorphs; ++m) {
      if (teeth.morphs[m].empty()) continue;
      teeth.morphs[m].resize(48);
      tongue.morphs[m].erase(tongue.morphs[m].begin(),
                             tongue.morphs[m].begin() + 48);
    }
    Head split;
    Check(BuildHead(mk, {skin, teeth, tongue}, split, why) &&
              split.tongue_verts == h.tongue_verts,
          "separate stationary tongue retains authored classification");
    std::vector<float> pose;
    Coefficients(h, &line, 0, pose);
    for (int ph = 0; ph < kNumPhonemes; ++ph) {
      float cues[kNumPhonemes] = {};
      cues[ph] = 1;
      std::vector<std::vector<float>> combined, separate;
      bool equal = Evaluate(h, pose.data(), kScales, kNoExpressions, cues,
                            TongueMode::kAuthored, combined) &&
                   Evaluate(split, pose.data(), kScales, kNoExpressions, cues,
                            TongueMode::kAuthored, separate);
      for (size_t i = 0; equal && i < separate[2].size(); ++i)
        equal = std::fabs(combined[1][48 + i] - separate[2][i]) < 1e-5f;
      Check(equal, "separate tongue preserves phoneme deformation");
    }
    for (auto& morph : tongue.morphs)
      std::fill(morph.begin(), morph.end(), 0.f);
    Check(BuildHead(mk, {skin, teeth, tongue}, split, why) &&
              split.tongue_verts == 0,
          "static separate oral part is not a tongue");
  }
  // Separate oral parts and head-scale checks use the tongue-free fixture.
  if (ok && mode == TongueMotion::kNone) {
    SharedMouthBorder(mk, skin, mouth);
    std::vector<float> pose;
    std::vector<std::vector<float>> d;
    Coefficients(h, &line, 0, pose);
    Evaluate(h, pose.data(), kScales, kNoExpressions, nullptr,
             TongueMode::kAuthored, d);
    // Asset-derived cases: all five supplied followers name their separate
    // Misc teeth this way. Their name/relationship hint is checked separately
    // in head_parts_test; here it must produce the known rigid-jaw result.
    for (float factor : {0.5f, 2.f}) {
      ShapeIn scaled = skin;
      for (float& value : scaled.verts) value *= factor;
      for (auto& morph : scaled.morphs) {
        for (float& value : morph) value *= factor;
      }
      Head mapped;
      bool passed = BuildHead(mk, {scaled}, mapped, why) &&
                    mapped.lips[0] == h.lips[0] && mapped.lips[1] == h.lips[1] &&
                    mapped.region_w.size() == h.region_w.size();
      for (size_t i = 0; passed && i < h.region_w.size(); ++i) {
        passed = std::fabs(mapped.region_w[i] - h.region_w[i]) < 1e-5f;
      }
      Check(passed, "anatomical scaling preserves lips and region weights");
      bool lips_match = mapped.lip_w.size() == h.lip_w.size();
      for (size_t i = 0; lips_match && i < h.lip_w.size(); ++i)
        lips_match = std::fabs(mapped.lip_w[i] - h.lip_w[i]) < 1e-5f;
      Check(lips_match, "anatomical scaling preserves upper/lower lip weights");
      // Isolate skin speech from jaw/followers and compare both slider ends.
      Head reference = h;
      reference.shapes.resize(1);
      reference.row.assign(reference.nv, 0);
      reference.basis = {1, 1, 1};
      mapped.row.assign(mapped.nv, 0);
      mapped.basis = {factor, factor, factor};
      std::vector<float> speech(1 + kNumMouthCoef, 0);
      speech[0] = 1;
      for (float upper : {0.f, 2.f}) {
        std::vector<std::vector<float>> base_deltas, scaled_deltas;
        bool equal = Evaluate(reference, speech.data(), kScales, kNoExpressions,
                              nullptr, TongueMode::kAuthored, base_deltas,
                              upper, 2 - upper) &&
                     Evaluate(mapped, speech.data(), kScales, kNoExpressions,
                              nullptr, TongueMode::kAuthored, scaled_deltas,
                              upper, 2 - upper);
        for (size_t i = 0; equal && i < base_deltas[0].size(); ++i)
          equal = std::fabs(scaled_deltas[0][i] / factor - base_deltas[0][i]) <
                  1e-5f;
        Check(equal, "lip slider deformation follows anatomical scale");
      }
    }
    ShapeIn collapsed = skin;
    for (int m : {kBlinkLeft, kBlinkRight, kBrowUpLeft, kBrowUpRight, kOohQ,
                  kHappy}) {
      collapsed.morphs[m] = skin.morphs[kBigAah];
    }
    Head invalid_head;
    Check(!BuildHead(mk, {collapsed}, invalid_head, why),
          "nonzero but collapsed landmarks are rejected");
    ShapeIn nonfinite = skin;
    nonfinite.verts[0] = std::numeric_limits<float>::quiet_NaN();
    Check(!BuildHead(mk, {nonfinite}, invalid_head, why),
          "non-finite skin vertices are rejected");
    nonfinite = skin;
    nonfinite.morphs[kBlinkLeft][0] = std::numeric_limits<float>::infinity();
    Check(!BuildHead(mk, {nonfinite}, invalid_head, why),
          "non-finite landmark morphs are rejected");
    ShapeIn candidate = mouth;
    candidate.part = Part::kPinned;
    candidate.mouth_candidate = true;
    for (const char* name :
         {"Female_Teeth_Lily", "Female_Teeth_Alma", "Female_Teeth_DVA",
          "Female_Teeth_Eve", "Female_Teeth_Squirrel_Girl",
          "FutureUpperTeethHead", "OralHeadLayer", "FutureDentition"}) {
      candidate.name = name;
      Head mapped;
      std::vector<std::vector<float>> actual;
      bool passed = BuildHead(mk, {skin, candidate, rings}, mapped, why) &&
                    mapped.shapes[1].part == Part::kMouth &&
                    Evaluate(mapped, pose.data(), kScales, kNoExpressions, nullptr,
                              TongueMode::kAuthored, actual) &&
                    actual == d;
      Check(passed, name);
      Head reordered;
      Check(BuildHead(mk, {candidate, skin, rings}, reordered, why) &&
                reordered.skin == 1 &&
                reordered.shapes[0].part == Part::kMouth &&
                Evaluate(reordered, pose.data(), kScales, kNoExpressions,
                         nullptr, TongueMode::kAuthored, actual) &&
                actual[0] == d[1] && actual[1] == d[0] && actual[2] == d[2],
            "oral extra before skin preserves mapping and deformation");
    }
    ShapeIn explicit_mouth = mouth;
    explicit_mouth.name = "00DZBMself_FemaleMouth";
    Head explicit_head;
    Check(BuildHead(mk, {explicit_mouth, skin}, explicit_head, why) &&
              explicit_head.skin == 1 &&
              explicit_head.shapes[0].part == Part::kMouth,
          "explicit mouth extra cannot become calibration skin");
    auto mapped_part = [&](ShapeIn part, Part expected, const char* label) {
      Head mapped;
      bool passed = BuildHead(mk, {skin, part}, mapped, why);
      passed = passed && (expected == Part::kNone
                             ? mapped.shapes.size() == 1
                             : mapped.shapes.size() == 2 &&
                                   mapped.shapes[1].part == expected);
      Check(passed, label);
    };
    for (int axis = 0; axis < 3; ++axis) {
      for (float distance : {-30.f, 30.f}) {
        ShapeIn remote = candidate;
        for (uint32_t i = 0; i < remote.nv; ++i)
          remote.verts[i * 3 + axis] += distance;
        mapped_part(remote, Part::kPinned, "remote anatomy name stays accessory");
      }
    }
    ShapeIn rejected = candidate;
    for (uint32_t i = 0; i < rejected.nv; ++i) rejected.verts[i * 3] += 30;
    rejected.morphs[kHappy].assign(rejected.verts.size(), 0.5f);
    Head rejected_head;
    Check(BuildHead(mk, {skin, rejected}, rejected_head, why) &&
              rejected_head.shapes[1].part == Part::kPinned &&
              rejected_head.shapes[1].expr[10].empty(),
          "rejected candidate does not gain expression writes");
    ShapeIn missing = candidate;
    missing.morphs.clear();
    mapped_part(missing, Part::kPinned, "missing morph data keeps fallback");
    ShapeIn unhinted = candidate;
    unhinted.mouth_candidate = false;
    mapped_part(unhinted, Part::kPinned, "position alone is insufficient");
    ShapeIn unknown = candidate;
    unknown.part = Part::kNone;
    mapped_part(unknown, Part::kMouth, "unregistered oral name with geometry");
    unknown.morphs.clear();
    mapped_part(unknown, Part::kNone, "rejected unregistered part stays excluded");
    ShapeIn upper_only = candidate;
    std::fill(upper_only.morphs[kBigAah].begin(),
              upper_only.morphs[kBigAah].end(), 0.f);
    Head upper_head;
    std::vector<std::vector<float>> upper_deltas;
    bool static_upper = BuildHead(mk, {skin, upper_only}, upper_head, why) &&
                        upper_head.shapes[1].part == Part::kMouth &&
                        Evaluate(upper_head, pose.data(), kScales, kNoExpressions,
                                 nullptr, TongueMode::kAuthored, upper_deltas);
    if (static_upper)
      for (float value : upper_deltas[1]) static_upper &= value == 0;
    Check(static_upper, "separate upper teeth remain stationary");
    ShapeIn skeleton = candidate;
    for (uint32_t i = 0; i < skeleton.nv; ++i) {
      skeleton.verts[i * 3 + 1] -= 1.55f;
      skeleton.verts[i * 3 + 2] += 120.34f;
    }
    mapped_part(skeleton, Part::kMouth, "skeleton-space teeth use head-space bounds");
    // Both the face and the cavity change size, so the gate must scale too.
    for (float scale : {0.5f, 2.f}) {
      ShapeIn scaled_skin = skin, scaled_part = candidate;
      for (ShapeIn* sh : {&scaled_skin, &scaled_part}) {
        for (float& value : sh->verts) value *= scale;
        for (auto& morph : sh->morphs)
          for (float& value : morph) value *= scale;
      }
      Head mapped;
      Check(BuildHead(mk, {scaled_skin, scaled_part}, mapped, why) &&
                mapped.shapes[1].part == Part::kMouth,
            "oral bounds scale with the face");
    }
    ShapeIn invalid = candidate;
    invalid.morphs[kBigAah][0] = std::numeric_limits<float>::quiet_NaN();
    invalid.part = Part::kNone;
    mapped_part(invalid, Part::kNone, "nonfinite oral morph rejected");
  }
  // A second controlled head with real lip motion exercises BuildHead's
  // weight storage, not just the standalone deformation test's weights.
  skin.morphs[kBigAah][5 * 3 + 2] = -.5f;
  Head lip_head;
  bool lip_ok = BuildHead(mk, {skin}, lip_head, why) &&
                lip_head.lip_w.size() == skin.nv * 2;
  float upper_weight = 0, lower_weight = 0;
  for (uint32_t v = 0; lip_ok && v < skin.nv; ++v) {
    float u = lip_head.lip_w[v * 2], l = lip_head.lip_w[v * 2 + 1];
    lip_ok = std::isfinite(u) && std::isfinite(l) && u >= 0 && l >= 0 &&
             u + l <= 1.00001f;
    upper_weight += u;
    lower_weight += l;
  }
  Check(lip_ok && upper_weight > 0 && lower_weight > 0, "mapped lip weights");
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

void SmileEyes(Head h) {
  h.region_w.assign(h.nv * kNumSkinRegions, 0.f);
  for (uint32_t v = 0; v < h.nv; ++v)
    h.region_w[v * kNumSkinRegions + (v == 0 ? kEyes : kMouth)] = 1;
  auto& happy = h.shapes[h.skin].expr[kHappy - kExpression];
  happy.assign(h.nv * 3, 1.f);
  std::vector<float> pose(h.k + kNumMouthCoef, 0.f);
  std::vector<std::vector<float>> base, boosted, neutral, muted;
  Evaluate(h, pose.data(), kScales, kNoExpressions, nullptr,
           TongueMode::kAuthored, base, 1, 1, 0, 1);
  Evaluate(h, pose.data(), kScales, kNoExpressions, nullptr,
           TongueMode::kAuthored, boosted, 1, 1, .5f, 1);
  Evaluate(h, pose.data(), kScales, kNoExpressions, nullptr,
           TongueMode::kAuthored, neutral, 1, 1, .5f, 0);
  bool ok = base == neutral;
  for (uint32_t i = 0; i < h.nv * 3; ++i)
    ok &= std::fabs(boosted[h.skin][i] - base[h.skin][i] -
                    (i < 3 ? .5f : 0.f)) < 1e-5f;
  Check(ok, "smile boosts eyes only; neutral speech and zero boost unchanged");
  float scales[kNumRegions] = {1, 0, 1, 1, 1, 1};
  Evaluate(h, pose.data(), scales, kNoExpressions, nullptr,
           TongueMode::kAuthored, muted, 1, 1, .5f, 1);
  Check(muted == base, "speech smile respects disabled eye motion");
  float levels[kNumExpressions] = {};
  levels[kHappy - kExpression] = .5f;
  Evaluate(h, nullptr, kScales, levels, nullptr, TongueMode::kAuthored, base, 1,
           1, 0);
  Evaluate(h, nullptr, kScales, levels, nullptr, TongueMode::kAuthored, boosted,
           1, 1, .5f);
  ok = true;
  for (uint32_t i = 0; i < h.nv * 3; ++i)
    ok &= std::fabs(boosted[h.skin][i] - base[h.skin][i] -
                    (i < 3 ? .25f : 0.f)) < 1e-5f;
  Check(ok, "native happiness gains eye motion without changing mouth");
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
      SmileEyes(fixture.head);
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
