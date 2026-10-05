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
  direct::FrameOffset(v.data(), static_cast<uint32_t>(v.size() / 3), off);
  bool ok = std::fabs(off[0]) < 1e-4f && std::fabs(off[1] - y) < 1e-4f &&
            std::fabs(off[2] - z) < 1e-4f;
  if (!ok) ++failures;
  printf("%s %s: offset %.2f %.2f %.2f, want 0 %.2f %.2f\n", ok ? "ok" : "FAIL",
         what, off[0], off[1], off[2], y, z);
}

// A head that is Mark himself (so the mapping is the identity), with a mouth
// part of upper teeth and lower teeth that BigAah turns about a hinge.
void MouthOnJaw() {
  using namespace direct;
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
  const int at[kNumLandmarks] = {kBlinkLeft,   kBlinkRight, kBrowUpLeft,
                                 kBrowUpRight, kBigAah,     kOohQ,
                                 kHappy,       kHappy};
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
  Head h;
  std::string why;
  bool ok = BuildHead(mk, {skin, mouth}, h, why) && h.shapes.size() == 2 &&
            h.jaw_verts == 8;
  // the jaw's points all 0.25 down in Mark's axes (y up): the head's z
  a2f::Line line;
  line.frames = 1;
  line.direct_dim = mk.k + kNumMouthCoef;
  line.direct.assign(line.direct_dim, 0.f);
  for (int i = 0; i < kNumJawPoints; ++i)
    line.direct[mk.k + a2f::kNumTongue + i * 3 + 1] = -0.25f;
  const float scales[kNumRegions] = {1, 1, 1, 1, 1, 1};
  const float kNoExpressions[kNumExpressions] = {};
  std::vector<std::vector<float>> d;
  std::vector<float> pose;
  ok = ok && Coefficients(h, &line, 0, pose) &&
       Evaluate(h, pose.data(), scales, kNoExpressions, d);
  float upper = 0, lower = 0;
  for (int i = 0; ok && i < 16; ++i)
    for (int c = 0; c < 3; ++c) {
      float want = i >= 8 && c == 2 ? -0.5f : 0;
      float& worst = i < 8 ? upper : lower;
      worst = std::fmax(worst, std::fabs(d[1][i * 3 + c] - want));
    }
  ok = ok && upper < 1e-4f && lower < 1e-3f;
  if (!ok) ++failures;
  printf("%s mouth on the jaw: %s upper teeth off by %.4f, lower by %.4f\n",
         ok ? "ok" : "FAIL", why.c_str(), upper, lower);
  // a pose carried over alone gives what its line gave
  std::vector<std::vector<float>> d2;
  bool same = ok && Coefficients(h, &line, 0, pose) &&
              Evaluate(h, pose.data(), scales, kNoExpressions, d2);
  float off = 0;
  for (size_t i = 0; same && i < d[1].size(); ++i)
    off = std::fmax(off, std::fabs(d2[1][i] - d[1][i]));
  same =
      same && off < 1e-5f && !Evaluate(h, nullptr, scales, kNoExpressions, d2);
  if (!same) ++failures;
  printf("%s a carried pose: off by %.5f\n", same ? "ok" : "FAIL", off);
}

}  // namespace

int main() {
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
  MouthOnJaw();
  printf("%s: %d failures\n", failures ? "FAIL" : "ok", failures);
  return failures ? 1 : 0;
}
