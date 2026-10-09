// Collapsed geometry must not drag landmark fits into hidden mesh debris.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

// Exercise private calibration helpers without expanding the production API.
#include "../src/animation/direct.cc"

#define CHECK(x)                                      \
  do {                                                \
    if (!(x)) {                                       \
      fprintf(stderr, "line %d: %s\n", __LINE__, #x); \
      std::exit(1);                                   \
    }                                                 \
  } while (0)
using namespace direct;

ShapeIn Fixture(bool conflicting = true, int count = 64) {
  ShapeIn s;
  s.skin = true;
  s.part = Part::kHead;
  // Distributed face samples; only OohQ needs motion for these centroid tests.
  for (int z = 0; z < 9; ++z)
    for (int x = 0; x < 9; ++x)
      s.verts.insert(s.verts.end(), {static_cast<float>(x - 4), 10.f,
                                     static_cast<float>(z - 4)});
  // A buried cluster with distinct authored movements after spatial collapse.
  for (int i = 0; i < count; ++i)
    s.verts.insert(s.verts.end(), {static_cast<float>(i % 4) * .0005f, 0.f,
                                   static_cast<float>(i / 4) * .0001f});
  s.nv = static_cast<uint32_t>(s.verts.size() / 3);
  s.morphs.resize(kNumMorphs, std::vector<float>(s.verts.size(), 0));
  for (uint32_t i = 0; i < s.nv; ++i) {
    s.morphs[kOohQ][i * 3 + 1] =
        i < 81 || !conflicting ? 1.f : (i % 2 ? 1.f : -1.f);
  }
  return s;
}

// Mark and a skin with a vertex at each of his landmarks (direct_test's),
// which BuildHead maps.
ShapeIn MarkSkin(Mark& mk) {
  const float lm[kNumLandmarks][3] = {
      {-3, 8, 4}, {3, 8, 4},  {-3, 8, 6},  {3, 8, 6},
      {0, 8, -5}, {0, 9, -2}, {-2, 8, -1}, {2, 8, -1}};
  for (int i = 0; i < kNumLandmarks * 3; ++i)
    mk.landmarks[i] = lm[i / 3][i % 3];
  std::vector<float> v(&lm[0][0], &lm[0][0] + kNumLandmarks * 3);
  for (int i = 0; i < 64; ++i)
    for (float c : {(i % 4) * 2.f - 3, 6.f + i % 3, (i / 4) - 8.f})
      v.push_back(c);
  v.insert(v.end(), {2.3f, 9.f, -2.f, -2.3f, 9.f, -2.f});
  mk.nv = static_cast<uint32_t>(v.size() / 3);
  mk.k = 1;
  mk.verts = v;
  mk.jaw.assign(mk.nv, 0.f);
  mk.jaw[4] = -1;
  mk.jaw[mk.nv - 2] = -0.1f;
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
  skin.morphs[kBigAah][v.size() - 4] = -0.1f;
  return skin;
}

// Mouth interiors may be hidden as a broad cloud rather than one point.
// Keep the visible skin and its authored morphs untouched.
ShapeIn BuriedMouth(const ShapeIn& clean, int count = 108, float y = -4.f,
                    float strength = 1.f) {
  ShapeIn skin = clean;
  for (int i = 0; i < count; ++i)
    skin.verts.insert(skin.verts.end(),
                      {(i % 9 - 4) * .3f, y + (i % 3) * .1f,
                       -2.f + (i / 9) * .07f});
  skin.nv += count;
  for (auto& morph : skin.morphs) morph.resize(skin.nv * 3, 0.f);
  for (uint32_t i = clean.nv; i < skin.nv; ++i) {
    skin.morphs[kOohQ][i * 3 + 1] = i % 2 ? strength : -strength;
    skin.morphs[kBigAah][i * 3 + 2] = -strength;
  }
  return skin;
}

void TestBuriedMouth() {
  Mark model;
  const ShapeIn clean = MarkSkin(model);
  const ShapeIn buried = BuriedMouth(clean);
  const auto mask = CalibrationExclusions(buried);
  CHECK(mask.size() == buried.nv);
  CHECK(std::count(mask.begin(), mask.end(), 1) == 108);
  CHECK(std::none_of(mask.begin(), mask.begin() + clean.nv,
                    [](uint8_t ignored) { return ignored != 0; }));
  CHECK(CalibrationExclusions(clean).empty());
  CHECK(CalibrationExclusions(BuriedMouth(clean, 15)).empty());
  CHECK(CalibrationExclusions(BuriedMouth(clean, 108, 0.f)).empty());
  CHECK(CalibrationExclusions(BuriedMouth(clean, 108, -4.f, .2f)).empty());

  // A mouth cue alone cannot establish which direction is behind the face.
  ShapeIn no_eyes = buried;
  for (Morph morph : {kBlinkLeft, kBlinkRight})
    std::fill(no_eyes.morphs[morph].begin(), no_eyes.morphs[morph].end(), 0.f);
  CHECK(CalibrationExclusions(no_eyes).empty());

  // A tight collapse with one stray point in the old detector's empty
  // shell is still buried mouth geometry.
  ShapeIn stray = buried;
  for (uint32_t i = clean.nv; i < stray.nv; ++i)
    Put(&stray.verts[i * 3], {0.f, -4.f, -2.f});
  stray.verts[clean.nv * 3] = .06f;
  const auto stray_mask = CalibrationExclusions(stray);
  CHECK(std::count(stray_mask.begin(), stray_mask.end(), 1) == 108);

  Head reference, mapped;
  std::string why;
  CHECK(BuildHead(model, {clean}, reference, why));
  CHECK(BuildHead(model, {buried}, mapped, why));
  CHECK(mapped.calibration_ignored == 108);
  CHECK(mapped.scale == reference.scale);
  CHECK(mapped.lips[0] == reference.lips[0]);
  CHECK(mapped.lips[1] == reference.lips[1]);
  CHECK(std::equal(std::begin(mapped.residual), std::end(mapped.residual),
                   std::begin(reference.residual)));
  CHECK(std::equal(clean.verts.begin(), clean.verts.end(), buried.verts.begin()));

  // The new rule uses Skyrim's head-local axes, like registration itself;
  // it must still be independent of translation and uniform asset scale.
  for (float scale : {.01f, 1.f, 100.f}) {
    ShapeIn moved = buried;
    for (uint32_t i = 0; i < moved.nv; ++i) {
      const V3 p = At(&buried.verts[i * 3]);
      Put(&moved.verts[i * 3],
          {(p.x + 5) * scale, (p.y - 3) * scale, (p.z + 12) * scale});
      for (auto& morph : moved.morphs)
        for (int c = 0; c < 3; ++c) morph[i * 3 + c] *= scale;
    }
    CHECK(CalibrationExclusions(moved) == mask);
  }
  std::vector<uint32_t> order(buried.nv);
  for (uint32_t i = 0; i < buried.nv; ++i) order[i] = i;
  std::shuffle(order.begin(), order.end(), std::mt19937(819));
  ShapeIn shuffled = buried;
  for (uint32_t i = 0; i < buried.nv; ++i) {
    std::copy_n(&buried.verts[order[i] * 3], 3, &shuffled.verts[i * 3]);
    for (int m = 0; m < kNumMorphs; ++m)
      std::copy_n(&buried.morphs[m][order[i] * 3], 3,
                  &shuffled.morphs[m][i * 3]);
  }
  const auto shuffled_mask = CalibrationExclusions(shuffled);
  CHECK(shuffled_mask.size() == buried.nv);
  for (uint32_t i = 0; i < buried.nv; ++i)
    CHECK(shuffled_mask[i] == mask[order[i]]);
}

int main() {
  TestBuriedMouth();
  // Compare the bounded density query with exhaustive counting and the old
  // nearest-neighbour predicate, including exact-radius and duplicate points.
  std::mt19937 rng(719);
  std::uniform_real_distribution<float> sample(-2.f, 2.f);
  std::vector<float> points;
  for (int i = 0; i < 256; ++i)
    points.insert(points.end(), {sample(rng), sample(rng), sample(rng)});
  for (int i = 0; i < 40; ++i) points.insert(points.end(), {0.f, 0.f, 0.f});
  points.insert(points.end(), {1.f, 0.f, 0.f, -1.f, 0.f, 0.f});
  const uint32_t n = static_cast<uint32_t>(points.size() / 3);
  KdTree tree(points.data(), n);
  for (int q = 0; q < 100; ++q) {
    const V3 p = q ? V3{sample(rng), sample(rng), sample(rng)} : V3{0, 0, 0};
    for (float radius : {0.f, .01f, .5f, std::nextafter(1.f, 0.f), 1.f,
                         std::nextafter(1.f, 2.f), 10.f}) {
      uint32_t expected = 0;
      for (uint32_t i = 0; i < n; ++i)
        expected += Len(At(&points[i * 3]) - p) <= radius;
      for (uint32_t limit : {0u, 1u, 31u, 32u, 33u, 400u})
        CHECK(tree.CountWithin(0, n, p, radius, limit) ==
              std::min(expected, limit));
      uint32_t indices[32];
      float distances[32];
      const bool previous = tree.Nearest(p, 32, indices, distances) == 32 &&
                            distances[31] <= radius;
      CHECK((tree.CountWithin(0, n, p, radius, 32) == 32) == previous);
    }
  }
  CHECK(tree.CountWithin(0, 0, {0, 0, 0}, 1, 32) == 0);

  auto s = Fixture();
  const auto verts = s.verts, morph = s.morphs[kOohQ];
  auto mask = CalibrationExclusions(s);
  CHECK(mask.size() == s.nv);
  CHECK(std::count(mask.begin(), mask.end(), 1) == 64);
  CHECK(std::none_of(mask.begin(), mask.begin() + 81,
                     [](uint8_t x) { return x != 0; }));
  auto old = MotionCentroid(s.verts, morph, s.nv);
  auto fit = MotionCentroid(s.verts, morph, s.nv, 0, 0, mask);
  CHECK(old.y < 6 && std::fabs(fit.y - 10) < 1e-6f);
  CHECK(s.verts == verts && s.morphs[kOohQ] == morph);
  CHECK(CalibrationExclusions(Fixture(false)).empty());  // coherent duplicates
  // Preserve groups no larger than the touching-lip/corner groups observed
  // in the corpus, even when their upper/lower lip motions conflict.
  CHECK(CalibrationExclusions(Fixture(true, 9)).empty());
  CHECK(CalibrationExclusions(Fixture(true, 15)).empty());
  const auto minimum_cluster = CalibrationExclusions(Fixture(true, 16));
  CHECK(std::count(minimum_cluster.begin(), minimum_cluster.end(), 1) == 16);
  const auto formerly_missed = CalibrationExclusions(Fixture(true, 24));
  CHECK(std::count(formerly_missed.begin(), formerly_missed.end(), 1) == 24);

  auto connected = Fixture();
  // One nearby surface sample in the surrounding shell defeats isolation.
  connected.verts[0] = .025f;
  connected.verts[1] = 0;
  connected.verts[2] = 0;
  CHECK(CalibrationExclusions(connected).empty());

  auto dense = Fixture();
  // A continuous, dense sheet spans far beyond any candidate collapse.
  for (int i = 81; i < static_cast<int>(dense.nv); ++i) {
    dense.verts[i * 3] = (i - 81) % 8 * .006f;
    dense.verts[i * 3 + 2] = (i - 81) / 8 * .006f;
  }
  CHECK(CalibrationExclusions(dense).empty());

  // The detection must not depend on an NPC, absolute position, size or axes.
  for (float scale : {.01f, 1.f, 100.f}) {
    auto moved = s;
    for (uint32_t i = 0; i < s.nv; ++i) {
      const V3 p = At(&s.verts[i * 3]);
      Put(&moved.verts[i * 3],
          {(p.z + 5) * scale, (p.x - 3) * scale, (p.y + 12) * scale});
      for (auto& m : moved.morphs) {
        const V3 d = At(&m[i * 3]);
        Put(&m[i * 3], {d.z * scale, d.x * scale, d.y * scale});
      }
    }
    CHECK(CalibrationExclusions(moved) == mask);
  }
  std::vector<uint32_t> order(s.nv);
  for (uint32_t i = 0; i < s.nv; ++i) order[i] = i;
  std::shuffle(order.begin(), order.end(), std::mt19937(314));
  auto shuffled = s;
  for (uint32_t i = 0; i < s.nv; ++i) {
    std::copy_n(&s.verts[order[i] * 3], 3, &shuffled.verts[i * 3]);
    for (int m = 0; m < kNumMorphs; ++m)
      std::copy_n(&s.morphs[m][order[i] * 3], 3, &shuffled.morphs[m][i * 3]);
  }
  const auto shuffled_mask = CalibrationExclusions(shuffled);
  CHECK(shuffled_mask.size() == s.nv);
  for (uint32_t i = 0; i < s.nv; ++i) CHECK(shuffled_mask[i] == mask[order[i]]);

  // A required calibration channel surviving only on debris is unsupported,
  // so BuildHead must retain the engine's native animation instead of fitting
  // it.
  auto unsupported = s;
  for (Morph m : {kBlinkLeft, kBlinkRight, kBrowUpLeft, kBrowUpRight, kBigAah,
                  kOohQ, kHappy})
    unsupported.morphs[m] = s.morphs[kOohQ];
  std::fill_n(unsupported.morphs[kOohQ].begin(), 81 * 3, 0.f);
  Head h;
  Mark model;
  std::string why;
  CHECK(!BuildHead(model, {unsupported}, h, why));
  CHECK(why == "no usable OohQ calibration");
  CHECK(h.calibration_ignored == 64);

  // A head maps the same with debris as without; a cluster below the debris
  // size is fitted as skin and moves the landmarks (or breaks the layout).
  Mark mk;
  const ShapeIn clean = MarkSkin(mk);
  Head reference;
  CHECK(BuildHead(mk, {clean}, reference, why));
  CHECK(reference.calibration_ignored == 0 && reference.scale > 0);
  for (int count : {64, 15}) {
    ShapeIn debris = clean;
    for (int i = 0; i < count; ++i)
      debris.verts.insert(debris.verts.end(),
                          {(i % 4) * .0005f, 7.f, -20.f + (i / 4) * .0001f});
    debris.nv += count;
    for (auto& m : debris.morphs) m.resize(debris.nv * 3, 0.f);
    for (int i = 0; i < count; ++i)
      debris.morphs[kOohQ][(clean.nv + i) * 3 + 1] = i % 2 ? 1.f : -1.f;
    Head mapped;
    const bool built = BuildHead(mk, {debris}, mapped, why);
    bool same = built && std::fabs(mapped.scale - reference.scale) < 1e-4f &&
                mapped.lips[0] == reference.lips[0] &&
                mapped.lips[1] == reference.lips[1];
    for (int i = 0; same && i < kNumLandmarks; ++i)
      same = std::fabs(mapped.residual[i] - reference.residual[i]) < 1e-4f;
    if (count == 64)
      CHECK(same && mapped.calibration_ignored == 64);
    else
      CHECK(!same && (!built || mapped.calibration_ignored == 0));
  }
  puts(
      "calibration clusters, preserved surfaces, transforms, unsupported "
      "fallback and mapped heads: pass");
}
