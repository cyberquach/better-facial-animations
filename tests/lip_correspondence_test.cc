// Correspondence must preserve lip identity even when target-space cropping
// excludes a whole source lip. Other rows retain their original basis.
#include <cstdio>
#include <cstdlib>

#include "../src/animation/direct.cc"
#define CHECK(x)                                           \
  do {                                                     \
    if (!(x)) {                                            \
      std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); \
      std::exit(1);                                        \
    }                                                      \
  } while (0)
using namespace direct;
void CheckExtendedControls(const Registration& g, const ShapeIn& shape,
                           const std::vector<float>& follow,
                           const std::vector<uint8_t>& labels,
                           const std::vector<float>& extension, Head h) {
  const Head unmapped = h;
  MapLipControls(g, shape, follow, extension, h);
  CHECK(labels[0] == kUpperLip && follow[0] == 0);
  CHECK(h.lip_w[0] > 0.99f && h.lip_w[1] == 0);
  CHECK(h.extended_lip_controls);
  CHECK(!h.calibrated_lips.empty());
  h.shapes.resize(1);
  h.shapes[0].part = Part::kHead;
  h.shapes[0].nv = h.nv;
  h.shapes[0].expr[0].assign(h.nv * 3, 0.125f);
  h.region_w.assign(h.nv * kNumSkinRegions, 0);
  for (uint32_t v = 0; v < h.nv; ++v)
    h.region_w[v * kNumSkinRegions + kMouth] = 1;
  const std::vector<float> blink(shape.verts.size(), 0);
  std::vector<KdTree> trees;
  for (uint32_t vertex : {0u, 1u}) {
    ShapeIn attachment;
    attachment.part = Part::kPinned;
    attachment.nv = 1;
    attachment.verts.assign(shape.verts.begin() + vertex * 3,
                            shape.verts.begin() + vertex * 3 + 3);
    for (bool pinned : {false, true}) {
      Head::Shape part{};
      part.part = Part::kPinned;
      part.nv = 1;
      Follow(attachment, shape.verts, shape.nv, labels, g.lm[kLandmarkLips].z,
             blink, g.mid, trees, part);
      if (pinned)
        Pin(attachment, shape.verts, shape.nv, labels, g.lm[kLandmarkLips].z,
            part);
      for (int j = 0; j < kFollow; ++j)
        if (part.w[j] > 0) CHECK(part.idx[j] == vertex);
      h.shapes.push_back(part);
    }
  }
  float scales[kNumRegions] = {1, 1, 1, 1, 1, 1};
  float expression[kNumExpressions] = {};
  std::vector<float> coefficients(h.k + kNumMouthCoef, 0);
  coefficients[0] = 1;
  std::vector<std::vector<float>> output, native;
  for (float level : {0.0f, 0.5f, 1.0f}) {
    expression[0] = level;
    Evaluate(h, nullptr, scales, expression, nullptr, TongueMode::kAuthored,
             native);
    for (float upper : {0.0f, 1.0f, 2.0f})
      for (float lower : {0.0f, 1.0f, 2.0f}) {
        CHECK(Evaluate(h, coefficients.data(), scales, expression, nullptr,
                       TongueMode::kAuthored, output, upper, lower));
        for (int j = 0; j < 3; ++j) {
          const float speech = unmapped.basis[h.row[0] * 3 + j] * upper;
          const float authored = native[0][j];
          CHECK(std::fabs(output[0][j] - speech - authored) < 1e-6f);
          CHECK(std::fabs(output[1][j] - output[0][j]) < 1e-6f);
          CHECK(std::fabs(output[2][j] - output[0][j]) < 1e-6f);
          CHECK(std::fabs(output[3][j] - output[0][3 + j]) < 1e-6f);
          CHECK(std::fabs(output[4][j] - output[0][3 + j]) < 1e-6f);
        }
      }
  }
  // The two sides of the old upper crop must share one continuous control.
  ShapeIn boundary = shape;
  boundary.verts[2] = g.lm[kLandmarkLips].z + 2 * g.unit - 0.0001f;
  boundary.verts[5] = g.lm[kLandmarkLips].z + 2 * g.unit + 0.0001f;
  Head adjacent;
  adjacent.nv = 2;
  adjacent.k = 1;
  adjacent.row = {0, 1};
  adjacent.basis.assign(6, 1);
  // A correct-source neighbor needs the same control, without a repair row.
  adjacent.calibrated_lips = {0};
  MapLipControls(g, boundary, {0, 0}, {1, 1}, adjacent);
  CHECK(std::fabs(adjacent.lip_w[0] - adjacent.lip_w[2]) < 1e-3f);
}

void CheckControlTransition(const Mark& mk, const Registration& g,
                            ShapeIn shape, const std::vector<float>& source,
                            const std::vector<float>& source_follow) {
  // Neighboring authored jaw values cross each end of the shared-lip band.
  const float jaw[] = {0.119f, 0.121f, 0.249f, 0.251f};
  shape.nv = 6;
  for (float f : jaw) shape.verts.insert(shape.verts.end(), {0, 9, 0.1f});
  for (auto& morph : shape.morphs) morph.resize(shape.nv * 3, 0);
  for (uint32_t i = 0; i < 4; ++i) {
    shape.morphs[kBigAah][(i + 2) * 3 + 2] = -jaw[i];
    shape.morphs[kOohQ][(i + 2) * 3] = 0.01f;
  }
  std::vector<float> follow = {-1, 1, -1, -1, -1, -1}, extension;
  std::vector<uint8_t> labels = {kNoLip, kLowerLip, kNoLip,
                                 kNoLip, kNoLip,    kNoLip};
  Head h;
  CHECK(MapSkin(mk, g, shape, source, follow, source_follow, labels, extension,
                h));
  CHECK(extension[2] > 0.99f && extension[3] > 0.99f);
  CHECK(extension[4] > 0 && extension[4] < 0.001f && extension[5] == 0);
  CHECK(labels[2] == kUpperLip && labels[3] == kNoLip);
  // Isolate control continuity from different correspondence rows: every
  // source row now has the same speech, before the neutral ratio is baked.
  std::fill(h.basis.begin(), h.basis.end(), 1);
  MapLipControls(g, shape, follow, extension, h);
  h.shapes.resize(1);
  h.shapes[0].nv = h.nv;
  h.shapes[0].expr[0].assign(h.nv * 3, 0.125f);
  h.region_w.assign(h.nv * kNumSkinRegions, 0);
  for (uint32_t v = 0; v < h.nv; ++v)
    h.region_w[v * kNumSkinRegions + kMouth] = 1;
  float scales[kNumRegions] = {1, 1, 1, 1, 1, 1};
  float expression[kNumExpressions] = {};
  std::vector<float> coefficients(h.k + kNumMouthCoef, 0);
  coefficients[0] = 1;
  std::vector<std::vector<float>> output;
  for (float native : {0.0f, 0.7f}) {
    expression[0] = native;
    for (float upper : {0.0f, 1.0f, 2.0f}) {
      for (float lower : {0.0f, 1.0f, 2.0f}) {
        CHECK(Evaluate(h, coefficients.data(), scales, expression, nullptr,
                       TongueMode::kAuthored, output, upper, lower));
        for (int axis = 0; axis < 3; ++axis) {
          CHECK(std::fabs(output[0][2 * 3 + axis] - output[0][3 * 3 + axis]) <
                0.001f);
          CHECK(std::fabs(output[0][4 * 3 + axis] - output[0][5 * 3 + axis]) <
                0.001f);
        }
      }
    }
  }
}

void CheckSmallConfidenceFollowers(const Mark& mk, const Registration& g,
                                   const ShapeIn& shape,
                                   const std::vector<float>& source_follow) {
  std::vector<Head::Shape> lower_followers;
  float small_support = 0;
  for (float distance : {1.0f, 0.1f, 0.05f}) {
    auto source = mk.verts;
    for (uint32_t i = 0; i < mk.nv; ++i) {
      Put(&source[i * 3], At(shape.verts.data()));
      source[i * 3] += i < kKnn - 1 ? 0 : i < 16 ? 100 : distance;
    }
    std::vector<float> follow = {-1, 1}, extension;
    std::vector<uint8_t> labels = {kNoLip, kLowerLip};
    Head h;
    CHECK(MapSkin(mk, g, shape, source, follow, source_follow, labels,
                  extension, h));
    if (distance == 1) {
      CHECK(extension.empty() && labels[0] == kNoLip);
    } else {
      CHECK(!extension.empty());
      CHECK(extension[0] > 0 && extension[0] < 0.0001f);
      CHECK(labels[0] == kUpperLip);
      small_support = extension[0];
    }
    const std::vector<float> blink(shape.verts.size(), 0);
    std::vector<KdTree> trees;
    for (bool pinned : {false, true}) {
      ShapeIn attached;
      attached.part = Part::kPinned;
      attached.nv = 1;
      attached.verts.assign(shape.verts.begin(), shape.verts.begin() + 3);
      Head::Shape upper;
      Follow(attached, shape.verts, shape.nv, labels, g.lm[kLandmarkLips].z,
             blink, g.mid, trees, upper);
      if (pinned)
        Pin(attached, shape.verts, shape.nv, labels, g.lm[kLandmarkLips].z,
            upper);
      for (int j = 0; j < kFollow; ++j)
        if (upper.w[j] > 0) CHECK(upper.idx[j] == 0);
      attached.verts = {0, 9, g.lm[kLandmarkLips].z - 0.1f};
      Head::Shape lower;
      Follow(attached, shape.verts, shape.nv, labels, g.lm[kLandmarkLips].z,
             blink, g.mid, trees, lower);
      if (pinned)
        Pin(attached, shape.verts, shape.nv, labels, g.lm[kLandmarkLips].z,
            lower);
      lower_followers.push_back(std::move(lower));
    }
  }
  auto upper_influence = [](const Head::Shape& shape) {
    float weight = 0;
    for (int j = 0; j < kFollow; ++j)
      if (shape.idx[j] == 0) weight += shape.w[j];
    return weight;
  };
  // Follower identity remains discrete: small positive calibration excludes
  // an upper lip from a lower attachment. Its weights are stable as that
  // positive correction changes, but not continuous across absent identity.
  for (int pinned = 0; pinned < 2; ++pinned) {
    CHECK(upper_influence(lower_followers[pinned]) > 0);
    CHECK(upper_influence(lower_followers[2 + pinned]) == 0);
    CHECK(lower_followers[2 + pinned].idx == lower_followers[4 + pinned].idx);
    CHECK(lower_followers[2 + pinned].w == lower_followers[4 + pinned].w);
  }
  std::printf(
      "small correction %.8f: lower attachment upper weight %.8f -> 0\n",
      small_support, upper_influence(lower_followers[0]));
}

// Comparable corrected patches with different authored pucker support must
// receive different control coverage without changing default speech.
void CheckAuthoredControlSupport(const Mark& mk, const Registration& g,
                                 const std::vector<float>& source,
                                 const std::vector<float>& source_follow) {
  ShapeIn skin;
  skin.nv = 3;
  skin.skin = true;
  skin.part = Part::kHead;
  skin.verts = {-0.7f, 9, 0.1f, 0.7f, 9, 0.1f, 0, 9, -2.2f};
  skin.morphs.assign(kNumMorphs, std::vector<float>(9, 0));
  skin.morphs[kBigAah][8] = -1;
  skin.morphs[kOohQ][0] = 0.01f;
  skin.morphs[kOohQ][3] = 0.04f;
  std::vector<float> follow = {-1, -1, 1}, extension;
  std::vector<uint8_t> labels = {kNoLip, kNoLip, kLowerLip};
  Head head;
  CHECK(MapSkin(mk, g, skin, source, follow, source_follow, labels, extension,
                head));
  head.region_w.assign(head.nv * kNumSkinRegions, 0);
  for (uint32_t v = 0; v < head.nv; ++v)
    head.region_w[v * kNumSkinRegions + kMouth] = 1;
  head.shapes.resize(1);
  head.shapes[0].part = Part::kHead;
  head.shapes[0].nv = head.nv;
  Head baseline = head;
  MapLipControls(g, skin, follow, {}, baseline);
  MapLipControls(g, skin, follow, extension, head);
  CHECK(head.lip_w[0] > 0.2f && head.lip_w[0] < 0.3f);
  CHECK(head.lip_w[2] > 0.99f);
  float scales[kNumRegions] = {1, 1, 1, 1, 1, 1};
  float expression[kNumExpressions] = {};
  std::vector<float> coef(head.k + kNumMouthCoef, 0);
  coef[0] = 1;
  std::vector<std::vector<float>> before, after, stronger;
  Evaluate(baseline, coef.data(), scales, expression, nullptr,
           TongueMode::kAuthored, before);
  Evaluate(head, coef.data(), scales, expression, nullptr,
           TongueMode::kAuthored, after);
  Evaluate(head, coef.data(), scales, expression, nullptr,
           TongueMode::kAuthored, stronger, 2, 1);
  for (size_t i = 0; i < before[0].size(); ++i)
    CHECK(std::fabs(before[0][i] - after[0][i]) < 1e-6f);
  CHECK(stronger[0][2] > after[0][2]);
  CHECK(stronger[0][5] - after[0][5] > 3 * (stronger[0][2] - after[0][2]));
}

int main() {
  Mark mk;
  const float lm[24] = {-3, 8, 4,  3, 8, 4,  -3, 8, 6,  3, 8, 6,
                        0,  8, -3, 0, 9, -2, -2, 9, -2, 2, 9, -2};
  std::copy_n(lm, 24, mk.landmarks);
  mk.k = 1;
  mk.nv = 32;
  for (uint32_t i = 0; i < mk.nv; ++i) {
    mk.verts.insert(mk.verts.end(),
                    {float(i % 8) * .2f - .7f, 9.f, i < 16 ? -1.8f : -2.2f});
    mk.jaw.push_back(i < 16 ? 0.f : -1.f);
    mk.row.push_back(i);
    mk.basis.insert(mk.basis.end(), {0.f, 0.f, i < 16 ? .1f : -1.f});
  }
  Registration g = NativeRegistration(mk);
  g.s = 1;
  g.rot[0] = g.rot[4] = g.rot[8] = 1;
  ShapeIn skin;
  skin.nv = 3;
  skin.skin = true;
  skin.part = Part::kHead;
  skin.verts = {0, 9, -2.2f, 0, 9, -2.2f, 0, 9, 5};
  skin.morphs.assign(kNumMorphs, std::vector<float>(9, 0));
  skin.morphs[kBigAah][5] = -1;
  const auto original_morphs = skin.morphs;
  std::vector<float> hf = {0, 1, -1}, mf(mk.nv, -1);
  // Target-space cropping excludes the entire source upper lip.
  for (uint32_t i = 16; i < mk.nv; ++i) mf[i] = 1;
  std::vector<uint8_t> labels = {kUpperLip, kLowerLip, kNoLip};
  std::vector<float> extension;
  Head repaired;
  CHECK(MapSkin(mk, g, skin, mk.verts, hf, mf, labels, extension, repaired));
  CHECK(!repaired.calibrated_lips.empty());
  CHECK(std::fabs(repaired.basis[repaired.row[0] * 3 + 2] - .1f) < 1e-6f);
  CHECK(std::fabs(repaired.basis[repaired.row[1] * 3 + 2] + 1.f) < 1e-6f);
  // Forehead transfer and the target's authored inputs are unchanged.
  Head clean;
  auto native = JawFollow(g, mk.jaw.data(), 1, mk.nv, mk.verts.data());
  CHECK(MapSkin(mk, g, skin, mk.verts, hf, native, labels, extension, clean));
  CHECK(clean.calibrated_lips.empty());
  for (int j = 0; j < 3; ++j)
    CHECK(clean.basis[clean.row[2] * 3 + j] ==
          repaired.basis[repaired.row[2] * 3 + j]);
  CHECK(skin.morphs == original_morphs);
  CHECK(skin.verts == std::vector<float>({0, 9, -2.2f, 0, 9, -2.2f, 0, 9, 5}));

  // One crossed row must not make an unrelated zero-correction row require
  // a distant replacement candidate, or allocate a new still playback row.
  Registration offset = g;
  offset.t.x = 6;
  auto offset_source = mk.verts;
  for (uint32_t v = 0; v < mk.nv; ++v) offset_source[v * 3] += 6;
  ShapeIn separated = skin;
  separated.verts[0] = 2.4f;
  separated.verts[3] = -2.4f;
  Head separated_head;
  auto separated_follow = hf;
  auto separated_labels = labels;
  CHECK(MapSkin(mk, offset, separated, offset_source, separated_follow, mf,
                separated_labels, extension, separated_head));
  CHECK(separated_head.calibrated_lips == std::vector<uint32_t>{0});
  CHECK(separated_head.basis[separated_head.row[1] * 3 + 2] == -1);
  Mark still_lower = mk;
  std::fill(still_lower.row.begin() + 16, still_lower.row.end(), -1);
  Head still_head;
  CHECK(MapSkin(still_lower, offset, separated, offset_source, separated_follow,
                mf, separated_labels, extension, still_head));
  CHECK(still_head.row[1] == -1);
  CHECK(still_head.calibrated_lips == std::vector<uint32_t>{0});

  Mark missing_upper = mk;
  std::fill(missing_upper.jaw.begin(), missing_upper.jaw.end(), -1.f);
  Head rejected;
  CHECK(!MapSkin(missing_upper, g, skin, mk.verts, hf, mf, labels, extension,
                 rejected));

  // A minor incorrect influence preserves the dominant correct-source motion.
  Mark mixed = mk;
  mixed.verts.insert(mixed.verts.end(), {0, 9, 5});
  mixed.jaw.push_back(0);
  mixed.row.push_back(mixed.nv++);
  mixed.basis.insert(mixed.basis.end(), {1, 0, .1f});
  auto mixed_positions = mixed.verts;
  mixed_positions[mixed_positions.size() - 3] = .05f;
  mixed_positions.back() = -2.2f;
  auto mixed_f = mf;
  mixed_f.push_back(0);
  Head partly_wrong;
  CHECK(MapSkin(mixed, g, skin, mixed_positions, hf, mixed_f, labels, extension,
                partly_wrong));
  CHECK(partly_wrong.basis[partly_wrong.row[0] * 3] > .5f);
  for (uint32_t i = 0; i < 16; ++i) mixed.row[i] = -1;
  Head still_replacement;
  CHECK(MapSkin(mixed, g, skin, mixed_positions, hf, mixed_f, labels, extension,
                still_replacement));
  CHECK(still_replacement.row[0] >= 0);
  CHECK(still_replacement.basis[still_replacement.row[0] * 3] > .5f);

  // Calibration distances scale with the head rather than world units.
  Registration scaled = g;
  scaled.s = scaled.unit = 2;
  for (auto& landmark : scaled.lm) landmark = landmark * 2;
  ShapeIn large = skin;
  for (float& x : large.verts) x *= 2;
  for (auto& morph : large.morphs)
    for (float& x : morph) x *= 2;
  auto large_source = mk.verts;
  for (float& x : large_source) x *= 2;
  Head large_head;
  CHECK(MapSkin(mk, scaled, large, large_source, hf, mf, labels, extension,
                large_head));
  for (size_t i = 0; i < repaired.basis.size(); ++i)
    CHECK(std::fabs(large_head.basis[i] - repaired.basis[i] * 2) < 1e-6f);

  // Pucker support identifies an upper lip beyond the old geometric crop.
  ShapeIn extended = skin;
  extended.nv = 2;
  extended.verts = {0, 9, .1f, 0, 9, -2.2f};
  for (auto& morph : extended.morphs) morph.resize(6);
  extended.morphs[kOohQ][0] = .01f;
  auto shifted_source = mk.verts;
  for (uint32_t i = 0; i < mk.nv; ++i) shifted_source[i * 3 + 2] += 2.3f;
  Head full_upper;
  std::vector<float> extended_follow = {-1, 1};
  std::vector<uint8_t> extended_labels = {kNoLip, kLowerLip};
  CHECK(MapSkin(mk, g, extended, shifted_source, extended_follow, mf,
                extended_labels, extension, full_upper));
  CHECK(std::fabs(full_upper.basis[full_upper.row[0] * 3 + 2] - .1f) < 1e-6f);
  CheckExtendedControls(g, extended, extended_follow, extended_labels,
                        extension, full_upper);
  CheckControlTransition(mk, g, extended, shifted_source, mf);
  CheckAuthoredControlSupport(mk, g, shifted_source, mf);
  CheckSmallConfidenceFollowers(mk, g, extended, mf);
  extended.morphs[kOohQ][0] = 0;
  Head unrelated;
  extended_follow = {-1, 1};
  extended_labels = {kNoLip, kLowerLip};
  CHECK(MapSkin(mk, g, extended, shifted_source, extended_follow, mf,
                extended_labels, extension, unrelated));
  CHECK(unrelated.basis[unrelated.row[0] * 3 + 2] < 0);
  std::puts(
      "lip calibration: identity, partial correction, scale, controls and "
      "followers passed");
}
