// Reused searches and local accumulation must preserve follower deformation.
#include <cassert>
#include <cstdio>
#include <tuple>

#include "../src/animation/direct.cc"

int main() {
  using namespace direct;
  const std::vector<float> skin = {-1, 0, -1, 1, 0, -1, -1, 0, 1, 1, 0, 1,
                                   -2, 0, -2, 2, 0, -2, -2, 0, 2, 2, 0, 2};
  const std::vector<uint8_t> lips = {kLowerLip, kLowerLip, kUpperLip, kUpperLip,
                                     0,         0,         0,         0};
  std::vector<float> blink(skin.size(), 0.f);
  for (size_t v = 0; v < lips.size(); ++v)
    blink[v * 3 + 2] = skin[v * 3 + 2] > 0 ? -1.f : 1.f;
  ShapeIn shape;
  shape.nv = static_cast<uint32_t>(lips.size());
  shape.verts = skin;
  shape.morphs.resize(kNumMorphs);
  shape.morphs[kBlinkLeft] = blink;
  shape.morphs[kBlinkRight] = blink;
  std::vector<KdTree> face_trees, eye_trees;
  for (Part part :
       {Part::kEyes, Part::kHead, Part::kPinned, Part::kEyes, Part::kHead}) {
    shape.part = part;
    for (float offset : {0.f, 0.1f, -0.1f}) {
      shape.verts = skin;
      for (float& value : shape.verts) value += offset;
      std::vector<KdTree> fresh;
      auto& shared = part == Part::kEyes ? eye_trees : face_trees;
      Head::Shape reference, reused;
      Follow(shape, skin, shape.nv, lips, 0, blink, 0, fresh, reference);
      const auto* storage = shared.empty() ? nullptr : shared[0].order.data();
      Follow(shape, skin, shape.nv, lips, 0, blink, 0, shared, reused);
      assert(reference.idx == reused.idx && reference.w == reused.w);
      assert(!storage || storage == shared[0].order.data());
    }
  }
  // Curved lips need not lie on their side of the global mouth-centre plane.
  // A goatee on a raised lower lip must open with the jaw; a moustache on a
  // drooping upper lip must stay with the upper face. The height-only mapping
  // attaches both of these fixtures to the opposite side.
  for (bool lower : {false, true}) {
    const float z = lower ? 0.2f : -0.2f;
    const uint8_t contact = lower ? kLowerLip : kUpperLip;
    const uint8_t opposite = lower ? kUpperLip : kLowerLip;
    const std::vector<float> curved = {-.1f, 0,  z,   .1f,  0,    z, 0,   -.1f,
                                       z,    0,  .1f, z,    -.1f, 0, -z,  .1f,
                                       0,    -z, 0,   -.1f, -z,   0, .1f, -z};
    const std::vector<uint8_t> labels = {contact,  contact,  contact,
                                         contact,  opposite, opposite,
                                         opposite, opposite};
    ShapeIn attachment;
    attachment.part = Part::kHead;
    attachment.nv = 6;
    attachment.verts = {-.1f, 0, z,  .1f, 0, z,  0, .1f, z,
                        -.1f, 0, -z, .1f, 0, -z, 0, .1f, -z};
    attachment.triangles = {0, 1, 2, 3, 4, 5};
    attachment.morphs.resize(kNumMorphs);
    attachment.morphs[kBigAah].resize(18);
    for (int i = 0; i < 6; ++i)
      attachment.morphs[kBigAah][i * 3 + 2] = (i < 3) == lower ? -1.f : 0.f;
    std::vector<float> opening(24);
    for (int i = 0; i < 8; ++i)
      opening[i * 3 + 2] = labels[i] == kLowerLip ? -1.f : 0.f;
    Head::Shape mapped;
    std::vector<KdTree> trees;
    Follow(attachment, curved, 8, labels, 0, std::vector<float>(24), 0, trees,
           mapped, opening);
    float jaw_motion = 0;
    for (int i = 0; i < kFollow; ++i) {
      const uint32_t vertex = mapped.idx[i];
      assert(labels[vertex] == contact);
      if (labels[vertex] == kLowerLip) jaw_motion -= mapped.w[i];
    }
    assert(std::fabs(jaw_motion - (lower ? -1.f : 0.f)) < 1e-6f);
  }
  // A moustache card's root belongs to the upper lip even if its hanging tip
  // is close to the lower lip. A tip its TRI opens less than half way stays
  // rigid with the root; one opened further spans the mouth and stretches as
  // authored. The separate goatee card must still move with the jaw. (The
  // vertices on a lip take that lip; with two skin vertices a side, the
  // other lip's are among the four nearest and weigh a five-hundredth.)
  for (float tip : {.3f, .6f}) {
    const std::vector<float> skin = {-1, 0, 1, 1, 0, 1, -1, .1f, 1, 1, .1f, 1,
                                     -1, 0, 0, 1, 0, 0, -1, .1f, 0, 1, .1f, 0};
    const std::vector<uint8_t> labels = {kUpperLip, kUpperLip, kUpperLip,
                                         kUpperLip, kLowerLip, kLowerLip,
                                         kLowerLip, kLowerLip};
    std::vector<float> opening(24);
    for (int i = 4; i < 8; ++i) opening[i * 3 + 2] = -1;
    ShapeIn cards;
    cards.part = Part::kHead;
    cards.nv = 6;
    cards.verts = {-1, 0, 1, 1, 0, 1, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, -2};
    cards.morphs.resize(kNumMorphs);
    cards.morphs[kBigAah] = {0, 0, 0,  0, 0, 0,  0, 0, -tip,
                             0, 0, -1, 0, 0, -1, 0, 0, -1};
    cards.triangles = {0, 1, 2, 3, 4, 5};
    std::vector<KdTree> trees;
    Head::Shape mapped;
    Follow(cards, skin, 8, labels, .5f, std::vector<float>(24), 0, trees,
           mapped, opening);
    for (uint32_t i = 0; i < cards.nv; ++i) {
      float movement = 0;
      for (int j = 0; j < kFollow; ++j)
        movement += mapped.w[i * kFollow + j] *
                    opening[mapped.idx[i * kFollow + j] * 3 + 2];
      const float expected = i < 2 ? 0.f : i > 2 ? -1.f : tip < .5f ? 0.f : -tip;
      assert(std::fabs(movement - expected) < (i == 2 ? 1e-6f : 5e-3f));
    }
    // Without usable topology a vertex clear of the skin follows the jaw
    // when its own TRI opens it.
    for (auto indices :
         {std::vector<uint16_t>{}, std::vector<uint16_t>{0, 1, 99}}) {
      cards.triangles = indices;
      Head::Shape fallback;
      Follow(cards, skin, 8, labels, .5f, std::vector<float>(24), 0, trees,
             fallback, opening);
      float movement = 0;
      for (int j = 0; j < kFollow; ++j)
        movement += fallback.w[2 * kFollow + j] *
                    opening[fallback.idx[2 * kFollow + j] * 3 + 2];
      assert(movement < -.99f);
    }
  }
  // A shape without morphs is split at the lips' height, which may lie under
  // the seam. In an unlisted shape a vertex above it still follows the lower
  // lip it lies on, one midway between the lips follows both, and one on the
  // upper lip stays.
  {
    const std::vector<float> skin = {-1, 0, 1, 1, 0, 1, -1, .1f, 1, 1, .1f, 1,
                                     -1, 0, 0, 1, 0, 0, -1, .1f, 0, 1, .1f, 0};
    const std::vector<uint8_t> labels = {kUpperLip, kUpperLip, kUpperLip,
                                         kUpperLip, kLowerLip, kLowerLip,
                                         kLowerLip, kLowerLip};
    std::vector<float> opening(24);
    for (int i = 4; i < 8; ++i) opening[i * 3 + 2] = -4;
    ShapeIn layer;
    layer.part = Part::kHead;
    layer.nv = 3;
    layer.verts = {1, 0, .1f, 1, 0, .5f, 1, 0, .9f};
    layer.unlisted = true;
    std::vector<KdTree> trees;
    Head::Shape mapped;
    Follow(layer, skin, 8, labels, -.5f, std::vector<float>(24), 0, trees,
           mapped, opening);
    const float expected[] = {-4, -2, 0};
    for (uint32_t i = 0; i < layer.nv; ++i) {
      float movement = 0;
      for (int j = 0; j < kFollow; ++j)
        movement += mapped.w[i * kFollow + j] *
                    opening[mapped.idx[i * kFollow + j] * 3 + 2];
      assert(std::fabs(movement - expected[i]) < 1e-5f);
    }
    // In a listed shape the height alone decides.
    layer.unlisted = false;
    Head::Shape kept;
    Follow(layer, skin, 8, labels, -.5f, std::vector<float>(24), 0, trees, kept,
           opening);
    for (size_t j = 0; j < kept.idx.size(); ++j)
      assert(kept.w[j] == 0 || labels[kept.idx[j]] == kUpperLip);
  }
  // A soul patch card is rooted on the lower lip. A vertex lying on the chin
  // moves with the chin under it rather than with the root; a strand end
  // hanging clear of the skin keeps the root's motion.
  {
    const std::vector<float> skin = {
        -1,  0, 1,  1,  0, 1,  -1,  .1f, 1,  1,  .1f, 1,
        -1,  0, 0,  1,  0, 0,  -1,  .1f, 0,  1,  .1f, 0,
        -.5f, 0, -1, .5f, 0, -1, -.5f, .1f, -1, .5f, .1f, -1};
    const std::vector<uint8_t> labels = {
        kUpperLip, kUpperLip, kUpperLip, kUpperLip, kLowerLip, kLowerLip,
        kLowerLip, kLowerLip, 0,         0,         0,         0};
    std::vector<float> opening(36);
    for (int i = 4; i < 8; ++i) opening[i * 3 + 2] = -1;
    for (int i = 8; i < 12; ++i) opening[i * 3 + 2] = -.5f;
    ShapeIn patch;
    patch.part = Part::kHead;
    patch.nv = 4;
    patch.verts = {-.2f, 0, 0, .2f, 0, 0, .5f, .05f, -1, 0, 3, -1};
    patch.morphs.resize(kNumMorphs);
    patch.morphs[kBigAah] = {0, 0, -1, 0, 0, -1, 0, 0, -.5f, 0, 0, -1};
    patch.triangles = {0, 1, 2, 0, 1, 3};
    std::vector<KdTree> trees;
    Head::Shape mapped;
    Follow(patch, skin, 12, labels, .5f, std::vector<float>(36), 0, trees,
           mapped, opening);
    for (uint32_t i = 0; i < patch.nv; ++i) {
      float movement = 0;
      for (int j = 0; j < kFollow; ++j)
        movement += mapped.w[i * kFollow + j] *
                    opening[mapped.idx[i * kFollow + j] * 3 + 2];
      assert(std::fabs(movement - (i == 2 ? -.5f : -1.f)) < 1e-6f);
    }
  }
  // A card rooted on still skin and authored to open at its tip stretches as
  // authored when it lies off the upper lip (a jaw line strand); the same
  // card over the upper lip (a moustache strand) stays rigid. The tip hangs
  // clear of the skin; lying on skin that opens as it does, it would follow
  // that skin in either card.
  for (const bool moustache : {false, true}) {
    const std::vector<float> skin = {
        -.3f, 0, 1, .3f, 0, 1, -.3f, .1f, 1,  .3f, .1f, 1,
        -.2f, 0, 0, .2f, 0, 0, -.2f, .1f, 0,  .2f, .1f, 0,
        -1,   0, -3, 1,  0, -3, -1,  .1f, -3, 1,   .1f, -3};
    std::vector<uint8_t> labels(12, 0);
    for (int i = 0; i < 4; ++i) labels[i] = moustache ? kUpperLip : 0;
    for (int i = 8; i < 12; ++i) labels[i] = kLowerLip;
    std::vector<float> opening(36);
    for (int i = 4; i < 8; ++i) opening[i * 3 + 2] = -.3f;
    for (int i = 8; i < 12; ++i) opening[i * 3 + 2] = -1;
    ShapeIn strand;
    strand.part = Part::kHead;
    strand.nv = 3;
    strand.verts = {-.2f, 0, 1, .2f, 0, 1, 0, .9f, .05f};
    strand.morphs.resize(kNumMorphs);
    strand.morphs[kBigAah] = {0, 0, 0, 0, 0, 0, 0, 0, -.3f};
    strand.triangles = {0, 1, 2};
    std::vector<KdTree> trees;
    Head::Shape mapped;
    Follow(strand, skin, 12, labels, -1.5f, std::vector<float>(36), 0, trees,
           mapped, opening);
    for (uint32_t i = 0; i < strand.nv; ++i) {
      float movement = 0;
      for (int j = 0; j < kFollow; ++j)
        movement += mapped.w[i * kFollow + j] *
                    opening[mapped.idx[i * kFollow + j] * 3 + 2];
      assert(std::fabs(movement - (i == 2 && !moustache ? -.3f : 0.f)) < 1e-5f);
    }
  }
  // A moustache card resting on the lower lip ignores a TRI that opens its
  // tip part of the way. With one vertex of four on the lip (a fringe) it
  // stays rigid with its root; with three of four on the jaw (a strand draped
  // past the mouth corner) those vertices move with the skin under them. A
  // fringe is as rigid when its TRI opens the tip in full, as the lip under
  // it does: following that lip, it would curtain the open mouth. Beside the
  // mouth, on a cheek that opens as that TRI says, the tip goes with the cheek.
  for (const bool cheek : {false, true})
  for (const float tip : {-.7f, -1.f})
  for (const bool draped : {false, true}) {
    const std::vector<float> skin = {-1, 0,   1,  1,  0,   1,  -1, .1f, 1,
                                     1,  .1f, 1,  -1, 0,   0,  1,  0,   0,
                                     -1, .1f, 0,  1,  .1f, 0,  -1, 0,   -1,
                                     1,  0,   -1, -1, .1f, -1, 1,  .1f, -1};
    std::vector<uint8_t> labels = {
        kUpperLip, kUpperLip, kUpperLip, kUpperLip, kLowerLip, kLowerLip,
        kLowerLip, kLowerLip, 0,         0,         0,         0};
    if (cheek) std::fill_n(labels.begin() + 4, 4, 0);
    std::vector<float> opening(36);
    for (int i = 4; i < 12; ++i) opening[i * 3 + 2] = -1;
    ShapeIn strand;
    strand.part = Part::kHead;
    strand.nv = 4;
    strand.verts = {-.9f, 0, 1, -.8f, 0, 1, -1, 0, draped ? -.5f : .7f,
                    -1,   0, 0};
    if (draped) strand.verts[5] = -1;
    strand.morphs.resize(kNumMorphs);
    strand.morphs[kBigAah] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, tip};
    if (draped) strand.morphs[kBigAah][5] = strand.morphs[kBigAah][8] = tip;
    strand.triangles = {0, 1, 2, 1, 2, 3};
    std::vector<KdTree> trees;
    Head::Shape mapped;
    Follow(strand, skin, 12, labels, .5f, std::vector<float>(36), 0, trees,
           mapped, opening);
    for (uint32_t i = 0; i < strand.nv; ++i) {
      float movement = 0;
      for (int j = 0; j < kFollow; ++j)
        movement += mapped.w[i * kFollow + j] *
                    opening[mapped.idx[i * kFollow + j] * 3 + 2];
      // The two on the upper lip take that lip; in so sparse a skin the lower
      // lip's vertices are among their four nearest and weigh a few hundredths.
      // Unlabelled, the ring the tip rests on is also among the root's four.
      const bool carried = cheek && tip == -1 && i == 3;
      assert(std::fabs(movement - ((draped && i) || carried ? -1.f : 0.f)) <
             ((i < 2 && !(draped && i)) || (cheek && !draped && i == 3)
                  ? .05f
                  : 1e-5f));
    }
  }
  // A card rooted on still skin whose lower edge lies on an upper lip that
  // rises as the jaw opens, and whose TRI raises that edge with it: the edge
  // stays on the lip. Rising counts as no opening for the skin as for the
  // hair; with the lip's rise counted against the hair's clamped 0, the edge
  // was held by the root and the lip rose through it.
  {
    std::vector<float> skin, opening;
    std::vector<uint8_t> labels;
    // still skin, the upper lip rising .2, the jaw opening .3, the lower lip
    for (const auto [z, drop, label] : {std::tuple{2.f, 0.f, uint8_t{0}},
                                        {1.f, -.2f, kUpperLip},
                                        {0.f, .3f, uint8_t{0}},
                                        {-3.f, 1.f, kLowerLip}})
      for (const float x : {-.3f, .3f})
        for (const float y : {0.f, .1f}) {
          skin.insert(skin.end(), {x, y, z});
          opening.insert(opening.end(), {0, 0, -drop});
          labels.push_back(label);
        }
    ShapeIn card;
    card.part = Part::kHead;
    card.nv = 4;
    card.verts = {-.2f, 0, 2, .2f, 0, 2, -.3f, .05f, 1, .3f, .05f, 1};
    card.morphs.resize(kNumMorphs);
    card.morphs[kBigAah] = {0, 0, 0, 0, 0, 0, 0, 0, .2f, 0, 0, .2f};
    card.triangles = {0, 1, 2, 1, 2, 3};
    std::vector<KdTree> trees;
    Head::Shape mapped;
    Follow(card, skin, 16, labels, -1.5f, std::vector<float>(48), 0, trees,
           mapped, opening);
    for (uint32_t i = 0; i < card.nv; ++i) {
      float movement = 0;
      for (int j = 0; j < kFollow; ++j)
        movement += mapped.w[i * kFollow + j] *
                    opening[mapped.idx[i * kFollow + j] * 3 + 2];
      assert(std::fabs(movement - (i < 2 ? 0.f : .2f)) < 1e-5f);
    }
  }
  // A beard lying on the cheek and the side of the jaw, where the opening of
  // the skin falls off gradually: every vertex takes the opening of the skin
  // under it, also where its own TRI was fitted to a head that opens a tenth
  // further, and with or without topology. Attached by a threshold on its own
  // opening to the skin that opens most, it drops further than the cheek it
  // lies on and the cheek shows through.
  for (const float stale : {0.f, .1f})
    for (const bool topology : {true, false}) {
      std::vector<float> skin, opening;
      std::vector<uint8_t> labels;
      auto fraction = [](float z) { return std::clamp((2 - z) / 6, 0.f, 1.f); };
      for (int row = 0; row <= 12; ++row)
        for (int column = 0; column <= 4; ++column) {
          const float z = 2 - row * .5f;
          skin.insert(skin.end(), {column * .5f, 0, z});
          opening.insert(opening.end(), {0, 0, -2 * fraction(z)});
          labels.push_back(0);
        }
      const uint32_t skin_count = static_cast<uint32_t>(labels.size());
      ShapeIn beard;
      beard.part = Part::kHead;
      beard.morphs.resize(kNumMorphs);
      for (int row = 0; row < 20; ++row)
        for (int column = 0; column < 2; ++column) {
          const float z = 1.25f - row * .25f;
          beard.verts.insert(beard.verts.end(), {.75f + column * .5f, .1f, z});
          beard.morphs[kBigAah].insert(beard.morphs[kBigAah].end(),
                                       {0, 0, -2 * (fraction(z) + stale)});
          if (!row || !topology) continue;
          const uint16_t i = static_cast<uint16_t>(row * 2 + column);
          beard.triangles.insert(
              beard.triangles.end(),
              {static_cast<uint16_t>(i - 2), static_cast<uint16_t>(i - 1 - column),
               static_cast<uint16_t>(i + 1 - column * 2)});
        }
      beard.nv = static_cast<uint32_t>(beard.verts.size() / 3);
      std::vector<KdTree> trees;
      Head::Shape mapped;
      Follow(beard, skin, skin_count, labels, 3, std::vector<float>(skin.size()),
             0, trees, mapped, opening);
      for (uint32_t i = 0; i < beard.nv; ++i) {
        float movement = 0, weight = 0;
        for (int j = 0; j < kFollow; ++j) {
          movement += mapped.w[i * kFollow + j] *
                      opening[mapped.idx[i * kFollow + j] * 3 + 2];
          weight += mapped.w[i * kFollow + j];
        }
        assert(std::fabs(weight - 1) < 1e-5f);
        assert(std::fabs(-movement / 2 - fraction(beard.verts[i * 3 + 2])) <
               .06f);
      }
    }
  // Exercise the original scalar summation independently, including a zero
  // weight with an unusable index and vertices with no following at all.
  Head head;
  head.nv = 4;
  head.shapes.resize(2);
  head.shapes[0].nv = 4;
  head.shapes[0].expr[0] = {1.25f, -2.5f, 0.3f, -4, 2, 3, 5, -6, 7, 8, 9, -10};
  head.region_w.assign(4 * kNumSkinRegions, 0.f);
  for (int v = 0; v < 4; ++v) head.region_w[v * kNumSkinRegions] = 1;
  auto& follower = head.shapes[1];
  follower.nv = 3;
  follower.idx = {0, 1, 2, 3, 3, 2, UINT32_MAX, 0, 0, 0, 0, 0};
  follower.w = {.1f, .2f, .3f, .4f, .5f, .25f, 0, .25f, 0, 0, 0, 0};
  const float scales[kNumRegions] = {1, 1, 1, 1, 1, 1};
  float expressions[kNumExpressions] = {};
  std::vector<std::vector<float>> out;
  for (Part part : {Part::kHead, Part::kEyes, Part::kPinned}) {
    follower.part = part;
    for (float strength : {.7f, 1.f, 0.f, .3f}) {
      expressions[0] = strength;
      Evaluate(head, nullptr, scales, expressions, nullptr,
               TongueMode::kAuthored, out);
      std::vector<float> expected(follower.nv * 3, 0.f);
      for (uint32_t v = 0; v < follower.nv; ++v)
        for (int k = 0; k < kFollow; ++k) {
          const float weight = follower.w[v * kFollow + k];
          if (weight == 0) continue;
          const uint32_t source = follower.idx[v * kFollow + k];
          for (int c = 0; c < 3; ++c)
            expected[v * 3 + c] += weight * out[0][source * 3 + c];
        }
      assert(out[1] == expected);
    }
  }
  std::puts("follower search reuse and scalar playback equivalence passed");
}
