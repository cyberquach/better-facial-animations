// Reused searches and local accumulation must preserve follower deformation.
#include <cassert>
#include <cstdio>

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
  // authored. The separate goatee card must still move with the jaw.
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
      assert(std::fabs(movement - expected) < 1e-6f);
    }
    // Missing or malformed topology keeps the ordinary vertex mapping.
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
