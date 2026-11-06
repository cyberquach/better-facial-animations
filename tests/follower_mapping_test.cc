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
