// Production lip weights and deformation, with controlled skin and followers.
#include <cstdlib>

#include "../src/direct.cc"

#define CHECK(x)                                      \
  do {                                                \
    if (!(x)) {                                       \
      fprintf(stderr, "line %d: %s\n", __LINE__, #x); \
      std::exit(1);                                   \
    }                                                 \
  } while (0)
int main() {
  using namespace direct;
  float w[2];
  const V3 lips{0, 9, -2};
  LipWeights({0, 9, -1.8f}, 0, lips, 0, w);
  CHECK(w[0] == 1 && w[1] == 0);
  LipWeights({0, 9, -2.2f}, 0, lips, 1, w);
  CHECK(w[0] == 0 && w[1] == 1);
  LipWeights(lips, 0, lips, .185f, w);
  CHECK(std::fabs(w[0] - .5f) < 1e-5f && std::fabs(w[1] - .5f) < 1e-5f);
  LipWeights({2.6f, 9, -2}, 0, lips, 0, w);
  CHECK(w[0] == 0 && w[1] == 0);
  LipWeights({2.3f, 9, -2}, 0, lips, 0, w);
  CHECK(w[0] > 0 && w[0] < 1 && w[1] == 0);

  Head h;
  h.k = 1;
  h.nv = 4;
  h.row = {0, 0, 0, 0};
  h.basis = {1, 2, 3};
  h.region_w.assign(4 * kNumSkinRegions, 0);
  for (int v = 0; v < 4; ++v) h.region_w[v * kNumSkinRegions + kMouth] = 1;
  h.lip_w = {1, 0, 0, 1, .5f, .5f, 0, 0};
  Head::Shape skin{}, follower{};
  skin.part = Part::kHead;
  skin.nv = 4;
  skin.expr[0].assign(12, 1);
  follower.part = Part::kPinned;
  follower.nv = 1;
  follower.idx = {0, 0, 0, 0};
  follower.w = {1, 0, 0, 0};
  h.shapes = {skin, follower};
  float scales[kNumRegions] = {1, 1, 1, 1, 1, 1};
  float c[1 + kNumMouthCoef] = {1}, expressions[kNumExpressions] = {};
  std::vector<std::vector<float>> baseline, changed;
  CHECK(Evaluate(h, c, scales, expressions, nullptr, TongueMode::kAuthored, baseline));
  CHECK(Evaluate(h, c, scales, expressions, nullptr, TongueMode::kAuthored, changed, 0, 2));
  CHECK(changed[0][0] == 0 && std::fabs(changed[0][3] - 2.4f) < 1e-5f);
  CHECK(std::fabs(changed[0][6] - 1.2f) < 1e-5f && changed[0][9] == 1);
  CHECK(changed[1][0] == 0);
  CHECK(Evaluate(h, c, scales, expressions, nullptr, TongueMode::kAuthored, changed, 2, 0));
  CHECK(changed[0][0] == 1.6f && changed[0][3] == 0 && changed[1][0] == 1.6f);
  CHECK(Evaluate(h, c, scales, expressions, nullptr, TongueMode::kAuthored, changed, 1, 1));
  CHECK(changed == baseline);
  CHECK(std::fabs(baseline[0][0] - .8f) < 1e-5f);
  CHECK(std::fabs(baseline[0][3] - 1.2f) < 1e-5f);
  CHECK(baseline[0][6] == 1 && baseline[0][9] == 1);
  CHECK(std::fabs(baseline[1][0] - .8f) < 1e-5f);
  expressions[0] = 1;
  CHECK(Evaluate(h, nullptr, scales, expressions, nullptr, TongueMode::kAuthored, baseline));
  CHECK(Evaluate(h, nullptr, scales, expressions, nullptr, TongueMode::kAuthored, changed, 0, 3));
  CHECK(changed == baseline);
  puts(
      "lip weights, independent strengths, neutral baseline and followers: "
      "pass");
}
