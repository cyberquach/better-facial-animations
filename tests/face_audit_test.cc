// Calibration diagnostics must not hide changes in the cached playback basis.
#include <cassert>

#include "../src/animation/direct.cc"
#define BFA_AUDIT_CALIBRATION
#include "../tools/face_audit/bridge.cc"

int main() {
  direct::Head head;
  head.nv = 2;
  head.k = 1;
  head.row = {0, 1};
  head.basis = {0, 0, -1, 0, 0, 0};
  head.region_w.assign(2 * direct::kNumSkinRegions, 0);
  head.region_w[direct::kMouth] = 1;
  head.region_w[direct::kNumSkinRegions + direct::kMouth] = 1;
  head.shapes.resize(1);
  head.shapes[0].nv = 2;
  const auto original = fingerprint(&head);
  head.calibrated_lips = {0};
  assert(fingerprint(&head) == original && !has_repair_state(&head));
  uint8_t mask[2] = {};
  repair_mask(&head, mask);
  assert(mask[0] == 1 && mask[1] == 0);
  head.extended_lip_controls = true;
  assert(has_extended_controls(&head));

  auto changed = head;
  changed.basis[2] = 0.2f;
  assert(fingerprint(&changed) != original);
  float scales[direct::kNumRegions] = {1, 1, 1, 1, 1, 1};
  float expressions[direct::kNumExpressions] = {};
  float coefficients[1 + direct::kNumMouthCoef] = {1};
  std::vector<std::vector<float>> before, after;
  direct::Evaluate(head, coefficients, scales, expressions, nullptr,
                   direct::TongueMode::kAuthored, before);
  direct::Evaluate(changed, coefficients, scales, expressions, nullptr,
                   direct::TongueMode::kAuthored, after);
  assert(std::fabs(before[0][2] - after[0][2]) > 1.1f);
  changed = head;
  changed.lip_w = {1, 0, 0, 1};
  assert(fingerprint(&changed) != original);
  changed = head;
  changed.lip_jaw.push_back({0, {0, 0, 0}, 1});
  assert(fingerprint(&changed) != original);
}
