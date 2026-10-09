#include "a2f/animation/pose.h"

#include <algorithm>

namespace a2f {

bool PoseTransition::Blend(const std::shared_ptr<const Line>& line, double now,
                           float fade, bool enabled, bool snap,
                           std::vector<float>& coefficients) {
  if (!enabled) {
    *this = PoseTransition();
    std::fill(coefficients.begin(), coefficients.end(), 0.f);
    return false;
  }
  if (pose_.size() != coefficients.size() || stale(now)) {
    pose_.assign(coefficients.size(), 0.f);
    offset_.clear();
    line_.reset();
  }
  if (snap) {
    line_ = line;
    offset_.clear();
  }
  if (line_ != line) {
    line_ = line;
    offset_.resize(coefficients.size());
    for (size_t i = 0; i < coefficients.size(); ++i)
      offset_[i] = pose_[i] - coefficients[i];
    started_at_ = now;
    fade_ = fade;
  }
  // One final rebuild clears the meshes after the offset reaches zero.
  moving_ = !offset_.empty();
  const float weight =
      fade_ > 0 ? 1 - Smoothstep(static_cast<float>(now - started_at_) / fade_)
                : 0;
  if (weight <= 0) offset_.clear();
  for (size_t i = 0; i < offset_.size(); ++i)
    coefficients[i] += weight * offset_[i];
  updated_at_ = now;
  pose_ = coefficients;
  return line != nullptr || !offset_.empty();
}

}  // namespace a2f
