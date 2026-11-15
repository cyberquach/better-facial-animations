#include "a2f/animation/pose.h"

#include <algorithm>
#include <cmath>

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
    share_ = 0;
  }
  if (snap) {
    line_ = line;
    offset_.clear();
  }
  const float target = line ? 1.f : 0.f;
  if (line_ != line) {
    line_ = line;
    offset_.resize(coefficients.size());
    for (size_t i = 0; i < coefficients.size(); ++i)
      offset_[i] = pose_[i] - coefficients[i];
    share_offset_ = share_ - target;
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
  share_ = offset_.empty() ? target : target + weight * share_offset_;
  updated_at_ = now;
  pose_ = coefficients;
  return line != nullptr || !offset_.empty();
}

float PoseTransition::share(double now) const {
  if (stale(now)) return 0;
  const float target = line_ ? 1.f : 0.f;
  if (offset_.empty() || fade_ <= 0) return target;
  const float weight =
      1 - Smoothstep(static_cast<float>(now - started_at_) / fade_);
  return target + weight * share_offset_;
}

bool PoseTransition::resting(double now) const {
  if (stale(now)) return true;
  for (float v : pose_)
    if (std::fabs(v) > 1e-6f) return false;
  return true;
}

}  // namespace a2f
