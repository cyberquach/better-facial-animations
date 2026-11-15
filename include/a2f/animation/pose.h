#ifndef INCLUDE_A2F_ANIMATION_POSE_H_
#define INCLUDE_A2F_ANIMATION_POSE_H_

#include <memory>
#include <vector>

#include "a2f/inference/live_core.h"

namespace a2f {

// Coefficient-space continuity across line starts, replacements and endings.
// Call only for poses actually submitted to the meshes. No engine or clock
// access.
class PoseTransition {
 public:
  // Samples are already interpolated; null line means a zero speech pose.
  // Modifies coefficients to the final pose. Returns whether speech
  // contributes.
  bool Blend(const std::shared_ptr<const Line>& line, double now, float fade,
             bool enabled, bool snap, std::vector<float>& coefficients);
  bool moving() const { return moving_; }
  void FadeToNeutral(double now, float duration) {
    line_.reset();
    offset_ = pose_;
    share_offset_ = share_;
    started_at_ = updated_at_ = now;
    fade_ = duration;
    moving_ = !offset_.empty();
  }
  bool stale(double now) const { return now - updated_at_ > kStaleSeconds; }
  // Whether the pose a new line would start from is the rest pose.
  bool resting(double now) const;
  // The plugin's share (0..1) of the mouth at now, the engine's vanilla lips
  // being the rest: 1 with a line in, 0 at rest, and in between it follows
  // the pose's transition (blended in, eased out). Blend at now on the same
  // line keeps it.
  float share(double now) const;

 private:
  static constexpr double kStaleSeconds = 0.5;
  std::shared_ptr<const Line> line_;
  std::vector<float> pose_;
  std::vector<float> offset_;
  float share_ = 0;         // of the last pose submitted
  float share_offset_ = 0;  // what offset_ is to pose_
  double updated_at_ = 0;
  double started_at_ = 0;
  float fade_ = 0;
  bool moving_ = false;
};

}  // namespace a2f

#endif  // INCLUDE_A2F_ANIMATION_POSE_H_
