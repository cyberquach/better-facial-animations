#ifndef SRC_POSE_H_
#define SRC_POSE_H_

#include <memory>
#include <vector>

#include "live_core.h"

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
  bool stale(double now) const { return now - updated_at_ > kStaleSeconds; }

 private:
  static constexpr double kStaleSeconds = 0.5;
  std::shared_ptr<const Line> line_;
  std::vector<float> pose_;
  std::vector<float> offset_;
  double updated_at_ = 0;
  double started_at_ = 0;
  float fade_ = 0;
  bool moving_ = false;
};

}  // namespace a2f

#endif  // SRC_POSE_H_
