#ifndef INCLUDE_A2F_INFERENCE_WINDOWS_PREREQUISITE_H_
#define INCLUDE_A2F_INFERENCE_WINDOWS_PREREQUISITE_H_

#include <string>

namespace a2f::independent {
struct WindowsVersion {
  unsigned major = 0, minor = 0, build = 0, revision = 0;
  bool known = false, revision_known = false, workstation = false, wine = false;
};
enum class Prerequisite { kUnknown, kTooOld, kMet };

// OS prerequisite only, never a driver capability test or startup gate.
// Microsoft introduction points: KB5029331, KB5029332, KB5029351 (2023-08-22).
// Unmapped releases (including Server/LTSC) stay unknown, not unsupported.
inline Prerequisite CheckWindows(const WindowsVersion& v) {
  if (v.wine || !v.known || !v.workstation || v.major != 10 || v.minor != 0)
    return Prerequisite::kUnknown;
  unsigned minimum = 0;
  switch (v.build) {
    case 19045:
      minimum = 3393;
      break;
    case 22000:
      minimum = 2360;
      break;
    case 22621:
      minimum = 2215;
      break;
    case 22631:
    case 26100:
    case 26200:
      return Prerequisite::kMet;
    default:
      return Prerequisite::kUnknown;
  }
  if (!v.revision_known) return Prerequisite::kUnknown;
  return v.revision >= minimum ? Prerequisite::kMet : Prerequisite::kTooOld;
}
inline std::string WindowsPrerequisiteReport(const WindowsVersion& v) {
  const std::string prefix = "Independent Devices OS prerequisite: ";
  if (v.wine)
    return prefix +
           "unknown (Wine/Proton: reported Windows version does not "
           "establish vkd3d-proton or driver support).";
  switch (CheckWindows(v)) {
    case Prerequisite::kMet:
      return prefix +
             "met (Windows version only; runtime API and driver support "
             "are separate checks).";
    case Prerequisite::kTooOld:
      return prefix +
             "too old for the documented Windows update; install "
             "current Windows cumulative updates and restart. See "
             "GPU-SETUP.txt. "
             "This does not test the loaded runtime or driver.";
    default:
      return prefix +
             "unknown (Windows version/revision unavailable or release "
             "not mapped; this does not mean unsupported).";
  }
}
}  // namespace a2f::independent
#endif
