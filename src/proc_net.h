// a2f::Net in a process of its own (BetterFacialAnimationsModel.exe, in the
// plugin folder): what is loaded into the game and hooks its Direct3D
// (ReShade, ENB, overlays, other mods) is not in that process, so it cannot
// break the session, and a session that dies leaves nothing behind in the
// game. The process ends with the game, however the game ends.
#ifndef SRC_PROC_NET_H_
#define SRC_PROC_NET_H_

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "live_core.h"
#include "ort_net.h"

namespace a2f {

// Receives the process's log lines (onnxruntime's warnings and errors).
using ProcLog = std::function<void(const std::string&)>;

// CreateOrtNet in a new process started from dir. started: false when the
// process itself did not come up (err says why): the session was not tried.
std::unique_ptr<Net> CreateProcNet(const std::wstring& dir,
                                   const std::wstring& model,
                                   const OrtOptions& o, const ProcLog& log,
                                   bool& started, std::string& err,
                                   std::string& info);

// DeviceReport and ModuleReport of such a process, in lines. False when it
// did not come up, with the reason in err.
bool ProcReport(const std::wstring& dir, std::vector<std::string>& lines,
                std::string& err);

}  // namespace a2f

#endif  // SRC_PROC_NET_H_
