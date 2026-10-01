// a2f::Net on ONNX Runtime (DirectML build) through its C API, loaded at
// runtime from the plugin folder.
#ifndef SRC_ORT_NET_H_
#define SRC_ORT_NET_H_

#include <memory>
#include <string>

#include "live_core.h"

namespace a2f {

struct OrtOptions {
  // DirectML (adapter 0); false = CPU, 2 threads (no VRAM, ~0.4x realtime).
  bool gpu = true;
  // Frames per run; fixed shape (the last batch of a line is padded). 2 =
  // least VRAM.
  int batch = 2;
};

// dll_dir: folder holding onnxruntime.dll / DirectML.dll. info receives a
// one-line description for the log.
std::unique_ptr<Net> CreateOrtNet(const std::wstring& dll_dir,
                                  const std::wstring& model,
                                  const OrtOptions& o, std::string& err,
                                  std::string& info);

}  // namespace a2f

#endif  // SRC_ORT_NET_H_
