// a2f::Net on ONNX Runtime (DirectML build) through its C API, loaded at
// runtime from the plugin folder.
#ifndef SRC_ORT_NET_H_
#define SRC_ORT_NET_H_

#include <memory>
#include <string>
#include <vector>

#include "gpu_state.h"
#include "live_core.h"

namespace a2f {

struct OrtOptions {
  // DirectML (adapter 0); false = CPU, 2 threads (no VRAM, ~0.4x realtime).
  bool gpu = true;
  // GPU only. The graphics driver's own versions of the network's operations
  // (metacommands); false: DirectML's shaders, the same on every driver.
  bool metacommands = true;
  // Frames per run; fixed shape (the last batch of a line is padded). 2 =
  // least VRAM.
  int batch = 2;
};

// dll_dir: folder holding onnxruntime.dll / DirectML.dll. info receives a
// one-line description for the log. The session has run one batch when this
// returns: on the GPU all its VRAM (weights and work buffers) is allocated
// and stays allocated until the Net is destroyed.
std::unique_ptr<Net> CreateOrtNet(const std::wstring& dll_dir,
                                  const std::wstring& model,
                                  const OrtOptions& o, std::string& err,
                                  std::string& info);

// Adapter 0 (the one DirectML uses) and this process's video memory there.
GpuState QueryGpu();

// What a bug report about the GPU session needs, one line each: Windows, the
// adapters and their drivers, what Direct3D 12 offers on adapter 0 (the
// driver's metacommands among it) and the modules loaded in the game that
// are not Windows' own, with their versions.
std::vector<std::string> GpuReport();

// True once onnxruntime.dll creates its Direct3D 12 device through the plugin
// (the test checks the import is still there).
bool DeviceRedirected();

// Where onnxruntime's own warnings and errors go (the first 200; any
// thread). Set before the first session.
void SetOrtLog(void (*log)(const std::string&));

}  // namespace a2f

#endif  // SRC_ORT_NET_H_
