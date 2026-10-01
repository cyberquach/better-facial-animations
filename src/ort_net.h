// a2f::Net on ONNX Runtime (DirectML build) through its C API, loaded at runtime from the plugin folder.
#pragma once
#include "live_core.h"
#include <memory>

namespace a2f {

struct OrtOptions {
    bool gpu = true;     // DirectML; false = CPU (no VRAM at all, ~0.4x realtime)
    int device = 0;      // DXGI adapter index
    int batch = 2;       // frames per run; fixed shape (the last batch of a line is padded). 2 = least VRAM
    int cpuThreads = 2;  // CPU execution provider threads
};

// dllDir: folder holding onnxruntime.dll / DirectML.dll. info receives a one-line description for the log.
std::unique_ptr<Net> CreateOrtNet(const std::wstring& dllDir, const std::wstring& model, const OrtOptions& o,
                                  std::string& err, std::string& info);

}  // namespace a2f
