// a2f::Net on ONNX Runtime (DirectML build) through its C API, loaded at
// runtime from the plugin folder.
#ifndef INCLUDE_A2F_INFERENCE_ORT_NET_H_
#define INCLUDE_A2F_INFERENCE_ORT_NET_H_

#include <memory>
#include <string>
#include <vector>

#include "a2f/inference/gpu_state.h"
#include "a2f/inference/live_core.h"

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
  // True: independent device/private queue. False: 4.0-style shared device.
  // The caller selects one route; failures go to CPU, never another GPU route.
  bool independent_device = false;
};

// dll_dir: folder holding onnxruntime.dll / DirectML.dll. info receives a
// one-line description for the log. The session has run one batch when this
// returns: on the GPU all its VRAM (weights and work buffers) is allocated
// and stays allocated until the Net is destroyed.
std::unique_ptr<Net> CreateOrtNet(const std::wstring& dll_dir,
                                  const std::wstring& model,
                                  const OrtOptions& o, std::string& err,
                                  std::string& info);

// The folded v3.0 network (tools/a2f3_fold.py), one window per run: o.batch
// is ignored. Otherwise as CreateOrtNet.
std::unique_ptr<Net> CreateOrtNet3(const std::wstring& dll_dir,
                                   const std::wstring& model,
                                   const OrtOptions& o, std::string& err,
                                   std::string& info);

// Fixed-shape Thin120 student; the same GPU/CPU options as the other models.
std::unique_ptr<Net> CreatePerformanceNet(const std::wstring& dll_dir,
                                          const std::wstring& model,
                                          const OrtOptions& o, std::string& err,
                                          std::string& info);
// Audio2Emotion v2.2, fixed 60000-sample windows, warmed before return.
std::unique_ptr<EmotionNet> CreateOrtEmotionNet(const std::wstring& dll_dir,
                                                const std::wstring& model,
                                                const OrtOptions& options,
                                                std::string& err);

// Adapter 0 (the one DirectML uses) and this process's video memory there.
GpuState QueryGpu();

// What a bug report about the GPU session needs, one line each: Windows, the
// adapters and their drivers, what Direct3D 12 offers on adapter 0 (the
// driver's metacommands among it) and the modules loaded in the game that
// are not Windows' own, with their versions. With probe_gpu=false, only
// passive OS/module metadata is collected; no graphics device is created.
std::vector<std::string> GpuReport(bool probe_gpu);

// True once onnxruntime.dll creates its Direct3D 12 device through the plugin
// (the test checks the import is still there).
bool DeviceRedirected();

// Where onnxruntime's own warnings and errors go (the first 200; any
// thread). Set before the first session.
void SetOrtLog(void (*log)(const std::string&));

// Worker only, before creating any session. The marker covers initialization
// and warmup, not later runs. An interrupted attempt makes the next launch CPU.
void SetGpuStartupMarker(const std::wstring& path);
// Shared by both GPU routes and all facial/sentiment models. No driver calls.
bool GpuAllowed(std::string& err);
void DisableGpu(const std::string& reason);

}  // namespace a2f

#endif  // INCLUDE_A2F_INFERENCE_ORT_NET_H_
