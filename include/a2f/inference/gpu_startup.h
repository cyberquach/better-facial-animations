// Worker-only, crash-persistent guard around GPU session creation and warmup.
#ifndef INCLUDE_A2F_INFERENCE_GPU_STARTUP_H_
#define INCLUDE_A2F_INFERENCE_GPU_STARTUP_H_

#include <windows.h>
#include <string>
#include <utility>

namespace a2f {
class GpuStartup {
 public:
  explicit GpuStartup(std::wstring path) : path_(std::move(path)) {}
  bool Begin(std::string& err) {
    // CREATE_NEW is deliberate: an unfinished attempt from another launch
    // (or process) must never be overwritten. Fail closed on IO errors too.
    HANDLE file = CreateFileW(path_.c_str(), GENERIC_WRITE, 0, nullptr,
                              CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
      err = "GPU startup guard exists or cannot be written (error " +
            std::to_string(GetLastError()) + "). Using CPU. After closing "
            "Skyrim, remove BetterFacialAnimations/gpu-startup.pending "
            "only if you want to try GPU startup again";
      return false;
    }
    const char message[] =
        "GPU session startup did not finish. CPU is selected on the next "
        "launch. Close Skyrim before deleting this file to retry.\r\n";
    DWORD written = 0;
    const bool ok = WriteFile(file, message, sizeof(message) - 1, &written,
                              nullptr) && written == sizeof(message) - 1 &&
                    FlushFileBuffers(file);
    CloseHandle(file);
    if (!ok) {
      err = "cannot flush GPU startup guard; using CPU";
      // Leave the file present: recovery protection could not be established.
      return false;
    }
    armed_ = true;
    return true;
  }
  bool Complete(std::string& err) {
    if (!armed_) return true;
    armed_ = false;
    if (DeleteFileW(path_.c_str())) return true;
    err = "cannot clear gpu-startup.pending; using CPU. Close Skyrim and "
          "remove the file before trying GPU again";
    return false;
  }
  // No destructor cleanup: an interrupted attempt must remain on disk.
 private:
  std::wstring path_;
  bool armed_ = false;
};
}  // namespace a2f
#endif
