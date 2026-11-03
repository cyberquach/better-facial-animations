#ifndef INCLUDE_A2F_INFERENCE_DIAGNOSTICS_H_
#define INCLUDE_A2F_INFERENCE_DIAGNOSTICS_H_

#include <windows.h>

#include <algorithm>
#include <array>
#include <string>
#include <utility>
#include <vector>

#include "a2f/runtime/scoped_lock.h"

namespace a2f::diagnostics {

enum class Component { kFace, kSentiment };
enum class Stage {
  kModelFiles,
  kRuntime,
  kRuntimeApi,
  kEnvironment,
  kAdapter,
  kFactory,
  kFlags,
  kDevice,
  kDirectML,
  kQueue,
  kOptions,
  kProvider,
  kSession,
  kMemory,
  kWarmup,
  kInference
};
inline const char* Name(Stage stage) {
  static const char* names[] = {"model files",
                                "runtime DLL loading",
                                "runtime API",
                                "runtime environment",
                                "adapter selection",
                                "independent device factory",
                                "isolation flags",
                                "D3D12 device creation",
                                "DirectML device",
                                "command queue",
                                "session options",
                                "execution provider",
                                "model session compilation",
                                "memory information",
                                "model warmup",
                                "inference"};
  return names[static_cast<unsigned>(stage)];
}
inline std::string Bounded(const std::string& text) {
  return text.size() <= 16384 ? text : text.substr(0, 16384) + " [truncated]";
}
struct Failure {
  unsigned attempt = 0;
  Stage stage = Stage::kModelFiles;
  std::string detail;
  bool gpu = false, independent = false;
};
struct Status {
  unsigned attempt = 0;
  bool loading = false, ready = false, gpu = false, independent = false;
  Stage stage = Stage::kModelFiles;
  ULONGLONG started = 0, stage_started = 0, finished = 0;
  std::string error;
  Failure last_failure;
};
struct Snapshot {
  std::array<Status, 2> components;
  std::vector<std::string> system;
  std::string independent_api =
      "not checked (no independent factory query yet)";
};

// A bounded reporting snapshot, separate from model counters and recovery
// policy. Never call a driver, logger or UI while holding this lock.
class Store {
 public:
  Snapshot Read() {
    bfa::ScopedLock guard(lock_, bfa::ScopedLock::kShared);
    return value_;
  }
  void Begin(Component component, bool gpu, bool independent) {
    bfa::ScopedLock guard(lock_);
    auto& s = value_.components[static_cast<unsigned>(component)];
    ++s.attempt;
    s.loading = true;
    s.ready = false;
    s.gpu = gpu;
    s.independent = independent;
    s.stage = Stage::kModelFiles;
    s.started = s.stage_started = GetTickCount64();
    s.finished = 0;
    s.error.clear();
  }
  void Progress(Component component, Stage stage) {
    bfa::ScopedLock guard(lock_);
    auto& s = value_.components[static_cast<unsigned>(component)];
    if (!s.loading) return;
    s.stage = stage;
    s.stage_started = GetTickCount64();
  }
  void Complete(Component component, bool ready, const std::string& error) {
    bfa::ScopedLock guard(lock_);
    auto& s = value_.components[static_cast<unsigned>(component)];
    s.loading = false;
    s.ready = ready;
    s.finished = GetTickCount64();
    s.error =
        ready ? ""
              : Bounded(error.empty() ? "No runtime detail returned" : error);
    if (!ready)
      s.last_failure = {s.attempt, s.stage, s.error, s.gpu, s.independent};
  }
  void RuntimeFailure(Component component, const std::string& error) {
    bfa::ScopedLock guard(lock_);
    auto& s = value_.components[static_cast<unsigned>(component)];
    s.ready = s.loading = false;
    s.stage = Stage::kInference;
    s.finished = GetTickCount64();
    s.error = Bounded(error);
    s.last_failure = {s.attempt, s.stage, s.error, s.gpu, s.independent};
  }
  void Unload(Component component) {
    bfa::ScopedLock guard(lock_);
    auto& s = value_.components[static_cast<unsigned>(component)];
    s.ready = false;
  }
  void System(std::vector<std::string> report) {
    report.resize((std::min)(report.size(), size_t{64}));
    for (auto& line : report) line = Bounded(line);
    bfa::ScopedLock guard(lock_);
    value_.system = std::move(report);
  }
  void IndependentApi(const std::string& result) {
    bfa::ScopedLock guard(lock_);
    value_.independent_api = Bounded(result);
  }

 private:
  SRWLOCK lock_ = SRWLOCK_INIT;
  Snapshot value_;
};
inline Store& State() {
  static auto* state = new Store;
  return *state;
}
// Model startup is serial on the inference worker; ORT logger threads do not
// publish stages.
inline thread_local Component current_component = Component::kFace;
inline void Progress(Stage stage) {
  State().Progress(current_component, stage);
}
class Attempt {
 public:
  Attempt(Component component, bool gpu, bool independent,
          const std::string& error)
      : component_(component), previous_(current_component), error_(error) {
    current_component = component;
    State().Begin(component, gpu, independent);
  }
  ~Attempt() {
    State().Complete(component_, ready_, error_);
    current_component = previous_;
  }
  void Ready() { ready_ = true; }

 private:
  Component component_, previous_;
  const std::string& error_;
  bool ready_ = false;
};

}  // namespace a2f::diagnostics
#endif
