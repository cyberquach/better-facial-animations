// Keeps the network loaded: the wanted session (GPU unless ForceCPU=1) is created
// once and held. A GPU failure selects CPU for the rest of this process.
#ifndef INCLUDE_A2F_INFERENCE_NET_KEEPER_H_
#define INCLUDE_A2F_INFERENCE_NET_KEEPER_H_

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "a2f/inference/gpu_state.h"
#include "a2f/inference/live_core.h"

namespace a2f {

// VRAM of the loaded GPU session (FP16; v2.3 at batch 2, v3.0 and the
// performance network); less free than this is a possible memory-pressure clue.
constexpr double kNetVramMb = 100;
constexpr double kNetVramMb3 = 250;
constexpr double kNetVramMbPerformance = 80;

// Human-readable diagnostics for the settings window.
std::vector<std::string> GpuWarnings(const GpuState& gpu);
// need_mb: the VRAM the session takes.
std::vector<std::string> DiagnoseNet(bool gpu, const std::string& error,
                                     const GpuState& state,
                                     double need_mb = kNetVramMb);

class NetKeeper {
 public:
  // Creates a session; on failure returns null with the reason in err.
  using Create = std::function<std::unique_ptr<Net>(bool gpu, std::string& err,
                                                    std::string& info)>;
  using Clock = std::function<double()>;  // seconds
  using Logger = std::function<void(const std::string&)>;
  static constexpr double kCpuRetry = 30;  // s between CPU session attempts

  NetKeeper(bool want_gpu, Create create, Clock now, Logger log = nullptr)
      : want_gpu_(want_gpu),
        create_(std::move(create)),
        now_(std::move(now)),
        log_(std::move(log)) {}

  // The session to use now (null: none could be created). Tries the GPU once
  // per process and the CPU every kCpuRetry while it fails.
  Net* Ensure();

  // use(Net&, err) -> bool with a session. A session that stops running
  // (Net::failed: the GPU device was lost) is replaced at once and use is
  // called again, once. no_net: it failed for want of a working session.
  template <typename Use>
  bool Run(Use&& use, std::string& err, bool& no_net,
           const std::function<bool()>& allowed = {});

  // Worker only, after generation has returned: release session resources.
  void Unload() {
    net_.reset();
    on_gpu_ = false;
    info_ = "not loaded";
    error_ = gpu_error_;
    retry_at_ = cpu_retry_at_ = 0;
  }
  // Worker only, once any submitted generation has returned.
  void Reconfigure(bool want_gpu) {
    Unload();
    want_gpu_ = want_gpu;
  }
  // The session the settings ask for is loaded (not the CPU fallback).
  bool Wanted() const { return net_ && (on_gpu_ || !want_gpu_); }
  // A loaded session is final, the CPU fallback too: nothing to poll for.
  bool loaded() const { return net_ != nullptr; }
  bool on_gpu() const { return net_ && on_gpu_; }
  int version() const { return net_ ? net_->Version() : 0; }
  // One-line description of the loaded session.
  const std::string& info() const { return info_; }
  // The wanted session's last failure (empty: it is loaded; with the CPU's
  // after "; CPU: " while that fails too) and the next CPU retry (zero when
  // CPU is working).
  const std::string& error() const { return error_; }
  double retry_at() const { return retry_at_; }

 private:
  void Log(const std::string& s) {
    if (log_) log_(s);
  }
  bool want_gpu_;
  Create create_;
  Clock now_;
  Logger log_;
  std::unique_ptr<Net> net_;
  bool on_gpu_ = false;
  std::string info_ = "not loaded", error_;
  std::string gpu_error_;  // survives Unload/Reconfigure; restart to retry
  double retry_at_ = 0, cpu_retry_at_ = 0;
};

// Template implementation.
template <typename Use>
bool NetKeeper::Run(Use&& use, std::string& err, bool& no_net,
                    const std::function<bool()>& allowed) {
  for (int attempt = 0; attempt < 2; ++attempt) {
    if (allowed && !allowed()) return false;
    Net* n = Ensure();
    no_net = !n;
    if (!n) {
      err = "network unavailable";
      return false;
    }
    if (use(*n, err)) return true;
    if (!n->failed) return false;
    no_net = true;
    if (on_gpu_) {
      gpu_error_ = error_ = err;
      retry_at_ = 0;
      Log("GPU disabled until restart: " + err);
    } else {
      Log("CPU network lost: " + err);
    }
    net_.reset();
  }
  return false;
}

}  // namespace a2f

#endif  // INCLUDE_A2F_INFERENCE_NET_KEEPER_H_
