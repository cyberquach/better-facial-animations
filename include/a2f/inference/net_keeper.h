// Keeps the network loaded: the wanted session (GPU unless ForceCPU=1) is created
// once and held. CPU is explicit; shared GPU failures retry with backoff.
#ifndef INCLUDE_A2F_INFERENCE_NET_KEEPER_H_
#define INCLUDE_A2F_INFERENCE_NET_KEEPER_H_

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "a2f/inference/gpu_state.h"
#include "a2f/inference/gpu_retry.h"
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
  static constexpr double kRetryMin = 2, kRetryMax = 30;
  static constexpr double kCpuRetry = 30;  // s between CPU session attempts

  NetKeeper(bool want_gpu, Create create, Clock now, Logger log = nullptr,
            bool retry_gpu = false)
      : want_gpu_(want_gpu),
        create_(std::move(create)),
        now_(std::move(now)),
        log_(std::move(log)), retry_gpu_(retry_gpu) {}

  // The session to use now (null: none could be created). Tries the GPU once
  // until an explicit retry for independent mode; shared retries back off.
  Net* Ensure();

  // use(Net&, err) -> bool with the selected session. A lost session is
  // discarded and its selected provider retry policy applies; no_net means
  // no usable session remains. Failures never change providers.
  template <typename Use>
  bool Run(Use&& use, std::string& err, bool& no_net,
           const std::function<bool()>& allowed = {});

  // Worker only, after generation has returned: release session resources.
  void Unload() {
    net_.reset();
    on_gpu_ = false;
    info_ = "not loaded";
    error_ = gpu_error_;
    cpu_retry_at_ = 0;
  }
  // Worker only, once any submitted generation has returned.
  void Reconfigure(bool want_gpu) {
    Unload();
    want_gpu_ = want_gpu;
    gpu_blocked_ = false;
    gpu_error_.clear();
    error_.clear();
    retry_at_ = retry_wait_ = cpu_retry_at_ = 0;
  }
  // Only the explicitly selected provider may be loaded.
  bool Wanted() const { return net_ && (on_gpu_ || !want_gpu_); }
  // Blocked independent attempts sleep until the user requests another attempt.
  bool needs_poll() const {
    return !loaded() && (!want_gpu_ || !gpu_blocked_);
  }
  bool loaded() const { return net_ != nullptr; }
  bool on_gpu() const { return net_ && on_gpu_; }
  int version() const { return net_ ? net_->Version() : 0; }
  // One-line description of the loaded session.
  const std::string& info() const { return info_; }
  // The selected provider's last failure (empty after successful creation).
  const std::string& error() const { return error_; }
  double retry_at() const {
    const double gpu = want_gpu_ && retry_gpu_ && !gpu_blocked_ && !on_gpu_ ? retry_at_ : 0;
    if (!want_gpu_ && !loaded() && cpu_retry_at_ > 0)
      return cpu_retry_at_;
    return gpu;
  }

 private:
  void Log(const std::string& s) {
    if (log_) log_(s);
  }
  void GpuFailed(const std::string& err);
  bool want_gpu_;
  Create create_;
  Clock now_;
  Logger log_;
  bool retry_gpu_ = false, gpu_blocked_ = false;
  std::unique_ptr<Net> net_;
  bool on_gpu_ = false;
  std::string info_ = "not loaded", error_;
  std::string gpu_error_;  // last GPU failure, cleared by an explicit selection
  double retry_at_ = 0, retry_wait_ = 0, cpu_retry_at_ = 0;
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
      err = error_.empty() ? "network unavailable" : error_;
      return false;
    }
    if (use(*n, err)) return true;
    if (!n->failed) return false;
    no_net = true;
    if (on_gpu_) {
      GpuFailed(err);
    } else {
      Log("CPU network lost: " + err);
      cpu_retry_at_ = now_() + kCpuRetry;
      error_ = err;
    }
    net_.reset();
    on_gpu_ = false;
    info_ = "not loaded";
  }
  return false;
}

}  // namespace a2f

#endif  // INCLUDE_A2F_INFERENCE_NET_KEEPER_H_
