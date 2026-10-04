// Keeps the network loaded: the wanted session (GPU unless GPU=0) is created
// once and held; while it cannot be created, or after it was lost, it is
// retried with a doubling wait and the CPU runs the lines meanwhile.
#ifndef SRC_NET_KEEPER_H_
#define SRC_NET_KEEPER_H_

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gpu_state.h"
#include "live_core.h"

namespace a2f {

// VRAM of the loaded GPU session (FP16, batch 2); less free than this reads
// as "no VRAM left".
constexpr double kNetVramMb = 100;

// Human-readable diagnostics for the settings window.
std::vector<std::string> GpuWarnings(const GpuState& gpu);
std::vector<std::string> DiagnoseNet(bool gpu, const std::string& error,
                                     int attempts, const GpuState& state);

// The graphics driver's metacommands (its own versions of the network's
// operations, which DirectML uses when the driver has them) and the GPU
// session: a driver whose metacommands fail takes the session with it, and
// DirectML's own shaders run the network without them.
struct Metacommands {
  enum State {
    kUntried,  // no GPU session was created yet
    kOn,       // the session loaded with them
    kOff,      // it failed with them and loaded without: they stay off
    kFailed,   // it failed with them and without: they are not the cause
  };
  State state = kUntried;
  // What the session with them failed with (kOff, kFailed).
  std::string error;
};
// One line for the settings window.
std::string DescribeMetacommands(const Metacommands& m);

// A GPU session by create, with the driver's metacommands and, when that
// fails, again without them (at once without, after they failed and the
// session loaded without). On failure returns null with the reason in err:
// the one with metacommands when both were tried. log receives what was
// found out about them.
using CreateGpu = std::function<std::unique_ptr<Net>(
    bool metacommands, std::string& err, std::string& info)>;
std::unique_ptr<Net> CreateGpuNet(
    const CreateGpu& create, Metacommands& meta, std::string& err,
    std::string& info, const std::function<void(const std::string&)>& log);

class NetKeeper {
 public:
  // Creates a session; on failure returns null with the reason in err.
  using Create = std::function<std::unique_ptr<Net>(bool gpu, std::string& err,
                                                    std::string& info)>;
  using Clock = std::function<double()>;  // seconds
  using Logger = std::function<void(const std::string&)>;
  static constexpr double kRetryMin = 2, kRetryMax = 30;

  NetKeeper(bool want_gpu, Create create, Clock now, Logger log = nullptr)
      : want_gpu_(want_gpu),
        create_(std::move(create)),
        now_(std::move(now)),
        log_(std::move(log)) {}

  // The session to use now (null: none could be created). Tries the wanted
  // one when it is missing and its wait is over.
  Net* Ensure();

  // use(Net&, err) -> bool with a session. A session that stops running
  // (Net::failed: the GPU device was lost) is replaced at once and use is
  // called again, once. no_net: it failed for want of a working session.
  template <typename Use>
  bool Run(Use&& use, std::string& err, bool& no_net);

  // The session the settings ask for is loaded (not the CPU fallback).
  bool Wanted() const { return net_ && (on_gpu_ || !want_gpu_); }
  bool loaded() const { return net_ != nullptr; }
  bool on_gpu() const { return net_ && on_gpu_; }
  // One-line description of the loaded session.
  const std::string& info() const { return info_; }
  // The wanted session's last failure (empty: it is loaded), the failed
  // attempts in a row and when the next one is.
  const std::string& error() const { return error_; }
  int attempts() const { return attempts_; }
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
  int attempts_ = 0;
  double retry_at_ = 0, retry_wait_ = 0, cpu_retry_at_ = 0;
};

// Template implementation.
template <typename Use>
bool NetKeeper::Run(Use&& use, std::string& err, bool& no_net) {
  for (int attempt = 0; attempt < 2; ++attempt) {
    Net* n = Ensure();
    no_net = !n;
    if (!n) {
      err = "network unavailable";
      return false;
    }
    if (use(*n, err)) return true;
    if (!n->failed) return false;
    no_net = true;
    Log("network lost (" + err + "): loading it again");
    // the wanted session is tried again at once; a lost CPU fallback
    // leaves the wanted one's wait as it is
    if (Wanted()) retry_at_ = 0;
    net_.reset();
  }
  return false;
}

}  // namespace a2f

#endif  // SRC_NET_KEEPER_H_
