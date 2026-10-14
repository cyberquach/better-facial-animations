// Losing the GPU session and getting it back (include/a2f/inference/net_keeper.h), with a fake
// GPU that fails the way DirectML does.
#include "a2f/inference/net_keeper.h"

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace {
int failures = 0;
void Expect(bool ok, const char* label) {
  printf("%s %s\n", ok ? "ok" : "FAIL", label);
  if (!ok) ++failures;
}
bool Has(const std::string& s, const char* what) {
  return s.find(what) != std::string::npos;
}

// What onnxruntime reports for each way of losing the GPU.
constexpr char kRemoved[] =
    "session: DmlGraphFusionHelper.cpp(906) 887A0005 The GPU device instance "
    "has been suspended.";
constexpr char kHung[] =
    "run: 887A0006 The GPU will not respond to more "
    "commands";
constexpr char kReset[] = "run: 887A0007 The GPU device was reset";
constexpr char kDriver[] =
    "session: 887A0020 The driver encountered a "
    "problem";
constexpr char kOom[] = "session: 8007000E Not enough memory resources";
constexpr char kOther[] = "session: DML api: provider not available";

struct Machine {  // the fake GPU / CPU and clock
  double now = 0;
  const char* gpu_error = nullptr;  // creating a GPU session fails with this
  const char* cpu_error = nullptr;
  const char* run_error = nullptr;  // the next GPU run fails with this
  int gpu_creates = 0, cpu_creates = 0, gpu_runs = 0, cpu_runs = 0;
};

class FakeNet : public a2f::Net {
 public:
  FakeNet(Machine* m, bool gpu) : m_(m), gpu_(gpu) {}
  int Batch() const override { return 2; }
  bool Run(const float*, const float*, int, float*, std::string& err) override {
    if (gpu_ && m_->run_error) {
      err = m_->run_error;
      m_->run_error = nullptr;  // the device comes back for a new session
      failed = true;
      return false;
    }
    ++(gpu_ ? m_->gpu_runs : m_->cpu_runs);
    return true;
  }

 private:
  Machine* m_;
  bool gpu_;
};

a2f::NetKeeper Keeper(Machine* m, bool want_gpu = true) {
  return a2f::NetKeeper(
      want_gpu,
      [m](bool gpu, std::string& err,
          std::string& info) -> std::unique_ptr<a2f::Net> {
        ++(gpu ? m->gpu_creates : m->cpu_creates);
        if (const char* e = gpu ? m->gpu_error : m->cpu_error) {
          err = e;
          return nullptr;
        }
        info = gpu ? "GPU" : "CPU";
        return std::make_unique<FakeNet>(m, gpu);
      },
      [m] { return m->now; });
}

// One line: a single network run.
bool Line(a2f::NetKeeper& k, bool* no_net = nullptr) {
  std::string err;
  bool none = false;
  bool ok = k.Run(
      [](a2f::Net& n, std::string& e) {
        return n.Run(nullptr, nullptr, 2, nullptr, e);
      },
      err, none);
  if (no_net) *no_net = none;
  return ok;
}

// Every returned GPU failure must settle on CPU for this process, even if
// the device later recovers or the user changes models/settings.
void LoadFails(const char* error, const char* name) {
  printf("-- load fails: %s\n", name);
  Machine m;
  m.gpu_error = error;
  auto k = Keeper(&m);
  Expect(k.Ensure() && !k.on_gpu() && !k.Wanted() && k.loaded(), "CPU fallback settles");
  Expect(k.error() == error && k.retry_at() == 0,
         "failure retained without a GPU retry timer");
  for (double now : {2., 30., 120., 1e6}) {
    m.now = now;
    Expect(Line(k), "fallback keeps running");
  }
  m.gpu_error = nullptr;
  k.Unload();
  Expect(k.Ensure() && !k.on_gpu(), "unloading cannot reset GPU failure");
  k.Reconfigure(false);
  Expect(k.Ensure() && k.Wanted() && k.error().empty(), "explicit CPU works");
  k.Reconfigure(true);
  Expect(k.Ensure() && !k.on_gpu() && k.error() == error,
         "switching back cannot reset GPU failure");
  Expect(m.gpu_creates == 1, "exactly one failed GPU creation");
}

void RunFails(const char* error, const char* name) {
  printf("-- device lost mid-line: %s\n", name);
  Machine m;
  auto k = Keeper(&m);
  Expect(k.Ensure() && k.on_gpu() && Line(k), "loaded on GPU");
  m.run_error = error;
  Expect(Line(k) && !k.on_gpu() && m.gpu_creates == 1 && m.cpu_runs == 1,
         "same line retried once on CPU, never on another GPU session");
  Expect(k.error() == error, "run failure remains visible");
  m.now = 1e6;
  k.Reconfigure(true);
  Expect(Line(k) && !k.on_gpu() && m.gpu_creates == 1,
         "run failure survives a model change");
}
}  // namespace

int main() {
  LoadFails(kRemoved, "device removed");
  LoadFails(kHung, "device hung");
  LoadFails(kReset, "device reset");
  LoadFails(kDriver, "driver error");
  LoadFails(kOom, "out of memory");
  LoadFails(kOther, "other");
  RunFails(kRemoved, "device removed");
  RunFails(kHung, "device hung");
  RunFails(kReset, "device reset");
  RunFails(kDriver, "driver error");

  printf("-- no CPU either\n");
  {
    Machine m;
    m.gpu_error = kOom;
    m.cpu_error = "session: cannot load";
    auto k = Keeper(&m);
    bool no_net = false;
    Expect(!k.Ensure() && !k.loaded() && k.info() == "not loaded",
           "nothing loaded");
    Expect(!Line(k, &no_net) && no_net, "the line fails for want of a network");
    m.now = 2;
    k.Ensure();
    Expect(m.gpu_creates == 1 && m.cpu_creates == 1,
           "neither provider retried before CPU backoff");
    m.gpu_error = nullptr;
    m.cpu_error = nullptr;
    m.now = 30;
    Expect(Line(k) && !k.on_gpu() && m.gpu_creates == 1, "CPU recovers without touching GPU");
  }
  printf("-- a failure that is not the session's\n");
  {
    Machine m;
    auto k = Keeper(&m);
    std::string err;
    bool no_net = true;
    Expect(!k.Run(
               [](a2f::Net&, std::string& e) {
                 e = "empty audio";
                 return false;
               },
               err, no_net) &&
               !no_net && err == "empty audio" && m.gpu_creates == 1 &&
               k.on_gpu(),
           "keeps the session and is the line's own");
  }
  printf("-- GPU=0\n");
  {
    Machine m;
    m.gpu_error = kOom;
    auto k = Keeper(&m, false);
    Expect(k.Ensure() && k.Wanted() && !k.on_gpu() && m.gpu_creates == 0 &&
               k.error().empty(),
           "CPU is the wanted session, the GPU is never tried");
  }
  printf("-- diagnosis shown in the settings window\n");
  using a2f::DiagnoseNet;
  a2f::GpuState roomy, full, unknown;
  roomy.known = full.known = true;
  roomy.adapter = full.adapter = "Test GPU";
  roomy.budget_mb = full.budget_mb = 8000;
  roomy.used_mb = 2000;
  full.used_mb = 7950;
  auto first = [](const std::vector<std::string>& d) {
    return d.empty() ? std::string() : d[0];
  };
  auto any = [](const std::vector<std::string>& d, const char* what) {
    for (const auto& l : d)
      if (Has(l, what)) return true;
    return false;
  };
  const std::string removed = kRemoved;
  const auto retained_failure = DiagnoseNet(true,
      "independent context: [DirectML removed reason 887A0020]; "
      "GPU disabled until restart", roomy);
  Expect(any(retained_failure, "CPU fallback") &&
             any(retained_failure, "Restart the game") &&
             !any(retained_failure, "each on a new device"),
         "retained failed context requires restart, not device recreation");
  Expect(any(DiagnoseNet(true, kOom, roomy), "restart the game"),
         "out of memory: advised to free VRAM and restart");
  Expect(Has(first(DiagnoseNet(true, kOom, roomy)), "memory allocation failed"),
         "out of memory: reports allocation failure");
  Expect(Has(first(DiagnoseNet(true, kRemoved, full)),
             "graphics device was lost (50 MB free of 8000 MB") &&
             any(DiagnoseNet(true, kRemoved, full), "likely cause"),
         "low memory does not misclassify device loss as proven OOM");
  Expect(any(DiagnoseNet(true, kRemoved, full), "no device-removal reason"),
         "absent device query does not claim a healthy D3D12 device");
  Expect(any(DiagnoseNet(true, removed + " [DirectML removed reason 887A0020]", roomy),
             "DirectML reported device failure reason 887A0020"),
         "reason from DirectML alone does not assert the original cause");
  for (const char* e : {kRemoved, kHung, kReset, kDriver})
    Expect(
        Has(first(DiagnoseNet(true, e, roomy)),
            "graphics device was lost (6000 MB free") &&
            Has(first(DiagnoseNet(true, e, unknown)), "free VRAM unknown"),
        "device lost with VRAM free: device lost");
  Expect(any(DiagnoseNet(true, removed + " [device removed reason 887A0006]",
                         roomy),
             "stopped responding"),
         "reason from Windows: hung");
  Expect(any(DiagnoseNet(true, removed + " [device removed reason 887A0007]",
                         roomy),
             "driver was reset"),
         "reason from Windows: reset");
  Expect(any(DiagnoseNet(true, removed + " [device removed reason 887A0020]",
                         roomy),
             "Update the graphics driver"),
         "reason from Windows: driver error");
  Expect(any(DiagnoseNet(true, removed + " [device removed reason 887A0005]",
                         roomy),
             "removed or disabled"),
         "reason from Windows: removed");
  Expect(first(DiagnoseNet(true, removed + " [device removed reason 8007000E]", roomy))
                 .find("memory allocation failed") != std::string::npos,
         "reason from Windows: out of memory");
  Expect(any(DiagnoseNet(true, kRemoved, roomy),
             "Restart the game to retry the GPU") &&
             !any(DiagnoseNet(true, kRemoved, roomy), "each on a new device"),
         "diagnosis does not promise retries or assign an unproven cause");
  Expect(Has(first(DiagnoseNet(true, kOther, roomy)),
             "DirectML is not available") &&
             any(DiagnoseNet(true, kOther, roomy), "Independent Devices API"),
         "DirectML missing: explains the independent-device requirement");
  Expect(Has(first(DiagnoseNet(true, "session: bad graph", roomy)),
             "did not load (6000 MB free"),
         "other: did not load");
  Expect(Has(first(DiagnoseNet(true, "model file not found: x.onnx", roomy)),
             "model file is missing (model file not found: x.onnx)"),
         "missing model file");
  Expect(any(DiagnoseNet(true, "cannot load onnxruntime.dll (error 126)",
                         roomy),
             "Reinstall the mod") &&
             any(DiagnoseNet(true, "onnxruntime.dll: wrong version", roomy),
                 "two mod versions"),
         "missing / wrong onnxruntime.dll");
  Expect(Has(first(DiagnoseNet(false, "x", unknown)), "on the CPU"),
         "CPU failure");
  Expect(a2f::GpuWarnings(roomy).empty() && a2f::GpuWarnings(unknown).empty(),
         "healthy GPU: no warnings");
  a2f::GpuState laptop = roomy;
  laptop.adapter = "Integrated";
  laptop.dedicated_mb = 128;
  laptop.bigger_adapter = "Discrete";
  laptop.bigger_mb = 8192;
  Expect(any(a2f::GpuWarnings(laptop),
             "model runs on \"Integrated\" (128 MB)") &&
             any(a2f::GpuWarnings(laptop), "high-performance GPU"),
         "model adapter diagnosis names the bigger adapter and the fix");
  a2f::GpuState over = roomy;
  over.used_mb = 8500;
  Expect(any(a2f::GpuWarnings(over), "500 MB over the 8000 MB"),
         "warns when over the VRAM budget");
  printf(failures ? "%d FAILED\n" : "ok: 0 failures\n", failures);
  return failures != 0;
}
