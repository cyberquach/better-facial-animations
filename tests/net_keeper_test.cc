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

a2f::NetKeeper Keeper(Machine* m, bool want_gpu = true, bool retry_gpu = false) {
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
      [m] { return m->now; }, nullptr, retry_gpu);
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

// Failure never selects CPU; the user can explicitly choose either provider.
void LoadFails(const char* error, const char* name) {
  printf("-- load fails: %s\n", name);
  Machine m;
  m.gpu_error = error;
  auto k = Keeper(&m);
  Expect(!k.Ensure() && !k.loaded() && !k.needs_poll(), "GPU failure leaves vanilla, no automatic CPU");
  Expect(k.error() == error && k.retry_at() == 0, "error retained without idle retries");
  for (double now : {2., 30., 120., 1e6}) {
    m.now = now;
    Expect(!Line(k) && m.cpu_creates == 0, "time and speech never select CPU");
  }
  m.gpu_error = nullptr;
  k.Unload();
  Expect(!k.Ensure(), "ordinary unload does not retry a failed attempt");
  k.Reconfigure(false);
  Expect(k.Ensure() && k.Wanted() && k.error().empty(), "explicit CPU works");
  k.Reconfigure(true);
  Expect(k.Ensure() && k.on_gpu() && k.error().empty(), "CPU to GPU works in the same process");
  Expect(m.gpu_creates == 2 && m.cpu_creates == 1, "only selected providers created");
}

void RunFails(const char* error, const char* name) {
  printf("-- device lost mid-line: %s\n", name);
  Machine m;
  auto k = Keeper(&m);
  Expect(k.Ensure() && k.on_gpu() && Line(k), "loaded on GPU");
  m.run_error = error;
  Expect(!Line(k) && !k.loaded() && m.gpu_creates == 1 && m.cpu_creates == 0,
         "lost GPU leaves vanilla, never replays on CPU");
  Expect(k.error() == error, "run failure remains visible");
  k.Reconfigure(true);
  Expect(Line(k) && k.on_gpu() && m.gpu_creates == 2,
         "manual GPU retry works without a restart");
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

  printf("-- shared GPU backoff and recovery\n");
  {
    Machine m;
    m.gpu_error = kOom;
    auto k = Keeper(&m, true, true);
    Expect(!k.Ensure() && !k.loaded() && k.needs_poll(), "shared GPU retry without CPU fallback");
    for (double wait : {2., 4., 8., 16., 30., 30.}) {
      const int before = m.gpu_creates;
      Expect(k.retry_at() == m.now + wait, "exponential GPU retry deadline, capped at 30 seconds");
      m.now = k.retry_at() - .01;
      Expect(!Line(k) && m.gpu_creates == before, "no provider change before GPU deadline");
      m.now += .01;
      Expect(!k.Ensure() && !k.loaded() && m.gpu_creates == before + 1,
             "shared GPU retried without CPU creation");
    }
    Expect(m.cpu_creates == 0, "GPU attempts never create CPU sessions");
    m.gpu_error = nullptr;
    m.now = k.retry_at();
    Expect(k.Ensure() && k.on_gpu() && !k.needs_poll() && k.error().empty() && !k.retry_at(),
           "shared GPU recovers and stops idle polling");
    m.run_error = kRemoved;
    Expect(!Line(k) && !k.loaded(), "run failure leaves vanilla during backoff");
    m.now = k.retry_at();
    Expect(Line(k) && k.on_gpu(), "lost shared session recovers on GPU");
  }
  {
    Machine m;
    m.gpu_error = "device error; GPU attempt failed";
    auto k = Keeper(&m, true, true);
    Expect(!k.Ensure() && !k.needs_poll(), "returned terminal attempt stops idle retries");
    m.gpu_error = nullptr;
    k.Reconfigure(true);
    Expect(Line(k) && m.gpu_creates == 2 && m.cpu_creates == 0, "explicit retry clears attempt state");
  }
  {
    Machine m;
    m.gpu_error = kOom;
    auto k = Keeper(&m, true, true);
    k.Ensure();
    k.Reconfigure(false);
    m.now = 1e6;
    Expect(k.Ensure() && k.Wanted() && !k.needs_poll() && !k.retry_at(), "Force CPU suspends shared retry");
    m.gpu_error = nullptr;
    k.Reconfigure(true);
    Expect(k.Ensure() && k.on_gpu(), "explicit GPU opt-in resumes shared recovery");
  }

  printf("-- explicitly selected CPU retry\n");
  {
    Machine m;
    m.cpu_error = "session: cannot load";
    auto k = Keeper(&m, false);
    Expect(!k.Ensure() && m.gpu_creates == 0 && m.cpu_creates == 1, "CPU failure never tries GPU");
    m.now = 2;
    Expect(!k.Ensure() && m.cpu_creates == 1, "CPU backoff");
    m.cpu_error = nullptr;
    m.now = 30;
    Expect(Line(k) && !k.on_gpu() && m.gpu_creates == 0, "selected CPU recovers");
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
      "GPU attempt failed", roomy);
  Expect(any(retained_failure, "Vanilla lip-sync") &&
             any(retained_failure, "Try GPU again") &&
             !any(retained_failure, "each on a new device"),
         "failed attempt offers explicit GPU retry");
  Expect(any(DiagnoseNet(true, kOom, roomy), "Try GPU again"),
         "out of memory: explains shared retry and independent restart");
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
             "Shared GPU retries automatically") &&
             !any(DiagnoseNet(true, kRemoved, roomy), "each on a new device"),
         "diagnosis distinguishes shared recovery without assigning an unproven cause");
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
