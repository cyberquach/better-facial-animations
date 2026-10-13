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

// The GPU cannot load the model at the main menu for `error`; it can later.
void LoadFails(const char* error, const char* name) {
  printf("-- load fails: %s\n", name);
  Machine m;
  m.gpu_error = error;
  auto k = Keeper(&m);
  Expect(k.Ensure() && !k.on_gpu() && !k.Wanted(), "CPU fallback is loaded");
  Expect(k.error() == error && k.attempts() == 1 && k.retry_at() == 2,
         "failure reported, retry in 2 s");
  Expect(Line(k) && m.cpu_runs == 1, "lines run on the CPU meanwhile");
  m.now = 1.9;
  k.Ensure();
  Expect(m.gpu_creates == 1, "no retry before its time");
  // 2, 4, 8, 16, 30, 30 s apart
  const double at[] = {2, 6, 14, 30, 60, 90};
  for (int i = 0; i < 6; ++i) {
    m.now = at[i];
    k.Ensure();
  }
  Expect(m.gpu_creates == 7 && k.attempts() == 7 && k.retry_at() == 120,
         "retries back off to every 30 s and never stop");
  Expect(m.cpu_creates == 1, "the CPU session is created once");
  m.gpu_error = nullptr;
  m.now = 120;
  Expect(k.Ensure() && k.on_gpu() && k.Wanted(), "GPU session takes over");
  Expect(k.error().empty() && k.attempts() == 0 && k.info() == "GPU",
         "error cleared");
  Expect(Line(k) && m.gpu_runs == 1 && m.cpu_runs == 1,
         "lines run on the GPU again");
  m.now = 1e6;
  k.Ensure();
  Expect(m.gpu_creates == 8 && k.on_gpu(), "loaded session is kept for good");
}

// The device is lost during a line for `error`.
void RunFails(const char* error, const char* name) {
  printf("-- device lost mid-line: %s\n", name);
  {
    Machine m;
    auto k = Keeper(&m);
    Expect(k.Ensure() && k.on_gpu() && Line(k), "loaded on the GPU");
    m.run_error = error;
    Expect(Line(k) && k.on_gpu() && m.gpu_creates == 2 && m.gpu_runs == 2 &&
               m.cpu_creates == 0,
           "new GPU session at once, the line is generated");
    Expect(k.error().empty(), "no error left");
  }
  {
    Machine m;
    auto k = Keeper(&m);
    k.Ensure();
    m.run_error = error;
    m.gpu_error = kRemoved;  // and it stays lost
    Expect(Line(k) && !k.on_gpu() && m.cpu_runs == 1,
           "GPU stays lost: the line is generated on the CPU");
    Expect(k.error() == kRemoved && k.attempts() == 1, "failure reported");
    m.gpu_error = nullptr;
    m.now = 2;
    Expect(Line(k) && k.on_gpu() && m.gpu_runs == 1 && k.error().empty(),
           "back on the GPU at the retry");
  }
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
    Expect(m.gpu_creates == 2 && m.cpu_creates == 1,
           "GPU retried, the CPU not at every retry");
    m.gpu_error = nullptr;
    m.now = 6;
    Expect(Line(k) && k.on_gpu(), "GPU recovers without the CPU");
  }
  printf("-- driver metacommands\n");
  {
    // What a GPU whose driver's metacommands fail reported (RTX 3080).
    const std::string kMeta =
        "session: Exception during initialization: E:\\_work\\1\\s\\"
        "onnxruntime\\core\\providers\\dml\\DmlExecutionProvider\\src\\"
        "DmlGraphFusionHelper.cpp(906)\\onnxruntime.dll!00007FFB3F1F833A: "
        "(caller: 00007FFB3F284FEE) Exception(12) tid(18ac) 887A0005 The GPU "
        "device instance has been suspended.";
    std::string with_error, without_error;  // empty: that session loads
    int with = 0, without = 0;
    std::vector<std::string> said;
    a2f::CreateGpu create = [&](bool meta, std::string& err, std::string& info)
        -> std::unique_ptr<a2f::Net> {
      ++(meta ? with : without);
      const std::string& e = meta ? with_error : without_error;
      if (!e.empty()) {
        err = e;
        return nullptr;
      }
      info = meta ? "on" : "off";
      static Machine machine;
      return std::make_unique<FakeNet>(&machine, true);
    };
    auto log = [&](const std::string& s) { said.push_back(s); };
    auto make = [&](a2f::Metacommands& m, std::string& err, std::string& info) {
      return a2f::CreateGpuNet(create, m, err, info, log);
    };
    std::string err, info;
    {
      a2f::Metacommands m;
      Expect(Has(a2f::DescribeMetacommands(m), "not tried"), "untried at first");
      Expect(make(m, err, info) && info == "on" && with == 1 && without == 0 &&
                 m.state == a2f::Metacommands::kOn,
             "a healthy driver: loaded with them, nothing tried without");
      Expect(said.size() == 1 && Has(said[0], "metacommands work"),
             "logged that they work");
      Expect(make(m, err, info) && said.size() == 1, "logged once");
      Expect(Has(a2f::DescribeMetacommands(m), "on ("), "shown as on");
    }
    {
      with = without = 0;
      said.clear();
      a2f::Metacommands m;
      with_error = kMeta;
      Expect(make(m, err, info) && info == "off" && with == 1 && without == 1 &&
                 m.state == a2f::Metacommands::kOff && m.error == kMeta,
             "they fail: the session is loaded without them at once");
      Expect(said.size() == 2 && Has(said[0], "failed") &&
                 Has(said[0], "887A0005") && Has(said[1], "off from now on"),
             "logged the failure and the fallback");
      Expect(Has(a2f::DescribeMetacommands(m), "off:"), "shown as off");
      Expect(make(m, err, info) && with == 1 && without == 2 && said.size() == 2,
             "the next session does not try them again");
      without_error = kHung;
      err.clear();
      Expect(!make(m, err, info) && err == kHung && with == 1 &&
                 m.state == a2f::Metacommands::kOff,
             "a later loss is reported as itself; they stay off");
    }
    {
      with = without = 0;
      said.clear();
      a2f::Metacommands m;
      with_error = kMeta;
      without_error = kOom;
      Expect(!make(m, err, info) && err == kMeta && with == 1 && without == 1 &&
                 m.state == a2f::Metacommands::kFailed,
             "fails both ways: reported with the first reason");
      Expect(said.size() == 2 && Has(said[1], "failed too") &&
                 Has(said[1], "8007000E"),
             "logged that they are not the cause");
      Expect(Has(a2f::DescribeMetacommands(m), "not the cause"),
             "shown as not the cause");
      Expect(!make(m, err, info) && with == 2 && without == 2,
             "the retry tries both again");
      with_error.clear();
      Expect(make(m, err, info) && info == "on" &&
                 m.state == a2f::Metacommands::kOn && m.error.empty(),
             "and takes them when they work");
    }
    {
      with = without = 0;
      a2f::Metacommands m;
      with_error = "cannot load onnxruntime.dll (error 126)";
      Expect(!make(m, err, info) && with == 1 && without == 0 &&
                 m.state == a2f::Metacommands::kUntried,
             "a runtime that does not load is not tried twice");
    }
    {
      // inside the keeper: no CPU fallback, no error left
      with = without = 0;
      with_error = kMeta;
      without_error.clear();
      a2f::Metacommands m;
      double now = 0;
      int cpu = 0;
      a2f::NetKeeper k(
          true,
          [&](bool gpu, std::string& e,
              std::string& i) -> std::unique_ptr<a2f::Net> {
            if (gpu) return a2f::CreateGpuNet(create, m, e, i, log);
            ++cpu;
            return nullptr;
          },
          [&] { return now; });
      Expect(k.Ensure() && k.on_gpu() && k.error().empty() && cpu == 0 &&
                 k.attempts() == 0 && k.info() == "off",
             "the keeper holds the GPU session without them");
    }
  }
  printf("-- the device inside a graphics hook's wrapper, or the wrapper\n");
  {
    std::string device_error, wrapper_error;
    bool wrapped = true;
    int on_device = 0, on_wrapper = 0;
    std::vector<std::string> said;
    a2f::CreateOnDevice create = [&](bool wrapper, std::string& err,
                                     std::string& info)
        -> std::unique_ptr<a2f::Net> {
      ++(wrapper ? on_wrapper : on_device);
      const std::string& e = wrapper ? wrapper_error : device_error;
      if (!e.empty()) {
        err = e;
        return nullptr;
      }
      info = wrapper ? "wrapper" : "device";
      static Machine machine;
      return std::make_unique<FakeNet>(&machine, true);
    };
    auto log = [&](const std::string& s) { said.push_back(s); };
    auto make = [&](a2f::Device& d, std::string& err, std::string& info) {
      return a2f::CreateDeviceNet(create, d, [&] { return wrapped; }, err,
                                  info, log);
    };
    std::string err, info;
    {
      a2f::Device d;
      Expect(make(d, err, info) && info == "device" && on_device == 1 &&
                 on_wrapper == 0 && !d.wrapper && said.empty(),
             "the device inside works: nothing tried on the wrapper");
    }
    {
      on_device = on_wrapper = 0;
      a2f::Device d;
      device_error = kRemoved;
      Expect(make(d, err, info) && info == "wrapper" && on_device == 1 &&
                 on_wrapper == 1 && d.wrapper && d.error == kRemoved,
             "lost on the device inside: loaded on the wrapper at once");
      Expect(said.size() == 2 && Has(said[0], "887A0005") &&
                 Has(said[1], "from now on"),
             "logged the loss and the switch");
      Expect(make(d, err, info) && on_device == 1 && on_wrapper == 2 &&
                 said.size() == 2,
             "the next session is on the wrapper without trying the device");
      wrapper_error = kHung;
      err.clear();
      Expect(!make(d, err, info) && err == kHung && on_device == 1 && d.wrapper,
             "a later loss on the wrapper is reported as itself; it stays");
    }
    {
      on_device = on_wrapper = 0;
      said.clear();
      a2f::Device d;
      device_error = kRemoved;
      wrapper_error = kDriver;
      Expect(!make(d, err, info) && err == kRemoved && on_device == 1 &&
                 on_wrapper == 1 && !d.wrapper,
             "lost both ways: reported with the device's reason, not switched");
      Expect(said.size() == 2 && Has(said[1], "failed too") &&
                 Has(said[1], "887A0020"),
             "logged that the wrapper failed too");
      Expect(!make(d, err, info) && on_device == 2 && on_wrapper == 2,
             "the retry tries both again");
    }
    {
      on_device = on_wrapper = 0;
      a2f::Device d;
      device_error = kOom;
      wrapper_error.clear();
      Expect(!make(d, err, info) && err == kOom && on_device == 1 &&
                 on_wrapper == 0,
             "a failure that is not a lost device does not try the wrapper");
      device_error = kRemoved;
      wrapped = false;
      Expect(!make(d, err, info) && err == kRemoved && on_device == 2 &&
                 on_wrapper == 0,
             "nor does a lost device when no wrapper was handed out");
      wrapped = true;
    }
    {
      on_device = on_wrapper = 0;
      a2f::Device d;
      d.fixed = true;
      d.wrapper = false;
      Expect(!make(d, err, info) && on_device == 1 && on_wrapper == 0,
             "forced no-wrapper never retries the wrapper on device loss");
      on_device = on_wrapper = 0;
      d.wrapper = d.fixed = true;  // UseDeviceWrapper=1
      device_error.clear();
      wrapper_error = kRemoved;
      Expect(!make(d, err, info) && err == kRemoved && on_device == 0 &&
                 on_wrapper == 1 && d.wrapper,
             "the ini's wrapper is never switched away from");
      wrapper_error.clear();
      Expect(make(d, err, info) && info == "wrapper" && on_device == 0,
             "and is used when it works");
    }
    {
      // inside the keeper: no CPU fallback, no error left after the switch
      on_device = on_wrapper = 0;
      device_error = kRemoved;
      wrapper_error.clear();
      a2f::Device d;
      double now = 0;
      int cpu = 0;
      a2f::NetKeeper k(
          true,
          [&](bool gpu, std::string& e,
              std::string& i) -> std::unique_ptr<a2f::Net> {
            if (gpu) return make(d, e, i);
            ++cpu;
            return nullptr;
          },
          [&] { return now; });
      Expect(k.Ensure() && k.on_gpu() && k.error().empty() && cpu == 0 &&
                 k.attempts() == 0 && k.info() == "wrapper",
             "the keeper holds the GPU session on the wrapper");
    }
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
      "GPU disabled until restart", 3, roomy);
  Expect(any(retained_failure, "CPU fallback") &&
             any(retained_failure, "Restart the game") &&
             !any(retained_failure, "each on a new device"),
         "retained failed context requires restart, not device recreation");
  Expect(any(DiagnoseNet(true, kOom, 1, roomy), "restart the game"),
         "out of memory: advised to free VRAM and restart");
  Expect(Has(first(DiagnoseNet(true, kOom, 1, roomy)), "no VRAM left"),
         "out of memory: no VRAM left");
  Expect(Has(first(DiagnoseNet(true, kRemoved, 1, full)),
             "no VRAM left (50 MB free of 8000 MB"),
         "device removed with VRAM full: no VRAM left, with the numbers");
  Expect(any(DiagnoseNet(true, kRemoved, 1, full), "DirectML gave no reason"),
         "and still says the device was lost");
  Expect(any(DiagnoseNet(true, removed + " [DirectML removed reason 887A0020]",
                         1, roomy),
             "DirectML stopped by itself (reason 887A0020)"),
         "reason from DirectML alone: Direct3D 12 is not blamed");
  for (const char* e : {kRemoved, kHung, kReset, kDriver})
    Expect(
        Has(first(DiagnoseNet(true, e, 1, roomy)),
            "graphics device was lost (6000 MB free") &&
            Has(first(DiagnoseNet(true, e, 1, unknown)), "free VRAM unknown"),
        "device lost with VRAM free: device lost");
  Expect(any(DiagnoseNet(true, removed + " [device removed reason 887A0006]", 1,
                         roomy),
             "stopped responding"),
         "reason from Windows: hung");
  Expect(any(DiagnoseNet(true, removed + " [device removed reason 887A0007]", 1,
                         roomy),
             "driver was reset"),
         "reason from Windows: reset");
  Expect(any(DiagnoseNet(true, removed + " [device removed reason 887A0020]", 1,
                         roomy),
             "Update the graphics driver"),
         "reason from Windows: driver error");
  Expect(any(DiagnoseNet(true, removed + " [device removed reason 887A0005]", 1,
                         roomy),
             "removed or disabled"),
         "reason from Windows: removed");
  Expect(first(DiagnoseNet(true, removed + " [device removed reason 8007000E]",
                           1, roomy))
                 .find("no VRAM left") != std::string::npos,
         "reason from Windows: out of memory");
  Expect(!any(DiagnoseNet(true, kRemoved, 2, roomy), "Still failing") &&
             any(DiagnoseNet(true, kRemoved, 3, roomy),
                 "Still failing after 3 attempts") &&
             !any(DiagnoseNet(true, kRemoved, 3, roomy), "estart the game"),
         "a session that keeps failing on a working device: send the log");
  Expect(any(DiagnoseNet(true, removed + " [device removed reason 887A0006]", 3,
                         roomy),
             "Still lost after 3 attempts: restart the game"),
         "a device that stays lost: restart the game");
  Expect(Has(first(DiagnoseNet(true, kOther, 1, roomy)),
             "DirectML is not available"),
         "DirectML missing");
  Expect(Has(first(DiagnoseNet(true, "session: bad graph", 1, roomy)),
             "did not load (6000 MB free"),
         "other: did not load");
  Expect(Has(first(DiagnoseNet(true, "model file not found: x.onnx", 1, roomy)),
             "model file is missing (model file not found: x.onnx)"),
         "missing model file");
  Expect(any(DiagnoseNet(true, "cannot load onnxruntime.dll (error 126)", 1,
                         roomy),
             "Reinstall the mod") &&
             any(DiagnoseNet(true, "onnxruntime.dll: wrong version", 1, roomy),
                 "two mod versions"),
         "missing / wrong onnxruntime.dll");
  Expect(Has(first(DiagnoseNet(false, "x", 1, unknown)), "on the CPU"),
         "CPU failure");
  Expect(a2f::GpuWarnings(roomy).empty() && a2f::GpuWarnings(unknown).empty(),
         "healthy GPU: no warnings");
  a2f::GpuState laptop = roomy;
  laptop.adapter = "Integrated";
  laptop.dedicated_mb = 128;
  laptop.bigger_adapter = "Discrete";
  laptop.bigger_mb = 8192;
  Expect(any(a2f::GpuWarnings(laptop),
             "run on \"Integrated\" (128 MB), not on \"Discrete\" (8192 MB)"),
         "warns when the game is on the smaller GPU");
  a2f::GpuState over = roomy;
  over.used_mb = 8500;
  Expect(any(a2f::GpuWarnings(over), "500 MB over the 8000 MB"),
         "warns when over the VRAM budget");
  printf(failures ? "%d FAILED\n" : "ok: 0 failures\n", failures);
  return failures != 0;
}
