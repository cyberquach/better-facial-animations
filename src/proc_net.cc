#include "proc_net.h"

#include <windows.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "net_wire.h"

namespace a2f {
namespace {

constexpr wchar_t kExe[] = L"BetterFacialAnimationsModel.exe";
// How long the process has for one answer (loading a session included)
// before it counts as hung.
constexpr DWORD kAnswerMs = 120000;
constexpr DWORD kPipeBytes = 1 << 17;  // more than one run's messages

struct Handle {
  ~Handle() {
    if (h && h != INVALID_HANDLE_VALUE) CloseHandle(h);
  }
  HANDLE h = nullptr;
};

// A pipe to the process (or from it). ours: the game's end, overlapped, so
// that waiting on it can be given up. theirs: the process's, inheritable.
bool OpenPipe(bool to_process, Handle& ours, Handle& theirs) {
  static std::atomic<unsigned> count;
  const std::wstring name = L"\\\\.\\pipe\\BetterFacialAnimations." +
                            std::to_wstring(GetCurrentProcessId()) + L"." +
                            std::to_wstring(++count);
  ours.h = CreateNamedPipeW(
      name.c_str(),
      (to_process ? PIPE_ACCESS_OUTBOUND : PIPE_ACCESS_INBOUND) |
          FILE_FLAG_OVERLAPPED | FILE_FLAG_FIRST_PIPE_INSTANCE,
      PIPE_TYPE_BYTE | PIPE_REJECT_REMOTE_CLIENTS, 1, kPipeBytes, kPipeBytes, 0,
      nullptr);
  if (ours.h == INVALID_HANDLE_VALUE) return false;
  SECURITY_ATTRIBUTES inherit{sizeof inherit, nullptr, TRUE};
  theirs.h = CreateFileW(name.c_str(), to_process ? GENERIC_READ : GENERIC_WRITE,
                         0, &inherit, OPEN_EXISTING, 0, nullptr);
  return theirs.h != INVALID_HANDLE_VALUE;
}

// The model's process and the pipes to its standard input and output.
// Destroying it ends the process, and so does the end of the game.
class Process {
 public:
  explicit Process(ProcLog log) : log_(std::move(log)) {}
  ~Process() {
    if (!process_.h) return;
    // not asked to exit: a session on a lost device may not come back.
    // Waited for: its VRAM is free before the next session wants it.
    TerminateProcess(process_.h, 0);
    WaitForSingleObject(process_.h, 1000);
  }

  // Runs "<dir>\<kExe> <args>" and waits for its hello.
  bool Start(const std::wstring& dir, const std::wstring& args,
             std::string& err) {
    auto fail = [&err](const char* what) {
      const DWORD error = GetLastError();
      err = std::string("cannot start the model's process (") + what +
            ", error " + std::to_string(error) + ")";
      return false;
    };
    Handle in, out;  // the process's ends
    event_.h = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (!event_.h || !OpenPipe(true, to_, in) || !OpenPipe(false, from_, out))
      return fail("pipes");

    // Only its two ends are inherited, and only by it.
    HANDLE ends[] = {in.h, out.h};
    SIZE_T size = 0;
    InitializeProcThreadAttributeList(nullptr, 1, 0, &size);
    std::vector<char> list_bytes(size);
    auto list =
        reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(list_bytes.data());
    if (!InitializeProcThreadAttributeList(list, 1, 0, &size))
      return fail("attributes");
    STARTUPINFOEXW si{};
    si.StartupInfo.cb = sizeof si;
    si.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
    si.StartupInfo.hStdInput = in.h;
    si.StartupInfo.hStdOutput = out.h;
    si.lpAttributeList = list;
    const std::wstring exe = dir + L"\\" + kExe;
    std::wstring command = L"\"" + exe + L"\" " + args;
    PROCESS_INFORMATION pi{};
    // Windows looks for the process's libraries in its own folder and then
    // in Windows': the Direct3D wrappers in the game's folder (ReShade's and
    // ENB's dxgi.dll / d3d11.dll) are not found. Started in dir so that the
    // game's folder is not searched at all. Below normal priority, as the
    // game's worker thread. Suspended until it is in its job.
    const bool created =
        UpdateProcThreadAttribute(list, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
                                  ends, sizeof ends, nullptr, nullptr) &&
        CreateProcessW(exe.c_str(), &command[0], nullptr, nullptr, TRUE,
                       CREATE_NO_WINDOW | EXTENDED_STARTUPINFO_PRESENT |
                           BELOW_NORMAL_PRIORITY_CLASS | CREATE_SUSPENDED,
                       nullptr, dir.c_str(), &si.StartupInfo, &pi);
    if (!created) fail("process");
    DeleteProcThreadAttributeList(list);
    if (!created) return false;
    process_.h = pi.hProcess;
    // A job that is closed when the game ends, in whatever way, and ends
    // the process with it: one that hangs never sees its input close.
    // (Without the job, where Windows refuses it, the closed input ends it.)
    job_.h = CreateJobObjectW(nullptr, nullptr);
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limit{};
    limit.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    if (job_.h && SetInformationJobObject(job_.h,
                                          JobObjectExtendedLimitInformation,
                                          &limit, sizeof limit))
      AssignProcessToJobObject(job_.h, process_.h);
    ResumeThread(pi.hThread);
    CloseHandle(pi.hThread);

    char type = 0;
    std::string data, why;
    if (Answer(type, data, why) && type == kMsgHello) return true;
    err = (why.empty() ? "the model's process does not answer as one" : why) +
          " at its start";
    return false;
  }

  // A process that stopped fails the answer, whatever the send did.
  void Send(char type, const std::string& data) {
    WriteMsg([this](char* p, size_t n) { return Transfer(to_.h, true, p, n); },
             type, data.data(), data.size());
  }
  // The next message that is not for the log. False when the process
  // stopped or hangs, with that in err.
  bool Answer(char& type, std::string& data, std::string& err) {
    auto read = [this](char* p, size_t n) {
      return Transfer(from_.h, false, p, n);
    };
    while (ReadMsg(read, type, data)) {
      if (type != kMsgLog) return true;
      if (log_) log_(data);
    }
    DWORD code = STILL_ACTIVE;
    WaitForSingleObject(process_.h, 2000);
    GetExitCodeProcess(process_.h, &code);
    char b[80] = "the model's process does not answer";
    if (code != STILL_ACTIVE)
      snprintf(b, sizeof b, "the model's process stopped (exit code %08lX)",
               code);
    err = b;
    return false;
  }

 private:
  // Writes or reads all n bytes; false when the process ended, the pipe
  // broke or kAnswerMs passed.
  bool Transfer(HANDLE pipe, bool write, char* p, size_t n) {
    for (DWORD k = 0; n; p += k, n -= k) {
      OVERLAPPED o{};
      o.hEvent = event_.h;
      const DWORD size = static_cast<DWORD>(n);
      if (!(write ? WriteFile(pipe, p, size, nullptr, &o)
                  : ReadFile(pipe, p, size, nullptr, &o))) {
        if (GetLastError() != ERROR_IO_PENDING) return false;
        HANDLE wait[] = {event_.h, process_.h};
        if (WaitForMultipleObjects(2, wait, FALSE, kAnswerMs) != WAIT_OBJECT_0)
          CancelIo(pipe);
      }
      if (!GetOverlappedResult(pipe, &o, &k, TRUE) || !k) return false;
    }
    return true;
  }

  ProcLog log_;
  Handle process_, job_, to_, from_, event_;
};

class ProcNet : public Net {
 public:
  ProcNet(std::unique_ptr<Process> process, int batch)
      : process_(std::move(process)), batch_(batch) {}
  int Batch() const override { return batch_; }
  bool Run(const float* x, const float* emo, int b, float* out,
           std::string& err) override {
    const size_t nx = static_cast<size_t>(b) * kWindow * sizeof(float),
                 ne = static_cast<size_t>(b) * kNumEmo * sizeof(float),
                 ny = static_cast<size_t>(b) * kNumOut * sizeof(float);
    std::string data(reinterpret_cast<const char*>(x), nx);
    data.append(reinterpret_cast<const char*>(emo), ne);
    char type = 0;
    process_->Send(kMsgRun, data);
    if (!process_->Answer(type, data, err)) {
      failed = true;
      return false;
    }
    if (type == kMsgOk && data.size() == ny) {
      memcpy(out, data.data(), ny);
      return true;
    }
    failed = type != kMsgError;
    err = type == kMsgOk ? "the model's process answered with the wrong size"
                         : data;
    return false;
  }

 private:
  std::unique_ptr<Process> process_;
  int batch_;
};

}  // namespace

std::unique_ptr<Net> CreateProcNet(const std::wstring& dir,
                                   const std::wstring& model,
                                   const OrtOptions& o, const ProcLog& log,
                                   bool& started, std::string& err,
                                   std::string& info) {
  const int batch = std::max(1, o.batch);
  auto process = std::make_unique<Process>(log);
  started = process->Start(dir,
                           std::wstring(o.gpu ? L"1 " : L"0 ") +
                               (o.metacommands ? L"1 " : L"0 ") +
                               std::to_wstring(batch) + L" \"" + model + L"\"",
                           err);
  char type = 0;
  std::string data;
  if (!started || !process->Answer(type, data, err)) return nullptr;
  if (type != kMsgOk) {
    err = data;
    return nullptr;
  }
  info = data + ", in its own process";
  return std::make_unique<ProcNet>(std::move(process), batch);
}

bool ProcReport(const std::wstring& dir, std::vector<std::string>& lines,
                std::string& err) {
  Process process([&lines](const std::string& s) { lines.push_back(s); });
  if (!process.Start(dir, L"report", err)) return false;
  char type = 0;
  std::string data;
  if (!process.Answer(type, data, err)) lines.push_back(err);
  return true;
}

}  // namespace a2f
