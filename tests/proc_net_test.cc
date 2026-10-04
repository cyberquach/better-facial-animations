// The model's process (BetterFacialAnimationsModel.exe): a CPU session in
// it gives what the session in this process gives, its failures arrive as
// the session's own, a process that dies reads as a lost session, and it
// does not outlive a game that died.
// usage: proc_net_test <plugin dir> <model>
//        proc_net_test game <frozen 0|1> <plugin dir> <model>
//          plays the game for the test: loads a session, freezes its
//          process if asked, and dies without closing anything
#include "../src/proc_net.h"

#include <windows.h>
#include <tlhelp32.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {
int failures = 0;
void Expect(bool ok, const char* label) {
  printf("%s %s\n", ok ? "ok" : "FAIL", label);
  if (!ok) ++failures;
}
std::wstring FullPath(const char* path) {
  wchar_t w[MAX_PATH], full[MAX_PATH];
  MultiByteToWideChar(CP_ACP, 0, path, -1, w, MAX_PATH);
  GetFullPathNameW(w, MAX_PATH, full, nullptr);
  return full;
}
void Log(const std::string& s) { printf("  log: %s\n", s.c_str()); }
constexpr int kBatch = 2;
constexpr wchar_t kExe[] = L"BetterFacialAnimationsModel.exe";
constexpr int kGameReady = 7;  // the game's exit code when it died as asked

// The model's processes started by the process parent.
std::vector<DWORD> ModelProcesses(DWORD parent) {
  std::vector<DWORD> ids;
  HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
  PROCESSENTRY32W e{};
  e.dwSize = sizeof e;
  for (BOOL more = Process32FirstW(snap, &e); more;
       more = Process32NextW(snap, &e))
    if (e.th32ParentProcessID == parent && !_wcsicmp(e.szExeFile, kExe))
      ids.push_back(e.th32ProcessID);
  CloseHandle(snap);
  return ids;
}

// The game: a session in its process, which then hangs (frozen) or idles,
// and the game is gone at once, as when it crashes or is killed.
int PlayGame(bool frozen, const std::wstring& dir, const std::wstring& model) {
  a2f::OrtOptions o;
  o.gpu = false;
  std::string err, info;
  bool started = false;
  auto net = a2f::CreateProcNet(dir, model, o, nullptr, started, err, info);
  const std::vector<DWORD> ids = ModelProcesses(GetCurrentProcessId());
  if (!net || ids.size() != 1) return 1;
  if (frozen) {
    using SuspendFn = LONG(WINAPI*)(HANDLE);
    auto suspend = reinterpret_cast<SuspendFn>(reinterpret_cast<void*>(
        GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "NtSuspendProcess")));
    HANDLE p = OpenProcess(PROCESS_SUSPEND_RESUME, FALSE, ids[0]);
    if (!p || !suspend || suspend(p) < 0) return 1;
  }
  TerminateProcess(GetCurrentProcess(), kGameReady);
  return 1;
}

// Runs the game above and waits for its model's process to be gone.
void ExpectNoOrphan(bool frozen, const char* dir, const char* model,
                    const char* label) {
  wchar_t self[MAX_PATH];
  GetModuleFileNameW(nullptr, self, MAX_PATH);
  std::wstring command = std::wstring(L"\"") + self + L"\" game " +
                         (frozen ? L"1" : L"0") + L" \"" + FullPath(dir) +
                         L"\" \"" + FullPath(model) + L"\"";
  STARTUPINFOW si{};
  si.cb = sizeof si;
  PROCESS_INFORMATION pi{};
  bool gone = false;
  if (CreateProcessW(self, &command[0], nullptr, nullptr, FALSE, 0, nullptr,
                     nullptr, &si, &pi)) {
    // its process is alive just before the game dies
    std::vector<DWORD> seen;
    while (WaitForSingleObject(pi.hProcess, 5) == WAIT_TIMEOUT)
      if (auto ids = ModelProcesses(pi.dwProcessId); !ids.empty()) seen = ids;
    DWORD code = 0;
    GetExitCodeProcess(pi.hProcess, &code);
    double waited = 0;
    for (; waited < 5000 && !ModelProcesses(pi.dwProcessId).empty();
         waited += 50)
      Sleep(50);
    gone = code == kGameReady && seen.size() == 1 &&
           ModelProcesses(pi.dwProcessId).empty();
    printf("  game %lu died (exit code %lu); its model process %lu was "
           "alive, %s %.0f ms later\n",
           pi.dwProcessId, code, seen.empty() ? 0 : seen[0],
           ModelProcesses(pi.dwProcessId).empty() ? "gone" : "STILL RUNNING",
           waited);
  }
  Expect(gone, label);
}
}  // namespace

int main(int argc, char** argv) {
  if (argc == 5 && !strcmp(argv[1], "game"))
    return PlayGame(argv[2][0] == '1', FullPath(argv[3]), FullPath(argv[4]));
  if (argc < 3) {
    printf("usage: proc_net_test <plugin dir> <model>\n");
    return 2;
  }
  const std::wstring dir = FullPath(argv[1]), model = FullPath(argv[2]);
  a2f::OrtOptions o;
  o.gpu = false;
  o.batch = kBatch;
  std::string err, info;
  bool started = false;
  std::vector<float> x(kBatch * a2f::kWindow), emo(kBatch * a2f::kNumEmo),
      y(kBatch * a2f::kNumOut), ref(y.size());
  for (size_t i = 0; i < x.size(); ++i)
    x[i] = 0.3f * static_cast<float>(
                      std::sin(i * 2 * 3.14159265 * 220 / a2f::kSampleRate));

  auto net = a2f::CreateProcNet(dir, model, o, Log, started, err, info);
  Expect(started && net, "CPU session loads in its own process");
  printf("  %s\n", net ? info.c_str() : err.c_str());
  auto here = a2f::CreateOrtNet(dir, model, o, err, info);
  bool same = net && here && net->Batch() == kBatch;
  for (int run = 0; same && run < 3; ++run) {
    same = net->Run(x.data(), emo.data(), kBatch, y.data(), err) &&
           here->Run(x.data(), emo.data(), kBatch, ref.data(), err);
    for (size_t i = 0; i < y.size(); ++i)
      same = same && std::fabs(y[i] - ref[i]) < 1e-4f;
  }
  Expect(same, "its runs give what the session in this process gives");

  if (net) {
    // every model process of this Wine prefix: a game running in it too
    system("taskkill /f /im BetterFacialAnimationsModel.exe >nul 2>&1");
    const bool ran = net->Run(x.data(), emo.data(), kBatch, y.data(), err);
    Expect(!ran && net->failed, "a process that died is a lost session");
    printf("  %s\n", err.c_str());
  }
  net.reset();
  Expect(ModelProcesses(GetCurrentProcessId()).empty(),
         "a session that is destroyed ends its process");

  net = a2f::CreateProcNet(dir, model + L".missing", o, Log, started, err,
                           info);
  Expect(started && !net && err.find("session: ") == 0,
         "a session that does not load fails with its own reason");
  printf("  %s\n", err.c_str());

  net = a2f::CreateProcNet(dir + L"\\licenses", model, o, Log, started, err,
                           info);
  Expect(!started && !net && err.find("cannot start") == 0,
         "a process that cannot be started says so");
  printf("  %s\n", err.c_str());

  // Another program under the process's name: it runs, prints text and
  // exits, as a process that an antivirus or a hook stops early.
  wchar_t self[MAX_PATH];
  GetModuleFileNameW(nullptr, self, MAX_PATH);
  std::wstring here_dir = self;
  here_dir.resize(here_dir.find_last_of(L'\\'));
  const std::wstring fake = here_dir + L"\\not_the_model";
  CreateDirectoryW(fake.c_str(), nullptr);
  CopyFileW((here_dir + L"\\voice_sync_test.exe").c_str(),
            (fake + L"\\" + kExe).c_str(), FALSE);
  net = a2f::CreateProcNet(fake, model, o, Log, started, err, info);
  Expect(!started && !net && err.find("at its start") != std::string::npos,
         "a process that does not come up as the model's says so");
  printf("  %s\n", err.c_str());

  std::vector<std::string> report;
  const bool reported = a2f::ProcReport(dir, report, err);
  bool lists_runtime = false;
  for (const auto& line : report) {
    printf("  %s\n", line.c_str());
    lists_runtime |= line.find("onnxruntime.dll") != std::string::npos;
  }
  Expect(reported && lists_runtime,
         "the process reports its device and modules");

  ExpectNoOrphan(false, argv[1], argv[2],
                 "the process does not outlive a game that died");
  ExpectNoOrphan(true, argv[1], argv[2],
                 "a hung process does not outlive a game that died");

  // The GPU session: Wine without vkd3d-proton may not have one.
  o.gpu = true;
  net = a2f::CreateProcNet(dir, model, o, Log, started, err, info);
  const bool gpu_ran =
      net && net->Run(x.data(), emo.data(), kBatch, y.data(), err);
  printf("  GPU session in its own process: %s\n",
         gpu_ran ? info.c_str() : err.c_str());

  printf(failures ? "%d FAILED\n" : "all ok\n", failures);
  return failures != 0;
}
