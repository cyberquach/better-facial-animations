// BetterFacialAnimationsModel.exe: the network in a process of its own (see
// proc_net.h), started by the plugin and talking to it over its standard
// input and output (net_wire.h). It says kMsgHello, then
//   BetterFacialAnimationsModel.exe <gpu 0|1> <metacommands 0|1> <batch> <model>
//     loads the session (kMsgOk with its description, or kMsgError) and
//     answers each kMsgRun until the game closes its end
//   BetterFacialAnimationsModel.exe report
//     logs DeviceReport and ModuleReport, then kMsgOk
#include <windows.h>

#include <cstdlib>
#include <cstring>
#include <cwchar>
#include <string>
#include <vector>

#include "net_wire.h"
#include "ort_net.h"

namespace {

// onnxruntime logs from its own threads too.
void Send(char type, const void* data, size_t n) {
  static SRWLOCK lock = SRWLOCK_INIT;
  auto write = [](char* p, size_t n) {
    for (DWORD k = 0; n; p += k, n -= k)
      if (!WriteFile(GetStdHandle(STD_OUTPUT_HANDLE), p,
                     static_cast<DWORD>(n), &k, nullptr))
        return false;
    return true;
  };
  AcquireSRWLockExclusive(&lock);
  a2f::WriteMsg(write, type, data, n);
  ReleaseSRWLockExclusive(&lock);
}
void Send(char type, const std::string& text) {
  Send(type, text.data(), text.size());
}

bool Read(char* p, size_t n) {
  for (DWORD k = 0; n; p += k, n -= k)
    if (!ReadFile(GetStdHandle(STD_INPUT_HANDLE), p, static_cast<DWORD>(n), &k,
                  nullptr) ||
        !k)
      return false;
  return true;
}

}  // namespace

int wmain(int argc, wchar_t** argv) {
  Send(a2f::kMsgHello, "");
  // the folder of this file: onnxruntime.dll and DirectML.dll are in it
  wchar_t path[MAX_PATH] = L"";
  GetModuleFileNameW(nullptr, path, MAX_PATH);
  std::wstring dir = path;
  dir.resize(dir.find_last_of(L'\\'));

  if (argc == 2 && !wcscmp(argv[1], L"report")) {
    // the libraries a session loads, for their versions among the modules
    for (const wchar_t* name : {L"\\DirectML.dll", L"\\onnxruntime.dll"})
      LoadLibraryExW((dir + name).c_str(), nullptr,
                     LOAD_WITH_ALTERED_SEARCH_PATH);
    for (auto report : {a2f::DeviceReport, a2f::ModuleReport})
      for (const auto& line : report()) Send(a2f::kMsgLog, line);
    Send(a2f::kMsgOk, "");
    return 0;
  }
  if (argc != 5) return 2;

  a2f::SetOrtLog([](const std::string& s) { Send(a2f::kMsgLog, s); });
  a2f::OrtOptions o;
  o.gpu = argv[1][0] == L'1';
  o.metacommands = argv[2][0] == L'1';
  o.batch = _wtoi(argv[3]);
  std::string err, info;
  auto net = a2f::CreateOrtNet(dir, argv[4], o, err, info);
  if (!net) {
    Send(a2f::kMsgError, err);
    return 1;
  }
  Send(a2f::kMsgOk, info);

  const size_t in_floats = a2f::kWindow + a2f::kNumEmo;
  char type = 0;
  std::string data;
  std::vector<float> x, y;
  while (a2f::ReadMsg(Read, type, data) && type == a2f::kMsgRun) {
    const int b = static_cast<int>(data.size() / (in_floats * sizeof(float)));
    x.resize(b * in_floats);
    memcpy(x.data(), data.data(), x.size() * sizeof(float));
    y.resize(static_cast<size_t>(b) * a2f::kNumOut);
    if (net->Run(x.data(), x.data() + b * a2f::kWindow, b, y.data(), err))
      Send(a2f::kMsgOk, y.data(), y.size() * sizeof(float));
    else
      Send(net->failed ? a2f::kMsgLost : a2f::kMsgError, err);
  }
  return 0;
}
