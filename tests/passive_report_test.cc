#include <windows.h>
#include <dxgi.h>
#include <cstdio>
#include "../src/ort_net.h"

namespace {
int factory_calls = 0;
HRESULT WINAPI Factory(REFIID, void**) {
  ++factory_calls;
  return E_FAIL;
}
}
// Intercept the imported factory so even a failed graphics probe is detected.
extern "C" {
decltype(&CreateDXGIFactory1) __wrap___imp_CreateDXGIFactory1 = Factory;
}
int main() {
  const auto report = a2f::GpuReport(false);
  bool skipped = false, system = false;
  for (const auto& line : report) {
    skipped |= line.find("GPU diagnostics skipped") != std::string::npos;
    system |= line.find("system:") == 0;
  }
  if (factory_calls || !skipped || !system) return 1;
  puts("passive diagnostics: system report without DXGI factory calls passed");
}
