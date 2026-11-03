#include <windows.h>
#include <dxgi.h>
#include <cstdio>
#include "a2f/inference/ort_net.h"
#include "a2f/inference/diagnostics.h"
#include "a2f/runtime/test_hooks.h"

namespace {
int factory_calls = 0;
HRESULT WINAPI Factory(REFIID, void**) {
  ++factory_calls;
  return E_FAIL;
}
}
int main() {
  bfa::testing::dxgi_factory = Factory;
  const HMODULE before = GetModuleHandleW(L"d3d12.dll");
  const auto report = a2f::GpuReport(false);
  bool skipped = false, system = false, prerequisite = false, runtime = false;
  for (const auto& line : report) {
    skipped |= line.find("GPU diagnostics skipped") != std::string::npos;
    system |= line.find("system:") == 0;
    prerequisite |= line.find("Independent Devices OS prerequisite:") == 0;
    runtime |= line.find("loaded runtime:") == 0;
  }
  if (factory_calls || !skipped || !system || !prerequisite || !runtime ||
      before != GetModuleHandleW(L"d3d12.dll") ||
      a2f::diagnostics::State().Read().independent_api.find("not checked") != 0) return 1;
  puts("passive diagnostics: system report without DXGI factory calls passed");
}
