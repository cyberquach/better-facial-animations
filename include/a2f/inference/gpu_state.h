// The adapter the model runs on and this process's video memory there, as
// QueryGpu (ort_net.h) reads them and net_keeper.h explains them.
#ifndef INCLUDE_A2F_INFERENCE_GPU_STATE_H_
#define INCLUDE_A2F_INFERENCE_GPU_STATE_H_

#include <string>
#include <vector>

namespace a2f {

// DXGI indices are preserved even when software adapters are omitted.
struct GpuAdapter {
  int index = 0;
  std::string name;
  double dedicated_mb = 0;
};
struct GpuAdapterList {
  std::vector<GpuAdapter> adapters;
  std::string error;
  bool requested = false, complete = false;
};

struct GpuState {
  bool known = false;   // DXGI answered
  std::string adapter;  // adapter 0's name
  double dedicated_mb = 0, budget_mb = 0, used_mb = 0;
  // Another hardware adapter with more dedicated memory than adapter 0
  // (a laptop running the game on its integrated GPU); empty when none.
  std::string bigger_adapter;
  double bigger_mb = 0;
};

}  // namespace a2f

#endif  // INCLUDE_A2F_INFERENCE_GPU_STATE_H_
