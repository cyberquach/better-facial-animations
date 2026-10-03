// The adapter the model runs on and this process's video memory there, as
// QueryGpu (ort_net.h) reads them and net_keeper.h explains them.
#ifndef SRC_GPU_STATE_H_
#define SRC_GPU_STATE_H_

#include <string>

namespace a2f {

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

#endif  // SRC_GPU_STATE_H_
