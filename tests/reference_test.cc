#include "runtime_fixture.h"
// Check the engine lookup output ABI and packed BSHandleRefObject release.
#include <cstdio>
#include <cstdlib>

#include "a2f/runtime/engine.h"

namespace {
alignas(8) uint8_t actor[0x30]{};
int lookups = 0, destroyed = 0, resolutions = 0;
void Check(bool ok) {
  if (!ok) std::abort();
}
void DeleteThis(uint8_t* object) {
  Check(object == actor + 0x20);
  ++destroyed;
}
bool Lookup(const uint32_t& handle, uint8_t** output) {
  ++lookups;
  Check(output && !*output);
  if (handle != 42) return false;
  InterlockedIncrement(&bfa::Field<LONG>(actor, 0x28));
  *output = actor;
  return true;
}
}  // namespace
namespace skse {
bool IsVr() { return false; }
bool IsSe() { return false; }
uint64_t Addr(uint64_t id) {
  ++resolutions;
  Check(id == kIdLookupReference.ae);
  return reinterpret_cast<uint64_t>(&Lookup);
}
}  // namespace skse
int main() {
  using namespace bfa;
  void* vtable[] = {nullptr, reinterpret_cast<void*>(&DeleteThis)};
  Field<void**>(actor, 0x20) = vtable;
  // Handle metadata must not prevent final reference destruction.
  Field<LONG>(actor, 0x28) = 0x8400;
  {
    ScopedReference empty(0), missing(7), first(42);
    Check(!empty.get() && !missing.get() && first.get() == actor);
    {
      ScopedReference second(42);
      Check((Field<LONG>(actor, 0x28) & 0x3FF) == 2);
    }
    Check(!destroyed && (Field<LONG>(actor, 0x28) & 0x3FF) == 1);
  }
  Check(lookups == 3 && destroyed == 1);
  for (int i = 0; i < 1000; ++i) {
    ScopedReference missing(7);
    Check(!missing.get());
  }
  Check(lookups == 1003 && resolutions == 1 && destroyed == 1);
  Check(Field<LONG>(actor, 0x28) == 0x8400);
  puts(
      "retained reference: null/missing handles, nested retention, masked "
      "final release passed");
}
