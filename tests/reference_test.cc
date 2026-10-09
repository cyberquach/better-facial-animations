// Check the engine lookup output ABI and packed BSHandleRefObject release.
#include "a2f/runtime/engine.h"
#include <cstdio>
#include <cstdlib>

namespace {
alignas(8) uint8_t actor[0x30]{};
int lookups = 0, destroyed = 0;
void Check(bool ok) { if (!ok) std::abort(); }
void DeleteThis(uint8_t* object) {
  Check(object == actor + bfa::kRefrHandleObj);
  ++destroyed;
}
bool Lookup(const uint32_t& handle, uint8_t** output) {
  ++lookups;
  Check(output && !*output);
  if (handle != 42) return false;
  InterlockedIncrement(&bfa::Field<LONG>(actor, bfa::kRefrRefCount));
  *output = actor;
  return true;
}
}
namespace skse {
bool IsVr() { return false; }
bool IsSe() { return false; }
uint64_t Addr(uint64_t id) {
  Check(id == kIdLookupReference.ae);
  return reinterpret_cast<uint64_t>(&Lookup);
}
}
int main() {
  using namespace bfa;
  void* vtable[] = {nullptr, reinterpret_cast<void*>(&DeleteThis)};
  Field<void**>(actor, kRefrHandleObj) = vtable;
  // Handle metadata must not prevent final reference destruction.
  Field<LONG>(actor, kRefrRefCount) = 0x8400;
  {
    ScopedReference empty(0), missing(7), first(42);
    Check(!empty.get() && !missing.get() && first.get() == actor);
    {
      ScopedReference second(42);
      Check((Field<LONG>(actor, kRefrRefCount) & 0x3FF) == 2);
    }
    Check(!destroyed && (Field<LONG>(actor, kRefrRefCount) & 0x3FF) == 1);
  }
  Check(lookups == 3 && destroyed == 1);
  Check(Field<LONG>(actor, kRefrRefCount) == 0x8400);
  puts("retained reference: null/missing handles, nested retention, masked final release passed");
}
