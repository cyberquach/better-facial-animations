#include "runtime_fixture.h"
// Exercise the real node and animation hooks with engine-independent clocks.
#include <array>
#include <cstdlib>

#include "../src/plugin/better_facial_animations.cc"

namespace {
void Check(bool ok, int line) {
  if (ok) return;
  std::fprintf(stderr, "FAIL face update: %d\n", line);
  std::exit(1);
}
#define CHECK(x) Check((x), __LINE__)

bfa::AnimUpdateFn animate;
void* vtable;
float elapsed = 0;
float delta = 0;
int calls = 0;
bool nested = false;
uint8_t other_face;

bool OriginalAnimation(uint8_t*, float dt, uint64_t flag) {
  CHECK(flag == 7);
  elapsed += dt;
  ++calls;
  return true;
}
void OriginalNode(uint8_t* node, bfa::NiUpdateData* data, uint32_t flags) {
  CHECK(flags == 42 && data->flags.underlying() == 13);
  auto* face = bfa::Field<uint8_t*>(node, (skse::IsVr() ? 0x178 : 0x150));
  if (nested) {
    nested = false;
    const bfa::ScopedFaceUpdate inner(&other_face, 4);
    CHECK(bfa::ScopedFaceUpdate::Delta(&other_face, 0.25f) == 0.25f);
    CHECK(bfa::ScopedFaceUpdate::Delta(face, 0.25f) == 0.25f);
  }
  CHECK(animate(face, delta, 7));
}
DWORD WINAPI OtherThread(void* face) {
  CHECK(bfa::ScopedFaceUpdate::Delta(face, 0.25f) == 0.25f);
  return 0;
}

bfa::AnimUpdateFn MakeCaller() {
  const auto base = reinterpret_cast<uint64_t>(GetModuleHandleA(nullptr));
  auto* stub =
      call_sites::Thunk(base, reinterpret_cast<void*>(OriginalAnimation));
  auto* caller =
      call_sites::Thunk(base, reinterpret_cast<void*>(OriginalAnimation));
  CHECK(stub && caller);
  const uint8_t code[] = {0x48, 0x83, 0xEC, 0x28, 0xE8, 0,    0,
                          0,    0,    0x48, 0x83, 0xC4, 0x28, 0xC3};
  std::memcpy(caller, code, sizeof(code));
  const auto at = reinterpret_cast<uint64_t>(caller + 4);
  CHECK(call_sites::PatchCall(at, stub));
  CHECK(call_sites::Hook(at, reinterpret_cast<void*>(bfa::AnimUpdateHook),
                         "face update test", base));
  return reinterpret_cast<bfa::AnimUpdateFn>(caller);
}
}  // namespace

extern "C" uint64_t TestAddress(uint64_t id) {
  if (id == kIdFaceAnimUpdate.Get())
    return reinterpret_cast<uint64_t>(OriginalAnimation);
  CHECK(id == kIdFaceNodeVtable.Get());
  return reinterpret_cast<uint64_t>(vtable);
}

int main() {
  bfa::testing::address = TestAddress;
  animate = MakeCaller();
  uint8_t face;
  bfa::partner_fg = &face;
  // Independent expected node offsets and virtual slots for SE, AE and VR.
  for (uint32_t runtime :
       {skse::Version(1, 5, 97), skse::Version(1, 6, 1170), skse::kVrRuntime}) {
    skse::SetRuntime(runtime);
    const bool vr = runtime == skse::kVrRuntime;
    {
      std::array<std::array<uint8_t, 0x140>, 33> ancestors{};
      uint8_t owner, other_owner;
      const size_t owner_offset = vr ? 0x110 : 0xF8;
      for (size_t i = 0; i + 1 < ancestors.size(); ++i)
        bfa::Field<uint8_t*>(ancestors[i].data(), 0x30) =
            ancestors[i + 1].data();
      // The 33rd owner must not be read through a recursive library getter.
      bfa::Field<uint8_t*>(ancestors[32].data(), owner_offset) = &owner;
      CHECK(!bfa::OfRef(ancestors[0].data(), &owner));
      bfa::Field<uint8_t*>(ancestors[31].data(), owner_offset) = &owner;
      bfa::Field<uint8_t*>(ancestors[0].data(), owner_offset) = &other_owner;
      CHECK(bfa::OfRef(ancestors[0].data(), &owner));
      CHECK(bfa::OfRef(ancestors[0].data(), &other_owner));
      CHECK(!bfa::OfRef(nullptr, &owner));
      // A cyclic graph with no owner must terminate within the same budget.
      for (auto& ancestor : ancestors)
        bfa::Field<uint8_t*>(ancestor.data(), owner_offset) = nullptr;
      bfa::Field<uint8_t*>(ancestors[31].data(), 0x30) = ancestors[0].data();
      CHECK(!bfa::OfRef(ancestors[0].data(), &owner));
    }
    alignas(16) std::array<uint8_t, 0x190> node{};
    bfa::Field<uint8_t*>(node.data(), vr ? 0x178 : 0x150) = &face;
    vtable =
        VirtualAlloc(nullptr, 4096, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    CHECK(vtable);
    auto* slot =
        static_cast<bfa::FaceNodeUpdateFn*>(vtable) + (vr ? 0x2D : 0x2C);
    *slot = OriginalNode;
    DWORD old;
    CHECK(VirtualProtect(vtable, 4096, PAGE_READONLY, &old));
    CHECK(bfa::InstallFaceNodeUpdateHook());
    CHECK(*slot == bfa::FaceNodeUpdateHook);
    MEMORY_BASIC_INFORMATION info{};
    CHECK(VirtualQuery(vtable, &info, sizeof(info)));
    CHECK(info.Protect == PAGE_READONLY);
    elapsed = 0;
    calls = 0;
    for (float dt : {0.016f, 0.02f, 0.25f, 1.0f, 0.0f}) {
      delta = dt;
      const float before = elapsed;
      bfa::NiUpdateData normal{10, static_cast<RE::NiUpdateData::Flag>(13)},
          refresh{0, static_cast<RE::NiUpdateData::Flag>(13)};
      (*slot)(node.data(), &normal, 42);
      CHECK(elapsed == before + dt);
      nested = true;
      (*slot)(node.data(), &refresh, 42);
      CHECK(elapsed == before + dt);
      // Repeated positive scene times and legitimate non-unit rates survive.
      (*slot)(node.data(), &normal, 42);
      CHECK(elapsed == (before + dt) + dt);
      CHECK(bfa::ScopedFaceUpdate::Delta(&face, dt) == dt);
    }
    CHECK(calls == 15);  // zero-time passes still run the native hook chain
    // An unrelated NPC is untouched even while the partner remains active.
    bfa::Field<uint8_t*>(node.data(), vr ? 0x178 : 0x150) = &other_face;
    delta = .25f;
    const float before = elapsed;
    bfa::NiUpdateData refresh{0, static_cast<RE::NiUpdateData::Flag>(13)};
    (*slot)(node.data(), &refresh, 42);
    CHECK(elapsed == before + delta && calls == 16);
    // With no active faces the same native-only path remains available.
    bfa::partner_fg = nullptr;
    (*slot)(node.data(), &refresh, 42);
    CHECK(elapsed == before + delta + delta && calls == 17);
    bfa::partner_fg = &face;
    CHECK(VirtualFree(vtable, 0, MEM_RELEASE));
  }
  {
    const bfa::ScopedFaceUpdate refresh(&face, 0);
    CHECK(bfa::ScopedFaceUpdate::Delta(&face, 0.25f) == 0);
    CHECK(bfa::ScopedFaceUpdate::Delta(&other_face, 0.25f) == 0.25f);
    CHECK(bfa::ScopedFaceUpdate::Delta(nullptr, 0.25f) == 0.25f);
    HANDLE thread = CreateThread(nullptr, 0, OtherThread, &face, 0, nullptr);
    CHECK(thread && WaitForSingleObject(thread, 5000) == WAIT_OBJECT_0);
    CloseHandle(thread);
    bfa::ResetPlayback(true);
    CHECK(bfa::ScopedFaceUpdate::Delta(&face, 0.25f) == 0);
  }
  CHECK(bfa::ScopedFaceUpdate::Delta(&face, 0.25f) == 0.25f);
  std::puts("face update: PASS");
}
