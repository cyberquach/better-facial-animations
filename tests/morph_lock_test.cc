#include "runtime_fixture.h"
// The morph hook keeps a played face's lock and every dynamic face shape's
// lock from before the engine's ApplyMorphs (which resets the vertices) until
// its own deltas are in. A release in between let the render thread copy a
// frame without the pose: a random one-frame flicker on speaking faces.
// A head with a static (non-dynamic) part that carries base and morph vertices
// never reaches the engine's ApplyMorphs: its morph kernel locks that part's
// dynamic data without checking that the part is dynamic.
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "../src/plugin/better_facial_animations.cc"
#define CHECK(x)                                           \
  do {                                                     \
    if (!(x)) {                                            \
      std::fprintf(stderr, "morph lock %d: %s\n", __LINE__, #x); \
      std::exit(1);                                        \
    }                                                      \
  } while (0)
namespace {
using namespace bfa;
constexpr int kShapes = 2;
alignas(16) std::array<uint8_t, 0x500> node{}, fg{}, node2{}, fg2{}, npc{},
    plain{};
alignas(16) std::array<uint8_t, 0x500> shapes[kShapes], fods[kShapes],
    plain_fod{};
std::array<float, 8> buffers[kShapes];
std::array<float, 6> base{1, 2, 3, 4, 5, 6};
uint8_t* children[] = {shapes[0].data(), plain.data(), shapes[1].data()};
uint8_t* extras[kShapes] = {fods[0].data(), fods[1].data()};
uint8_t* plain_extra = plain_fod.data();
uint8_t* no_children[] = {nullptr};
bool plain_morphs = false;

// what the fakes saw
int depth[kShapes]{}, lock_calls = 0, orig_calls = 0, lookups = 0;
int orig_depth[kShapes]{}, lookup_depth = 99;
bool orig_done = false, in_hook = false, released_in_gap = false,
     face_owned_in_orig = false;
uint32_t face_count_in_orig = 0;

int Index(uint8_t* s) {
  for (int i = 0; i < kShapes; ++i)
    if (s == shapes[i].data()) return i;
  CHECK(false);  // only the face shapes are ever locked
  return -1;
}
float* Lock(uint8_t* s) {
  const int i = Index(s);
  ++depth[i];
  ++lock_calls;
  return buffers[i].data();
}
void Unlock(uint8_t* s) {
  const int i = Index(s);
  CHECK(depth[i] > 0);
  // a release after the reset and before the deltas pass is the flicker
  if (--depth[i] == 0 && in_hook && orig_done && !lookups)
    released_in_gap = true;
}
void Update(uint8_t* s, const float* xyz, bool) {
  const int i = Index(s);
  CHECK(depth[i] > 0);
  for (int v = 0; v < 2; ++v) std::copy_n(xyz + v * 3, 3, buffers[i].data() + v * 4);
}
// The engine's ApplyMorphs: resets each shape under its own lock (which
// nests under the hook's) and clears the node's reset flag.
uint64_t Morph(uint8_t* n, uint8_t force) {
  CHECK((n == node.data() || n == node2.data()) && force == 1);
  ++orig_calls;
  auto* f = Field<uint8_t*>(n, 0x150);
  face_owned_in_orig = Field<uint32_t>(f, 0x220) == GetCurrentThreadId();
  face_count_in_orig = Field<uint32_t>(f, 0x224);
  for (int i = 0; i < kShapes; ++i) orig_depth[i] = depth[i];
  if (n == node.data())
    for (int i = 0; i < kShapes; ++i) {
      const int before = depth[i];
      float* data = Lock(shapes[i].data());
      CHECK(depth[i] == before + 1);
      for (int v = 0; v < 2; ++v) std::copy_n(base.data() + v * 3, 3, data + v * 4);
      Unlock(shapes[i].data());
      CHECK(depth[i] == before);
    }
  Field<uint16_t>(n, 0x160) &= ~1;
  orig_done = true;
  return 123;
}
// The engine's shape -> morph data.
void* MorphData(uint8_t* s) {
  return s == plain.data() && plain_morphs ? plain_fod.data() : nullptr;
}
// AddDirect retains the line's actor through this; it runs after the
// original and must still be under the locks.
bool Lookup(const uint32_t& h, uint8_t** out) {
  CHECK(h == 42);
  ++lookups;
  lookup_depth = std::min(depth[0], depth[1]);
  *out = nullptr;
  return false;
}
}  // namespace
extern "C" uint64_t TestAddress(uint64_t id) {
  if (id == kIdApplyMorphs.Get()) return reinterpret_cast<uint64_t>(&Morph);
  if (id == kIdLockDynamic.Get()) return reinterpret_cast<uint64_t>(&Lock);
  if (id == kIdUnlockDynamic.Get()) return reinterpret_cast<uint64_t>(&Unlock);
  if (id == kIdUpdateDynamic.Get()) return reinterpret_cast<uint64_t>(&Update);
  if (id == kIdMorphData.Get()) return reinterpret_cast<uint64_t>(&MorphData);
  if (id == kIdLookupReference.Get()) return reinterpret_cast<uint64_t>(&Lookup);
  return 0;
}
namespace {
void Shape(int i) {
  auto* s = shapes[i].data();
  s[0x150] = 4;  // BSDynamicTriShape
  Field<const char*>(s, 0x10) = FixtureString("FemaleHead");
  Field<uint16_t>(s, 0x15A) = 2;   // vertexCount
  Field<uint32_t>(s, 0x170) = 32;  // dataSize = nv * 16
  Field<uint8_t**>(s, 0x20) = &extras[i];
  Field<uint16_t>(s, 0x28) = 1;
  Field<const char*>(fods[i].data(), 0x10) = FixtureString("FOD");
  Field<const float*>(fods[i].data(), 0x18) = base.data();
  Field<uint32_t>(fods[i].data(), 0x24) = 2;
  buffers[i] = {1, 2, 3, 99, 4, 5, 6, 98};
}
void Seed() {
  for (int i = 0; i < kShapes; ++i) Shape(i);
  plain[0x150] = 1;  // a non-dynamic child is never locked
  Field<uint8_t*>(node.data(), 0x150) = fg.data();
  Field<uint8_t**>(node.data(), 0x118) = children;
  Field<uint16_t>(node.data(), 0x122) = 3;
  Field<uint8_t*>(node2.data(), 0x150) = fg2.data();
  Field<uint8_t**>(node2.data(), 0x118) = no_children;
  Field<uint16_t>(node2.data(), 0x122) = 0;
  Field<uint32_t>(npc.data(), 0x14) = 0x123;
  ready = true;
  loading_game = false;
}
void Reset() {
  for (int i = 0; i < kShapes; ++i) depth[i] = orig_depth[i] = 0;
  lock_calls = orig_calls = lookups = 0;
  lookup_depth = 99;
  orig_done = released_in_gap = face_owned_in_orig = false;
  face_count_in_orig = 0;
  Field<uint16_t>(node.data(), 0x160) = 1;
}
uint64_t Run(uint8_t* n) {
  Reset();
  in_hook = true;
  const uint64_t r = ApplyMorphsHook(n, 1);
  in_hook = false;
  CHECK(r == 123 && orig_calls == 1);
  for (int i = 0; i < kShapes; ++i) CHECK(depth[i] == 0);  // all released
  CHECK(Field<uint32_t>(fg.data(), 0x224) == 0 &&
        Field<uint32_t>(fg2.data(), 0x224) == 0);
  return r;
}
void Play(uint8_t* face) {
  Playing p;
  p.actor = npc.data();
  p.actor_handle = 42;
  p.direct_frame = 0;
  p.ready_at = 0;
  p.entry = std::make_shared<Entry>();
  p.entry->state = State::kReady;
  playing[face] = p;
  Count();
}
void PlayedFace() {
  Play(fg.data());
  Run(node.data());
  // every face shape is locked by the hook before the engine resets it
  for (int i = 0; i < kShapes; ++i) CHECK(orig_depth[i] == 1);
  CHECK(face_owned_in_orig && face_count_in_orig == 1);
  // the engine's own locking nested: 2 of ours, 2 of its
  CHECK(lock_calls == 2 * kShapes);
  // the deltas pass ran while the locks were still held
  CHECK(lookups == 1 && lookup_depth == 1);
  CHECK(!released_in_gap);
  CHECK(!(Field<uint16_t>(node.data(), 0x160) & 1));
}
void OtherFaceWhilePlaying() {
  // another actor's face: no lock work at all
  Run(node2.data());
  for (int i = 0; i < kShapes; ++i) CHECK(orig_depth[i] == 0);
  CHECK(!face_owned_in_orig && lock_calls == 0 && lookups == 0);
}
void NothingPlaying() {
  playing.clear();
  Count();
  Run(node.data());
  for (int i = 0; i < kShapes; ++i) CHECK(orig_depth[i] == 0);
  CHECK(!face_owned_in_orig && lock_calls == kShapes && lookups == 0);
}
void NotReady() {
  Play(fg.data());
  ready = false;
  Run(node.data());
  for (int i = 0; i < kShapes; ++i) CHECK(orig_depth[i] == 0);
  CHECK(!face_owned_in_orig && lock_calls == kShapes && lookups == 0);
  ready = true;
  playing.clear();
  Count();
}
// The static child gains base vertices, then morph data.
void StaticMorphPartSkipsEngine() {
  plain[0x150] = 3;  // BSTriShape
  Field<const char*>(plain.data(), 0x10) = FixtureString("FemaleEyes");
  Field<const char*>(plain_fod.data(), 0x10) = FixtureString("FOD");
  Field<const float*>(plain_fod.data(), 0x18) = base.data();
  Field<uint32_t>(plain_fod.data(), 0x24) = 2;
  Field<uint8_t**>(plain.data(), 0x20) = &plain_extra;
  Field<uint16_t>(plain.data(), 0x28) = 1;
  // base vertices alone: the engine skips the shape itself
  CHECK(!StaticMorphPart(node.data()));
  Run(node.data());
  plain_morphs = true;
  CHECK(!std::strcmp(StaticMorphPart(node.data()), "FemaleEyes"));
  Play(fg.data());
  for (bool is_ready : {true, false}) {
    ready = is_ready;
    Reset();
    CHECK(ApplyMorphsHook(node.data(), 1) == 0);
    CHECK(orig_calls == 0 && lock_calls == 0 && lookups == 0);
    CHECK(Field<uint32_t>(fg.data(), 0x224) == 0);
  }
  ready = true;
  // a head without such a part is unaffected
  CHECK(!StaticMorphPart(node2.data()));
  Run(node2.data());
  // a dynamic shape with morph data is the normal case
  plain[0x150] = 4;
  CHECK(!StaticMorphPart(node.data()));
  plain[0x150] = 3;
  plain_morphs = false;
  playing.clear();
  Count();
}
}  // namespace

int main() {
  bfa::testing::address = TestAddress;
  skse::SetRuntime(skse::Version(1, 6, 1170));
  Seed();
  PlayedFace();
  OtherFaceWhilePlaying();
  NothingPlaying();
  NotReady();
  StaticMorphPartSkipsEngine();
  puts("ok: face and shape locks span the morph reset and the deltas");
}
