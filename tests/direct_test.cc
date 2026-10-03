// Checks direct::FrameOffset on parts placed like real FaceGen meshes: a part
// other than the head skin is centred off the head, and that must not move it.
// ./build.sh test builds and runs it.
#include <cmath>
#include <cstdio>
#include <vector>

#include "../src/direct.h"

namespace {

int failures;

// A box of 8 corners around centre with half extents h.
std::vector<float> Box(float cx, float cy, float cz, float hx, float hy,
                       float hz) {
  std::vector<float> v;
  for (int i = 0; i < 8; ++i) {
    v.push_back(cx + (i & 1 ? hx : -hx));
    v.push_back(cy + (i & 2 ? hy : -hy));
    v.push_back(cz + (i & 4 ? hz : -hz));
  }
  return v;
}

void Expect(const direct::RefMesh& ref, const char* what,
            const std::vector<float>& v, float y, float z) {
  float off[3];
  direct::FrameOffset(ref, v.data(), static_cast<uint32_t>(v.size() / 3), off);
  bool ok = std::fabs(off[0]) < 1e-4f && std::fabs(off[1] - y) < 1e-4f &&
            std::fabs(off[2] - z) < 1e-4f;
  if (!ok) ++failures;
  printf("%s %s: offset %.2f %.2f %.2f, want 0 %.2f %.2f\n",
         ok ? "ok" : "FAIL", what, off[0], off[1], off[2], y, z);
}

}  // namespace

int main() {
  // only the reference's box matters: centred on the origin, head sized
  direct::RefMesh ref;
  ref.verts = Box(0, 0, 0, 6, 8, 11);
  ref.nv = 8;
  // shape-local, the head and parts around it (a long beard hangs ~8 below
  // and ~7 in front of the head's centre)
  Expect(ref, "local head", Box(0, 0, 0, 6, 8, 11), 0, 0);
  Expect(ref, "local brows", Box(0, 7, 4, 4, 1, 1), 0, 0);
  Expect(ref, "local long beard", Box(0, 6.6f, -7.8f, 3, 2, 5), 0, 0);
  // skeleton space (High Poly Head replacers): the vanilla head transform
  // (0, -1.55, 120.34) back, whatever the part's own box
  const float sy = -1.55f, sz = 120.34f;
  Expect(ref, "skeleton head", Box(0, sy, sz, 6, 8, 11), -sy, -sz);
  Expect(ref, "skeleton beard", Box(0, sy + 2.8f, sz - 1.8f, 4, 3, 4), -sy,
         -sz);
  Expect(ref, "skeleton brows", Box(0, sy + 5.3f, sz + 4, 4, 1, 1), -sy, -sz);
  Expect(ref, "skeleton eyes", Box(0, sy + 5.3f, sz + 3, 3, 1, 1), -sy, -sz);
  printf("%s: %d failures\n", failures ? "FAIL" : "ok", failures);
  return failures ? 1 : 0;
}
