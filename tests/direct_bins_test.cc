// The shipped A2FDirect bins load as the plugin loads them: A2FDirect.bin is
// Mark for the v2.3 network, A2FDirect3.bin is Claire (female) and Mark
// (male) for the folded v3.0 network, and a line of either is as wide as a
// head mapped onto any of its faces expects.
// usage: direct_bins_test <A2FDirect.bin> <A2FDirect3.bin>
#include <cmath>
#include <cstdio>
#include <string>

#include "a2f/animation/direct.h"

namespace {
int failures = 0;
void Expect(bool ok, const char* label) {
  printf("%s %s\n", ok ? "ok" : "FAIL", label);
  if (!ok) ++failures;
}
bool Finite(const std::vector<float>& v) {
  for (float x : v)
    if (!std::isfinite(x)) return false;
  return !v.empty();
}
// Every array of the face is there, finite and of the model's sizes.
bool Whole(const direct::Model& m, const direct::Mark& f) {
  size_t rows = 0;
  for (int32_t r : f.row) rows += r >= 0;
  return f.nv && f.k && f.verts.size() == f.nv * 3u && f.jaw.size() == f.nv &&
         f.row.size() == f.nv && rows && f.basis.size() == rows * f.k * 3 &&
         f.proj.size() == static_cast<size_t>(f.k) * m.n_skin &&
         f.rest.size() == m.n_out && f.tongue_nv &&
         f.tongue_verts.size() == f.tongue_nv * 3u &&
         f.tongue_basis.size() ==
             static_cast<size_t>(f.tongue_nv) * a2f::kNumTongue * 3 &&
         Finite(f.verts) && Finite(f.jaw) && Finite(f.basis) &&
         Finite(f.proj) && Finite(f.rest) && Finite(f.tongue_verts) &&
         Finite(f.tongue_basis) && f.jaw_strength > 0;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    printf("usage: direct_bins_test <A2FDirect.bin> <A2FDirect3.bin>\n");
    return 2;
  }
  auto path = [](const char* s) {
    std::string n = s;
    return std::wstring(n.begin(), n.end());
  };
  std::string err;
  direct::Model m2, m3;
  Expect(m2.Load(path(argv[1]), err), "A2FDirect.bin loads");
  if (!err.empty()) printf("  %s\n", err.c_str());
  Expect(m2.network == 2 && m2.faces.size() == 1 &&
             m2.n_skin == a2f::kNumPca && m2.n_out == a2f::kNumOut &&
             Whole(m2, m2.faces[0]) && m2.faces[0].tongue_proj.empty(),
         "it is Mark for the v2.3 network");
  Expect(m2.faces.size() == 1 && &m2.Face(true) == &m2.Face(false),
         "women and men are both mapped onto him");
  Expect(m3.Load(path(argv[2]), err), "A2FDirect3.bin loads");
  if (!err.empty()) printf("  %s\n", err.c_str());
  bool two = m3.network == 3 && m3.faces.size() == 2 &&
             m3.n_skin == a2f::kLatent3 && m3.n_out == a2f::kNumOut3;
  for (size_t i = 0; two && i < 2; ++i)
    two = Whole(m3, m3.faces[i]) &&
          m3.faces[i].tongue_proj.size() ==
              static_cast<size_t>(a2f::kNumTongue) * a2f::kLatent3 &&
          Finite(m3.faces[i].tongue_proj);
  Expect(two, "it is two faces for the v3.0 network");
  if (two) {
    Expect(&m3.Face(true) == &m3.faces[0] && &m3.Face(false) == &m3.faces[1],
           "women are mapped onto the first (Claire), men onto the second");
    // lines of both sexes are as wide: a face's pose blends across them
    Expect(m3.faces[0].k == m3.faces[1].k, "both faces have as many "
                                          "coefficients");
    // The performance network emits a fixed width onto these same faces;
    // Coefficients drops a line whose width differs from the head's.
    Expect(m3.faces[0].k + a2f::kNumTongue + a2f::kNumJaw ==
               static_cast<uint32_t>(a2f::kPerformanceChannels),
           "performance lines are as wide as a v3 head expects");
    // Claire is the smaller face
    float hi[2] = {-INFINITY, -INFINITY}, lo[2] = {INFINITY, INFINITY};
    for (int f = 0; f < 2; ++f)
      for (uint32_t v = 0; v < m3.faces[f].nv; ++v) {
        hi[f] = std::fmax(hi[f], m3.faces[f].verts[v * 3]);
        lo[f] = std::fmin(lo[f], m3.faces[f].verts[v * 3]);
      }
    printf("  widths: female %.1f, male %.1f\n", hi[0] - lo[0], hi[1] - lo[1]);
    Expect(hi[0] - lo[0] < hi[1] - lo[1], "the female face is the narrower");
  }
  printf(failures ? "%d FAILED\n" : "all ok\n", failures);
  return failures != 0;
}
