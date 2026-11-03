// Independent front-surface oracle: rasterize actual triangles, not the
// correction's support vertices or its clearance formula.
#include "a2f/animation/direct.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Triangle = std::array<uint32_t, 3>;
constexpr int kWidth = 400, kHeight = 300;
constexpr double kX = -3.2, kZ = -4.6;
constexpr double kDX = 6.4 / (kWidth - 1), kDZ = 4.8 / (kHeight - 1);
const float kScales[direct::kNumRegions] = {1, 1, 1, 1, 1, 1};
void Require(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error(what);
}
uint32_t Uint(std::ifstream& in) {
  uint32_t value = 0;
  Require(bool(in.read(reinterpret_cast<char*>(&value), 4)), "truncated integer");
  return value;
}
void Floats(std::ifstream& in, std::vector<float>& v, size_t n) {
  v.resize(n);
  Require(bool(in.read(reinterpret_cast<char*>(v.data()), n * 4)), "truncated floats");
  Require(std::all_of(v.begin(), v.end(), [](float x) { return std::isfinite(x); }),
          "nonfinite input");
}
std::vector<Triangle> Triangles(std::ifstream& in, uint32_t nv) {
  const uint32_t n = Uint(in);
  Require(n > 0 && n < 100000, "invalid triangle count");
  std::vector<Triangle> triangles(n);
  for (auto& t : triangles)
    for (auto& v : t) { v = Uint(in); Require(v < nv, "triangle index out of range"); }
  return triangles;
}
struct Surface {
  std::vector<float> vertices;
  std::vector<Triangle> triangles;
};
std::vector<double> Depth(const Surface& s) {
  std::vector<double> depth(kWidth * kHeight, -std::numeric_limits<double>::infinity());
  for (const auto& t : s.triangles) {
    const float *a = &s.vertices[t[0] * 3], *b = &s.vertices[t[1] * 3],
                *c = &s.vertices[t[2] * 3];
    const int x0 = std::max(0, int(std::ceil((std::min({a[0], b[0], c[0]}) - kX) / kDX)));
    const int x1 = std::min(kWidth - 1, int(std::floor((std::max({a[0], b[0], c[0]}) - kX) / kDX)));
    const int z0 = std::max(0, int(std::ceil((std::min({a[2], b[2], c[2]}) - kZ) / kDZ)));
    const int z1 = std::min(kHeight - 1, int(std::floor((std::max({a[2], b[2], c[2]}) - kZ) / kDZ)));
    const double det = double(b[2] - c[2]) * (a[0] - c[0]) +
                       double(c[0] - b[0]) * (a[2] - c[2]);
    if (std::fabs(det) < 1e-8) continue;
    for (int z = z0; z <= z1; ++z)
      for (int x = x0; x <= x1; ++x) {
        const double px = kX + x * kDX - c[0], pz = kZ + z * kDZ - c[2];
        const double u = ((b[2] - c[2]) * px + (c[0] - b[0]) * pz) / det;
        const double v = ((c[2] - a[2]) * px + (a[0] - c[0]) * pz) / det;
        if (u < 0 || v < 0 || u + v > 1) continue;
        double& d = depth[z * kWidth + x];
        d = std::max(d, u * a[1] + v * b[1] + (1 - u - v) * c[1]);
      }
  }
  return depth;
}
struct Measure { uint32_t pixels = 0; double peak = 0; };
Measure Penetration(const Surface& face, const Surface& lining) {
  auto f = Depth(face), m = Depth(lining);
  Measure result;
  for (size_t i = 0; i < f.size(); ++i)
    if (std::isfinite(f[i]) && std::isfinite(m[i]) && f[i] > 5 && m[i] > f[i] + .01) {
      ++result.pixels;
      result.peak = std::max(result.peak, m[i] - f[i]);
    }
  return result;
}
struct Fixture {
  direct::Head head;
  Surface face, lining;
  uint32_t protected_begin = 0;
};
Measure CheckFrame(const Fixture& f, const float* c, const float* expr,
                   direct::TongueMode mode, bool uncorrected = false) {
  auto legacy = f.head;
  legacy.shapes[1].walls.clear();
  std::vector<std::vector<float>> actual, reference;
  direct::Evaluate(uncorrected ? legacy : f.head, c, kScales, expr, nullptr, mode, actual);
  direct::Evaluate(legacy, c, kScales, expr, nullptr, mode, reference);
  Require(actual[0] == reference[0], "outer face changed");
  const bool neutral = !c || std::all_of(c, c + f.head.k + direct::kNumMouthCoef,
                                       [](float x) { return x == 0; });
  for (size_t i = 0; i < actual[1].size(); ++i) {
    Require(std::isfinite(actual[1][i]), "nonfinite mouth output");
    if (i >= f.protected_begin * 3 || i % 3 != 1 || neutral)
      Require(actual[1][i] == reference[1][i], "teeth/tongue/opening/native expressions changed");
    else
      Require(actual[1][i] <= reference[1][i] + 1e-6f, "lining pushed forward");
  }
  Surface face = f.face, lining = f.lining;
  for (size_t i = 0; i < face.vertices.size(); ++i) face.vertices[i] += actual[0][i];
  for (size_t i = 0; i < lining.vertices.size(); ++i) lining.vertices[i] += actual[1][i];
  return Penetration(face, lining);
}

// A sloped face slides sideways over a nearby wall. A Y-only follow or an
// averaged displacement cannot keep the wall inside this independently defined
// surface. The removed-correction control must visibly cross it.
void Synthetic() {
  Fixture f;
  f.face = {{-2, 8, -2, 2, 12, -2, 2, 12, 2, -2, 8, 2}, {{0, 1, 2}, {0, 2, 3}}};
  f.lining = {{-1, 8.8f, -1, 1, 10.8f, -1, 0, 9.8f, 1,
               0, 8, -1, 0, 8, -2}, {{0, 1, 2}}};
  f.protected_begin = 3;
  auto& h = f.head;
  h.k = 1; h.nv = 4; h.skin = 0; h.row = {0, 1, 2, 3};
  h.basis = {1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0};
  h.region_w.assign(4 * direct::kNumSkinRegions, 0);
  for (int i = 0; i < 4; ++i) h.region_w[i * direct::kNumSkinRegions + direct::kMouth] = 1;
  const float jaw[direct::kNumJawPoints * 3] = {-1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1};
  std::copy(std::begin(jaw), std::end(jaw), h.jaw_points);
  h.shapes.resize(2);
  h.shapes[0].part = direct::Part::kHead; h.shapes[0].nv = 4;
  auto& mouth = h.shapes[1]; mouth.part = direct::Part::kMouth; mouth.nv = 5;
  mouth.pos = f.lining.vertices; mouth.jaw.assign(5, 0); mouth.model_jaw.assign(5, 0);
  for (uint32_t i = 0; i < 3; ++i) {
    direct::Head::Shape::Wall wall{}; wall.vertex = i; wall.count = 4;
    for (uint32_t j = 0; j < 4; ++j) wall.skin[j] = j;
    mouth.walls.push_back(wall);
  }
  for (size_t si = 0; si < 2; ++si) {
    auto& d = h.shapes[si].expr[direct::kHappy - direct::kExpression];
    d.assign(h.shapes[si].nv * 3, 0);
    for (size_t i = 1; i < d.size(); i += 3) d[i] = -.2f;
  }
  std::vector<float> c(h.k + direct::kNumMouthCoef, 0);
  float expr[direct::kNumExpressions] = {};
  for (auto mode : {direct::TongueMode::kAuthored, direct::TongueMode::kModel})
    for (float happy : {0.f, 1.f}) {
      expr[direct::kHappy - direct::kExpression] = happy;
      CheckFrame(f, nullptr, expr, mode);
      for (float amplitude : {0.f, .25f, .5f, 1.f, 2.f}) {
        c[0] = amplitude;
        Require(CheckFrame(f, c.data(), expr, mode).pixels == 0, "synthetic wall crosses sliding face");
        if (amplitude >= .5f)
          Require(CheckFrame(f, c.data(), expr, mode, true).pixels > 100,
                  "negative control did not expose wall");
      }
    }
  std::cout << "sliding-face surface oracle, negative control, both modes and native expressions: pass\n";
}

void Assets(int argc, char** argv) {
  Require(argc == 8, "usage: test shape.bin triangles.bin speech.coef model.bin pixel_limit depth_limit require_control");
  std::ifstream in(argv[1], std::ios::binary);
  Require(Uint(in) == 2, "expected head and mouth");
  std::vector<direct::ShapeIn> shapes(2);
  for (size_t i = 0; i < 2; ++i) {
    auto& s = shapes[i]; s.skin = i == 0;
    s.part = i == 0 ? direct::Part::kHead : direct::Part::kMouth;
    s.nv = Uint(in); Require(s.nv > 0 && s.nv < 100000, "invalid vertex count");
    Floats(in, s.verts, s.nv * 3);
    s.morphs.resize(direct::kNumMorphs);
    for (auto& m : s.morphs) Floats(in, m, s.nv * 3);
  }
  Require(in.peek() == EOF, "trailing shape bytes");
  Fixture f; f.face.vertices = shapes[0].verts; f.lining.vertices = shapes[1].verts;
  std::ifstream topology(argv[2], std::ios::binary);
  f.protected_begin = Uint(topology);
  Require(f.protected_begin > 0 && f.protected_begin < shapes[1].nv, "invalid anatomical boundary");
  f.face.triangles = Triangles(topology, shapes[0].nv);
  f.lining.triangles = Triangles(topology, f.protected_begin);
  Require(topology.peek() == EOF, "trailing topology bytes");
  direct::Model model; std::string why, model_path = argv[4];
  Require(model.Load(std::wstring(model_path.begin(), model_path.end()), why), why);
  Require(direct::BuildHead(model.Face(true), shapes, f.head, why), why);
  std::ifstream speech(argv[3], std::ios::binary);
  const uint32_t frames = Uint(speech), dim = Uint(speech);
  std::vector<float> fps, coefficients;
  Floats(speech, fps, 1);
  Require(frames > 0 && frames < 10000 && fps[0] > 0 &&
          dim == f.head.k + direct::kNumMouthCoef, "invalid speech dimensions");
  Floats(speech, coefficients, size_t(frames) * dim);
  Require(speech.peek() == EOF, "trailing speech bytes");
  const uint32_t limit = static_cast<uint32_t>(std::stoul(argv[5]));
  const double depth_limit = std::stod(argv[6]);
  const bool require_control = std::stoi(argv[7]) != 0;
  std::vector<float> neutral(dim, 0);
  float expr[direct::kNumExpressions] = {};
  uint32_t negative_peak = 0;
  for (auto mode : {direct::TongueMode::kAuthored, direct::TongueMode::kModel})
    for (float happy : {0.f, 1.f}) {
      expr[direct::kHappy - direct::kExpression] = happy;
      CheckFrame(f, nullptr, expr, mode);
      CheckFrame(f, neutral.data(), expr, mode);
      uint32_t worst = 0; double peak = 0;
      for (uint32_t frame = 0; frame < frames; ++frame) {
        const float* c = &coefficients[frame * dim];
        const auto m = CheckFrame(f, c, expr, mode);
        worst = std::max(worst, m.pixels); peak = std::max(peak, m.peak);
        Require(m.pixels <= limit && m.peak <= depth_limit,
                "mouth lining protrudes at frame " + std::to_string(frame) +
                ": pixels=" + std::to_string(m.pixels) + " depth=" + std::to_string(m.peak));
        if (require_control && mode == direct::TongueMode::kAuthored && happy == 0)
          negative_peak = std::max(negative_peak, CheckFrame(f, c, expr, mode, true).pixels);
      }
      std::cout << "mode=" << int(mode) << " happy=" << happy << " frames=" << frames
                << " worst_pixels=" << worst << " peak_depth=" << peak << '\n';
    }
  if (require_control) Require(negative_peak > 1000, "broken-mouth negative control did not reproduce");
  std::cout << "uncorrected_peak=" << negative_peak << " face/teeth/tongue/opening preserved\n";
}
}  // namespace
int main(int argc, char** argv) {
  try { if (argc == 1) Synthetic(); else Assets(argc, argv); }
  catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
  return 0;
}
