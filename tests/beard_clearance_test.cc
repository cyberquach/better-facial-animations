// Facial hair must not sink into the face it lies on. An independent ray
// measure of the deformed geometry: no production mapping code decides what
// counts. Without arguments: the measure's own controls. With a pinned local
// capture: production BuildHead/Evaluate over recorded speech.
// No third-party geometry is compiled into this test or distributed with it.
#include "a2f/animation/direct.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

void Check(bool ok, const std::string& message) {
  if (ok) return;
  std::cerr << message << '\n';
  std::exit(1);
}

struct P3 {
  float x, y, z;
};
P3 operator-(P3 a, P3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
P3 operator+(P3 a, P3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
P3 operator*(P3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
float Dot(P3 a, P3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
P3 Cross(P3 a, P3 b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
float Norm(P3 a) { return std::sqrt(Dot(a, a)); }
P3 At(const std::vector<float>& v, uint32_t i) {
  return {v[i * 3], v[i * 3 + 1], v[i * 3 + 2]};
}

// Triangles in a uniform grid; the rays are short, so a query visits few
// cells.
class Grid {
 public:
  Grid(const std::vector<float>& verts, const std::vector<uint32_t>& tris,
       float cell)
      : v_(verts), t_(tris), cell_(cell) {
    lo_ = {1e30f, 1e30f, 1e30f};
    P3 hi{-1e30f, -1e30f, -1e30f};
    for (uint32_t i = 0; i < verts.size() / 3; ++i) {
      const P3 p = At(verts, i);
      lo_ = {std::min(lo_.x, p.x), std::min(lo_.y, p.y), std::min(lo_.z, p.z)};
      hi = {std::max(hi.x, p.x), std::max(hi.y, p.y), std::max(hi.z, p.z)};
    }
    n_[0] = Cells(hi.x - lo_.x);
    n_[1] = Cells(hi.y - lo_.y);
    n_[2] = Cells(hi.z - lo_.z);
    start_.assign(size_t(n_[0]) * n_[1] * n_[2] + 1, 0);
    for (int pass = 0; pass < 2; ++pass) {
      std::vector<uint32_t> fill(start_.begin(), start_.end() - 1);
      for (uint32_t t = 0; t < tris.size() / 3; ++t) {
        int a[3], b[3];
        Bounds(t, a, b);
        for (int x = a[0]; x <= b[0]; ++x)
          for (int y = a[1]; y <= b[1]; ++y)
            for (int z = a[2]; z <= b[2]; ++z) {
              const size_t c = Index(x, y, z);
              if (pass)
                items_[fill[c]++] = t;
              else
                ++start_[c + 1];
            }
      }
      if (!pass) {
        for (size_t c = 1; c < start_.size(); ++c) start_[c] += start_[c - 1];
        items_.resize(start_.back());
      }
    }
  }
  // First hit of origin + t * dir, 0 < t <= tmax; tmax + 1: none. front:
  // whether the ray meets the triangle's e1 x e2 side.
  float Ray(P3 o, P3 d, float tmax, bool* front = nullptr) const {
    const P3 e = o + d * tmax;
    int a[3], b[3];
    Cell({std::min(o.x, e.x), std::min(o.y, e.y), std::min(o.z, e.z)}, a);
    Cell({std::max(o.x, e.x), std::max(o.y, e.y), std::max(o.z, e.z)}, b);
    float best = tmax + 1;
    for (int x = a[0]; x <= b[0]; ++x)
      for (int y = a[1]; y <= b[1]; ++y)
        for (int z = a[2]; z <= b[2]; ++z) {
          const size_t c = Index(x, y, z);
          for (uint32_t i = start_[c]; i < start_[c + 1]; ++i) {
            const uint32_t t = items_[i];
            const P3 p0 = At(v_, t_[t * 3]), e1 = At(v_, t_[t * 3 + 1]) - p0,
                     e2 = At(v_, t_[t * 3 + 2]) - p0;
            const P3 h = Cross(d, e2);
            const float det = Dot(e1, h);  // = -d . (e1 x e2)
            if (std::fabs(det) < 1e-12f) continue;
            const float inv = 1 / det;
            const P3 s = o - p0;
            const float u = Dot(s, h) * inv;
            if (u < -1e-4f || u > 1 + 1e-4f) continue;
            const P3 q = Cross(s, e1);
            const float w = Dot(d, q) * inv;
            if (w < -1e-4f || u + w > 1 + 1e-4f) continue;
            const float hit = Dot(e2, q) * inv;
            if (hit > 0 && hit <= tmax && hit < best) {
              best = hit;
              if (front) *front = det > 0;
            }
          }
        }
    return best;
  }
  // Closest point of the surface within radius: its triangle and weights.
  bool Closest(P3 p, float radius, uint32_t& tri, float bary[3]) const {
    int a[3], b[3];
    Cell({p.x - radius, p.y - radius, p.z - radius}, a);
    Cell({p.x + radius, p.y + radius, p.z + radius}, b);
    float best = radius * radius;
    bool found = false;
    for (int x = a[0]; x <= b[0]; ++x)
      for (int y = a[1]; y <= b[1]; ++y)
        for (int z = a[2]; z <= b[2]; ++z) {
          const size_t c = Index(x, y, z);
          for (uint32_t i = start_[c]; i < start_[c + 1]; ++i) {
            const uint32_t t = items_[i];
            float w[3];
            const P3 q =
                ClosestOnTriangle(p, At(v_, t_[t * 3]), At(v_, t_[t * 3 + 1]),
                                  At(v_, t_[t * 3 + 2]), w);
            const float d2 = Dot(q - p, q - p);
            if (d2 <= best) {
              best = d2;
              tri = t;
              std::copy_n(w, 3, bary);
              found = true;
            }
          }
        }
    return found;
  }

 private:
  static P3 ClosestOnTriangle(P3 p, P3 a, P3 b, P3 c, float w[3]) {
    const P3 ab = b - a, ac = c - a, ap = p - a;
    const float d1 = Dot(ab, ap), d2 = Dot(ac, ap);
    auto set = [&](float u, float v, float x) {
      w[0] = u;
      w[1] = v;
      w[2] = x;
      return a * u + b * v + c * x;
    };
    if (d1 <= 0 && d2 <= 0) return set(1, 0, 0);
    const P3 bp = p - b;
    const float d3 = Dot(ab, bp), d4 = Dot(ac, bp);
    if (d3 >= 0 && d4 <= d3) return set(0, 1, 0);
    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0) {
      const float v = d1 / (d1 - d3);
      return set(1 - v, v, 0);
    }
    const P3 cp = p - c;
    const float d5 = Dot(ab, cp), d6 = Dot(ac, cp);
    if (d6 >= 0 && d5 <= d6) return set(0, 0, 1);
    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0) {
      const float x = d2 / (d2 - d6);
      return set(1 - x, 0, x);
    }
    const float va = d3 * d6 - d5 * d4;
    if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
      const float x = (d4 - d3) / ((d4 - d3) + (d5 - d6));
      return set(0, 1 - x, x);
    }
    const float den = 1 / (va + vb + vc), v = vb * den, x = vc * den;
    return set(1 - v - x, v, x);
  }
  int Cells(float extent) const {
    return std::max(1, static_cast<int>(extent / cell_) + 1);
  }
  void Cell(P3 p, int c[3]) const {
    c[0] = std::clamp(static_cast<int>((p.x - lo_.x) / cell_), 0, n_[0] - 1);
    c[1] = std::clamp(static_cast<int>((p.y - lo_.y) / cell_), 0, n_[1] - 1);
    c[2] = std::clamp(static_cast<int>((p.z - lo_.z) / cell_), 0, n_[2] - 1);
  }
  size_t Index(int x, int y, int z) const {
    return (size_t(x) * n_[1] + y) * n_[2] + z;
  }
  void Bounds(uint32_t t, int a[3], int b[3]) const {
    P3 lo = At(v_, t_[t * 3]), hi = lo;
    for (int j = 1; j < 3; ++j) {
      const P3 p = At(v_, t_[t * 3 + j]);
      lo = {std::min(lo.x, p.x), std::min(lo.y, p.y), std::min(lo.z, p.z)};
      hi = {std::max(hi.x, p.x), std::max(hi.y, p.y), std::max(hi.z, p.z)};
    }
    Cell(lo, a);
    Cell(hi, b);
  }
  const std::vector<float>& v_;
  const std::vector<uint32_t>& t_;
  float cell_;
  P3 lo_;
  int n_[3];
  std::vector<uint32_t> start_, items_;
};

constexpr float kCell = .75f;
constexpr float kReach = 1.5f;  // how deep under the skin a sample is sought

// Depth of p under the surface along dir: the first surface met is seen from
// behind (outward: +1 when e1 x e2 points out of the head, else -1). 0: free.
float Burial(const Grid& grid, P3 p, P3 dir, float outward) {
  bool front = false;
  const float hit = grid.Ray(p, dir, kReach, &front);
  // From inside, the ray leaves through the side e1 x e2 points away from
  // when the triangles face outwards.
  return hit <= kReach && front == (outward < 0) ? hit : 0.f;
}

// +1 when the triangles' e1 x e2 points out of the head. The foremost vertex
// (nose tip, +y) faces forwards.
float Outward(const std::vector<float>& verts,
              const std::vector<uint32_t>& tris) {
  uint32_t tip = 0;
  for (uint32_t i = 1; i < verts.size() / 3; ++i)
    if (verts[i * 3 + 1] > verts[tip * 3 + 1]) tip = i;
  float forward = 0;
  for (size_t t = 0; t < tris.size(); t += 3)
    if (tris[t] == tip || tris[t + 1] == tip || tris[t + 2] == tip)
      forward += Cross(At(verts, tris[t + 1]) - At(verts, tris[t]),
                       At(verts, tris[t + 2]) - At(verts, tris[t]))
                     .y;
  return forward > 0 ? 1.f : -1.f;
}

struct Surface {
  std::vector<float> verts;     // rest
  std::vector<uint32_t> tris;
  std::vector<std::vector<float>> frames;  // deltas, one per frame
};

struct Result {
  uint32_t samples = 0, visible = 0, clip10 = 0, clip20 = 0, clip40 = 0;
  uint32_t stretched = 0, squashed = 0;  // edges beyond 2x / under .5x
  float deepest = 0;
};

// Seven samples per hair triangle. A sample counts while it is visible from
// outside at rest; it clips in a frame when it lies under the skin along the
// rest normal of the skin nearest to it and the skin it rests on is not itself
// covered by other skin (pressed lips, closed lids).
Result Measure(const Surface& skin, const Surface& hair) {
  static const float kBary[7][3] = {
      {1, 0, 0},     {0, 1, 0},     {0, 0, 1},          {.5f, .5f, 0},
      {0, .5f, .5f}, {.5f, 0, .5f}, {1 / 3.f, 1 / 3.f, 1 / 3.f}};
  Result out;
  const float outward = Outward(skin.verts, skin.tris);
  const uint32_t snv = static_cast<uint32_t>(skin.verts.size() / 3);
  std::vector<float> normals(skin.verts.size(), 0.f);
  for (size_t t = 0; t < skin.tris.size(); t += 3) {
    const P3 c = Cross(At(skin.verts, skin.tris[t + 1]) -
                           At(skin.verts, skin.tris[t]),
                       At(skin.verts, skin.tris[t + 2]) -
                           At(skin.verts, skin.tris[t]));
    for (int j = 0; j < 3; ++j) {
      float* n = &normals[size_t{skin.tris[t + j]} * 3];
      n[0] += c.x;
      n[1] += c.y;
      n[2] += c.z;
    }
  }
  for (uint32_t i = 0; i < snv; ++i) {
    const float length = Norm(At(normals, i));
    for (int c = 0; c < 3 && length > 0; ++c) normals[i * 3 + c] /= length;
  }
  struct Sample {
    uint32_t corner[3];
    const float* bary;
    P3 normal;
    uint32_t base;  // skin triangle under it
    float base_bary[3];
    float worst = 0;
  };
  std::vector<Sample> samples;
  {
    const Grid rest(skin.verts, skin.tris, kCell);
    for (size_t t = 0; t < hair.tris.size(); t += 3)
      for (const auto& bary : kBary) {
        ++out.samples;
        Sample s{{hair.tris[t], hair.tris[t + 1], hair.tris[t + 2]}, bary};
        P3 p{};
        for (int j = 0; j < 3; ++j)
          p = p + At(hair.verts, s.corner[j]) * bary[j];
        if (!rest.Closest(p, 3.f, s.base, s.base_bary)) continue;
        P3 n{};
        for (int j = 0; j < 3; ++j)
          n = n + At(normals, skin.tris[s.base * 3 + j]) * s.base_bary[j];
        if (Norm(n) < 1e-6f) continue;
        s.normal = n * (outward / Norm(n));
        // Visible: of the normal and eight directions 35 degrees off it, at
        // least six leave the head without meeting skin. The inside of the
        // mouth and of an eye socket sees out through a slit at most, and so
        // does anything under the skin.
        const P3 axis = std::fabs(s.normal.z) < .9f ? P3{0, 0, 1} : P3{1, 0, 0};
        P3 a = Cross(s.normal, axis);
        a = a * (1 / Norm(a));
        const P3 b = Cross(s.normal, a);
        int escaped = 0;
        for (int k = 0; k < 9; ++k) {
          P3 dir = s.normal;
          if (k) {
            const float angle = (k - 1) * 0.78539816f;
            dir = s.normal + (a * std::cos(angle) + b * std::sin(angle)) * .7f;
            dir = dir * (1 / Norm(dir));
          }
          bool front = false;
          const bool clear =
              rest.Ray(p + s.normal * .06f, dir, 4.f, &front) > 4.f;
          escaped += clear;
          // Under a back face (a fur fin, the inside of a fold): not outside.
          if (!clear && front == (outward < 0)) escaped = -9;
        }
        if (escaped < 6 || Burial(rest, p, s.normal, outward) > 0) continue;
        samples.push_back(s);
      }
  }
  out.visible = static_cast<uint32_t>(samples.size());
  // Edges of the hair, each once.
  std::vector<std::pair<uint32_t, uint32_t>> edges;
  for (size_t t = 0; t < hair.tris.size(); t += 3)
    for (int j = 0; j < 3; ++j) {
      uint32_t a = hair.tris[t + j], b = hair.tris[t + (j + 1) % 3];
      if (a > b) std::swap(a, b);
      if (Norm(At(hair.verts, a) - At(hair.verts, b)) > .05f)
        edges.emplace_back(a, b);
    }
  std::sort(edges.begin(), edges.end());
  edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
  std::vector<float> longest(edges.size(), 1.f), shortest(edges.size(), 1.f);
  std::vector<float> moved(skin.verts.size()), hair_moved(hair.verts.size());
  for (size_t f = 0; f < skin.frames.size(); ++f) {
    for (size_t i = 0; i < moved.size(); ++i)
      moved[i] = skin.verts[i] + skin.frames[f][i];
    for (size_t i = 0; i < hair_moved.size(); ++i)
      hair_moved[i] = hair.verts[i] + hair.frames[f][i];
    const Grid grid(moved, skin.tris, kCell);
    for (auto& s : samples) {
      P3 p{}, base{};
      for (int j = 0; j < 3; ++j) {
        p = p + At(hair_moved, s.corner[j]) * s.bary[j];
        base = base + At(moved, skin.tris[s.base * 3 + j]) * s.base_bary[j];
      }
      const float depth = Burial(grid, p, s.normal, outward);
      if (depth <= s.worst) continue;
      if (Burial(grid, base + s.normal * .02f, s.normal, outward) > 0) continue;
      s.worst = depth;
    }
    for (size_t e = 0; e < edges.size(); ++e) {
      const float ratio =
          Norm(At(hair_moved, edges[e].first) -
               At(hair_moved, edges[e].second)) /
          Norm(At(hair.verts, edges[e].first) - At(hair.verts, edges[e].second));
      longest[e] = std::max(longest[e], ratio);
      shortest[e] = std::min(shortest[e], ratio);
    }
  }
  for (const auto& s : samples) {
    out.clip10 += s.worst > .1f;
    out.clip20 += s.worst > .2f;
    out.clip40 += s.worst > .4f;
    out.deepest = std::max(out.deepest, s.worst);
  }
  for (size_t e = 0; e < edges.size(); ++e) {
    out.stretched += longest[e] > 2;
    out.squashed += shortest[e] < .5f;
  }
  return out;
}

// ---- the measure's own controls ----

// A patch of a cylinder about the z axis facing +y, n x n vertices.
Surface Patch(float radius, int n, bool flip = false) {
  Surface s;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      const float angle = -1.f + 2.f * j / (n - 1);
      s.verts.insert(s.verts.end(),
                     {radius * std::sin(angle), radius * std::cos(angle),
                      -4.f + 8.f * i / (n - 1)});
    }
  for (int i = 0; i + 1 < n; ++i)
    for (int j = 0; j + 1 < n; ++j) {
      const uint32_t a = i * n + j, b = a + 1, c = a + n, d = c + 1;
      const uint32_t quad[6] = {a, b, c, b, d, c};
      for (int k = 0; k < 6; k += 3) {
        s.tris.push_back(quad[k]);
        s.tris.push_back(quad[k + (flip ? 2 : 1)]);
        s.tris.push_back(quad[k + (flip ? 1 : 2)]);
      }
    }
  return s;
}

void SelfTest() {
  for (const bool flip : {false, true}) {
    Surface skin = Patch(6, 41, flip);
    // a small beard patch .05 above the middle of the skin
    Surface hair = Patch(6.05f, 41);
    {
      Surface part;
      const int n = 41;
      std::vector<int> index(n * n, -1);
      for (int i = 14; i < 27; ++i)
        for (int j = 14; j < 27; ++j) {
          index[i * n + j] = static_cast<int>(part.verts.size() / 3);
          for (int c = 0; c < 3; ++c)
            part.verts.push_back(hair.verts[(i * n + j) * 3 + c]);
        }
      for (size_t t = 0; t < hair.tris.size(); t += 3) {
        const int a = index[hair.tris[t]], b = index[hair.tris[t + 1]],
                  c = index[hair.tris[t + 2]];
        if (a < 0 || b < 0 || c < 0) continue;
        part.tris.insert(part.tris.end(),
                         {uint32_t(a), uint32_t(b), uint32_t(c)});
      }
      hair = part;
    }
    const std::vector<float> still_skin(skin.verts.size(), 0.f),
        still_hair(hair.verts.size(), 0.f);
    auto radial = [](const Surface& s, float amount) {
      std::vector<float> d(s.verts.size(), 0.f);
      for (size_t i = 0; i < d.size(); i += 3) {
        const float r = std::hypot(s.verts[i], s.verts[i + 1]);
        d[i] = amount * s.verts[i] / r;
        d[i + 1] = amount * s.verts[i + 1] / r;
      }
      return d;
    };
    // Hair that moves with its skin does not clip; neither does hair the
    // skin moves away from.
    skin.frames = {still_skin, radial(skin, .5f), radial(skin, -.5f)};
    hair.frames = {still_hair, radial(hair, .5f), still_hair};
    Result r = Measure(skin, hair);
    Check(r.visible == r.samples && r.samples > 1000, "visible hair not seen");
    Check(r.clip10 == 0 && r.deepest < .01f, "moving together reads as clipping");
    // The skin bulging .5 through still hair buries all of it .45 deep.
    skin.frames = {still_skin, radial(skin, .5f)};
    hair.frames = {still_hair, still_hair};
    r = Measure(skin, hair);
    Check(r.clip40 == r.visible && std::fabs(r.deepest - .45f) < .02f,
          "skin through still hair not measured");
    // Hair pulled to 2.5 times its height along the face stays above the
    // skin; the stretching is read off its edges.
    std::vector<float> stretch(hair.verts.size(), 0.f);
    for (size_t i = 0; i < stretch.size(); i += 3)
      stretch[i + 2] = 1.5f * hair.verts[i + 2];
    skin.frames = {still_skin, still_skin};
    hair.frames = {still_hair, stretch};
    r = Measure(skin, hair);
    Check(r.stretched > 0 && r.squashed == 0 && r.clip10 == 0,
          "stretched edges not counted");
    // Hair under the skin at rest is not visible and is not counted.
    Surface under = hair;
    under.frames = {still_hair, still_hair};
    skin.frames = {still_skin, radial(skin, .5f)};
    for (size_t i = 0; i < under.verts.size(); i += 3) {
      under.verts[i] *= 5.8f / 6.05f;
      under.verts[i + 1] *= 5.8f / 6.05f;
    }
    r = Measure(skin, under);
    Check(r.visible == 0 && r.clip10 == 0, "hidden hair counted");
    // Skin covered by other skin (a second sheet closing over it, as a lip
    // over a lip) hides what lies on it: not the hair's clipping.
    Surface lips = skin;
    const uint32_t base = static_cast<uint32_t>(lips.verts.size() / 3);
    Surface cover = Patch(8.5f, 41, flip);
    lips.verts.insert(lips.verts.end(), cover.verts.begin(), cover.verts.end());
    for (uint32_t index : cover.tris) lips.tris.push_back(base + index);
    std::vector<float> close(lips.verts.size(), 0.f);
    // The cover rests far out of sight (radius 25.5) and closes to .5 over
    // the skin, .45 over the hair.
    for (size_t i = base * 3; i < lips.verts.size(); i += 3) {
      lips.verts[i] *= 3;
      lips.verts[i + 1] *= 3;
      close[i] = lips.verts[i] * (6.5f / 25.5f - 1);
      close[i + 1] = lips.verts[i + 1] * (6.5f / 25.5f - 1);
    }
    lips.frames = {std::vector<float>(lips.verts.size(), 0.f), close};
    hair.frames = {still_hair, still_hair};
    r = Measure(lips, hair);
    Check(r.visible == r.samples && r.clip10 == 0,
          "skin covering skin read as hair clipping");
  }
  std::cout << "clearance measure controls passed\n";
}

// ---- pinned captures ----

uint32_t ReadUint(std::ifstream& input) {
  unsigned char bytes[4];
  Check(static_cast<bool>(input.read(reinterpret_cast<char*>(bytes), 4)),
        "truncated fixture integer");
  return uint32_t{bytes[0]} | (uint32_t{bytes[1]} << 8) |
         (uint32_t{bytes[2]} << 16) | (uint32_t{bytes[3]} << 24);
}

float ReadFloat(std::ifstream& input) {
  const uint32_t bits = ReadUint(input);
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  Check(std::isfinite(value), "non-finite fixture value");
  return value;
}

std::string ReadString(std::ifstream& input) {
  const uint32_t size = ReadUint(input);
  Check(size > 0 && size <= 1024, "invalid fixture name length");
  std::string value(size, '\0');
  Check(static_cast<bool>(input.read(value.data(), size)), "truncated name");
  return value;
}

std::vector<float> ReadFloats(std::ifstream& input, size_t count) {
  std::vector<float> values(count);
  for (float& value : values) value = ReadFloat(input);
  return values;
}

struct Capture {
  bool female = false;
  std::vector<direct::ShapeIn> shapes;
  std::vector<uint8_t> hair;  // per shape: facial hair or a scar, measured
};

// "BFABEARD" 1, female, shapes: name, vertices, part, skin, mouth candidate,
// unlisted, measured, positions, morph count (0, or every production morph
// with its name), triangles.
Capture ReadCapture(const char* path) {
  std::ifstream input(path, std::ios::binary);
  char magic[8];
  Check(static_cast<bool>(input.read(magic, sizeof(magic))) &&
            std::memcmp(magic, "BFABEARD", sizeof(magic)) == 0 &&
            ReadUint(input) == 1,
        "invalid capture header");
  Capture capture;
  capture.female = ReadUint(input) != 0;
  const uint32_t count = ReadUint(input);
  Check(count > 0 && count <= 64, "invalid shape count");
  capture.shapes.resize(count);
  for (auto& shape : capture.shapes) {
    shape.name = ReadString(input);
    shape.nv = ReadUint(input);
    Check(shape.nv > 0 && shape.nv <= 65535, "invalid vertex count");
    const uint32_t part = ReadUint(input);
    Check(part <= static_cast<uint32_t>(direct::Part::kNone), "invalid part");
    shape.part = static_cast<direct::Part>(part);
    shape.skin = ReadUint(input) != 0;
    shape.mouth_candidate = ReadUint(input) != 0;
    shape.unlisted = ReadUint(input) != 0;
    capture.hair.push_back(ReadUint(input) != 0);
    shape.verts = ReadFloats(input, size_t{shape.nv} * 3);
    const uint32_t morphs = ReadUint(input);
    Check(morphs == 0 || morphs == direct::kNumMorphs,
          "capture needs regeneration for the current morph inventory");
    if (morphs) shape.morphs.resize(direct::kNumMorphs);
    for (uint32_t m = 0; m < morphs; ++m) {
      Check(ReadString(input) == direct::kMorphIds[m].name,
            "capture morph order differs from production");
      auto values = ReadFloats(input, size_t{shape.nv} * 3);
      if (direct::Reads(shape.skin, shape.part, static_cast<int>(m),
                        shape.mouth_candidate))
        shape.morphs[m] = std::move(values);
    }
    const uint32_t triangles = ReadUint(input);
    Check(triangles <= 200000, "invalid triangle count");
    shape.triangles.resize(size_t{triangles} * 3);
    for (auto& index : shape.triangles) {
      const uint32_t value = ReadUint(input);
      Check(value < shape.nv, "triangle index out of range");
      index = static_cast<uint16_t>(value);
    }
  }
  Check(input.peek() == std::char_traits<char>::eof(), "trailing capture data");
  return capture;
}

// uint32 frames, uint32 coefficients, float fps, frame-major coefficients.
std::vector<std::vector<float>> ReadSpeech(const char* path, size_t dim) {
  std::ifstream input(path, std::ios::binary);
  const uint32_t frames = ReadUint(input), channels = ReadUint(input);
  ReadFloat(input);
  Check(frames > 0 && frames <= 100000 && channels == dim,
        "speech does not match the model");
  std::vector<std::vector<float>> speech(frames);
  for (auto& frame : speech) frame = ReadFloats(input, channels);
  return speech;
}

std::vector<uint32_t> Wide(const std::vector<uint16_t>& indices) {
  return std::vector<uint32_t>(indices.begin(), indices.end());
}

struct Totals {
  uint32_t visible = 0, clip20 = 0, stretched = 0;
};

// Every measured shape of the head over the speech. withdrawn: the measured
// shapes instead take the mapping this gate was written against (a vertex its
// own TRI opens by over .12 of the jaw follows the nearest skin opening over
// .25), which sinks cheek and jaw-line hair into the face.
Totals Replay(const Capture& capture, const direct::Model& model,
              const std::vector<std::vector<float>>& speech, bool triangles,
              bool withdrawn, bool print) {
  auto shapes = capture.shapes;
  if (!triangles)
    for (size_t i = 0; i < shapes.size(); ++i)
      if (!shapes[i].skin) shapes[i].triangles.clear();
  direct::Head head;
  std::string why;
  Check(direct::BuildHead(model.Face(capture.female), shapes, head, why), why);
  Check(head.shapes.size() == shapes.size(), "shape count changed");
  const auto& in_skin = capture.shapes[head.skin];
  Check(!in_skin.triangles.empty(), "capture has no skin triangles");
  Surface skin;
  skin.verts = in_skin.verts;
  skin.tris = Wide(in_skin.triangles);
  std::vector<Surface> hair(shapes.size());
  float expressions[direct::kNumExpressions] = {};
  std::vector<std::vector<float>> deltas;
  for (const auto& frame : speech) {
    Check(direct::Evaluate(head, frame.data(), head.auto_scale, expressions,
                           nullptr, direct::TongueMode::kAuthored, deltas),
          "evaluation failed");
    skin.frames.push_back(deltas[head.skin]);
    for (size_t i = 0; i < shapes.size(); ++i)
      if (capture.hair[i] && i != head.skin) hair[i].frames.push_back(deltas[i]);
  }
  const uint32_t snv = in_skin.nv;
  const auto& open = in_skin.morphs[direct::kBigAah];
  float jaw = 0;
  for (uint32_t i = 0; i < snv; ++i) jaw = std::max(jaw, -open[i * 3 + 2]);
  Totals totals;
  for (size_t i = 0; i < shapes.size(); ++i) {
    if (!capture.hair[i] || i == head.skin) continue;
    const auto& in = capture.shapes[i];
    if (in.triangles.empty()) continue;
    hair[i].verts = in.verts;
    hair[i].tris = Wide(in.triangles);
    if (withdrawn && in.morphs.size() > direct::kBigAah &&
        in.morphs[direct::kBigAah].size() == in.verts.size()) {
      for (uint32_t v = 0; v < in.nv; ++v) {
        if (-in.morphs[direct::kBigAah][v * 3 + 2] <= .12f * jaw) continue;
        uint32_t nearest[direct::kFollow];
        float distance[direct::kFollow];
        int found = 0;
        for (uint32_t k = 0; k < snv; ++k) {
          if (-open[k * 3 + 2] <= .25f * jaw) continue;
          const float d = Norm(At(in.verts, v) - At(in_skin.verts, k));
          int at = found;
          while (at > 0 && distance[at - 1] > d) --at;
          if (at >= direct::kFollow) continue;
          found = std::min(found + 1, int{direct::kFollow});
          for (int j = found - 1; j > at; --j) {
            nearest[j] = nearest[j - 1];
            distance[j] = distance[j - 1];
          }
          nearest[at] = k;
          distance[at] = d;
        }
        double weights[direct::kFollow], sum = 0;
        for (int j = 0; j < found; ++j)
          sum += weights[j] = 1 / ((distance[j] + .05) * (distance[j] + .05));
        for (auto& frame : hair[i].frames) {
          const auto& moved = skin.frames[&frame - hair[i].frames.data()];
          for (int c = 0; c < 3; ++c) {
            double value = 0;
            for (int j = 0; j < found; ++j)
              value += weights[j] / sum * moved[nearest[j] * 3 + c];
            frame[v * 3 + c] = static_cast<float>(value);
          }
        }
      }
    }
    const Result r = Measure(skin, hair[i]);
    totals.visible += r.visible;
    totals.clip20 += r.clip20;
    totals.stretched += r.stretched;
    if (print)
      std::cout << "  " << in.name << ": visible " << r.visible << " of "
                << r.samples << ", under the skin >.1 " << r.clip10 << " >.2 "
                << r.clip20 << " >.4 " << r.clip40 << " deepest " << r.deepest
                << ", edges >2x " << r.stretched << " <.5x " << r.squashed
                << '\n';
  }
  return totals;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 1) {
    SelfTest();
    return 0;
  }
  if (argc != 8) {
    std::cerr << "usage: beard_clearance_test [capture.bin A2FDirect3.bin "
                 "speech.coef min-visible max-clipped max-stretched "
                 "min-withdrawn-clipped]\n";
    return 2;
  }
  const Capture capture = ReadCapture(argv[1]);
  direct::Model model;
  std::string why;
  const std::string path = argv[2];
  Check(model.Load(std::wstring(path.begin(), path.end()), why), why);
  // The model's coefficient count is the head's; read it from a built head.
  direct::Head probe;
  Check(direct::BuildHead(model.Face(capture.female), capture.shapes, probe,
                          why),
        why);
  const auto speech =
      ReadSpeech(argv[3], size_t{probe.k} + direct::kNumMouthCoef);
  const uint32_t min_visible = std::strtoul(argv[4], nullptr, 10),
                 max_clipped = std::strtoul(argv[5], nullptr, 10),
                 max_stretched = std::strtoul(argv[6], nullptr, 10),
                 min_withdrawn = std::strtoul(argv[7], nullptr, 10);
  bool passed = true;
  for (const bool triangles : {true, false}) {
    std::cout << (triangles ? "with" : "without")
              << " follower triangles\n";
    const Totals totals =
        Replay(capture, model, speech, triangles, false, true);
    std::cout << "  total: visible " << totals.visible << ", under the skin >.2 "
              << totals.clip20 << " (limit " << max_clipped << "), edges >2x "
              << totals.stretched << " (limit " << max_stretched << ")\n";
    if (totals.visible < min_visible) {
      std::cerr << "too little visible hair measured\n";
      passed = false;
    }
    if (totals.clip20 > max_clipped) {
      std::cerr << "facial hair sinks into the face\n";
      passed = false;
    }
    if (totals.stretched > max_stretched) {
      std::cerr << "facial hair stretches\n";
      passed = false;
    }
  }
  const Totals control = Replay(capture, model, speech, true, true, false);
  std::cout << "withdrawn mapping control: under the skin >.2 " << control.clip20
            << " (at least " << min_withdrawn << ")\n";
  if (control.clip20 < min_withdrawn) {
    std::cerr << "the measure no longer sees the withdrawn mapping clip\n";
    passed = false;
  }
  return passed ? 0 : 1;
}
