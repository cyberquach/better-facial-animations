// A2F Direct model, evaluation and the reference -> target resampler
// declared in direct.h.
#include "direct.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace direct {
namespace {

// Same topology and positions within the NPC's FaceGen sculpt (~0.2 units).
constexpr float kIdentityMeanDist = 1.0f;
constexpr int kMinNeighbours = 6;

struct V3 {
  float x, y, z;
};
inline V3 At(const float* p) { return {p[0], p[1], p[2]}; }
inline V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline V3 operator*(V3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
inline float Dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline float Len(V3 a) { return std::sqrt(Dot(a, a)); }
inline V3 Unit(V3 a) { return a * (1 / std::max(Len(a), 1e-6f)); }

float Half(uint16_t h) {
  uint32_t s = static_cast<uint32_t>(h >> 15) << 31, e = (h >> 10) & 0x1F,
           m = h & 0x3FF, bits;
  if (e == 0) {
    if (!m) {
      bits = s;
    } else {  // subnormal
      e = 1;
      while (!(m & 0x400)) m <<= 1, --e;
      bits = s | ((e + 112) << 23) | ((m & 0x3FF) << 13);
    }
  } else if (e == 31) {
    bits = s | 0x7F800000 | (m << 13);
  } else {
    bits = s | ((e + 112) << 23) | (m << 13);
  }
  float f;
  memcpy(&f, &bits, 4);
  return f;
}

// Sequential reader over the file bytes; ok turns false on overrun.
struct Reader {
  const uint8_t* p;
  size_t left;
  bool ok = true;
  template <typename T>
  T Get() {
    T v{};
    if (left < sizeof v) {
      ok = false;
      left = 0;
      return v;
    }
    memcpy(&v, p, sizeof v);
    p += sizeof v;
    left -= sizeof v;
    return v;
  }
  template <typename T>
  void Vec(std::vector<T>& v, size_t n) {
    v.resize(n);
    for (size_t i = 0; i < n && ok; ++i) v[i] = Get<T>();
  }
  void Halves(std::vector<float>& v, size_t n) {
    v.resize(n);
    for (size_t i = 0; i < n && ok; ++i) v[i] = Half(Get<uint16_t>());
  }
  bool Mesh(RefMesh& m) {
    uint32_t nv = Get<uint32_t>(), nt = Get<uint32_t>();
    if (!ok || !nv || nv > 65535) return false;
    m.nv = nv;
    Vec(m.verts, static_cast<size_t>(nv) * 3);
    std::vector<uint16_t> t;
    Vec(t, static_cast<size_t>(nt) * 3);
    m.tris.assign(t.begin(), t.end());
    Vec(m.labels, nv);
    for (uint32_t i : m.tris)
      if (i >= nv) return false;
    return ok;
  }
};

// Closest point of p on triangle abc as barycentric weights (Ericson,
// Real-Time Collision Detection 5.1.5).
void ClosestOnTri(V3 p, V3 a, V3 b, V3 c, float w[3]) {
  V3 ab = b - a, ac = c - a, ap = p - a;
  float d1 = Dot(ab, ap), d2 = Dot(ac, ap);
  if (d1 <= 0 && d2 <= 0) {
    w[0] = 1, w[1] = 0, w[2] = 0;
    return;
  }
  V3 bp = p - b;
  float d3 = Dot(ab, bp), d4 = Dot(ac, bp);
  if (d3 >= 0 && d4 <= d3) {
    w[0] = 0, w[1] = 1, w[2] = 0;
    return;
  }
  float vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) {
    float v = d1 / (d1 - d3);
    w[0] = 1 - v, w[1] = v, w[2] = 0;
    return;
  }
  V3 cp = p - c;
  float d5 = Dot(ab, cp), d6 = Dot(ac, cp);
  if (d6 >= 0 && d5 <= d6) {
    w[0] = 0, w[1] = 0, w[2] = 1;
    return;
  }
  float vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) {
    float v = d2 / (d2 - d6);
    w[0] = 1 - v, w[1] = 0, w[2] = v;
    return;
  }
  float va = d3 * d6 - d5 * d4;
  if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
    float v = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    w[0] = 0, w[1] = 1 - v, w[2] = v;
    return;
  }
  float sum = va + vb + vc;
  if (sum <= 0) {  // degenerate
    w[0] = w[1] = w[2] = 1.f / 3;
    return;
  }
  float v = vb / sum, u = vc / sum;
  w[0] = 1 - v - u, w[1] = v, w[2] = u;
}

// Uniform grid over the reference vertices.
struct Grid {
  V3 lo;
  float cell;
  int n[3];
  std::vector<uint32_t> start, items;  // CSR per cell

  Grid(const std::vector<float>& verts, uint32_t nv) {
    V3 hi = At(verts.data());
    lo = hi;
    for (uint32_t i = 1; i < nv; ++i) {
      V3 p = At(&verts[i * 3]);
      lo = {std::min(lo.x, p.x), std::min(lo.y, p.y), std::min(lo.z, p.z)};
      hi = {std::max(hi.x, p.x), std::max(hi.y, p.y), std::max(hi.z, p.z)};
    }
    V3 ext = hi - lo;
    float vol = std::max(ext.x, 1e-3f) * std::max(ext.y, 1e-3f) *
                std::max(ext.z, 1e-3f);
    cell = std::max(std::cbrt(vol / nv), 1e-3f);
    const float* e = &ext.x;
    for (int k = 0; k < 3; ++k)
      n[k] = std::min(64, static_cast<int>(e[k] / cell) + 1);
    std::vector<uint32_t> cnt(static_cast<size_t>(n[0]) * n[1] * n[2] + 1, 0);
    std::vector<int> c(static_cast<size_t>(nv) * 3);
    for (uint32_t i = 0; i < nv; ++i) {
      Coord(At(&verts[i * 3]), &c[i * 3]);
      ++cnt[Cell(&c[i * 3]) + 1];
    }
    for (size_t i = 1; i < cnt.size(); ++i) cnt[i] += cnt[i - 1];
    start = cnt;
    items.resize(nv);
    for (uint32_t i = 0; i < nv; ++i) items[cnt[Cell(&c[i * 3])]++] = i;
  }
  void Coord(V3 p, int c[3]) const {
    V3 r = (p - lo) * (1 / cell);
    const float* f = &r.x;
    for (int k = 0; k < 3; ++k)
      c[k] = std::clamp(static_cast<int>(std::floor(f[k])), 0, n[k] - 1);
  }
  size_t Cell(const int c[3]) const {
    return (static_cast<size_t>(c[2]) * n[1] + c[1]) * n[0] + c[0];
  }
  // Vertices in the cells at Chebyshev distance r from c.
  void Shell(const int c[3], int r, std::vector<uint32_t>& out) const {
    int lo3[3], hi3[3];
    for (int k = 0; k < 3; ++k) {
      lo3[k] = std::max(c[k] - r, 0);
      hi3[k] = std::min(c[k] + r, n[k] - 1);
    }
    int q[3];
    for (q[2] = lo3[2]; q[2] <= hi3[2]; ++q[2])
      for (q[1] = lo3[1]; q[1] <= hi3[1]; ++q[1])
        for (q[0] = lo3[0]; q[0] <= hi3[0]; ++q[0]) {
          if (std::max({std::abs(q[0] - c[0]), std::abs(q[1] - c[1]),
                        std::abs(q[2] - c[2])}) != r)
            continue;
          size_t i = Cell(q);
          out.insert(out.end(), items.begin() + start[i],
                     items.begin() + start[i + 1]);
        }
  }
  // Rings outward until some vertices are found, then one more ring (so the
  // true nearest is included) and extra beyond that.
  void Collect(V3 p, int extra, std::vector<uint32_t>& out) const {
    out.clear();
    int c[3];
    Coord(p, c);
    int rmax = std::max({n[0], n[1], n[2]}), found = -1;
    for (int r = 0; r <= rmax; ++r) {
      Shell(c, r, out);
      if (found < 0 && !out.empty()) found = r;
      if (found >= 0 && r >= found + 1 + extra &&
          static_cast<int>(out.size()) >= kMinNeighbours)
        break;
    }
  }
};

}  // namespace

bool Model::Load(const std::wstring& path, std::string& err) {
#ifdef _WIN32
  FILE* f = _wfopen(path.c_str(), L"rb");
#else  // offline tests
  FILE* f = fopen(std::string(path.begin(), path.end()).c_str(), "rb");
#endif
  if (!f) {
    err = "cannot open";
    return false;
  }
  std::vector<uint8_t> d;
  fseek(f, 0, SEEK_END);
  d.resize(ftell(f));
  fseek(f, 0, SEEK_SET);
  size_t got = fread(d.data(), 1, d.size(), f);
  fclose(f);
  Reader r{d.data(), got};
  err = "bad file";
  if (got < 12 || memcmp(d.data(), "A2FD", 4)) return false;
  r.p += 4, r.left -= 4;
  uint32_t version = r.Get<uint32_t>();  // 2 adds region weights + scales
  if (version < 1 || version > 2 || r.Get<uint32_t>() != 2) return false;
  for (int s = 0; s < 2; ++s) {
    if (r.Get<uint8_t>() != s) return false;
    SexModel& m = sex[s];
    if (!r.Mesh(m.head) || !r.Mesh(m.mouth)) return false;
    r.Vec(m.jaw_w, m.mouth.nv);
    m.k = r.Get<uint32_t>();
    m.nm = r.Get<uint32_t>();
    if (!r.ok || !m.k || m.k > 1024 || m.nm > m.head.nv) return false;
    r.Vec(m.midx, m.nm);
    r.Halves(m.basis, static_cast<size_t>(m.k) * m.nm * 3);
    r.Vec(m.proj, static_cast<size_t>(m.k) * a2f::kNumPca);
    uint32_t nt = r.Get<uint32_t>();
    if (!r.ok || nt > m.mouth.nv) return false;
    r.Vec(m.tongue_idx, nt);
    r.Halves(m.tongue_basis, static_cast<size_t>(a2f::kNumTongue) * nt * 3);
    uint32_t nl = r.Get<uint32_t>();
    if (!r.ok || nl > m.head.nv) return false;
    std::vector<uint16_t> lip;
    r.Vec(lip, nl);
    for (float& v : m.lip_big) v = r.Get<float>();
    float rot[9];
    for (float& v : rot) v = r.Get<float>();
    for (float& v : m.t_big) v = r.Get<float>();
    std::vector<uint8_t> rw(static_cast<size_t>(m.head.nv) * 4, 0);
    if (version >= 2) {
      r.Vec(rw, rw.size());
    } else {  // no weights: all skin counts as mouth
      for (uint32_t i = 0; i < m.head.nv; ++i) rw[i * 4 + kMouth] = 255;
    }
    if (!r.ok) return false;
    for (uint16_t i : m.midx)
      if (i >= m.head.nv) return false;
    for (uint16_t i : m.tongue_idx)
      if (i >= m.mouth.nv) return false;
    // per-vertex region weights -> per basis row
    m.region_w.resize(static_cast<size_t>(m.nm) * 4);
    for (uint32_t i = 0; i < m.nm; ++i)
      for (int c = 0; c < 4; ++c)
        m.region_w[i * 4 + c] = rw[m.midx[i] * 4 + c] / 255.f;
    // lip rows -> mean basis row; rotation matrix -> axis-angle
    std::vector<int32_t> row(m.head.nv, -1);
    for (uint32_t i = 0; i < m.nm; ++i) row[m.midx[i]] = static_cast<int>(i);
    m.lip_mean.assign(static_cast<size_t>(m.k) * 3, 0.f);
    uint32_t used = 0;
    for (uint16_t v : lip) {
      if (row[v] < 0) continue;
      ++used;
      for (uint32_t k = 0; k < m.k; ++k)
        for (int c = 0; c < 3; ++c)
          m.lip_mean[k * 3 + c] +=
              m.basis[(static_cast<size_t>(k) * m.nm + row[v]) * 3 + c];
    }
    for (float& v : m.lip_mean) v /= std::max(used, 1u);
    float cs = std::clamp((rot[0] + rot[4] + rot[8] - 1) / 2, -1.f, 1.f);
    m.angle = std::acos(cs);
    V3 ax = Unit({rot[7] - rot[5], rot[2] - rot[6], rot[3] - rot[1]});
    m.axis[0] = ax.x, m.axis[1] = ax.y, m.axis[2] = ax.z;
  }
  r.Vec(rest, a2f::kNumRaw);
  for (float& v : default_scale) v = version >= 2 ? r.Get<float>() : 1.f;
  if (!r.ok) return false;
  err.clear();
  return true;
}

a2f::DirectProj Model::Proj(bool female) const {
  const SexModel& m = sex[female ? 0 : 1];
  return {m.proj.data(), rest.data(), m.k};
}

void Evaluate(const Track& tr, float frame, Part part, const float* scales,
              std::vector<float>& out) {
  const SexModel& m = *tr.model;
  const RefMesh& ref = part == Part::kHead ? m.head : m.mouth;
  out.assign(static_cast<size_t>(ref.nv) * 3, 0.f);
  if (!tr.frames || tr.dim != m.k + a2f::kNumTongue) return;
  if (frame < 0) frame = 0;
  uint32_t i0 = static_cast<uint32_t>(frame);
  float a = frame - static_cast<float>(i0);
  bool past = i0 + 1 >= tr.frames;
  const float* c0 =
      tr.coeffs + static_cast<size_t>(past ? tr.frames - 1 : i0) * tr.dim;
  const float* c1 = past ? c0 : c0 + tr.dim;
  // ease out over 0.1 s, like the morph curves
  float g =
      past ? std::fmax(0.f,
                       1.f - (frame - static_cast<float>(tr.frames - 1)) / 3.f)
           : 1.f;
  if (g <= 0) return;
  std::vector<float> c(tr.dim);
  for (uint32_t k = 0; k < tr.dim; ++k)
    c[k] = ((1 - a) * c0[k] + a * c1[k]) * g;

  if (part == Part::kHead) {
    std::vector<float> s(m.nm, 0.f);
    for (uint32_t i = 0; i < m.nm; ++i)
      for (int r = 0; r < 4; ++r) s[i] += scales[r] * m.region_w[i * 4 + r];
    for (uint32_t k = 0; k < m.k; ++k) {
      if (c[k] == 0) continue;
      const float* b = &m.basis[static_cast<size_t>(k) * m.nm * 3];
      for (uint32_t i = 0; i < m.nm; ++i) {
        float* o = &out[m.midx[i] * 3];
        for (int j = 0; j < 3; ++j) o[j] += c[k] * s[i] * b[i * 3 + j];
      }
    }
    return;
  }
  // tongue
  const uint32_t nt = static_cast<uint32_t>(m.tongue_idx.size());
  for (int k = 0; k < a2f::kNumTongue; ++k) {
    float ck = c[m.k + k] * scales[kTongue];
    const float* b = &m.tongue_basis[static_cast<size_t>(k) * nt * 3];
    for (uint32_t i = 0; i < nt; ++i) {
      float* o = &out[m.tongue_idx[i] * 3];
      for (int j = 0; j < 3; ++j) o[j] += ck * b[i * 3 + j];
    }
  }
  // lower jaw: BigAah's rigid lower-teeth motion scaled by how far the lower
  // lip has dropped (rotation, x/z) and retracted (y) relative to BigAah
  V3 lip{0, 0, 0};
  for (uint32_t k = 0; k < m.k; ++k) lip = lip + At(&m.lip_mean[k * 3]) * c[k];
  // opening / retraction ratios from the unscaled lip, then the jaw scale
  float alpha = std::fmax(lip.z / m.lip_big[2], 0.f);
  float beta = std::fmax(lip.y / m.lip_big[1], alpha) * scales[kJaw];
  alpha *= scales[kJaw];
  if (alpha <= 0 && beta <= 0) return;
  float th = alpha * m.angle, sn = std::sin(th), cs = std::cos(th);
  V3 ax = At(m.axis),
     t{m.t_big[0] * alpha, m.t_big[1] * beta, m.t_big[2] * alpha};
  for (uint32_t v = 0; v < m.mouth.nv; ++v) {
    if (!m.jaw_w[v]) continue;
    V3 p = At(&m.mouth.verts[v * 3]);
    // Rodrigues
    V3 cross{ax.y * p.z - ax.z * p.y, ax.z * p.x - ax.x * p.z,
             ax.x * p.y - ax.y * p.x};
    V3 q = p * cs + cross * sn + ax * (Dot(ax, p) * (1 - cs)) + t;
    V3 d = (q - p) * (m.jaw_w[v] / 255.f);
    out[v * 3] += d.x, out[v * 3 + 1] += d.y, out[v * 3 + 2] += d.z;
  }
}

void LipLabels(const float* verts, const float* bigaah, uint32_t nv,
               std::vector<uint8_t>& labels) {
  labels.assign(nv, 0);
  float drop = 0;
  auto zone = [&](uint32_t i) {
    const float* v = &verts[i * 3];
    return std::fabs(v[0]) < 2.6f && v[1] > 6.5f && v[2] > -3.5f && v[2] < 0.5f;
  };
  for (uint32_t i = 0; i < nv; ++i)
    if (zone(i)) drop = std::max(drop, -bigaah[i * 3 + 2]);
  for (uint32_t i = 0; i < nv; ++i) {
    if (!zone(i)) continue;
    float dz = bigaah[i * 3 + 2];
    labels[i] = dz < -0.25f * drop ? 2 : dz > -0.12f * drop ? 1 : 0;
  }
}

void BuildMapping(const RefMesh& ref, const float* target, uint32_t nv,
                  const uint8_t* target_labels, Mapping& out) {
  out = Mapping();
  out.nv = nv;
  if (nv == ref.nv) {
    double sum = 0;
    for (uint32_t i = 0; i < nv; ++i)
      sum += Len(At(&target[i * 3]) - At(&ref.verts[i * 3]));
    out.mean_dist = static_cast<float>(sum / nv);
    if (out.mean_dist < kIdentityMeanDist) {
      out.identity = true;
      return;
    }
  }
  const uint32_t nt = static_cast<uint32_t>(ref.tris.size() / 3);
  const bool labelled = ref.labels.size() == ref.nv;
  // incident triangles per reference vertex (CSR)
  std::vector<uint32_t> inc_start(ref.nv + 1, 0), inc(nt * 3);
  for (uint32_t t : ref.tris) ++inc_start[t + 1];
  for (uint32_t i = 1; i <= ref.nv; ++i) inc_start[i] += inc_start[i - 1];
  {
    std::vector<uint32_t> fill(inc_start.begin(), inc_start.end() - 1);
    for (uint32_t t = 0; t < nt; ++t)
      for (int k = 0; k < 3; ++k) inc[fill[ref.tris[t * 3 + k]]++] = t;
  }
  // a candidate triangle must not touch a different region
  auto allowed = [&](uint32_t t, uint8_t label) {
    if (!label) return true;
    for (int k = 0; k < 3; ++k) {
      uint8_t l = ref.labels[ref.tris[t * 3 + k]];
      if (l && l != label) return false;
    }
    return true;
  };

  Grid grid(ref.verts, ref.nv);
  out.idx.assign(static_cast<size_t>(nv) * 3, 0);
  out.w.assign(static_cast<size_t>(nv) * 3, 0.f);
  std::vector<uint32_t> near, stamp(nt, 0);
  uint32_t gen = 0;
  double sum = 0;
  for (uint32_t i = 0; i < nv; ++i) {
    V3 p = At(&target[i * 3]);
    grid.Collect(p, 0, near);
    if (near.empty()) continue;  // off the grid: no deltas
    auto nearest = [&](uint8_t label) {
      int nn = -1;
      float nd = INFINITY;
      for (uint32_t v : near) {
        if (label && ref.labels[v] != label) continue;
        float d = Len(p - At(&ref.verts[v * 3]));
        if (d < nd) nd = d, nn = static_cast<int>(v);
      }
      return nn;
    };
    const uint8_t label = labelled && target_labels ? target_labels[i] : 0;
    int best = -1;
    float best_d = INFINITY, best_w[3] = {1, 0, 0};
    ++gen;
    for (uint32_t v : near)
      for (uint32_t j = inc_start[v]; j < inc_start[v + 1]; ++j) {
        uint32_t t = inc[j];
        if (stamp[t] == gen) continue;
        stamp[t] = gen;
        if (!allowed(t, label)) continue;
        const uint32_t* tv = &ref.tris[t * 3];
        V3 a = At(&ref.verts[tv[0] * 3]), b = At(&ref.verts[tv[1] * 3]),
           c = At(&ref.verts[tv[2] * 3]);
        float w[3];
        ClosestOnTri(p, a, b, c, w);
        float d = Len(p - (a * w[0] + b * w[1] + c * w[2]));
        if (d < best_d) {
          best_d = d;
          best = static_cast<int>(t);
          std::copy(w, w + 3, best_w);
        }
      }
    if (best >= 0) {
      for (int k = 0; k < 3; ++k) {
        out.idx[i * 3 + k] = ref.tris[best * 3 + k];
        out.w[i * 3 + k] = best_w[k];
      }
    } else {
      // every nearby triangle touches another region: snap to the nearest
      // vertex of this region, looking further out
      grid.Collect(p, 3, near);
      int v = nearest(label);
      if (v < 0) v = nearest(0);
      ++out.snapped;
      best_d = Len(p - At(&ref.verts[v * 3]));
      for (int k = 0; k < 3; ++k) {
        out.idx[i * 3 + k] = static_cast<uint32_t>(v);
        out.w[i * 3 + k] = k == 0 ? 1.f : 0.f;
      }
    }
    sum += best_d;
    out.max_dist = std::max(out.max_dist, best_d);
  }
  out.mean_dist = static_cast<float>(sum / std::max(nv, 1u));
}

void AddDeltas(const Mapping& m, const std::vector<float>& d, float* xyzw) {
  if (m.identity) {
    if (d.size() < static_cast<size_t>(m.nv) * 3) return;
    for (uint32_t i = 0; i < m.nv; ++i)
      for (int c = 0; c < 3; ++c) xyzw[i * 4 + c] += d[i * 3 + c];
    return;
  }
  for (uint32_t i = 0; i < m.nv; ++i) {
    const uint32_t* ix = &m.idx[i * 3];
    const float* w = &m.w[i * 3];
    for (int c = 0; c < 3; ++c)
      xyzw[i * 4 + c] += w[0] * d[ix[0] * 3 + c] + w[1] * d[ix[1] * 3 + c] +
                         w[2] * d[ix[2] * 3 + c];
  }
}

}  // namespace direct
