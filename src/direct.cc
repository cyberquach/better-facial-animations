// A2F Direct model, the head -> Mark mapping and evaluation declared in
// direct.h.
#include "direct.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <utility>

namespace direct {

const MorphId kMorphIds[kNumMorphs] = {
    {2, 0, "BlinkLeft"},
    {2, 1, "BlinkRight"},
    {2, 6, "BrowUpLeft"},
    {2, 7, "BrowUpRight"},
    {0, 1, "BigAah"},
    {0, 12, "OohQ"},
    // the tongue phonemes
    {0, 4, "DST"},
    {0, 10, "N"},
    {0, 9, "K"},
    {0, 14, "Th"},
    {0, 13, "R"},
    // the expressions
    {1, 0, "DialogueAnger"},
    {1, 1, "DialogueFear"},
    {1, 2, "DialogueHappy"},
    {1, 3, "DialogueSad"},
    {1, 4, "DialogueSurprise"},
    {1, 5, "DialoguePuzzled"},
    {1, 6, "DialogueDisgusted"},
    {1, 7, "MoodNeutral"},
    {1, 8, "MoodAnger"},
    {1, 9, "MoodFear"},
    {1, 10, "MoodHappy"},
    {1, 11, "MoodSad"},
    {1, 12, "MoodSurprise"},
    {1, 13, "MoodPuzzled"},
    {1, 14, "MoodDisgusted"},
    {1, 15, "CombatAnger"},
    // Remaining speech shapes; the others share the entries above.
    {0, 0, "Aah"},
    {0, 2, "BMP"},
    {0, 3, "ChJSh"},
    {0, 5, "Eee"},
    {0, 6, "Eh"},
    {0, 7, "FV"},
    {0, 8, "I"},
    {0, 11, "Oh"},
    {0, 15, "W"},
};

namespace {

// Mark's landmarks (tools/mark_export.py LANDMARKS) and the head morph each
// pairs with; the smile is the expression's motion on one side of the face.
constexpr Morph kLandmarkMorph[kNumLandmarks] = {
    kBlinkLeft, kBlinkRight, kBrowUpLeft, kBrowUpRight,
    kBigAah,    kOohQ,       kHappy,      kHappy};
constexpr int kLandmarkSide[kNumLandmarks] = {0, 0, 0, 0, 0, 0, -1, 1};
constexpr int kLandmarkLips = 5;  // the pucker: centre of the lip box
// skin vertex labels in the lip box (else kNoLip)
enum Lip : uint8_t { kNoLip, kUpperLip, kLowerLip };
constexpr int kKnn = 8;        // Mark vertices per head vertex
// the lips' search: units of distance per whole jaw drop in how far a vertex
// follows the jaw
constexpr float kJawFollow = 3.f;
// vanilla head parts' shape translation
constexpr float kHeadToSkeleton[3] = {0, -1.55f, 120.34f};

struct V3 {
  float x, y, z;
};
inline V3 At(const float* p) { return {p[0], p[1], p[2]}; }
inline V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline V3 operator*(V3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
inline float Dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline float Len(V3 a) { return std::sqrt(Dot(a, a)); }
inline void Put(float* o, V3 p) {
  o[0] = p.x;
  o[1] = p.y;
  o[2] = p.z;
}

float Half(uint16_t h) {
  uint32_t s = static_cast<uint32_t>(h >> 15) << 31, e = (h >> 10) & 0x1F,
           m = h & 0x3FF, bits;
  if (e == 0) {
    if (!m) {
      bits = s;
    } else {  // subnormal
      e = 1;
      while (!(m & 0x400)) {
        m <<= 1;
        --e;
      }
      bits = s | ((e + 112) << 23) | ((m & 0x3FF) << 13);
    }
  } else if (e == 31) {
    bits = s | 0x7F800000 | (m << 13);
  } else {
    bits = s | ((e + 112) << 23) | (m << 13);
  }
  float f;
  memcpy(&f, &bits, sizeof f);
  return f;
}

// Sequential reader over the file bytes; ok turns false on overrun (nothing
// is read or allocated past the bytes there are).
struct Reader {
  const uint8_t* p;
  size_t left;
  bool ok = true;
  // Whether n more items of size bytes each are there; an overrun if not.
  bool Has(size_t n, size_t size) {
    if (ok && n <= left / size) return true;
    ok = false;
    left = 0;
    return false;
  }
  template <typename T>
  T Get() {
    T v{};
    if (!Has(1, sizeof v)) return v;
    memcpy(&v, p, sizeof v);
    p += sizeof v;
    left -= sizeof v;
    return v;
  }
  template <typename T>
  void Vec(std::vector<T>& v, size_t n) {
    if (!Has(n, sizeof(T))) return;
    v.resize(n);
    for (size_t i = 0; i < n; ++i) v[i] = Get<T>();
  }
  void Halves(std::vector<float>& v, size_t n) {
    if (!Has(n, sizeof(uint16_t))) return;
    v.resize(n);
    for (size_t i = 0; i < n; ++i) v[i] = Half(Get<uint16_t>());
  }
};

// Static kd-tree over a point set for exact k nearest neighbours.
struct KdTree {
  std::vector<float> own;    // a subset's points
  std::vector<uint32_t> id;  // a subset's indices in the full set
  const float* pts;          // n x dim
  int dim;
  std::vector<uint32_t> order;  // the points, as a balanced tree in place
  std::vector<uint8_t> axis;    // split axis of the node at order[i]

  KdTree(const float* p, uint32_t n, int d = 3) : pts(p), dim(d) { Init(n); }
  // The points ids of p; Nearest gives indices into p.
  KdTree(const float* p, std::vector<uint32_t> ids, int d = 3)
      : id(std::move(ids)), dim(d) {
    for (uint32_t i : id) own.insert(own.end(), p + i * d, p + i * d + d);
    pts = own.data();
    Init(static_cast<uint32_t>(id.size()));
  }
  KdTree(const KdTree&) = delete;  // pts may point into own
  KdTree(KdTree&&) = default;
  void Init(uint32_t n) {
    order.resize(n);
    axis.assign(n, 0);
    for (uint32_t i = 0; i < n; ++i) order[i] = i;
    Build(0, n);
  }
  void Build(uint32_t b, uint32_t e) {
    if (e - b <= 1) return;
    float lo[4] = {INFINITY, INFINITY, INFINITY, INFINITY},
          hi[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
    for (uint32_t i = b; i < e; ++i)
      for (int k = 0; k < dim; ++k) {
        lo[k] = std::min(lo[k], pts[order[i] * dim + k]);
        hi[k] = std::max(hi[k], pts[order[i] * dim + k]);
      }
    int ax = 0;
    for (int k = 1; k < dim; ++k)
      if (hi[k] - lo[k] > hi[ax] - lo[ax]) ax = k;
    uint32_t m = b + (e - b) / 2;
    std::nth_element(order.begin() + b, order.begin() + m, order.begin() + e,
                     [&](uint32_t x, uint32_t y) {
                       return pts[x * dim + ax] < pts[y * dim + ax];
                     });
    axis[m] = static_cast<uint8_t>(ax);
    Build(b, m);
    Build(m + 1, e);
  }
  // best: a max-heap of (squared distance, point), at most k
  void Search(uint32_t b, uint32_t e, const float* q, int k,
              std::vector<std::pair<float, uint32_t>>& best) const {
    if (b >= e) return;
    uint32_t m = b + (e - b) / 2, i = order[m];
    float d2 = 0;
    for (int c = 0; c < dim; ++c) {
      float t = q[c] - pts[i * dim + c];
      d2 += t * t;
    }
    if (static_cast<int>(best.size()) < k) {
      best.emplace_back(d2, i);
      std::push_heap(best.begin(), best.end());
    } else if (d2 < best.front().first) {
      std::pop_heap(best.begin(), best.end());
      best.back() = {d2, i};
      std::push_heap(best.begin(), best.end());
    }
    if (e - b == 1) return;
    float diff = q[axis[m]] - pts[i * dim + axis[m]];
    bool left = diff < 0;
    Search(left ? b : m + 1, left ? m : e, q, k, best);
    if (static_cast<int>(best.size()) < k || diff * diff < best.front().first)
      Search(left ? m + 1 : b, left ? e : m, q, k, best);
  }
  // The k nearest of q (dim floats): indices and distances, nearest first;
  // how many.
  int Nearest(const float* q, int k, uint32_t* idx, float* dist) const {
    thread_local std::vector<std::pair<float, uint32_t>> best;
    best.clear();
    Search(0, static_cast<uint32_t>(order.size()), q, k, best);
    std::sort_heap(best.begin(), best.end());
    int n = static_cast<int>(best.size());
    for (int i = 0; i < n; ++i) {
      dist[i] = std::sqrt(best[i].first);
      idx[i] = id.empty() ? best[i].second : id[best[i].second];
    }
    return n;
  }
  int Nearest(V3 p, int k, uint32_t* idx, float* dist) const {
    return Nearest(&p.x, k, idx, dist);
  }
};

// Inverse squared distance weights 1 / (dist + eps)^2 of n neighbours,
// normalised.
void IdwWeights(const float* dist, int n, double eps, double* w) {
  double ws = 0;
  for (int j = 0; j < n; ++j)
    ws += w[j] = 1 / ((dist[j] + eps) * (dist[j] + eps));
  for (int j = 0; j < n; ++j) w[j] /= ws;
}

// ---------------- registration ----------------
// Centroid of verts weighted by |delta|^4 (where the morph moves most); side
// -1 / +1 keeps one side of x = mid.
V3 MotionCentroid(const std::vector<float>& verts, const std::vector<float>& d,
                  uint32_t nv, int side = 0, float mid = 0) {
  double s[3] = {}, ws = 0;
  for (uint32_t i = 0; i < nv; ++i) {
    if (side && (verts[i * 3] - mid) * side <= 0) continue;
    double m = Dot(At(&d[i * 3]), At(&d[i * 3]));
    double w = m * m;
    for (int c = 0; c < 3; ++c) s[c] += w * verts[i * 3 + c];
    ws += w;
  }
  if (ws <= 0) return {0, 0, 0};
  return {static_cast<float>(s[0] / ws), static_cast<float>(s[1] / ws),
          static_cast<float>(s[2] / ws)};
}

// Eigenvector of the largest eigenvalue of a symmetric 4x4 (cyclic Jacobi).
void TopEigenvector(double a[4][4], double v[4]) {
  double e[4][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
  for (int sweep = 0; sweep < 50; ++sweep) {
    double off = 0;
    for (int p = 0; p < 4; ++p)
      for (int q = p + 1; q < 4; ++q) off += a[p][q] * a[p][q];
    if (off < 1e-20) break;
    for (int p = 0; p < 4; ++p)
      for (int q = p + 1; q < 4; ++q) {
        if (std::fabs(a[p][q]) < 1e-30) continue;
        double th = (a[q][q] - a[p][p]) / (2 * a[p][q]);
        double t =
            (th >= 0 ? 1 : -1) / (std::fabs(th) + std::sqrt(th * th + 1));
        double c = 1 / std::sqrt(t * t + 1), s = t * c;
        for (int k = 0; k < 4; ++k) {  // A <- A J
          double akp = a[k][p], akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
        }
        for (int k = 0; k < 4; ++k) {  // A <- J^T A
          double apk = a[p][k], aqk = a[q][k];
          a[p][k] = c * apk - s * aqk;
          a[q][k] = s * apk + c * aqk;
        }
        for (int k = 0; k < 4; ++k) {
          double ekp = e[k][p], ekq = e[k][q];
          e[k][p] = c * ekp - s * ekq;
          e[k][q] = s * ekp + c * ekq;
        }
      }
  }
  int best = 0;
  for (int i = 1; i < 4; ++i)
    if (a[i][i] > a[best][best]) best = i;
  for (int k = 0; k < 4; ++k) v[k] = e[k][best];
}

// Least-squares similarity b ~ s R a + t (Horn's unit quaternion method);
// rigid: s = 1.
void Similarity(const V3* a, const V3* b, int n, float& s, float r[9], V3& t,
                bool rigid = false) {
  double ma[3] = {}, mb[3] = {};
  for (int i = 0; i < n; ++i) {
    const float *pa = &a[i].x, *pb = &b[i].x;
    for (int c = 0; c < 3; ++c) {
      ma[c] += pa[c] / n;
      mb[c] += pb[c] / n;
    }
  }
  double m[3][3] = {}, var = 0;
  for (int i = 0; i < n; ++i) {
    const float *pa = &a[i].x, *pb = &b[i].x;
    for (int j = 0; j < 3; ++j) {
      var += (pa[j] - ma[j]) * (pa[j] - ma[j]);
      for (int k = 0; k < 3; ++k) m[j][k] += (pa[j] - ma[j]) * (pb[k] - mb[k]);
    }
  }
  double xx = m[0][0], xy = m[0][1], xz = m[0][2], yx = m[1][0], yy = m[1][1],
         yz = m[1][2], zx = m[2][0], zy = m[2][1], zz = m[2][2];
  double nq[4][4] = {{xx + yy + zz, yz - zy, zx - xz, xy - yx},
                     {yz - zy, xx - yy - zz, xy + yx, zx + xz},
                     {zx - xz, xy + yx, -xx + yy - zz, yz + zy},
                     {xy - yx, zx + xz, yz + zy, -xx - yy + zz}};
  double q[4];
  TopEigenvector(nq, q);
  double w = q[0], x = q[1], y = q[2], z = q[3];
  double rd[9] = {1 - 2 * (y * y + z * z), 2 * (x * y - w * z),
                  2 * (x * z + w * y),     2 * (x * y + w * z),
                  1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                  2 * (x * z - w * y),     2 * (y * z + w * x),
                  1 - 2 * (x * x + y * y)};
  double num = 0;  // sum (b - mb) . R (a - ma)
  for (int j = 0; j < 3; ++j)
    for (int k = 0; k < 3; ++k) num += rd[j * 3 + k] * m[k][j];
  s = static_cast<float>(var > 0 && !rigid ? num / var : 1);
  double tt[3];
  for (int j = 0; j < 3; ++j) {
    tt[j] = mb[j];
    for (int k = 0; k < 3; ++k) tt[j] -= s * rd[j * 3 + k] * ma[k];
  }
  for (int i = 0; i < 9; ++i) r[i] = static_cast<float>(rd[i]);
  t = {static_cast<float>(tt[0]), static_cast<float>(tt[1]),
       static_cast<float>(tt[2])};
}

inline V3 Rotate(const float r[9], V3 p) {
  return {r[0] * p.x + r[1] * p.y + r[2] * p.z,
          r[3] * p.x + r[4] * p.y + r[5] * p.z,
          r[6] * p.x + r[7] * p.y + r[8] * p.z};
}

float MaxAbs(const std::vector<float>& v) {
  float m = 0;
  for (float x : v) m = std::max(m, std::fabs(x));
  return m;
}

bool IsFinite(const std::vector<float>& values) {
  return std::all_of(values.begin(), values.end(),
                     [](float value) { return std::isfinite(value); });
}

std::vector<float> Shifted(const std::vector<float>& v, const float off[3]) {
  std::vector<float> o(v);
  for (size_t i = 0; i < o.size(); ++i) o[i] += off[i % 3];
  return o;
}

// Region weights of a skin vertex (brows, eyes, cheeks & nose, mouth; sum to
// 1, no seams): b1 midway between the eye line and the brow line, b2 1.2
// under the eye line (lower lid -> cheek), b3 under the nose on the midline
// (2.35 above the lips) dipping by 1.8 at |x| >= 3 so the cheeks and the
// nasolabial area stay with the cheeks; each a smoothstep 1-1.5 units wide.
void RegionWeights(V3 p, float mid, float z_eye, float z_brow, float z_lips,
                   float w[kNumSkinRegions]) {
  using a2f::Smoothstep;
  float b1 = 0.5f * (z_eye + z_brow), b2 = z_eye - 1.2f,
        b3 = z_lips + 2.35f -
             1.8f * std::clamp((std::fabs(p.x - mid) - 1.5f) / 1.5f, 0.f, 1.f);
  w[kBrows] = Smoothstep((p.z - b1) / 1.0f + 0.5f);
  w[kEyes] = (1 - w[kBrows]) * Smoothstep((p.z - b2) / 1.0f + 0.5f);
  w[kCheeks] =
      (1 - w[kBrows] - w[kEyes]) * Smoothstep((p.z - b3) / 1.5f + 0.5f);
  w[kMouth] = 1 - w[kBrows] - w[kEyes] - w[kCheeks];
}

// Lip controls taper into the surrounding skin and share mouth corners.
void LipWeights(V3 p, float mid, V3 lips, float jaw_follow, float w[2]) {
  const float edge = std::min({(2.6f - std::fabs(p.x - mid)) / .6f,
                               (p.y - lips.y + 1.5f) / .5f,
                               (2.f - std::fabs(p.z - lips.z)) / .5f});
  const float weight = a2f::Smoothstep(edge);
  // Use the same jaw-follow distinction as mapping: upper below .12,
  // lower above .25, smoothly shared at the corners.
  const float lower = a2f::Smoothstep((jaw_follow - .12f) / .13f);
  w[0] = weight * (1 - lower);
  w[1] = weight * lower;
}

// A skin vertex's scale: scales (kNumSkinRegions) by its region weights.
float RegionScale(const Head& h, uint32_t v, const float* scales) {
  float sv = 0;
  for (int r = 0; r < kNumSkinRegions; ++r)
    sv += scales[r] * h.region_w[v * kNumSkinRegions + r];
  return sv;
}

// Mean z of the vertices a morph moves by more than t, or fallback.
float MovedZ(const std::vector<float>& v, const std::vector<float>& a,
             const std::vector<float>& b, uint32_t nv, float t,
             float fallback) {
  double s = 0;
  uint32_t n = 0;
  for (uint32_t i = 0; i < nv; ++i)
    if (Len(At(&a[i * 3])) > t || Len(At(&b[i * 3])) > t) {
      s += v[i * 3 + 2];
      ++n;
    }
  return n ? static_cast<float>(s / n) : fallback;
}

// Brows, beards, scars: the kFollow nearest skin vertices, never across the
// lips (a moustache doesn't drop with the jaw, a goatee does). Eyes (lashes,
// lid line): weighted by how much the shape's own blink moves the vertex
// (eyeballs ~ 0), among skin vertices the blinks move on the same side and
// the same lid (upper lids close down, lower lids up).
void Follow(const ShapeIn& s, const std::vector<float>& skin, uint32_t snv,
            const std::vector<uint8_t>& lips, float lips_z,
            const std::vector<float>& skin_blink, float mid, Head::Shape& out) {
  float off[3];
  FrameOffset(s.verts.data(), s.nv, off);
  std::vector<float> v = Shifted(s.verts, off);
  out.idx.assign(static_cast<size_t>(s.nv) * kFollow, 0);
  out.w.assign(out.idx.size(), 0.f);
  std::vector<float> follow(s.nv, 1.f), bz(s.nv, 0.f);
  bool eyes = s.part == Part::kEyes;
  if (eyes) {
    std::vector<float> mag(s.nv, 0.f);
    bool blinks = s.morphs.size() > kBlinkRight &&
                  s.morphs[kBlinkLeft].size() == s.nv * 3 &&
                  s.morphs[kBlinkRight].size() == s.nv * 3;
    for (uint32_t i = 0; blinks && i < s.nv; ++i) {
      V3 l = At(&s.morphs[kBlinkLeft][i * 3]),
         r = At(&s.morphs[kBlinkRight][i * 3]);
      V3 m{std::max(std::fabs(l.x), std::fabs(r.x)),
           std::max(std::fabs(l.y), std::fabs(r.y)),
           std::max(std::fabs(l.z), std::fabs(r.z))};
      mag[i] = Len(m);
      bz[i] = l.z + r.z;
    }
    std::vector<float> moving;
    for (float m : mag)
      if (m > 0) moving.push_back(m);
    float p90 = 0;
    if (!moving.empty()) {
      size_t k = moving.size() * 9 / 10;
      std::nth_element(moving.begin(), moving.begin() + k, moving.end());
      p90 = moving[k];
    }
    for (uint32_t i = 0; i < s.nv; ++i)
      follow[i] = p90 > 0 ? std::clamp(mag[i] / p90, 0.f, 1.f) : 0.f;
  }
  // candidate skin vertices: all but the lower lip (above the lips) / all
  // but the upper lip (below), or for the eyes those the blinks move, by
  // side and lid
  std::vector<uint32_t> ids[4];  // eyes: side (x < mid, x > mid) x lid
  for (uint32_t i = 0; i < snv; ++i) {
    if (!eyes) {
      if (lips[i] != kLowerLip) ids[0].push_back(i);
      if (lips[i] != kUpperLip) ids[1].push_back(i);
      continue;
    }
    V3 b = At(&skin_blink[i * 3]);
    if (Len(b) <= 0.1f) continue;
    ids[(skin[i * 3] > mid ? 1 : 0) * 2 + (b.z > 0 ? 1 : 0)].push_back(i);
  }
  for (int side = 0; eyes && side < 2; ++side)  // a lid's fallback: the eye
    for (int lid = 0; lid < 2; ++lid)
      if (ids[side * 2 + lid].empty())
        ids[side * 2 + lid] = ids[side * 2 + 1 - lid];
  std::vector<KdTree> trees;
  for (int c = 0; c < 4; ++c) trees.emplace_back(skin.data(), ids[c]);
  for (uint32_t i = 0; i < s.nv; ++i) {
    if (follow[i] <= 0.02f) continue;
    V3 p = At(&v[i * 3]);
    int c = eyes           ? (p.x > mid ? 1 : 0) * 2 + (bz[i] > 0 ? 1 : 0)
            : p.z < lips_z ? 1
                           : 0;
    if (ids[c].empty()) continue;
    uint32_t idx[kFollow];
    float dist[kFollow];
    double w[kFollow];
    int n = trees[c].Nearest(p, kFollow, idx, dist);
    IdwWeights(dist, n, 0.05, w);
    for (int j = 0; j < n; ++j) {
      out.idx[i * kFollow + j] = idx[j];
      out.w[i * kFollow + j] = static_cast<float>(w[j] * follow[i]);
    }
  }
}

// Pinned parts (piercings), after Follow: every vertex of a piece (vertices
// within kPinLink of one another) takes the same skin vertices, so the piece
// moves as one; Follow alone pulls a ring through the lower lip apart with
// the upper. The piece is pinned where its vertices come nearest the skin,
// on the lip more of it touches. A piece larger than kPinSize, or a shape of
// more than kPinMaxVerts, keeps Follow's weights.
constexpr float kPinLink = 0.25f, kPinSize = 4.f;
constexpr uint32_t kPinMaxVerts = 5000;
void Pin(const ShapeIn& s, const std::vector<float>& skin, uint32_t snv,
         const std::vector<uint8_t>& lips, float lips_z, Head::Shape& out) {
  if (!snv || s.nv > kPinMaxVerts) return;
  float off[3];
  FrameOffset(s.verts.data(), s.nv, off);
  const std::vector<float> v = Shifted(s.verts, off);
  std::vector<uint32_t> root(s.nv);
  for (uint32_t i = 0; i < s.nv; ++i) root[i] = i;
  auto find = [&root](uint32_t i) {
    while (root[i] != i) i = root[i] = root[root[i]];
    return i;
  };
  for (uint32_t i = 1; i < s.nv; ++i)
    for (uint32_t j = 0; j < i; ++j)
      if (Len(At(&v[i * 3]) - At(&v[j * 3])) < kPinLink)
        root[find(i)] = find(j);
  std::vector<std::vector<uint32_t>> pieces(s.nv);
  for (uint32_t i = 0; i < s.nv; ++i) pieces[find(i)].push_back(i);
  // the skin a piece may be pinned to: all but the lower lip / the upper
  std::vector<uint32_t> ids[2];
  for (uint32_t i = 0; i < snv; ++i) {
    if (lips[i] != kLowerLip) ids[0].push_back(i);
    if (lips[i] != kUpperLip) ids[1].push_back(i);
  }
  const KdTree all(skin.data(), snv);
  const KdTree sides[2] = {KdTree(skin.data(), ids[0]),
                           KdTree(skin.data(), ids[1])};
  for (const std::vector<uint32_t>& piece : pieces) {
    if (piece.empty()) continue;
    float lo[3] = {INFINITY, INFINITY, INFINITY},
          hi[3] = {-INFINITY, -INFINITY, -INFINITY};
    double at[3] = {}, sum = 0, lip[3] = {};
    for (uint32_t i : piece) {
      uint32_t near;
      float dist;
      all.Nearest(&v[i * 3], 1, &near, &dist);
      const double w = 1 / ((dist + 0.05) * (dist + 0.05));
      for (int c = 0; c < 3; ++c) {
        lo[c] = std::min(lo[c], v[i * 3 + c]);
        hi[c] = std::max(hi[c], v[i * 3 + c]);
        at[c] += w * skin[near * 3 + c];
      }
      sum += w;
      lip[lips[near]] += w;
    }
    if (std::max({hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]}) > kPinSize)
      continue;
    const float p[3] = {static_cast<float>(at[0] / sum),
                        static_cast<float>(at[1] / sum),
                        static_cast<float>(at[2] / sum)};
    const int side = lip[kLowerLip] != lip[kUpperLip]
                         ? lip[kLowerLip] > lip[kUpperLip]
                         : p[2] < lips_z;
    if (ids[side].empty()) continue;
    uint32_t idx[kFollow] = {};
    float dist[kFollow];
    double w[kFollow] = {};
    const int n = sides[side].Nearest(p, kFollow, idx, dist);
    IdwWeights(dist, n, 0.05, w);
    for (uint32_t i : piece)
      for (int j = 0; j < kFollow; ++j) {
        out.idx[i * kFollow + j] = idx[j];
        out.w[i * kFollow + j] = static_cast<float>(w[j]);
      }
  }
}

// Least-squares affine motion d ~ a [x - c, 1] of the vertices ids about
// their centroid c. False: too few or degenerate.
bool AffineFit(const float* pos, const float* d,
               const std::vector<uint32_t>& ids, double a[3][4], double c[3]) {
  if (ids.size() < 4) return false;
  double m[4][7] = {};  // normal equations, three right-hand sides
  c[0] = c[1] = c[2] = 0;
  for (uint32_t i : ids)
    for (int j = 0; j < 3; ++j) c[j] += pos[i * 3 + j] / ids.size();
  for (uint32_t i : ids) {
    const double x[4] = {pos[i * 3] - c[0], pos[i * 3 + 1] - c[1],
                         pos[i * 3 + 2] - c[2], 1};
    for (int j = 0; j < 4; ++j) {
      for (int k = 0; k < 4; ++k) m[j][k] += x[j] * x[k];
      for (int k = 0; k < 3; ++k) m[j][4 + k] += x[j] * d[i * 3 + k];
    }
  }
  double trace = m[0][0] + m[1][1] + m[2][2];
  for (int j = 0; j < 3; ++j) m[j][j] += 1e-6 * trace;  // flat vertex sets
  for (int j = 0; j < 4; ++j) {  // Gauss-Jordan, partial pivoting
    int piv = j;
    for (int k = j + 1; k < 4; ++k)
      if (std::fabs(m[k][j]) > std::fabs(m[piv][j])) piv = k;
    if (std::fabs(m[piv][j]) < 1e-12) return false;
    for (int k = 0; k < 7; ++k) std::swap(m[j][k], m[piv][k]);
    for (int k = 0; k < 4; ++k) {
      if (k == j) continue;
      const double f = m[k][j] / m[j][j];
      for (int l = j; l < 7; ++l) m[k][l] -= f * m[j][l];
    }
  }
  for (int j = 0; j < 4; ++j)
    for (int k = 0; k < 3; ++k) a[k][j] = m[j][4 + k] / m[j][j];
  return true;
}

// The value a fraction q of v lies under.
float Quantile(std::vector<float> v, double q) {
  if (v.empty()) return 0;
  size_t k = std::min(v.size() - 1, static_cast<size_t>(q * v.size()));
  std::nth_element(v.begin(), v.begin() + k, v.end());
  return v[k];
}

// Preserve the pre-feature mapping for the live opt-out switch. The jaw
// fit is shared, but the old tongue classifier and model basis stay intact.
void ModelTongue(const Mark& mk, const ShapeIn& sh,
                const std::vector<uint32_t>& teeth,
                const std::vector<float>& mag, float most, float s,
                const float r[9], Head::Shape& out) {
  constexpr int kTongueKnn = 4;
  const uint32_t nv = sh.nv;
  const float* big = sh.morphs[kBigAah].data();
  const float* p = out.pos.data();
  out.model_jaw = out.jaw;
  // the tongue: the most any tongue phoneme moves a vertex off the jaw's own
  // motion in it (BigAah's, scaled to the phoneme on the lower teeth); the
  // teeth's own level of that is the noise, and they are never tongue
  std::vector<float> score(nv, 0.f);
  for (int ph = 0; ph < kNumTonguePhonemes; ++ph) {
    const size_t m = kTonguePhoneme + ph;
    if (m >= sh.morphs.size() || sh.morphs[m].size() != nv * 3) continue;
    const float* d = sh.morphs[m].data();
    double num = 0, den = 0;
    for (uint32_t i : teeth) {
      num += Dot(At(&d[i * 3]), At(&big[i * 3]));
      den += Dot(At(&big[i * 3]), At(&big[i * 3]));
    }
    const float a = den > 0 ? static_cast<float>(num / den) : 0;
    for (uint32_t i = 0; i < nv; ++i)
      if (mag[i] >= 0.02f * most)
        score[i] = std::max(score[i], Len(At(&d[i * 3]) - At(&big[i * 3]) * a));
  }
  std::vector<float> noise;
  for (uint32_t i : teeth) noise.push_back(score[i]);
  for (uint32_t i : teeth) score[i] = 0;
  const float lo = 1.5f * Quantile(noise, 0.5),
              top = *std::max_element(score.begin(), score.end());
  if (top < 2.5f * lo || top < 0.05f * most || !mk.tongue_nv) return;
  std::vector<float> tongue(nv);
  std::vector<uint32_t> ids;
  for (uint32_t i = 0; i < nv; ++i) {
    tongue[i] = std::clamp((score[i] - lo) / (0.5f * top - lo), 0.f, 1.f);
    if (tongue[i] > 0.5f) ids.push_back(i);
  }
  if (ids.size() < 4) return;
  // its tip: the front end, at the middle
  V3 tip{0, -INFINITY, 0};
  for (uint32_t i : ids) {
    tip.x += p[i * 3] / ids.size();
    tip.y = std::max(tip.y, p[i * 3 + 1]);
  }
  uint32_t front = 0;
  for (uint32_t i : ids)
    if (p[i * 3 + 1] > tip.y - 0.3f) {
      tip.z += p[i * 3 + 2];
      ++front;
    }
  tip.z /= front;
  const KdTree tree(mk.tongue_verts.data(), mk.tongue_nv);
  const float rt[9] = {r[0], r[3], r[6], r[1], r[4], r[7], r[2], r[5], r[8]};
  out.model_tongue.assign(static_cast<size_t>(nv) * a2f::kNumTongue * 3, 0.f);
  for (uint32_t i = 0; i < nv; ++i) {
    if (tongue[i] <= 0) continue;
    out.model_jaw[i] = std::max(out.model_jaw[i], tongue[i]);
    V3 q = At(mk.tongue_tip) + Rotate(rt, At(&p[i * 3]) - tip) * (1 / s);
    uint32_t idx[kTongueKnn];
    float dist[kTongueKnn];
    double w[kTongueKnn];
    int n = tree.Nearest(q, kTongueKnn, idx, dist);
    IdwWeights(dist, n, 1e-3, w);
    float* o = &out.model_tongue[static_cast<size_t>(i) * a2f::kNumTongue * 3];
    for (int c = 0; c < a2f::kNumTongue; ++c) {
      V3 d{0, 0, 0};
      for (int j = 0; j < n; ++j)
        d = d +
            At(&mk.tongue_basis
                    [(static_cast<size_t>(idx[j]) * a2f::kNumTongue + c) * 3]) *
                static_cast<float>(w[j]);
      Put(&o[c * 3], Rotate(r, d) * (s * tongue[i]));
    }
  }
}

// Map the jaw from BigAah and identify the tongue by its independent
// phoneme motion. Keep authored tongue shapes in the head's own space.
void Mouth(const Mark& mk, const ShapeIn& sh, std::vector<float> pos,
           float scale, const float rotation[9], Head& head, Head::Shape& out) {
  const uint32_t nv = sh.nv;
  out.pos = std::move(pos);
  out.jaw.assign(nv, 0.f);
  out.model_jaw.assign(nv, 0.f);
  if (sh.morphs.size() <= kBigAah || sh.morphs[kBigAah].size() != nv * 3)
    return;
  const float* big = sh.morphs[kBigAah].data();
  const float* p = out.pos.data();
  std::vector<float> mag(nv);
  float most = 0;
  for (uint32_t i = 0; i < nv; ++i)
    most = std::max(most, mag[i] = Len(At(&big[i * 3])));
  const bool has_jaw_motion = most >= 1e-4f;
  // the lower teeth: the most moved, less those off their common motion
  std::vector<uint32_t> teeth;
  for (uint32_t i = 0; i < nv; ++i)
    if (has_jaw_motion && mag[i] >= 0.5f * most) teeth.push_back(i);
  std::vector<float> rigid(static_cast<size_t>(nv) * 3);
  bool fitted = false;
  for (int pass = 0; pass < 4; ++pass) {
    double a[3][4], c[3];
    if (!AffineFit(p, big, teeth, a, c)) break;
    fitted = true;
    for (uint32_t i = 0; i < nv; ++i)
      for (int k = 0; k < 3; ++k)
        rigid[i * 3 + k] = static_cast<float>(
            a[k][0] * (p[i * 3] - c[0]) + a[k][1] * (p[i * 3 + 1] - c[1]) +
            a[k][2] * (p[i * 3 + 2] - c[2]) + a[k][3]);
    auto off = [&](uint32_t i) {
      return Len(At(&big[i * 3]) - At(&rigid[i * 3]));
    };
    std::vector<float> res;
    for (uint32_t i : teeth) res.push_back(off(i));
    const float limit = std::max(3 * Quantile(res, 0.5), 0.02f * most);
    std::vector<uint32_t> kept;
    for (uint32_t i = 0; i < nv; ++i)
      if (mag[i] >= 0.5f * most && off(i) <= limit) kept.push_back(i);
    if (kept.size() < 4 || kept == teeth) break;
    teeth = std::move(kept);
  }
  for (uint32_t i = 0; i < nv; ++i) {
    if (mag[i] < 1e-4f) continue;
    V3 d = At(&big[i * 3]), m = At(&rigid[i * 3]);
    float w = fitted ? Dot(d, m) / std::max(Dot(m, m), 0.01f * most * most)
                     : mag[i] / most;
    out.jaw[i] = std::clamp(w, 0.f, 1.f);
  }
  if (has_jaw_motion)
    ModelTongue(mk, sh, teeth, mag, most, scale, rotation, out);
  // Refine the jaw sample per phoneme: BigAah can also move the tongue.
  // With no jaw sample, classify independent phoneme motion directly; a
  // separate tongue can be still in BigAah, while static teeth have no signal.
  std::vector<float> score(nv, 0.f);
  float noise = 0;
  for (int ph = 0; ph < kNumTonguePhonemes; ++ph) {
    const size_t m = kTonguePhoneme + ph;
    if (m >= sh.morphs.size() || sh.morphs[m].size() != nv * 3) continue;
    const float* d = sh.morphs[m].data();
    std::vector<uint32_t> sample = teeth;
    std::vector<float> residual(nv);
    for (int pass = 0; pass < 8; ++pass) {
      double num = 0, den = 0;
      for (uint32_t i : sample) {
        num += Dot(At(&d[i * 3]), At(&big[i * 3]));
        den += Dot(At(&big[i * 3]), At(&big[i * 3]));
      }
      const float a = den > 0 ? static_cast<float>(num / den) : 0;
      for (uint32_t i = 0; i < nv; ++i)
        residual[i] = Len(At(&d[i * 3]) - At(&big[i * 3]) * a);
      std::vector<float> errors;
      for (uint32_t i : sample) errors.push_back(residual[i]);
      const float median = Quantile(errors, 0.5);
      const float limit = std::max(3 * median, 0.02f * most);
      std::vector<uint32_t> kept;
      for (uint32_t i : sample)
        if (residual[i] <= limit) kept.push_back(i);
      if (kept.size() < 4 || kept == sample || pass == 7) {
        noise = std::max(noise, median);
        break;
      }
      sample = std::move(kept);
    }
    for (uint32_t i = 0; i < nv; ++i)
      score[i] = std::max(score[i], residual[i]);
  }
  const float lo = std::max({1e-4f, 1.5f * noise, 0.02f * most}),
              top = *std::max_element(score.begin(), score.end());
  for (uint32_t i = 0; i < nv; ++i) head.jaw_verts += out.jaw[i] > 0;
  if (top < 2.5f * lo || top < 0.05f * most) return;
  std::vector<float> tongue(nv);
  uint32_t tongue_count = 0;
  for (uint32_t i = 0; i < nv; ++i) {
    tongue[i] = std::clamp((score[i] - lo) / (0.5f * top - lo), 0.f, 1.f);
    if (tongue[i] > 0.5f) ++tongue_count;
  }
  if (tongue_count < 4) return;
  out.tongue_weight = tongue;
  out.tongue.assign(static_cast<size_t>(nv) * kNumPhonemes * 3, 0.f);
  for (uint32_t i = 0; i < nv; ++i) {
    if (tongue[i] <= 0) continue;
    for (int ph = 0; ph < kNumPhonemes; ++ph) {
      const size_t m = kSpeechMorphs[ph];
      if (m >= sh.morphs.size() || sh.morphs[m].size() != nv * 3) continue;
      for (int j = 0; j < 3; ++j)
        out.tongue[(static_cast<size_t>(i) * kNumPhonemes + ph) * 3 + j] =
            sh.morphs[m][i * 3 + j] * tongue[i];
    }
    ++head.tongue_verts;
  }
}

// One face of the file (version 8 or 9), as the tool writes it.
bool LoadFace(Reader& r, uint32_t version, const Model& model, Mark& m) {
  m.nv = r.Get<uint32_t>();
  if (!r.ok || !m.nv || m.nv > 1000000) return false;
  r.Vec(m.verts, static_cast<size_t>(m.nv) * 3);
  r.Halves(m.jaw, m.nv);
  for (float& v : m.landmarks) v = r.Get<float>();
  m.k = r.Get<uint32_t>();
  uint32_t nm = r.Get<uint32_t>();
  if (!r.ok || !m.k || m.k > 256 || nm > m.nv) return false;
  std::vector<uint32_t> midx;
  r.Vec(midx, nm);
  r.Halves(m.basis, static_cast<size_t>(nm) * m.k * 3);
  r.Vec(m.proj, static_cast<size_t>(m.k) * model.n_skin);
  r.Vec(m.rest, model.n_out);
  for (float& v : m.jaw_points) v = r.Get<float>();
  m.jaw_strength = r.Get<float>();
  m.tongue_nv = r.Get<uint32_t>();
  if (!r.ok || m.tongue_nv > 1000000) return false;
  r.Vec(m.tongue_verts, static_cast<size_t>(m.tongue_nv) * 3);
  for (float& v : m.tongue_tip) v = r.Get<float>();
  r.Halves(m.tongue_basis,
           static_cast<size_t>(m.tongue_nv) * a2f::kNumTongue * 3);
  if (version == 9)
    r.Vec(m.tongue_proj, static_cast<size_t>(a2f::kNumTongue) * a2f::kLatent3);
  if (!r.ok) return false;
  m.row.assign(m.nv, -1);
  for (uint32_t i = 0; i < nm; ++i) {
    if (midx[i] >= m.nv) return false;
    m.row[midx[i]] = static_cast<int32_t>(i);
  }
  return true;
}

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
  const long size = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (size > 0) d.resize(static_cast<size_t>(size));
  size_t got = fread(d.data(), 1, d.size(), f);
  fclose(f);
  err = "bad file";
  constexpr char kMagic[] = "A2FD";
  constexpr size_t kMagicSize = sizeof kMagic - 1;
  if (got < kMagicSize || memcmp(d.data(), kMagic, kMagicSize)) return false;
  Reader r{d.data() + kMagicSize, got - kMagicSize};
  // version 8: Mark (tools/mark_export.py); version 9: the v3.0 network's
  // faces (tools/mark_export3.py)
  const uint32_t version = r.Get<uint32_t>();
  if (version != 8 && version != 9) {
    err = "not a version-8 or version-9 A2FDirect bin";
    return false;
  }
  network = version == 9 ? 3 : 2;
  uint32_t count = 1;
  if (version == 9) {
    count = r.Get<uint32_t>();
    n_skin = r.Get<uint32_t>();
    const uint32_t n_tongue = r.Get<uint32_t>();
    n_out = r.Get<uint32_t>();
    if (!r.ok || count != 2 || n_skin != a2f::kLatent3 ||
        n_tongue != a2f::kLatent3 || n_out != a2f::kNumOut3)
      return false;
  }
  faces.assign(count, Mark());
  for (Mark& m : faces)
    if (!LoadFace(r, version, *this, m)) return false;
  if (r.left) return false;
  err.clear();
  return true;
}

void FrameOffset(const float* verts, uint32_t nv, float offset[3]) {
  float lo = INFINITY, hi = -INFINITY;
  for (uint32_t i = 0; i < nv; ++i) {
    lo = std::min(lo, verts[i * 3 + 2]);
    hi = std::max(hi, verts[i * 3 + 2]);
  }
  bool skel = nv && (lo + hi) / 2 > kHeadToSkeleton[2] / 2;
  for (int c = 0; c < 3; ++c) offset[c] = skel ? -kHeadToSkeleton[c] : 0;
}

namespace {

// Where a head stands against Mark: its midline and landmarks, and the
// similarity that puts Mark on them.
struct Registration {
  float mid, s, rot[9];
  float unit;  // Head-space distance per nominal fitting unit.
  V3 t, lm[kNumLandmarks];
  V3 ToHead(V3 mark) const { return Rotate(rot, mark) * s + t; }
  V3 InFittingUnits(V3 point) const { return point * (1.f / unit); }
};

Registration Register(const Mark& mk, const ShapeIn& skin, Head& out) {
  const uint32_t nv = skin.nv;
  auto centroid = [&](Morph m, int side, float mid) {
    return MotionCentroid(skin.verts, skin.morphs[m], nv, side, mid);
  };
  Registration g;
  g.mid = 0.5f * (centroid(kBlinkLeft, 0, 0).x + centroid(kBlinkRight, 0, 0).x);
  V3 ml[kNumLandmarks];
  for (int i = 0; i < kNumLandmarks; ++i) {
    g.lm[i] = centroid(kLandmarkMorph[i], kLandmarkSide[i], g.mid);
    ml[i] = At(&mk.landmarks[i * 3]);
  }
  // Normalize fitting distances to a nominal six-unit eye separation.
  g.unit = Len(g.lm[1] - g.lm[0]) / 6.f;
  Similarity(ml, g.lm, kNumLandmarks, g.s, g.rot, g.t);
  for (int i = 0; i < kNumLandmarks; ++i)
    out.residual[i] = Len(g.lm[i] - g.ToHead(ml[i]));
  out.scale = g.s;
  return g;
}

// Head::auto_scale: the skin's own morph peaks over Mark's scale s, against
// the standard heads'.
void AutoScale(const ShapeIn& skin, float s, Head& out) {
  auto ratio = [&](Morph m, float ref) {
    const std::vector<float>& d = skin.morphs[m];
    float top = 0;
    for (uint32_t v = 0; v < skin.nv; ++v)
      top = std::max(top, Len(At(&d[v * 3])));
    return s > 0 ? std::clamp(top / s / ref, kAutoMin, 1.f) : 1.f;
  };
  out.auto_scale[kBrows] = std::max(ratio(kBrowUpLeft, kAutoBrowUp),
                                    ratio(kBrowUpRight, kAutoBrowUp));
  out.auto_scale[kCheeks] = ratio(kHappy, kAutoHappy);
  out.auto_scale[kMouth] = out.auto_scale[kJaw] = ratio(kBigAah, kAutoBigAah);
}

// The lips: a box around the pucker's centroid.
bool InLipBox(const Registration& g, V3 p) {
  const V3 lips = g.lm[kLandmarkLips];
  return std::fabs(p.x - g.mid) < 2.6f * g.unit &&
         p.y > lips.y - 1.5f * g.unit &&
         std::fabs(p.z - lips.z) < 2.0f * g.unit;
}

// Ambiguous parts must fit behind this face's lips, with extents scaled by
// its smile and eye separation. Work in the skin's space, as Mouth does.
// Existing explicitly identified mouths keep their established mapping.
bool OralGeometry(const ShapeIn& shape, const Registration& g,
                  const float skin_offset[3]) {
  if (!shape.nv || shape.morphs.size() <= kBigAah ||
      shape.morphs[kBigAah].size() != shape.verts.size())
    return false;
  const V3 lips = g.lm[kLandmarkLips];
  const float width = std::max(0.5f * Len(g.lm[7] - g.lm[6]),
                               0.3f * Len(g.lm[1] - g.lm[0]));
  if (!std::isfinite(width) || width < 1e-4f) return false;
  float offset[3];
  FrameOffset(shape.verts.data(), shape.nv, offset);
  uint32_t inside = 0;
  for (uint32_t i = 0; i < shape.nv; ++i) {
    const V3 p = At(&shape.verts[i * 3]) + At(offset) - At(skin_offset);
    const V3 delta = At(&shape.morphs[kBigAah][i * 3]);
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) ||
        !std::isfinite(delta.x) || !std::isfinite(delta.y) ||
        !std::isfinite(delta.z))
      return false;
    inside += std::fabs(p.x - g.mid) <= 1.5f * width &&
              p.y >= lips.y - 3.f * width && p.y <= lips.y + 0.6f * width &&
              std::fabs(p.z - lips.z) <= 1.5f * width;
  }
  // A few gum seam vertices may extend beyond the cavity. A static BigAah
  // is valid for separate upper teeth; a missing morph vector is not.
  return inside >= 0.9 * shape.nv;
}

// How far each of n vertices (pos; dz: its z under the open jaw, every
// stride-th float) follows the jaw: its drop over the lip box's most, 0..1;
// -1 outside the box.
std::vector<float> JawFollow(const Registration& g, const float* dz, int stride,
                             uint32_t n, const float* pos) {
  float drop = 0;
  for (uint32_t i = 0; i < n; ++i)
    if (InLipBox(g, At(&pos[i * 3]))) drop = std::max(drop, -dz[i * stride]);
  std::vector<float> f(n, -1.f);
  for (uint32_t i = 0; drop > 0 && i < n; ++i)
    if (InLipBox(g, At(&pos[i * 3])))
      f[i] = std::clamp(-dz[i * stride] / drop, 0.f, 1.f);
  return f;
}

// Skin vertex -> its nearest Mark vertices (mw: Mark on the head); the basis
// rows through the similarity. Lips search Mark's lips by position and how
// far they follow the jaw (head_f, mark_f), so that the corners of a mouth
// wider or narrower than Mark's still get the corners' motion and the
// opening keeps its own shape. A vertex whose neighbours are all still gets
// no row.
void MapSkin(const Mark& mk, const Registration& g, const ShapeIn& skin,
             const std::vector<float>& mw, const std::vector<float>& head_f,
             const std::vector<float>& mark_f, const std::vector<uint8_t>& lab,
             Head& out) {
  const uint32_t nv = skin.nv, nk = mk.k;
  std::vector<float> mw4(static_cast<size_t>(mk.nv) * 4);  // + jaw follow
  std::vector<uint32_t> mark_lips;
  for (uint32_t i = 0; i < mk.nv; ++i) {
    std::copy_n(&mw[i * 3], 3, &mw4[i * 4]);
    mw4[i * 4 + 3] = kJawFollow * g.unit * mark_f[i];
    if (mark_f[i] >= 0) mark_lips.push_back(i);
  }
  const KdTree all(mw.data(), mk.nv), lip(mw4.data(), mark_lips, 4);
  out.k = nk;
  out.nv = nv;
  out.row.assign(nv, -1);
  std::vector<float> acc(static_cast<size_t>(nk) * 3);
  double lip_sum = 0;
  uint32_t lip_n = 0, rows = 0;
  for (uint32_t v = 0; v < nv; ++v) {
    uint32_t idx[kKnn];
    float dist[kKnn];
    V3 p = At(&skin.verts[v * 3]);
    const float q[4] = {p.x, p.y, p.z, kJawFollow * g.unit * head_f[v]};
    int n = head_f[v] >= 0 && !mark_lips.empty()
                ? lip.Nearest(q, kKnn, idx, dist)
                : all.Nearest(p, kKnn, idx, dist);
    double w[kKnn];
    IdwWeights(dist, n, 1e-3 * g.unit, w);
    std::fill(acc.begin(), acc.end(), 0.f);
    V3 matched{0, 0, 0};
    bool moves = false;
    for (int j = 0; j < n; ++j) {
      uint32_t mi = idx[j];
      float wj = static_cast<float>(w[j]);
      matched = matched + At(&mw[mi * 3]) * wj;
      int32_t row = mk.row[mi];
      if (row < 0) continue;
      moves = true;
      const float* b = &mk.basis[static_cast<size_t>(row) * nk * 3];
      for (uint32_t c = 0; c < nk * 3; ++c) acc[c] += wj * b[c];
    }
    if (lab[v]) {
      lip_sum += Len(matched - p);
      ++lip_n;
    }
    if (!moves) continue;
    out.row[v] = static_cast<int32_t>(rows++);
    const size_t at = out.basis.size();
    out.basis.resize(at + static_cast<size_t>(nk) * 3);
    for (uint32_t k = 0; k < nk; ++k)
      Put(&out.basis[at + k * 3], Rotate(g.rot, At(&acc[k * 3])) * g.s);
  }
  out.lip_dist = lip_n ? static_cast<float>(lip_sum / lip_n) : 0;
}

// Mark's jaw on this head: its points at rest, and the network's
// displacement of a point (Mark's axes: x left, y up, z forward) to here.
void MapJaw(const Mark& mk, const Registration& g, Head& out) {
  for (int i = 0; i < kNumJawPoints; ++i)
    Put(&out.jaw_points[i * 3], g.ToHead(At(&mk.jaw_points[i * 3])));
  const float gain = g.s * mk.jaw_strength;
  for (int i = 0; i < 3; ++i) {
    out.jaw_motion[i * 3] = -gain * g.rot[i * 3];
    out.jaw_motion[i * 3 + 1] = gain * g.rot[i * 3 + 2];
    out.jaw_motion[i * 3 + 2] = gain * g.rot[i * 3 + 1];
  }
}

}  // namespace

bool BuildHead(const Mark& mk, const std::vector<ShapeIn>& shapes,
               Head& out, std::string& why) {
  out = Head();
  const ShapeIn* skin = nullptr;
  for (const ShapeIn& s : shapes)
    if (s.skin && s.part == Part::kHead && !skin) skin = &s;
  if (!skin || !skin->nv || skin->verts.size() != skin->nv * 3) {
    why = "no head skin shape";
    return false;
  }
  if (!IsFinite(skin->verts)) {
    why = "non-finite head vertices";
    return false;
  }
  const uint32_t nv = skin->nv;
  const size_t n3 = static_cast<size_t>(nv) * 3;
  auto has = [](const ShapeIn& s, size_t m) {
    return m < s.morphs.size() && s.morphs[m].size() == s.nv * 3 &&
           IsFinite(s.morphs[m]) && MaxAbs(s.morphs[m]) >= 1e-4f;
  };
  for (Morph m : {kBlinkLeft, kBlinkRight, kBrowUpLeft, kBrowUpRight, kBigAah,
                  kOohQ, kHappy})
    if (!has(*skin, m)) {
      why = std::string("no ") + kMorphIds[m].name + " morph";
      return false;
    }
  const std::vector<float>& verts = skin->verts;
  auto morph = [&](Morph m) -> const std::vector<float>& {
    return skin->morphs[m];
  };
  const Registration g = Register(mk, *skin, out);
  // Nonzero morphs can still identify the same point. Reject collapsed or
  // inverted layouts before they produce a plausible-looking invalid fit.
  const float eye_span = 6.f * g.unit;
  const float eye_z = 0.5f * (g.lm[0].z + g.lm[1].z);
  if (!std::isfinite(g.s) || g.s <= 0 || !std::isfinite(eye_span) ||
      eye_span < 1e-4f ||
      std::fabs(g.lm[1].x - g.lm[0].x) < 0.5f * eye_span ||
      eye_z - g.lm[kLandmarkLips].z < 0.1f * eye_span ||
      Len(g.lm[3] - g.lm[2]) < 0.1f * eye_span ||
      Len(g.lm[7] - g.lm[6]) < 0.1f * eye_span ||
      !std::all_of(std::begin(out.residual), std::end(out.residual),
                   [eye_span](float residual) {
                     return std::isfinite(residual) && residual < eye_span;
                   })) {
    why = "invalid facial landmark layout";
    return false;
  }
  AutoScale(*skin, g.s, out);
  std::vector<float> mw(static_cast<size_t>(mk.nv) * 3);  // Mark on the head
  for (uint32_t i = 0; i < mk.nv; ++i)
    Put(&mw[i * 3], g.ToHead(At(&mk.verts[i * 3])));
  // How far a vertex follows the jaw (the head's BigAah, Mark's jawOpen)
  // labels the head's lips: lower where over a quarter, upper where under an
  // eighth
  const std::vector<float> head_f = JawFollow(g, &morph(kBigAah)[2], 3, nv,
                                              verts.data()),
                           mark_f =
                               JawFollow(g, mk.jaw.data(), 1, mk.nv, mw.data());
  std::vector<uint8_t> lab(nv, kNoLip);
  for (uint32_t i = 0; i < nv; ++i) {
    if (head_f[i] < 0) continue;
    lab[i] = head_f[i] > 0.25f   ? kLowerLip
             : head_f[i] < 0.12f ? kUpperLip
                                 : kNoLip;
    if (lab[i]) ++out.lips[lab[i] - kUpperLip];
  }
  MapSkin(mk, g, *skin, mw, head_f, mark_f, lab, out);
  // region weights from the head's own eye and brow lines
  const V3 lips = g.lm[kLandmarkLips];
  float z_eye = MovedZ(verts, morph(kBlinkLeft), morph(kBlinkRight), nv,
                       0.2f * g.unit, 0.5f * (g.lm[0].z + g.lm[1].z));
  float z_brow = MovedZ(verts, morph(kBrowUpLeft), morph(kBrowUpRight), nv,
                        0.1f * g.unit, 0.5f * (g.lm[2].z + g.lm[3].z));
  out.region_w.resize(static_cast<size_t>(nv) * kNumSkinRegions);
  for (uint32_t v = 0; v < nv; ++v)
    RegionWeights(g.InFittingUnits(At(&verts[v * 3])), g.mid / g.unit,
                  z_eye / g.unit, z_brow / g.unit, lips.z / g.unit,
                  &out.region_w[v * kNumSkinRegions]);
  out.lip_w.assign(static_cast<size_t>(nv) * 2, 0.f);
  for (uint32_t v = 0; v < nv; ++v) {
    if (head_f[v] < 0) continue;
    const V3 p = At(&verts[v * 3]);
    LipWeights(g.InFittingUnits(p), g.mid / g.unit, g.InFittingUnits(lips),
               head_f[v], &out.lip_w[v * 2]);
  }
  MapJaw(mk, g, out);
  // Every shape: skin, followers, model jaw and authored tongue motion.
  float off[3];
  FrameOffset(verts.data(), nv, off);
  std::vector<float> local = Shifted(verts, off);
  std::vector<float> blink(n3);
  for (size_t i = 0; i < n3; ++i)
    blink[i] = morph(kBlinkLeft)[i] + morph(kBlinkRight)[i];
  for (const ShapeIn& sh : shapes) {
    if (!sh.nv || sh.verts.size() != static_cast<size_t>(sh.nv) * 3) continue;
    Part part = sh.part;
    if (sh.mouth_candidate && (part == Part::kPinned || part == Part::kNone) &&
        OralGeometry(sh, g, off))
      part = Part::kMouth;
    if (part == Part::kNone) continue;
    Head::Shape hs;
    hs.name = sh.name;
    hs.part = part;
    hs.nv = sh.nv;
    for (int e = 0; e < kNumExpressions; ++e)
      if (PlaysExpression(e) && Reads(sh.skin, part, kExpression + e) &&
          has(sh, kExpression + e))
        hs.expr[e] = sh.morphs[kExpression + e];
    if (&sh == skin) {
      out.skin = static_cast<uint32_t>(out.shapes.size());
    } else if (part == Part::kMouth) {
      // into the skin's space: both to shape-local, the skin's back
      float mo[3];
      FrameOffset(sh.verts.data(), sh.nv, mo);
      for (int c = 0; c < 3; ++c) mo[c] -= off[c];
      Mouth(mk, sh, Shifted(sh.verts, mo), g.s, g.rot, out, hs);
    } else {
      Follow(sh, local, nv, lab, lips.z + off[2], blink, g.mid, hs);
      if (part == Part::kPinned)
        Pin(sh, local, nv, lab, lips.z + off[2], hs);
    }
    out.shapes.push_back(std::move(hs));
  }
  why.clear();
  return true;
}

bool Coefficients(const Head& h, const a2f::Line* line, float frame,
                  std::vector<float>& c) {
  const uint32_t dim = h.k + kNumMouthCoef;
  c.assign(dim, 0.f);
  if (!line || !line->frames || line->direct_dim != dim) return false;
  const a2f::FrameSample f = a2f::SampleFrame(frame, line->frames);
  const float* c0 = &line->direct[static_cast<size_t>(f.i0) * dim];
  const float* c1 = &line->direct[static_cast<size_t>(f.i1) * dim];
  for (uint32_t k = 0; k < dim; ++k) c[k] = (1 - f.a) * c0[k] + f.a * c1[k];
  return true;
}

bool Evaluate(const Head& h, const float* coefficients, const float* scales,
              const float* expressions, const float* phonemes,
              TongueMode tongue_mode, std::vector<std::vector<float>>& out,
              float upper_lip, float lower_lip) {
  // Keep the authored tongue default at 1 in the user settings.
  constexpr float kAuthoredTongueGain = 1.5f;
  // The authored cues' total: overlapping phonemes share it instead of
  // stacking into an exaggerated held pose.
  constexpr float kTongueCueTotal = 0.35f;
  const float authored_tongue_scale = scales[kTongue] * kAuthoredTongueGain;
  const uint32_t nk = h.k;
  out.resize(h.shapes.size());
  for (size_t i = 0; i < h.shapes.size(); ++i)
    out[i].assign(static_cast<size_t>(h.shapes[i].nv) * 3, 0.f);
  const float* c = coefficients;
  const bool speaking = c != nullptr;
  bool any = speaking;
  for (int e = 0; e < kNumExpressions; ++e) any |= expressions[e] > 0;
  if (!any) return false;
  thread_local std::vector<float> skin;
  skin.assign(static_cast<size_t>(h.nv) * 3, 0.f);
  if (speaking) {
    for (uint32_t v = 0; v < h.nv; ++v) {
      if (h.row[v] < 0) continue;
      float sv = RegionScale(h, v, scales);
      if (!h.lip_w.empty())
        sv *= 1 + (kUpperLipCalibration * upper_lip - 1) * h.lip_w[v * 2] +
              (kLowerLipCalibration * lower_lip - 1) * h.lip_w[v * 2 + 1];
      const float* b = &h.basis[static_cast<size_t>(h.row[v]) * nk * 3];
      float d[3] = {};
      for (uint32_t k = 0; k < nk; ++k)
        for (int j = 0; j < 3; ++j) d[j] += c[k] * b[k * 3 + j];
      for (int j = 0; j < 3; ++j) skin[v * 3 + j] = sv * d[j];
    }
  }
  // the expressions, region scaled, before the followers take the skin
  for (int e = 0; e < kNumExpressions; ++e) {
    const std::vector<float>& d = h.shapes[h.skin].expr[e];
    if (expressions[e] <= 0 || d.empty()) continue;
    for (uint32_t v = 0; v < h.nv; ++v) {
      const float sv = expressions[e] * RegionScale(h, v, kExpressionScale);
      for (int j = 0; j < 3; ++j) skin[v * 3 + j] += sv * d[v * 3 + j];
    }
  }
  // the jaw: the rigid motion that takes its points at rest to where the
  // network puts them
  const float* tongue = speaking ? c + nk : nullptr;
  float jr[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  V3 jt{0, 0, 0};
  if (speaking) {
    V3 rest[kNumJawPoints], now[kNumJawPoints];
    for (int i = 0; i < kNumJawPoints; ++i) {
      rest[i] = At(&h.jaw_points[i * 3]);
      now[i] =
          rest[i] + Rotate(h.jaw_motion, At(&tongue[a2f::kNumTongue + i * 3])) *
                        scales[kJaw];
    }
    float one;
    Similarity(rest, now, kNumJawPoints, one, jr, jt, true);
  }
  // Use every authored phoneme on the tongue mask. These are visual tuning
  // defaults, not an anatomical reconstruction from the LIP track.
  float tongue_cues[kNumPhonemes] = {};
  float cue_sum = 0;
  for (int ph = 0; phonemes && ph < kNumPhonemes; ++ph) {
    tongue_cues[ph] = std::max(0.f, phonemes[ph]);
    cue_sum += tongue_cues[ph];
  }
  for (float& cue : tongue_cues)
    cue *= kTongueCueTotal / std::max(1.f, cue_sum);
  for (size_t si = 0; si < h.shapes.size(); ++si) {
    const Head::Shape& s = h.shapes[si];
    std::vector<float>& o = out[si];
    if (si == h.skin) {
      o = skin;
    } else if (s.part == Part::kMouth) {
      const size_t n3 = static_cast<size_t>(s.nv) * 3;
      for (int e = 0; e < kNumExpressions; ++e) {
        if (expressions[e] <= 0 || s.expr[e].size() != n3) continue;
        const float* d = s.expr[e].data();
        const float we = expressions[e] * kExpressionScale[kMouth];
        for (size_t i = 0; i < n3; ++i) o[i] += we * d[i];
      }
      if (!speaking || s.pos.size() != n3 || s.jaw.size() != s.nv) continue;
      const bool vanilla = tongue_mode == TongueMode::kAuthored && phonemes &&
                           s.tongue_weight.size() == s.nv &&
                           s.tongue.size() == n3 * kNumPhonemes;
      const bool model =
          tongue_mode == TongueMode::kModel && s.model_jaw.size() == s.nv;
      const bool model_bends =
          model && s.model_tongue.size() == n3 * a2f::kNumTongue;
      for (uint32_t i = 0; i < s.nv; ++i) {
        V3 p = At(&s.pos[i * 3]);
        const float mask = vanilla ? s.tongue_weight[i] : 0;
        // Rest with half the jaw movement, including tongues still in BigAah.
        const float jaw = model ? s.model_jaw[i]
                                : s.jaw[i] * (1 - mask) + 0.5f * mask;
        V3 d = (Rotate(jr, p) + jt - p) * jaw;
        if (vanilla && mask > 0) {
          const float* b = &s.tongue[static_cast<size_t>(i) * kNumPhonemes * 3];
          for (int ph = 0; ph < kNumPhonemes; ++ph) {
            d = d + At(&b[ph * 3]) * (tongue_cues[ph] * authored_tongue_scale);
          }
        }
        if (model_bends && jaw > 0) {
          const float* b =
              &s.model_tongue[static_cast<size_t>(i) * a2f::kNumTongue * 3];
          for (int k = 0; k < a2f::kNumTongue; ++k)
            d = d + At(&b[k * 3]) * (tongue[k] * scales[kTongue]);
        }
        o[i * 3] += d.x;
        o[i * 3 + 1] += d.y;
        o[i * 3 + 2] += d.z;
      }
    } else {
      for (uint32_t i = 0; i < s.nv; ++i)
        for (int k = 0; k < kFollow; ++k) {
          float wk = s.w[i * kFollow + k];
          if (wk == 0) continue;
          uint32_t r = s.idx[i * kFollow + k];
          for (int j = 0; j < 3; ++j) o[i * 3 + j] += wk * skin[r * 3 + j];
        }
    }
  }
  return true;
}

}  // namespace direct
