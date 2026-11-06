#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "a2f/animation/direct.h"
static direct::Model model;
extern "C" int init(const char* path) {
  std::string p(path), why;
  return model.Load(std::wstring(p.begin(), p.end()), why);
}
extern "C" void* fit(int female, int nv, const float* v, const float* morph,
                     float* stats, char* why) {
  direct::ShapeIn s;
  s.skin = true;
  s.part = direct::Part::kHead;
  s.nv = nv;
  s.verts.assign(v, v + nv * 3);
  s.morphs.resize(direct::kNumMorphs);
  for (int m = 0; m < direct::kNumMorphs; ++m)
    s.morphs[m].assign(morph + m * nv * 3, morph + (m + 1) * nv * 3);
  auto* h = new direct::Head;
  std::string reason;
  bool ok = direct::BuildHead(model.Face(female), {s}, *h, reason);
  std::strncpy(why, reason.c_str(), 1023);
  stats[0] = ok;
  stats[1] = h->calibration_ignored;
  stats[2] = h->scale;
  stats[3] = h->lip_dist;
  stats[4] = h->lips[0];
  stats[5] = h->lips[1];
  stats[6] = h->k;
#ifdef BFA_AUDIT_CALIBRATION
  stats[21] = static_cast<float>(h->calibrated_lips.size());
#elif !defined(BFA_AUDIT_LEGACY_HEAD)
  stats[21] = static_cast<float>(h->lip_repairs.size());
#endif
  for (int i = 0; i < 8; ++i) stats[7 + i] = h->residual[i];
  for (int i = 0; i < 6; ++i) stats[15 + i] = h->auto_scale[i];
  if (!ok) {
    delete h;
    return nullptr;
  }
  // Skin-only inputs lack oral seam shapes. Exercise the jaw-cap branch on
  // every weighted lower-lip vertex as a synthetic stress case, not evidence
  // that this particular NPC has a shared oral border in game.
  if (h->lip_jaw.empty())
    for (uint32_t i = 0; i < h->nv; ++i)
      if (!h->lip_w.empty() && h->lip_w[i * 2 + 1] > 0)
        h->lip_jaw.push_back(
            {i, {v[i * 3], v[i * 3 + 1], v[i * 3 + 2]}, h->lip_w[i * 2 + 1]});
  return h;
}
// Playback state for skin-only inputs. Legacy contact state is checked
// separately; current calibration is fully represented by the cached arrays.
extern "C" uint64_t fingerprint(void* ptr) {
  auto& h = *(direct::Head*)ptr;
  uint64_t hash = 14695981039346656037ull;
  auto add = [&](const void* p, size_t n) {
    auto* b = (const unsigned char*)p;
    for (size_t i = 0; i < n; ++i) {
      hash ^= b[i];
      hash *= 1099511628211ull;
    }
  };
  add(h.row.data(), h.row.size() * 4);
  for (auto* v : {&h.basis, &h.lip_w, &h.region_w})
    add(v->data(), v->size() * 4);
#ifdef BFA_AUDIT_NEUTRAL_WEIGHTS
  add(h.lip_neutral_w.data(), h.lip_neutral_w.size() * sizeof(float));
#endif
  add(h.auto_scale, sizeof(h.auto_scale));
  add(h.jaw_points, sizeof(h.jaw_points));
  add(h.jaw_motion, sizeof(h.jaw_motion));
  for (const auto& lip : h.lip_jaw) {
    add(&lip.vertex, sizeof(lip.vertex));
    add(lip.pos, sizeof(lip.pos));
    add(&lip.weight, sizeof(lip.weight));
  }
  return hash;
}
extern "C" bool has_repair_state(void* ptr) {
#if defined(BFA_AUDIT_LEGACY_HEAD) || defined(BFA_AUDIT_CALIBRATION)
  return false;
#else
  const auto& h = *static_cast<const direct::Head*>(ptr);
#ifdef BFA_AUDIT_NEUTRAL_WEIGHTS
  if (!h.lip_neutral_w.empty()) return true;
#endif
  return !h.lip_repairs.empty() || !h.lip_contacts.empty() ||
         !h.lip_blend.empty();
#endif
}
extern "C" bool has_extended_controls(void* ptr) {
#ifdef BFA_AUDIT_CALIBRATION
  return static_cast<const direct::Head*>(ptr)->extended_lip_controls;
#elif defined(BFA_AUDIT_NEUTRAL_WEIGHTS)
  return !static_cast<const direct::Head*>(ptr)->lip_neutral_w.empty();
#else
  return false;
#endif
}
extern "C" int evaluate_controls(void* ptr, int frames, int dim,
                                 const float* coef, float* out, int limited,
                                 float upper, float lower, int expression) {
  auto& h = *(direct::Head*)ptr;
  if (dim != int(h.k + direct::kNumMouthCoef)) return 0;
  float expr[direct::kNumExpressions] = {};
  if (expression >= 0 && expression < direct::kNumExpressions)
    expr[expression] = 0.7f;
  std::vector<std::vector<float>> d;
  for (int t = 0; t < frames; ++t) {
    if (!direct::Evaluate(h, coef + t * dim, h.auto_scale, expr, nullptr,
                          direct::TongueMode::kAuthored, d, upper, lower, 0, 0,
                          limited != 0))
      return 0;
    std::copy(d[0].begin(), d[0].end(), out + t * h.nv * 3);
  }
  return 1;
}
extern "C" int evaluate_mode(void* ptr, int frames, int dim, const float* coef,
                             float* out, int limited) {
  return evaluate_controls(ptr, frames, dim, coef, out, limited, 1, 1, -1);
}
extern "C" int evaluate(void* ptr, int frames, int dim, const float* coef,
                        float* out) {
  return evaluate_mode(ptr, frames, dim, coef, out, 0);
}
extern "C" void release(void* ptr) { delete (direct::Head*)ptr; }
extern "C" int benchmark(int female, int nv, const float* v, const float* morph,
                         int loops, double* ms) {
  direct::ShapeIn s;
  s.skin = true;
  s.part = direct::Part::kHead;
  s.nv = nv;
  s.verts.assign(v, v + nv * 3);
  s.morphs.resize(direct::kNumMorphs);
  for (int m = 0; m < direct::kNumMorphs; ++m)
    s.morphs[m].assign(morph + m * nv * 3, morph + (m + 1) * nv * 3);
  std::vector<direct::ShapeIn> shapes;
  shapes.push_back(std::move(s));
  for (int i = 0; i < loops; ++i) {
    direct::Head h;
    std::string why;
    auto start = std::chrono::steady_clock::now();
    bool ok = direct::BuildHead(model.Face(female), shapes, h, why);
    ms[i] = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start)
                .count();
    if (!ok) return 0;
  }
  return 1;
}
extern "C" void repair_mask(void* ptr, uint8_t* mask) {
  auto& h = *(direct::Head*)ptr;
  std::fill_n(mask, h.nv, 0);
#ifdef BFA_AUDIT_CALIBRATION
  for (uint32_t vertex : h.calibrated_lips) mask[vertex] = 1;
#elif !defined(BASELINE) && !defined(BFA_AUDIT_LEGACY_HEAD)
  for (const auto& r : h.lip_repairs) mask[r.vertex] = 1;
#endif
}
extern "C" uint64_t expression_fingerprint(void* ptr) {
  auto& h = *(direct::Head*)ptr;
  uint64_t hash = 14695981039346656037ull;
  float scales[direct::kNumRegions] = {1, 1, 1, 1, 1, 1};
  for (int e = 0; e < direct::kNumExpressions; ++e) {
    float expr[direct::kNumExpressions] = {};
    expr[e] = 1;
    std::vector<std::vector<float>> out;
    if (!direct::Evaluate(h, nullptr, scales, expr, nullptr,
                          direct::TongueMode::kAuthored, out))
      return 0;
    for (const auto& shape : out) {
      auto* bytes = reinterpret_cast<const uint8_t*>(shape.data());
      for (size_t i = 0; i < shape.size() * 4; ++i)
        hash = (hash ^ bytes[i]) * 1099511628211ull;
    }
  }
  return hash;
}
