// Rest positions of dynamic face shapes the engine keeps no base vertices
// for: geometry baked into a FaceGen mesh whose head part the NPC does not
// list. The engine never rebuilds such a shape, so the plugin remembers what
// it looked like before its first write and puts that back itself.
#ifndef INCLUDE_A2F_ANIMATION_REST_STORE_H_
#define INCLUDE_A2F_ANIMATION_REST_STORE_H_

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace bfa {

// Shapes are keyed by address and recognized by content: an entry holds only
// while the shape still has the vertices last written through it, so
// a new shape at a freed one's address starts over from its own vertices.
// data: the shape's nv x 4 vertices; xyz: nv x 3.
class RestStore {
 public:
  // The shape's rest positions (nv x 3), taken from data on first sight.
  std::shared_ptr<const std::vector<float>> Rest(const void* shape,
                                                 const float* data,
                                                 uint32_t nv) {
    std::lock_guard<std::mutex> guard(lock_);
    Entry& e = entries_[shape];
    if (!Holds(e, data, nv)) {
      if (entries_.size() > kMaxShapes) Prune(shape);
      auto rest = std::make_shared<std::vector<float>>(size_t{nv} * 3);
      for (uint32_t i = 0; i < nv; ++i)
        for (int c = 0; c < 3; ++c) (*rest)[i * 3 + c] = data[i * 4 + c];
      e.written = *rest;
      e.rest = std::move(rest);
      count_ = entries_.size();
    }
    return e.rest;
  }

  // The shape now holds xyz. Unknown shapes are ignored.
  void Wrote(const void* shape, const float* xyz, uint32_t nv) {
    if (!count_) return;
    std::lock_guard<std::mutex> guard(lock_);
    auto it = entries_.find(shape);
    if (it == entries_.end() || it->second.written.size() != size_t{nv} * 3)
      return;
    it->second.written.assign(xyz, xyz + size_t{nv} * 3);
  }

  // Whether the shape is away from its rest positions; xyz then holds them
  // and the shape counts as holding them, so the caller must write xyz.
  bool Displaced(const void* shape, const float* data, uint32_t nv,
                 std::vector<float>& xyz) {
    if (!count_) return false;
    std::lock_guard<std::mutex> guard(lock_);
    auto it = entries_.find(shape);
    if (it == entries_.end() || !Holds(it->second, data, nv) ||
        !it->second.rest || it->second.written == *it->second.rest)
      return false;
    xyz = *it->second.rest;
    it->second.written = *it->second.rest;
    return true;
  }

  bool empty() const { return !count_; }

 private:
  struct Entry {
    std::shared_ptr<const std::vector<float>> rest;
    std::vector<float> written;
  };
  static constexpr size_t kMaxShapes = 256;
  // slack for the engine's copy of a write; far below any real difference
  // between two shapes
  static constexpr float kSame = 1e-4f;

  static bool Holds(const Entry& e, const float* data, uint32_t nv) {
    if (e.written.empty() || e.written.size() != size_t{nv} * 3) return false;
    for (uint32_t i = 0; i < nv; ++i)
      for (int c = 0; c < 3; ++c)
        if (!(std::fabs(e.written[i * 3 + c] - data[i * 4 + c]) <= kSame))
          return false;
    return true;
  }

  // Forgets shapes at rest; a displaced one may still need putting back.
  void Prune(const void* keep) {
    for (auto it = entries_.begin(); it != entries_.end();)
      it = it->first != keep &&
                   (!it->second.rest || it->second.written == *it->second.rest)
               ? entries_.erase(it)
               : std::next(it);
  }

  std::mutex lock_;
  std::unordered_map<const void*, Entry> entries_;
  std::atomic<size_t> count_{0};
};

}  // namespace bfa

#endif  // INCLUDE_A2F_ANIMATION_REST_STORE_H_
