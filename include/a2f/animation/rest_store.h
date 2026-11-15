// Rest positions of dynamic face shapes the engine keeps no base vertices
// for: geometry baked into a FaceGen mesh whose head part the NPC does not
// list. The engine never rebuilds such a shape, so the plugin remembers what
// it looked like before its first write and puts that back itself.
#ifndef INCLUDE_A2F_ANIMATION_REST_STORE_H_
#define INCLUDE_A2F_ANIMATION_REST_STORE_H_

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace bfa {

// A shape's identity is its address, its vertex count and its dynamic data
// (data: the shape's nv x 4 vertices as LockDynamic returns them). The rest
// (nv x 3) is taken from data when an identity is first seen and kept for
// that identity's lifetime: whatever writes the shape later, the plugin or
// another writer, never changes it. A new identity at the same address
// starts over from its own vertices.
class RestStore {
 public:
  // The shape's rest positions.
  std::shared_ptr<const std::vector<float>> Rest(const void* shape,
                                                 const float* data,
                                                 uint32_t nv) {
    std::lock_guard<std::mutex> guard(lock_);
    Entry& e = entries_[shape];
    if (!e.rest || e.data != data || e.nv != nv) {
      auto rest = std::make_shared<std::vector<float>>(size_t{nv} * 3);
      for (uint32_t i = 0; i < nv; ++i)
        for (int c = 0; c < 3; ++c) (*rest)[i * 3 + c] = data[i * 4 + c];
      e = Entry{std::move(rest), data, nv, false};
    }
    e.used = ++clock_;
    auto rest = e.rest;
    if (entries_.size() > kMaxShapes) Prune();
    count_ = entries_.size();
    return rest;
  }

  // The plugin wrote the shape (the identity Rest last saw at that address).
  // Unknown shapes are ignored.
  void Wrote(const void* shape, uint32_t nv) {
    if (!count_) return;
    std::lock_guard<std::mutex> guard(lock_);
    auto it = entries_.find(shape);
    if (it == entries_.end() || it->second.nv != nv) return;
    it->second.written = true;
    it->second.used = ++clock_;
  }

  // Whether the plugin wrote this identity since it was last put back,
  // whether or not another writer touched it since; xyz then holds the rest
  // and the shape counts as at rest, so the caller must write xyz.
  bool Displaced(const void* shape, const float* data, uint32_t nv,
                 std::vector<float>& xyz) {
    if (!count_) return false;
    std::lock_guard<std::mutex> guard(lock_);
    auto it = entries_.find(shape);
    if (it == entries_.end() || !it->second.written ||
        it->second.data != data || it->second.nv != nv)
      return false;
    xyz = *it->second.rest;
    it->second.written = false;
    it->second.used = ++clock_;
    return true;
  }

  bool empty() const { return !count_; }

  // Forgets every shape (a new world: none of its shapes are the old ones).
  void Clear() {
    std::lock_guard<std::mutex> guard(lock_);
    entries_.clear();
    count_ = 0;
  }

 private:
  struct Entry {
    std::shared_ptr<const std::vector<float>> rest;
    const float* data = nullptr;
    uint32_t nv = 0;
    bool written = false;
    uint64_t used = 0;  // when last seen, put back or written (clock_)
  };
  static constexpr size_t kMaxShapes = 256;
  // written shapes never put back (unloaded mid-pose) beyond this many go
  static constexpr size_t kMaxWritten = 1024;

  // Forgets the least recently used shapes at rest down to half the bound,
  // so one between being put back and written again (most recently used)
  // stays; a written one still needs putting back, unless far too many are.
  void Prune() {
    std::vector<std::pair<uint64_t, const void*>> idle, written;
    for (const auto& [shape, e] : entries_)
      (e.written ? written : idle).emplace_back(e.used, shape);
    const auto drop = [this](auto& by_use, size_t keep) {
      if (by_use.size() <= keep) return;
      std::sort(by_use.begin(), by_use.end());
      for (size_t i = 0; i < by_use.size() - keep; ++i)
        entries_.erase(by_use[i].second);
    };
    drop(idle, kMaxShapes / 2);
    drop(written, kMaxWritten);
  }

  std::mutex lock_;
  std::unordered_map<const void*, Entry> entries_;
  uint64_t clock_ = 0;
  std::atomic<size_t> count_{0};
};

}  // namespace bfa

#endif  // INCLUDE_A2F_ANIMATION_REST_STORE_H_
