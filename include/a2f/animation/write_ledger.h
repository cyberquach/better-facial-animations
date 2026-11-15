// What the plugin last wrote to each dynamic face shape since the shape's
// last reset. A reset the morph pass claims (force, or the node's reset
// flag) may not have happened: a hook behind the plugin's can skip the
// engine's pass and leave the plugin's last pose in the buffer, and deltas
// added on top of it would double the pose. A shape still holding the
// plugin's last write was not reset. A node with writes noted keeps the
// plugin's shape-lock span on its next reset, so its shapes go back together.
#ifndef INCLUDE_A2F_ANIMATION_WRITE_LEDGER_H_
#define INCLUDE_A2F_ANIMATION_WRITE_LEDGER_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace bfa {

// Nodes and shapes are compared by address, never dereferenced. data: a
// shape's nv x 4 vertex buffer.
class WriteLedger {
 public:
  // The plugin wrote the node's shape whose vertex buffer is data, which
  // now hashes (Hash) to hash. Only a write that changed the buffer is to be
  // noted: one that left it as it was cannot be told from a reset.
  void Wrote(const void* node, const void* shape, const float* data,
             uint32_t nv, uint64_t hash) {
    const Write w{shape, data, nv, hash};
    std::lock_guard<std::mutex> guard(lock_);
    Node& n = nodes_[node];
    n.stamp = ++clock_;
    for (Write& old : n.writes)
      if (old.shape == shape) {
        old = w;
        return;
      }
    n.writes.push_back(w);
    if (nodes_.size() > kMaxNodes) Prune();
    count_ = nodes_.size();
  }

  // Whether a write to one of the node's shapes is noted since its reset.
  bool Pending(const void* node) const {
    if (!count_) return false;
    std::lock_guard<std::mutex> guard(lock_);
    return nodes_.count(node) != 0;
  }

  // Whether the node's shape (its vertex buffer data) still holds the
  // plugin's last write to it: a claimed reset did not reach it.
  bool Holds(const void* node, const void* shape, const float* data,
             uint32_t nv) const {
    if (!count_) return false;
    Write w{};
    {
      std::lock_guard<std::mutex> guard(lock_);
      const auto it = nodes_.find(node);
      if (it == nodes_.end()) return false;
      for (const Write& noted : it->second.writes)
        if (noted.shape == shape) w = noted;
    }
    return w.shape && w.data == data && w.nv == nv &&
           Hash(data, nv) == w.hash;
  }

  // The node's shapes were reset: their writes are forgotten.
  void Reset(const void* node) {
    if (!count_) return;
    std::lock_guard<std::mutex> guard(lock_);
    nodes_.erase(node);
    count_ = nodes_.size();
  }

  // Everything is forgotten (a new world: no node of the old one is drawn).
  void Clear() {
    std::lock_guard<std::mutex> guard(lock_);
    nodes_.clear();
    count_ = 0;
  }

  // FNV-1a over the bits of each vertex's x, y, z in a shape's nv x 4
  // vertex buffer.
  static uint64_t Hash(const float* data, uint32_t nv) {
    uint64_t h = 14695981039346656037ull;
    for (uint32_t i = 0; i < nv; ++i)
      for (int c = 0; c < 3; ++c) {
        uint32_t bits;
        std::memcpy(&bits, &data[size_t{i} * 4 + c], sizeof bits);
        h = (h ^ bits) * 1099511628211ull;
      }
    return h;
  }

 private:
  struct Write {
    const void* shape;
    const float* data;
    uint32_t nv;
    uint64_t hash;
  };
  struct Node {
    std::vector<Write> writes;
    uint64_t stamp = 0;
  };
  // Nodes unloaded before their next reset are never reset: the least
  // recently written beyond this many are forgotten.
  static constexpr size_t kMaxNodes = 64;

  void Prune() {
    auto oldest = nodes_.begin();
    for (auto it = nodes_.begin(); it != nodes_.end(); ++it)
      if (it->second.stamp < oldest->second.stamp) oldest = it;
    nodes_.erase(oldest);
  }

  mutable std::mutex lock_;
  std::unordered_map<const void*, Node> nodes_;
  uint64_t clock_ = 0;
  std::atomic<size_t> count_{0};
};

}  // namespace bfa

#endif  // INCLUDE_A2F_ANIMATION_WRITE_LEDGER_H_
