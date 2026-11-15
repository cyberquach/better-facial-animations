#include <cassert>
#include <cstdio>
#include <vector>

#include "a2f/animation/rest_store.h"

namespace {

std::vector<float> Data(const std::vector<float>& xyz) {
  std::vector<float> data(xyz.size() / 3 * 4, 1.f);
  for (size_t i = 0; i < xyz.size() / 3; ++i)
    for (int c = 0; c < 3; ++c) data[i * 4 + c] = xyz[i * 3 + c];
  return data;
}

// Writes xyz into a shape's dynamic data in place, as the engine does.
void Set(std::vector<float>& data, const std::vector<float>& xyz) {
  for (size_t i = 0; i < xyz.size() / 3; ++i)
    for (int c = 0; c < 3; ++c) data[i * 4 + c] = xyz[i * 3 + c];
}

}  // namespace

int main() {
  bfa::RestStore store;
  const int shape = 0, other = 0;
  const std::vector<float> rest = {0, 0, 0, 1, 2, 3};
  const std::vector<float> moved = {0, 0, -1, 1, 2, 2};
  std::vector<float> data = Data(rest);  // the shape's dynamic data
  std::vector<float> back;
  assert(store.empty());
  assert(!store.Displaced(&shape, data.data(), 2, back));

  // first sight: the shape's own vertices are its rest; nothing to put back
  assert(*store.Rest(&shape, data.data(), 2) == rest && !store.empty());
  assert(!store.Displaced(&shape, data.data(), 2, back));

  // a write displaces it; rest stays what it was, frame after frame
  Set(data, moved);
  store.Wrote(&shape, 2);
  assert(*store.Rest(&shape, data.data(), 2) == rest);
  // putting it back counts as written: there is nothing more to put back
  assert(store.Displaced(&shape, data.data(), 2, back) && back == rest);
  Set(data, back);
  assert(!store.Displaced(&shape, data.data(), 2, back));

  // Another writer working from the plugin's pose (here moving it further)
  // does not make that the rest, and the reset puts the rest back over it.
  Set(data, moved);
  store.Wrote(&shape, 2);
  Set(data, {0, 0, -2, 1, 2, 1});
  assert(*store.Rest(&shape, data.data(), 2) == rest);
  assert(store.Displaced(&shape, data.data(), 2, back) && back == rest);
  Set(data, back);
  // A shape the plugin has not written since its reset is left to the other
  // writer, and what it writes is still not the rest.
  Set(data, moved);
  assert(!store.Displaced(&shape, data.data(), 2, back));
  assert(*store.Rest(&shape, data.data(), 2) == rest);
  Set(data, rest);

  // a write of another vertex count is not this shape's
  store.Wrote(&shape, 3);
  assert(!store.Displaced(&shape, data.data(), 2, back));

  // writes to shapes never asked about are not recorded
  store.Wrote(&other, 2);
  assert(!store.Displaced(&other, data.data(), 2, back));

  // A new shape at a freed shape's address must not inherit the old rest,
  // even while the old shape was written: it is told by identity (its own
  // dynamic data or vertex count), never by content, which other writers
  // change.
  store.Wrote(&shape, 2);
  const std::vector<float> replaced = {5, 5, 5, 6, 6, 6};
  std::vector<float> replaced_data = Data(replaced);
  // the old shape's write is not the new one's
  assert(!store.Displaced(&shape, replaced_data.data(), 2, back));
  assert(*store.Rest(&shape, replaced_data.data(), 2) == replaced);
  assert(!store.Displaced(&shape, replaced_data.data(), 2, back));
  const std::vector<float> longer = {5, 5, 5, 6, 6, 6, 7, 7, 7};
  std::vector<float> longer_data = Data(longer);
  assert(*store.Rest(&shape, longer_data.data(), 3) == longer);
  // same dynamic data, another vertex count (its first two vertices)
  assert(*store.Rest(&shape, longer_data.data(), 2) == replaced);

  // many shapes at rest are forgotten; a written one is kept
  assert(*store.Rest(&shape, longer_data.data(), 3) == longer);
  store.Wrote(&shape, 3);
  Set(longer_data, std::vector<float>(9, 1.f));
  std::vector<int> many(600);
  for (int& m : many) store.Rest(&m, data.data(), 2);
  assert(store.Displaced(&shape, longer_data.data(), 3, back) &&
         back == longer);

  // a shape put back stays known while others come and go before it is
  // written again (another thread's passes), so the write is noted
  std::vector<int> others(100);
  for (int& m : others) store.Rest(&m, data.data(), 2);
  store.Wrote(&shape, 3);
  assert(store.Displaced(&shape, longer_data.data(), 3, back));

  // a new world forgets every shape
  store.Wrote(&shape, 3);
  store.Clear();
  assert(store.empty() && !store.Displaced(&shape, longer_data.data(), 3, back));
  std::puts("rest_store_test: ok");
}
