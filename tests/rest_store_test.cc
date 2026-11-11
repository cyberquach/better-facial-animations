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

}  // namespace

int main() {
  bfa::RestStore store;
  const int shape = 0, other = 0;
  const std::vector<float> rest = {0, 0, 0, 1, 2, 3};
  std::vector<float> back;
  assert(store.empty());
  assert(!store.Displaced(&shape, Data(rest).data(), 2, back));

  // first sight: the shape's own vertices are its rest
  assert(*store.Rest(&shape, Data(rest).data(), 2) == rest && !store.empty());
  assert(!store.Displaced(&shape, Data(rest).data(), 2, back));

  // a write displaces it; rest stays what it was, frame after frame
  std::vector<float> moved = {0, 0, -1, 1, 2, 2};
  store.Wrote(&shape, moved.data(), 2);
  assert(*store.Rest(&shape, Data(moved).data(), 2) == rest);
  // the engine's copy of a write may differ slightly; it is still that write
  std::vector<float> nudged = moved;
  for (float& v : nudged) v += 5e-5f;
  assert(*store.Rest(&shape, Data(nudged).data(), 2) == rest);
  // putting it back counts as written: there is nothing more to put back
  assert(store.Displaced(&shape, Data(moved).data(), 2, back) && back == rest);
  assert(!store.Displaced(&shape, Data(rest).data(), 2, back));

  // a write of another vertex count is not this shape's
  store.Wrote(&shape, std::vector<float>(9, 7.f).data(), 3);
  assert(!store.Displaced(&shape, Data(rest).data(), 2, back));
  assert(*store.Rest(&shape, Data(rest).data(), 2) == rest);

  // writes to shapes never asked about are not recorded
  store.Wrote(&other, moved.data(), 2);
  assert(!store.Displaced(&other, Data(moved).data(), 2, back));

  // a different shape at the same address starts over from its own vertices
  store.Wrote(&shape, moved.data(), 2);
  const std::vector<float> replaced = {5, 5, 5, 6, 6, 6};
  assert(!store.Displaced(&shape, Data(replaced).data(), 2, back));
  assert(*store.Rest(&shape, Data(replaced).data(), 2) == replaced);
  const std::vector<float> longer = {5, 5, 5, 6, 6, 6, 7, 7, 7};
  assert(*store.Rest(&shape, Data(longer).data(), 3) == longer);
  assert(!store.Displaced(&shape, Data(replaced).data(), 2, back));

  // many shapes at rest are forgotten; a displaced one is kept
  store.Wrote(&shape, std::vector<float>(9, 1.f).data(), 3);
  std::vector<int> many(600);
  for (int& m : many) store.Rest(&m, Data(rest).data(), 2);
  assert(store.Displaced(&shape, Data(std::vector<float>(9, 1.f)).data(), 3,
                         back) &&
         back == longer);
  std::puts("rest_store_test: ok");
}
