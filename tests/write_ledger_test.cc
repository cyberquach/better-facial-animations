#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "a2f/animation/write_ledger.h"

#define CHECK(x)                                              \
  do {                                                        \
    if (!(x)) {                                               \
      std::fprintf(stderr, "write ledger %d: %s\n", __LINE__, #x); \
      std::exit(1);                                           \
    }                                                         \
  } while (0)

namespace {

// A shape's vertex buffer (nv x 4) holding xyz (nv x 3).
std::vector<float> Buffer(const std::vector<float>& xyz) {
  std::vector<float> data(xyz.size() / 3 * 4, 1.f);
  for (size_t i = 0; i < xyz.size() / 3; ++i)
    for (int c = 0; c < 3; ++c) data[i * 4 + c] = xyz[i * 3 + c];
  return data;
}

// The plugin wrote the shape whose buffer (nv 2) now holds xyz.
void Note(bfa::WriteLedger& ledger, const void* node, const void* shape,
          const float* data, const std::vector<float>& xyz) {
  const std::vector<float> written = Buffer(xyz);
  ledger.Wrote(node, shape, data, 2, bfa::WriteLedger::Hash(written.data(), 2));
}

}  // namespace

int main() {
  bfa::WriteLedger ledger;
  const int node = 0, other_node = 0, shape = 0, other_shape = 0;
  const std::vector<float> pose = {0, 0, -1, 1, 2, 2};
  std::vector<float> buffer = Buffer(pose);
  CHECK(!ledger.Pending(&node));
  CHECK(!ledger.Holds(&node, &shape, buffer.data(), 2));

  // the write is still there: a claimed reset did not reach the shape
  Note(ledger, &node, &shape, buffer.data(), pose);
  CHECK(ledger.Pending(&node) && !ledger.Pending(&other_node));
  CHECK(ledger.Holds(&node, &shape, buffer.data(), 2));
  CHECK(!ledger.Holds(&node, &other_shape, buffer.data(), 2));
  CHECK(!ledger.Holds(&other_node, &shape, buffer.data(), 2));
  // the 4th float of each vertex is not the plugin's
  buffer[3] = 7;
  CHECK(ledger.Holds(&node, &shape, buffer.data(), 2));

  // the engine rebuilt it (any vertex moved), or the buffer or its size is
  // another's: reset
  buffer[6] += 1e-6f;
  CHECK(!ledger.Holds(&node, &shape, buffer.data(), 2));
  const std::vector<float> fresh = Buffer(pose);
  std::copy(fresh.begin(), fresh.end(), buffer.begin());  // same storage
  CHECK(ledger.Holds(&node, &shape, buffer.data(), 2));
  std::vector<float> copy = buffer;
  CHECK(!ledger.Holds(&node, &shape, copy.data(), 2));
  CHECK(!ledger.Holds(&node, &shape, buffer.data(), 1));

  // a later write replaces the earlier one
  const std::vector<float> next = {0, 0, -2, 1, 2, 1};
  Note(ledger, &node, &shape, buffer.data(), next);
  CHECK(!ledger.Holds(&node, &shape, buffer.data(), 2));
  buffer = Buffer(next);
  Note(ledger, &node, &shape, buffer.data(), next);
  CHECK(ledger.Holds(&node, &shape, buffer.data(), 2));

  // a verified reset forgets the node's writes, not another node's
  std::vector<float> other_buffer = Buffer(pose);
  Note(ledger, &other_node, &other_shape, other_buffer.data(), pose);
  ledger.Reset(&node);
  CHECK(!ledger.Pending(&node) && !ledger.Holds(&node, &shape, buffer.data(), 2));
  CHECK(ledger.Pending(&other_node));
  ledger.Reset(&other_node);
  CHECK(!ledger.Pending(&other_node));

  // nodes never reset again (unloaded) do not accumulate: the least
  // recently written go first
  std::vector<char> nodes(200);
  for (char& n : nodes) Note(ledger, &n, &shape, buffer.data(), next);
  CHECK(!ledger.Pending(&nodes[0]) && ledger.Pending(&nodes.back()));
  int pending = 0;
  for (char& n : nodes) pending += ledger.Pending(&n);
  CHECK(pending == 64);
  // a new world forgets every node
  ledger.Clear();
  CHECK(!ledger.Pending(&nodes.back()));
  std::puts("ok: write ledger");
}
