// Production detection and playback calibration; only resource I/O is mocked.
// Optional arguments exercise the user's MaleHead.tri and FemaleHead.tri files.
#include <array>
#include <cassert>
#include <fstream>

#include "../src/better_facial_animations.cc"

namespace {
std::vector<uint8_t> files[2];
std::array<uint8_t, 16> resource{};
const std::vector<uint8_t>* active;
size_t cursor;
bool short_read = false;
int opened = 0, closed = 0;

uint32_t Read(void*, void* out, uint32_t count, uint32_t*, uint32_t) {
  if (short_read && cursor >= active->size() / 2) return 0;
  const size_t n =
      std::min({size_t(count), active->size() - cursor, size_t(4096)});
  memcpy(out, active->data() + cursor, n);
  cursor += n;
  return static_cast<uint32_t>(n);
}

void* Open(void* stream, const char* path, uint64_t, void*) {
  using namespace bfa;
  const bool female = strcmp(path, kExpressiveHeads[1].path) == 0;
  assert(strcmp(path, kExpressiveHeads[female].path) == 0);
  active = &files[female];
  cursor = 0;
  ++opened;
  auto* s = static_cast<uint8_t*>(stream);
  Field<void*>(s, 0x20) = active->empty() ? nullptr : resource.data();
  Field<uint32_t>(resource.data(), 8) = static_cast<uint32_t>(active->size());
  Field<decltype(&Read)>(s, 0x10) = Read;
  return stream;
}

void Close(void*) { ++closed; }

void Detect(bool male, bool female) {
  bfa::LoadExpressiveFaces();
  assert(bfa::config.expressive_faces[0] == male);
  assert(bfa::config.expressive_faces[1] == female);
  assert(opened == closed);
}

void CheckScales() {
  using namespace bfa;
  direct::Head head;
  head.k = head.nv = 1;
  head.row = {0};
  head.basis = {0, 0, 1};
  head.region_w = {0, 0, 0, 1};
  head.shapes.resize(1);
  head.shapes[0].nv = 1;
  head.auto_scale[direct::kMouth] = 0.5f;
  float coefficients[1 + direct::kNumMouthCoef] = {1};
  float expressions[direct::kNumExpressions] = {};
  std::vector<std::vector<float>> deltas;
  // Reuse the same head to catch sex-dependent state leaking through a cache,
  // including when the old model maps both sexes onto Mark.
  for (int mask = 0; mask < 4; ++mask) {
    config.expressive_faces[0] = mask & 1;
    config.expressive_faces[1] = mask & 2;
    for (bool female : {false, true}) {
      for (float value : {0.f, 0.5f, 1.f, 2.f}) {
        Tune tune;
        tune.direct_scale[direct::kMouth] = value;
        float scales[direct::kNumRegions];
        SpeechScales(head, tune, female, scales);
        const float gain = (mask & (female ? 2 : 1)) ? 0.85f : 1.f;
        assert(std::fabs(scales[direct::kMouth] - value * 0.5f * gain) < 1e-6f);
        for (int r = 0; r < direct::kNumRegions; ++r)
          if (r != direct::kMouth) assert(scales[r] == 1);
        assert(direct::Evaluate(head, coefficients, scales, expressions,
                                nullptr, direct::TongueMode::kAuthored,
                                deltas));
        assert(std::fabs(deltas[0][2] - value * 0.5f * gain) < 1e-6f);
      }
    }
  }
}
}  // namespace

extern "C" uint64_t __wrap__ZN4skse4AddrEy(uint64_t id) {
  if (id == kIdStreamCtor.Get()) return reinterpret_cast<uint64_t>(&Open);
  if (id == kIdStreamDtor.Get()) return reinterpret_cast<uint64_t>(&Close);
  return 0;
}

int main(int argc, char** argv) {
  assert(argc == 1 || argc == 3);
  std::vector<uint8_t> originals[2];
  if (argc == 3) {
    for (int i = 0; i < 2; ++i) {
      std::ifstream f(argv[i + 1], std::ios::binary);
      if (!f) {
        fprintf(stderr, "Missing fixture: %s\n", argv[i + 1]);
        return 2;
      }
      originals[i].assign(std::istreambuf_iterator<char>(f), {});
      assert(bfa::IsExpressiveHead(originals[i], i != 0));
    }
  }
  for (uint32_t runtime : {skse::Version(1, 5, 97), skse::Version(1, 6, 1170),
                           skse::Version(1, 4, 15)}) {
    skse::SetRuntime(runtime);
    files[0].clear();
    files[1].clear();
    Detect(false, false);
    // A filename or even the correct byte count must not establish EFA.
    for (int i = 0; i < 2; ++i)
      files[i].assign(bfa::kExpressiveHeads[i].size, 0);
    Detect(false, false);
    if (argc == 3) {
      files[0] = originals[0];
      files[1] = originals[1];
      Detect(true, true);
      files[1].clear();
      Detect(true, false);
      files[0].clear();
      files[1] = originals[1];
      Detect(false, true);
      files[0] = originals[0];
      files[0].back() ^= 1;
      Detect(false, true);
      files[0] = originals[1];
      files[1] = originals[0];
      Detect(false, false);
      files[0] = originals[0];
      files[1] = originals[1];
      short_read = true;
      Detect(false, false);
      short_read = false;
      files[0].clear();
      files[1].clear();
      Detect(false, false);
    }
    CheckScales();
  }
  puts(
      "EFA contracts passed: resource dispatch, missing/unknown rejection, "
      "per-sex scaling, shared head and no double calibration");
  if (argc == 3)
    puts(
        "Local EFA assets passed: both, male-only, female-only, modified, "
        "wrong-sex, partial-read and removed controls");
}
