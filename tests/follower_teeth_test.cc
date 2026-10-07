// Pinned local FaceGen/TRI captures: separate teeth must use the jaw mapping.
// No third-party geometry is compiled into this test or distributed with it.
#include "direct.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

void Check(bool ok, const std::string& message) {
  if (ok) return;
  std::cerr << message << '\n';
  std::exit(1);
}

uint32_t ReadUint(std::ifstream& input) {
  unsigned char bytes[4];
  Check(static_cast<bool>(input.read(reinterpret_cast<char*>(bytes), 4)),
        "truncated fixture integer");
  return uint32_t{bytes[0]} | (uint32_t{bytes[1]} << 8) |
         (uint32_t{bytes[2]} << 16) | (uint32_t{bytes[3]} << 24);
}

std::string ReadString(std::ifstream& input) {
  const uint32_t size = ReadUint(input);
  Check(size > 0 && size <= 1024, "invalid fixture name length");
  std::string value(size, '\0');
  Check(static_cast<bool>(input.read(value.data(), size)), "truncated name");
  return value;
}

std::vector<float> ReadVertices(std::ifstream& input, uint32_t count) {
  std::vector<float> values(count * 3);
  for (float& value : values) {
    const uint32_t bits = ReadUint(input);
    static_assert(sizeof(value) == sizeof(bits));
    std::memcpy(&value, &bits, sizeof(value));
    Check(std::isfinite(value), "non-finite fixture value");
  }
  return values;
}

std::vector<direct::ShapeIn> ReadFixture(const char* path) {
  std::ifstream input(path, std::ios::binary);
  char magic[8];
  Check(static_cast<bool>(input.read(magic, sizeof(magic))) &&
            std::memcmp(magic, "BFATEETH", sizeof(magic)) == 0 &&
            ReadUint(input) == 1,
        "invalid fixture header");
  const uint32_t count = ReadUint(input);
  Check(count > 0 && count <= 64, "invalid shape count");
  std::vector<direct::ShapeIn> shapes(count);
  for (auto& shape : shapes) {
    shape.name = ReadString(input);
    shape.nv = ReadUint(input);
    Check(shape.nv > 0 && shape.nv <= 65535, "invalid vertex count");
    const uint32_t part = ReadUint(input);
    Check(part <= static_cast<uint32_t>(direct::Part::kNone), "invalid part");
    shape.part = static_cast<direct::Part>(part);
    shape.skin = ReadUint(input) != 0;
    shape.mouth_candidate = ReadUint(input) != 0;
    shape.verts = ReadVertices(input, shape.nv);
    Check(ReadUint(input) == direct::kNumMorphs,
          "fixture needs regeneration for current morph inventory");
    shape.morphs.resize(direct::kNumMorphs);
    for (int m = 0; m < direct::kNumMorphs; ++m) {
      Check(ReadString(input) == direct::kMorphIds[m].name,
            "fixture morph order differs from production");
      auto values = ReadVertices(input, shape.nv);
      if (direct::Reads(shape.skin, shape.part, m, shape.mouth_candidate))
        shape.morphs[m] = std::move(values);
    }
  }
  Check(input.peek() == std::char_traits<char>::eof(), "trailing fixture data");
  return shapes;
}

float Peak(const std::vector<float>& values) {
  float peak = 0;
  for (float value : values) {
    Check(std::isfinite(value), "non-finite deformation");
    peak = std::max(peak, std::fabs(value));
  }
  return peak;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: follower_teeth_test fixture.bin A2FDirect.bin tooth-regions.bin\n";
    return 2;
  }
  auto shapes = ReadFixture(argv[1]);
  size_t tooth_index = shapes.size();
  for (size_t i = 0; i < shapes.size(); ++i) {
    if (shapes[i].name.find("Female_Teeth_") != 0) continue;
    Check(tooth_index == shapes.size(), "ambiguous teeth shape");
    tooth_index = i;
  }
  Check(tooth_index < shapes.size(), "missing separate teeth");
  const auto& teeth = shapes[tooth_index];
  Check(teeth.nv == 4468 && teeth.mouth_candidate &&
            teeth.part == direct::Part::kPinned,
        "unexpected follower tooth topology or classification input");
  // Independent anatomical labels from inspected TRI connected components.
  // BigAah-stationary vertices include the tongue root, not just upper teeth.
  std::ifstream regions_file(argv[3], std::ios::binary);
  std::vector<uint8_t> regions(teeth.nv);
  Check(static_cast<bool>(regions_file.read(
            reinterpret_cast<char*>(regions.data()), regions.size())) &&
            regions_file.peek() == std::char_traits<char>::eof(),
        "invalid anatomical label fixture");
  size_t region_counts[3] = {};
  for (uint8_t region : regions) {
    Check(region < 3, "unknown anatomical region");
    ++region_counts[region];
  }
  Check(region_counts[0] == 1784 && region_counts[1] == 1945 &&
            region_counts[2] == 739,
        "upper/lower dentition and tongue topology changed");
  direct::Model model;
  std::string why;
  const std::string path = argv[2];
  Check(model.Load(std::wstring(path.begin(), path.end()), why), why);
  direct::Head head, oracle;
  Check(direct::BuildHead(model.Face(true), shapes, head, why), why);
  Check(head.shapes.size() == shapes.size() &&
            head.shapes[tooth_index].name == teeth.name &&
            head.shapes[tooth_index].part == direct::Part::kMouth,
        "separate teeth were not promoted to jaw mapping");
  shapes[tooth_index].mouth_candidate = false;
  shapes[tooth_index].part = direct::Part::kMouth;
  Check(direct::BuildHead(model.Face(true), shapes, oracle, why), why);

  std::vector<float> coefficients(head.k + direct::kNumMouthCoef);
  float scales[direct::kNumRegions] = {1, 1, 1, 1, 1, 1};
  float expressions[direct::kNumExpressions] = {};
  float phonemes[direct::kNumPhonemes] = {};
  std::vector<std::vector<float>> deltas, reference;
  bool passed = true;
  for (auto mode : {direct::TongueMode::kAuthored, direct::TongueMode::kModel}) {
    float articulation_peak = 0;
    // All skin, model tongue/jaw, expression and authored phoneme basis cues.
    const size_t count = coefficients.size() + direct::kNumExpressions +
                         direct::kNumPhonemes;
    for (size_t cue = 0; cue < count; ++cue) {
      std::fill(coefficients.begin(), coefficients.end(), 0);
      std::fill_n(expressions, direct::kNumExpressions, 0);
      std::fill_n(phonemes, direct::kNumPhonemes, 0);
      if (cue < coefficients.size())
        coefficients[cue] = 1;
      else if (cue < coefficients.size() + direct::kNumExpressions)
        expressions[cue - coefficients.size()] = 1;
      else
        phonemes[cue - coefficients.size() - direct::kNumExpressions] = 1;
      Check(direct::Evaluate(head, coefficients.data(), scales, expressions,
                             phonemes, mode, deltas) &&
                direct::Evaluate(oracle, coefficients.data(), scales,
                                 expressions, phonemes, mode, reference),
            "basis evaluation failed");
      Check(deltas == reference, "candidate differs from explicit-mouth control");
      for (const auto& shape : deltas) Peak(shape);
      if (cue >= coefficients.size() + direct::kNumExpressions) {
        for (uint32_t v = 0; v < teeth.nv; ++v)
          if (regions[v] == 2)
            for (int axis = 0; axis < 3; ++axis)
              articulation_peak = std::max(
                  articulation_peak, std::fabs(deltas[tooth_index][v * 3 + axis]));
      }
      if (cue < head.k)
        Check(Peak(deltas[tooth_index]) < 1e-5f,
              "teeth incorrectly follow skin coefficients");
      // Switching tongue modes must not alter actual dental vertices, even
      // though the same mesh also contains a tongue. This catches accidental
      // tongue masks on both upper and lower teeth across every basis cue.
      std::vector<std::vector<float>> other_mode;
      Check(direct::Evaluate(head, coefficients.data(), scales, expressions,
                             phonemes,
                             mode == direct::TongueMode::kAuthored
                                 ? direct::TongueMode::kModel
                                 : direct::TongueMode::kAuthored,
                             other_mode), "tongue mode control failed");
      for (uint32_t v = 0; v < teeth.nv; ++v) {
        if (regions[v] == 2) continue;
        for (int axis = 0; axis < 3; ++axis)
          Check(std::fabs(deltas[tooth_index][v * 3 + axis] -
                          other_mode[tooth_index][v * 3 + axis]) < 1e-5f,
                "tongue mode deforms dental anatomy");
      }
    }
    if (mode == direct::TongueMode::kAuthored)
      Check(articulation_peak > .01f, "embedded tongue lost authored articulation");
    std::fill(coefficients.begin(), coefficients.end(), 0);
    std::fill_n(expressions, direct::kNumExpressions, 0);
    std::fill_n(phonemes, direct::kNumPhonemes, 0);
    for (int i = 0; i < direct::kNumJawPoints; ++i)
      coefficients[head.k + a2f::kNumTongue + i * 3 + 1] = -.25f;
    Check(direct::Evaluate(head, coefficients.data(), scales, expressions,
                           phonemes, mode, deltas), "jaw evaluation failed");
    float lower_peak = 0, upper_peak = 0, tongue_peak = 0;
    float lower_min = INFINITY;
    for (uint32_t v = 0; v < teeth.nv; ++v) {
      float delta = 0;
      for (int axis = 0; axis < 3; ++axis)
        delta = std::max(delta, std::fabs(deltas[tooth_index][v * 3 + axis]));
      if (regions[v] == 0) {
        upper_peak = std::max(upper_peak, delta);
      } else if (regions[v] == 1) {
        lower_peak = std::max(lower_peak, delta);
        lower_min = std::min(lower_min, delta);
      } else {
        tongue_peak = std::max(tongue_peak, delta);
      }
    }
    std::cout << teeth.name << " mode=" << static_cast<int>(mode)
              << " upper_peak=" << upper_peak << " lower_peak=" << lower_peak
              << " lower_min=" << lower_min << " tongue_peak=" << tongue_peak
              << '\n';
    if (upper_peak >= 1e-4f || lower_min <= .01f || tongue_peak <= .01f) {
      std::cerr << "upper teeth move or lower teeth/tongue fail to follow jaw\n";
      passed = false;
    }
    const auto jaw = deltas[tooth_index];
    for (float strength : {0.f, 3.f}) {
      Check(direct::Evaluate(head, coefficients.data(), scales, expressions,
                             phonemes, mode, deltas, strength, strength),
            "lip strength evaluation failed");
      Check(deltas[tooth_index] == jaw, "lip sliders deform the separate teeth");
    }
  }
  std::cout << teeth.name << ": model " << model.network
            << ", both tongue modes, upper/lower teeth and lip controls "
            << (passed ? "pass" : "FAIL") << '\n';
  return passed ? 0 : 1;
}
