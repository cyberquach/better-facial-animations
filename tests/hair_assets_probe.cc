// Replay pinned installed HDPT registries through production face traversal.
// Reuse the engine mocks; the synthetic contract remains a separate executable.
#define main SyntheticHeadPartsMain
#include "head_parts_test.cc"
#undef main

#include <fstream>
#include <stdexcept>

namespace {
uint32_t ReadU32(std::istream& in) {
  uint32_t value;
  if (!in.read(reinterpret_cast<char*>(&value), sizeof(value)))
    throw std::runtime_error("short fixture input");
  return value;
}
std::string ReadString(std::istream& in) {
  std::string value(ReadU32(in), '\0');
  if (!in.read(value.data(), value.size()))
    throw std::runtime_error("short fixture string");
  return value;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) return 2;
  try {
    std::ifstream in(argv[1], std::ios::binary);
    std::ofstream out(argv[2], std::ios::binary);
    using namespace bfa;
    const uint32_t scenarios = ReadU32(in);
    for (uint32_t sc = 0; sc < scenarios; ++sc) {
      const uint32_t n = ReadU32(in);
      std::vector<std::array<uint8_t, 0x120>> forms(n);
      std::vector<std::string> names(n);
      std::vector<uint8_t*> ptrs(n);
      std::vector<std::vector<uint8_t*>> extras(n);
      for (uint32_t i = 0; i < n; ++i) ptrs[i] = forms[i].data();
      for (uint32_t i = 0; i < n; ++i) {
        names[i] = ReadString(in);
        const uint32_t type = ReadU32(in), count = ReadU32(in);
        for (uint32_t j = 0; j < count; ++j) {
          const uint32_t index = ReadU32(in);
          if (index >= n) throw std::runtime_error("invalid extra-part index");
          extras[i].push_back(ptrs[index]);
        }
        auto* p = ptrs[i];
        p[kFormType] = kFtHeadPart;
        Field<uint32_t>(p, kHeadPartType) = type;
        Field<const char*>(p, kHeadPartName) = names[i].c_str();
        Field<BSTArray>(p, kHeadPartExtras) = {extras[i].data(),
                                               uint32_t(extras[i].size()), 0,
                                               uint32_t(extras[i].size())};
      }
      Field<BSTArray>(handler,
                      kDhFormArrays + kFtHeadPart * sizeof(BSTArray)) = {
          ptrs.data(), n, 0, n};
      LoadHeadPartTypes();
      std::array<uint8_t, 0x1000> node{}, shape{}, fod{}, shader{};
      uint8_t* children[] = {shape.data()};
      uint8_t* extra[] = {fod.data()};
      float vertices[] = {1, 2, 3, 4, 5, 6};
      Field<uint8_t**>(node.data(), kNodeChildren.Get()) = children;
      Field<uint16_t>(node.data(), kNodeNumChildren.Get()) = 1;
      shape[kGeomType.Get()] = 4;
      Field<uint16_t>(shape.data(), kDynNumVerts.Get()) = 2;
      Field<uint32_t>(shape.data(), kDynDataSize.Get()) = 32;
      Field<uint8_t**>(shape.data(), kObjExtra) = extra;
      Field<uint16_t>(shape.data(), kObjNumExtra) = 1;
      // Independent AE shader layout; SE/AE/VR are covered by head-parts.
      Field<uint8_t*>(shape.data(), 0x128) = shader.data();
      Field<const char*>(fod.data(), kObjName) = "FOD";
      Field<uint32_t>(fod.data(), kBaseNumVerts) = 2;
      Field<const float*>(fod.data(), kBaseVerts) = vertices;
      const uint32_t queries = ReadU32(in);
      for (uint32_t i = 0; i < queries; ++i) {
        const std::string name = ReadString(in);
        uint64_t shader_flags;
        if (!in.read(reinterpret_cast<char*>(&shader_flags),
                     sizeof(shader_flags)))
          throw std::runtime_error("short shader flags");
        Field<uint64_t>(shader.data(), 0x38) = shader_flags;
        Field<const char*>(shape.data(), kObjName) = name.c_str();
        bool accepted = false;
        ForFaceShapes(node.data(),
                      [&](uint8_t*, const char*, direct::Part, bool, uint32_t,
                          const float*) { accepted = true; });
        bool skin = false;
        const int32_t part = static_cast<int32_t>(Classify(name.c_str(), skin));
        const uint8_t flags = uint8_t(skin) |
                              (uint8_t(MouthCandidate(name.c_str())) << 1) |
                              (uint8_t(accepted) << 2);
        out.write(reinterpret_cast<const char*>(&part), sizeof(part));
        out.put(flags);
      }
    }
    if (in.peek() != std::char_traits<char>::eof())
      throw std::runtime_error("trailing fixture input");
    return out ? 0 : 1;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
