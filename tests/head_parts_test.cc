// Exercise the production registry/classifier using synthetic engine HDPTs.
#include <algorithm>
#include <array>
#include <cmath>
#include <cassert>
#include <cctype>
#include <cstdio>
#include <vector>

#include "../src/head_cache.h"
namespace {
std::array<uint8_t, 0x1000> storage{};
uint8_t* handler = storage.data();
std::array<float, 8> dynamic_vertices{1, 2, 3, 99, 4, 5, 6, 98};
bool has_morphs = true;
int apply_count = 0;
float* LockDynamic(uint8_t*) { return dynamic_vertices.data(); }
void UnlockDynamic(uint8_t*) {}
void* MorphData(uint8_t*) { return has_morphs ? storage.data() : nullptr; }
void ApplyMorph(void*, int type, int index, uint8_t*, float weight) {
  ++apply_count;
  for (int i : {0, 1, 2, 4, 5, 6})
    dynamic_vertices[i] += weight * (type * 100 + index + 1);
}
}  // namespace
namespace skse {
bool IsSe() { return false; }
bool IsVr() { return false; }
uint64_t Addr(uint64_t id) {
  if (id == kIdDataHandler.ae) return reinterpret_cast<uint64_t>(&handler);
  if (id == kIdLockDynamic.ae) return reinterpret_cast<uint64_t>(&LockDynamic);
  if (id == kIdUnlockDynamic.ae) return reinterpret_cast<uint64_t>(&UnlockDynamic);
  if (id == kIdMorphData.ae) return reinterpret_cast<uint64_t>(&MorphData);
  if (id == kIdApplyMorph.ae) return reinterpret_cast<uint64_t>(&ApplyMorph);
  return 0;
}
void Log(const char*, ...) {}
}  // namespace skse
namespace bfa {
direct::Model instance;
direct::Model& model = instance;
double Now() { return 0; }
std::string Lower(const char* s) {
  std::string out(s);
  for (char& c : out) c = std::tolower(static_cast<unsigned char>(c));
  return out;
}
}  // namespace bfa
// Include the capture implementation so engine calls can be controlled while
// exercising its real candidate-morph selection and vertex restoration.
#include "../src/head_cache.cc"

int main() {
  using namespace bfa;
  struct Part {
    const char* name;
    uint32_t type;
  } specs[] = {{"InigoHead", kHpFace},
               {"InigoHair", kHpHair},
               {"InigoEP", kHpMisc},
               {"PAHP-Septum-Ring-HP", 434},
               {"PAHP-Left-Snakebite-Ring-HP", kHpMisc},
               {"PAHP-Left-Brow-Barbell-HP", 420},
               {"CustomSlotHeadDecoration", 257},
               {"UnknownSlotSeven", 7},
               {"InigoEyes", kHpEyes},
               {"MaleMouthKhajiit", kHpMisc},
               {"KhajiitBeard01Alpha", kHpMisc},
               {"InigoBeard", kHpFacialHair},
               {"CustomLashes", kHpMisc},
               {"Female_Mouth_Lily", kHpMisc},
               {"Female_Teeth_Lily", kHpMisc},
               {"Female_Teeth_Alma", kHpMisc},
               {"Female_Teeth_DVA", kHpMisc},
               {"Female_Teeth_Eve", kHpMisc},
               {"Female_Teeth_Squirrel_Girl", kHpMisc},
               {"OralChild", kHpMisc},
               {"NestedChild", kHpMisc},
               {"Tooth_Piercing", kHpMisc},
               {"Teeth_Eyes", kHpEyes},
               {"Teeth_Hair", kHpHair},
               {"Teeth_CustomSlot", 434}};
  std::vector<std::array<uint8_t, 0x120>> forms(std::size(specs));
  std::vector<uint8_t*> ptrs;
  for (size_t i = 0; i < forms.size(); ++i) {
    auto* p = forms[i].data();
    p[kFormType] = kFtHeadPart;
    Field<uint32_t>(p, kHeadPartType) = specs[i].type;
    Field<const char*>(p, kHeadPartName) = specs[i].name;
    ptrs.push_back(p);
  }
  uint8_t* hair_extra[] = {ptrs[2]};
  Field<BSTArray>(ptrs[1], kHeadPartExtras) = {hair_extra, 1, 0, 1};
  uint8_t* beard_extra[] = {ptrs[10]};
  Field<BSTArray>(ptrs[11], kHeadPartExtras) = {beard_extra, 1, 0, 1};
  Field<BSTArray>(handler, kDhFormArrays + kFtHeadPart * sizeof(BSTArray)) = {
      ptrs.data(), uint32_t(ptrs.size()), 0, uint32_t(ptrs.size())};
  uint8_t* mouth_extra[] = {ptrs[14], ptrs[19], ptrs[21], ptrs[22], ptrs[23], ptrs[24]};
  uint8_t* nested_extra[] = {ptrs[20]};
  uint8_t* cyclic_extra[] = {ptrs[19]};
  Field<BSTArray>(ptrs[13], kHeadPartExtras) = {mouth_extra, 6, 0, 6};
  Field<BSTArray>(ptrs[19], kHeadPartExtras) = {nested_extra, 1, 0, 1};
  Field<BSTArray>(ptrs[20], kHeadPartExtras) = {cyclic_extra, 1, 0, 1};
  LoadHeadPartTypes();
  auto expect = [](const char* n, direct::Part part, bool is_skin) {
    bool skin = true;
    assert(Classify(n, skin) == part);
    assert(skin == is_skin);
  };
  expect("InigoHead", direct::Part::kHead, true);
  expect("InigoHair", direct::Part::kNone, false);
  // accessories are rigid pieces, also one named like a soft part (brow)
  for (auto n : {"InigoEP", "PAHP-Septum-Ring-HP",
                 "PAHP-Left-Snakebite-Ring-HP", "PAHP-Left-Brow-Barbell-HP",
                 "CustomSlotHeadDecoration", "UnknownSlotSeven"})
    expect(n, direct::Part::kPinned, false);
  expect("KhajiitBeard01Alpha", direct::Part::kHead, false);
  expect("CustomLashes", direct::Part::kHead, false);
  expect("InigoEyes", direct::Part::kEyes, false);
  expect("MaleMouthKhajiit", direct::Part::kMouth, false);
  expect("UnregisteredDecoration", direct::Part::kNone, false);
  expect("FemaleHeadNord", direct::Part::kHead, true);
  for (const char* name : {"Female_Teeth_Lily", "Female_Teeth_Alma",
                            "Female_Teeth_DVA", "Female_Teeth_Eve",
                            "Female_Teeth_Squirrel_Girl", "OralChild", "NestedChild",
                            "UnregisteredTEETH", "UpperTooth", "LowerGums",
                            "CustomDentition", "SeparateTongue"}) {
    assert(MouthCandidate(name));
  }
  for (const char* name : {"Tooth_Piercing", "MouthRing", "Teeth_Eyes",
                            "Teeth_Hair", "Teeth_CustomSlot", "InigoEP",
                            "UnregisteredDecoration", "FemaleHeadNord"}) {
    assert(!MouthCandidate(name));
  }
  // Candidate status does not prematurely change the accessory fallback.
  expect("Female_Teeth_Lily", direct::Part::kPinned, false);
  expect("Tooth_Piercing", direct::Part::kPinned, false);
  expect("MouthRing", direct::Part::kNone, false);
  expect("UnregisteredTEETH", direct::Part::kNone, false);
  for (const char* name : {"Female_Teeth_Lily", "UnregisteredTEETH"}) {
    direct::ShapeIn captured;
    captured.part = Classify(name, captured.skin);
    captured.mouth_candidate = MouthCandidate(name);
    captured.nv = 2;
    const auto before = dynamic_vertices;
    ReadMorphs(nullptr, captured);
    assert(dynamic_vertices == before);
    for (int m = 0; m < direct::kNumMorphs; ++m) {
      if (!direct::Reads(false, direct::Part::kMouth, m)) {
        assert(captured.morphs[m].empty());
        continue;
      }
      assert(captured.morphs[m].size() == 6);
      const auto& id = direct::kMorphIds[m];
      for (float value : captured.morphs[m])
        assert(value == id.type * 100 + id.index + 1);
    }
  }
  direct::ShapeIn no_morphs;
  no_morphs.nv = 2;
  no_morphs.mouth_candidate = true;
  has_morphs = false;
  const int previous_count = apply_count;
  ReadMorphs(nullptr, no_morphs);
  assert(apply_count == previous_count);
  for (const auto& morph : no_morphs.morphs) assert(morph.empty());

  // The shared capture/playback traversal must admit unregistered candidates
  // without admitting unrelated geometry.
  std::array<uint8_t, 0x1000> node{}, fod{};
  std::array<std::array<uint8_t, 0x1000>, 3> geometry{};
  uint8_t* children[] = {geometry[0].data(), geometry[1].data(), geometry[2].data()};
  uint8_t* extras[] = {fod.data()};
  const char* names[] = {"UnregisteredTEETH", "UnregisteredDecoration", "MouthRing"};
  float vertices[] = {1, 2, 3, 4, 5, 6};
  Field<uint8_t**>(node.data(), kNodeChildren.Get()) = children;
  Field<uint16_t>(node.data(), kNodeNumChildren.Get()) = 3;
  Field<const char*>(fod.data(), kObjName) = "FOD";
  Field<uint32_t>(fod.data(), kBaseNumVerts) = 2;
  Field<const float*>(fod.data(), kBaseVerts) = vertices;
  for (int i = 0; i < 3; ++i) {
    auto* shape = children[i];
    shape[kGeomType.Get()] = 4;
    Field<const char*>(shape, kObjName) = names[i];
    Field<uint16_t>(shape, kDynNumVerts.Get()) = 2;
    Field<uint32_t>(shape, kDynDataSize.Get()) = 32;
    Field<uint8_t**>(shape, kObjExtra) = extras;
    Field<uint16_t>(shape, kObjNumExtra) = 1;
  }
  int seen = 0;
  ForFaceShapes(node.data(), [&](uint8_t*, const char* name, direct::Part part,
                                bool skin, uint32_t nv, const float* base) {
    ++seen;
    assert(std::string(name) == "UnregisteredTEETH");
    assert(part == direct::Part::kNone && !skin && nv == 2 && base == vertices);
  });
  assert(seen == 1);
  Field<const char*>(geometry[1].data(), kObjName) = "OtherTeeth";
  const std::string key = HeadKey(node.data(), false);
  std::swap(children[0], children[1]);
  assert(HeadKey(node.data(), false) == key);
  dynamic_vertices[0] += 1;
  assert(HeadKey(node.data(), false) == key);
  vertices[0] = std::nextafter(vertices[0], 2.f);
  assert(HeadKey(node.data(), false) != key);
  vertices[0] = 1;
  has_morphs = true;
  assert(HeadKey(node.data(), false) != key);
  has_morphs = false;
  Field<uint16_t>(node.data(), kNodeNumChildren.Get()) = 1;
  assert(HeadKey(node.data(), false) != key);
  Field<uint16_t>(node.data(), kNodeNumChildren.Get()) = 3;
  Field<const char*>(geometry[1].data(), kObjName) = "RenamedTeeth";
  assert(HeadKey(node.data(), false) != key);
  Field<BSTArray>(ptrs[13], kHeadPartExtras) = {};
  LoadHeadPartTypes();
  assert(!MouthCandidate("OralChild"));
  assert(!MouthCandidate("NestedChild"));
  assert(MouthCandidate("Female_Teeth_Lily"));
  puts(
      "head-part classification passed: custom 32-bit slots, Misc, hair "
      "extras, vanilla and unknown controls");
}
