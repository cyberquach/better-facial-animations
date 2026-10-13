// Exercise the production registry/classifier using synthetic engine HDPTs.
#include <algorithm>
#include <array>
#include <cmath>
#include <cassert>
#include <cctype>
#include <cstdio>
#include <vector>

#include "a2f/animation/head_cache.h"
namespace {
std::array<uint8_t, 0x1000> storage{};
uint8_t* handler = storage.data();
std::array<float, 8> dynamic_vertices{1, 2, 3, 99, 4, 5, 6, 98};
bool has_morphs = true;
bool test_se = false, test_vr = false;
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
bool IsSe() { return test_se; }
bool IsVr() { return test_vr; }
uint64_t Addr(uint64_t id) {
  if (id == kIdDataHandler.ae || id == kIdDataHandler.se ||
      id == kIdDataHandler.vr)
    return reinterpret_cast<uint64_t>(&handler);
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
std::shared_ptr<const ModelState> SnapshotModel() {
  return std::make_shared<const ModelState>(
      ModelState{3, true, L"",
                 std::shared_ptr<const direct::Model>(
                     &model, [](const direct::Model*) {})});
}
double Now() { return 0; }
std::string Lower(const char* s) {
  std::string out(s);
  for (char& c : out) c = std::tolower(static_cast<unsigned char>(c));
  return out;
}
}  // namespace bfa
// Include the capture implementation so engine calls can be controlled while
// exercising its real candidate-morph selection and vertex restoration.
#include "../src/animation/head_cache.cc"

// Verify exclusion at the shared capture/playback boundary, not just the
// classifier: mouth candidates must not reopen the path for excluded extras.
void HairExclusion() {
  using namespace bfa;
  struct Spec {
    const char* name;
    uint32_t type;
    bool excluded;
  } specs[] = {{"Style01", kHpHair, true},
               {"Underlayer", kHpMisc, true},
               {"HeadBrowOverlay", kHpMisc, true},
               {"MouthTeethOverlay", kHpMisc, true},
               {"HairOwnedRing", 434, true},
               {"", kHpMisc, true},
               {"DeepOverlay", kHpMisc, true},
               {"BeardParent", kHpFacialHair, false},
               {"MouthParent", kHpMisc, false},
               {"IndependentRing", 434, false},
               {"FacialHairBeard", kHpFacialHair, false},
               {"HairyBrows", kHpEyebrows, false},
               {"CustomLashes", kHpMisc, false},
               {"CustomHeadHairline", kHpMisc, true},
               {"ScalpTeeth", 434, true},
               {"FallbackChild", kHpMisc, true},
               {"VirtualBody", kHpMisc, true},
               {"IndependentEyes", kHpEyes, false},
               {"TintedFaceNord", kHpFace, false},
               {"TintedScar", kHpScar, false},
               {"FacialHair", kHpMisc, false},
               {"Facial_Hair", kHpMisc, false},
               {"Facial Hair", 434, false},
               {"HairOwnedFacialHair", kHpMisc, true}};
  std::vector<std::array<uint8_t, 0x120>> forms(std::size(specs));
  std::vector<uint8_t*> ptrs;
  for (size_t i = 0; i < forms.size(); ++i) {
    auto* p = forms[i].data();
    p[kFormType] = kFtHeadPart;
    Field<uint32_t>(p, kHeadPartType) = specs[i].type;
    Field<const char*>(p, kHeadPartName) = specs[i].name;
    ptrs.push_back(p);
  }
  uint8_t* root[] = {ptrs[1], ptrs[4], ptrs[5], nullptr, ptrs[23]};
  uint8_t* nested[] = {ptrs[2], ptrs[3]};
  uint8_t* cycle[] = {ptrs[1]};
  uint8_t* unnamed[] = {ptrs[6]};
  uint8_t* shared[] = {ptrs[2], ptrs[3]};
  uint8_t* fallback[] = {ptrs[15]};
  uint8_t* eye_lashes[] = {ptrs[12]};
  Field<BSTArray>(ptrs[0], kHeadPartExtras) = {root, 5, 0, 5};
  Field<BSTArray>(ptrs[1], kHeadPartExtras) = {nested, 2, 0, 2};
  Field<BSTArray>(ptrs[2], kHeadPartExtras) = {cycle, 1, 0, 1};
  Field<BSTArray>(ptrs[5], kHeadPartExtras) = {unnamed, 1, 0, 1};
  Field<BSTArray>(ptrs[7], kHeadPartExtras) = {shared, 2, 0, 2};
  Field<BSTArray>(ptrs[8], kHeadPartExtras) = {shared, 2, 0, 2};
  Field<BSTArray>(ptrs[13], kHeadPartExtras) = {fallback, 1, 0, 1};
  Field<BSTArray>(ptrs[17], kHeadPartExtras) = {eye_lashes, 1, 0, 1};
  Field<BSTArray>(handler, kDhFormArrays + kFtHeadPart * sizeof(BSTArray)) = {
      ptrs.data(), uint32_t(ptrs.size()), 0, uint32_t(ptrs.size())};

  std::array<uint8_t, 0x1000> node{}, shape{}, fod{};
  uint8_t* children[] = {shape.data()};
  uint8_t* extras[] = {fod.data()};
  float vertices[] = {1, 2, 3, 4, 5, 6};
  Field<uint8_t**>(node.data(), kNodeChildren.Get()) = children;
  Field<uint16_t>(node.data(), kNodeNumChildren.Get()) = 1;
  shape[kGeomType.Get()] = 4;
  Field<uint16_t>(shape.data(), kDynNumVerts.Get()) = 2;
  Field<uint32_t>(shape.data(), kDynDataSize.Get()) = 32;
  Field<uint8_t**>(shape.data(), kObjExtra) = extras;
  Field<uint16_t>(shape.data(), kObjNumExtra) = 1;
  Field<const char*>(fod.data(), kObjName) = "FOD";
  Field<uint32_t>(fod.data(), kBaseNumVerts) = 2;
  Field<const float*>(fod.data(), kBaseVerts) = vertices;
  auto check = [&](const char* name, bool excluded) {
    bool skin = true;
    const auto part = Classify(name, skin);
    if (excluded) {
      assert(part == direct::Part::kNone && !skin);
      assert(!MouthCandidate(name));
    }
    Field<const char*>(shape.data(), kObjName) = name;
    int seen = 0;
    ForFaceShapes(node.data(), [&](uint8_t*, const char*, direct::Part, bool,
                                   uint32_t, const float*) { ++seen; });
    assert(seen == (excluded ? 0 : 1));
  };
  for (int order = 0; order < 2; ++order) {
    LoadHeadPartTypes();
    for (const auto& spec : specs)
      if (*spec.name) check(spec.name, spec.excluded);
    check("hEaDbRoWoVeRlAy", true);
    check("UnregisteredHeadHairline", true);
    check("UnregisteredHairMouth", true);
    check("UnregisteredScalpHead", true);
    check("UnregisteredBeardHair", false);
    std::reverse(ptrs.begin(), ptrs.end());
  }
  // Independent CommonLib layout fixtures for SE, AE and VR. Hair tint is
  // also used by legitimate beards/brows, but never authorizes scalp hair.
  for (int runtime = 0; runtime < 3; ++runtime) {
    test_se = runtime == 0;
    test_vr = runtime == 2;
    const uint32_t shader_offset = test_vr ? 0x168 : 0x128;
    node.fill(0);
    shape.fill(0);
    Field<uint8_t**>(node.data(), kNodeChildren.Get()) = children;
    Field<uint16_t>(node.data(), kNodeNumChildren.Get()) = 1;
    shape[kGeomType.Get()] = 4;
    Field<uint16_t>(shape.data(), kDynNumVerts.Get()) = 2;
    Field<uint32_t>(shape.data(), kDynDataSize.Get()) = 32;
    Field<uint8_t**>(shape.data(), kObjExtra) = extras;
    Field<uint16_t>(shape.data(), kObjNumExtra) = 1;
    std::array<uint8_t, 0x100> shader{};
    Field<uint8_t*>(shape.data(), shader_offset) = shader.data();
    Field<uint64_t>(shader.data(), 0x38) = uint64_t{1} << 18;
    auto admitted = [&](const char* name) {
      Field<const char*>(shape.data(), kObjName) = name;
      int seen = 0;
      ForFaceShapes(node.data(), [&](uint8_t*, const char*, direct::Part, bool,
                                     uint32_t, const float*) { ++seen; });
      return seen != 0;
    };
    // Actual KS Hairdos shape alias; "scar" used to admit it as a scar.
    assert(!admitted("Scarlet"));
    assert(!admitted("VirtualBody"));
    assert(!admitted("UnknownHeadOverlay"));
    assert(!admitted("UnknownTeethOverlay"));
    assert(!admitted("HeadBrowOverlay"));  // hair-owned: even a brow name loses
    assert(admitted("MaleBeards002"));     // actual Lucien beard model alias
    assert(admitted("FacialHairBeard"));
    assert(admitted("FacialHair"));
    assert(admitted("Facial_Hair"));
    assert(admitted("Facial Hair"));
    assert(!admitted("HairOwnedFacialHair"));
    assert(admitted("HairyBrows"));
    assert(admitted("CustomLashes"));
    // Registered face, scar and eye parts keep their controls with the flag;
    // registered hair and hair-owned extras never do.
    assert(admitted("TintedFaceNord"));
    assert(admitted("TintedScar"));
    assert(admitted("IndependentEyes"));
    assert(!admitted("Style01"));
    assert(!admitted("Underlayer"));
    // No shader or a different flag must not disable ordinary face parts.
    Field<uint64_t>(shader.data(), 0x38) = uint64_t{1} << 17;
    assert(admitted("UnknownHeadOverlay"));
    assert(admitted("UnknownTeethOverlay"));
    Field<uint8_t*>(shape.data(), shader_offset) = nullptr;
    assert(admitted("UnknownHeadOverlay"));
  }
  test_se = test_vr = false;
  node.fill(0);
  shape.fill(0);
  Field<uint8_t**>(node.data(), kNodeChildren.Get()) = children;
  Field<uint16_t>(node.data(), kNodeNumChildren.Get()) = 1;
  shape[kGeomType.Get()] = 4;
  Field<uint16_t>(shape.data(), kDynNumVerts.Get()) = 2;
  Field<uint32_t>(shape.data(), kDynDataSize.Get()) = 32;
  Field<uint8_t**>(shape.data(), kObjExtra) = extras;
  Field<uint16_t>(shape.data(), kObjNumExtra) = 1;
  // Registry reload must clear old ownership. A missing registry still uses
  // the conservative hair/scalp name guard before any head or mouth fallback.
  Field<BSTArray>(handler, kDhFormArrays + kFtHeadPart * sizeof(BSTArray)) = {};
  LoadHeadPartTypes();
  check("HeadBrowOverlay", false);
  check("UnregisteredHeadHairline", true);
  check("UnregisteredHairMouth", true);
  check("UnregisteredScalpHead", true);
  check("UnregisteredBeardHair", false);
  check("FemaleHeadNord", false);
}

// True 3D Mouth's Face -> Misc mouth graph, plus future oral names and
// nested/shared extras. Registry order and runtime must not turn oral parts
// into skin; explicit engine slots and hair exclusions retain precedence.
void OralInheritance() {
  using namespace bfa;
  struct Spec {
    const char* name;
    uint32_t type;
  } specs[] = {{"00KLHM_FemaleHeadNord", kHpFace},
               {"00DZBMself_FemaleMouth", kHpMisc},
               {"FutureUpperTeethHead", kHpMisc},
               {"FutureLowerTooth", kHpMisc},
               {"FutureTongue", kHpMisc},
               {"FutureGums", kHpMisc},
               {"FutureDentition", kHpMisc},
               {"OralHeadLayer", kHpMisc},
               {"NestedLayer", kHpMisc},
               {"MouthPiercing", kHpMisc},
               {"TeethEyes", kHpEyes},
               {"TeethCustom", 434},
               {"MouthHead", kHpFace},
               {"HairOwner", kHpHair},
               {"HairSharedTeeth", kHpMisc},
               {"HeadOverlay", kHpMisc},
               {"EyesOwner", kHpEyes},
               {"BeardOwner", kHpFacialHair},
               {"", kHpMisc},
               {"UnlabelledDescendant", kHpMisc}};
  for (int runtime = 0; runtime < 3; ++runtime) {
    test_se = runtime == 1;
    test_vr = runtime == 2;
    std::vector<std::array<uint8_t, 0x120>> forms(std::size(specs));
    std::vector<uint8_t*> ptrs;
    for (size_t i = 0; i < forms.size(); ++i) {
      auto* p = forms[i].data();
      p[kFormType] = kFtHeadPart;
      Field<uint32_t>(p, kHeadPartType) = specs[i].type;
      Field<const char*>(p, kHeadPartName) = specs[i].name;
      ptrs.push_back(p);
    }
    uint8_t* face[] = {ptrs[1], ptrs[2], ptrs[3],  ptrs[4], ptrs[5],
                       ptrs[6], ptrs[7], ptrs[14], ptrs[15]};
    uint8_t* oral[] = {ptrs[7],  ptrs[9],  ptrs[10], ptrs[11],
                       ptrs[12], ptrs[14], ptrs[18]};
    uint8_t* bridge[] = {ptrs[19]};
    uint8_t* nested[] = {ptrs[8]};
    uint8_t* cycle[] = {ptrs[7]};
    uint8_t* hair[] = {ptrs[14]};
    uint8_t* shared[] = {ptrs[2], ptrs[7]};
    Field<BSTArray>(ptrs[0], kHeadPartExtras) = {face, 9, 0, 9};
    Field<BSTArray>(ptrs[1], kHeadPartExtras) = {oral, 7, 0, 7};
    Field<BSTArray>(ptrs[18], kHeadPartExtras) = {bridge, 1, 0, 1};
    Field<BSTArray>(ptrs[7], kHeadPartExtras) = {nested, 1, 0, 1};
    Field<BSTArray>(ptrs[8], kHeadPartExtras) = {cycle, 1, 0, 1};
    Field<BSTArray>(ptrs[13], kHeadPartExtras) = {hair, 1, 0, 1};
    Field<BSTArray>(ptrs[16], kHeadPartExtras) = {shared, 2, 0, 2};
    Field<BSTArray>(ptrs[17], kHeadPartExtras) = {shared, 2, 0, 2};
    Field<BSTArray>(handler, kDhFormArrays + kFtHeadPart * sizeof(BSTArray)) = {
        ptrs.data(), uint32_t(ptrs.size()), 0, uint32_t(ptrs.size())};
    auto expect = [](const char* name, direct::Part part, bool skin,
                     bool oral) {
      bool actual_skin = !skin;
      assert(Classify(name, actual_skin) == part);
      assert(actual_skin == skin && MouthCandidate(name) == oral);
    };
    for (int order = 0; order < 2; ++order) {
      LoadHeadPartTypes();
      expect(specs[0].name, direct::Part::kHead, true, false);
      expect(specs[1].name, direct::Part::kMouth, false, true);
      for (int i = 2; i <= 8; ++i)
        expect(specs[i].name, direct::Part::kPinned, false, true);
      expect(specs[19].name, direct::Part::kPinned, false, true);
      expect(specs[9].name, direct::Part::kPinned, false, false);
      expect(specs[10].name, direct::Part::kEyes, false, false);
      expect(specs[11].name, direct::Part::kPinned, false, false);
      expect(specs[12].name, direct::Part::kHead, true, false);
      expect(specs[14].name, direct::Part::kNone, false, false);
      expect(specs[15].name, direct::Part::kHead, true, false);
      expect("UnregisteredTeethHead", direct::Part::kNone, false, true);
      std::reverse(ptrs.begin(), ptrs.end());
    }
    // Registry rebuild must discard inherited hints from the old graph.
    Field<BSTArray>(forms[1].data(), kHeadPartExtras) = {};
    LoadHeadPartTypes();
    assert(!MouthCandidate(specs[7].name));
    assert(!MouthCandidate(specs[8].name));
    assert(!MouthCandidate(specs[19].name));
    assert(MouthCandidate(specs[2].name));
  }
  test_se = test_vr = false;
}

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
  // Hair-owned ear pieces stay with the hair; independent accessories follow.
  expect("InigoEP", direct::Part::kNone, false);
  for (auto n : {"PAHP-Septum-Ring-HP", "PAHP-Left-Snakebite-Ring-HP",
                 "PAHP-Left-Brow-Barbell-HP", "CustomSlotHeadDecoration",
                 "UnknownSlotSeven"})
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
  auto different_model = std::make_shared<direct::Model>();
  different_model->network = model.network == 2 ? 3 : 2;
  assert(HeadKey(node.data(), false, different_model) != key);
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
  HairExclusion();
  OralInheritance();
  puts(
      "head-part classification passed: custom 32-bit slots, Misc, hair "
      "extras, vanilla and unknown controls");
  return 0;
}
