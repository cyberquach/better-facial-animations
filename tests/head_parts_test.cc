// Exercise the production registry/classifier using synthetic engine HDPTs.
#include <array>
#include <cassert>
#include <cctype>
#include <cstdio>
#include <vector>

#include "../src/head_cache.h"
namespace {
std::array<uint8_t, 0x1000> storage{};
uint8_t* handler = storage.data();
}  // namespace
namespace skse {
bool IsSe() { return false; }
bool IsVr() { return false; }
uint64_t Addr(uint64_t id) {
  return id == kIdDataHandler.ae ? reinterpret_cast<uint64_t>(&handler) : 0;
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
               {"CustomLashes", kHpMisc}};
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
  puts(
      "head-part classification passed: custom 32-bit slots, Misc, hair "
      "extras, vanilla and unknown controls");
}
