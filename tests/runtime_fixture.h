#pragma once
#include "a2f/runtime/engine.h"
namespace bfa {
template <typename T> T& Field(void* data, uint64_t offset) {
  return *reinterpret_cast<T*>(static_cast<uint8_t*>(data) + offset);
}
}
inline const bool fixture_runtime = [] {
  REL::Module::mock(REL::Version(1, 6, 353, 0), REL::Module::Runtime::AE,
                   L"SkyrimSE.exe", reinterpret_cast<uintptr_t>(GetModuleHandleW(nullptr)));
  return true;
}();
// Independent byte fixtures, not a second production engine implementation.
struct FixtureArray {
  uint8_t** items;
  uint32_t cap, pad, n;
};
inline RE::BSFaceGenKeyframeMultiple& FixtureKeyframe(void* object, size_t offset) {
  return *reinterpret_cast<RE::BSFaceGenKeyframeMultiple*>(static_cast<uint8_t*>(object) + offset);
}

// Fixed strings point after a real upstream pool entry; c_str() validates its flags.
inline const char* FixtureString(const char* value) {
  if (!value) return nullptr;
  static std::map<std::string, std::unique_ptr<std::byte[]>> strings;
  auto& memory = strings[value];
  if (!memory) {
    memory = std::make_unique<std::byte[]>(sizeof(RE::BSStringPool::Entry) + std::strlen(value) + 1);
    auto* entry = std::construct_at(reinterpret_cast<RE::BSStringPool::Entry*>(memory.get()));
    entry->_flags = 1;
    entry->_length = static_cast<uint32_t>(std::strlen(value));
    std::strcpy(reinterpret_cast<char*>(entry + 1), value);
  }
  return reinterpret_cast<const char*>(memory.get() + sizeof(RE::BSStringPool::Entry));
}
