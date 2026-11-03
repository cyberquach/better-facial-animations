// The engine as the plugin sees it: object layouts and function types (the
// Address Library ids are in ids.h).
#ifndef INCLUDE_A2F_RUNTIME_ENGINE_H_
#define INCLUDE_A2F_RUNTIME_ENGINE_H_

#include <cstdint>

#include "a2f/runtime/ids.h"
#include "a2f/runtime/skse.h"

namespace bfa {

using NiUpdateData = RE::NiUpdateData;
constexpr uint32_t kPlayerFormId = 0x14;
constexpr uint64_t kShaderHairTint = uint64_t{1} << 18;
constexpr uint8_t kFtRace = 0x0E, kFtFaction = 0x0B, kFtHeadPart = 0x0C,
                  kFtNpc = 0x2B, kFtActor = 0x3E;
enum HeadPartType : uint8_t {
  kHpMisc,
  kHpFace,
  kHpEyes,
  kHpHair,
  kHpFacialHair,
  kHpScar,
  kHpEyebrows
};
// Project hook slots (upstream exposes methods, but not their vtable indices).
constexpr skse::Rel kVfUpdateDownwardPass{0x2C, 0x2C, 0x2D};

using SpeakFn = float (*)(uint8_t* actor, const char* voice, void* handle,
                          uint64_t emo_type, uint64_t emo_value, uint64_t,
                          uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
                          uint64_t, uint64_t, uint64_t);
using ResponseCtorFn = uint8_t* (*)(uint8_t* self, void* topic, void* info,
                                    uint8_t* speaker, void* response);
using AnimUpdateFn = bool (*)(uint8_t* fg, float dt, uint64_t flag);
using ApplyMorphsFn = uint64_t (*)(uint8_t* node, uint8_t force);
using ApplyMorphFn = void (*)(void* morph_data, int type, int index,
                              uint8_t* shape, float value);
using GenerateFromPathFn = void (*)(skse::ResId*, const char*);
using MorphDataFn = void* (*)(uint8_t* shape);
using LockDynamicFn = float* (*)(uint8_t* shape);
using UnlockDynamicFn = void (*)(uint8_t* shape);
using UpdateDynamicFn = void (*)(uint8_t* shape, float* xyz, bool);

// An engine function by Address Library id, resolved on first use (after the
// library is loaded).
template <typename Fn, const skse::Rel& kId>
Fn Engine() {
  static const Fn fn = reinterpret_cast<Fn>(skse::Addr(kId));
  return fn;
}

// The name of the plugin file a form was loaded from, or null.
inline const char* SourceFileName(uint8_t* form) {
  auto* typed = reinterpret_cast<RE::TESForm*>(form);
  if (!typed || !typed->sourceFiles.array || !typed->sourceFiles.array->data())
    return nullptr;
  auto* file = typed->GetFile(0);
  return file ? file->fileName : nullptr;
}
// The form id without its load-order index: 12 bits for light (FE) plugins,
// 24 for full ones.
inline uint32_t LocalFormId(uint32_t id) {
  return id & ((id >> 24) == 0xFE ? 0xFFF : 0xFFFFFF);
}

// The reference a handle is of, held (reference counted, as the engine's
// NiPointer<TESObjectREFR> the lookup writes) while this lives; null once it
// is gone. Its face and 3D are not held. To be destroyed with neither the
// plugin's nor a face's lock held: letting go of the last reference destroys
// the engine's object.
class ScopedReference {
 public:
  explicit ScopedReference(uint32_t handle) {
    if (handle) {
      const auto lookup =
          Engine<bool (*)(const uint32_t&, RE::NiPointer<RE::TESObjectREFR>&),
                 kIdLookupReference>();
      if (lookup) lookup(handle, ref_);
    }
  }
  uint8_t* get() const { return reinterpret_cast<uint8_t*>(ref_.get()); }

 private:
  RE::NiPointer<RE::TESObjectREFR> ref_;
};

// Reject malformed engine arrays before walking their storage.
template <typename T>
bool ValidArray(const RE::BSTArray<T>& values) {
  return values.size() <= values.capacity() && values.size() <= 1000000 &&
         (values.empty() || values.data());
}

inline const RE::BSTArray<RE::TESForm*>* FormArray(uint8_t type) {
  const auto address = skse::Addr(kIdDataHandler);
  auto* handler =
      address ? *reinterpret_cast<RE::TESDataHandler**>(address) : nullptr;
  if (!handler) return nullptr;
  const auto& forms = handler->formArrays[type];
  return ValidArray(forms) ? &forms : nullptr;
}

}  // namespace bfa

#endif  // INCLUDE_A2F_RUNTIME_ENGINE_H_
