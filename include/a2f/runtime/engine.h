// The engine as the plugin sees it: object layouts and function types (the
// Address Library ids are in ids.h).
#ifndef INCLUDE_A2F_RUNTIME_ENGINE_H_
#define INCLUDE_A2F_RUNTIME_ENGINE_H_

#include <cstdint>

#include "a2f/runtime/ids.h"
#include "a2f/runtime/skse.h"

namespace bfa {

// BSFaceGenNiNode / BSDynamicTriShape / NiObjectNET. skse::Rel offsets are
// {SE, AE, VR}: VR's NiAVObject and BSGeometry are larger.
constexpr skse::Rel kNodeChildren{0x118, 0x118, 0x140};
constexpr skse::Rel kNodeNumChildren{0x122, 0x122, 0x14A};
constexpr skse::Rel kNodeFg{0x150, 0x150, 0x178};
constexpr skse::Rel kVfUpdateDownwardPass{0x2C, 0x2C, 0x2D};
struct NiUpdateData {
  float time;
  uint32_t flags;
};
static_assert(sizeof(NiUpdateData) == 8);
// u16, bit0 = reset needed
constexpr skse::Rel kNodeFlags{0x160, 0x160, 0x188};
constexpr uint32_t kObjName = 0x10;
constexpr uint32_t kObjExtra = 0x20;
constexpr uint32_t kObjNumExtra = 0x28;
constexpr skse::Rel kGeomType{0x150, 0x150, 0x190};  // 4 = BSDynamicTriShape
constexpr skse::Rel kDynNumVerts{0x15A, 0x15A, 0x19A};
constexpr skse::Rel kDynDataSize{0x170, 0x170, 0x1B0};
constexpr uint32_t kBaseVerts = 0x18;  // BSFaceGenBaseMorphExtraData ("FOD")
constexpr uint32_t kBaseNumVerts = 0x24;
constexpr uint32_t kFgDirty = 0x218;  // byte: run the face update next frame
// Actor runtime data moved by eight bytes in 1.6.629 (CommonLibSSE-NG
// Actor::GetActorRuntimeData). Earlier AE still uses AE Address Library ids.
// Unlike Rel, Get depends on the exact runtime, so it is not constexpr.
struct ActorOffset {
  uint32_t before_629;
  uint32_t Get() const {
    return before_629 + (skse::Runtime() >= skse::Version(1, 6, 629) ? 8 : 0);
  }
};
constexpr ActorOffset kActorProcess{0xF0};
constexpr uint32_t kProcessHigh = 0x10;
constexpr uint32_t kHighVoiceHandle = 0x2FC;
constexpr uint32_t kFormId = 0x14;
constexpr uint32_t kPlayerFormId = 0x14;  // the form id itself
constexpr uint32_t kFormType = 0x1A;
constexpr uint32_t kRefrBase = 0x40;
constexpr uint32_t kNpcFlags = 0x38;  // TESNPC actor data flags; bit 0: female
// MenuTopicManager: the handle of the actor the player is talking to
constexpr uint32_t kMtmSpeaker = 0x68;
// TESObjectREFR's BSHandleRefObject and its reference count, whose low bits
// are the count (the rest: "has a handle" and the handle's entry index)
constexpr uint32_t kRefrHandleObj = 0x20;
constexpr uint32_t kRefrRefCount = kRefrHandleObj + 8;
constexpr uint32_t kRefCountMask = 0x3FF;
// NiAVObject: parent NiNode, and the reference a 3D root belongs to
constexpr uint32_t kAvParent = 0x30;
constexpr skse::Rel kAvUserData{0xF8, 0xF8, 0x110};
constexpr ActorOffset kActorRace{0x1F0};
constexpr uint32_t kRaceFlags = 0x108;
constexpr uint32_t kRaceName = 0x28;  // TESFullName::fullName
constexpr uint32_t kRaceEditorId = 0x2E0;  // TESRace::formEditorID
constexpr uint8_t kFtRace = 0x0E;
constexpr uint8_t kFtFaction = 0x0B;
constexpr uint8_t kFtHeadPart = 0x0C;
constexpr uint8_t kFtNpc = 0x2B;
constexpr uint8_t kFtActor = 0x3E;
constexpr uint32_t kVfGetFaceGenAnimationData = 0x63;
// Actor::IsInFaction(const TESFaction*), as {SE, AE, VR}
constexpr skse::Rel kVfIsInFaction{0xF9, 0xF9, 0xFB};
// TESForm::sourceFiles: TESFileArray* {TESFile** files; u32 count}, the
// first being the file the form is from; TESFile::fileName
constexpr uint32_t kFormFiles = 0x08;
constexpr uint32_t kFilesArray = 0x00;
constexpr uint32_t kFilesCount = 0x08;
constexpr uint32_t kFileName = 0x58;
// TESDataHandler::formArrays, one BSTArray per form type
constexpr uint32_t kDhFormArrays = 0x10;
// BGSHeadPart
constexpr uint32_t kHeadPartType = 0x6C;
constexpr uint32_t kHeadPartExtras = 0x70;  // BSTArray<BGSHeadPart*>
constexpr uint32_t kHeadPartName = 0x118;   // BSFixedString formEditorID
enum HeadPartType : uint8_t {
  kHpMisc,
  kHpFace,
  kHpEyes,
  kHpHair,
  kHpFacialHair,
  kHpScar,
  kHpEyebrows
};
// DialogueResponse
constexpr uint32_t kRespEmoType = 0x10;
constexpr uint32_t kRespEmoValue = 0x14;
constexpr uint32_t kRespVoice = 0x18;

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
using IsInFactionFn = bool (*)(const uint8_t* actor, const uint8_t* faction);

// An engine object's field at byte offset off.
template <typename T>
T& Field(void* p, uint64_t off) {
  return *reinterpret_cast<T*>(static_cast<uint8_t*>(p) + off);
}
// An engine function by Address Library id, resolved on first use (after the
// library is loaded).
template <typename Fn, const skse::Rel& kId>
Fn Engine() {
  static const Fn fn = reinterpret_cast<Fn>(skse::Addr(kId));
  return fn;
}

// The name of the plugin file a form was loaded from, or null.
inline const char* SourceFileName(uint8_t* form) {
  uint8_t* files = Field<uint8_t*>(form, kFormFiles);
  if (!files || !Field<uint32_t>(files, kFilesCount)) return nullptr;
  uint8_t** items = Field<uint8_t**>(files, kFilesArray);
  if (!items || !items[0]) return nullptr;
  return reinterpret_cast<const char*>(items[0] + kFileName);
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
    if (!handle) return;
    auto lookup = Engine<bool (*)(const uint32_t&, uint8_t**),
                         kIdLookupReference>();
    if (lookup) lookup(handle, &ref_);
  }
  ~ScopedReference() {
    if (!ref_) return;
    const LONG count = InterlockedDecrement(&Field<LONG>(ref_, kRefrRefCount));
    if (!(count & kRefCountMask)) {
      uint8_t* object = ref_ + kRefrHandleObj;
      auto destroy = reinterpret_cast<void (*)(uint8_t*)>(
          Field<void**>(object, 0)[1]);
      destroy(object);
    }
  }
  ScopedReference(const ScopedReference&) = delete;
  ScopedReference& operator=(const ScopedReference&) = delete;
  uint8_t* get() const { return ref_; }

 private:
  uint8_t* ref_ = nullptr;
};

// BSTArray<T*>: items, capacity, size.
struct BSTArray {
  uint8_t** items;
  uint32_t cap, pad, n;
  bool Ok() const { return n <= cap && n <= 1000000 && (items || !n); }
};
static_assert(sizeof(BSTArray) == 0x18, "BSTArray");

// The game's forms of a type (TESDataHandler::formArrays), or null when the
// array is not there or does not look right.
inline const BSTArray* FormArray(uint8_t type) {
  const auto* handler =
      reinterpret_cast<uint8_t* const*>(skse::Addr(kIdDataHandler));
  if (!handler || !*handler) return nullptr;
  const BSTArray& forms =
      Field<const BSTArray>(*handler, kDhFormArrays + type * sizeof(BSTArray));
  return forms.Ok() ? &forms : nullptr;
}

}  // namespace bfa

#endif  // INCLUDE_A2F_RUNTIME_ENGINE_H_
