// The engine as the plugin sees it: object layouts and function types (the
// Address Library ids are in ids.h).
#ifndef SRC_ENGINE_H_
#define SRC_ENGINE_H_

#include <cstdint>

#include "skse.h"

namespace bfa {

// BSFaceGenNiNode / BSDynamicTriShape / NiObjectNET. skse::Rel offsets are
// {SE, AE, VR}: VR's NiAVObject and BSGeometry are larger.
constexpr skse::Rel kNodeChildren{0x118, 0x118, 0x140};
constexpr skse::Rel kNodeNumChildren{0x122, 0x122, 0x14A};
constexpr skse::Rel kNodeFg{0x150, 0x150, 0x178};
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
constexpr skse::Rel kActorProcess{0xF0, 0xF8, 0xF0};
constexpr uint32_t kProcessHigh = 0x10;
constexpr uint32_t kHighVoiceHandle = 0x2FC;
constexpr uint32_t kFormId = 0x14;
constexpr uint32_t kFormType = 0x1A;
constexpr uint32_t kRefrBase = 0x40;
// MenuTopicManager: the handle of the actor the player is talking to
constexpr uint32_t kMtmSpeaker = 0x68;
// BSPointerHandleManager entries: bits (in use, age), BSHandleRefObject*
constexpr uint32_t kHandleIndexMask = 0xFFFFF, kHandleAgeMask = 0x3F00000,
                   kHandleInUse = 0x4000000, kHandleEntrySize = 0x10;
constexpr uint32_t kRefrHandleObj = 0x20;  // TESObjectREFR's BSHandleRefObject
// its reference count: the count, "has a handle", the handle's entry index
constexpr uint32_t kRefrRefCount = kRefrHandleObj + 8;
constexpr uint32_t kRefCountHasHandle = 0x400, kRefCountIndexShift = 11;
// NiAVObject: parent NiNode, and the reference a 3D root belongs to
constexpr uint32_t kAvParent = 0x30;
constexpr skse::Rel kAvUserData{0xF8, 0xF8, 0x110};
constexpr skse::Rel kActorRace{0x1F0, 0x1F8, 0x1F0};
constexpr uint32_t kRaceFlags = 0x108;
// Skyrim.esm ArgonianRace / ArgonianRaceVampire: their snout doesn't match the
// human-trained model, so they keep vanilla lip-sync.
constexpr uint32_t kArgonianRace = 0x00013740;
constexpr uint32_t kArgonianRaceVampire = 0x0008883A;
constexpr uint8_t kFtHeadPart = 0x0C;
constexpr uint8_t kFtNpc = 0x2B;
constexpr uint8_t kFtActor = 0x3E;
constexpr uint32_t kVfGetFaceGenAnimationData = 0x63;
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
  kHpEyebrows,
  kHpNumTypes
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

}  // namespace bfa

#endif  // SRC_ENGINE_H_
