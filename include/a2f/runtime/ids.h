// Address Library ids of the engine functions the plugin hooks and calls, as
// {SE 1.5.97, AE, VR 1.4.15} with VR's as offsets into SkyrimVR.exe; 0 where a
// runtime has no such function. Ids in comments are AE ids. SE ids and VR
// offsets were matched against the 1.5.97 and SkyrimVR.exe disassembly;
// tests/runtimes_test.cc only checks that each SE/AE id exists in that
// runtime's Address Library.
#ifndef INCLUDE_A2F_RUNTIME_IDS_H_
#define INCLUDE_A2F_RUNTIME_IDS_H_

#include "a2f/runtime/skse.h"

constexpr skse::Rel kIdSpeakVoiceLine{36541, 37542, 0x5F0E20};
constexpr skse::Rel kIdDialogueResponseCtor{34429, 35249, 0x573B70};
// play-response, UpdateInDialogue, 44673, console SpeakSound
constexpr skse::Rel kSpeakCallers[] = {{38747, 39772, 0x681F80},
                                       {36540, 37541, 0x5F0790},
                                       {43467, 44673, 0x7940F0},
                                       {22389, 22864, 0x3247E0}};
// DialogueItem ctor, MenuTopicManager x2 (35254 is inlined into the other on
// SE, which leaves one call there)
constexpr skse::Rel kResponseCallers[] = {
    {34413, 35220, 0x572FD0}, {0, 35254, 0}, {34436, 35256, 0x574360}};
// BSResourceNiBinaryStream
constexpr skse::Rel kIdStreamCtor{69636, 71014, 0xCBC490};
constexpr skse::Rel kIdStreamDtor{69638, 71016, 0xCBC770};
// BSFaceGenAnimationData::Update(fg, dt, bool)
constexpr skse::Rel kIdFaceAnimUpdate{25983, 26598, 0x3D38B0};
constexpr skse::Rel kIdFaceNodeVtable{252410, 200333, 0x1678620};
// BSFaceGenNiNode: update keyframes, apply morphs
constexpr skse::Rel kFaceCallers[] = {{26417, 26998, 0x3E8DC0}};
// BSFaceGenNiNode::ApplyMorphs(node, force): reset to base + morphs. Every
// call in the executable is hooked, not only the "Face morphing" job (26999)
// and the player's face update (36564): the engine's face rebuild that
// RaceMenu runs (UpdateModelFace), and so Mu Dynamic NPC Replacer when it
// re-applies a preset, calls it from elsewhere, and a reset the hook does not
// see drops that frame's speech deltas (a flicker).
constexpr skse::Rel kIdApplyMorphs{26407, 26988, 0x3E82F0};
// shape -> BSFaceGenMorphDataHead
constexpr skse::Rel kIdMorphData{26441, 27020, 0x3E9B90};
// add one morph to a shape
constexpr skse::Rel kIdApplyMorph{26348, 26926, 0x3E70B0};
// BSDynamicTriShape data lock
constexpr skse::Rel kIdLockDynamic{69566, 70950, 0xCB8770};
constexpr skse::Rel kIdUnlockDynamic{69568, 70952, 0xCB87D0};
// BSDynamicTriShape::UpdateDynamicData(float3*, bool): what the engine's morph
// kernel writes through; renderers that mirror face meshes (Creation Engine
// Raytracing) hook it, so Direct writes through it too
constexpr skse::Rel kIdUpdateDynamic{69570, 70954, 0xCB8880};
// bool(BSSoundHandle*)
constexpr skse::Rel kIdSoundIsPlaying{66359, 67620, 0xC284F0};
// ResId from "Sound\...\X.lip"
constexpr skse::Rel kIdGenerateFromPath{68635, 69979, 0xC8F6C0};
// TESDataHandler* singleton (CommonLibSSE-NG Offset::TESDataHandler)
constexpr skse::Rel kIdDataHandler{514141, 400269, 0x1F82AD8};
// MenuTopicManager* singleton (CommonLibSSE-NG GetSingleton)
constexpr skse::Rel kIdMenuTopicManager{514959, 401099, 0x2FC4878};
// LookupReferenceByHandle(handle, NiPointer<TESObjectREFR>&) and
// CreateRefHandle(handle, TESObjectREFR*) (CommonLibSSE-NG Misc.cpp)
constexpr skse::Rel kIdLookupReference{12204, 12332, 0x143180};
constexpr skse::Rel kIdCreateHandle{12193, 12326, 0x1428A0};
// PlayerCharacter* singleton (CommonLibSSE-NG PlayerCharacter::GetSingleton).
// None on VR: DBReV supports SE/AE, and VR's player is not verified.
constexpr skse::Rel kIdPlayer{517014, 403521, 0};
// Optional desktop NPC targeting/menu guards (CommonLibSSE-NG).
constexpr skse::Rel kIdCrosshairPick{515446, 401585, 0};
constexpr skse::Rel kIdUI{514178, 400327, 0};
constexpr skse::Rel kIdControlMap{514705, 400863, 0};
// TESForm::LookupByID(FormID) (CommonLibSSE-NG TESForm.h). None on VR: the
// provider interface (BFA_API.h) resolves actors by form id on SE/AE only.
constexpr skse::Rel kIdLookupFormById{14461, 14617, 0};
// AIProcess::GetHeadtrackTarget: hidden result pointer after this, returns
// that pointer. Unavailable on VR: its target-selection ABI is unverified.
constexpr skse::Rel kIdHeadtrackTarget{38483, 39484, 0};

// The single ids above that every runtime has, for tests/runtimes_test.cc.
constexpr skse::Rel kIds[] = {
    kIdSpeakVoiceLine,   kIdDialogueResponseCtor, kIdStreamCtor,
    kIdStreamDtor,       kIdFaceAnimUpdate,       kIdApplyMorphs,
    kIdMorphData,        kIdApplyMorph,           kIdLockDynamic,
    kIdUnlockDynamic,    kIdUpdateDynamic,        kIdSoundIsPlaying,
    kIdGenerateFromPath, kIdDataHandler,          kIdMenuTopicManager,
    kIdLookupReference,  kIdCreateHandle,         kIdFaceNodeVtable,
};
// And those some runtime lacks.
constexpr skse::Rel kOptionalIds[] = {
    kIdPlayer,     kIdCrosshairPick,   kIdUI,
    kIdControlMap, kIdHeadtrackTarget, kIdLookupFormById};

#endif  // INCLUDE_A2F_RUNTIME_IDS_H_
