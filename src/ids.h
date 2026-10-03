// Address Library ids of the engine functions the plugin hooks and calls, as
// {SE 1.5.97, AE} pairs; 0 where a runtime has no such function. Ids in
// comments are AE ids. SE ids were matched against the 1.5.97 disassembly;
// tests/runtimes_test.cc only checks that each exists on every supported
// runtime.
#ifndef SRC_IDS_H_
#define SRC_IDS_H_

#include "skse.h"

constexpr skse::Rel kIdSpeakVoiceLine{36541, 37542};
constexpr skse::Rel kIdDialogueResponseCtor{34429, 35249};
// play-response, UpdateInDialogue, 44673, console SpeakSound
constexpr skse::Rel kSpeakCallers[] = {
    {38747, 39772}, {36540, 37541}, {43467, 44673}, {22389, 22864}};
// DialogueItem ctor, MenuTopicManager x2 (35254 is inlined into the other on
// SE, which leaves one call there)
constexpr skse::Rel kResponseCallers[] = {
    {34413, 35220}, {0, 35254}, {34436, 35256}};
constexpr skse::Rel kIdStreamCtor{69636, 71014};  // BSResourceNiBinaryStream
constexpr skse::Rel kIdStreamDtor{69638, 71016};
// BSFaceGenAnimationData::Update(fg, dt, bool)
constexpr skse::Rel kIdFaceAnimUpdate{25983, 26598};
// BSFaceGenNiNode: update keyframes, apply morphs
constexpr skse::Rel kFaceCallers[] = {{26417, 26998}};
// BSFaceGenNiNode::ApplyMorphs(node, force): reset to base + morphs, run by
// the "Face morphing" job (26999) and the player's face update (36564)
constexpr skse::Rel kIdApplyMorphs{26407, 26988};
constexpr skse::Rel kMorphCallers[] = {{26418, 26999}, {35565, 36564}};
// shape -> BSFaceGenMorphDataHead
constexpr skse::Rel kIdMorphData{26441, 27020};
constexpr skse::Rel kIdApplyMorph{26348, 26926};  // add one morph to a shape
// BSDynamicTriShape data lock
constexpr skse::Rel kIdLockDynamic{69566, 70950};
constexpr skse::Rel kIdUnlockDynamic{69568, 70952};
// BSDynamicTriShape::UpdateDynamicData(float3*, bool): what the engine's morph
// kernel writes through; renderers that mirror face meshes (Creation Engine
// Raytracing) hook it, so Direct writes through it too
constexpr skse::Rel kIdUpdateDynamic{69570, 70954};
// bool(BSSoundHandle*)
constexpr skse::Rel kIdSoundIsPlaying{66359, 67620};
// ResId from "Sound\...\X.lip"
constexpr skse::Rel kIdGenerateFromPath{68635, 69979};
// TESDataHandler* singleton (CommonLibSSE-NG Offset::TESDataHandler)
constexpr skse::Rel kIdDataHandler{514141, 400269};
// MenuTopicManager* singleton and the reference handle table
// (CommonLibSSE-NG MenuTopicManager::GetSingleton, GetHandleEntries)
constexpr skse::Rel kIdMenuTopicManager{514959, 401099};
constexpr skse::Rel kIdHandleEntries{514478, 400622};

// The single ids above, for tests/runtimes_test.cc.
constexpr skse::Rel kIds[] = {
    kIdSpeakVoiceLine,   kIdDialogueResponseCtor, kIdStreamCtor,
    kIdStreamDtor,       kIdFaceAnimUpdate,       kIdApplyMorphs,
    kIdMorphData,        kIdApplyMorph,           kIdLockDynamic,
    kIdUnlockDynamic,    kIdUpdateDynamic,        kIdSoundIsPlaying,
    kIdGenerateFromPath, kIdDataHandler,          kIdMenuTopicManager,
    kIdHandleEntries,
};

#endif  // SRC_IDS_H_
