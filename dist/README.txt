Better Facial Animations @VERSION@ - real-time Audio2Face lip-sync for Skyrim

Every voiced line spoken by an NPC with a FaceGen head (all playable races except Argonians, who keep vanilla lip-sync; creatures and animals are skipped)
is run through NVIDIA Audio2Face-3D v3.0 on your GPU (women speak with its Claire face, men with Mark; v2.3 Mark for
everyone with UseOldModel=1 in the ini), which deforms the face mesh directly: skin, lips,
brows and eyelids, with the lower teeth and gums on Audio2Face's jaw. The tongue follows half the jaw movement with reduced, overlap-limited tongue cues from the game's speech track. The game's own emotion
expressions (dialogue, mood, combat) fade out while a line plays, when Audio2Face alone shapes the face, and back
in after it. A line's emotion comes from the game's emotion for it.
Each head is fitted to the animated face the first time it speaks, from its own expression morphs, so custom heads,
NPC replacers and High Poly Head animate like vanilla ones; a head whose .tri lacks one of the morphs it needs keeps
vanilla lip-sync. Works with Creation Engine Raytracing. Per-region scales and the
emotion strength are in the settings page. Vanilla, DLC and modded voices all work, packed or loose, with or without .lip files.

Requirements
  Skyrim Special Edition 1.6.1170, 1.6.1179 (GOG), 1.7.99 or 1.7.104 (AE), matching SKSE, Address Library for SKSE Plugins.
  Or Skyrim VR 1.4.15 with SKSEVR (no Address Library needed; not yet looked at in a headset).
  Not the Epic Games Store version: SKSE does not support it.
  A DirectX 12 GPU (~250 MB of VRAM, ~100 MB with UseOldModel=1, held from the main menu until the game exits). If the model cannot
  be loaded on the GPU the settings window says why, it is retried and lines run on the CPU meanwhile (slow).
  Optional: SKSE Menu Framework, for the in-game settings page (Better Facial Animations -> Settings).

Player dialogue (optional)
  Dragonborn ReVoiced (DBReV) 1.4.4+ with a compatible voice pack animates the player through BFA.
  Select the voice in DBReV's MCM. A third-person/cinematic camera lets you see the player's face.
  General -> Player facial animations toggles this separately from NPC faces; the INI equivalent is
  [General] PlayerFacialAnimations=0. The master Enabled setting still applies.
  With Cinematic Conversation Camera, use [Performance] bSynthLipSync=0 in SKSE/Plugins/SD_user.ini
  to disable its synthesized fallback lips. Turning off BFA's player feature leaves DBReV's own lip sync.
  The same supported-head/race requirements apply. DBReV is optional, and its DLL is not patched.

Settings
  Data/SKSE/Plugins/BetterFacialAnimations.ini, or in game through SKSE Menu Framework.
Log
  Documents/My Games/Skyrim Special Edition/SKSE/BetterFacialAnimations.log (Skyrim VR/SKSE/ on VR)
  The GPU, the driver and what went wrong loading the model are always written there: attach it to a bug
  report. Logging (menu Debug section or [Debug] Logging=1, on by default) adds the generated and played lines.

Contents of this folder
  onnxruntime.dll, onnxruntime_providers_shared.dll   ONNX Runtime 1.24.4, DirectML build (MIT)
  DirectML.dll                                         Microsoft DirectML 1.15.4 (licenses/DirectML-LICENSE.txt)
  a2f3_fp16.onnx                                       Audio2Face-3D v3.0, FP16 (NOTICE.txt)
  A2FDirect3.bin                                       its faces (Claire, Mark): basis, landmarks and network projection
  network_fp16.onnx                                    Audio2Face-3D v2.3 Mark, FP16 (NOTICE.txt): UseOldModel=1
  A2FDirect.bin                                        its face (Mark)
BetterFacialAnimations.dll statically links FFmpeg 8.0.1 (LGPL 2.1): wmav1/wmav2/pcm decoders, xwma/wav demuxers, swresample.
Better Facial Animations is free software under the GNU GPL v3 (licenses/BetterFacialAnimations-GPL-3.0.txt).
Additional term under GPL v3 section 7(b): any copy or modified version must keep the attribution "Based on Better Facial Animations by Cyberquach" in its documentation and on any page it is distributed from.
Source code: https://gitlab.com/cyberquach/better-facial-animations
