Better Facial Animations 1.2.6 - real-time Audio2Face lip-sync for Skyrim

Every voiced line spoken by an NPC with a FaceGen head (all playable races except Argonians, who keep vanilla lip-sync; creatures and animals are skipped)
is run through NVIDIA Audio2Face-3D v2.3 (Mark) on your GPU, which deforms the face mesh directly: skin, lips,
brows and eyelids, with the jaw, teeth and gums on Skyrim's own mouth shapes and the tongue on top, plus the dialogue
emotion expression. Works with High Poly Head and with Creation Engine Raytracing. Per-region scales and the
expression are in Settings -> Direct. Vanilla, DLC and modded voices all work, packed or loose, with or without .lip files.

Requirements
  Skyrim Special Edition 1.6.1170, 1.6.1179 (GOG) or 1.7.104 (AE), matching SKSE, Address Library for SKSE Plugins.
  A DirectX 12 GPU (~85 MB of VRAM). Without one it falls back to the CPU.
  Optional: SKSE Menu Framework, for the in-game settings page (Better Facial Animations -> Settings).

Settings
  Data/SKSE/Plugins/BetterFacialAnimations.ini, or in game through SKSE Menu Framework.
Log
  Off by default. Turn on Logging (menu Debug section or [Debug] Logging=1), then see
  Documents/My Games/Skyrim Special Edition/SKSE/BetterFacialAnimations.log

Contents of this folder
  onnxruntime.dll, onnxruntime_providers_shared.dll   ONNX Runtime 1.24.4, DirectML build (MIT)
  DirectML.dll                                         Microsoft DirectML 1.15.4 (licenses/DirectML-LICENSE.txt)
  network_fp16.onnx                                    Audio2Face-3D v2.3 Mark, FP16 (NOTICE.txt)
  A2FDirect.bin                                        Direct mode face basis, region weights and mouth rig
BetterFacialAnimations.dll statically links FFmpeg 8.0.1 (LGPL 2.1): wmav1/wmav2/pcm decoders, xwma/wav demuxers, swresample.
Better Facial Animations is free software under the GNU GPL v3 (licenses/BetterFacialAnimations-GPL-3.0.txt).
Additional term under GPL v3 section 7(b): any copy or modified version must keep the attribution "Based on Better Facial Animations by Cyberquach" in its documentation and on any page it is distributed from.
Source code: the optional file "Better Facial Animations - Source" on the Nexus Mods page.
