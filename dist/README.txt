Better Facial Animations @VERSION@ - real-time Audio2Face lip-sync for Skyrim

Every voiced line spoken by an NPC with a FaceGen head (enabled races; creatures and animals without FaceGen heads are skipped)
is run through NVIDIA Audio2Face-3D v3.0 on your GPU (women speak with its Claire face, men with Mark; v2.3 Mark for
everyone with UseOldModel=1 in the ini), which deforms the face mesh directly: skin, lips,
brows and eyelids, with the lower teeth and gums on Audio2Face's jaw. The tongue follows half the jaw movement with reduced, overlap-limited tongue cues from the game's speech track. The game's own emotion
expressions (dialogue, mood, combat) fade out while a line plays, when Audio2Face alone shapes the face, and back
in after it. A line's emotion comes from the game's emotion for it.
Each head is fitted to the animated face the first time it speaks, from its own expression morphs, so custom heads,
NPC replacers and High Poly Head animate like vanilla ones; a head whose .tri lacks one of the morphs it needs keeps
vanilla lip-sync. Works with Creation Engine Raytracing. Per-region scales and the
emotion strength are in the settings page. Vanilla, DLC and modded voices all work, packed or loose, with or without .lip files.
Lines with missing or silent voice audio (including Fuz Ro D'oh dialogue) keep vanilla lip-sync and expressions. BFA takes over only when an animation from audible audio is ready.

Requirements
  Skyrim Special Edition 1.5.97, 1.6.353, 1.6.640, 1.6.659 (GOG), 1.6.1170,
  1.6.1179 (GOG), 1.7.99 or 1.7.104, matching SKSE, Address Library for SKSE Plugins.
  Or Skyrim VR 1.4.15 with SKSEVR (no Address Library needed; not yet looked at in a headset).
  Not the Epic Games Store version: SKSE does not support it.
  V3/Old GPU inference needs DirectX 12 (~250 MB VRAM / ~100 MB respectively).
  Settings -> Model offers mutually exclusive V3, Performance and Old model checkboxes.
  Hover for tradeoffs. Model changes apply live; playing lines finish with their
  original animation. Performance (55k Thin96) uses the GPU by default or CPU with ForceCPU=1:
  smaller, but less emotive, 30 fps, no eye gaze feature.
  It is experimental and Skyrim-trained; not yet verified on Steam Deck.
  UsePerformanceModel=1 selects it; ForceCPU applies to all three models. PotatoMode=1 processes only the active
  conversation NPC and the player (DBReV or experimental DBVO 2 support). Other NPC speech stays vanilla.
  Both modes keep the model loaded from the main menu until exit. An already submitted GPU run must finish
  before cancellation. PotatoMode=0 (default) processes all speakers. If the model cannot
  be loaded on the GPU the settings window says why, it is retried and lines run on the CPU meanwhile (slow).
  Under ReShade the model runs on the device inside ReShade's wrapper, and switches to the wrapper itself
  when that device is lost (DLSS 5 bridges); UseDeviceWrapper=1 in the ini always takes the wrapper.
  Experimental [General] IndependentGpuDevice=1 requires an independent D3D12 device and private compute queue.
  Restart after changing it. It launches no EXE and falls back to CPU on failure, never to the shared GPU device.
  It requires supporting Windows D3D12 runtime/driver; ReShade/RenoDX/frame-generation compatibility is unverified.
  UseDeviceWrapper is ignored and metacommands remain enabled in this mode. Default 0 preserves the legacy path.
  Optional: SKSE Menu Framework, for the in-game settings page (Better Facial Animations -> Settings).

Player dialogue (optional)
  Dragonborn ReVoiced (DBReV) 1.4.4+ with a compatible voice pack animates the player through BFA.
  Select the voice in DBReV's MCM. A third-person/cinematic camera lets you see the player's face.
  General -> Player facial animations toggles this separately from NPC faces; the INI equivalent is
  [General] PlayerFacialAnimations=0. Player animation also works in Potato mode. The master Enabled setting still applies.
  With Cinematic Conversation Camera, use [Performance] bSynthLipSync=0 in SKSE/Plugins/SD_user.ini
  to disable its synthesized fallback lips. Turning off BFA's player feature leaves DBReV's own lip sync.
  The same supported-head/race requirements apply. DBReV is optional, and its DLL is not patched.

Settings
  Data/SKSE/Plugins/BetterFacialAnimations.ini, or in game through SKSE Menu Framework (saved automatically).
  Male and Female NPCs and the Player each have independent playback, strength and emotion sliders.
  Each profile has ten separate model emotion caps (0..2, default 2), applied after strength and volume swell from the next line. Lower Max joy / happiness to soften smiles independently. Grief, out of breath and pain are not currently fed by Skyrim.
  Upper lip and Lower lip multiply Mouth strength; 1 uses permanent 0.8 upper / 1.2 lower calibration.
  Eye gaze adds speech-driven eye motion when the speaking conversation NPC looks at you (SE/AE).
  Toggle Enable eye gaze in Better Facial Animations -> Settings -> Strength; changes apply immediately.
  It preserves the game's target and stops when the actor looks elsewhere. Enabled by default;
  [LocalAnimationStrength] ConversationGaze=0 disables it, GazeStrength=0..2 adjusts it (default 1).
  VR keeps its existing gaze. Tested offline on original eye meshes; in-game/custom-eye calibration is unverified.
  Older shared settings seed both NPC profiles; older lip values convert to preserve their effect.
Log
  Documents/My Games/Skyrim Special Edition/SKSE/BetterFacialAnimations.log (Skyrim VR/SKSE/ on VR)
  The GPU, the driver and what went wrong loading the model are always written there: attach it to a bug
  report. Logging (menu Debug section or [Debug] Logging=1, on by default) adds the generated and played lines.

Contents of this folder
  onnxruntime.dll, onnxruntime_providers_shared.dll   ONNX Runtime 1.24.4, DirectML build (MIT)
  DirectML.dll                                         Microsoft DirectML 1.15.4 (licenses/DirectML-LICENSE.txt)
  a2f3_fp16.onnx                                       Audio2Face-3D v3.0, FP16 (NOTICE.txt)
  A2FDirect3.bin                                       its faces (Claire, Mark): basis, landmarks and network projection
  performance.onnx                                   Experimental Thin96 performance model
  network_fp16.onnx                                    Audio2Face-3D v2.3 Mark, FP16 (NOTICE.txt): UseOldModel=1
  A2FDirect.bin                                        its face (Mark)
BetterFacialAnimations.dll statically links FFmpeg 8.0.1 (LGPL 2.1): wmav1/wmav2/pcm decoders, xwma/wav demuxers, swresample.
Better Facial Animations is free software under the GNU GPL v3 (licenses/BetterFacialAnimations-GPL-3.0.txt).
Additional term under GPL v3 section 7(b): any copy or modified version must keep the attribution "Based on Better Facial Animations by Cyberquach" in its documentation and on any page it is distributed from.
Source code: https://gitlab.com/cyberquach/better-facial-animations

DBVO 2 EXPERIMENTAL PROTOTYPE
DBVO2 prototype for Skyrim: detected automatically alongside DBReV.
PlayerFacialAnimations controls both integrations and is on by default.
The old NativePlayerSpeech setting is ignored. Use default DBVO pitch (100).
The observer reuses BFA's player rendering without modifying DBVO. It refuses
unrecognized or already-hooked engine entry bytes. Enabled experimentally on
BFA-supported desktop runtimes: 1.5.97, 1.6.353, 1.6.640, 1.6.659 (GOG), 1.6.1170, 1.6.1179 (GOG),
1.7.99 and 1.7.104. VR and unknown runtimes are excluded. DBVO itself must
support the runtime; the pinned DBVO 2.0.2.2 DLL excludes 1.7.99.
1.6.353, 1.6.640, GOG 1.6.659, GOG 1.6.1179 and 1.7.99 are enabled but unverified.
Non-default pitch, no-LIP audio, skips and rapid replacements need validation.
1.6.1170 has in-game playback evidence. On 1.5.97 and 1.7.104, binary
hook/ABI and live SKSE/plugin startup checks pass; player playback
still needs in-game validation.

Races owned by DWAlfiqKhajiit.esp (Creatures of Nirn - Khajiit Alfiq)
default off in the Races menu, including corrupted and boss variants.
They can be enabled individually; compatible face morphs are still required.
Saved race choices are preserved. Ordinary Khajiit remain supported.
Renamed or merged origin plugins need a separate compatibility check.
