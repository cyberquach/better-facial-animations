Better Facial Animations @VERSION@ - real-time Audio2Face lip-sync for Skyrim

Every voiced line spoken by an NPC with a FaceGen head (enabled races; creatures and animals without FaceGen heads are skipped)
is run through NVIDIA Audio2Face-3D v3.0 on your GPU (women speak with its Claire face, men with Mark; v2.3 Mark for
everyone with UseOldModel=1 in the ini), which deforms the face mesh directly: skin, lips,
brows and eyelids, with the lower teeth and gums on Audio2Face's jaw. The tongue follows half the jaw movement with reduced, overlap-limited tongue cues from the game's speech track. The game's own emotion
expressions (dialogue, mood, combat) fade out while a line plays, when Audio2Face alone shapes the face, and back
in after it. Sentiment Analysis can add Audio2Emotion voice sentiment to the game's dialogue emotion.
Vanilla / scene emotion inputs (on by default) passes authored scene and dialogue-mod emotions to the model.
This toggle is independent of Sentiment Analysis; both off keeps audio-driven lip-sync without these emotion inputs.
The settings menu groups shared controls separately from selectable Male NPCs, Female NPCs and Player profiles.
Maximum emotion inputs are ceilings on what the model receives, not fixed emotion levels or movement strength.
Audio2Emotion offers Disabled, Full and Performance (default). Hover over the selected
value or an open dropdown option for resource use and tradeoffs. Full and Performance
support compatible AMD and NVIDIA GPUs through DirectML, or CPU via the separate
Audio2Emotion processing device selector. Each model defaults to GPU 0 and saves
its device choice independently.
Full uses the original model for each spoken line; typically about 600 MB of extra VRAM
plus additional processing time. CPU uses system RAM and extra CPU processing.
Performance is an experimental 21.6 MB student model trained to memorize Full's answers
on the training dialogue. It does not reliably handle lines it was not trained on:
results can be hit or miss, especially in obscure quest mods. Even training answers
are approximate. The weight-file size is not a runtime VRAM estimate. Use Full when
coverage of unfamiliar dialogue matters. See emotion_performance.json for checkpoint
identity and measured export validation; no in-game quality acceptance is claimed.
Both models only add voice-driven emotion alongside authored scene emotion. The
Sentiment addition multiplier (0-2, default 1) scales that contribution; zero skips it.
The combined inputs obey the profile's maximum emotion caps. Selecting Disabled
unloads the extra session after current inference returns. Audio2Face must be enabled
for Audio2Emotion to contribute; disabling Audio2Face remembers the emotion selection.
Both archives include Performance. The with-sentiment archive also includes Full;
without-sentiment omits only Full.
Restart after adding/removing model files. INI: SentimentAnalysis=0 disables emotion;
otherwise UsePerformanceEmotion=0 selects Full and UsePerformanceEmotion=1 Performance.
Each head is fitted to the animated face the first time it speaks, from its own expression morphs, so custom heads,
NPC replacers and High Poly Head animate like vanilla ones; a head whose .tri lacks one of the morphs it needs keeps
vanilla lip-sync. Works with Creation Engine Raytracing. Per-region scales and the
emotion strength are in the settings page. Vanilla, DLC and modded voices all work, packed or loose, with or without .lip files.
Lines with missing or silent voice audio (including Fuz Ro D'oh dialogue) keep vanilla lip-sync and expressions. BFA takes over only when an animation from audible audio is ready.

Requirements
  Microsoft Visual C++ v14 x64 runtime (tested with 14.51).
  Skyrim Special Edition 1.5.97, 1.6.353, 1.6.640, 1.6.659 (GOG), 1.6.1170,
  1.6.1179 (GOG), 1.7.99 or 1.7.104, matching SKSE, Address Library for SKSE Plugins.
  Or Skyrim VR 1.4.15 with SKSEVR (no Address Library needed; not yet looked at in a headset).
  Not the Epic Games Store version: SKSE does not support it.
  V3/Old GPU inference needs DirectX 12 (~250 MB VRAM / ~100 MB respectively).
  Settings -> Animation models offers Audio2Face: Disabled, V3, Performance, Ultra Performance, Legacy.
  Audio2Emotion has its own Disabled, Full, Performance dropdown.
  Hover for tradeoffs. Model changes apply live; playing lines finish with their
  original animation. Performance (4M Block AttnRes) runs on GPU, or CPU with ForceCPU=1:
  compact V3-trained motion with per-line memory, 30 fps, no eye gaze feature.
  It is trained to reproduce V3's answers on the training dialogue, prioritizing
  close reproduction of familiar lines over broad generalization. Unfamiliar
  dialogue can be less reliable; it approximates answers, not stored animations.
  Visual equivalence to V3 has not been established. Not yet verified on Steam Deck.
  Ultra Performance is the original 80,729-parameter Performance model, retained
  for the lowest computation cost. It trades facial detail and lip-sync accuracy
  for speed, especially on unfamiliar dialogue. GPU or CPU, 30 fps, no eye gaze.
  Older saved Performance selections migrate once to Ultra Performance, preserving
  that model choice. Select Performance explicitly to opt into the larger model.
  Audio2Face processing device and Audio2Emotion processing device each select
  a named GPU or CPU (Force CPU). Both default to GPU 0 and save their own choice;
  changing one does not change the other. GpuAdapter (default 0) and ForceCPU
  select facial processing. EmotionDevice=-1 selects emotion CPU, and 0 or higher
  selects an emotion DXGI GPU (default 0).
  The shared/independent GPU route applies globally to both models. Open either
  dropdown after game entry to discover GPUs; Refresh GPU list rescans. Missing
  GPUs report a failure; choose another device explicitly. Reselect by name after
  hardware/driver changes that reorder adapters.
  UsePerformanceModel=1 selects Performance; UseUltraPerformanceModel=1 selects
  Ultra Performance; UseOldModel=1 selects Legacy. Set the other model switches
  to 0; all three at 0 selects V3. ForceCPU applies to all four facial models.
  Potato Mode (PotatoMode=1) processes only the active
  conversation NPC and the player (DBReV or experimental DBVO 2 support). Other NPC speech stays vanilla.
  Both modes initialize inference after the first successful save load, or after the main/loading
  screens close for a new game, then keep the model loaded while enabled. Disabled unloads both sessions. Early lines use vanilla animation while it warms up. An already submitted GPU run must finish
  before cancellation. PotatoMode=0 (default) processes all speakers. If the model cannot
  be loaded on the GPU the settings window says why; vanilla lip-sync remains active.
  GPU inference defaults to a shared device (IndependentGpuDevice=0) that unwraps
  ReShade by itself. The opt-in installer checkbox "Independent GPU mode" selects a
  separate device/private queue (IndependentGpuDevice=1). It requires Windows 10 22H2
  or Windows 11 with the August 2023 or later update (D3D12 Independent Devices API;
  Proton: current vkd3d-proton) AND a graphics driver that supports it; ordinary
  DirectX 12 support is insufficient. Read the installer text and GPU-SETUP.txt. No
  universal minimum driver version is verified. Neither route guarantees compatibility
  with graphics mods. Change the installer/INI and restart to switch routes. Shared
  mode retries without metacommands and, after eligible device loss, through an
  observed graphics wrapper; further attempts wait 2/4/8/16/30 seconds. Independent
  mode creates a lost device again, 3 retries at 2/4/8 seconds; a missing API or
  failed device creation waits for Try GPU again. Sentiment retries after 30 seconds.
  No Auto probe or automatic independent/shared switch. GPU-first V3 is the normal preset.
  GPU errors never select CPU automatically. Choose Try GPU again for a new attempt,
  or select CPU for the relevant model. Selecting a GPU in either processing-device
  dropdown tries GPU in the same session, including after an earlier failure.
  The FOMOD selects the GPU route and separate facial/emotion processing devices.
  Both device groups offer GPU 0 (default), GPU 1, GPU 2, GPU 3, or CPU. The
  installer cannot detect adapter names. Select other slots only when you know
  their Windows DXGI indices; use the menu or INI for any higher index. Each
  model saves its choice independently. For recovery without GPU inference,
  select CPU for both facial and emotion processing. Choices persist until changed;
  CPU processing remembers the selected GPU route for later opt-in.
  Choose facial and emotion models, or disable either. Full emotion is offered
  only in with-sentiment. Independent options cover vanilla emotion cues, Potato
  Mode, player animation, early preparation, tongue, eye gaze, logging and ordinary
  Khajiit/Argonian animation. No settings menu is required. Fine strength/timing
  controls and individual NPC/custom-race settings remain in the INI/menu.
  Installer defaults live in BetterFacialAnimations/Installer. Explicit values in
  BetterFacialAnimations.ini override them; menu saves preserve effective choices.
  Back up custom settings before reinstalling. Check MO2 Overwrite for a winning
  INI; deploy changes in Vortex. Manual installation defaults to shared GPU, V3
  and Performance emotion after game entry.
  GPU-SETUP.txt in this folder has the installer guide: Windows/driver and Proton
  requirements, exact CPU recovery INI edits, and how to try GPU from a working CPU session.
  No startup recovery marker is created or checked; old gpu-startup.pending files
  are ignored. If GPU startup hangs/crashes, select CPU for both models, or set
  ForceCPU=1 and EmotionDevice=-1 in the winning INI before restarting.
  A driver call already hanging cannot safely be cancelled inside Skyrim. No helper is launched.
  Old GpuDevice, AutoGpuDevice and UseDeviceWrapper keys are ignored.
  Optional: SKSE Menu Framework, for the in-game settings page (Better Facial Animations -> Settings).

Player dialogue (optional)
  Dragonborn ReVoiced (DBReV) 1.4.4+ with a compatible voice pack animates the player through BFA.
  Select the voice in DBReV's MCM. A third-person/cinematic camera lets you see the player's face.
  General -> Player facial animations toggles this separately from NPC faces; the INI equivalent is
  [General] PlayerFacialAnimations=0. Player animation also works with Potato Mode. The master Enabled setting still applies.
  With Cinematic Conversation Camera, use [Performance] bSynthLipSync=0 in SKSE/Plugins/SD_user.ini
  to disable its synthesized fallback lips. Turning off BFA's player feature leaves DBReV's own lip sync.
  The same supported-head/race requirements apply. DBReV is optional, and its DLL is not patched.

Speech providers (mod authors)
  AI dialogue and TTS mods can animate their own lines through the C interface in
  Data/SKSE/Plugins/BetterFacialAnimations/BFA_API.h (WAV bytes and actor ID,
  playback controls, result codes, per-line emotion and lifecycle events). SE/AE only.
  EndUtterance discards the reply's queued/cached animations and cancels inference.
  Unreleased ABI changed in place; rebuild callers with this header. Version remains 1.

Settings
  Data/SKSE/Plugins/BetterFacialAnimations.ini, or in game through SKSE Menu Framework (saved automatically).
  Male and Female NPCs and the Player each have independent playback, strength and emotion sliders.
  Each profile has ten separate model emotion caps (0..2, default 1), applied after strength and volume swell from the next line. Lower Max joy / happiness to soften smiles independently. Grief, out of breath and pain are not currently fed by Skyrim.
  Smile reaches eyes adds happy-expression motion around the eyes per profile (SmileEyes, 0..2, default 1.0 for all profiles; 0 restores the original pose). Speech uses the line's mean joy and fades with the speech pose; native happy expressions receive the same eye boost. Appearance depends on the head's authored happy morph.
  Upper lip and Lower lip multiply Mouth strength; 1 uses permanent 0.8 upper / 1.2 lower calibration.
  Eye gaze adds speech-driven eye motion when the speaking conversation NPC looks at you (SE/AE).
  Toggle Enable eye gaze in Better Facial Animations -> Settings -> Strength; changes apply immediately.
  It preserves the game's target and stops when the actor looks elsewhere. Enabled by default;
  [LocalAnimationStrength] ConversationGaze=0 disables it, GazeStrength=0..2 adjusts it (default 1).
  VR keeps its existing gaze. Tested offline on original eye meshes; in-game/custom-eye calibration is unverified.
  Older shared settings seed both NPC profiles; older lip values convert to preserve their effect.
  Overrides page: assign an unused keyboard hotkey, close settings, aim at an NPC and press it
  to edit their playback, strength, tongue, gaze and emotion settings. Edits save automatically.
  Sentiment addition multiplier (0..2) overrides the shared amount for that NPC.
  The NPC Sentiment Analysis checkbox turns it off while keeping their multiplier saved.
  Global Sentiment Analysis must be enabled; a zero multiplier also skips it for the NPC.
  Hotkeys require Menu Framework 3 and SE/AE; initially unbound. Escape closes or cancels binding.
  Select NPC under crosshair also works with older frameworks. Edit reopens saved NPC settings.
  Use global clears one control; Reset NPC to global settings or Remove clears all NPC overrides.
  Disable facial animations for an NPC to keep vanilla. Global enable/race restrictions still apply.
  Overrides apply across saves to the NPC base (including copies), survive load-order changes,
  and are preserved by global Defaults. Player and temporary NPC bases cannot be selected.
  Manual configuration (also VR): put Plugin.esp|001234=1 in [Overrides], then create
  [Override:Plugin.esp|001234] with Enabled=0 to disable BFA or settings such as DirectMouth=0.8,
  VoiceEmotion=0.5 and MaxJoy=0.4. Omitted settings inherit the current global profile.
  IDs are six-digit hexadecimal local NPC base IDs; light IDs are padded to six. Restart after edits.
  Missing plugins retain entries. Legacy blacklist entries/hotkey migrate when settings save.
Log
  Documents/My Games/Skyrim Special Edition/SKSE/BetterFacialAnimations.log (Skyrim VR/SKSE/ on VR)
  The GPU, the driver and what went wrong loading the model are always written there: attach it to a bug
  report. Logging (menu Debug section or [Debug] Logging=1, on by default) adds the generated and played lines.

Contents of this folder
  onnxruntime.dll, onnxruntime_providers_shared.dll   ONNX Runtime 1.24.4, DirectML build (MIT)
  DirectML.dll                                         Microsoft DirectML 1.15.4 (licenses/DirectML-LICENSE.txt)
  a2f3_fp16.onnx                                       Audio2Face-3D v3.0, FP16 (NOTICE.txt)
  A2FDirect3.bin                                       its faces (Claire, Mark): basis, landmarks and network projection
  performance.onnx                                   Performance Block AttnRes facial model
  ultra-performance.onnx                             Ultra Performance Thin120 facial model
  emotion_fp16.onnx                                  Full Audio2Emotion (with-sentiment only)
  emotion_performance.onnx, emotion_performance.json  Performance Audio2Emotion and provenance (both archives)
  network_fp16.onnx                                    Audio2Face-3D v2.3 Mark, FP16 (NOTICE.txt): UseOldModel=1
  A2FDirect.bin                                        its face (Mark)
BetterFacialAnimations.dll statically links FFmpeg 8.0.1 (LGPL 2.1): wmav1/wmav2/pcm decoders, xwma/wav demuxers, swresample.
Better Facial Animations plugin source code is free software under the GNU GPL v3 (licenses/BetterFacialAnimations-GPL-3.0.txt).
Additional term under GPL v3 section 7(b): any copy or modified version must keep the attribution "Based on Better Facial Animations by Cyberquach" in its documentation and on any page it is distributed from.
Bundled third-party libraries and NVIDIA models retain their separate licenses; the GPL does not replace those terms.
Audio2Emotion, when included, is governed by licenses/NVIDIA-Audio2Emotion-License.txt: Audio2Face project use only,
not for the purpose of emotion recognition. See NOTICE.txt for model attribution and modifications.
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
