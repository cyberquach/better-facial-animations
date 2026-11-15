# Changelog

## 7.0.0 (changes since 6.1.0)

Major version: the default face model and the GPU memory behaviour change for
everyone who upgrades. Existing INI model choices are kept.

### Changed

- **Performance is the new default face model.** It is now a 22M-parameter
  student (width 608), replacing the 4M one. It is trained to reproduce V3 on
  the training dialogue and runs at 30 fps on GPU or CPU, using about 185 MB of
  VRAM. It is listed first and recommended in the installer (`UsePerformanceModel=1`).
  Unfamiliar dialogue can still be less reliable than V3. An INI that sets any
  model switch keeps its model, and the switches it leaves out count as off;
  one that sets none gets Performance.
- **DirectML graph fusion is off by default**, which lowers VRAM use: V3 drops
  from about 410 to 270 MB, Performance from 315 to 185 MB, Full emotion from
  1,340 to 745 MB and Performance emotion from 52 to 31 MB. Generation is
  1–13% slower, still several times faster than speech. Two new settings bring
  back the earlier behaviour:
  `GraphFusion` (Audio2Face) and `EmotionGraphFusion` (Audio2Emotion). Each is
  in the installer and is a checkbox under its model's processing device in the
  menu. A change rebuilds only that model, live; the setting does nothing on CPU.
- **Dialogue lines are generated first.** Lines of the player and of the NPC
  you are in conversation with go before other lines, and spoken lines before
  ones prepared ahead. A spoken line waiting behind a prefetch stops it
  between runs of the network and goes first; the prefetch is generated again
  after it. The inference thread runs above normal priority while it
  generates one of these lines.
- **A line's speech shows from its first frame.** A face at rest showed a
  line's generated speech faded in over `FaceFadeIn` (0.5 s), so the first
  half second looked vanilla. A face at rest now takes it at once; a face
  still posed (a line before it) or a line ready late blends over `BlendIn`.
  `FaceFadeIn` now fades only the game's own expressions.
- The Performance model's weights are stored as float16: `performance.onnx` is
  48 MB instead of 93 MB, and its output differs by at most 0.016 mm.
- The VRAM figures in the INI, menu, README, GPU-SETUP and installer are now
  measured values from an RTX 5070 Ti, replacing the earlier estimates.

### Fixed

- Cheek and jaw-side hair of vanilla-shaped beards (e.g. Faithful Faces) no
  longer drops through the cheek. Facial hair now follows the skin it lies on;
  hair that hangs clear of the skin keeps the jaw/upper-face rule.
- Lines from a loose WAV that a mod rewrites under the same name (Mantella) are
  generated again each time they are spoken, so a speaker no longer gets an
  earlier line's animation; Fuz lines stay cached. A muted line is animated
  from an audible copy of the same length beside it, `<line>.bfa.wav`
  (Mantella's `bfa_facial_animation` option).
- Legacy is offered in the in-game Audio2Face dropdown again. A Legacy choice
  made in the installer or INI was shown as Disabled. *(6.1.2)*
- Heads with a static part that carries morph data are animated again. Only
  that part is hidden from the engine's face morphs, not the whole head.
  *(6.1.1)*
- Another plugin that patches a shared call site after BFA and calls BFA's hook
  from its own code (e.g. with SMP Fixes in front) now reaches the plugins
  chained behind it. Expression writes from mods such as PPA reach the face
  again. *(6.1.1)*
- A line generated without its sentiment, because Audio2Emotion was turned off
  while the sentiment was read, is now marked degraded and generated again
  once sentiment is back.
- Gaze-beat moods no longer lose the sentiment at the end of clips whose length
  is not a whole number of frames.

### Internal

- Each face model, profile, and per-channel/landmark/mood table now comes from
  a single definition, with the counts checked by the compiler and by tests.
- New compatibility contracts: model texts against
  `performance-model.json`, beard clearance over 51 head captures, and the
  installer flags.
- Lines log how long they waited in the queue, for sentiment, the face
  session and the model, and how long after its speech began each line was
  first shown (vanilla heard, by cause); a summary every 20 lines.
- Training checkpoints are kept out of Git; the exported ONNX is the tracked
  model.
