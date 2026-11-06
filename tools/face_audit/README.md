# Local face regression audit

Read-only comparison against a retained `direct.cc` baseline. Requires Python
with NumPy and lz4, a native C++17 compiler, local Skyrim/MO2 assets and recorded
BFA coefficient clips. Does not download, install or launch the game.

```sh
python tools/face_audit/run.py --game /path/to/Skyrim \
  --profile 'MO2 profile name' --baseline /path/to/baseline.cc \
  --stimuli /path/to/stimuli.json --output /path/to/new-evidence \
  --installed --full --jobs 4
```

`stimuli.json` maps model `2`/`3`, then sex `0` (male)/`1` (female), to coefficient
file paths. Each clip is little-endian `u32 frames, u32 channels, f32 fps` followed
by `frames * channels` float32 coefficients. Use absolute paths. The baseline
uses the current public header unless `--baseline-include /path/to/include`
supplies its matching include tree. Sources and both header trees are copied
into the evidence directory before compilation. Reusing an output containing
results is rejected. `--reuse-inventory` reuses a prior `records.json` and optional
`installed.json`; active plugin hashes are checked before using cached records.
Refresh the inventory when installed assets change.

`--strength-sweep` additionally replays every unique changed head, including
heads with an extended lip control envelope, at upper/lower strengths 0/1, 2/1, 1/0, 1/2 and 2/2, with
native MoodHappy at 0.7. These nondefault surface screens remain review gates,
including where corrected control semantics deliberately change the motion.

The audit enumerates active FaceGen candidates and, with `--installed`, every
installed source variant (including inactive mods). It resolves expression TRI
files, checks vertex counts, evaluates both models and records source hashes.
Inactive variants use their own morph assets where present and otherwise the
active profile's assets; these are hypothetical combinations, not certified
alternative load orders. NPC/template coverage is reported separately.

Gates compare accepted fits, finite speech, opening, per-frame projected penetration,
bit-identical authored expressions and unchanged speech outside the declared calibration region.
Speech is screened both without the jaw cap and with it synthetically applied
to all weighted lower-lip vertices. Skin-only inputs lack oral seam shapes, so
this stress case does not establish which NPCs use that cap in game.
Aperture pairs come from authored TRI geometry and opening, independently of the
mapper's transferred weights. Upper-mouth TRI edges also screen surface continuity:
flag either a rise in the 95th-percentile maximum edge strain (>0.001) or an
increase in edges stretching beyond twice their resting length. Missing or
inconsistent edge measurements also require review. The anatomical
region comes from native opening and independently sampled mouth landmarks.
This catches a repaired/untouched split that a larger aperture alone misses. They are screening measurements, not ground-truth
lip anatomy or full triangle collision. `--full` uses complete recorded clips;
otherwise it uses deterministic samples. No speech evaluation is needed when
the skin playback arrays are byte-identical and neither mapping retains legacy
repair/contact state. Current calibration is baked into those arrays; legacy
contact-only changes still require speech replay. Expressions
are always evaluated.

`summary.json`, per-asset JSONL, flagged rows and NPC coverage retain the result.
Exit status: **0** complete pass, **1** regression review required, **2** coverage
blocked. Missing assets, unresolved morphs and runtime-generated heads are never
counted as verified. In-game appearance, other shapes and simultaneous native
expression/speech combinations need separate validation; use the follower-teeth,
mouth-lining and direct-mapping contracts alongside this audit.
