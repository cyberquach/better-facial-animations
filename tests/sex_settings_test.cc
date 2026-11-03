// Actual INI migration/persistence and production emotion request selection.
#include <cstdlib>

#include "../src/plugin/better_facial_animations.cc"

#define CHECK(x)                                      \
  do {                                                \
    if (!(x)) {                                       \
      fprintf(stderr, "line %d: %s\n", __LINE__, #x); \
      std::exit(1);                                   \
    }                                                 \
  } while (0)
int main() {
  using namespace bfa;
  char old[MAX_PATH], temp[MAX_PATH], root[MAX_PATH];
  CHECK(GetCurrentDirectoryA(MAX_PATH, old));
  CHECK(GetTempPathA(MAX_PATH, temp));
  CHECK(GetTempFileNameA(temp, "bfa", 0, root));
  CHECK(DeleteFileA(root) && CreateDirectoryA(root, nullptr));
  CHECK(SetCurrentDirectoryA(root));
  CHECK(CreateDirectoryA("Data", nullptr));
  CHECK(CreateDirectoryA("Data\\SKSE", nullptr));
  CHECK(CreateDirectoryA("Data\\SKSE\\Plugins", nullptr));
  const char* ini = ".\\Data\\SKSE\\Plugins\\BetterFacialAnimations.ini";
  auto write = [&](const char* section, const char* key, const char* value) {
    CHECK(WritePrivateProfileStringA(section, key, value, ini));
  };
  // CPU uses the legacy flag and remembers a GPU choice across
  // save/reload/reset.
  LoadIni();
  CHECK(SnapshotTune().gpu_adapter == 0);
  CHECK(SnapshotTune().sentiment_analysis &&
        SnapshotTune().performance_sentiment);
  // Explicit saved choices survive the new Performance default.
  SelectEmotionOption(1);
  SaveIni();
  LoadIni();
  CHECK(SnapshotTune().sentiment_analysis &&
        !SnapshotTune().performance_sentiment);
  SelectEmotionOption(0);
  SaveIni();
  LoadIni();
  CHECK(!SnapshotTune().sentiment_analysis &&
        !SnapshotTune().performance_sentiment);
  SetProcessingDevice(true, 2);
  SaveIni();
  LoadIni();
  CHECK(SnapshotTune().force_cpu && SnapshotTune().gpu_adapter == 2);
  CHECK(config.gpu_adapter == 2);
  ResetTune();
  CHECK(SnapshotTune().force_cpu && SnapshotTune().gpu_adapter == 2);
  SetProcessingDevice(false, 2);
  SaveIni();
  LoadIni();
  CHECK(!SnapshotTune().force_cpu && SnapshotTune().gpu_adapter == 2);
  write("General", "GpuAdapter", "-1");
  LoadIni();
  CHECK(SnapshotTune().gpu_adapter == 0);
  CHECK(DeleteFileA(ini));
  // Retired GPU route keys cannot override the selected device route.
  for (const char* mode : {"0", "1", "2", "3", "99"}) {
    write("General", "GpuDevice", mode);
    write("General", "AutoGpuDevice", mode);
    write("General", "IndependentGpuDevice", "0");
    write("Debug", "UseDeviceWrapper", "1");
    LoadIni();
    SaveIni();
    LoadIni();
    CHECK(config.gpu == !SnapshotTune().force_cpu);
  }
  CHECK(DeleteFileA(
      ini));  // keep the migration fixture independent of SaveIni above
  LoadIni();
  CHECK(
      !config.independent_device);  // a missing INI starts on the shared route
  CHECK(SnapshotTune().male.upper_lip == 1 &&
        SnapshotTune().female.lower_lip == 1);
  CHECK(SnapshotTune().conversation_gaze && SnapshotTune().gaze_strength == 1);
  CHECK(SnapshotTune().vanilla_emotions);
  write("General", "VanillaEmotions", "0");
  LoadIni();
  CHECK(!SnapshotTune().vanilla_emotions);
  SaveIni();
  ResetTune();
  CHECK(SnapshotTune().vanilla_emotions);
  LoadIni();
  CHECK(!SnapshotTune().vanilla_emotions);
  CHECK(DeleteFileA(ini));  // isolate the legacy migration fixture below
  LoadIni();
  write("General", "IniVersion", "3");
  write("General", "SentimentAnalysis", "0");
  write("General", "UsePerformanceEmotion", "0");
  write("General", "UseOldModel", "1");
  write("Debug", "Logging", "0");
  write("LocalAnimationStrength", "DirectMouth", "0.6");
  write("Playback", "TimeOffset", "0.12");
  write("Emotion", "VoiceEmotion", "0.375");
  write("LocalAnimationStrengthFemale", "DirectMouth", "0.8");
  LoadIni();
  auto t = SnapshotTune();
  CHECK(t.male.direct_scale[direct::kMouth] == .6f);
  CHECK(t.female.direct_scale[direct::kMouth] == .8f);
  CHECK(t.male.time_offset == .12f && t.female.time_offset == .12f);
  CHECK(t.male.voice_emotion == .375f && t.female.voice_emotion == .375f);
  CHECK(t.old_model && !t.logging);
  CHECK(GetPrivateProfileIntA("General", "IniVersion", 0, ini) == 9);
  CHECK(t.player_face.upper_lip == 1 && t.player_face.lower_lip == 1);
  CHECK(t.player_face.voice_emotion == 1 && t.player_face.time_offset == 0);
  CHECK(&t.ForActor(false, true) == &t.player_face);
  CHECK(&t.ForActor(true, true) == &t.player_face);
  CHECK(&t.ForActor(false, false) == &t.male);
  CHECK(&t.ForActor(true, false) == &t.female);
  write("General", "IniVersion", "4");
  write("LocalAnimationStrengthMale", "UpperLip", "0.8");
  write("LocalAnimationStrengthFemale", "LowerLip", "1.2");
  LoadIni();
  CHECK(SnapshotTune().male.upper_lip == 1);
  CHECK(SnapshotTune().female.lower_lip == 1);
  LoadIni();  // conversion is applied once, not on each load
  CHECK(SnapshotTune().male.upper_lip == 1);
  CHECK(SnapshotTune().female.lower_lip == 1);
  // Version 7: caps saved at the old default of 2 take the new default of 1;
  // deliberate caps, and caps written at version 7, are kept.
  write("General", "IniVersion", "6");
  write("EmotionMale", "MaxJoy", "2");
  write("EmotionFemale", "MaxAnger", "1.5");
  write("EmotionPlayer", "MaxFear", "2");
  LoadIni();
  CHECK(SnapshotTune().male.max_emotion[a2f::kJoy] == 1);
  CHECK(SnapshotTune().female.max_emotion[a2f::kAnger] == 1.5f);
  CHECK(SnapshotTune().player_face.max_emotion[a2f::kFear] == 1);
  write("EmotionMale", "MaxJoy", "2");
  LoadIni();
  CHECK(SnapshotTune().male.max_emotion[a2f::kJoy] == 2);
  // The shared menu entry must remain a checkbox bound to live gaze tuning.
  const Param* gaze = nullptr;
  for (const Param& p : kParams) {
    if (strcmp(p.key, "ConversationGaze")) continue;
    CHECK(!gaze);
    gaze = &p;
  }
  CHECK(gaze && gaze->flag && !gaze->value);
  CHECK(gaze->sec == Sec::kDirect && gaze->profile == -1);
  *gaze->flag = false;
  CHECK(!SnapshotTune().conversation_gaze);
  *gaze->flag = true;
  CHECK(SnapshotTune().conversation_gaze);
  write("LocalAnimationStrength", "ConversationGaze", "0");
  write("LocalAnimationStrength", "GazeStrength", "1.5");
  LoadIni();
  CHECK(!SnapshotTune().conversation_gaze &&
        SnapshotTune().gaze_strength == 1.5f);
  SaveIni();
  ResetTune();
  CHECK(SnapshotTune().conversation_gaze && SnapshotTune().gaze_strength == 1);
  LoadIni();
  CHECK(!SnapshotTune().conversation_gaze &&
        SnapshotTune().gaze_strength == 1.5f);
  CHECK(SnapshotTune().male.smile_eyes == 1.0f &&
        SnapshotTune().female.smile_eyes == 1.0f &&
        SnapshotTune().player_face.smile_eyes == 1.0f);
  int smile_profiles = 0;
  for (const Param& p : kParams)
    if (!strcmp(p.key, "SmileEyes")) {
      CHECK(p.value && p.sec == Sec::kDirect && p.profile >= 0 &&
            p.profile < 3);
      smile_profiles |= 1 << p.profile;
    }
  CHECK(smile_profiles == 7);
  // Every slider has independent storage and survives saving and reloading.
  std::vector<float> expected;
  for (const Param& p : kParams) {
    if (!p.value) continue;
    *p.value = p.lo + (p.hi - p.lo) * (p.profile == 2 ? .5f
                                       : p.profile    ? .75f
                                                      : .25f);
    expected.push_back(*p.value);
  }
  SaveIni();
  ResetTune();
  CHECK(SnapshotTune().male.upper_lip == 1 &&
        SnapshotTune().female.lower_lip == 1);
  CHECK(SnapshotTune().old_model && !SnapshotTune().logging);
  LoadIni();
  size_t i = 0;
  for (const Param& p : kParams)
    if (p.value) CHECK(std::fabs(*p.value - expected[i++]) < 1e-5f);
  write("LocalAnimationStrengthMale", "UpperLip", "nan");
  write("LocalAnimationStrengthFemale", "LowerLip", "9");
  LoadIni();
  t = SnapshotTune();
  CHECK(t.male.upper_lip == 1 && t.female.lower_lip == 3);
  CHECK(!t.sentiment_analysis);
  // A 4.2 ini with the retired per-profile key is migrated at load.
  write("General", "IniVersion", "7");
  for (const char* section :
       {"Emotion", "EmotionMale", "EmotionFemale", "EmotionPlayer"})
    write(section, "Sentiment", "0");
  LoadIni();
  CHECK(SnapshotTune().sentiment_multiplier == t.sentiment_multiplier);
  SaveIni();
  for (const char* section :
       {"Emotion", "EmotionMale", "EmotionFemale", "EmotionPlayer"}) {
    char legacy[32];
    GetPrivateProfileStringA(section, "Sentiment", "missing", legacy,
                             sizeof legacy, ini);
    CHECK(!strcmp(legacy, "missing"));
  }
  for (const Param& p : kParams) CHECK(strcmp(p.key, "Sentiment"));
  config.sentiment_available = true;
  t.sentiment_analysis = true;
  // The old model shares geometry, but must still honor sex-specific emotion.
  model.faces.resize(1);
  auto male = Request("profile_test", 1, 50, false, false, t, 0, world_epoch);
  auto female = Request("profile_test", 1, 50, true, false, t, 0, world_epoch);
  CHECK(male != female && queue.size() == 2);
  CHECK(queue[0].emotion.gain == t.male.voice_emotion);
  CHECK(queue[1].emotion.gain == t.female.voice_emotion);
  CHECK(queue[0].emotion.swell == t.male.voice_swell);
  CHECK(queue[1].emotion.swell == t.female.voice_swell);
  CHECK(Request("profile_test", 1, 50, false, false, t, 0, world_epoch) ==
        male);
  auto player =
      Request("profile_test", 1, 50, false, false, t, 0, world_epoch, true);
  CHECK(player != male && player != female && queue.size() == 3);
  CHECK(queue[2].emotion.gain == t.player_face.voice_emotion);
  CHECK(queue[2].emotion.swell == t.player_face.voice_swell);
  CHECK(Request("profile_test", 1, 50, true, false, t, 0, world_epoch, true) ==
        player);  // same old-model geometry and independent player emotion
  // A cap alone separates cached and queued work, for all three profiles.
  Tune caps;
  caps.SelectModel(t.SelectedModel());
  caps.male.max_emotion[a2f::kJoy] = .25f;
  caps.female.max_emotion[a2f::kJoy] = .5f;
  caps.player_face.max_emotion[a2f::kJoy] = .75f;
  auto capped_male =
      Request("cap_test", 3, 100, false, false, caps, 0, world_epoch);
  auto capped_female =
      Request("cap_test", 3, 100, true, false, caps, 0, world_epoch);
  auto capped_player =
      Request("cap_test", 3, 100, false, false, caps, 0, world_epoch, true);
  CHECK(capped_male != capped_female && capped_male != capped_player &&
        capped_female != capped_player);
  CHECK(queue[3].emotion.maximum[a2f::kJoy] == .25f);
  CHECK(queue[4].emotion.maximum[a2f::kJoy] == .5f);
  CHECK(queue[5].emotion.maximum[a2f::kJoy] == .75f);
  CHECK(Request("cap_test", 3, 100, true, false, caps, 0, world_epoch, true) ==
        capped_player);
  caps.male.max_emotion[a2f::kJoy] = .251f;
  CHECK(Request("cap_test", 3, 100, false, false, caps, 0, world_epoch) !=
        capped_male);
  // Every model channel participates in the cache key, including unused ones.
  for (int c = 0; c < a2f::kNumMod; ++c) {
    auto before =
        Request("channel_cache", 5, 100, false, false, caps, 0, world_epoch);
    caps.male.max_emotion[c] = .123f + c * .01f;
    auto after =
        Request("channel_cache", 5, 100, false, false, caps, 0, world_epoch);
    CHECK(before != after);
    CHECK(queue.back().emotion.maximum == caps.male.max_emotion);
  }
  // Thirty independent controls bind the named INI keys in network order.
  const char* const kKeys[a2f::kNumMod] = {
      "MaxAmazement", "MaxAnger", "MaxCheekiness",  "MaxDisgust", "MaxFear",
      "MaxGrief",     "MaxJoy",   "MaxOutOfBreath", "MaxPain",    "MaxSadness"};
  for (int profile = 0; profile < 3; ++profile) {
    for (int c = 0; c < a2f::kNumMod; ++c) {
      int found = 0;
      for (const Param& p : kParams) {
        if (p.profile != profile || strcmp(p.key, kKeys[c])) continue;
        CHECK(p.value && !p.flag && p.lo == 0 && p.hi == 2);
        *p.value = .1f * (c + 1) + .01f * profile;
        ++found;
      }
      CHECK(found == 1);
    }
  }
  SaveIni();
  ResetTune();
  LoadIni();
  const auto saved_caps = SnapshotTune();
  for (int profile = 0; profile < 3; ++profile) {
    const auto& face = saved_caps.ForActor(profile == 1, profile == 2);
    for (int c = 0; c < a2f::kNumMod; ++c)
      CHECK(std::fabs(face.max_emotion[c] - (.1f * (c + 1) + .01f * profile)) <
            1e-5f);
  }
  write("EmotionMale", "MaxJoy", "nan");
  write("EmotionFemale", "MaxJoy", "-1");
  write("EmotionPlayer", "MaxJoy", "9");
  LoadIni();
  CHECK(SnapshotTune().male.max_emotion[a2f::kJoy] == 1);
  CHECK(SnapshotTune().female.max_emotion[a2f::kJoy] == 0);
  CHECK(SnapshotTune().player_face.max_emotion[a2f::kJoy] == 2);
  ResetTune();
  CHECK(SnapshotTune().male.max_emotion == a2f::kDefaultEmotionCaps);
  CHECK(SnapshotTune().female.max_emotion == a2f::kDefaultEmotionCaps);
  CHECK(SnapshotTune().player_face.max_emotion == a2f::kDefaultEmotionCaps);
  config.sentiment_available = true;
  CHECK(queue[0].emotion.sentiment_multiplier == t.sentiment_multiplier);
  CHECK(queue[1].emotion.sentiment_multiplier == t.sentiment_multiplier);
  CHECK(queue[2].emotion.sentiment_multiplier == t.sentiment_multiplier);
  t.male.voice_emotion = 0;
  auto sentiment_only =
      Request("profile_test", 1, 50, false, false, t, 0, world_epoch);
  CHECK(sentiment_only != male && queue.back().emotion.gain == 0 &&
        queue.back().emotion.sentiment_multiplier == t.sentiment_multiplier);
  t.sentiment_analysis = false;
  auto off = Request("profile_test", 1, 50, false, false, t, 0, world_epoch);
  CHECK(off != sentiment_only &&
        queue.back().emotion.sentiment_multiplier == 0);
  t.sentiment_analysis = true;
  t.sentiment_multiplier = .5f;
  auto scaled = Request("profile_test", 1, 50, false, false, t, 0, world_epoch);
  CHECK(scaled != sentiment_only &&
        queue.back().emotion.sentiment_multiplier == .5f);
  t.sentiment_multiplier = .501f;
  CHECK(Request("profile_test", 1, 50, false, false, t, 0, world_epoch) !=
        scaled);
  // The shared source toggle suppresses both native and explicit authored
  // inputs while preserving independent sentiment and the configured limits.
  Tune sources;
  sources.performance_sentiment = false;  // Full-model source fixture
  sources.sentiment_analysis = true;
  auto authored =
      Request("source_toggle", 1, 50, false, false, sources, 0, world_epoch);
  CHECK(authored && queue.back().emotion.gain == 1);
  sources.vanilla_emotions = false;
  auto without_authored =
      Request("source_toggle", 1, 50, false, false, sources, 0, world_epoch);
  CHECK(without_authored != authored && queue.back().emotion.gain == 0 &&
        queue.back().emotion.type == 0 && queue.back().emotion.swell == 0);
  CHECK(queue.back().emotion.sentiment_multiplier == 1 &&
        queue.back().emotion.maximum == sources.male.max_emotion);
  a2f::Emotion provider;
  provider.type = a2f::kExplicitEmotion;
  provider.channels[a2f::kAnger] = .75f;
  CHECK(Request("provider_source_toggle", 0, 0, false, false, sources, 0,
                world_epoch, false, &provider));
  CHECK(queue.back().emotion.gain == 0 &&
        queue.back().emotion.channels[a2f::kAnger] == 0);
  sources.sentiment_analysis = false;
  CHECK(Request("both_sources_off", 1, 50, false, false, sources, 0,
                world_epoch));
  CHECK(queue.back().emotion.gain == 0 &&
        queue.back().emotion.sentiment_multiplier == 0);
  sources.vanilla_emotions = true;
  CHECK(Request("provider_source_toggle", 0, 0, false, false, sources, 0,
                world_epoch, false, &provider));
  CHECK(queue.back().emotion.gain == 1 &&
        queue.back().emotion.channels[a2f::kAnger] == .75f);
  Param const* upper = nullptr;
  for (const Param& p : kParams)
    if (p.value && p.profile == 0 && !strcmp(p.key, "UpperLip")) upper = &p;
  CHECK(upper);
  for (int edit = 0; edit < 20; ++edit) {
    {
      ScopedLock guard(settings_lock);
      *upper->value = 1.f + edit * .05f;
    }
    SaveIni();
  }
  // The save call must finish writing before it returns.
  auto written = [&](const char* section, const char* key) {
    char saved[32];
    GetPrivateProfileStringA(section, key, "", saved, sizeof saved, ini);
    return std::strtof(saved, nullptr);
  };
  CHECK(std::fabs(written("LocalAnimationStrengthMale", "UpperLip") - 1.95f) <
        1e-5f);
  // Every switch, flipped in the menu's way, is on disk before the next
  // launch and comes back from it (a switch's tune field, not its ini key).
  for (const Param& p : kParams) {
    if (!p.flag || !strcmp(p.key, "UsePerformanceModel") ||
        !strcmp(p.key, "UseOldModel"))
      continue;
    const bool flipped = !*p.flag;
    {
      ScopedLock guard(settings_lock);
      *p.flag = flipped;
    }
    SaveIni();
    const char* section = p.sec == Sec::kDebug    ? "Debug"
                          : p.sec == Sec::kDirect ? "LocalAnimationStrength"
                                                  : "General";
    CHECK(written(section, p.key) == flipped);
    *p.flag = !flipped;
    LoadIni();
    CHECK(*p.flag == flipped);
  }
  // The model dropdown keeps exactly one model selected, and the choice
  // survives save, load and reset.
  for (const char* independent : {"0", "1"}) {
    write("General", "IndependentGpuDevice", independent);
    for (int selected : {4, 2, 3, 4}) {
      SelectFaceOption(selected);
      const Tune selected_tune = SnapshotTune();
      CHECK(selected_tune.SelectedModel() == selected);
      CHECK(int(selected_tune.old_model) +
                int(selected_tune.performance_model) ==
            (selected != a2f::kNetV3));
      SaveIni();
      LoadIni();
      CHECK(config.network == selected);
      CHECK(config.gpu == !SnapshotTune().force_cpu);
      CHECK(config.direct ==
            (selected == 2 ? L"A2FDirect.bin" : L"A2FDirect3.bin"));
      ResetTune();
      CHECK(SnapshotTune().SelectedModel() == selected);
    }
  }
  // Each network loads the mapping its lines are made for.
  CHECK(ModelFor(a2f::kNetPerformance).mapping_network == a2f::kNetV3 &&
        ModelFor(a2f::kNetV3).mapping_network == a2f::kNetV3 &&
        ModelFor(a2f::kNetV2).mapping_network == a2f::kNetV2);
  CHECK(std::wstring(ModelFor(a2f::kNetPerformance).mapping_file) ==
            ModelFor(a2f::kNetV3).mapping_file &&
        std::wstring(ModelFor(a2f::kNetPerformance).model_file) ==
            L"performance.onnx");
  // V3 is the absence of both keys; UseV3Model is ignored and never
  // stored. Performance wins over Old.
  CHECK(GetPrivateProfileIntA("General", "UseV3Model", -1, ini) == -1);
  write("General", "UsePerformanceModel", "1");
  write("General", "UseOldModel", "1");
  write("General", "UseV3Model", "1");
  LoadIni();
  CHECK(SnapshotTune().SelectedModel() == 4 &&
        config.gpu == !SnapshotTune().force_cpu);

  SaveIni();
  CHECK(DeleteFileA(ini));
  CHECK(RemoveDirectoryA("Data\\SKSE\\Plugins"));
  CHECK(RemoveDirectoryA("Data\\SKSE"));
  CHECK(RemoveDirectoryA("Data"));
  CHECK(SetCurrentDirectoryA(old) && RemoveDirectoryA(root));
  puts(
      "sex profiles, migration, persistence, defaults and request selection: "
      "pass");
}
