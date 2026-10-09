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
  LoadIni();
  CHECK(!config.independent_device);
  write("General", "IndependentGpuDevice", "1");
  LoadIni();
  CHECK(config.independent_device);
  SaveIni();
  LoadIni();
  CHECK(config.independent_device);
  write("General", "IndependentGpuDevice", "0");
  LoadIni();
  CHECK(!config.independent_device);
  CHECK(DeleteFileA(ini));  // keep the migration fixture independent of SaveIni above
  LoadIni();
  CHECK(SnapshotTune().male.upper_lip == 1 &&
        SnapshotTune().female.lower_lip == 1);
  CHECK(SnapshotTune().conversation_gaze && SnapshotTune().gaze_strength == 1);
  write("General", "IniVersion", "3");
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
  CHECK(GetPrivateProfileIntA("General", "IniVersion", 0, ini) == 6);
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
  CHECK(!SnapshotTune().conversation_gaze && SnapshotTune().gaze_strength == 1.5f);
  SaveIni();
  ResetTune();
  CHECK(SnapshotTune().conversation_gaze && SnapshotTune().gaze_strength == 1);
  LoadIni();
  CHECK(!SnapshotTune().conversation_gaze && SnapshotTune().gaze_strength == 1.5f);
  // Every slider has independent storage and survives saving and reloading.
  std::vector<float> expected;
  for (const Param& p : kParams) {
    if (!p.value) continue;
    *p.value = p.lo + (p.hi - p.lo) * (p.profile == 2 ? .5f : p.profile ? .75f : .25f);
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
  // The old model shares geometry, but must still honor sex-specific emotion.
  model.faces.resize(1);
  auto male = Request("profile_test", 1, 50, false, false, t, 0, world_epoch);
  auto female = Request("profile_test", 1, 50, true, false, t, 0, world_epoch);
  CHECK(male != female && queue.size() == 2);
  CHECK(queue[0].emotion.gain == t.male.voice_emotion);
  CHECK(queue[1].emotion.gain == t.female.voice_emotion);
  CHECK(queue[0].emotion.swell == t.male.voice_swell);
  CHECK(queue[1].emotion.swell == t.female.voice_swell);
  CHECK(Request("profile_test", 1, 50, false, false, t, 0, world_epoch) == male);
  auto player = Request("profile_test", 1, 50, false, false, t, 0,
                        world_epoch, true);
  CHECK(player != male && player != female && queue.size() == 3);
  CHECK(queue[2].emotion.gain == t.player_face.voice_emotion);
  CHECK(queue[2].emotion.swell == t.player_face.voice_swell);
  CHECK(Request("profile_test", 1, 50, true, false, t, 0, world_epoch, true)
        == player);  // same old-model geometry and independent player emotion
  // A cap alone separates cached and queued work, for all three profiles.
  Tune caps;
  caps.SelectModel(t.SelectedModel());
  caps.male.max_emotion[a2f::kJoy] = .25f;
  caps.female.max_emotion[a2f::kJoy] = .5f;
  caps.player_face.max_emotion[a2f::kJoy] = .75f;
  auto capped_male = Request("cap_test", 3, 100, false, false, caps, 0, world_epoch);
  auto capped_female = Request("cap_test", 3, 100, true, false, caps, 0, world_epoch);
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
      "MaxAmazement", "MaxAnger", "MaxCheekiness", "MaxDisgust", "MaxFear",
      "MaxGrief", "MaxJoy", "MaxOutOfBreath", "MaxPain", "MaxSadness"};
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
      CHECK(std::fabs(face.max_emotion[c] -
                      (.1f * (c + 1) + .01f * profile)) < 1e-5f);
  }
  write("EmotionMale", "MaxJoy", "nan");
  write("EmotionFemale", "MaxJoy", "-1");
  write("EmotionPlayer", "MaxJoy", "9");
  LoadIni();
  CHECK(SnapshotTune().male.max_emotion[a2f::kJoy] == 2);
  CHECK(SnapshotTune().female.max_emotion[a2f::kJoy] == 0);
  CHECK(SnapshotTune().player_face.max_emotion[a2f::kJoy] == 2);
  ResetTune();
  CHECK(SnapshotTune().male.max_emotion == a2f::kDefaultEmotionCaps);
  CHECK(SnapshotTune().female.max_emotion == a2f::kDefaultEmotionCaps);
  CHECK(SnapshotTune().player_face.max_emotion == a2f::kDefaultEmotionCaps);
  Param const* upper = nullptr;
  for (const Param& p : kParams)
    if (p.value && p.profile == 0 && !strcmp(p.key, "UpperLip")) upper = &p;
  CHECK(upper);
  for (int edit = 0; edit < 20; ++edit) {
    {
      ScopedLock guard(settings_lock);
      *upper->value = 1.f + edit * .05f;
    }
    ScheduleSaveIni();
  }
  // The writer thread's save: the key's value in the file, once it is value
  // (as a number) or 5 s passed.
  auto written = [&](const char* section, const char* key, float value) {
    const ULONGLONG deadline = GetTickCount64() + 5000;
    char saved[32];
    do {
      Sleep(20);
      GetPrivateProfileStringA(section, key, "", saved, sizeof saved, ini);
    } while (!(std::fabs(std::strtof(saved, nullptr) - value) < 1e-5f) &&
             GetTickCount64() < deadline);  // also while it is "nan"
    return std::strtof(saved, nullptr);
  };
  CHECK(std::fabs(written("LocalAnimationStrengthMale", "UpperLip", 1.95f) -
                  1.95f) < 1e-5f);
  // Every switch, flipped in the menu's way, is on disk before the next
  // launch and comes back from it (a switch's tune field, not its ini key).
  for (const Param& p : kParams) {
    if (!p.flag || !strcmp(p.key, "UseV3Model") ||
        !strcmp(p.key, "UsePerformanceModel") || !strcmp(p.key, "UseOldModel"))
      continue;
    const bool flipped = !*p.flag;
    {
      ScopedLock guard(settings_lock);
      *p.flag = flipped;
    }
    ScheduleSaveIni();
    const char* section = p.sec == Sec::kDebug    ? "Debug"
                          : p.sec == Sec::kDirect ? "LocalAnimationStrength"
                                                  : "General";
    CHECK(written(section, p.key, flipped) == flipped);
    *p.flag = !flipped;
    LoadIni();
    CHECK(*p.flag == flipped);
  }
  // The menu helper keeps exactly one model selected, even when unchecking
  // the selected one, and the choice survives save, load and reset.
  for (const char* independent : {"0", "1"}) {
    write("General", "IndependentGpuDevice", independent);
    for (int selected : {4, 2, 3, 4}) {
      const char* key = selected == 4   ? "UsePerformanceModel"
                        : selected == 2 ? "UseOldModel"
                                        : "UseV3Model";
      for (const Param& p : kParams) {
        if (strcmp(p.key, key)) continue;
        ScopedLock guard(settings_lock);
        *p.flag = !*p.flag;
        SelectModelCheckbox(p);
      }
      const Tune selected_tune = SnapshotTune();
      CHECK(selected_tune.SelectedModel() == selected);
      CHECK(int(selected_tune.old_model) + int(selected_tune.v3_model) +
                int(selected_tune.performance_model) ==
            1);
      SaveIni();
      LoadIni();
      CHECK(config.network == selected);
      CHECK(config.independent_device == (independent[0] == '1'));
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
  // UseV3Model is derived, never stored; Performance wins over Old.
  CHECK(GetPrivateProfileIntA("General", "UseV3Model", -1, ini) == -1);
  write("General", "UsePerformanceModel", "1");
  write("General", "UseOldModel", "1");
  write("General", "UseV3Model", "1");
  LoadIni();
  CHECK(SnapshotTune().SelectedModel() == 4 &&
        config.gpu == !SnapshotTune().force_cpu);

  SaveIni();  // joins any current write before removing the test directory
  CHECK(DeleteFileA(ini));
  CHECK(RemoveDirectoryA("Data\\SKSE\\Plugins"));
  CHECK(RemoveDirectoryA("Data\\SKSE"));
  CHECK(RemoveDirectoryA("Data"));
  CHECK(SetCurrentDirectoryA(old) && RemoveDirectoryA(root));
  puts(
      "sex profiles, migration, persistence, defaults and request selection: "
      "pass");
}
