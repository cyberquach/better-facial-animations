// Actual INI migration/persistence and production emotion request selection.
#include <cstdlib>

#include "../src/better_facial_animations.cc"

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
  CHECK(SnapshotTune().male.upper_lip == 1 &&
        SnapshotTune().female.lower_lip == 1);
  CHECK(config.use_device_wrapper);
  write("Debug", "UseDeviceWrapper", "0");
  LoadIni();
  CHECK(config.use_device_wrapper);  // Old shipped default is not a new opt-in.
  write("Debug", "UnwrapDevice", "1");
  LoadIni();
  CHECK(!config.use_device_wrapper);
  write("Debug", "UnwrapDevice", "0");
  LoadIni();
  CHECK(config.use_device_wrapper);
  CHECK(DeleteFileA(ini));  // Separate fixture for legacy slider migration.
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
  CHECK(GetPrivateProfileIntA("General", "IniVersion", 0, ini) == 5);
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
  const ULONGLONG deadline = GetTickCount64() + 5000;
  char saved[32];
  do {
    Sleep(20);
    GetPrivateProfileStringA("LocalAnimationStrengthMale", "UpperLip", "",
                             saved, sizeof saved, ini);
  } while (!(std::fabs(std::strtof(saved, nullptr) - 1.95f) < 1e-5f) &&
           GetTickCount64() < deadline);
  CHECK(std::fabs(std::strtof(saved, nullptr) - 1.95f) < 1e-5f);
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
