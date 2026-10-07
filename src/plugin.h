// What the plugin's files share: the settings (settings.cc) and the state
// of better_facial_animations.cc that the head cache and the settings page
// use.
#ifndef SRC_PLUGIN_H_
#define SRC_PLUGIN_H_

#include <windows.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "direct.h"

namespace bfa {

extern const char kVersion[];  // from the VERSION file

enum class State { kPending, kReady, kFailed };

// ---------------- settings ----------------
struct Config {  // fixed after load
  bool gpu = true;
  // 3: Audio2Face-3D v3.0 (a2f3_fp16.onnx, A2FDirect3.bin); 2: v2.3 Mark
  // (network_fp16.onnx, A2FDirect.bin)
  int network = 3;
  int batch = 2;
  int log_lines = 400;
  bool use_device_wrapper = true;  // a2f::SetUseDeviceWrapper
  int cache_lines = 300;
  // Winning EFA head assets, detected at data loaded: male, female. Their
  // mouths were made for less speech: that face's Mouth scale is multiplied.
  bool expressive_faces[2] = {};
  std::wstring model, direct;  // the network's files, by network
};
constexpr float kExpressiveMouthScale = 0.85f;
struct FaceTune {
  // how much louder speech adds to a line's emotion
  float voice_swell = 0.25f;
  // s for the game's expressions to fade out when a line starts (the face
  // becomes the model's alone) and back in after it
  float face_in = 0.5f;
  float face_out = 1;
  float time_offset = 0;
  float blend_in = 0.15f;
  float direct_scale[direct::kNumRegions] = {1, 1, 1, 1, 1, 1};
  // how strongly a line's emotion drives the network
  float voice_emotion = 1;
  float upper_lip = 1;
  float lower_lip = 1;
};
struct Tune {  // menu; defaults as dist/BetterFacialAnimations.ini
  bool enabled = true;
  bool player = true;
  bool prefetch = true;
  bool potato_mode = false;
  bool disable_tongue = false;
  // the v2.3 network (Config::network 2) and no GPU session (Config::gpu);
  // they take effect at the next start
  bool old_model = false;
  bool force_cpu = false;
  bool logging = true;
  FaceTune male, female, player_face;
  const FaceTune& ForActor(bool is_female, bool is_player) const {
    return is_player ? player_face : ForSex(is_female);
  }
  const FaceTune& ForSex(bool is_female) const {
    return is_female ? female : male;
  }
};
extern Config& config;
Tune SnapshotTune();  // coherent copy for playback and workers
extern SRWLOCK
    settings_lock;  // guards all Param pointers; readers use SnapshotTune()

// Param pointers may only be accessed with settings_lock held.
// Every ini key / menu item: a slider (value, lo..hi) or, with flag set, a
// checkbox, in a menu group (kModel is in the ini's General section). Table
// order = menu order.
enum class Sec { kGeneral, kModel, kPlayback, kDirect, kEmotion, kDebug };
struct Param {
  const char* key;
  float* value;
  float lo, hi;
  const char* label;
  const char* tip;
  Sec sec;
  bool* flag;
  int profile = -1;  // -1: shared, 0: male NPC, 1: female NPC, 2: player
};
extern const std::vector<Param> kParams;
extern const size_t kNumParams;

void LoadIni();  // at load
void SaveIni();
void ScheduleSaveIni();  // debounce and persist on a dedicated writer thread
void WakeWorker();  // after menu settings change, with settings_lock released
void ResetTune();  // menu defaults
// Turns the log on / off as tune.logging says; a new log starts with the
// version.
void ApplyLogging();
// Whether one more log line of a kind may be written: logging is on and
// fewer than LogLines of the kind (counted in written) were.
bool LogMore(std::atomic<int>& written);

// ---------------- state ----------------
// The network is loaded. Only unrestricted mode keeps generating while
// disabled.
extern std::atomic<bool> net_loaded;
// The A2FDirect bin is loaded and the morph hook in place.
extern std::atomic<bool> ready;
extern direct::Model& model;

double Now();  // s
std::string Lower(const char* s);

struct Stats {
  uint32_t lines = 0, failed = 0, late = 0;
  double audio_sec = 0, gen_sec = 0, last_ms = 0;
  std::string net_info, last_error;
  // The wanted session failed: the diagnosis (problem, cause, advice), the
  // runtime's own message, the attempts so far and when (Now()) the next one
  // is.
  std::vector<std::string> net_error;
  std::string net_detail;
  int net_attempts = 0;
  double net_retry_at = 0;
  bool net_gpu_disabled = false;
  bool net_gpu = false;  // the loaded session is the GPU one
  // Why the plugin is not running (the A2FDirect bin).
  std::string startup_error;
};
// The statistics now, and the lines cached and waiting to be generated.
Stats SnapshotStats(size_t& cached, size_t& queued);
// Forgets the generated lines (not those still being generated).
void ClearLines();

}  // namespace bfa

#endif  // SRC_PLUGIN_H_
