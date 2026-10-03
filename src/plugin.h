// What the plugin's files share: the settings (settings.cc) and the state
// of better_facial_animations.cc that the head cache and the settings page
// use.
#ifndef SRC_PLUGIN_H_
#define SRC_PLUGIN_H_

#include <windows.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>
#include <utility>
#include <vector>

#include "direct.h"

namespace bfa {

extern const char kVersion[];  // from the VERSION file

enum class State { kPending, kReady, kFailed };

// ---------------- settings ----------------
struct Config {  // fixed after load
  bool gpu = true;
  int batch = 2;
  int log_lines = 400;
  int cache_lines = 300;
  std::wstring model;
};
struct Tune {  // menu; defaults as dist/BetterFacialAnimations.ini
  bool enabled = true;
  bool prefetch = true;
  bool logging = false;
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
};
extern Config& config;
Tune SnapshotTune();  // coherent copy for playback and workers
extern SRWLOCK
    settings_lock;  // guards all Param pointers; readers use SnapshotTune()

// Param pointers may only be accessed with settings_lock held.
// Every ini key / menu item: a slider (value, lo..hi) or, with flag set, a
// checkbox, in an ini section / menu group. Table order = menu order.
enum class Sec { kGeneral, kPlayback, kDirect, kEmotion, kDebug };
struct Param {
  const char* key;
  float* value;
  float lo, hi;
  const char* label;
  const char* tip;
  Sec sec;
  bool* flag;
};
extern const Param kParams[];
extern const size_t kNumParams;

void LoadIni();  // at load
void SaveIni();
void ResetTune();  // menu defaults
// Turns the log on / off as tune.logging says; a new log starts with the
// version.
void ApplyLogging();
// Whether one more log line of a kind may be written: logging is on and
// fewer than LogLines of the kind (counted in written) were.
bool LogMore(std::atomic<int>& written);

// ---------------- state ----------------
// The network is loaded: lines keep generating while disabled (instant A/B).
extern std::atomic<bool> net_loaded;
// A2FDirect.bin is loaded and the morph hook in place.
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
  bool net_gpu = false;  // the loaded session is the GPU one
  // Why the plugin is not running (A2FDirect.bin).
  std::string startup_error;
  // The last network events (loads, failures, losses) and when (Now()).
  std::deque<std::pair<double, std::string>> events;
};
// The statistics now, and the lines cached and waiting to be generated.
Stats SnapshotStats(size_t& cached, size_t& queued);
// Forgets the generated lines (not those still being generated).
void ClearLines();

}  // namespace bfa

#endif  // SRC_PLUGIN_H_
