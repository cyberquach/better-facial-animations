#ifndef INCLUDE_A2F_SPEECH_PLAYER_SPEECH_H_
#define INCLUDE_A2F_SPEECH_PLAYER_SPEECH_H_

#include <windows.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "a2f/inference/live_core.h"

namespace skse {
struct SKSEInterface;
}  // namespace skse

namespace bfa {

// Owned playback notification: adapters never pass borrowed provider data to
// the face system. Cancellation also invalidates a queued main-thread start.
// DBReV and DBVO lines carry a path and a start; a provider interface line
// (BFA_API.h, provider.cc) also carries its handle, audio and clock.
struct PlayerSpeech {
  std::string path;  // relative to Data/Sound; a provider's relative to Data
  double started_at = 0;
  float duration = 0;  // audio only, zero if unknown
  std::atomic<bool> stopped{false};

  // ---- provider interface lines; handle 0 otherwise ----
  uint64_t handle = 0, utterance_id = 0;
  uint32_t segment = 0;
  uint32_t form_id = 0;             // the speaker's
  // provider RIFF WAVE bytes; DBVO/DBReV use path instead
  std::shared_ptr<const std::vector<uint8_t>> audio;
  std::string stem;  // the line's name in the cache and the log
  // A provider line's emotion (explicit channels; neutral when none given).
  a2f::Emotion emotion;
  // Pending terminal event; accessed only by provider.cc under provider_lock.
  uint32_t end_event = 1;  // BFA_kEnded
  uint32_t end_reason = 0;
  std::atomic<bool> play_requested{false}, animation_started{false}, expired{false};
  std::atomic<bool> on_face{false};   // Start was accepted; cleared when it left
  std::atomic<bool> reported{false};  // its end (ended, failed, declined) was
  std::atomic<double> touched{0};     // the provider's last call about it

  // The audio's position: `at` seconds at wall clock `now`, counting on
  // from there unless paused.
  void Place(double now, double at, bool pause) {
    AcquireSRWLockExclusive(&clock_lock);
    anchor = now;
    offset = at;
    paused = pause;
    ReleaseSRWLockExclusive(&clock_lock);
  }
  double Position(double now) const {
    AcquireSRWLockShared(&clock_lock);
    const double t = paused ? offset : offset + (now - anchor);
    ReleaseSRWLockShared(&clock_lock);
    return t < 0 ? 0 : t;
  }
  bool Paused() const {
    AcquireSRWLockShared(&clock_lock);
    const bool p = paused;
    ReleaseSRWLockShared(&clock_lock);
    return p;
  }

 private:
  mutable SRWLOCK clock_lock = SRWLOCK_INIT;
  double anchor = 0, offset = 0;
  bool paused = false;
};
using PlayerSpeechPtr = std::shared_ptr<PlayerSpeech>;

// Called on the game thread; uses the same renderer as ordinary actor speech.
void StartPlayerSpeech(PlayerSpeechPtr speech);
// Populated at SKSE PostLoad, after all plugins have registered.
bool DbvoInstalled();
void InitPlayerSpeech(const skse::SKSEInterface* iface);
void PlayerSpeechMessage(unsigned type);

}  // namespace bfa

#endif  // INCLUDE_A2F_SPEECH_PLAYER_SPEECH_H_
