#ifndef SRC_PLAYER_SPEECH_H_
#define SRC_PLAYER_SPEECH_H_

#include <atomic>
#include <memory>
#include <string>

namespace skse {
struct SKSEInterface;
}  // namespace skse

namespace bfa {

// Owned playback notification: adapters never pass borrowed provider data to
// the face system. Cancellation also invalidates a queued main-thread start.
struct PlayerSpeech {
  std::string path;  // relative to Data/Sound
  double started_at = 0;
  float duration = 0;  // audio only, zero if unknown
  std::atomic<bool> stopped{false};
};
using PlayerSpeechPtr = std::shared_ptr<PlayerSpeech>;

// Called on the game thread; uses the same renderer as ordinary actor speech.
void StartPlayerSpeech(PlayerSpeechPtr speech);
// Populated at SKSE PostLoad, after all plugins have registered.
bool DbvoInstalled();
void InitPlayerSpeech(const skse::SKSEInterface* iface);
void PlayerSpeechMessage(unsigned type);

}  // namespace bfa

#endif  // SRC_PLAYER_SPEECH_H_
