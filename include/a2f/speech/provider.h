// The provider interface (BFA_API.h) inside the plugin. provider.cc owns the
// handles, listeners and the event thread; the plugin
// (better_facial_animations.cc) admits actors, queues inference and plays
// the lines. Each side calls the other's functions declared here; all of
// them may be called from any thread. The plugin's lock is the outer one:
// provider.cc never calls the plugin's side with its own lock held.
#ifndef INCLUDE_A2F_SPEECH_PROVIDER_H_
#define INCLUDE_A2F_SPEECH_PROVIDER_H_

#include <cstdint>

#include "a2f/inference/live_core.h"
#include "a2f/plugin/plugin.h"
#include "a2f/runtime/skse.h"  // namespace skse
#include "a2f/speech/player_speech.h"

namespace bfa {

// ---- the plugin's side ----
// The actor of a form id as the plugin would animate it now.
struct ProviderActor {
  uint8_t* actor = nullptr;
  uint32_t handle = 0;
  uint8_t* fg = nullptr;
  bool female = false, player = false;
  State head_state = State::kPending;  // of its face's mapping, when known
};
// BFA_kResultOK with out filled in, else the BFA_ResultCode.
uint32_t ResolveProviderActor(uint32_t form_id, ProviderActor& out);
// Queues the line's inference: BFA_kResultOK, and ProviderLineFailed follows
// if it fails (at once when the failure is cached); else the BFA_ResultCode.
uint32_t ProviderPrepare(const PlayerSpeechPtr& line, const ProviderActor& who);
// The line takes its actor's face now (BFA_kResultOK), or why it cannot.
uint32_t ProviderStart(const PlayerSpeechPtr& line);
// Discard queued/cached inference owned by the reply; running work cancels.
void ProviderDiscardUtterance(uint64_t utterance);

// ---- provider.cc's side; the plugin's lock may be held ----
void ProviderLineStarted(const PlayerSpeechPtr& line);
void ProviderReset();
void ProviderLineFailed(const PlayerSpeechPtr& line, uint32_t reason);
void ProviderLineDeclined(const PlayerSpeechPtr& line, uint32_t reason);
// Record completion atomically with provider Stop; notify after hand-back.
void ProviderLineStopping(const PlayerSpeechPtr& line, uint32_t stop_reason);
void ProviderLineLeft(const PlayerSpeechPtr& line, uint32_t stop_reason);
bool ProviderClaims(uint32_t form_id);  // an utterance holds the face
// The utterance holding the actor's face between its segments, 0 for none.
uint64_t ProviderHolder(uint32_t form_id);
void InitProvider(const skse::SKSEInterface* iface);
void ProviderMessage(unsigned type);

}  // namespace bfa

#endif  // INCLUDE_A2F_SPEECH_PROVIDER_H_
