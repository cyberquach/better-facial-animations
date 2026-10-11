// The small event-driven speech provider API. Engine access is queued on SKSE.
//
// Locks, outermost first: `calls` (serialises the provider's engine-bound
// calls), the plugin's lock, `provider_lock`, `events_lock`. Nothing here
// calls the plugin with provider_lock held; the plugin calls the functions
// below with its lock held.
#include "a2f/speech/provider.h"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "a2f/runtime/scoped_lock.h"
#include "a2f/runtime/skse.h"
#include "a2f/speech/BFA_API.h"

namespace bfa {
namespace {
SRWLOCK provider_lock = SRWLOCK_INIT;
std::recursive_mutex calls;
auto& lines = *new std::unordered_map<uint64_t, PlayerSpeechPtr>;
struct Utterance {
  uint32_t actor;
  double touched;
  bool started = false;
};
auto& utterances = *new std::unordered_map<uint64_t, Utterance>;
struct Listener {
  BFA_EventCallback fn;
  void* context;
};
auto& listeners = *new std::unordered_map<uint64_t, Listener>;
uint64_t next_handle = 0, next_listener = 0;
std::atomic<uint64_t> next_utterance{0};
skse::SKSETaskInterface* tasks = nullptr;
skse::SKSEMessagingInterface* messaging = nullptr;
uint32_t plugin_handle = 0;
SRWLOCK events_lock = SRWLOCK_INIT;
CONDITION_VARIABLE events_cv = CONDITION_VARIABLE_INIT;
auto& events = *new std::deque<BFA_Event>;
HANDLE event_thread = nullptr;
constexpr double kExpiry = 120;
constexpr size_t kCapacity = 256;
constexpr uint32_t kMinWaveBytes = 12;  // "RIFF" size "WAVE"

// provider_lock held. Subscribers receive events in lifecycle order.
void Post(uint32_t event, uint32_t reason, const PlayerSpeechPtr& line) {
  if (listeners.empty()) return;
  BFA_Event e{BFA_API_VERSION,    sizeof(BFA_Event), event,
              reason,             line->handle,      line->utterance_id,
              line->segment,      line->form_id};
  ScopedLock guard(events_lock);
  events.push_back(e);
  WakeConditionVariable(&events_cv);
}
// provider_lock held. End is idempotent, including late inference callbacks.
void Finish(const PlayerSpeechPtr& line, uint32_t event, uint32_t reason) {
  if (line->reported.exchange(true)) return;
  line->stopped = true;
  line->on_face = false;
  Post(event, reason, line);
  lines.erase(line->handle);
}
// provider_lock held. The first terminal cause wins until expiry/reset.
void RecordEnd(const PlayerSpeechPtr& line, uint32_t event, uint32_t reason) {
  if (line->stopped) return;
  line->end_event = event;
  line->end_reason = reason;
  line->stopped = true;
}
void RecordStop(const PlayerSpeechPtr& line, uint32_t reason) {
  RecordEnd(line, reason == BFA_kSuperseded ? BFA_kOwnerChanged : BFA_kEnded,
            reason == BFA_kSuperseded ? BFA_kResultTakenByOtherLine : reason);
}
// provider_lock held. Playback may still be fading after it stops.
void FinishPending(const PlayerSpeechPtr& line) {
  if (line->expired)
    Finish(line, BFA_kExpired, BFA_kResultExpired);
  else
    Finish(line, line->end_event, line->end_reason);
}
PlayerSpeechPtr Find(uint64_t handle) {
  ScopedLock guard(provider_lock, ScopedLock::kShared);
  const auto it = lines.find(handle);
  return it == lines.end() ? nullptr : it->second;
}
// provider_lock held.
void Touch(const PlayerSpeechPtr& line) {
  line->touched = Now();
  auto it = utterances.find(line->utterance_id);
  if (it != utterances.end()) it->second.touched = line->touched;
}
// Never calls the plugin while holding provider_lock. Expiry marks stopped so
// a queued engine task cannot revive the handle; the renderer drops stopped
// lines.
void Expire() {
  std::lock_guard<std::recursive_mutex> serial(calls);
  std::vector<uint64_t> discard;
  {
    ScopedLock guard(provider_lock);
    const double now = Now();
    std::vector<PlayerSpeechPtr> expired;
    for (const auto& [handle, line] : lines)
      if (now - line->touched >= kExpiry) expired.push_back(line);
    for (const auto& line : expired) {
      // The renderer sends the terminal event after its last write.
      RecordStop(line, BFA_kAborted);
      line->expired = true;
      line->audio.reset();
      if (!line->on_face) Finish(line, BFA_kExpired, BFA_kResultExpired);
    }
    for (auto it = utterances.begin(); it != utterances.end();) {
      if (now - it->second.touched >= kExpiry) {
        discard.push_back(it->first);
        it = utterances.erase(it);
      } else {
        ++it;
      }
    }
  }
  for (uint64_t id : discard) ProviderDiscardUtterance(id);
}
DWORD WINAPI EventThread(void*) {
  for (;;) {
    Expire();
    BFA_Event e{};
    {
      ScopedLock guard(events_lock);
      if (events.empty())
        SleepConditionVariableSRW(&events_cv, &events_lock, 1000, 0);
      if (events.empty()) continue;
      e = events.front();
      events.pop_front();
    }
    std::vector<Listener> copy;
    {
      ScopedLock guard(provider_lock, ScopedLock::kShared);
      for (const auto& [id, listener] : listeners) copy.push_back(listener);
    }
    for (const auto& listener : copy) listener.fn(&e, listener.context);
  }
}
// provider_lock held.
bool EnsureThread() {
  if (!event_thread)
    event_thread = CreateThread(nullptr, 0, EventThread, nullptr, 0, nullptr);
  return event_thread != nullptr;
}
// The engine-bound half of Prepare (admission and inference) or Start (the
// face), on the game thread.
struct LineTask final : skse::TaskDelegate {
  PlayerSpeechPtr line;
  bool prepare;
  LineTask(PlayerSpeechPtr l, bool p) : line(std::move(l)), prepare(p) {}
  void Run() override {
    std::lock_guard<std::recursive_mutex> serial(calls);
    if (line->stopped) return;
    ProviderActor who;
    uint32_t result = ResolveProviderActor(line->form_id, who);
    if (result == BFA_kResultOK && who.head_state == State::kFailed)
      result = BFA_kResultHeadMapFailed;
    if (result != BFA_kResultOK) {
      ProviderLineDeclined(line, result);
      return;
    }
    if (prepare) {
      result = ProviderPrepare(line, who);
      if (result != BFA_kResultOK) ProviderLineFailed(line, result);
      return;
    }
    result = ProviderStart(line);
    if (result != BFA_kResultOK) {
      ProviderLineDeclined(line, result);
      return;
    }
    ScopedLock guard(provider_lock);
    if (!line->stopped) line->on_face = true;
  }
  void Dispose() override { delete this; }
};
// The line's emotion as the network takes it: explicit channels, neutral
// when the provider gives none.
bool ReadEmotion(const BFA_Emotion* in, a2f::Emotion& out) {
  out.type = a2f::kExplicitEmotion;
  if (!in) return true;
  if (in->version != BFA_API_VERSION || in->size < sizeof(*in)) return false;
  for (size_t i = 0; i < BFA_kNumEmotionChannels; ++i) {
    const float v = in->channels[i];
    if (!std::isfinite(v) || v < 0 || v > 2) return false;
    out.channels[i] = v;
  }
  return true;
}
bool ValidLine(const BFA_Line* in, a2f::Emotion& emotion) {
  if (!in || in->version != BFA_API_VERSION || in->size < sizeof(*in) ||
      !in->speaker_form_id || !in->utterance_id || !in->audio ||
      in->audio_bytes < kMinWaveBytes || in->audio_bytes > BFA_MAX_AUDIO_BYTES)
    return false;
  const auto* bytes = static_cast<const uint8_t*>(in->audio);
  return !memcmp(bytes, "RIFF", 4) && !memcmp(bytes + 8, "WAVE", 4) &&
         ReadEmotion(in->emotion, emotion);
}
// The line's name in the cache and the log: its reply and its audio's hash.
std::string ProviderStem(uint64_t utterance,
                         const std::vector<uint8_t>& audio) {
  uint64_t hash = 1469598103934665603ull;
  for (uint8_t byte : audio) hash = (hash ^ byte) * 1099511628211ull;
  char stem[64];
  snprintf(stem, sizeof stem, "provider:%016llx:%016llx",
           static_cast<unsigned long long>(utterance),
           static_cast<unsigned long long>(hash));
  return stem;
}
uint64_t NewUtteranceId() { return ++next_utterance; }
uint64_t Prepare(const BFA_Line* in, BFA_Result* result) {
  if (!result) return 0;
  *result = BFA_kResultInvalidArgument;
  a2f::Emotion emotion;
  if (!ValidLine(in, emotion)) return 0;
  std::lock_guard<std::recursive_mutex> serial(calls);
  if (!tasks || !tasks->AddTask) {
    *result = BFA_kResultUnavailable;
    return 0;
  }
  const auto* bytes = static_cast<const uint8_t*>(in->audio);
  auto line = std::make_shared<PlayerSpeech>();
  line->utterance_id = in->utterance_id;
  line->segment = in->segment;
  line->form_id = in->speaker_form_id;
  line->emotion = emotion;
  line->audio = std::make_shared<const std::vector<uint8_t>>(
      bytes, bytes + in->audio_bytes);
  line->stem = ProviderStem(in->utterance_id, *line->audio);
  line->Place(Now(), 0, true);
  {
    ScopedLock guard(provider_lock);
    if (in->utterance_id > next_utterance) {  // not ours
      *result = BFA_kResultNotFound;
      return 0;
    }
    auto it = utterances.find(in->utterance_id);
    if (lines.size() >= kCapacity ||
        (utterances.size() >= kCapacity && it == utterances.end())) {
      *result = BFA_kResultCapacity;
      return 0;
    }
    // A reply has one speaker.
    if (it != utterances.end() && it->second.actor != in->speaker_form_id)
      return 0;
    if (!EnsureThread()) {
      *result = BFA_kResultInternalError;
      return 0;
    }
    if (it == utterances.end())
      utterances.emplace(in->utterance_id, Utterance{line->form_id, Now()});
    line->handle = ++next_handle;
    Touch(line);
    lines[line->handle] = line;
  }
  tasks->AddTask(new LineTask(line, true));
  *result = BFA_kResultPending;
  return line->handle;
}
BFA_Result Start(uint64_t handle, double offset) {
  if (!std::isfinite(offset) || offset < 0) return BFA_kResultInvalidArgument;
  std::lock_guard<std::recursive_mutex> serial(calls);
  auto line = Find(handle);
  if (!line) return BFA_kResultInvalidHandle;
  {
    ScopedLock guard(provider_lock);
    if (line->stopped) return BFA_kResultInvalidState;
    Touch(line);
    line->Place(Now(), offset, false);
    if (line->play_requested) return BFA_kResultOK;  // a second Start seeks
    line->play_requested = true;
    line->started_at = Now();
    auto it = utterances.find(line->utterance_id);
    if (it != utterances.end()) it->second.started = true;
  }
  tasks->AddTask(new LineTask(line, false));
  return BFA_kResultPending;
}
// The started line's audio clock, as the provider reports it.
BFA_Result Clock(uint64_t handle, double offset, bool pause) {
  if (!std::isfinite(offset) || offset < 0) return BFA_kResultInvalidArgument;
  ScopedLock guard(provider_lock);
  auto it = lines.find(handle);
  if (it == lines.end()) return BFA_kResultInvalidHandle;
  auto line = it->second;
  if (line->stopped || !line->play_requested) return BFA_kResultInvalidState;
  Touch(line);
  const double now = Now();
  if (pause) {
    const bool was_paused = line->Paused();
    line->Place(now, line->Position(now), true);
    return was_paused ? BFA_kResultNoChange : BFA_kResultOK;
  }
  line->Place(now, offset, false);
  return BFA_kResultOK;
}
BFA_Result Pause(uint64_t handle) { return Clock(handle, 0, true); }
BFA_Result Seek(uint64_t handle, double offset) {
  return Clock(handle, offset, false);
}
BFA_Result Stop(uint64_t handle, uint32_t reason) {
  if (reason > BFA_kAborted) return BFA_kResultInvalidArgument;
  std::lock_guard<std::recursive_mutex> serial(calls);
  ScopedLock guard(provider_lock);
  auto it = lines.find(handle);
  if (it == lines.end()) return BFA_kResultInvalidHandle;
  auto line = it->second;
  if (line->stopped) return BFA_kResultNoChange;
  Touch(line);
  RecordStop(line, reason);
  line->Place(Now(), line->Position(Now()), true);
  if (!line->on_face) Finish(line, BFA_kEnded, reason);
  return line->reported ? BFA_kResultOK : BFA_kResultPending;
}
BFA_Result EndUtterance(uint64_t utterance, uint32_t reason) {
  if (!utterance || reason > BFA_kAborted) return BFA_kResultInvalidArgument;
  std::lock_guard<std::recursive_mutex> serial(calls);
  std::vector<PlayerSpeechPtr> stop;
  {
    ScopedLock guard(provider_lock);
    if (!utterances.erase(utterance)) return BFA_kResultNotFound;
    for (const auto& [handle, line] : lines)
      if (line->utterance_id == utterance) stop.push_back(line);
  }
  bool pending = false;
  for (const auto& line : stop) {
    pending |= Stop(line->handle, reason) == BFA_kResultPending;
    line->audio.reset();
  }
  ProviderDiscardUtterance(utterance);
  return pending ? BFA_kResultPending : BFA_kResultOK;
}
uint64_t RegisterListener(BFA_EventCallback fn, void* context,
                          BFA_Result* result) {
  if (!result) return 0;
  *result = BFA_kResultInvalidArgument;
  if (!fn) return 0;
  ScopedLock guard(provider_lock);
  if (!EnsureThread()) {
    *result = BFA_kResultInternalError;
    return 0;
  }
  const auto id = ++next_listener;
  listeners[id] = {fn, context};
  *result = BFA_kResultOK;
  return id;
}
BFA_Result UnregisterListener(uint64_t id) {
  ScopedLock guard(provider_lock);
  return listeners.erase(id) ? BFA_kResultOK : BFA_kResultNotFound;
}
const BFA_Interface kInterface = {
    BFA_API_VERSION, sizeof(BFA_Interface), NewUtteranceId, Prepare,
    Start,           Pause,                 Seek,           Stop,
    EndUtterance,    RegisterListener,      UnregisterListener};
void SendInterface(const char* receiver) {
  if (messaging && messaging->Dispatch)
    messaging->Dispatch(plugin_handle, BFA_kInterface,
                        const_cast<BFA_Interface*>(&kInterface),
                        sizeof kInterface, receiver);
}
void Receive(skse::SKSEMessage* m) {
  if (m && m->type == BFA_kRequestInterface && m->sender &&
      strcmp(m->sender, "SKSE"))
    SendInterface(m->sender);
}
}  // namespace

void ProviderLineStarted(const PlayerSpeechPtr& line) {
  ScopedLock guard(provider_lock);
  if (!line->stopped && !line->reported &&
      !line->animation_started.exchange(true))
    Post(BFA_kStarted, 0, line);
}
void ProviderLineFailed(const PlayerSpeechPtr& line, uint32_t result) {
  ScopedLock guard(provider_lock);
  RecordEnd(line, BFA_kFailed, result);
  if (!line->on_face) FinishPending(line);
}
void ProviderLineDeclined(const PlayerSpeechPtr& line, uint32_t result) {
  ScopedLock guard(provider_lock);
  RecordEnd(line, BFA_kDeclined, result);
  if (!line->on_face) FinishPending(line);
}
void ProviderLineStopping(const PlayerSpeechPtr& line, uint32_t reason) {
  ScopedLock guard(provider_lock);
  RecordStop(line, reason);
}
void ProviderLineLeft(const PlayerSpeechPtr& line, uint32_t reason) {
  ScopedLock guard(provider_lock);
  RecordStop(line, reason);
  FinishPending(line);
}
void ProviderReset() {
  ScopedLock guard(provider_lock);
  std::vector<PlayerSpeechPtr> copy;
  for (const auto& [handle, line] : lines) copy.push_back(line);
  for (const auto& line : copy) Finish(line, BFA_kEnded, BFA_kAborted);
  utterances.clear();
}
uint64_t ProviderHolder(uint32_t actor) {
  ScopedLock guard(provider_lock, ScopedLock::kShared);
  for (const auto& [id, u] : utterances)
    if (u.actor == actor && u.started && Now() - u.touched < kExpiry)
      return id;
  return 0;
}
bool ProviderClaims(uint32_t actor) { return ProviderHolder(actor) != 0; }
void InitProvider(const skse::SKSEInterface* iface) {
  plugin_handle = iface->GetPluginHandle();
  messaging = static_cast<skse::SKSEMessagingInterface*>(
      iface->QueryInterface(skse::kInterfaceMessaging));
  tasks = static_cast<skse::SKSETaskInterface*>(
      iface->QueryInterface(skse::kInterfaceTask));
}
void ProviderMessage(unsigned type) {
  if (type != skse::kMessagePostPostLoad) return;
  if (messaging) messaging->RegisterListener(plugin_handle, nullptr, Receive);
  SendInterface(nullptr);
}
}  // namespace bfa

extern "C" __declspec(dllexport) const BFA_Interface* BFA_QueryInterface(
    uint32_t version) {
  return version == BFA_API_VERSION ? &bfa::kInterface : nullptr;
}
