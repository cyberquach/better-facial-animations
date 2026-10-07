// DBReV is optional. This adapter only copies its public notification into an
// owned speech session; audio playback and dialogue control stay with DBReV.
#include "player_speech.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>

#include "DBReV_API.h"
#include "plugin.h"
#include "scoped_lock.h"
#include "skse.h"

namespace bfa {
namespace {
skse::SKSEMessagingInterface* messaging = nullptr;
skse::SKSETaskInterface* tasks = nullptr;
uint32_t plugin_handle = 0;
const void* (*plugin_info)(const char*) = nullptr;
bool dbvo_installed = false;
SRWLOCK session_lock = SRWLOCK_INIT;
// not destroyed at exit: the Papyrus thread may still be notifying
PlayerSpeechPtr& active = *new PlayerSpeechPtr;
uint32_t topic = 0;

struct StartTask final : skse::TaskDelegate {
  PlayerSpeechPtr speech;
  explicit StartTask(PlayerSpeechPtr value) : speech(std::move(value)) {}
  void Run() override {
    if (!speech->stopped) StartPlayerSpeech(speech);
  }
  void Dispose() override { delete this; }
};

template <class T>
bool Read(const skse::SKSEMessage& message, T& out) {
  if (!message.data || message.dataLen < sizeof(T)) return false;
  std::memcpy(&out, message.data, sizeof(T));
  return out.version >= 1 && out.structSize >= sizeof(T) &&
         out.structSize <= message.dataLen;
}

void Receive(skse::SKSEMessage* message) {
  if (!message) return;
  ScopedLock guard(session_lock);
  if (message->type == DBReV::kPlayerLineStart) {
    DBReV::PlayerLineStart line{};
    if (!Read(*message, line)) return;
    if (active) active->stopped = true;
    active.reset();
    if (!line.voicePath || !*line.voicePath) return;
    auto speech = std::make_shared<PlayerSpeech>();
    speech->path = line.voicePath;
    speech->started_at = Now();
    speech->duration = std::isfinite(line.audioSeconds)
                           ? std::max(0.f, line.audioSeconds) : 0.f;
    topic = line.topicIndex;
    active = speech;
    tasks->AddTask(new StartTask(std::move(speech)));
  } else if (message->type == DBReV::kPlayerLineEnd) {
    DBReV::PlayerLineEnd end{};
    // A superseded line was already stopped by the start that replaced it;
    // its end may come after that start, with the same topic index.
    if (Read(*message, end) && end.reason != DBReV::kEndReason_Superseded &&
        active && topic == end.topicIndex) {
      active->stopped = true;
      active.reset();
    }
  }
}
}  // namespace

bool DbvoInstalled() { return dbvo_installed; }

void InitPlayerSpeech(const skse::SKSEInterface* iface) {
  plugin_info = iface->GetPluginInfo;
  dbvo_installed = false;
  plugin_handle = iface->GetPluginHandle();
  messaging = static_cast<skse::SKSEMessagingInterface*>(
      iface->QueryInterface(skse::kInterfaceMessaging));
  tasks = static_cast<skse::SKSETaskInterface*>(
      iface->QueryInterface(skse::kInterfaceTask));
}

void PlayerSpeechMessage(unsigned type) {
  if (type == skse::kMessagePostLoad) {
    dbvo_installed = plugin_info && plugin_info("DBVO");
    skse::Log("player speech: DBVO %s", dbvo_installed ? "detected" : "not present");
    bool found = messaging && tasks &&
        messaging->RegisterListener(plugin_handle, DBReV::kSender, Receive);
    skse::Log("player speech: DBReV listener %s",
              found ? "registered" : "not present");
  } else if (type == skse::kMessagePreLoadGame ||
             type == skse::kMessageNewGame ||
             type == skse::kMessagePostLoadGame) {
    ScopedLock guard(session_lock);
    if (active) active->stopped = true;
    active.reset();
  }
}
}  // namespace bfa
