// The DBReV adapter (src/speech/player_speech.cc): message validation, the owned
// copy, replacement, cancellation and the reset on a load.
#include "a2f/speech/player_speech.h"

#include <cstdio>
#include <cstring>
#include <vector>

#include "a2f/speech/DBReV_API.h"
#include "a2f/runtime/skse.h"

namespace {
int failures = 0;
void Expect(bool ok, const char* label) {
  printf("%s %s\n", ok ? "ok" : "FAIL", label);
  if (!ok) ++failures;
}

// SKSE: the listener the adapter registers and the tasks it queues.
void (*listener)(skse::SKSEMessage*) = nullptr;
std::vector<skse::TaskDelegate*> queued;
// The lines the adapter started (StartPlayerSpeech below).
std::vector<bfa::PlayerSpeechPtr> started;

bool dbvo_present = false, dbrev_present = true;
const void* PluginInfo(const char* name) {
  return dbvo_present && !strcmp(name, "DBVO") ? &dbvo_present : nullptr;
}

bool Listen(uint32_t, const char*, void (*cb)(skse::SKSEMessage*)) {
  if (!dbrev_present) return false;
  listener = cb;
  return true;
}
void Add(skse::TaskDelegate* task) { queued.push_back(task); }
skse::SKSEMessagingInterface messaging{1, Listen};
skse::SKSETaskInterface tasks{2, Add, nullptr};
void* Query(uint32_t id) {
  return id == skse::kInterfaceTask ? static_cast<void*>(&tasks)
                                    : static_cast<void*>(&messaging);
}
uint32_t Handle() { return 1; }

// Runs the queued tasks, as the game thread does.
void Drain() {
  for (skse::TaskDelegate* task : queued) {
    task->Run();
    task->Dispose();
  }
  queued.clear();
}

template <class T>
void Send(uint32_t type, T& data, uint32_t size = sizeof(T)) {
  skse::SKSEMessage message{DBReV::kSender, type, size, &data};
  listener(&message);
}
}  // namespace

namespace skse {
void Log(const char*, ...) {}
}  // namespace skse

namespace bfa {
double Now() { return 12.5; }
void StartPlayerSpeech(PlayerSpeechPtr speech) { started.push_back(speech); }
}  // namespace bfa

int main() {
  skse::SKSEInterface api{};
  api.QueryInterface = Query;
  api.GetPluginHandle = Handle;
  api.GetPluginInfo = PluginInfo;
  for (bool dbvo : {false, true}) {
    for (bool dbrev : {false, true}) {
      dbvo_present = dbvo;
      dbrev_present = dbrev;
      listener = nullptr;
      bfa::InitPlayerSpeech(&api);
      Expect(!bfa::DbvoInstalled(), "provider detection waits for PostLoad");
      bfa::PlayerSpeechMessage(skse::kMessagePostLoad);
      Expect(bfa::DbvoInstalled() == dbvo, "detect only registered DBVO");
      Expect(bool(listener) == dbrev, "register only installed DBReV");
    }
  }
  dbvo_present = false;
  dbrev_present = true;
  listener = nullptr;
  bfa::InitPlayerSpeech(&api);
  Expect(!listener, "nothing is registered at load");
  bfa::PlayerSpeechMessage(skse::kMessagePostLoad);
  if (!listener) {
    printf("FAIL no listener registered at PostLoad\n");
    return 1;
  }

  char path[] = "DBReV/pack/hello.fuz";
  DBReV::PlayerLineStart begin{};
  begin.version = 1;
  begin.structSize = sizeof(begin);
  begin.audioSeconds = 2;
  begin.totalSeconds = 5;
  begin.topicIndex = 4;
  begin.voicePath = path;
  Send(DBReV::kPlayerLineStart, begin, 8);
  Expect(queued.empty(), "a short payload is ignored");
  begin.version = 0;
  Send(DBReV::kPlayerLineStart, begin);
  Expect(queued.empty(), "version 0 is ignored");
  begin.version = 1;
  Send(DBReV::kPlayerLineStart, begin);
  path[0] = 'X';  // the provider's storage is gone before the task runs
  Drain();
  Expect(started.size() == 1 && started.back()->path == "DBReV/pack/hello.fuz",
         "the path is copied");
  Expect(started.back()->duration == 2 && started.back()->started_at == 12.5,
         "the audio length and the start time are kept");

  DBReV::PlayerLineEnd end{1, sizeof(end), 99, DBReV::kEndReason_Skipped};
  Send(DBReV::kPlayerLineEnd, end);
  Expect(!started.back()->stopped, "another topic's end is ignored");
  end.topicIndex = 4;
  Send(DBReV::kPlayerLineEnd, end);
  Expect(started.back()->stopped, "the line's end stops it");

  Send(DBReV::kPlayerLineStart, begin);
  Send(DBReV::kPlayerLineEnd, end);
  Drain();
  Expect(started.size() == 1, "a line ended before its task runs never starts");

  Send(DBReV::kPlayerLineStart, begin);
  begin.topicIndex = 5;
  Send(DBReV::kPlayerLineStart, begin);
  Drain();
  Expect(started.size() == 2, "a replaced queued start is ignored");
  Send(DBReV::kPlayerLineEnd, end);
  Expect(!started.back()->stopped, "the replaced line's end is ignored");
  DBReV::PlayerLineEnd late{1, sizeof(late), 5, DBReV::kEndReason_Superseded};
  Send(DBReV::kPlayerLineEnd, late);
  Expect(!started.back()->stopped, "a late superseded end is ignored");
  bfa::PlayerSpeechMessage(skse::kMessagePreLoadGame);
  Expect(started.back()->stopped, "loading a game stops the line");

  struct Future {
    DBReV::PlayerLineStart line;
    uint64_t extra;
  } future{};
  future.line = begin;
  future.line.version = 2;
  future.line.structSize = sizeof(future);
  Send(DBReV::kPlayerLineStart, future);
  Drain();
  Expect(started.size() == 3, "a later API version with more fields is read");

  Send(DBReV::kPlayerLineStart, begin);
  bfa::PlayerSpeechMessage(skse::kMessageNewGame);
  Drain();
  Expect(started.size() == 3 && started.back()->stopped,
         "a new game stops the lines, queued too");

  Send(DBReV::kPlayerLineStart, begin);
  bfa::PlayerSpeechMessage(skse::kMessagePostLoadGame);
  Drain();
  Expect(started.size() == 3,
         "post-load cancels provider starts queued during loading");

  printf("%s: %d failures\n", failures ? "FAIL" : "ok", failures);
  return failures ? 1 : 0;
}
