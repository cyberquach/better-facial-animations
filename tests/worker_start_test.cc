#include "runtime_fixture.h"
// Exercise the real SKSE handler, with inference thread creation intercepted.
#include <windows.h>

HANDLE WINAPI CaptureThread(LPSECURITY_ATTRIBUTES, SIZE_T,
                            LPTHREAD_START_ROUTINE, LPVOID, DWORD, LPDWORD);
#define CreateThread CaptureThread
#include "../src/plugin/better_facial_animations.cc"
#undef CreateThread

#include <cstdlib>

#define CHECK(x)                                      \
  do {                                                \
    if (!(x)) {                                       \
      fprintf(stderr, "line %d: %s\n", __LINE__, #x);  \
      std::exit(1);                                   \
    }                                                 \
  } while (0)

namespace {
int starts = 0;
bool startup_broadcast = false, scan_finished = false, fail_thread = false;
void* Open(void* stream, const char*, uint64_t, void*) { return stream; }
void Close(void*) {}
void Message(uint32_t type, void* data = nullptr) {
  skse::SKSEMessage message{"SKSE", type, 0, data};
  bfa::OnMessage(&message);
}
}  // namespace

extern "C" uint64_t TestAddress(uint64_t id) {
  if (id == kIdStreamCtor.Get()) return reinterpret_cast<uint64_t>(&Open);
  if (id == kIdStreamDtor.Get()) return reinterpret_cast<uint64_t>(&Close);
  return 0;
}

HANDLE WINAPI CaptureThread(LPSECURITY_ATTRIBUTES, SIZE_T,
                            LPTHREAD_START_ROUTINE entry, LPVOID parameter,
                            DWORD flags, LPDWORD) {
  CHECK(!startup_broadcast && scan_finished);
  CHECK(entry == bfa::Worker && !parameter && !flags);
  ++starts;
  if (fail_thread) {
    SetLastError(ERROR_NOT_ENOUGH_MEMORY);
    return nullptr;
  }
  // A real closeable handle; no inference or graphics libraries are loaded.
  return CreateEventW(nullptr, FALSE, FALSE, nullptr);
}

int main() {
  bfa::testing::address = TestAddress;
  using namespace bfa;
  for (bool logger_first : {false, true}) {
    for (uint32_t enter : {skse::kMessagePostLoadGame, skse::kMessageNewGame}) {
      for (bool gpu : {false, true}) {
        config.gpu = gpu;
        startup_data_loaded = worker_started = false;
        starts = 0;
        scan_finished = false;
        Message(enter, reinterpret_cast<void*>(1));
        CHECK(starts == 0);  // No data yet.
        startup_broadcast = true;
        if (logger_first) scan_finished = true;
        Message(skse::kMessageDataLoaded);
        CHECK(starts == 0 && startup_data_loaded && stats.waiting_for_game);
        if (!logger_first) scan_finished = true;
        startup_broadcast = false;
        Message(skse::kMessagePostPostLoad);
        Message(skse::kMessagePreLoadGame);
        Message(skse::kMessagePostLoadGame);  // Failed load.
        CHECK(starts == 0);
        Message(enter, reinterpret_cast<void*>(1));
        CHECK(starts == 1 && worker_started && !stats.waiting_for_game);
        for (int i = 0; i < 3; ++i) {
          Message(skse::kMessagePreLoadGame);
          Message(skse::kMessagePostLoadGame, reinterpret_cast<void*>(1));
          Message(skse::kMessageNewGame);
        }
        CHECK(starts == 1);  // Never recreate a running worker/context.
      }
    }
  }
  worker_started = false;
  fail_thread = true;
  starts = 0;
  Message(skse::kMessageNewGame);
  CHECK(starts == 1 && !worker_started && !stats.startup_error.empty());
  fail_thread = false;
  Message(skse::kMessageNewGame);
  CHECK(starts == 2 && worker_started && stats.startup_error.empty());
  puts("startup scan exclusion, CPU/GPU, load/new game, failed load, one worker and thread failure passed");
}
