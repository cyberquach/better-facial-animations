#include "runtime_fixture.h"
// Exercise the real message handler, menu sink and queued game task. Preserve
// the October 13 startup-scan control, and add the November 8 new-game case.
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
      fprintf(stderr, "line %d: %s\n", __LINE__, #x); \
      std::exit(1);                                   \
    }                                                 \
  } while (0)

namespace {
int starts = 0;
bool startup_broadcast = false, scan_finished = false, fail_thread = false;
bool in_new_game = false, in_menu_event = false;
bool main_open = false, loading_open = false, ui_available = true;
std::vector<SKSE::Impl::TaskDelegate*> tasks;
RE::BSTEventSink<RE::MenuOpenCloseEvent>* menu_sink = nullptr;
void* Open(void* stream, const char*, uint64_t, void*) { return stream; }
void Close(void*) {}
void Message(uint32_t type, void* data = nullptr) {
  skse::SKSEMessage message{"SKSE", type, 0, data};
  in_new_game = type == skse::kMessageNewGame;
  bfa::OnMessage(&message);
  in_new_game = false;
}
void AddTask(void* task) {
  // Enqueue must not hold the plugin lock (including on SKSE VR).
  CHECK(TryAcquireSRWLockExclusive(&bfa::lock));
  ReleaseSRWLockExclusive(&bfa::lock);
  tasks.push_back(static_cast<SKSE::Impl::TaskDelegate*>(task));
}
SKSE::Impl::SKSETaskInterface task_api{2, AddTask, nullptr};
void DrainTasks() {
  auto pending = std::move(tasks);
  tasks.clear();
  for (auto* task : pending) {
    task->Run();
    task->Dispose();
  }
  CHECK(tasks.empty());  // Never poll/requeue inside a task dispatch loop.
}
void Menu(const char* name, bool opening) {
  // Independent fixed-string storage; never invoke the game's intern pool.
  std::array<uint8_t, sizeof(RE::MenuOpenCloseEvent)> storage{};
  auto* event = reinterpret_cast<RE::MenuOpenCloseEvent*>(storage.data());
  bfa::Field<const char*>(event, 0) = FixtureString(name);
  event->opening = opening;
  in_menu_event = true;
  CHECK(menu_sink);
  menu_sink->ProcessEvent(event, nullptr);
  in_menu_event = false;
}
void Reset() {
  DrainTasks();
  using namespace bfa;
  startup_data_loaded = worker_started = false;
  new_game_start_epoch = new_game_task_epoch = 0;
  ready = false;
  starts = 0;
  scan_finished = false;
  fail_thread = false;
  main_open = loading_open = false;
  ui_available = true;
  startup_tasks = reinterpret_cast<skse::SKSETaskInterface*>(&task_api);
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
  CHECK(!in_new_game && !in_menu_event);
  CHECK(entry == bfa::Worker && !parameter && !flags);
  ++starts;
  if (fail_thread) {
    SetLastError(ERROR_NOT_ENOUGH_MEMORY);
    return nullptr;
  }
  return CreateEventW(nullptr, FALSE, FALSE, nullptr);
}

int main() {
  using namespace bfa;
  testing::address = TestAddress;
  testing::startup_register_menu_sink = [](auto* sink) { menu_sink = sink; };
  testing::startup_menus_closed = [] {
    CHECK(!in_menu_event);  // Query after the menu event returns.
    return ui_available && !main_open && !loading_open;
  };
  for (bool logger_first : {false, true}) {
    for (uint32_t enter : {skse::kMessagePostLoadGame, skse::kMessageNewGame}) {
      for (bool gpu : {false, true}) {
        Reset();
        config.gpu = gpu;
        Message(enter, reinterpret_cast<void*>(1));
        DrainTasks();
        CHECK(starts == 0);  // No data yet.
        startup_broadcast = true;
        if (logger_first) scan_finished = true;
        Message(skse::kMessageDataLoaded);
        CHECK(starts == 0 && startup_data_loaded && stats.waiting_for_game);
        Menu("Loading Menu", false);
        Menu("Main Menu", false);
        // An early provider request cannot start inference either.
        ProviderActor actor;
        CHECK(ResolveProviderActor(0x100, actor) == BFA_kResultModelNotLoaded);
        RequestGpuRetry();
        RequestGpuAdapters();
        DrainTasks();
        CHECK(starts == 0);
        if (!logger_first) scan_finished = true;
        startup_broadcast = false;
        Message(skse::kMessagePostPostLoad);
        Message(skse::kMessagePreLoadGame);
        Message(skse::kMessagePostLoadGame);  // Failed load.
        CHECK(starts == 0);
        Message(enter, reinterpret_cast<void*>(1));
        if (enter == skse::kMessageNewGame) {
          CHECK(starts == 0 && tasks.size() == 1);
          DrainTasks();  // Handles menus that closed before NewGame arrived.
        }
        CHECK(starts == 1 && worker_started && !stats.waiting_for_game);
        for (int i = 0; i < 3; ++i) {
          Message(skse::kMessagePreLoadGame);
          Message(skse::kMessagePostLoadGame, reinterpret_cast<void*>(1));
          Message(skse::kMessageNewGame);
          Menu("Loading Menu", false);
        }
        DrainTasks();
        CHECK(starts == 1);  // Retain existing worker/context across worlds.
      }
    }
  }
  for (bool main_first : {false, true}) {
    Reset();
    Message(skse::kMessageDataLoaded);
    scan_finished = true;
    main_open = loading_open = true;
    Message(skse::kMessageNewGame);
    DrainTasks();
    CHECK(starts == 0 && !worker_started);
    Menu("InventoryMenu", false);
    Menu("Loading Menu", true);
    CHECK(tasks.empty());
    (main_first ? main_open : loading_open) = false;
    Menu(main_first ? "Main Menu" : "Loading Menu", false);
    DrainTasks();
    CHECK(starts == 0);  // The other menu still blocks initialization.
    (main_first ? loading_open : main_open) = false;
    Menu(main_first ? "Loading Menu" : "Main Menu", false);
    Menu(main_first ? "Loading Menu" : "Main Menu", false);
    CHECK(tasks.size() == 1 && starts == 0);
    DrainTasks();
    CHECK(starts == 1 && worker_started && !new_game_start_epoch);
  }
  // An obsolete task must neither start inference nor consume a newer task.
  Reset();
  Message(skse::kMessageDataLoaded);
  scan_finished = true;
  Message(skse::kMessageNewGame);
  Message(skse::kMessagePreLoadGame);
  Message(skse::kMessagePostLoadGame);  // Failed load cancels pending start.
  DrainTasks();
  CHECK(starts == 0);
  Message(skse::kMessageNewGame);
  Message(skse::kMessageNewGame);
  CHECK(tasks.size() == 2);
  DrainTasks();
  CHECK(starts == 1 && !new_game_task_epoch);

  Reset();
  Message(skse::kMessageDataLoaded);
  scan_finished = true;
  ui_available = false;
  Message(skse::kMessageNewGame);
  DrainTasks();
  CHECK(starts == 0);  // Unknown UI state cannot authorize startup.
  ui_available = true;
  fail_thread = true;
  Menu("Loading Menu", false);
  DrainTasks();
  CHECK(starts == 1 && !worker_started && !stats.startup_error.empty());
  fail_thread = false;
  Menu("Loading Menu", false);
  DrainTasks();
  CHECK(starts == 2 && worker_started && stats.startup_error.empty());

  Reset();
  Message(skse::kMessageDataLoaded);
  scan_finished = true;
  startup_tasks = nullptr;
  Message(skse::kMessageNewGame);
  CHECK(starts == 0 && tasks.empty() && !stats.startup_error.empty());
  // A successful save load still starts without the new-game task path.
  Message(skse::kMessagePostLoadGame, reinterpret_cast<void*>(1));
  CHECK(starts == 1 && worker_started && stats.startup_error.empty());
  worker_started = false;
  fail_thread = true;
  Message(skse::kMessagePostLoadGame, reinterpret_cast<void*>(1));
  CHECK(starts == 2 && !worker_started && !stats.startup_error.empty());
  fail_thread = false;
  Message(skse::kMessagePostLoadGame, reinterpret_cast<void*>(1));
  CHECK(starts == 3 && worker_started && stats.startup_error.empty());
  puts(
      "startup scan exclusion, new-game menu/task deferral, saved games, "
      "CPU/GPU, stale tasks, one worker and thread failure passed");
}
