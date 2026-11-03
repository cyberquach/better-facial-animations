#include "a2f/inference/diagnostics.h"
#include "a2f/inference/windows_prerequisite.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <thread>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); std::exit(1); } } while (0)

int main() {
  using namespace a2f::diagnostics;
  using namespace a2f::independent;
  WindowsVersion v;
  CHECK(CheckWindows(v) == Prerequisite::kUnknown);
  v.known = v.workstation = true;
  v.major = 10;
  for (const auto& pair : {std::pair<unsigned, unsigned>{19045, 3393},
                         {22000, 2360}, {22621, 2215}}) {
    v.build = pair.first;
    v.revision_known = false;
    CHECK(CheckWindows(v) == Prerequisite::kUnknown);
    v.revision_known = true;
    v.revision = pair.second - 1;
    CHECK(CheckWindows(v) == Prerequisite::kTooOld);
    CHECK(WindowsPrerequisiteReport(v).find("restart") != std::string::npos);
    ++v.revision;
    CHECK(CheckWindows(v) == Prerequisite::kMet);
    ++v.revision;
    CHECK(CheckWindows(v) == Prerequisite::kMet);
  }
  for (unsigned build : {22631, 26100, 26200}) {
    v.build = build;
    v.revision_known = false;
    CHECK(CheckWindows(v) == Prerequisite::kMet);
  }
  v.wine = true;
  CHECK(CheckWindows(v) == Prerequisite::kUnknown);
  CHECK(WindowsPrerequisiteReport(v).find("vkd3d-proton") != std::string::npos);
  v.wine = false;
  v.workstation = false;
  CHECK(CheckWindows(v) == Prerequisite::kUnknown);
  v.workstation = true;
  for (unsigned build : {19044, 20348, 22622, 30000}) {
    v.build = build;
    CHECK(CheckWindows(v) == Prerequisite::kUnknown);
  }
  v.build = 26100;
  v.known = false;
  CHECK(CheckWindows(v) == Prerequisite::kUnknown);
  std::string error;
  {
    Attempt attempt(Component::kFace, true, true, error);
    Progress(Stage::kDevice);
    const auto pending = State().Read().components[0];
    CHECK(pending.loading && !pending.ready && pending.stage == Stage::kDevice);
    error = "CreateDevice failed [HRESULT 0x887A0004]";
  }
  auto failed = State().Read().components[0];
  CHECK(!failed.loading && !failed.ready && failed.last_failure.detail == error);
  CHECK(failed.last_failure.stage == Stage::kDevice);
  // Successful explicit CPU recovery must not erase the original GPU failure.
  {
    Attempt attempt(Component::kFace, false, false, error);
    Progress(Stage::kWarmup);
    attempt.Ready();
  }
  auto recovered = State().Read().components[0];
  CHECK(recovered.ready && !recovered.gpu && recovered.error.empty());
  CHECK(recovered.attempt == 2 && recovered.last_failure.attempt == 1);
  CHECK(recovered.last_failure.detail == failed.error);
  CHECK(recovered.last_failure.gpu && recovered.last_failure.independent);
  {
    Attempt sentiment(Component::kSentiment, true, false, error);
    Progress(Stage::kProvider);
    error = "DirectML provider failed [ORT error 1]";
  }
  CHECK(State().Read().components[0].ready);
  CHECK(State().Read().components[1].last_failure.stage == Stage::kProvider);
  State().RuntimeFailure(Component::kFace, "device removed 887A0005");
  CHECK(State().Read().components[0].last_failure.stage == Stage::kInference);
  // The menu can snapshot a blocked operation without waiting for its completion.
  Store shared;
  CHECK(shared.Read().independent_api.find("not checked") == 0);
  shared.IndependentApi("available");
  shared.Begin(Component::kFace, false, false);
  CHECK(shared.Read().independent_api == "available");
  std::atomic<bool> entered{false}, release{false};
  std::thread worker([&] {
    shared.Begin(Component::kFace, true, false);
    shared.Progress(Component::kFace, Stage::kSession);
    entered = true;
    while (!release) Sleep(1);
    shared.Complete(Component::kFace, false, std::string(100000, 'x'));
  });
  while (!entered) Sleep(1);
  CHECK(shared.Read().components[0].stage == Stage::kSession);
  CHECK(shared.Read().components[0].loading);
  release = true;
  worker.join();
  CHECK(shared.Read().components[0].error.size() < 16500);
  shared.System(std::vector<std::string>(1000, std::string(20000, 'x')));
  CHECK(shared.Read().system.size() == 64);
  CHECK(shared.Read().system[0].size() < 16500);
  puts("diagnostics: Windows prerequisite boundaries/unknowns, API status, pending operation, recovery history, bounded concurrent snapshots passed");
}
