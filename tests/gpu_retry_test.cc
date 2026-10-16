#include "a2f/inference/gpu_retry.h"
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <utility>
#include <vector>
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "line %d: %s\n", __LINE__, #x); std::exit(1); } } while (0)
int main() {
  using namespace a2f;
  using Calls = std::vector<std::pair<bool, bool>>;
  std::string err;
  // Driver-specific operators fail; shader-only succeeds and remains selected.
  SharedGpuSettings settings;
  Calls calls;
  auto shaders = [&](bool meta, bool wrapper, std::string& error) {
    calls.emplace_back(meta, wrapper);
    error = "session: driver operator failed";
    return meta ? std::unique_ptr<int>{} : std::make_unique<int>(1);
  };
  CHECK(CreateSharedGpu(shaders, settings, [] { return true; }, err));
  CHECK((calls == Calls{{true, false}, {false, false}}));
  calls.clear();
  CHECK(CreateSharedGpu(shaders, settings, [] { return true; }, err));
  CHECK((calls == Calls{{false, false}}));
  // A lost unwrapped device retries the observed wrapper, including shaders.
  for (const char* code : {"887A0005", "887A0006", "887A0007", "887A0020"}) {
    settings = {}; calls.clear();
    auto wrapper_only = [&](bool meta, bool wrapper, std::string& error) {
      calls.emplace_back(meta, wrapper);
      error = code;
      return wrapper && !meta ? std::make_unique<int>(1) : std::unique_ptr<int>{};
    };
    CHECK(CreateSharedGpu(wrapper_only, settings, [] { return true; }, err));
    CHECK((calls == Calls{{true, false}, {false, false}, {true, true}, {false, true}}));
    calls.clear();
    CHECK(CreateSharedGpu(wrapper_only, settings, [] { return true; }, err));
    CHECK((calls == Calls{{false, true}}));
  }
  // No wrapper retries without actual wrapper evidence, or for unrelated errors.
  for (bool wrapped : {false, true}) {
    settings = {}; calls.clear();
    auto fail = [&](bool meta, bool wrapper, std::string& error) -> std::unique_ptr<int> {
      calls.emplace_back(meta, wrapper);
      error = wrapped ? "8007000E" : "887A0005";
      return nullptr;
    };
    CHECK(!CreateSharedGpu(fail, settings, [=] { return wrapped; }, err));
    CHECK((calls == Calls{{true, false}, {false, false}}));
    CHECK(settings.metacommands && !settings.wrapper);
  }
  for (const char* terminal : {"session; GPU attempt failed", "cannot load onnxruntime.dll",
                              "onnxruntime.dll: wrong version", "model file not found"}) {
    settings = {}; calls.clear();
    auto fail = [&](bool meta, bool wrapper, std::string& error) -> std::unique_ptr<int> {
      calls.emplace_back(meta, wrapper); error = terminal; return nullptr;
    };
    CHECK(!CreateSharedGpu(fail, settings, [] { return true; }, err));
    CHECK(calls.size() == 1);
  }
  // Guard failure on the second attempt must stop wrapper and later retries.
  settings = {}; calls.clear();
  auto guard = [&](bool meta, bool wrapper, std::string& error) -> std::unique_ptr<int> {
    calls.emplace_back(meta, wrapper);
    error = calls.size() == 1 ? "887A0005" : "session; GPU attempt failed";
    return nullptr;
  };
  CHECK(!CreateSharedGpu(guard, settings, [] { return true; }, err));
  CHECK(calls.size() == 2 && !RetryableGpuFailure(err));
  puts("shared GPU: shader/wrapper recovery, remembered successes and terminal guards passed");
}
