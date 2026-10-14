#include "a2f/inference/gpu_startup.h"
#include <cstdio>
#include <cstdlib>
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "line %d: %s\n", __LINE__, #x); std::exit(1); } } while (0)
int main() {
  wchar_t temp[MAX_PATH], path[MAX_PATH];
  CHECK(GetTempPathW(MAX_PATH, temp));
  CHECK(GetTempFileNameW(temp, L"bfa", 0, path));
  CHECK(DeleteFileW(path));
  std::string err;
  {
    a2f::GpuStartup attempt(path);
    CHECK(attempt.Begin(err));
    // Simulate process exit before completion; destruction must leave guard.
  }
  a2f::GpuStartup next_launch(path);
  CHECK(!next_launch.Begin(err));
  CHECK(err.find("Using CPU") != std::string::npos);
  CHECK(next_launch.Complete(err)); // cannot delete someone else's guard
  CHECK(GetFileAttributesW(path) != INVALID_FILE_ATTRIBUTES);
  CHECK(DeleteFileW(path)); // deliberate user retry after closing Skyrim
  a2f::GpuStartup retry(path);
  CHECK(retry.Begin(err) && retry.Complete(err));
  CHECK(GetFileAttributesW(path) == INVALID_FILE_ATTRIBUTES);
  a2f::GpuStartup unwritable(std::wstring(path) + L"\\missing\\guard");
  CHECK(!unwritable.Begin(err));
  // Failure to remove a guard is reported, not advertised as a clean launch.
  a2f::GpuStartup locked(path);
  CHECK(locked.Begin(err));
  HANDLE reader = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ, nullptr,
                               OPEN_EXISTING, 0, nullptr);
  CHECK(reader != INVALID_HANDLE_VALUE);
  CHECK(!locked.Complete(err));
  CloseHandle(reader);
  CHECK(DeleteFileW(path));
  puts("GPU startup crash recovery, deliberate retry and IO fail-closed: pass");
}
