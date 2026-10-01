// Log, Address Library loader and engine helpers declared in skse.h.
#include "skse.h"

#include <shlobj.h>
#include <windows.h>

#include <algorithm>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

namespace skse {
namespace {

FILE* log_file;
uint64_t image_base;
std::vector<std::pair<uint64_t, uint64_t>> ids;  // id -> offset, sorted by id

}  // namespace

void Log(const char* fmt, ...) {
  if (!log_file) return;
  va_list a;
  va_start(a, fmt);
  vfprintf(log_file, fmt, a);
  va_end(a);
  fputc('\n', log_file);
  fflush(log_file);
}

void OpenLog() {
  wchar_t* docs = nullptr;
  wchar_t path[MAX_PATH];
  if (SHGetKnownFolderPath(FOLDERID_Documents, 0, nullptr, &docs) != S_OK)
    return;
  _snwprintf(path, MAX_PATH,
             L"%ls\\My Games\\Skyrim Special Edition\\SKSE\\A2FLive.log", docs);
  CoTaskMemFree(docs);
  log_file = _wfopen(path, L"w");
}

// versionlib format 2 (CommonLibSSE-NG REL/ID.h).
bool LoadAddressLibrary(const char* path) {
  image_base = reinterpret_cast<uint64_t>(GetModuleHandleA(nullptr));
  FILE* f = fopen(path, "rb");
  if (!f) return false;
  std::vector<uint8_t> d;
  fseek(f, 0, SEEK_END);
  d.resize(ftell(f));
  fseek(f, 0, SEEK_SET);
  fread(d.data(), 1, d.size(), f);
  fclose(f);
  size_t p = 0;
  auto rd = [&](size_t n) -> uint64_t {
    uint64_t v = 0;
    memcpy(&v, &d[p], n);
    p += n;
    return v;
  };
  if (static_cast<int32_t>(rd(4)) != 2) return false;
  p += 16;
  int32_t name_len = static_cast<int32_t>(rd(4));
  p += name_len;
  int32_t ptr_size = static_cast<int32_t>(rd(4));
  int32_t count = static_cast<int32_t>(rd(4));
  ids.reserve(count);
  uint64_t pid = 0, poff = 0;
  for (int32_t i = 0; i < count; ++i) {
    uint8_t t = static_cast<uint8_t>(rd(1));
    uint8_t lo = t & 0xF, hi = t >> 4;
    uint64_t id, off;
    switch (lo) {
      case 0:
        id = rd(8);
        break;
      case 1:
        id = pid + 1;
        break;
      case 2:
        id = pid + rd(1);
        break;
      case 3:
        id = pid - rd(1);
        break;
      case 4:
        id = pid + rd(2);
        break;
      case 5:
        id = pid - rd(2);
        break;
      case 6:
        id = rd(2);
        break;
      case 7:
        id = rd(4);
        break;
      default:
        return false;
    }
    uint64_t base = (hi & 8) ? poff / ptr_size : poff;
    switch (hi & 7) {
      case 0:
        off = rd(8);
        break;
      case 1:
        off = base + 1;
        break;
      case 2:
        off = base + rd(1);
        break;
      case 3:
        off = base - rd(1);
        break;
      case 4:
        off = base + rd(2);
        break;
      case 5:
        off = base - rd(2);
        break;
      case 6:
        off = rd(2);
        break;
      case 7:
        off = rd(4);
        break;
      default:
        return false;
    }
    if (hi & 8) off *= ptr_size;
    ids.emplace_back(id, off);
    pid = id;
    poff = off;
  }
  std::sort(ids.begin(), ids.end());
  return true;
}

uint64_t ImageBase() { return image_base; }

uint64_t Addr(uint64_t id) {
  auto it =
      std::lower_bound(ids.begin(), ids.end(), std::make_pair(id, uint64_t{0}));
  return (it != ids.end() && it->first == id) ? image_base + it->second : 0;
}

void SpinLock::Lock() {
  LONG me = static_cast<LONG>(GetCurrentThreadId());
  if (owner == me) {
    InterlockedIncrement(&count);
    return;
  }
  for (uint32_t spins = 0; InterlockedCompareExchange(&count, 1, 0) != 0;
       ++spins) {
    if (spins < 10000) {
      YieldProcessor();
    } else {
      Sleep(0);
    }
  }
  owner = me;
}

void SpinLock::Unlock() {
  if (count == 1) {
    owner = 0;
    InterlockedCompareExchange(&count, 0, 1);
  } else {
    InterlockedDecrement(&count);
  }
}

const ResId* ActiveLine(uint8_t* fg) {
  void* entry = *reinterpret_cast<void**>(fg + kFgDialogue);
  if (!entry) return nullptr;
  uint32_t ctrl =
      *reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(entry) + 0x0C);
  if ((((ctrl & 0x70000000u) + 0xD0000000u) & 0xEFFFFFFFu) != 0)
    return nullptr;  // lip resource not loaded
  return static_cast<const ResId*>(entry);
}

}  // namespace skse
