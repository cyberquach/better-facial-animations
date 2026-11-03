#pragma once
// Offline contracts substitute engine calls and deterministic scheduling only.
// This header contributes no state or branches to the production DLL.
#ifdef ENABLE_COMMONLIBSSE_TESTING
#include <windows.h>

#include <atomic>
#include <cstdint>
namespace bfa::testing {
inline uint64_t (*address)(uint64_t);
inline void (*log)(const char*, ...);
inline void (*report)(const char*, ...);
inline void (*before_face_lock)();
inline void (*save_ini)();
inline std::atomic<double> clock_offset{0};
inline HRESULT(WINAPI* dxgi_factory)(REFIID, void**);
}  // namespace bfa::testing
#endif
