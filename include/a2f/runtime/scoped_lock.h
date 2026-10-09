#ifndef INCLUDE_A2F_RUNTIME_SCOPED_LOCK_H_
#define INCLUDE_A2F_RUNTIME_SCOPED_LOCK_H_

#include <windows.h>

namespace bfa {

// Owns one SRW lock. Unlock/Lock allow work outside a worker's critical
// section.
class ScopedLock {
 public:
  enum Mode { kExclusive, kShared };

  explicit ScopedLock(SRWLOCK& lock, Mode mode = kExclusive)
      : lock_(lock), mode_(mode) {
    Lock();
  }
  ~ScopedLock() { Unlock(); }
  ScopedLock(const ScopedLock&) = delete;
  ScopedLock& operator=(const ScopedLock&) = delete;

  void Lock() {
    if (mode_ == kShared)
      AcquireSRWLockShared(&lock_);
    else
      AcquireSRWLockExclusive(&lock_);
    locked_ = true;
  }
  void Unlock() {
    if (!locked_) return;
    if (mode_ == kShared)
      ReleaseSRWLockShared(&lock_);
    else
      ReleaseSRWLockExclusive(&lock_);
    locked_ = false;
  }

 private:
  SRWLOCK& lock_;
  Mode mode_;
  bool locked_ = false;
};

}  // namespace bfa

#endif  // INCLUDE_A2F_RUNTIME_SCOPED_LOCK_H_
