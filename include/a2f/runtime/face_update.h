#ifndef INCLUDE_A2F_RUNTIME_FACE_UPDATE_H_
#define INCLUDE_A2F_RUNTIME_FACE_UPDATE_H_

namespace bfa {

// A zero-time scene-graph refresh must not consume the frame's animation
// delta again. Keep the context local to this face, thread and nested call.
class ScopedFaceUpdate {
 public:
  ScopedFaceUpdate(const void* face, float time)
      : face_(face), refresh_(time == 0), previous_(current_) {
    current_ = this;
  }
  ~ScopedFaceUpdate() { current_ = previous_; }
  ScopedFaceUpdate(const ScopedFaceUpdate&) = delete;
  ScopedFaceUpdate& operator=(const ScopedFaceUpdate&) = delete;

  static float Delta(const void* face, float dt) {
    return current_ && face && current_->face_ == face && current_->refresh_
               ? 0.0f
               : dt;
  }

 private:
  const void* face_;
  bool refresh_;
  const ScopedFaceUpdate* previous_;
  inline static thread_local const ScopedFaceUpdate* current_ = nullptr;
};

}  // namespace bfa

#endif  // INCLUDE_A2F_RUNTIME_FACE_UPDATE_H_
