// Where a conversation partner looks: at the player, except for looks away
// that are placed like an actor's: into a pause between phrases (a glance
// that is back as the words resume, or a thought that is held into the next
// phrase), at the start of a line, and on a large change of emotion. The
// line's emotion sets how likely each is and where it goes.
#ifndef INCLUDE_A2F_ANIMATION_GAZE_DIRECTOR_H_
#define INCLUDE_A2F_ANIMATION_GAZE_DIRECTOR_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace a2f {

enum GazeMood {
  kGazeNeutral,
  kGazeAngry,
  kGazeHappy,
  kGazeSad,  // sadness, grief, pain: also shame and embarrassment
  kGazeAfraid,
  kGazeSurprised,
  kGazeDisgusted,
  kNumGazeMoods
};
using GazeMoods = std::array<float, kNumGazeMoods>;

// The moods' shares (summing to 1) of the network's emotion (kNumMod values).
GazeMoods GazeMoodsOf(const float* emotion);

enum GazeBeatKind {
  kGazeStart,  // the line's first words
  kGazePause,  // a silence between phrases
  kGazeTurn,   // a large change of emotion within a phrase
};

// A moment the gaze may shift at. Times are seconds into the line.
struct GazeBeat {
  float time;       // a pause: where the silence begins
  GazeMoods moods;  // of the words that follow
  GazeBeatKind kind = kGazeStart;
  float resume = 0;      // where the words begin again (otherwise: time)
  float phrase_end = 0;  // where those words next stop
};

// A line's beats, in order: where speech starts, each pause between phrases,
// and where the emotion (kFps x kNumMod, may be empty) has changed a lot
// within a phrase. None for a silent clip.
void FindGazeBeats(const float* pcm, size_t n,
                   const std::vector<float>& emotion,
                   std::vector<GazeBeat>& beats);

// Whether a ray from the eyes toward something distance away reached it.
// hit: the fraction of the way where the ray stopped (1 or more: nowhere).
// The thing's own surface stops the ray up to its size short of its centre.
bool GazeRayReaches(float hit, float distance, float size);

constexpr GazeMoods kGazePlayerOdds = {0.7f, 0.9f, 0.55f, 0.15f,
                                       0.41f, 0.87f, 0.46f};
// The away multiplier for a "holds your eye" setting of 0..2: 1 leaves the
// per-mood odds as they are, 2 never looks away, 0 looks away twice as often.
constexpr float GazeAwayFromHold(float hold) {
  return hold == hold ? (hold < 0 ? 2.f : hold > 2 ? 0.f : 2 - hold) : 1.f;
}
struct GazeSettings {
  // The chance, per mood, that a beat leaves the eyes on the player. A line's
  // start looks away half as often as a pause does.
  GazeMoods player_odds = kGazePlayerOdds;
  // Multiplies the chance of every look away, and the rate of idle glances;
  // 0: the eyes never leave the player.
  float away = 1;
  // After the eyes come back to the player, they stay this long at least.
  float rest_seconds = 3;
  // A conversation opens on the player: no look away in its first seconds.
  // It starts with the first Update after construction or Reset.
  float settle_seconds = 3;
  // Typical seconds between idle glances past the player's head, outside the
  // beats, at away 1; 0 none.
  float glance_seconds = 16;
  // Objects far from the player's direction are looked at with the head.
  bool turn_head = true;
};

// Something to look at: a world position, and the reference it belongs to
// (0: none) for looking at it with the head.
struct GazePoint {
  float position[3];
  uint32_t ref = 0;
  bool actor = false;
};

// The speaker's eyes and the player's head, world positions; and the camera
// when it is not the player's own eyes. The speaker never looks at it: looks
// away keep clear of its direction, and with the camera beside the player's
// head the eyes rest a little to the head's other side.
struct GazeView {
  float eye[3];
  float player[3];
  float camera[3] = {};
  bool camera_known = false;
};

// The line being spoken: any value that differs between lines, seconds into
// it, its length, and its beats (may be null).
struct GazeLine {
  uint64_t id;
  float time, duration;
  const std::vector<GazeBeat>* beats;
};

// Where the eyes are, in degrees from the direction of the player (right and
// down positive), and the reference the head should follow (0: the player).
// While the head follows it the game aims the eyes too, and the offset is
// not needed.
struct GazeAim {
  float right = 0, down = 0;
  uint32_t head_ref = 0;
  bool on_player = true;
};

constexpr float kMaxGazeRight = 30, kMaxGazeDown = 24, kMaxGazeUp = 14;

class GazeDirector {
 public:
  explicit GazeDirector(uint32_t seed = 0x9E3779B9u) { Seed(seed); }
  void Seed(uint32_t seed) { random_ = seed ? seed : 1; }
  // Back to looking at the player with no line.
  void Reset();
  // now: any monotonic clock, in seconds. line: null between lines.
  GazeAim Update(double now, const GazeLine* line, const GazeView& view,
                 const std::vector<GazePoint>& points,
                 const GazeSettings& settings);

 private:
  enum Where { kDown, kSide, kPast, kLowPast, kUp, kObject, kNumWheres };
  float Uniform(float lo = 0, float hi = 1);
  float PlayerOdds(const GazeSettings& settings) const;
  // Whether a look away may start now, and by chance does.
  bool Leaves(double now, float chance, const GazeSettings& settings);
  void LookAtPlayer(double now, const GazeSettings& settings);
  // seconds: how long before the eyes come back. glance: with the eyes only.
  // idle: outside the beats, just past the player whatever the mood.
  void LookAway(double now, float seconds, bool glance, bool idle,
                const GazeView& view, const std::vector<GazePoint>& points,
                const GazeSettings& settings);
  void Shift(double now);

  uint32_t random_ = 1;
  GazeMoods moods_ = {1, 0, 0, 0, 0, 0, 0};
  uint64_t line_ = 0;
  float line_time_ = 0;
  size_t next_beat_ = 0;
  bool line_ending_ = false;
  double spoke_at_ = -1e9;
  // The fixation: the player, or a world position (and its reference).
  bool on_player_ = true, head_ = false, started_ = false;
  float target_[3] = {};
  uint32_t ref_ = 0;
  double until_ = 0;
  double returned_at_ = -1e9;  // from the last look away
  double conversation_at_ = 0;
  // The saccade to it: from these angles, over this long.
  float from_[2] = {}, last_[2] = {};
  double shift_at_ = -1e9;
  float shift_seconds_ = 0;
};

}  // namespace a2f

#endif  // INCLUDE_A2F_ANIMATION_GAZE_DIRECTOR_H_
