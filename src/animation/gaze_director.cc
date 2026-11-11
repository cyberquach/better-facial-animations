#include "a2f/animation/gaze_director.h"

#include <algorithm>
#include <cmath>
#include <iterator>

#include "a2f/inference/live_core.h"

namespace a2f {
namespace {

constexpr float kDegrees = 57.29578f;
constexpr GazeMoods kNeutralMoods = {1, 0, 0, 0, 0, 0, 0};

// Beats.
constexpr size_t kLevelHop = kSampleRate / 50;  // 20 ms
constexpr float kLevelSeconds = static_cast<float>(kLevelHop) / kSampleRate;
constexpr float kPauseSeconds = 0.35f;       // a silence between phrases
constexpr float kMinPhraseSeconds = 1;       // between a beat and a pause
constexpr float kEmotionBeatSeconds = 1.5f;  // into a phrase, and from its end
constexpr float kEmotionBeatChange = 0.9f;   // of 2: the moods' L1 distance
constexpr float kMoodSeconds = 0.5f;         // a beat's moods: the mean over

// Where a look away from the player goes, by mood: down, to the side, just
// past the player's head, past it and lower, up, at any object.
constexpr float kWhere[kNumGazeMoods][6] = {
    {0.10f, 0.15f, 0.30f, 0.15f, 0.10f, 0.20f},  // neutral
    {0.05f, 0.15f, 0.15f, 0.55f, 0.00f, 0.10f},  // angry: frustration
    {0.10f, 0.20f, 0.25f, 0.05f, 0.15f, 0.25f},  // happy
    {0.50f, 0.10f, 0.05f, 0.25f, 0.00f, 0.10f},  // sad
    {0.20f, 0.35f, 0.10f, 0.10f, 0.00f, 0.25f},  // afraid
    {0.05f, 0.20f, 0.30f, 0.05f, 0.15f, 0.25f},  // surprised
    {0.20f, 0.40f, 0.05f, 0.25f, 0.00f, 0.10f},  // disgusted
};
constexpr float kGlanceWhere[6] = {0, 0, 0.8f, 0.2f, 0, 0};
// Those directions, in degrees: to either side, and down; and how far off a
// look in one with nothing there rests, in distances to the player.
constexpr struct Zone {
  float side_lo, side_hi, down_lo, down_hi, depth;
} kZones[5] = {
    {4, 16, 16, kMaxGazeDown, 1.5f},   // down
    {16, kMaxGazeRight, -4, 8, 1.5f},  // side
    {5, 11, -4, 3, 2},                 // past
    {4, 12, 6, 15, 2},                 // past and lower
    {6, 18, -kMaxGazeUp, -8, 1.5f},    // up
};
constexpr float kZoneMargin = 3;  // an object this far outside still counts
// The chance, by mood, that a look away in a pause is a thought, held into
// the next phrase, and not a glance that is back as the words resume.
constexpr GazeMoods kThought = {0.5f, 0.25f, 0.4f, 0.75f, 0.25f, 0.3f, 0.5f};

// Looking away.
constexpr float kMinDistance = 100, kMaxDistance = 1500;  // game units
// The game aims both eyes at what the head follows: nearer than this they
// would visibly cross.
constexpr float kHeadMinDistance = 200;
constexpr float kMinSeparation = 4;    // degrees from the player, for objects
constexpr float kHeadSeparation = 12;  // from here the head turns too
constexpr float kMaxHeadRight = 55, kMaxHeadDown = 40, kMaxHeadUp = 25;
// Looks away keep this far, in degrees, from the direction of the camera, and
// go to the other side of the player from it; the eyes on the player, the
// second.
constexpr float kCameraDegrees = 20, kCameraPlayerDegrees = 8;
constexpr int kCameraTries = 8;
constexpr float kLineEndSeconds = 0.45f;  // the gaze comes back to the listener
constexpr float kMoodMemorySeconds = 2;   // after the line

// The speaker's horizontal frame toward the player.
struct Frame {
  bool ok = false;
  float forward[2], side[2];  // side: the speaker's right
  float player_down, distance;
  // The camera's angles from the player's direction, when it is in view.
  bool camera = false;
  float camera_right = 0, camera_down = 0;
  bool NearCamera(float right, float down, float degrees) const {
    return camera &&
           std::hypot(right - camera_right, down - camera_down) < degrees;
  }
};

float Angles(const Frame& frame, const float* eye, const float* position,
             float& right, float& down);

Frame FrameOf(const GazeView& view) {
  Frame frame;
  const float d[3] = {view.player[0] - view.eye[0],
                      view.player[1] - view.eye[1],
                      view.player[2] - view.eye[2]};
  const float flat = std::hypot(d[0], d[1]);
  if (!std::isfinite(flat) || !std::isfinite(d[2]) || flat < 1) return frame;
  frame.ok = true;
  frame.forward[0] = d[0] / flat;
  frame.forward[1] = d[1] / flat;
  // Heading runs clockwise seen from above: facing +Y, the right is +X.
  frame.side[0] = frame.forward[1];
  frame.side[1] = -frame.forward[0];
  frame.player_down = -std::atan2(d[2], flat) * kDegrees;
  frame.distance = std::hypot(flat, d[2]);
  if (view.camera_known && std::isfinite(view.camera[0]) &&
      std::isfinite(view.camera[1]) && std::isfinite(view.camera[2])) {
    const float distance = Angles(frame, view.eye, view.camera,
                                  frame.camera_right, frame.camera_down);
    frame.camera = distance >= 1 && std::abs(frame.camera_right) < 90;
  }
  return frame;
}

// The position's angles from the player's direction, and its distance.
float Angles(const Frame& frame, const float* eye, const float* position,
             float& right, float& down) {
  const float d[3] = {position[0] - eye[0], position[1] - eye[1],
                      position[2] - eye[2]};
  const float forward = d[0] * frame.forward[0] + d[1] * frame.forward[1];
  const float side = d[0] * frame.side[0] + d[1] * frame.side[1];
  const float flat = std::hypot(forward, side);
  right = std::atan2(side, forward) * kDegrees;
  down = -std::atan2(d[2], flat) * kDegrees - frame.player_down;
  return std::hypot(flat, d[2]);
}

void PositionAt(const Frame& frame, const float* eye, float right, float down,
                float distance, float* position) {
  const float yaw = right / kDegrees;
  const float pitch = (frame.player_down + down) / kDegrees;
  const float flat = std::cos(pitch) * distance;
  for (int axis = 0; axis < 2; ++axis)
    position[axis] = eye[axis] + flat * (frame.forward[axis] * std::cos(yaw) +
                                         frame.side[axis] * std::sin(yaw));
  position[2] = eye[2] - std::sin(pitch) * distance;
}

bool Finite(const float* position) {
  return std::isfinite(position[0]) && std::isfinite(position[1]) &&
         std::isfinite(position[2]);
}

}  // namespace

GazeMoods GazeMoodsOf(const float* emotion) {
  auto level = [&](int channel) {
    const float value = emotion[channel];
    return std::isfinite(value) ? std::clamp(value, 0.f, 1.f) : 0.f;
  };
  GazeMoods moods = {};
  moods[kGazeAngry] = level(kAnger);
  moods[kGazeHappy] = std::max(level(kJoy), level(kCheekiness));
  moods[kGazeSad] =
      std::max({level(kSadness), level(kGrief), 0.7f * level(kPain)});
  moods[kGazeAfraid] = level(kFear);
  moods[kGazeSurprised] = level(kAmazement);
  moods[kGazeDisgusted] = level(kDisgust);
  float sum = 0;
  for (float mood : moods) sum += mood;
  if (sum > 1)
    for (float& mood : moods) mood /= sum;
  moods[kGazeNeutral] = 1 - std::min(sum, 1.f);
  return moods;
}

void FindGazeBeats(const float* pcm, size_t n,
                   const std::vector<float>& emotion,
                   std::vector<GazeBeat>& beats) {
  beats.clear();
  const size_t windows = pcm ? n / kLevelHop : 0;
  if (!windows) return;
  std::vector<float> level(windows);
  float peak = 0;
  for (size_t w = 0; w < windows; ++w) {
    double sum = 0;
    for (size_t i = 0; i < kLevelHop; ++i) {
      const double sample = pcm[w * kLevelHop + i];
      if (std::isfinite(sample)) sum += sample * sample;
    }
    level[w] = static_cast<float>(std::sqrt(sum / kLevelHop));
    peak = std::max(peak, level[w]);
  }
  if (peak < 1e-4f) return;
  const float threshold = std::max(peak * 0.1f, 1e-4f);
  const size_t frames = emotion.size() / kNumMod;
  auto moods_at = [&](float time) {
    const size_t first = static_cast<size_t>(time * kFps);
    const size_t last =
        std::min(frames, first + static_cast<size_t>(kMoodSeconds * kFps));
    if (first >= last)
      return frames ? GazeMoodsOf(&emotion[(frames - 1) * kNumMod])
                    : kNeutralMoods;
    GazeMoods mean = {};
    for (size_t frame = first; frame < last; ++frame) {
      const GazeMoods moods = GazeMoodsOf(&emotion[frame * kNumMod]);
      for (int mood = 0; mood < kNumGazeMoods; ++mood)
        mean[mood] += moods[mood] / (last - first);
    }
    return mean;
  };
  // The phrases: runs of speech, joined over silences shorter than a pause.
  struct Phrase {
    float start, end;
  };
  std::vector<Phrase> phrases;
  for (size_t w = 0; w < windows; ++w) {
    if (level[w] <= threshold) continue;
    const float time = w * kLevelSeconds;
    if (phrases.empty() || time - phrases.back().end >= kPauseSeconds)
      phrases.push_back({time, time});
    phrases.back().end = time + kLevelSeconds;
  }
  for (size_t i = 0; i < phrases.size(); ++i) {
    const Phrase& phrase = phrases[i];
    if (!i) {
      beats.push_back({phrase.start, moods_at(phrase.start), kGazeStart,
                       phrase.start, phrase.end});
    } else if (phrase.start - beats.back().resume >= kMinPhraseSeconds) {
      beats.push_back({phrases[i - 1].end, moods_at(phrase.start), kGazePause,
                       phrase.start, phrase.end});
    } else {
      beats.back().phrase_end = phrase.end;  // one stuttered phrase
    }
    // The turns of emotion within it, clear of its ends.
    for (float time = phrase.start; time + kEmotionBeatSeconds <= phrase.end;
         time += kLevelSeconds) {
      if (time - beats.back().resume < kEmotionBeatSeconds) continue;
      const GazeMoods moods = moods_at(time);
      float change = 0;
      for (int mood = 0; mood < kNumGazeMoods; ++mood)
        change += std::abs(moods[mood] - beats.back().moods[mood]);
      // The beat is the turn toward the new emotion: its moods are from there.
      if (change >= kEmotionBeatChange)
        beats.push_back(
            {time, moods_at(time + kMoodSeconds), kGazeTurn, time, phrase.end});
    }
  }
}

bool GazeRayReaches(float hit, float distance, float size) {
  constexpr float kMargin = 10;  // game units
  if (!std::isfinite(hit) || !std::isfinite(distance) || hit < 0 ||
      distance < 0)
    return false;
  if (hit >= 1) return true;
  return hit * distance >=
         distance - std::max(std::isfinite(size) ? size : 0.f, kMargin);
}

void GazeDirector::Reset() {
  const uint32_t random = random_;
  *this = GazeDirector(random);
}

float GazeDirector::Uniform(float lo, float hi) {
  random_ ^= random_ << 13;
  random_ ^= random_ >> 17;
  random_ ^= random_ << 5;
  return lo + (hi - lo) * ((random_ >> 8) * (1.f / 16777216.f));
}

float GazeDirector::PlayerOdds(const GazeSettings& settings) const {
  float odds = 0;
  for (int mood = 0; mood < kNumGazeMoods; ++mood) {
    const float chance = settings.player_odds[mood];
    odds += moods_[mood] *
            (std::isfinite(chance) ? std::clamp(chance, 0.f, 1.f) : 1.f);
  }
  return odds;
}

void GazeDirector::Shift(double now) {
  from_[0] = last_[0];
  from_[1] = last_[1];
  shift_at_ = now;
  shift_seconds_ = -1;  // set from the size of the movement
}

bool GazeDirector::Leaves(double now, float chance,
                          const GazeSettings& settings) {
  const float away = std::isfinite(settings.away) ? settings.away : 1;
  const float roll = Uniform();  // drawn either way: one per beat
  return on_player_ && now - returned_at_ >= settings.rest_seconds &&
         now - conversation_at_ >= settings.settle_seconds &&
         roll < chance * away;
}

void GazeDirector::LookAtPlayer(double now, const GazeSettings& settings) {
  if (!on_player_) {
    Shift(now);
    returned_at_ = now;
  }
  on_player_ = true;
  head_ = false;
  ref_ = 0;
  // Those who look at the player less also glance away sooner.
  until_ = settings.glance_seconds > 0 && settings.away > 0
               ? now + settings.glance_seconds / settings.away *
                           Uniform(0.5f, 1.5f) *
                           (0.4f + 1.2f * PlayerOdds(settings))
               : now + 1;
}

void GazeDirector::LookAway(double now, float seconds, bool glance, bool idle,
                            const GazeView& view,
                            const std::vector<GazePoint>& points,
                            const GazeSettings& settings) {
  static_assert(std::size(kWhere[0]) == kNumWheres &&
                std::size(kGlanceWhere) == kNumWheres &&
                std::size(kZones) == kObject);
  const Frame frame = FrameOf(view);
  const bool may_turn = settings.turn_head && !glance;
  struct Candidate {
    const GazePoint* point;
    float right, down, weight;
    bool head;
  };
  std::vector<Candidate> candidates;
  for (const GazePoint& point : points) {
    if (!Finite(point.position)) continue;
    Candidate c{&point};
    const float distance =
        Angles(frame, view.eye, point.position, c.right, c.down);
    const float separation = std::hypot(c.right, c.down);
    if (distance < kMinDistance || distance > kMaxDistance ||
        separation < kMinSeparation ||
        frame.NearCamera(c.right, c.down, kCameraDegrees))
      continue;
    const bool eyes = std::abs(c.right) <= kMaxGazeRight &&
                      c.down <= kMaxGazeDown && c.down >= -kMaxGazeUp;
    c.head = may_turn && point.ref && distance >= kHeadMinDistance &&
             separation >= kHeadSeparation &&
             std::abs(c.right) <= kMaxHeadRight && c.down <= kMaxHeadDown &&
             c.down >= -kMaxHeadUp;
    if (!eyes && !c.head) continue;
    c.weight = (point.actor ? 3.f : 1.f) / (1 + separation / 25);
    candidates.push_back(c);
  }
  float weights[kNumWheres] = {};
  for (int where = 0; where < kNumWheres; ++where) {
    if (idle) {
      weights[where] = kGlanceWhere[where];
      continue;
    }
    for (int mood = 0; mood < kNumGazeMoods; ++mood)
      weights[where] += moods_[mood] * kWhere[mood][where];
  }
  if (candidates.empty()) weights[kObject] = 0;
  float sum = 0;
  for (float weight : weights) sum += weight;
  float pick = Uniform(0, sum);
  int where = kPast;
  for (int i = 0; i < kNumWheres; ++i) {
    if (pick < weights[i]) {
      where = i;
      break;
    }
    pick -= weights[i];
  }
  // The eyes rest on something that is there: of the things in that
  // direction, or of them all. Only with nothing there, on the direction.
  const Zone* zone = where == kObject ? nullptr : &kZones[where];
  auto within = [&](const Candidate& c) {
    if (!zone) return true;
    const float side = std::abs(c.right);
    return side >= zone->side_lo - kZoneMargin &&
           side <= zone->side_hi + kZoneMargin &&
           c.down >= zone->down_lo - kZoneMargin &&
           c.down <= zone->down_hi + kZoneMargin;
  };
  float total = 0;
  for (const Candidate& c : candidates)
    if (within(c)) total += c.weight;
  const Candidate* chosen = nullptr;
  if (total > 0) {
    float at = Uniform(0, total);
    for (const Candidate& c : candidates) {
      if (!within(c)) continue;
      chosen = &c;
      if (at < c.weight) break;
      at -= c.weight;
    }
  }
  Shift(now);
  on_player_ = false;
  if (chosen) {
    std::copy_n(chosen->point->position, 3, target_);
    ref_ = chosen->point->ref;
    head_ = chosen->head;
  } else {
    if (!zone) zone = &kZones[kPast];
    float sign = Uniform() < 0.5f ? -1.f : 1.f;
    // Never toward the camera's side of the player.
    const bool sided = frame.camera && std::abs(frame.camera_right) >= 0.5f;
    if (sided) sign = frame.camera_right > 0 ? -1.f : 1.f;
    float right = 0, down = 0;
    bool clear = false;
    // A zone that is all too near the camera: further aside instead.
    for (int pass = 0; pass < 2 && !clear; ++pass) {
      if (pass) zone = &kZones[kSide];
      for (int attempt = 0; attempt < kCameraTries && !clear; ++attempt) {
        right = sign * Uniform(zone->side_lo, zone->side_hi);
        down = Uniform(zone->down_lo, zone->down_hi);
        clear = !frame.NearCamera(right, down, kCameraDegrees);
        if (!sided) sign = -sign;
      }
    }
    if (!clear) {  // nowhere there that is not the camera: no look away
      LookAtPlayer(now, settings);
      return;
    }
    PositionAt(frame, view.eye, right, down, frame.distance * zone->depth,
               target_);
    ref_ = 0;
    head_ = false;
  }
  until_ = now + seconds;
}

GazeAim GazeDirector::Update(double now, const GazeLine* line,
                             const GazeView& view,
                             const std::vector<GazePoint>& points,
                             const GazeSettings& settings) {
  const Frame frame = FrameOf(view);
  if (!frame.ok || !Finite(view.eye) || !std::isfinite(now)) {
    Reset();
    return {};
  }
  if (!started_) {
    started_ = true;
    conversation_at_ = now;
    LookAtPlayer(now, settings);
  }
  const GazeBeat* beat = nullptr;
  if (line) {
    if (line->id != line_ || line->time < line_time_ - 0.5f) {
      line_ = line->id;
      next_beat_ = 0;
      line_ending_ = false;
    }
    line_time_ = line->time;
    spoke_at_ = now;
    if (line->beats) {
      for (; next_beat_ < line->beats->size() &&
             (*line->beats)[next_beat_].time <= line->time;
           ++next_beat_)
        beat = &(*line->beats)[next_beat_];
    }
  } else {
    line_ = 0;
    if (now - spoke_at_ > kMoodMemorySeconds) moods_ = kNeutralMoods;
  }
  if (beat) moods_ = beat->moods;
  const float odds = PlayerOdds(settings);
  auto positive = [](float seconds) {
    return std::isfinite(seconds) ? std::max(seconds, 0.f) : 0.f;
  };
  if (beat && beat->kind == kGazeStart) {
    // Some lines open in thought and come to the listener within the first
    // phrase; the rest open on the listener.
    if (!on_player_) LookAtPlayer(now, settings);
    if (Leaves(now, 0.5f * (1 - odds), settings))
      LookAway(now,
               std::clamp(Uniform(0.25f, 0.5f) *
                              positive(beat->phrase_end - beat->time),
                          0.6f, 3.f),
               false, false, view, points, settings);
  } else if (beat && beat->kind == kGazePause) {
    // The pause is where the eyes leave: for the silence alone, or in
    // thought, coming back part of the way through the words that follow.
    // A look that is already away is left to finish.
    if (Leaves(now, 1 - odds, settings)) {
      float thought = 0;
      for (int mood = 0; mood < kNumGazeMoods; ++mood)
        thought += moods_[mood] * kThought[mood];
      const float silence = positive(beat->resume - line->time);
      if (Uniform() < thought)
        LookAway(
            now,
            std::clamp(silence + Uniform(0.35f, 0.65f) *
                                     positive(beat->phrase_end - beat->resume),
                       0.8f, 5.f),
            false, false, view, points, settings);
      else
        LookAway(now, std::clamp(silence + Uniform(0.05f, 0.25f), 0.6f, 1.5f),
                 true, false, view, points, settings);
    }
  } else if (beat) {
    // A turn of emotion pulls the eyes to the listener, or briefly off them.
    if (Leaves(now, 1 - odds, settings))
      LookAway(now, Uniform(0.6f, 1.2f), true, false, view, points, settings);
    else if (!on_player_ && Uniform() < odds)
      LookAtPlayer(now, settings);
  } else if (line && !line_ending_ &&
             line->duration - line->time < kLineEndSeconds) {
    // A speaker's gaze returns to the listener as the turn ends.
    line_ending_ = true;
    if (!on_player_ && Uniform() < 0.5f + 0.5f * odds)
      LookAtPlayer(now, settings);
  } else if (now >= until_) {
    if (!on_player_)
      LookAtPlayer(now, settings);
    else if (!(settings.glance_seconds > 0 && settings.away > 0))
      until_ = now + 1;
    else if (now - returned_at_ < settings.rest_seconds)
      until_ = returned_at_ + settings.rest_seconds;
    else if (now - conversation_at_ < settings.settle_seconds)
      until_ = conversation_at_ + settings.settle_seconds;
    else
      LookAway(now, Uniform(0.5f, 1.2f), true, true, view, points, settings);
  }
  float want[2] = {0, 0};
  if (!on_player_) {
    Angles(frame, view.eye, target_, want[0], want[1]);
    // The camera has cut to where the eyes were resting.
    if (frame.NearCamera(want[0], want[1], kCameraDegrees))
      LookAtPlayer(now, settings);
  }
  if (on_player_) {
    want[0] = want[1] = 0;
    // With the camera beside the player's head, the eyes rest on the head's
    // other side: an eyeline past the lens, not into it.
    if (frame.NearCamera(0, 0, kCameraPlayerDegrees)) {
      float away[2] = {-frame.camera_right, -frame.camera_down};
      const float apart = std::hypot(away[0], away[1]);
      if (apart < 0.5f) {
        away[0] = 0;
        away[1] = 1;  // the camera is the head: below it
      } else {
        away[0] /= apart;
        away[1] /= apart;
      }
      want[0] = away[0] * (kCameraPlayerDegrees - apart);
      want[1] = away[1] * (kCameraPlayerDegrees - apart);
    }
  } else {
    // The two have moved and the eyes alone no longer reach it.
    if (!head_ &&
        (std::abs(want[0]) > 1.5f * kMaxGazeRight ||
         want[1] > 1.5f * kMaxGazeDown || want[1] < -1.5f * kMaxGazeUp)) {
      LookAtPlayer(now, settings);
      want[0] = want[1] = 0;
    } else {
      want[0] = std::clamp(want[0], -kMaxGazeRight, kMaxGazeRight);
      want[1] = std::clamp(want[1], -kMaxGazeUp, kMaxGazeDown);
    }
  }
  // A saccade: quick, and longer the further the eyes travel.
  if (shift_seconds_ < 0)
    shift_seconds_ =
        0.03f + 0.0022f * std::hypot(want[0] - from_[0], want[1] - from_[1]);
  const float moved =
      Smoothstep(static_cast<float>(now - shift_at_) / shift_seconds_);
  for (int axis = 0; axis < 2; ++axis)
    last_[axis] = from_[axis] + (want[axis] - from_[axis]) * moved;
  GazeAim aim;
  aim.right = last_[0];
  aim.down = last_[1];
  aim.on_player = on_player_;
  aim.head_ref = !on_player_ && head_ ? ref_ : 0;
  return aim;
}

}  // namespace a2f
