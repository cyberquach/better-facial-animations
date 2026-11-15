#include "a2f/animation/gaze_director.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "a2f/inference/live_core.h"

#define CHECK(x)                                      \
  do {                                                \
    if (!(x)) {                                       \
      fprintf(stderr, "FAIL %d: %s\n", __LINE__, #x); \
      std::exit(1);                                   \
    }                                                 \
  } while (0)

namespace {
using namespace a2f;

// The speaker at the origin's eye height, the player 120 units north.
const GazeView kView = {{0, 0, 120}, {0, 120, 120}};
constexpr double kFrame = 1.0 / 60;

// The tests below look from the first moment of a conversation.
GazeSettings Unsettled() {
  GazeSettings settings;
  settings.settle_seconds = 0;
  return settings;
}

GazeMoods Only(GazeMood mood) {
  GazeMoods moods = {};
  moods[mood] = 1;
  return moods;
}

std::vector<float> Tone(float seconds) {
  std::vector<float> pcm(static_cast<size_t>(seconds * kSampleRate));
  for (size_t i = 0; i < pcm.size(); ++i) pcm[i] = 0.3f * std::sin(i * 0.05f);
  return pcm;
}

void Moods() {
  float emotion[kNumMod] = {};
  GazeMoods moods = GazeMoodsOf(emotion);
  CHECK(moods[kGazeNeutral] == 1);
  emotion[kAnger] = 2;  // the network's range reaches 2
  moods = GazeMoodsOf(emotion);
  CHECK(moods[kGazeAngry] == 1 && moods[kGazeNeutral] == 0);
  emotion[kAnger] = 0.25f;
  emotion[kGrief] = 0.25f;
  emotion[kFear] = NAN;
  moods = GazeMoodsOf(emotion);
  CHECK(moods[kGazeAngry] == 0.25f && moods[kGazeSad] == 0.25f &&
        moods[kGazeAfraid] == 0 && moods[kGazeNeutral] == 0.5f);
  emotion[kJoy] = emotion[kAnger] = emotion[kGrief] = 1;
  moods = GazeMoodsOf(emotion);
  float sum = 0;
  for (float mood : moods) sum += mood;
  CHECK(std::abs(sum - 1) < 1e-5f && moods[kGazeNeutral] == 0);
}

void Beats() {
  std::vector<GazeBeat> beats = {{1, {}}};
  const std::vector<float> none;
  // Silence and missing audio have no beats.
  std::vector<float> pcm(kSampleRate, 0.f);
  FindGazeBeats(pcm.data(), pcm.size(), none, beats);
  CHECK(beats.empty());
  FindGazeBeats(nullptr, 100000, none, beats);
  CHECK(beats.empty());
  // Two phrases 0.4 s apart: the start, and the pause between them. A 0.1 s
  // gap is not one.
  pcm = std::vector<float>(kSampleRate / 2, 0.f);
  const std::vector<float> tone = Tone(1);
  pcm.insert(pcm.end(), tone.begin(), tone.end());
  pcm.insert(pcm.end(), kSampleRate / 10, 0.f);
  pcm.insert(pcm.end(), tone.begin(), tone.end());
  pcm.insert(pcm.end(), kSampleRate * 4 / 10, 0.f);
  pcm.insert(pcm.end(), tone.begin(), tone.end());
  FindGazeBeats(pcm.data(), pcm.size(), none, beats);
  CHECK(beats.size() == 2);
  CHECK(beats[0].kind == kGazeStart && std::abs(beats[0].time - 0.5f) < 0.03f);
  CHECK(std::abs(beats[0].phrase_end - 2.6f) < 0.03f);
  CHECK(beats[1].kind == kGazePause && std::abs(beats[1].time - 2.6f) < 0.03f);
  CHECK(std::abs(beats[1].resume - 3) < 0.03f);
  CHECK(std::abs(beats[1].phrase_end - 4) < 0.03f);
  CHECK(beats[0].moods[kGazeNeutral] == 1);
  // Short phrases in a row are one: no pause beat within a second of a beat.
  pcm.clear();
  for (int phrase = 0; phrase < 3; ++phrase) {
    pcm.insert(pcm.end(), tone.begin(), tone.begin() + kSampleRate / 4);
    pcm.insert(pcm.end(), kSampleRate * 4 / 10, 0.f);
  }
  FindGazeBeats(pcm.data(), pcm.size(), none, beats);
  CHECK(beats.size() == 2 && beats[1].kind == kGazePause);
  CHECK(std::abs(beats[1].resume - 1.3f) < 0.03f);
  // One unbroken phrase that turns from anger to grief: a beat at the turn,
  // carrying the new mood.
  pcm = Tone(4);
  std::vector<float> emotion(4 * kFps * kNumMod);
  for (int frame = 0; frame < 4 * kFps; ++frame)
    emotion[frame * kNumMod + (frame < 2 * kFps ? kAnger : kGrief)] = 1;
  FindGazeBeats(pcm.data(), pcm.size(), emotion, beats);
  CHECK(beats.size() == 2);
  CHECK(beats[0].time < 0.03f && beats[0].moods[kGazeAngry] == 1);
  CHECK(beats[1].kind == kGazeTurn);
  CHECK(beats[1].time > 1.5f && beats[1].time < 2.1f);
  CHECK(beats[1].moods[kGazeSad] > 0.7f);
  // A partial change is not a turn, and a steady emotion adds none.
  for (int frame = 2 * kFps; frame < 4 * kFps; ++frame) {
    emotion[frame * kNumMod + kAnger] = 0.7f;
    emotion[frame * kNumMod + kGrief] = 0.3f;
  }
  FindGazeBeats(pcm.data(), pcm.size(), emotion, beats);
  CHECK(beats.size() == 1);
  for (int frame = 0; frame < 4 * kFps; ++frame) {
    emotion[frame * kNumMod + kAnger] = 1;
    emotion[frame * kNumMod + kGrief] = 0;
  }
  FindGazeBeats(pcm.data(), pcm.size(), emotion, beats);
  CHECK(beats.size() == 1);
}

// A pause from 1 to 1.5 s, then words until 4.5 s.
std::vector<GazeBeat> Pause(GazeMood mood) {
  return {{1, Only(mood), kGazePause, 1.5f, 4.5f}};
}

// Of pauses in one mood: the share that leave the eyes on the player, and of
// the looks away, the shares at the floor and low just past the player.
void BeatOdds(GazeMood mood, const GazeSettings& settings, float& player,
              float& down, float& low_past) {
  const std::vector<GazeBeat> beats = Pause(mood);
  const std::vector<GazePoint> points;
  int on_player = 0, away = 0, lowered = 0, past = 0;
  constexpr int kTrials = 2000;
  for (int trial = 0; trial < kTrials; ++trial) {
    GazeDirector director(trial * 7919 + 1);
    GazeLine line{1, 0.99f, 6, &beats};
    director.Update(0, &line, kView, points, settings);
    GazeAim aim;
    // The pause, then long enough for the saccade to land.
    for (int frame = 1; frame <= 20; ++frame) {
      line.time = 1 + frame * static_cast<float>(kFrame);
      aim = director.Update(frame * kFrame, &line, kView, points, settings);
    }
    if (aim.on_player) {
      CHECK(aim.right == 0 && aim.down == 0);
      ++on_player;
    } else {
      ++away;
      lowered += aim.down > 15.5f;
      past +=
          std::abs(aim.right) < 12.5f && aim.down > 5.5f && aim.down < 15.5f;
    }
  }
  player = static_cast<float>(on_player) / kTrials;
  down = away ? static_cast<float>(lowered) / away : 0;
  low_past = away ? static_cast<float>(past) / away : 0;
}

void Odds() {
  GazeSettings settings = Unsettled();
  settings.glance_seconds = 0;
  float player, down, low_past;
  BeatOdds(kGazeAngry, settings, player, down, low_past);
  CHECK(std::abs(player - kGazePlayerOdds[kGazeAngry]) < 0.03f);
  BeatOdds(kGazeHappy, settings, player, down, low_past);
  CHECK(std::abs(player - kGazePlayerOdds[kGazeHappy]) < 0.05f);
  BeatOdds(kGazeSad, settings, player, down, low_past);
  CHECK(std::abs(player - kGazePlayerOdds[kGazeSad]) < 0.05f);
  CHECK(down > 0.45f);  // the embarrassed look at the floor
  // The menu's chances are the odds.
  settings.player_odds[kGazeAngry] = 0;
  BeatOdds(kGazeAngry, settings, player, down, low_past);
  CHECK(player == 0);
  CHECK(low_past > 0.5f && down < 0.15f);  // frustration: low, past the player
  settings.player_odds[kGazeSad] = 1;
  BeatOdds(kGazeSad, settings, player, down, low_past);
  CHECK(player == 1);
}

// A look away in a pause comes back once: as the words resume, or, held in
// thought, part of the way through them. Nothing else moves the eyes until
// the line's last moments.
void Thought() {
  GazeSettings settings = Unsettled();
  settings.glance_seconds = 0;
  settings.player_odds = {};
  const std::vector<GazePoint> points;
  const float step = static_cast<float>(kFrame);
  for (GazeMood mood : {kGazeSad, kGazeAngry}) {
    const std::vector<GazeBeat> beats = Pause(mood);
    int glances = 0, thoughts = 0;
    for (int seed = 1; seed <= 400; ++seed) {
      GazeDirector director(seed);
      GazeLine line{1, 0, 6, &beats};
      float left = -1, back = -1;
      for (int frame = 0; frame * step < 5.4f; ++frame) {
        line.time = frame * step;
        const GazeAim aim =
            director.Update(frame * kFrame, &line, kView, points, settings);
        if (line.time < 0.99f) CHECK(aim.on_player);
        if (!aim.on_player && left < 0) left = line.time;
        if (aim.on_player && left >= 0 && back < 0) back = line.time;
        if (back >= 0) CHECK(aim.on_player);  // and stays
      }
      CHECK(left > 0.99f && left < 1.05f);
      if (back < 1.8f) {
        CHECK(back > 1.5f);  // out for the silence
        ++glances;
      } else {
        CHECK(back > 2.5f && back < 3.5f);  // into the phrase, not to its end
        ++thoughts;
      }
    }
    CHECK(glances > 40 && thoughts > 40);
    // The sad drift into thought, the angry look away and straight back.
    CHECK(mood == kGazeSad ? thoughts > 2 * glances : glances > 2 * thoughts);
  }
  // A line opens on the listener more often than a pause leaves them, and a
  // look away at its start is back within the first phrase.
  const std::vector<GazeBeat> beats = {{0, Only(kGazeHappy), kGazeStart, 0, 4}};
  settings = Unsettled();
  settings.glance_seconds = 0;
  int away = 0;
  for (int seed = 1; seed <= 2000; ++seed) {
    GazeDirector director(seed * 31);
    GazeLine line{1, 0, 6, &beats};
    bool left = false;
    for (int frame = 0; frame * step < 2.2f; ++frame) {
      line.time = frame * step;
      const GazeAim aim =
          director.Update(frame * kFrame, &line, kView, points, settings);
      left |= !aim.on_player;
      if (line.time > 2.1f) CHECK(aim.on_player);
    }
    away += left;
  }
  CHECK(std::abs(away / 2000.f - 0.5f * (1 - kGazePlayerOdds[kGazeHappy])) <
        0.04f);
}

// After a look away the eyes stay on the player for the rest time, however
// many pauses and turns come; the frequency scales every look away, and at 0
// there are none.
void Rest() {
  std::vector<GazeBeat> beats;
  for (int i = 0; i < 40; ++i)
    beats.push_back({1 + i * 1.5f, Only(kGazeSad),
                     i % 2 ? kGazeTurn : kGazePause, 1.3f + i * 1.5f,
                     2.4f + i * 1.5f});
  const std::vector<GazePoint> points;
  int looks[3] = {};
  for (int pass = 0; pass < 3; ++pass) {
    GazeSettings settings = Unsettled();
    settings.player_odds = {};
    settings.glance_seconds = 0;
    settings.rest_seconds = 4;
    settings.away = pass == 2 ? 0 : pass == 1 ? 0.15f : 1;
    for (int seed = 1; seed <= 40; ++seed) {
      GazeDirector director(seed);
      GazeLine line{1, 0, 62, &beats};
      bool was_on_player = true;
      double returned = -1e9;
      for (int frame = 0; frame < 61 * 60; ++frame) {
        const double now = frame * kFrame;
        line.time = static_cast<float>(now);
        const GazeAim aim =
            director.Update(now, &line, kView, points, settings);
        if (was_on_player && !aim.on_player) {
          CHECK(now - returned >= settings.rest_seconds - 0.02);
          ++looks[pass];
        }
        if (!was_on_player && aim.on_player) returned = now;
        was_on_player = aim.on_player;
      }
    }
  }
  // 61 s with a 4 s rest: at most one look in about every 4.6 s.
  CHECK(looks[0] > 40 * 6 && looks[0] <= 40 * 14);
  CHECK(looks[1] < looks[0] * 0.7);
  CHECK(looks[2] == 0);
  // A conversation opens on the player: nothing in its first 3 s, by
  // default; and again after a reset.
  GazeSettings settings;
  settings.player_odds = {};
  settings.glance_seconds = 0.5f;
  settings.rest_seconds = 0;
  GazeDirector director(9);
  GazeLine line{1, 0, 62, &beats};
  int early = 0, late = 0;
  for (double start : {100.0, 130.0}) {
    director.Reset();
    for (int frame = 0; frame < 20 * 60; ++frame) {
      const double now = start + frame * kFrame;
      line.time = static_cast<float>(frame * kFrame);
      const GazeAim aim = director.Update(now, &line, kView, points, settings);
      (frame * kFrame < 2.98 ? early : late) += !aim.on_player;
    }
  }
  CHECK(early == 0 && late > 200);
}

// Listening, with no line: mostly on the player, with short glances that
// stay near the player's head, each a quick continuous movement.
void Glances() {
  GazeSettings settings = Unsettled();
  const std::vector<GazePoint> points;
  GazeDirector director(3);
  int on_player = 0, glances = 0, frames = 0;
  bool was_on_player = true;
  GazeAim last;
  for (double now = 0; now < 300; now += kFrame, ++frames) {
    const GazeAim aim = director.Update(now, nullptr, kView, points, settings);
    CHECK(aim.head_ref == 0);
    CHECK(std::abs(aim.right) <= 12.5f && aim.down <= 15.5f &&
          aim.down >= -4.5f);
    CHECK(std::hypot(aim.right - last.right, aim.down - last.down) < 8);
    on_player += aim.on_player;
    glances += was_on_player && !aim.on_player;
    was_on_player = aim.on_player;
    last = aim;
  }
  CHECK(on_player > frames * 0.9);
  CHECK(glances > 10 && glances < 50);
  // Without glances the eyes never leave.
  settings.glance_seconds = 0;
  director.Reset();
  for (double now = 0; now < 60; now += kFrame) {
    const GazeAim aim = director.Update(now, nullptr, kView, points, settings);
    CHECK(aim.on_player && aim.right == 0 && aim.down == 0);
  }
}

// Glances land on something that is just past the player's head, and the
// eyes stay on it while the speaker moves.
void Snap() {
  GazeSettings settings = Unsettled();
  // 8 degrees to the speaker's right (east when facing north), 240 away.
  const float angle = 8 / 57.29578f;
  const std::vector<GazePoint> points = {
      {{240 * std::sin(angle), 240 * std::cos(angle), 120}, 5, false}};
  GazeDirector director(11);
  int landed = 0;
  GazeView view = kView;
  for (double now = 0; now < 300; now += kFrame) {
    const GazeAim aim = director.Update(now, nullptr, view, points, settings);
    CHECK(aim.head_ref == 0);  // a glance never turns the head
    if (std::abs(aim.right - 8) < 0.01f && std::abs(aim.down) < 0.01f) ++landed;
  }
  CHECK(landed > 100);
  // Sidestepping changes the angle to the same point.
  settings.player_odds = {};
  const std::vector<GazeBeat> beats = {
      {0, Only(kGazeNeutral), kGazeStart, 0, 9}};
  bool followed = false;
  for (int seed = 1; seed < 200 && !followed; ++seed) {
    director = GazeDirector(seed);
    GazeLine line{1, 0, 9, &beats};
    GazeAim aim;
    for (int frame = 0; frame < 30; ++frame)
      aim = director.Update(frame * kFrame, &line, kView, points, settings);
    if (std::abs(aim.right - 8) > 0.01f) continue;
    view = kView;
    view.eye[0] = -10;
    view.player[0] = -10;
    aim = director.Update(31 * kFrame, &line, view, points, settings);
    CHECK(aim.right > 9.5f && aim.right < 12);
    followed = true;
  }
  CHECK(followed);
}

// An object well to the side is looked at with the head, when allowed; the
// eyes alone never leave their range.
void Head() {
  GazeSettings settings = Unsettled();
  settings.player_odds = {};
  settings.glance_seconds = 0;
  const float angle = 40 / 57.29578f;
  const std::vector<GazePoint> points = {
      {{300 * std::sin(angle), 300 * std::cos(angle), 100}, 9, false},
      {{0, -300, 120}, 10, false},  // behind the speaker: never
      // As far to the side but within arm's reach: the eyes would cross.
      {{-150 * std::sin(angle), 150 * std::cos(angle), 100}, 11, false},
      {{60 * std::sin(angle / 2), 60 * std::cos(angle / 2), 110}, 12, false}};
  const std::vector<GazeBeat> beats = {{0, Only(kGazeHappy), kGazeStart, 0, 9}};
  for (int turn = 0; turn < 2; ++turn) {
    settings.turn_head = turn;
    int turned = 0;
    for (int seed = 1; seed <= 300; ++seed) {
      GazeDirector director(seed);
      GazeLine line{1, 0, 9, &beats};
      for (int frame = 0; frame < 240; ++frame) {
        line.time = frame * static_cast<float>(kFrame);
        const GazeAim aim =
            director.Update(frame * kFrame, &line, kView, points, settings);
        CHECK(aim.head_ref == 0 || aim.head_ref == 9);
        CHECK(std::abs(aim.right) <= kMaxGazeRight + 1e-3f &&
              aim.down <= kMaxGazeDown + 1e-3f &&
              aim.down >= -kMaxGazeUp - 1e-3f);
        turned += aim.head_ref == 9;
      }
    }
    CHECK(turn ? turned > 0 : turned == 0);
  }
}

// A camera over the player's shoulder is never looked at: not by a look
// away, not by a glance, and not when it cuts to where the eyes rest.
void Camera() {
  GazeSettings settings = Unsettled();
  settings.player_odds = {};
  settings.glance_seconds = 1;
  // 9 degrees to the speaker's right of the player, 60 units further back,
  // with something to look at right beside it and something well clear.
  auto at = [](float degrees, float distance) {
    const float angle = degrees / 57.29578f;
    return GazePoint{
        {distance * std::sin(angle), distance * std::cos(angle), 120},
        7,
        false};
  };
  GazeView view = kView;
  const GazePoint camera = at(9, 180);
  std::copy_n(camera.position, 3, view.camera);
  view.camera_known = true;
  const std::vector<GazePoint> points = {at(10, 240), at(-14, 240)};
  const std::vector<GazeBeat> beats = {
      {0, Only(kGazeNeutral), kGazeStart, 0, 4},
      {4, Only(kGazeAngry), kGazePause, 4.5f, 8}};
  int away = 0, toward = 0;
  for (int seed = 1; seed <= 300; ++seed) {
    for (int known = 0; known < 2; ++known) {
      view.camera_known = known;
      GazeDirector director(seed);
      GazeLine line{1, 0, 9, &beats};
      int resting = 0;  // frames in a row stopped near the camera
      for (int frame = 0; frame < 9 * 60; ++frame) {
        line.time = frame * static_cast<float>(kFrame);
        const GazeAim aim =
            director.Update(frame * kFrame, &line, view, points, settings);
        if (aim.on_player) {
          resting = 0;
          continue;
        }
        const bool close = std::hypot(aim.right - 9, aim.down) < 19.9f;
        // A saccade may cross it; the eyes never stop there.
        if (known) {
          CHECK(aim.right < 1);  // and only ever to the player's other side
          away += !close;
          resting = close ? resting + 1 : 0;
          CHECK(resting < 6);
        } else {
          toward += close;
        }
      }
    }
  }
  CHECK(away > 1000);
  CHECK(toward > 1000);  // without the camera known, that is where they look
  // The camera cuts onto a look that is being held: back to the player.
  view = kView;
  settings.glance_seconds = 0;
  const std::vector<GazePoint> one = {at(8, 240)};
  bool cut = false;
  for (int seed = 1; seed < 200 && !cut; ++seed) {
    GazeDirector director(seed);
    GazeLine line{1, 0, 9, &beats};
    GazeAim aim;
    for (int frame = 0; frame < 30; ++frame)
      aim = director.Update(frame * kFrame, &line, view, one, settings);
    if (std::abs(aim.right - 8) > 0.01f) continue;
    std::copy_n(one[0].position, 3, view.camera);
    view.camera_known = true;
    for (int frame = 30; frame < 45; ++frame)
      aim = director.Update(frame * kFrame, &line, view, one, settings);
    CHECK(aim.on_player && aim.right == 0 && aim.down == 0);
    cut = true;
  }
  CHECK(cut);
  // The camera just beside the player's head: the eyes rest on the head's
  // other side, 8 degrees from the lens; from the head itself, below it.
  GazeDirector director(1);
  view = kView;
  view.camera_known = true;
  const GazePoint beside = at(1.5f, 150);
  std::copy_n(beside.position, 3, view.camera);
  GazeAim aim = director.Update(0, nullptr, view, {}, settings);
  CHECK(aim.on_player && std::abs(aim.right + 6.5f) < 0.05f &&
        std::abs(aim.down) < 0.05f);
  std::copy_n(view.player, 3, view.camera);
  aim = director.Update(1, nullptr, view, {}, settings);
  CHECK(aim.on_player && aim.right == 0 && std::abs(aim.down - 8) < 0.05f);
  // Behind the speaker, or invalid: as if there were none.
  view.camera[1] = -200;
  aim = director.Update(2, nullptr, view, {}, settings);
  CHECK(aim.right == 0 && aim.down == 0);
  view.camera[0] = NAN;
  aim = director.Update(3, nullptr, view, {}, settings);
  CHECK(aim.right == 0 && aim.down == 0);
}

// A ray that stops at the thing's own surface reached it; one that stops at
// a wall in front of it did not.
// A line ready late has its beats only once it is generated, seconds into
// it: the moments already past move no eyes. A beat met within
// half a second still does.
void Late() {
  const std::vector<GazeBeat> beats = {
      {0, Only(kGazeNeutral), kGazeStart, 0, 1},
      {1, Only(kGazeSad), kGazePause, 1.5f, 4.5f}};
  GazeSettings settings = Unsettled();
  settings.player_odds = {};
  settings.glance_seconds = 0;
  const std::vector<GazePoint> points;
  int left[2] = {};
  for (int late = 0; late < 2; ++late)
    for (int seed = 1; seed <= 200; ++seed) {
      GazeDirector director(seed);
      // From 0.8 s without beats; they are there from 2 s (late) or 1.2 s.
      const float found = late ? 2 : 1.2f;
      bool away = false;
      for (int frame = 0; frame < 90; ++frame) {
        GazeLine line{1, 0.8f + frame * static_cast<float>(kFrame), 6, nullptr};
        if (line.time >= found) line.beats = &beats;
        away |= !director.Update(frame * kFrame, &line, kView, points, settings)
                     .on_player;
      }
      left[late] += away;
    }
  CHECK(left[1] == 0);
  CHECK(left[0] > 150);
}

void Sight() {
  CHECK(GazeRayReaches(1, 300, 20));
  CHECK(GazeRayReaches(0.95f, 300, 20));  // 15 short: its near side
  CHECK(!GazeRayReaches(0.9f, 300, 20));  // 30 short: something else
  CHECK(!GazeRayReaches(0.5f, 300, 20));
  CHECK(GazeRayReaches(0.97f, 300, 0));  // small things: a fixed margin
  CHECK(!GazeRayReaches(0.9f, 300, NAN));
  CHECK(!GazeRayReaches(NAN, 300, 20) && !GazeRayReaches(0.5f, NAN, 20) &&
        !GazeRayReaches(-1, 300, 20));
}

void Invalid() {
  GazeSettings settings = Unsettled();
  const std::vector<GazePoint> points = {{{NAN, 0, 0}, 1, false}};
  GazeDirector director(5);
  GazeView view = kView;
  view.player[1] = 0;  // on top of the speaker
  GazeAim aim = director.Update(0, nullptr, view, points, settings);
  CHECK(aim.on_player && aim.right == 0 && aim.down == 0);
  view = kView;
  view.eye[2] = NAN;
  aim = director.Update(1, nullptr, view, points, settings);
  CHECK(aim.on_player && aim.right == 0 && aim.down == 0);
  settings.player_odds = {NAN, NAN, NAN, NAN, NAN, NAN, NAN};
  for (double now = 0; now < 30; now += kFrame) {
    aim = director.Update(now, nullptr, kView, points, settings);
    CHECK(std::isfinite(aim.right) && std::isfinite(aim.down) &&
          aim.head_ref == 0);
  }
}

}  // namespace

int main() {
  CHECK(a2f::GazeAwayFromHold(1) == 1 && a2f::GazeAwayFromHold(2) == 0 &&
        a2f::GazeAwayFromHold(0) == 2 && a2f::GazeAwayFromHold(5) == 0 &&
        a2f::GazeAwayFromHold(-1) == 2);
  Moods();
  Beats();
  Odds();
  Thought();
  Rest();
  Glances();
  Snap();
  Head();
  Camera();
  Late();
  Sight();
  Invalid();
  puts("gaze director ok");
  return 0;
}
