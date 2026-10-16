#include <cstdlib>
#include "a2f/inference/ort_net.h"

namespace a2f {
int creates = 0;
OrtOptions seen;
bool gpu_shaders_work = false, cpu_works = false;
class ReadyEmotion : public EmotionNet {
 public:
  bool Run(const float*, float* logits, std::string&) override {
    std::fill_n(logits, kNumVoiceEmo, 0.f);
    return true;
  }
};
std::unique_ptr<EmotionNet> CaptureEmotion(const std::wstring&,
                                         const std::wstring&,
                                         const OrtOptions& options,
                                         std::string& err) {
  ++creates;
  seen = options;
  if ((options.gpu && !options.metacommands && gpu_shaders_work) ||
      (!options.gpu && cpu_works)) return std::make_unique<ReadyEmotion>();
  err = "injected creation failure";
  return nullptr;
}
}
#define CreateOrtEmotionNet CaptureEmotion
#include "../src/plugin/better_facial_animations.cc"
#undef CreateOrtEmotionNet
#define CHECK(x)                                      \
  do {                                                \
    if (!(x)) {                                       \
      fprintf(stderr, "line %d: %s\n", __LINE__, #x); \
      std::exit(1);                                   \
    }                                                 \
  } while (0)
namespace {
int destroyed = 0, calls = 0;
bool fail = false, disable = false;
const bfa::Job* cancel = nullptr;  // its line is cancelled during the run
void Set(const char* key, bool value) {
  for (const auto& p : bfa::kParams)
    if (!strcmp(p.key, key)) *p.flag = value;
}
void Enable(bool enabled) { Set("SentimentAnalysis", enabled); }
struct Voice : a2f::EmotionNet {
  ~Voice() override { ++destroyed; }
  bool Run(const float*, float* logits, std::string& err) override {
    ++calls;
    if (disable) Enable(false);
    if (cancel) cancel->entry->cancelled = true;
    if (fail) {
      err = "test model failure";
      return false;
    }
    std::fill_n(logits, a2f::kNumVoiceEmo, 0.f);
    return true;
  }
};
}  // namespace
int main() {
  using namespace bfa;
  CHECK(!SnapshotTune().sentiment_analysis);
  Tune t = SnapshotTune();
  auto entry =
      Request("sentiment_lifecycle", 0, 0, false, false, t, 0, world_epoch);
  CHECK(entry && queue.back().emotion.sentiment_multiplier == 0);
  Job job = queue.back();
  queue.clear();
  std::vector<float> pcm(16000, .1f);
  auto emotion = job.emotion;
  CHECK(PrepareSentiment(job, pcm, emotion) == Sentiment::kReady &&
        !sentiment_net && calls == 0);
  Enable(true);
  auto missing = Request("missing_model", 0, 0, false, false, SnapshotTune(), 0, world_epoch);
  CHECK(missing && queue.back().emotion.sentiment_multiplier == 0);
  queue.clear();
  Enable(false);
  config.sentiment_available = true;
  emotion.sentiment_multiplier = 1;
  CHECK(PrepareSentiment(job, pcm, emotion) == Sentiment::kCancelled &&
        !sentiment_net && calls == 0);
  Enable(true);
  sentiment_net = std::make_unique<Voice>();
  CHECK(PrepareSentiment(job, pcm, emotion) == Sentiment::kReady &&
        !emotion.sentiment.empty() && calls == 1);
  emotion.sentiment.clear();
  fail = true;
  CHECK(PrepareSentiment(job, pcm, emotion) == Sentiment::kUnavailable &&
        emotion.sentiment.empty());
  CHECK(!sentiment_net && destroyed == 1 && sentiment_retry_at > Now());
  CHECK(PrepareSentiment(job, pcm, emotion) == Sentiment::kUnavailable &&
        !sentiment_net && calls == 2);
  fail = false;
  sentiment_retry_at = 0;
  // Sentiment Analysis turned off during a run: the line goes without sentiment;
  // the idle worker unloads the session (below).
  disable = true;
  sentiment_net = std::make_unique<Voice>();
  CHECK(PrepareSentiment(job, pcm, emotion) == Sentiment::kCancelled &&
        emotion.sentiment.empty());
  CHECK(sentiment_net && destroyed == 1 && sentiment_retry_at == 0);
  // A cancelled line keeps the session for the next one.
  disable = false;
  Enable(true);
  cancel = &job;
  CHECK(PrepareSentiment(job, pcm, emotion) == Sentiment::kCancelled &&
        emotion.sentiment.empty());
  CHECK(sentiment_net && destroyed == 1 && sentiment_retry_at == 0);
  cancel = nullptr;
  job.entry->cancelled = false;
  CHECK(PrepareSentiment(job, pcm, emotion) == Sentiment::kReady &&
        !emotion.sentiment.empty());
  sentiment_net.reset();
  CHECK(destroyed == 2);
  // Shared GPU variants remain GPU-only. Failure leaves authored emotion
  // usable; CPU is created only after an explicit settings change.
  config.independent_device = false;
  Set("ForceCPU", false);
  a2f::gpu_shaders_work = true;
  CHECK(PrepareSentiment(job, pcm, emotion) == Sentiment::kReady);
  CHECK(sentiment_gpu && a2f::seen.gpu && !a2f::seen.metacommands);
  sentiment_net.reset();
  shared_gpu_settings[0] = {};
  a2f::gpu_shaders_work = false;
  a2f::cpu_works = true;
  CHECK(PrepareSentiment(job, pcm, emotion) == Sentiment::kUnavailable);
  CHECK(a2f::seen.gpu && sentiment_retry_at > Now());
  const int before_backoff = a2f::creates;
  CHECK(PrepareSentiment(job, pcm, emotion) == Sentiment::kUnavailable);
  CHECK(a2f::creates == before_backoff);
  Set("ForceCPU", true);
  UnloadSentiment();
  CHECK(PrepareSentiment(job, pcm, emotion) == Sentiment::kReady && !sentiment_gpu);
  Set("ForceCPU", false);
  UnloadSentiment();
  a2f::gpu_shaders_work = true;
  a2f::RetryGpu();
  CHECK(PrepareSentiment(job, pcm, emotion) == Sentiment::kReady && sentiment_gpu);
  sentiment_net.reset();
  shared_gpu_settings[0] = {};
  a2f::gpu_shaders_work = a2f::cpu_works = false;
  for (bool independent : {false, true}) {
    config.independent_device = independent;
    for (bool cpu : {false, true}) {
      Set("ForceCPU", cpu);
      UnloadSentiment();
      a2f::RetryGpu();
      const int before = a2f::creates;
      const int expected = cpu || independent ? 1 : 2;
      CHECK(PrepareSentiment(job, pcm, emotion) == Sentiment::kUnavailable);
      CHECK(a2f::creates == before + expected && !sentiment_net);
      CHECK(a2f::seen.gpu == !cpu && a2f::seen.independent_device == independent);
      CHECK(sentiment_retry_at > Now());
      CHECK(PrepareSentiment(job, pcm, emotion) == Sentiment::kUnavailable);
      CHECK(a2f::creates == before + expected);
    }
  }
  // Idle worker must release weights too, without needing another spoken line.
  sentiment_net = std::make_unique<Voice>();
  Set("Enabled", false);
  int face_loads = 0;
  a2f::NetKeeper keeper(
      false,
      [&](bool, std::string&, std::string&) {
        ++face_loads;
        return std::unique_ptr<a2f::Net>{};
      },
      Now);
  WorkerStep(keeper, false);
  CHECK(!sentiment_net && destroyed == 3 && face_loads == 0);
  // A line generated while sentiment was unavailable stays cached (prefetch
  // keeps working) until the worker would try sentiment again.
  Set("Enabled", true);
  Enable(true);
  sentiment_retry_at = Now() + 60;
  auto degraded = Request("degraded", 0, 0, false, false, SnapshotTune(), 0,
                          world_epoch);
  CHECK(degraded && queue.back().emotion.sentiment_multiplier > 0);
  queue.clear();
  degraded->state = State::kReady;
  degraded->degraded = true;
  CHECK(Request("degraded", 0, 0, false, false, SnapshotTune(), 0,
                world_epoch) == degraded && queue.empty());
  sentiment_retry_at = 0;
  CHECK(Request("degraded", 0, 0, false, false, SnapshotTune(), 0,
                world_epoch) != degraded && queue.size() == 1);
  puts(
      "sentiment loading gate, failure backoff, cancellation and idle unload: "
      "pass");
}
