// Better Facial Animations core: decode, the line's emotion, network windows,
// A2F Direct coefficients.
#include "live_core.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libswresample/swresample.h>
}

namespace a2f {

namespace {
// Audio2Face-3D v2.3 Mark: audio gain and the implicit emotion vector (SDK
// model config).
constexpr float kInputStrength = 1.3f;
constexpr float kImplicit[kNumImplicit] = {
    0.031689223f,  -0.023175493f, -0.0046390453f, -0.011679624f,
    0.0044094822f, -0.039738461f, -0.013413748f,  0.013034184f,
    0.0074624573f, -0.026291862f, -0.057227135f,  -0.02242188f,
    0.0015903788f, 0.03309321f,   -0.030960485f,  0.017880375f};
}  // namespace

// ---------------- audio ----------------
namespace {
// In-memory AVIO source.
struct Mem {
  const uint8_t* p;
  size_t n, pos;
};
int MemRead(void* o, uint8_t* buf, int sz) {
  auto* m = static_cast<Mem*>(o);
  size_t k = std::min(static_cast<size_t>(sz), m->n - m->pos);
  if (!k) return AVERROR_EOF;
  memcpy(buf, m->p + m->pos, k);
  m->pos += k;
  return static_cast<int>(k);
}
int64_t MemSeek(void* o, int64_t off, int whence) {
  auto* m = static_cast<Mem*>(o);
  if (whence & AVSEEK_SIZE) return static_cast<int64_t>(m->n);
  whence &= ~AVSEEK_FORCE;
  int64_t np = whence == SEEK_SET   ? off
               : whence == SEEK_CUR ? static_cast<int64_t>(m->pos) + off
                                    : static_cast<int64_t>(m->n) + off;
  if (np < 0 || np > static_cast<int64_t>(m->n)) return -1;
  m->pos = static_cast<size_t>(np);
  return np;
}

// Frees an FFmpeg object as its type asks.
struct AvDelete {
  void operator()(AVIOContext* p) const {
    av_freep(&p->buffer);  // the one in use: avio may have replaced it
    avio_context_free(&p);
  }
  void operator()(AVFormatContext* p) const { avformat_close_input(&p); }
  void operator()(AVCodecContext* p) const { avcodec_free_context(&p); }
  void operator()(SwrContext* p) const { swr_free(&p); }
  void operator()(AVPacket* p) const { av_packet_free(&p); }
  void operator()(AVFrame* p) const { av_frame_free(&p); }
};
template <typename T>
using AvPtr = std::unique_ptr<T, AvDelete>;
}  // namespace

bool HasAudibleAudio(const float* pcm, size_t n) {
  constexpr size_t window = kSampleRate / 50;  // 20 ms
  constexpr double min_power = 1e-7;           // -70 dBFS
  if (!pcm) return false;
  for (size_t start = 0; start < n; start += window) {
    const size_t count = std::min(window, n - start);
    double sum = 0, square = 0;
    for (size_t i = start; i < start + count; ++i) {
      const double v = pcm[i];
      if (!std::isfinite(v)) return false;
      sum += v;
      square += v * v;
    }
    const double mean = sum / count;
    if (square / count - mean * mean > min_power) return true;
  }
  return false;
}

bool DecodeAudio(const uint8_t* data, size_t n, std::vector<float>& pcm,
                 std::string& err) {
  pcm.clear();
  if (n >= 12 && !memcmp(data, "FUZE", 4)) {
    // FUZE: magic, version, lip size, lip data, audio
    uint32_t lip;
    memcpy(&lip, data + 8, 4);
    if (12 + static_cast<size_t>(lip) > n) {
      err = "bad fuz";
      return false;
    }
    data += 12 + lip;
    n -= 12 + lip;
  }
  if (n >= 8 && !memcmp(data, "RIFF", 4)) {
    uint32_t riff_size;
    memcpy(&riff_size, data + 4, 4);
    if (static_cast<uint64_t>(riff_size) + 8 > n) {
      err = "truncated audio container";
      return false;
    }
  }
  Mem mem{data, n, 0};
  const int iosz = 32768;
  uint8_t* iobuf = static_cast<uint8_t*>(av_malloc(iosz));
  // declared before the format context, which reads through it
  const AvPtr<AVIOContext> io(
      avio_alloc_context(iobuf, iosz, 0, &mem, MemRead, nullptr, MemSeek));
  const AvPtr<AVPacket> pkt(av_packet_alloc());
  const AvPtr<AVFrame> fr(av_frame_alloc());
  AvPtr<AVCodecContext> dec;
  AvPtr<SwrContext> swr;
  std::vector<int16_t> s16;
  auto drain = [&](AVFrame* in) -> bool {  // in == nullptr flushes
    int cap = in ? swr_get_out_samples(swr.get(), in->nb_samples)
                 : swr_get_out_samples(swr.get(), 0);
    if (cap <= 0 && !in) return true;
    size_t at = s16.size();
    s16.resize(at + std::max(cap, 0) + 256);
    uint8_t* outp = reinterpret_cast<uint8_t*>(s16.data() + at);
    int got = swr_convert(
        swr.get(), &outp, static_cast<int>(s16.size() - at),
        in ? const_cast<const uint8_t**>(in->extended_data) : nullptr,
        in ? in->nb_samples : 0);
    if (got < 0) {
      err = "resample failed";
      return false;
    }
    s16.resize(at + got);
    return true;
  };
  auto receive = [&]() -> bool {
    while (true) {
      int r = avcodec_receive_frame(dec.get(), fr.get());
      if (r == AVERROR(EAGAIN) || r == AVERROR_EOF) return true;
      if (r < 0) {
        err = "decode failed";
        return false;
      }
      bool k = drain(fr.get());
      av_frame_unref(fr.get());
      if (!k) return false;
    }
  };
  AVFormatContext* opening = avformat_alloc_context();
  opening->pb = io.get();
  opening->flags |= AVFMT_FLAG_CUSTOM_IO;
  // frees the context when it fails
  if (avformat_open_input(&opening, nullptr, nullptr, nullptr) < 0) {
    err = "unknown audio format";
    return false;
  }
  const AvPtr<AVFormatContext> fmt(opening);
  if (avformat_find_stream_info(fmt.get(), nullptr) < 0) {
    err = "no stream info";
    return false;
  }
  const int st =
      av_find_best_stream(fmt.get(), AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);
  if (st < 0) {
    err = "no audio stream";
    return false;
  }
  const AVCodecParameters* par = fmt->streams[st]->codecpar;
  const AVCodec* codec = avcodec_find_decoder(par->codec_id);
  if (!codec) {
    err = "unsupported codec";
    return false;
  }
  dec.reset(avcodec_alloc_context3(codec));
  if (avcodec_parameters_to_context(dec.get(), par) < 0 ||
      avcodec_open2(dec.get(), codec, nullptr) < 0) {
    err = "cannot open decoder";
    return false;
  }
  AVChannelLayout mono = AV_CHANNEL_LAYOUT_MONO;
  SwrContext* resampler = nullptr;
  const int allocated = swr_alloc_set_opts2(
      &resampler, &mono, AV_SAMPLE_FMT_S16, kSampleRate, &dec->ch_layout,
      dec->sample_fmt, dec->sample_rate, 0, nullptr);
  swr.reset(resampler);
  if (allocated < 0 || swr_init(swr.get()) < 0) {
    err = "cannot create resampler";
    return false;
  }
  int read_result = 0;
  while ((read_result = av_read_frame(fmt.get(), pkt.get())) >= 0) {
    bool ok = true;
    if (pkt->stream_index == st) {
      if (avcodec_send_packet(dec.get(), pkt.get()) < 0) {
        err = "audio packet rejected";
        ok = false;
      } else {
        ok = receive();
      }
    }
    av_packet_unref(pkt.get());
    if (!ok) return false;
  }
  if (read_result != AVERROR_EOF) {
    err = "audio read failed";
    return false;
  }
  if (avcodec_send_packet(dec.get(), nullptr) < 0) {
    err = "audio decoder flush failed";
    return false;
  }
  if (!receive() || !drain(nullptr)) return false;
  if (s16.empty()) {
    err = "no audio decoded";
    return false;
  }
  pcm.resize(s16.size());
  for (size_t i = 0; i < s16.size(); ++i) pcm[i] = s16[i] * (1.0f / 32768.0f);
  return true;
}

namespace {
// emotion_map.MAP in network order (amazement anger cheekiness disgust fear
// grief joy outofbreath pain sadness) per Skyrim emotion type.
constexpr float kMap[8][kNumMod] = {{},
                                    {0, 1},
                                    {0, 0, 0, 1},
                                    {0, 0, 0, 0, 1},
                                    {0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
                                    {0, 0, 0, 0, 0, 0, 1},
                                    {1},
                                    {0.3f, 0, 0.3f}};
// numpy default (linear).
double Percentile(std::vector<double> v, double q) {
  std::sort(v.begin(), v.end());
  double pos = q / 100.0 * (v.size() - 1);
  size_t i = static_cast<size_t>(pos);
  double f = pos - i;
  return i + 1 < v.size() ? v[i] + (v[i + 1] - v[i]) * f : v[i];
}

// Forward-backward one-pole, `passes` times.
void ZeroPhase(std::vector<double>& e, double a, int passes) {
  size_t n = e.size();
  for (int pass = 0; pass < 2 * passes; ++pass) {
    for (size_t i = 1; i < n; ++i) e[i] = a * e[i - 1] + (1 - a) * e[i];
    std::reverse(e.begin(), e.end());
  }
}

// The voice's level per frame, 0..1: RMS in dB, normalised between the
// line's 20th and 97th percentile, slowly smoothed.
std::vector<double> Loudness(const float* pcm, size_t n, int frames) {
  std::vector<double> db(frames);
  double hop = static_cast<double>(kSampleRate) / kFps;
  for (int i = 0; i < frames; ++i) {
    size_t a = static_cast<size_t>(i * hop),
           b = std::min(static_cast<size_t>((i + 1) * hop), n);
    double e = 0;
    for (size_t k = a; k < b; ++k) e += static_cast<double>(pcm[k]) * pcm[k];
    db[i] = 20 * std::log10(std::sqrt((b > a ? e / (b - a) : 0) + 1e-12));
  }
  double lo = Percentile(db, 20), hi = Percentile(db, 97);
  for (double& v : db)
    v = std::clamp((v - lo) / std::max(hi - lo, 1e-3), 0.0, 1.0);
  ZeroPhase(db, 0.82, 2);
  return db;
}

}  // namespace

void EmotionTrack(const float* pcm, size_t n, int frames,
                  const Emotion& emotion, std::vector<float>& track) {
  double game[kNumMod] = {};
  if (emotion.type >= 1 && emotion.type <= 7)
    for (int c = 0; c < kNumMod; ++c)
      game[c] =
          kMap[emotion.type][c] * std::clamp(emotion.value / 100.0, 0.0, 1.0);
  std::vector<double> loud;
  if (emotion.swell > 0 && emotion.gain > 0) loud = Loudness(pcm, n, frames);
  track.resize(static_cast<size_t>(frames) * kNumMod);
  for (int t = 0; t < frames; ++t) {
    const double g =
        emotion.gain * (loud.empty() ? 1 : 1 + emotion.swell * loud[t]);
    for (int c = 0; c < kNumMod; ++c)
      track[static_cast<size_t>(t) * kNumMod + c] =
          static_cast<float>(std::clamp(g * game[c], 0.0, kMaxEmotion));
  }
}

// ---------------- the whole line ----------------
bool Generate(Net& net, const float* pcm, size_t n, const Emotion& emotion,
              const float* proj, const float* rest, uint32_t k, Line& out,
              std::string& err, const std::function<bool()>& allowed) {
  out = Line();  // a second run (after a lost session) starts clean
  if (!n) {
    err = "empty audio";
    return false;
  }
  const int num_frames = static_cast<int>(
      (static_cast<int64_t>(n) * kFps + kSampleRate - 1) / kSampleRate);
  std::vector<float> track;
  EmotionTrack(pcm, n, num_frames, emotion, track);
  for (int t = 0; t < num_frames; ++t)
    for (int c = 0; c < kNumMod; ++c) {
      float v = track[static_cast<size_t>(t) * kNumMod + c];
      out.emotion_peak[c] = std::max(out.emotion_peak[c], v);
      out.emotion_mean[c] += v / num_frames;
    }

  // --- network: audio windows + [implicit, explicit] emotion per frame; the
  // outputs straight to the A2F Direct coefficients ---
  int batch = std::max(1, net.Batch());
  std::vector<float> x(static_cast<size_t>(batch) * kWindow),
      em(static_cast<size_t>(batch) * kNumEmo),
      y(static_cast<size_t>(batch) * kNumOut);
  out.frames = num_frames;
  const uint32_t dim = k + kNumTongue + kNumJaw;
  out.direct_dim = dim;
  out.direct.assign(static_cast<size_t>(num_frames) * dim, 0.f);
  for (int f0 = 0; f0 < num_frames; f0 += batch) {
    int b = std::min(batch, num_frames - f0);
    std::fill(x.begin(), x.end(), 0.f);
    for (int i = 0; i < batch; ++i) {  // the padding: the last frame's
      float* ei = &em[static_cast<size_t>(i) * kNumEmo];
      memcpy(ei, kImplicit, sizeof kImplicit);
      memcpy(ei + kNumImplicit,
             &track[static_cast<size_t>(std::min(f0 + i, num_frames - 1)) *
                    kNumMod],
             kNumMod * sizeof(float));
    }
    for (int i = 0; i < b; ++i) {
      int64_t start =
          static_cast<int64_t>(f0 + i) * kSampleRate / kFps - kWindowOffset;
      float* xi = &x[static_cast<size_t>(i) * kWindow];
      for (int j = 0; j < kWindow; ++j) {
        int64_t p = start + j;
        xi[j] = (p >= 0 && p < static_cast<int64_t>(n))
                    ? pcm[p] * kInputStrength
                    : 0.f;
      }
    }
    if (allowed && !allowed()) {
      err = "generation cancelled";
      return false;
    }
    if (!net.Run(x.data(), em.data(), batch, y.data(), err)) return false;
    for (int i = 0; i < b; ++i) {
      const float* yt = &y[static_cast<size_t>(i) * kNumOut];
      float* o = &out.direct[static_cast<size_t>(f0 + i) * dim];
      for (uint32_t c = 0; c < k; ++c) {
        const float* p = proj + static_cast<size_t>(c) * kNumPca;
        for (int j = 0; j < kNumPca; ++j) o[c] += (yt[j] - rest[j]) * p[j];
      }
      for (int j = kNumPca; j < kNumPca + kNumTongue + kNumJaw; ++j)
        o[k + j - kNumPca] = yt[j] - rest[j];
    }
  }
  return true;
}

bool Generate3(Net& net, const float* pcm, size_t n, const Emotion& emotion,
               const Face3& face, Line& out, std::string& err,
               const std::function<bool()>& allowed) {
  out = Line();  // a second run (after a lost session) starts clean
  if (!n) {
    err = "empty audio";
    return false;
  }
  if (face.identity < 0 || face.identity >= kNumIdentity3) {
    err = "no such identity";
    return false;
  }
  // the emotion at kFps, as v2.3 takes it; each 60 fps frame takes its own
  const int emo_frames = static_cast<int>(
      (static_cast<int64_t>(n) * kFps + kSampleRate - 1) / kSampleRate);
  std::vector<float> track;
  EmotionTrack(pcm, n, emo_frames, emotion, track);
  for (int t = 0; t < emo_frames; ++t)
    for (int c = 0; c < kNumMod; ++c) {
      float v = track[static_cast<size_t>(t) * kNumMod + c];
      out.emotion_peak[c] = std::max(out.emotion_peak[c], v);
      out.emotion_mean[c] += v / emo_frames;
    }

  // frames at 60 fps covering the clip; window i holds the audio
  // [kStride3 i - kWindow3, kStride3 i) and gives frames 30 i - 45 .. + 29
  const int num_frames = static_cast<int>(
      (static_cast<int64_t>(n) * kFps3 + kSampleRate - 1) / kSampleRate);
  const int windows = (kSkip3 - 1 + num_frames) / kFrames3 + 1;
  float ident[kNumIdentity3] = {};
  ident[face.identity] = 1;
  std::vector<float> x(kWindow3), em(kFrames3 * kNumMod),
      skin(kFrames3 * kLatent3), tongue(kFrames3 * kLatent3),
      jaw(kFrames3 * kJawEyes3), state(kState3), next(kState3);
  out.frames = num_frames;
  out.fps = kFps3;
  const uint32_t k = face.k, dim = k + kNumTongue + kNumJaw;
  out.direct_dim = dim;
  out.direct.assign(static_cast<size_t>(num_frames) * dim, 0.f);
  const float *rest_tongue = face.rest + kLatent3,
              *rest_jaw = face.rest + 2 * kLatent3;
  for (int i = 0; i < windows; ++i) {
    const int64_t a = static_cast<int64_t>(kStride3) * i - kWindow3;
    for (int j = 0; j < kWindow3; ++j)
      x[j] = a + j >= 0 && a + j < static_cast<int64_t>(n) ? pcm[a + j] : 0.f;
    const int f0 = kFrames3 * i - kSkip3;
    for (int j = 0; j < kFrames3; ++j) {
      const int f = std::clamp(f0 + j, 0, num_frames - 1);
      memcpy(&em[static_cast<size_t>(j) * kNumMod],
             &track[static_cast<size_t>(
                        std::min(f * kFps / kFps3, emo_frames - 1)) *
                    kNumMod],
             kNumMod * sizeof(float));
    }
    if (allowed && !allowed()) {
      err = "generation cancelled";
      return false;
    }
    if (!net.Run3(x.data(), ident, em.data(), state.data(), skin.data(),
                  tongue.data(), jaw.data(), next.data(), err))
      return false;
    state.swap(next);
    for (int j = 0; j < kFrames3; ++j) {
      const int f = f0 + j;
      if (f < 0 || f >= num_frames) continue;
      float* o = &out.direct[static_cast<size_t>(f) * dim];
      const float* s = &skin[static_cast<size_t>(j) * kLatent3];
      const float* t = &tongue[static_cast<size_t>(j) * kLatent3];
      const float* g = &jaw[static_cast<size_t>(j) * kJawEyes3];
      for (uint32_t c = 0; c < k; ++c) {
        const float* p = face.proj + static_cast<size_t>(c) * kLatent3;
        for (int l = 0; l < kLatent3; ++l) o[c] += (s[l] - face.rest[l]) * p[l];
      }
      for (int c = 0; c < kNumTongue; ++c) {
        const float* p = face.tongue_proj + static_cast<size_t>(c) * kLatent3;
        for (int l = 0; l < kLatent3; ++l)
          o[k + c] += (t[l] - rest_tongue[l]) * p[l];
      }
      for (int c = 0; c < kNumJaw; ++c)
        o[k + kNumTongue + c] = g[c] - rest_jaw[c];
    }
  }
  return true;
}

}  // namespace a2f
