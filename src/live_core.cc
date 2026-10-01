// Better Facial Animations core: decode, network windows, A2F Direct
// coefficients, expression curve.
#include "live_core.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
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
// Expression morph: level = floor + (1 - floor) * value / 100, times gain
// (max 1); eased in / out over attack / release from onset; emphasis = how
// much it follows loudness.
constexpr double kExprFloor = 0.4, kExprGain = 1.5, kOnset = 0.2,
                 kAttack = 0.35, kRelease = 0.4, kEmphasis = 0.35;
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
}  // namespace

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
  Mem mem{data, n, 0};
  const int iosz = 32768;
  uint8_t* iobuf = static_cast<uint8_t*>(av_malloc(iosz));
  AVIOContext* io =
      avio_alloc_context(iobuf, iosz, 0, &mem, MemRead, nullptr, MemSeek);
  AVFormatContext* fmt = avformat_alloc_context();
  AVCodecContext* dec = nullptr;
  SwrContext* swr = nullptr;
  AVPacket* pkt = av_packet_alloc();
  AVFrame* fr = av_frame_alloc();
  std::vector<int16_t> s16;
  bool ok = false;
  int st = -1;
  fmt->pb = io;
  fmt->flags |= AVFMT_FLAG_CUSTOM_IO;
  auto drain = [&](AVFrame* in) -> bool {  // in == nullptr flushes
    int cap = in ? swr_get_out_samples(swr, in->nb_samples)
                 : swr_get_out_samples(swr, 0);
    if (cap <= 0 && !in) return true;
    size_t at = s16.size();
    s16.resize(at + std::max(cap, 0) + 256);
    uint8_t* outp = reinterpret_cast<uint8_t*>(s16.data() + at);
    int got = swr_convert(
        swr, &outp, static_cast<int>(s16.size() - at),
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
      int r = avcodec_receive_frame(dec, fr);
      if (r == AVERROR(EAGAIN) || r == AVERROR_EOF) return true;
      if (r < 0) {
        err = "decode failed";
        return false;
      }
      bool k = drain(fr);
      av_frame_unref(fr);
      if (!k) return false;
    }
  };
  do {
    if (avformat_open_input(&fmt, nullptr, nullptr, nullptr) < 0) {
      fmt = nullptr;
      err = "unknown audio format";
      break;
    }
    if (avformat_find_stream_info(fmt, nullptr) < 0) {
      err = "no stream info";
      break;
    }
    st = av_find_best_stream(fmt, AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);
    if (st < 0) {
      err = "no audio stream";
      break;
    }
    const AVCodecParameters* par = fmt->streams[st]->codecpar;
    const AVCodec* codec = avcodec_find_decoder(par->codec_id);
    if (!codec) {
      err = "unsupported codec";
      break;
    }
    dec = avcodec_alloc_context3(codec);
    if (avcodec_parameters_to_context(dec, par) < 0 ||
        avcodec_open2(dec, codec, nullptr) < 0) {
      err = "cannot open decoder";
      break;
    }
    AVChannelLayout mono = AV_CHANNEL_LAYOUT_MONO;
    if (swr_alloc_set_opts2(&swr, &mono, AV_SAMPLE_FMT_S16, kSampleRate,
                            &dec->ch_layout, dec->sample_fmt, dec->sample_rate,
                            0, nullptr) < 0 ||
        swr_init(swr) < 0) {
      err = "cannot create resampler";
      break;
    }
    bool bad = false;
    while (!bad && av_read_frame(fmt, pkt) >= 0) {
      if (pkt->stream_index == st && avcodec_send_packet(dec, pkt) >= 0)
        bad = !receive();
      av_packet_unref(pkt);
    }
    if (bad) break;
    avcodec_send_packet(dec, nullptr);
    if (!receive() || !drain(nullptr)) break;
    ok = !s16.empty();
    if (!ok) err = "no audio decoded";
  } while (false);
  av_frame_free(&fr);
  av_packet_free(&pkt);
  swr_free(&swr);
  avcodec_free_context(&dec);
  if (fmt) avformat_close_input(&fmt);
  avio_context_free(&io);  // the buffer may have been reallocated by avio
  if (ok) {
    pcm.resize(s16.size());
    for (size_t i = 0; i < s16.size(); ++i) pcm[i] = s16[i] * (1.0f / 32768.0f);
  }
  return ok;
}

namespace {
double Ease(double x) {
  double c = std::clamp(x, 0.0, 1.0);
  return c * c * (3 - 2 * c);
}
double Envelope(double s, double dur) {
  double e = Ease(s / kAttack) * Ease((dur - s) / kRelease);
  return kOnset + (1 - kOnset) * e;
}

// emotion_map.MAP in network order (amazement anger cheekiness disgust fear
// grief joy outofbreath pain sadness), at value / 100.
void ExplicitEmotion(int type, int value, float out[kNumMod]) {
  static const float kMap[8][kNumMod] = {{},
                                         {0, 1},
                                         {0, 0, 0, 1},
                                         {0, 0, 0, 0, 1},
                                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
                                         {0, 0, 0, 0, 0, 0, 1},
                                         {1},
                                         {0.3f, 0, 0.3f}};
  for (int i = 0; i < kNumMod; ++i) out[i] = 0;
  if (type < 1 || type > 7) return;
  double k = std::clamp(value / 100.0, 0.0, 1.0);
  for (int i = 0; i < kNumMod; ++i)
    out[i] = static_cast<float>(std::nearbyint(kMap[type][i] * k * 1e3) / 1e3);
}

// numpy default (linear).
double Percentile(std::vector<double> v, double q) {
  std::sort(v.begin(), v.end());
  double pos = q / 100.0 * (v.size() - 1);
  size_t i = static_cast<size_t>(pos);
  double f = pos - i;
  return i + 1 < v.size() ? v[i] + (v[i + 1] - v[i]) * f : v[i];
}

// expression.loudness smoothing: forward-backward one-pole, twice.
void ZeroPhase(std::vector<double>& e, double a) {
  size_t n = e.size();
  for (int pass = 0; pass < 2; ++pass) {
    for (size_t i = 1; i < n; ++i) e[i] = a * e[i - 1] + (1 - a) * e[i];
    std::reverse(e.begin(), e.end());
  }
}

// Engine expression index (channels.EXPRESSIONS) for Skyrim emotion types
// 1..7.
const uint32_t kEngineExpr[8] = {255, 0, 6, 1, 3, 2, 4, 5};
}  // namespace

// ---------------- the whole line ----------------
bool Generate(Net& net, const float* pcm, size_t n, int emotion, int value,
              const DirectProj& direct, Line& out, std::string& err,
              Timing* timing) {
  using Clock = std::chrono::steady_clock;
  auto t0 = Clock::now();
  if (!n) {
    err = "empty audio";
    return false;
  }
  int num_frames = 0;
  while (static_cast<int64_t>(num_frames) * kSampleRate / kFps <
         static_cast<int64_t>(n))
    ++num_frames;

  // --- network: audio windows + [implicit, explicit] emotion per frame,
  // the emotion constant over the line ---
  float emo10[kNumMod];
  ExplicitEmotion(emotion, value, emo10);
  int batch = std::max(1, net.Batch());
  std::vector<float> x(static_cast<size_t>(batch) * kWindow),
      em(static_cast<size_t>(batch) * kNumEmo),
      y(static_cast<size_t>(batch) * kNumOut),
      raw(static_cast<size_t>(num_frames) * kNumRaw);
  for (int i = 0; i < batch; ++i) {
    float* ei = &em[static_cast<size_t>(i) * kNumEmo];
    memcpy(ei, kImplicit, sizeof kImplicit);
    memcpy(ei + kNumImplicit, emo10, sizeof emo10);
  }
  for (int f0 = 0; f0 < num_frames; f0 += batch) {
    int b = std::min(batch, num_frames - f0);
    std::fill(x.begin(), x.end(), 0.f);
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
    if (!net.Run(x.data(), em.data(), batch, y.data(), err)) return false;
    for (int i = 0; i < b; ++i)
      memcpy(&raw[static_cast<size_t>(f0 + i) * kNumRaw],
             &y[static_cast<size_t>(i) * kNumOut], kNumRaw * sizeof(float));
  }
  auto t1 = Clock::now();

  // --- A2F Direct coefficients ---
  out.frames = num_frames;
  out.direct_dim = direct.k + kNumTongue;
  out.direct.assign(static_cast<size_t>(num_frames) * out.direct_dim, 0.f);
  for (int t = 0; t < num_frames; ++t) {
    const float* yt = &raw[static_cast<size_t>(t) * kNumRaw];
    float* o = &out.direct[static_cast<size_t>(t) * out.direct_dim];
    for (uint32_t k = 0; k < direct.k; ++k) {
      const float* p = direct.proj + static_cast<size_t>(k) * kNumPca;
      for (int j = 0; j < kNumPca; ++j) o[k] += (yt[j] - direct.rest[j]) * p[j];
    }
    for (int j = 0; j < kNumTongue; ++j)
      o[direct.k + j] = yt[kNumPca + j] - direct.rest[kNumPca + j];
  }

  // --- expression curve: level x envelope x loudness (expression.loudness:
  // per-frame RMS in dB, normalised between the 20th and 97th percentile,
  // slow smoothing) ---
  out.expr = emotion >= 1 && emotion <= 7 ? kEngineExpr[emotion] : 255;
  out.expression.clear();
  if (out.expr != 255) {
    std::vector<double> db(num_frames);
    double hop = static_cast<double>(kSampleRate) / kFps;
    for (int i = 0; i < num_frames; ++i) {
      size_t a = static_cast<size_t>(i * hop),
             b = std::min(static_cast<size_t>((i + 1) * hop), n);
      double e = 0;
      for (size_t k = a; k < b; ++k) e += static_cast<double>(pcm[k]) * pcm[k];
      db[i] = 20 * std::log10(std::sqrt((b > a ? e / (b - a) : 0) + 1e-12));
    }
    double lo = Percentile(db, 20), hi = Percentile(db, 97);
    for (double& v : db)
      v = std::clamp((v - lo) / std::max(hi - lo, 1e-3), 0.0, 1.0);
    ZeroPhase(db, 0.82);
    double level = std::min(
               1.0,
               (kExprFloor + (1 - kExprFloor) * std::min(1.0, value / 100.0)) *
                   kExprGain),
           dur = static_cast<double>(num_frames) / kFps;
    out.expression.resize(num_frames);
    for (int t = 0; t < num_frames; ++t)
      out.expression[t] = static_cast<float>(
          std::clamp(level * Envelope(static_cast<double>(t) / kFps, dur) *
                         (1 - kEmphasis + kEmphasis * db[t]),
                     0.0, 1.0));
  }
  if (timing) {
    timing->net = std::chrono::duration<double>(t1 - t0).count();
    timing->post = std::chrono::duration<double>(Clock::now() - t1).count();
  }
  return true;
}

}  // namespace a2f
