// A2FLive core: decode, network windows, SDK skin animator, retarget solve,
// expression curve.
#include "live_core.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libswresample/swresample.h>
}

namespace a2f {

// clang-format off
const char* const kChannelNames[kNumSolve] = {
    "Aah", "BigAah", "BMP", "ChJSh", "DST", "Eee", "Eh", "FV",
    "I", "K", "N", "Oh", "OohQ", "R", "Th", "W",
    "BlinkLeft", "BlinkRight", "BrowDownLeft", "BrowDownRight",
    "BrowInLeft", "BrowInRight", "BrowUpLeft", "BrowUpRight",
    "SquintLeft", "SquintRight"};
// clang-format on

namespace {
// Channel indices used by the tongue layer.
constexpr int kAah = 0, kBigAah = 1, kDst = 4, kTh = 14;
}  // namespace

// ---------------- data file ----------------
bool Data::Load(const std::wstring& path, std::string& err) {
  FILE* f = _wfopen(path.c_str(), L"rb");
  if (!f) {
    err = "cannot open data file";
    return false;
  }
  char magic[4];
  uint32_t ver = 0, count = 0;
  bool ok = fread(magic, 1, 4, f) == 4 && !memcmp(magic, "A2FD", 4) &&
            fread(&ver, 4, 1, f) == 1 && ver == 1 &&
            fread(&count, 4, 1, f) == 1;
  for (uint32_t i = 0; ok && i < count; ++i) {
    char name[33] = {};
    Array x;
    ok = fread(name, 1, 32, f) == 32 && fread(&x.rows, 4, 1, f) == 1 &&
         fread(&x.cols, 4, 1, f) == 1 &&
         static_cast<size_t>(x.rows) * x.cols < (1u << 24);
    if (!ok) break;
    x.values.resize(static_cast<size_t>(x.rows) * x.cols);
    ok = fread(x.values.data(), 4, x.values.size(), f) == x.values.size();
    arrays[name] = std::move(x);
  }
  fclose(f);
  if (!ok) {
    err = "bad data file";
    return false;
  }
  const Array *cfg, *tun, *emo, *ch;
  if (!(cfg = Get("cfg", 1, 7, err)) || !(tun = Get("tun", 1, 9, err)) ||
      !(emo = Get("emo", 1, 9, err)) || !(ch = Get("ch", kNumSolve, 6, err)))
    return false;
  Settings& s = defaults;
  const float* c = cfg->values.data();
  s.input_strength = c[0];
  s.upper_smoothing = c[1];
  s.lower_smoothing = c[2];
  s.upper_face_strength = c[3];
  s.lower_face_strength = c[4];
  s.skin_strength = c[5];
  s.tongue_strength = c[6];
  const float* t = tun->values.data();
  s.strength = t[0];
  s.ridge = t[1];
  s.smooth = t[2];
  s.feature_weight = t[3];
  s.tongue_on = t[4] != 0;
  s.tip_up_full = t[5];
  s.tip_up_min = t[6];
  s.th_fwd = t[7];
  s.tongue_share = t[8];
  const float* e = emo->values.data();
  s.a2f_floor = e[0];
  s.a2f_gain = e[1];
  s.a2f_cap = e[2];
  s.expr_floor = e[3];
  s.expr_gain = e[4];
  s.onset = e[5];
  s.attack = e[6];
  s.release = e[7];
  s.emphasis = e[8];
  for (int j = 0; j < kNumSolve; ++j) {
    const float* r = (*ch)[j];
    s.gain[j] = r[0];
    s.gamma[j] = r[1];
    s.offset[j] = r[2];
    s.max[j] = r[3];
    s.cost[j] = r[4];
    s.solve[j] = r[5] != 0;
  }
  for (const char* sx : {"m_", "f_"}) {
    std::string p = sx;
    for (const char* side : {"U_", "L_"})
      if (!Get(p + side + "b1", kNumPca, kNumMouth, err) ||
          !Get(p + side + "b2", kNumPca, kNumSolve, err) ||
          !Get(p + side + "f", kNumPca, kNumFeat, err))
        return false;
    if (!Get(p + "G1", kNumMouth, kNumMouth, err) ||
        !Get(p + "PF", kNumFeat, kNumMouth, err) ||
        !Get(p + "G2", kNumSolve, kNumSolve, err) ||
        !Get(p + "scal", 1, 7, err))
      return false;
  }
  return Get("implicit", 1, kNumImplicit, err) &&
         Get("tip", kNumTongue, 3, err) && arrays.count("sil_raw") &&
         arrays["sil_raw"].cols == kNumRaw && arrays["sil_raw"].rows >= 45;
}

const Array* Data::Get(const std::string& name, uint32_t rows, uint32_t cols,
                       std::string& err) const {
  auto it = arrays.find(name);
  if (it == arrays.end() || it->second.rows != rows ||
      it->second.cols != cols) {
    err = "data file: missing or bad array " + name;
    return nullptr;
  }
  return &it->second;
}

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

// ---------------- small linear algebra ----------------
namespace {
// Largest eigenvalue of a symmetric k x k matrix (cyclic Jacobi).
double MaxEig(std::vector<double> a, int k) {
  for (int sweep = 0; sweep < 64; ++sweep) {
    double off = 0;
    for (int p = 0; p < k; ++p)
      for (int q = p + 1; q < k; ++q) off += a[p * k + q] * a[p * k + q];
    if (off < 1e-22) break;
    for (int p = 0; p < k; ++p)
      for (int q = p + 1; q < k; ++q) {
        double apq = a[p * k + q];
        if (std::fabs(apq) < 1e-300) continue;
        double th = (a[q * k + q] - a[p * k + p]) / (2 * apq);
        double t = (th >= 0 ? 1 : -1) /
                   (std::fabs(th) + std::sqrt(th * th + 1)),
               c = 1 / std::sqrt(t * t + 1), s = t * c;
        for (int r = 0; r < k; ++r) {  // a = J^T a J
          double arp = a[r * k + p], arq = a[r * k + q];
          a[r * k + p] = c * arp - s * arq;
          a[r * k + q] = s * arp + c * arq;
        }
        for (int r = 0; r < k; ++r) {
          double apr = a[p * k + r], aqr = a[q * k + r];
          a[p * k + r] = c * apr - s * aqr;
          a[q * k + r] = s * apr + c * aqr;
        }
      }
  }
  double m = -1e300;
  for (int i = 0; i < k; ++i) m = std::max(m, a[i * k + i]);
  return m;
}

// In place, lower triangle.
bool Cholesky(std::vector<double>& l, int k) {
  for (int j = 0; j < k; ++j) {
    double d = l[j * k + j];
    for (int i = 0; i < j; ++i) d -= l[j * k + i] * l[j * k + i];
    if (d <= 0) return false;
    d = std::sqrt(d);
    l[j * k + j] = d;
    for (int r = j + 1; r < k; ++r) {
      double v = l[r * k + j];
      for (int i = 0; i < j; ++i) v -= l[r * k + i] * l[j * k + i];
      l[r * k + j] = v / d;
    }
  }
  return true;
}
void CholSolve(const std::vector<double>& l, int k, double* x) {
  for (int i = 0; i < k; ++i) {
    double v = x[i];
    for (int j = 0; j < i; ++j) v -= l[i * k + j] * x[j];
    x[i] = v / l[i * k + i];
  }
  for (int i = k - 1; i >= 0; --i) {
    double v = x[i];
    for (int j = i + 1; j < k; ++j) v -= l[j * k + i] * x[j];
    x[i] = v / l[i * k + i];
  }
}

// solve.Retargeter.solve's bls: min |P w - d|^2 + |r scale diag(cost) w|^2,
// 0 <= w <= 1, for all frames. q: k x k Gram (without ridge), rhs: f x k
// right-hand sides (rhs = d P). x: f x k out.
void BoundedLS(std::vector<double> q, int k, const std::vector<double>& rhs,
               int f, const double* ridge2, std::vector<double>& x) {
  x.assign(static_cast<size_t>(f) * k, 0.0);
  if (!k) return;
  for (int i = 0; i < k; ++i) q[i * k + i] += ridge2[i];
  double lmax = MaxEig(q, k);
  std::vector<double> chol = q;
  if (!Cholesky(chol, k)) return;
  std::vector<double> g(k);
  for (int t = 0; t < f; ++t) {
    double* xt = &x[static_cast<size_t>(t) * k];
    const double* b = &rhs[static_cast<size_t>(t) * k];
    memcpy(xt, b, k * sizeof(double));
    CholSolve(chol, k, xt);
    for (int i = 0; i < k; ++i) xt[i] = std::clamp(xt[i], 0.0, 1.0);
    for (int it = 0; it < 400; ++it) {  // projected gradient onto [0,1]
      for (int i = 0; i < k; ++i) {
        double v = -b[i];
        for (int j = 0; j < k; ++j) v += xt[j] * q[j * k + i];
        g[i] = v;
      }
      for (int i = 0; i < k; ++i)
        xt[i] = std::clamp(xt[i] - g[i] / lmax, 0.0, 1.0);
    }
  }
}

// SDK skin animator smoothing: two cascaded one-pole filters, state starts
// at the first frame.
void Smooth2(const std::vector<float>& raw, int f, float a,
             std::vector<float>& out) {
  out.resize(static_cast<size_t>(f) * kNumPca);
  for (int k = 0; k < kNumPca; ++k) {
    float y1 = raw[k], y2 = raw[k];
    out[k] = raw[k];
    for (int t = 1; t < f; ++t) {
      float p = raw[static_cast<size_t>(t) * kNumRaw + k];
      if (a > 0) {
        y1 += (p - y1) * a;
        y2 += (y1 - y2) * a;
      } else {
        y1 = y2 = p;
      }
      out[static_cast<size_t>(t) * kNumPca + k] = y2;
    }
  }
}
float Alpha(float s) {
  return s > 0 ? 1.0f - std::pow(0.5f, (1.0f / kFps) / s) : -1.0f;
}

float Ease(double x) {
  double c = std::clamp(x, 0.0, 1.0);
  return static_cast<float>(c * c * (3 - 2 * c));
}
double Envelope(double s, double dur, const Settings& st) {
  double e = Ease(s / std::max(static_cast<double>(st.attack), 1e-3)) *
             Ease((dur - s) / std::max(static_cast<double>(st.release), 1e-3));
  return st.onset + (1 - st.onset) * e;
}
double Intensity(int value, double floor, double gain, double cap) {
  return std::min(cap,
                  (floor + (1 - floor) * std::min(1.0, value / 100.0)) * gain);
}
double Round(double x, double scale) {
  return std::nearbyint(x * scale) / scale;
}

// emotion_map.MAP in network order (amazement anger cheekiness disgust fear
// grief joy outofbreath pain sadness).
void ExplicitEmotion(int type, int value, const Settings& s,
                     float out[kNumMod]) {
  static const float kMap[8][kNumMod] = {{},
                                         {0, 1},
                                         {0, 0, 0, 1},
                                         {0, 0, 0, 0, 1},
                                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
                                         {0, 0, 0, 0, 0, 0, 1},
                                         {1},
                                         {0.3f, 0, 0.3f}};
  for (int i = 0; i < kNumMod; ++i) out[i] = 0;
  if (type < 1 || type > 7 || !s.emotion_to_a2f) return;
  double k = Intensity(value, s.a2f_floor, s.a2f_gain, s.a2f_cap);
  for (int i = 0; i < kNumMod; ++i)
    if (kMap[type][i] > 0)
      out[i] = static_cast<float>(Round(
          std::min(static_cast<double>(s.a2f_cap), kMap[type][i] * k), 1e3));
}

// emotion_map.envelope_keys -> (sample, scale) keyframes as a2f-bake feeds
// the emotion accumulator.
std::vector<std::pair<int64_t, float>> EnvelopeKeys(double dur,
                                                    const Settings& s) {
  double a = std::min(static_cast<double>(s.attack), dur / 2),
         r = std::min(static_cast<double>(s.release), dur / 2);
  std::vector<double> t;
  for (int i = 0; i < 6; ++i) t.push_back(a * i / 5);
  for (int i = 0; i < 6; ++i) t.push_back((dur - r) + r * i / 5);
  for (double& x : t) x = std::clamp(x, 0.0, dur);
  std::sort(t.begin(), t.end());
  t.erase(std::unique(t.begin(), t.end()), t.end());
  std::vector<std::pair<int64_t, float>> k;
  for (double x : t) {
    float sec = static_cast<float>(Round(x, 1e4)),
          v = static_cast<float>(Round(Envelope(x, dur, s), 1e4));
    k.emplace_back(static_cast<int64_t>(sec * static_cast<float>(kSampleRate)),
                   v);
  }
  return k;
}
float SampleKeys(const std::vector<std::pair<int64_t, float>>& k, int64_t t) {
  if (t <= k.front().first) return k.front().second;
  if (t >= k.back().first) return k.back().second;
  size_t i = 1;
  while (k[i].first < t) ++i;
  if (k[i].first == t) return k[i].second;
  float a = static_cast<float>(t - k[i - 1].first) /
            static_cast<float>(k[i].first - k[i - 1].first);
  return k[i - 1].second * (1 - a) + k[i].second * a;
}

// numpy default (linear).
double Percentile(std::vector<double> v, double q) {
  std::sort(v.begin(), v.end());
  double pos = q / 100.0 * (v.size() - 1);
  size_t i = static_cast<size_t>(pos);
  double f = pos - i;
  return i + 1 < v.size() ? v[i] + (v[i + 1] - v[i]) * f : v[i];
}

// expression.loudness / solve smoothing: forward-backward one-pole, twice.
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
bool Generate(const Data& d, Net& net, const Settings& s, const float* pcm,
              size_t n, int emotion, int value, bool female, Line& out,
              std::string& err, Timing* timing, const DirectProj* direct) {
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

  // --- network: audio windows + [implicit, explicit] emotion per frame ---
  float emo10[kNumMod];
  ExplicitEmotion(emotion, value, s, emo10);
  auto keys = EnvelopeKeys(static_cast<double>(n) / kSampleRate, s);
  const float* implicit = d.arrays.at("implicit").values.data();
  int batch = std::max(1, net.Batch());
  std::vector<float> x(static_cast<size_t>(batch) * kWindow),
      em(static_cast<size_t>(batch) * kNumEmo),
      y(static_cast<size_t>(batch) * kNumOut),
      raw(static_cast<size_t>(num_frames) * kNumRaw);
  for (int f0 = 0; f0 < num_frames; f0 += batch) {
    int b = std::min(batch, num_frames - f0);
    std::fill(x.begin(), x.end(), 0.f);
    std::fill(em.begin(), em.end(), 0.f);
    for (int i = 0; i < b; ++i) {
      int64_t start =
          static_cast<int64_t>(f0 + i) * kSampleRate / kFps - kWindowOffset;
      float* xi = &x[static_cast<size_t>(i) * kWindow];
      for (int j = 0; j < kWindow; ++j) {
        int64_t p = start + j;
        xi[j] = (p >= 0 && p < static_cast<int64_t>(n))
                    ? pcm[p] * s.input_strength
                    : 0.f;
      }
      float* ei = &em[static_cast<size_t>(i) * kNumEmo];
      memcpy(ei, implicit, kNumImplicit * sizeof(float));
      float sc = SampleKeys(keys, start + kWindowOffset);
      for (int j = 0; j < kNumMod; ++j) ei[kNumImplicit + j] = emo10[j] * sc;
    }
    if (!net.Run(x.data(), em.data(), batch, y.data(), err)) return false;
    for (int i = 0; i < b; ++i)
      memcpy(&raw[static_cast<size_t>(f0 + i) * kNumRaw],
             &y[static_cast<size_t>(i) * kNumOut], kNumRaw * sizeof(float));
  }
  auto t1 = Clock::now();

  // --- SDK skin animator (in PCA space) relative to the silence rest pose ---
  const Array& sil = d.arrays.at("sil_raw");
  float alpha_u = Alpha(s.upper_smoothing), alpha_l = Alpha(s.lower_smoothing);
  std::vector<float> pca_u, pca_l, sil_u, sil_l;
  Smooth2(raw, num_frames, alpha_u, pca_u);
  Smooth2(raw, num_frames, alpha_l, pca_l);
  Smooth2(sil.values, static_cast<int>(sil.rows), alpha_u, sil_u);
  Smooth2(sil.values, static_cast<int>(sil.rows), alpha_l, sil_l);
  double rest_u[kNumPca] = {}, rest_l[kNumPca] = {}, rest_t[kNumTongue] = {};
  for (int t = 15; t < 45; ++t) {
    for (int k = 0; k < kNumPca; ++k) {
      rest_u[k] += sil_u[static_cast<size_t>(t) * kNumPca + k] / 30.0;
      rest_l[k] += sil_l[static_cast<size_t>(t) * kNumPca + k] / 30.0;
    }
    for (int k = 0; k < kNumTongue; ++k)
      rest_t[k] += sil[t][kNumPca + k] / 30.0;
  }

  // --- folded retarget: right-hand sides per frame ---
  std::string p = female ? "f_" : "m_";
  const Array &ub1 = d.arrays.at(p + "U_b1"), &lb1 = d.arrays.at(p + "L_b1"),
              &ub2 = d.arrays.at(p + "U_b2"), &lb2 = d.arrays.at(p + "L_b2");
  const Array &uf = d.arrays.at(p + "U_f"), &lf = d.arrays.at(p + "L_f"),
              &g1 = d.arrays.at(p + "G1"), &pf = d.arrays.at(p + "PF"),
              &g2 = d.arrays.at(p + "G2");
  const float* scal = d.arrays.at(p + "scal").values.data();
  const double scale = scal[0], fw_base = scal[1], feat_scale = scal[2],
               jawk[4] = {scal[3], scal[4], scal[5], scal[6]};
  const double cu = static_cast<double>(s.skin_strength) *
                    s.upper_face_strength * s.strength,
               cl = static_cast<double>(s.skin_strength) *
                    s.lower_face_strength * s.strength;
  std::vector<double> b1(static_cast<size_t>(num_frames) * kNumMouth, 0.0),
      b2(static_cast<size_t>(num_frames) * kNumSolve, 0.0),
      fm(static_cast<size_t>(num_frames) * kNumFeat, 0.0);
  for (int t = 0; t < num_frames; ++t) {
    double* b1t = &b1[static_cast<size_t>(t) * kNumMouth];
    double* b2t = &b2[static_cast<size_t>(t) * kNumSolve];
    double* fmt = &fm[static_cast<size_t>(t) * kNumFeat];
    for (int k = 0; k < kNumPca; ++k) {
      double du =
                 cu * (pca_u[static_cast<size_t>(t) * kNumPca + k] - rest_u[k]),
             dl =
                 cl * (pca_l[static_cast<size_t>(t) * kNumPca + k] - rest_l[k]);
      const float *u1 = ub1[k], *l1 = lb1[k], *u2 = ub2[k], *l2 = lb2[k],
                  *ufk = uf[k], *lfk = lf[k];
      for (int j = 0; j < kNumMouth; ++j) b1t[j] += du * u1[j] + dl * l1[j];
      for (int j = 0; j < kNumSolve; ++j) b2t[j] += du * u2[j] + dl * l2[j];
      for (int j = 0; j < kNumFeat; ++j) fmt[j] += du * ufk[j] + dl * lfk[j];
    }
    for (int j = 0; j < kNumFeat; ++j) fmt[j] *= feat_scale;
  }

  // --- stage 1: mouth channels on lower-face rows + mouth features ---
  const double fw = s.feature_weight * fw_base, fw2 = fw * fw, r = s.ridge;
  std::vector<double> w(static_cast<size_t>(num_frames) * kNumSolve, 0.0), x_s,
      q, rhs;
  {
    std::vector<int> sel;
    for (int j = 0; j < kNumMouth; ++j)
      if (s.solve[j]) sel.push_back(j);
    int k = static_cast<int>(sel.size());
    q.assign(static_cast<size_t>(k) * k, 0.0);
    rhs.assign(static_cast<size_t>(num_frames) * k, 0.0);
    std::vector<double> ridge2(k);
    for (int a = 0; a < k; ++a) {
      ridge2[a] = std::pow(r * scale * s.cost[sel[a]], 2);
      for (int b = 0; b < k; ++b) {
        double v = g1[sel[a]][sel[b]];
        for (int qq = 0; qq < kNumFeat; ++qq)
          v += fw2 * pf[qq][sel[a]] * pf[qq][sel[b]];
        q[a * k + b] = v;
      }
    }
    for (int t = 0; t < num_frames; ++t)
      for (int a = 0; a < k; ++a) {
        double v = b1[static_cast<size_t>(t) * kNumMouth + sel[a]];
        for (int qq = 0; qq < kNumFeat; ++qq)
          v +=
              fw2 * fm[static_cast<size_t>(t) * kNumFeat + qq] * pf[qq][sel[a]];
        rhs[static_cast<size_t>(t) * k + a] = v;
      }
    BoundedLS(q, k, rhs, num_frames, ridge2.data(), x_s);
    for (int t = 0; t < num_frames; ++t)
      for (int a = 0; a < k; ++a)
        w[static_cast<size_t>(t) * kNumSolve + sel[a]] =
            x_s[static_cast<size_t>(t) * k + a];
  }
  // --- stage 2: modifiers on upper-face rows, fitting what the mouth
  // channels leave over ---
  {
    std::vector<int> sel;
    for (int j = kNumMouth; j < kNumSolve; ++j)
      if (s.solve[j]) sel.push_back(j);
    int k = static_cast<int>(sel.size());
    q.assign(static_cast<size_t>(k) * k, 0.0);
    rhs.assign(static_cast<size_t>(num_frames) * k, 0.0);
    std::vector<double> ridge2(k);
    for (int a = 0; a < k; ++a) {
      ridge2[a] = std::pow(r * scale * s.cost[sel[a]], 2);
      for (int b = 0; b < k; ++b) q[a * k + b] = g2[sel[a]][sel[b]];
    }
    for (int t = 0; t < num_frames; ++t)
      for (int a = 0; a < k; ++a) {
        double v = b2[static_cast<size_t>(t) * kNumSolve + sel[a]];
        for (int m = 0; m < kNumMouth; ++m)
          v -= w[static_cast<size_t>(t) * kNumSolve + m] * g2[m][sel[a]];
        rhs[static_cast<size_t>(t) * k + a] = v;
      }
    BoundedLS(q, k, rhs, num_frames, ridge2.data(), x_s);
    for (int t = 0; t < num_frames; ++t)
      for (int a = 0; a < k; ++a)
        w[static_cast<size_t>(t) * kNumSolve + sel[a]] =
            x_s[static_cast<size_t>(t) * k + a];
  }

  // --- tongue layer: hand part of the jaw opening to DST / Th when A2F's
  // tongue tip rises ---
  if (s.tongue_on) {
    const Array& tip = d.arrays.at("tip");
    for (int t = 0; t < num_frames; ++t) {
      double td[3] = {};
      for (int k = 0; k < kNumTongue; ++k) {
        double dq =
            s.tongue_strength *
            (raw[static_cast<size_t>(t) * kNumRaw + kNumPca + k] - rest_t[k]);
        for (int c = 0; c < 3; ++c) td[c] += dq * tip[k][c];
      }
      double* wt = &w[static_cast<size_t>(t) * kNumSolve];
      double up = std::clamp(
          (td[1] - s.tip_up_min) / (s.tip_up_full - s.tip_up_min), 0.0, 1.0);
      bool fwd = td[2] > s.th_fwd;
      double jaw = wt[kAah] * jawk[0] + wt[kBigAah] * jawk[1],
             share = up * s.tongue_share;
      for (int pass = 0; pass < 2; ++pass) {
        int j = pass == 0 ? kTh : kDst;
        bool mask = pass == 0 ? fwd : !fwd;
        double k = std::max(jawk[pass == 0 ? 3 : 2], 1e-3);
        double ww = std::clamp(share * (mask ? 1.0 : 0.0) * jaw / k, 0.0, 1.0),
               moved = ww * k;
        wt[j] = std::max(wt[j], ww);
        double take_aah = std::min(wt[kAah] * jawk[0], moved);
        wt[kAah] -= take_aah / jawk[0];
        wt[kBigAah] =
            std::clamp(wt[kBigAah] - (moved - take_aah) / jawk[1], 0.0, 1.0);
      }
    }
  }

  // --- zero-phase smoothing, per-channel response, expression curve,
  // quantize ---
  out.frames = num_frames;
  out.data.assign(static_cast<size_t>(num_frames) * kNumChannels, 0);
  std::vector<double> c(num_frames);
  for (int j = 0; j < kNumSolve; ++j) {
    for (int t = 0; t < num_frames; ++t)
      c[t] = w[static_cast<size_t>(t) * kNumSolve + j];
    ZeroPhase(c, s.smooth);
    for (int t = 0; t < num_frames; ++t) {
      double v = std::clamp(c[t] * s.gain[j] + s.offset[j], 0.0, 1.0);
      v = std::clamp(std::pow(v, static_cast<double>(s.gamma[j])), 0.0,
                     static_cast<double>(s.max[j]));
      out.data[static_cast<size_t>(t) * kNumChannels + j] =
          static_cast<uint8_t>(std::nearbyint(std::clamp(v, 0.0, 1.0) * 255));
    }
  }
  out.expr = (emotion >= 1 && emotion <= 7 && s.expression_on)
                 ? kEngineExpr[emotion]
                 : 255;
  if (out.expr != 255) {
    // expression.loudness: per-frame RMS in dB, normalised between the 20th
    // and 97th percentile, slow smoothing
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
    double level = Intensity(value, s.expr_floor, s.expr_gain, 1.0),
           dur = static_cast<double>(num_frames) / kFps;
    for (int t = 0; t < num_frames; ++t) {
      double v = level * Envelope(static_cast<double>(t) / kFps, dur, s) *
                 (1 - s.emphasis + s.emphasis * db[t]);
      out.data[static_cast<size_t>(t) * kNumChannels + kNumSolve] =
          static_cast<uint8_t>(std::nearbyint(std::clamp(v, 0.0, 1.0) * 255));
    }
  }
  // --- A2F Direct coefficients ---
  out.direct_dim = 0;
  out.direct.clear();
  if (direct) {
    out.direct_dim = direct->k + kNumTongue;
    out.direct.assign(static_cast<size_t>(num_frames) * out.direct_dim, 0.f);
    for (int t = 0; t < num_frames; ++t) {
      const float* y = &raw[static_cast<size_t>(t) * kNumRaw];
      float* o = &out.direct[static_cast<size_t>(t) * out.direct_dim];
      for (uint32_t k = 0; k < direct->k; ++k) {
        const float* p = direct->proj + static_cast<size_t>(k) * kNumPca;
        for (int j = 0; j < kNumPca; ++j)
          o[k] += (y[j] - direct->rest[j]) * p[j];
      }
      for (int j = 0; j < kNumTongue; ++j)
        o[direct->k + j] = y[kNumPca + j] - direct->rest[kNumPca + j];
    }
  }
  if (timing) {
    timing->net = std::chrono::duration<double>(t1 - t0).count();
    timing->solve = std::chrono::duration<double>(Clock::now() - t1).count();
  }
  return true;
}

}  // namespace a2f
