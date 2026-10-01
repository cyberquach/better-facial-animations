// A2FLive core: decode, network windows, SDK skin animator, retarget solve, expression curve.
#include "live_core.h"
#include <cmath>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <chrono>
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswresample/swresample.h>
#include <libavutil/channel_layout.h>
}

namespace a2f {

const char* const CHANNEL_NAMES[NSOLVE] = {
    "Aah", "BigAah", "BMP", "ChJSh", "DST", "Eee", "Eh", "FV", "I", "K", "N", "Oh", "OohQ", "R", "Th", "W",
    "BlinkLeft", "BlinkRight", "BrowDownLeft", "BrowDownRight", "BrowInLeft", "BrowInRight", "BrowUpLeft", "BrowUpRight",
    "SquintLeft", "SquintRight"};
enum { AAH = 0, BIGAAH = 1, DST = 4, TH = 14 };

// ---------------- data file ----------------
bool Data::Load(const std::wstring& path, std::string& err) {
    FILE* f = _wfopen(path.c_str(), L"rb");
    if (!f) { err = "cannot open data file"; return false; }
    char magic[4]; uint32_t ver = 0, count = 0;
    bool ok = fread(magic, 1, 4, f) == 4 && !memcmp(magic, "A2FD", 4) && fread(&ver, 4, 1, f) == 1 && ver == 1 && fread(&count, 4, 1, f) == 1;
    for (uint32_t i = 0; ok && i < count; ++i) {
        char name[33] = {}; Array x;
        ok = fread(name, 1, 32, f) == 32 && fread(&x.rows, 4, 1, f) == 1 && fread(&x.cols, 4, 1, f) == 1 && (size_t)x.rows * x.cols < (1u << 24);
        if (!ok) break;
        x.v.resize((size_t)x.rows * x.cols);
        ok = fread(x.v.data(), 4, x.v.size(), f) == x.v.size();
        a[name] = std::move(x);
    }
    fclose(f);
    if (!ok) { err = "bad data file"; return false; }
    const Array *cfg, *tun, *emo, *ch;
    if (!(cfg = Get("cfg", 1, 7, err)) || !(tun = Get("tun", 1, 9, err)) || !(emo = Get("emo", 1, 9, err)) || !(ch = Get("ch", NSOLVE, 6, err))) return false;
    Settings& s = defaults; const float* c = cfg->v.data();
    s.inputStrength = c[0]; s.upperSmoothing = c[1]; s.lowerSmoothing = c[2]; s.upperFaceStrength = c[3]; s.lowerFaceStrength = c[4];
    s.skinStrength = c[5]; s.tongueStrength = c[6];
    const float* t = tun->v.data();
    s.strength = t[0]; s.ridge = t[1]; s.smooth = t[2]; s.featureWeight = t[3];
    s.tongueOn = t[4] != 0; s.tipUpFull = t[5]; s.tipUpMin = t[6]; s.thFwd = t[7]; s.tongueShare = t[8];
    const float* e = emo->v.data();
    s.a2fFloor = e[0]; s.a2fGain = e[1]; s.a2fCap = e[2]; s.exprFloor = e[3]; s.exprGain = e[4];
    s.onset = e[5]; s.attack = e[6]; s.release = e[7]; s.emphasis = e[8];
    for (int j = 0; j < NSOLVE; ++j) {
        const float* r = (*ch)[j];
        s.gain[j] = r[0]; s.gamma[j] = r[1]; s.offset[j] = r[2]; s.max[j] = r[3]; s.cost[j] = r[4]; s.solve[j] = r[5] != 0;
    }
    for (const char* sx : {"m_", "f_"}) {
        std::string p = sx;
        for (const char* side : {"U_", "L_"})
            if (!Get(p + side + "b1", NPCA, NMOUTH, err) || !Get(p + side + "b2", NPCA, NSOLVE, err) || !Get(p + side + "f", NPCA, NFEAT, err)) return false;
        if (!Get(p + "G1", NMOUTH, NMOUTH, err) || !Get(p + "PF", NFEAT, NMOUTH, err) || !Get(p + "G2", NSOLVE, NSOLVE, err) || !Get(p + "scal", 1, 7, err)) return false;
    }
    return Get("implicit", 1, NIMPLICIT, err) && Get("tip", NTONGUE, 3, err) && a.count("sil_raw") && a["sil_raw"].cols == NRAW && a["sil_raw"].rows >= 45;
}

const Array* Data::Get(const std::string& name, uint32_t rows, uint32_t cols, std::string& err) const {
    auto it = a.find(name);
    if (it == a.end() || it->second.rows != rows || it->second.cols != cols) { err = "data file: missing or bad array " + name; return nullptr; }
    return &it->second;
}

// ---------------- audio ----------------
namespace {
struct Mem { const uint8_t* p; size_t n, pos; };
int MemRead(void* o, uint8_t* buf, int sz) {
    auto* m = (Mem*)o; size_t k = std::min((size_t)sz, m->n - m->pos);
    if (!k) return AVERROR_EOF;
    memcpy(buf, m->p + m->pos, k); m->pos += k; return (int)k;
}
int64_t MemSeek(void* o, int64_t off, int whence) {
    auto* m = (Mem*)o;
    if (whence & AVSEEK_SIZE) return (int64_t)m->n;
    whence &= ~AVSEEK_FORCE;
    int64_t np = whence == SEEK_SET ? off : whence == SEEK_CUR ? (int64_t)m->pos + off : (int64_t)m->n + off;
    if (np < 0 || np > (int64_t)m->n) return -1;
    m->pos = (size_t)np; return np;
}
}  // namespace

bool DecodeAudio(const uint8_t* data, size_t n, std::vector<float>& pcm, std::string& err) {
    pcm.clear();
    if (n >= 12 && !memcmp(data, "FUZE", 4)) {  // FUZE: magic, version, lip size, lip data, audio
        uint32_t lip; memcpy(&lip, data + 8, 4);
        if (12 + (size_t)lip > n) { err = "bad fuz"; return false; }
        data += 12 + lip; n -= 12 + lip;
    }
    Mem mem{data, n, 0};
    const int iosz = 32768;
    uint8_t* iobuf = (uint8_t*)av_malloc(iosz);
    AVIOContext* io = avio_alloc_context(iobuf, iosz, 0, &mem, MemRead, nullptr, MemSeek);
    AVFormatContext* fmt = avformat_alloc_context();
    AVCodecContext* dec = nullptr; SwrContext* swr = nullptr; AVPacket* pkt = av_packet_alloc(); AVFrame* fr = av_frame_alloc();
    std::vector<int16_t> s16;
    bool ok = false; int st = -1;
    fmt->pb = io; fmt->flags |= AVFMT_FLAG_CUSTOM_IO;
    auto drain = [&](AVFrame* in) -> bool {  // in == nullptr flushes the resampler
        int cap = in ? swr_get_out_samples(swr, in->nb_samples) : swr_get_out_samples(swr, 0);
        if (cap <= 0 && !in) return true;
        size_t at = s16.size(); s16.resize(at + std::max(cap, 0) + 256);
        uint8_t* outp = (uint8_t*)(s16.data() + at);
        int got = swr_convert(swr, &outp, (int)(s16.size() - at), in ? (const uint8_t**)in->extended_data : nullptr, in ? in->nb_samples : 0);
        if (got < 0) { err = "resample failed"; return false; }
        s16.resize(at + got); return true;
    };
    auto receive = [&]() -> bool {
        while (true) {
            int r = avcodec_receive_frame(dec, fr);
            if (r == AVERROR(EAGAIN) || r == AVERROR_EOF) return true;
            if (r < 0) { err = "decode failed"; return false; }
            bool k = drain(fr); av_frame_unref(fr);
            if (!k) return false;
        }
    };
    do {
        if (avformat_open_input(&fmt, nullptr, nullptr, nullptr) < 0) { fmt = nullptr; err = "unknown audio format"; break; }
        if (avformat_find_stream_info(fmt, nullptr) < 0) { err = "no stream info"; break; }
        st = av_find_best_stream(fmt, AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);
        if (st < 0) { err = "no audio stream"; break; }
        const AVCodecParameters* par = fmt->streams[st]->codecpar;
        const AVCodec* codec = avcodec_find_decoder(par->codec_id);
        if (!codec) { err = "unsupported codec"; break; }
        dec = avcodec_alloc_context3(codec);
        if (avcodec_parameters_to_context(dec, par) < 0 || avcodec_open2(dec, codec, nullptr) < 0) { err = "cannot open decoder"; break; }
        AVChannelLayout mono = AV_CHANNEL_LAYOUT_MONO;
        if (swr_alloc_set_opts2(&swr, &mono, AV_SAMPLE_FMT_S16, SR, &dec->ch_layout, dec->sample_fmt, dec->sample_rate, 0, nullptr) < 0 || swr_init(swr) < 0) {
            err = "cannot create resampler"; break;
        }
        bool bad = false;
        while (!bad && av_read_frame(fmt, pkt) >= 0) {
            if (pkt->stream_index == st && avcodec_send_packet(dec, pkt) >= 0) bad = !receive();
            av_packet_unref(pkt);
        }
        if (bad) break;
        avcodec_send_packet(dec, nullptr);
        if (!receive() || !drain(nullptr)) break;
        ok = !s16.empty();
        if (!ok) err = "no audio decoded";
    } while (false);
    av_frame_free(&fr); av_packet_free(&pkt); swr_free(&swr); avcodec_free_context(&dec);
    if (fmt) avformat_close_input(&fmt);
    avio_context_free(&io);  // frees the context; its buffer may have been reallocated by avio
    if (ok) { pcm.resize(s16.size()); for (size_t i = 0; i < s16.size(); ++i) pcm[i] = s16[i] * (1.0f / 32768.0f); }
    return ok;
}

// ---------------- small linear algebra ----------------
namespace {
// largest eigenvalue of a symmetric k x k matrix (cyclic Jacobi)
double MaxEig(std::vector<double> A, int k) {
    for (int sweep = 0; sweep < 64; ++sweep) {
        double off = 0;
        for (int p = 0; p < k; ++p) for (int q = p + 1; q < k; ++q) off += A[p * k + q] * A[p * k + q];
        if (off < 1e-22) break;
        for (int p = 0; p < k; ++p) for (int q = p + 1; q < k; ++q) {
            double apq = A[p * k + q]; if (std::fabs(apq) < 1e-300) continue;
            double th = (A[q * k + q] - A[p * k + p]) / (2 * apq);
            double t = (th >= 0 ? 1 : -1) / (std::fabs(th) + std::sqrt(th * th + 1)), c = 1 / std::sqrt(t * t + 1), s = t * c;
            for (int r = 0; r < k; ++r) {  // A = J^T A J
                double arp = A[r * k + p], arq = A[r * k + q];
                A[r * k + p] = c * arp - s * arq; A[r * k + q] = s * arp + c * arq;
            }
            for (int r = 0; r < k; ++r) {
                double apr = A[p * k + r], aqr = A[q * k + r];
                A[p * k + r] = c * apr - s * aqr; A[q * k + r] = s * apr + c * aqr;
            }
        }
    }
    double m = -1e300; for (int i = 0; i < k; ++i) m = std::max(m, A[i * k + i]); return m;
}

bool Cholesky(std::vector<double>& L, int k) {  // in place, lower triangle
    for (int j = 0; j < k; ++j) {
        double d = L[j * k + j]; for (int i = 0; i < j; ++i) d -= L[j * k + i] * L[j * k + i];
        if (d <= 0) return false;
        d = std::sqrt(d); L[j * k + j] = d;
        for (int r = j + 1; r < k; ++r) {
            double v = L[r * k + j]; for (int i = 0; i < j; ++i) v -= L[r * k + i] * L[j * k + i];
            L[r * k + j] = v / d;
        }
    }
    return true;
}
void CholSolve(const std::vector<double>& L, int k, double* x) {
    for (int i = 0; i < k; ++i) { double v = x[i]; for (int j = 0; j < i; ++j) v -= L[i * k + j] * x[j]; x[i] = v / L[i * k + i]; }
    for (int i = k - 1; i >= 0; --i) { double v = x[i]; for (int j = i + 1; j < k; ++j) v -= L[j * k + i] * x[j]; x[i] = v / L[i * k + i]; }
}

// solve.Retargeter.solve's bls: min |P w - d|^2 + |r scale diag(cost) w|^2, 0 <= w <= 1, for all frames.
// Q: k x k Gram (without ridge), B: F x k right-hand sides (B = d P). X: F x k out.
void BoundedLS(std::vector<double> Q, int k, const std::vector<double>& B, int F, const double* ridge2, std::vector<double>& X) {
    X.assign((size_t)F * k, 0.0);
    if (!k) return;
    for (int i = 0; i < k; ++i) Q[i * k + i] += ridge2[i];
    double Lmax = MaxEig(Q, k);
    std::vector<double> C = Q;
    if (!Cholesky(C, k)) return;
    std::vector<double> g(k);
    for (int f = 0; f < F; ++f) {
        double* x = &X[(size_t)f * k]; const double* b = &B[(size_t)f * k];
        memcpy(x, b, k * sizeof(double)); CholSolve(C, k, x);
        for (int i = 0; i < k; ++i) x[i] = std::clamp(x[i], 0.0, 1.0);
        for (int it = 0; it < 400; ++it) {  // projected gradient onto [0,1]
            for (int i = 0; i < k; ++i) { double v = -b[i]; for (int j = 0; j < k; ++j) v += x[j] * Q[j * k + i]; g[i] = v; }
            for (int i = 0; i < k; ++i) x[i] = std::clamp(x[i] - g[i] / Lmax, 0.0, 1.0);
        }
    }
}

// SDK skin animator smoothing: two cascaded one-pole filters, state starts at the first frame
void Smooth2(const std::vector<float>& raw, int F, float a, std::vector<float>& out) {
    out.resize((size_t)F * NPCA);
    for (int k = 0; k < NPCA; ++k) {
        float y1 = raw[k], y2 = raw[k]; out[k] = raw[k];
        for (int t = 1; t < F; ++t) {
            float p = raw[(size_t)t * NRAW + k];
            if (a > 0) { y1 += (p - y1) * a; y2 += (y1 - y2) * a; } else { y1 = y2 = p; }
            out[(size_t)t * NPCA + k] = y2;
        }
    }
}
float Alpha(float s) { return s > 0 ? 1.0f - std::pow(0.5f, (1.0f / FPS) / s) : -1.0f; }

float Ease(double x) { double c = std::clamp(x, 0.0, 1.0); return (float)(c * c * (3 - 2 * c)); }
double Envelope(double s, double dur, const Settings& st) {
    double e = Ease(s / std::max((double)st.attack, 1e-3)) * Ease((dur - s) / std::max((double)st.release, 1e-3));
    return st.onset + (1 - st.onset) * e;
}
double Intensity(int value, double floor, double gain, double cap) {
    return std::min(cap, (floor + (1 - floor) * std::min(1.0, value / 100.0)) * gain);
}
double Round(double x, double scale) { return std::nearbyint(x * scale) / scale; }

// emotion_map.MAP in network order (amazement anger cheekiness disgust fear grief joy outofbreath pain sadness)
void ExplicitEmotion(int type, int value, const Settings& s, float out[NMOD]) {
    static const float MAP[8][NMOD] = {
        {}, {0, 1}, {0, 0, 0, 1}, {0, 0, 0, 0, 1}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 1}, {0, 0, 0, 0, 0, 0, 1}, {1}, {0.3f, 0, 0.3f}};
    for (int i = 0; i < NMOD; ++i) out[i] = 0;
    if (type < 1 || type > 7 || !s.emotionToA2F) return;
    double k = Intensity(value, s.a2fFloor, s.a2fGain, s.a2fCap);
    for (int i = 0; i < NMOD; ++i) if (MAP[type][i] > 0) out[i] = (float)Round(std::min((double)s.a2fCap, MAP[type][i] * k), 1e3);
}

// emotion_map.envelope_keys -> (sample, scale) keyframes as a2f-bake feeds the emotion accumulator
std::vector<std::pair<int64_t, float>> EnvelopeKeys(double dur, const Settings& s) {
    double a = std::min((double)s.attack, dur / 2), r = std::min((double)s.release, dur / 2);
    std::vector<double> t;
    for (int i = 0; i < 6; ++i) t.push_back(a * i / 5);
    for (int i = 0; i < 6; ++i) t.push_back((dur - r) + r * i / 5);
    for (double& x : t) x = std::clamp(x, 0.0, dur);
    std::sort(t.begin(), t.end()); t.erase(std::unique(t.begin(), t.end()), t.end());
    std::vector<std::pair<int64_t, float>> k;
    for (double x : t) {
        float sec = (float)Round(x, 1e4), v = (float)Round(Envelope(x, dur, s), 1e4);
        k.emplace_back((int64_t)(sec * (float)SR), v);
    }
    return k;
}
float SampleKeys(const std::vector<std::pair<int64_t, float>>& k, int64_t t) {
    if (t <= k.front().first) return k.front().second;
    if (t >= k.back().first) return k.back().second;
    size_t i = 1; while (k[i].first < t) ++i;
    if (k[i].first == t) return k[i].second;
    float a = (float)(t - k[i - 1].first) / (float)(k[i].first - k[i - 1].first);
    return k[i - 1].second * (1 - a) + k[i].second * a;
}

double Percentile(std::vector<double> v, double q) {  // numpy default (linear)
    std::sort(v.begin(), v.end());
    double pos = q / 100.0 * (v.size() - 1); size_t i = (size_t)pos; double f = pos - i;
    return i + 1 < v.size() ? v[i] + (v[i + 1] - v[i]) * f : v[i];
}

void ZeroPhase(std::vector<double>& e, double a) {  // expression.loudness / solve smoothing: forward-backward one-pole, twice
    size_t n = e.size();
    for (int pass = 0; pass < 2; ++pass) {
        for (size_t i = 1; i < n; ++i) e[i] = a * e[i - 1] + (1 - a) * e[i];
        std::reverse(e.begin(), e.end());
    }
}

// Engine expression index (channels.EXPRESSIONS) for Skyrim emotion types 1..7
const uint32_t ENGINE_EXPR[8] = {255, 0, 6, 1, 3, 2, 4, 5};
}  // namespace

// ---------------- the whole line ----------------
bool Generate(const Data& d, Net& net, const Settings& s, const float* pcm, size_t n, int emotion, int value, bool female,
              Line& out, std::string& err, Timing* timing) {
    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();
    if (!n) { err = "empty audio"; return false; }
    int F = 0; while ((int64_t)F * SR / FPS < (int64_t)n) ++F;

    // --- network: audio windows + [implicit, explicit] emotion per frame ---
    float emo10[NMOD]; ExplicitEmotion(emotion, value, s, emo10);
    auto keys = EnvelopeKeys((double)n / SR, s);
    const float* implicit = d.a.at("implicit").v.data();
    int B = std::max(1, net.Batch());
    std::vector<float> x((size_t)B * WIN), em((size_t)B * NEMO), y((size_t)B * NOUT), raw((size_t)F * NRAW);
    for (int f0 = 0; f0 < F; f0 += B) {
        int b = std::min(B, F - f0);
        std::fill(x.begin(), x.end(), 0.f); std::fill(em.begin(), em.end(), 0.f);
        for (int i = 0; i < b; ++i) {
            int64_t start = (int64_t)(f0 + i) * SR / FPS - WIN_OFS;
            float* xi = &x[(size_t)i * WIN];
            for (int j = 0; j < WIN; ++j) { int64_t p = start + j; xi[j] = (p >= 0 && p < (int64_t)n) ? pcm[p] * s.inputStrength : 0.f; }
            float* ei = &em[(size_t)i * NEMO];
            memcpy(ei, implicit, NIMPLICIT * sizeof(float));
            float sc = SampleKeys(keys, start + WIN_OFS);
            for (int j = 0; j < NMOD; ++j) ei[NIMPLICIT + j] = emo10[j] * sc;
        }
        if (!net.Run(x.data(), em.data(), B, y.data(), err)) return false;
        for (int i = 0; i < b; ++i) memcpy(&raw[(size_t)(f0 + i) * NRAW], &y[(size_t)i * NOUT], NRAW * sizeof(float));
    }
    auto t1 = clk::now();

    // --- SDK skin animator (in PCA space) relative to the silence rest pose ---
    const Array& sil = d.a.at("sil_raw");
    float aU = Alpha(s.upperSmoothing), aL = Alpha(s.lowerSmoothing);
    std::vector<float> pU, pL, sU, sL;
    Smooth2(raw, F, aU, pU); Smooth2(raw, F, aL, pL);
    Smooth2(sil.v, (int)sil.rows, aU, sU); Smooth2(sil.v, (int)sil.rows, aL, sL);
    double rU[NPCA] = {}, rL[NPCA] = {}, rT[NTONGUE] = {};
    for (int t = 15; t < 45; ++t) {
        for (int k = 0; k < NPCA; ++k) { rU[k] += sU[(size_t)t * NPCA + k] / 30.0; rL[k] += sL[(size_t)t * NPCA + k] / 30.0; }
        for (int k = 0; k < NTONGUE; ++k) rT[k] += sil[t][NPCA + k] / 30.0;
    }

    // --- folded retarget: right-hand sides per frame ---
    std::string p = female ? "f_" : "m_";
    const Array &Ub1 = d.a.at(p + "U_b1"), &Lb1 = d.a.at(p + "L_b1"), &Ub2 = d.a.at(p + "U_b2"), &Lb2 = d.a.at(p + "L_b2");
    const Array &Uf = d.a.at(p + "U_f"), &Lf = d.a.at(p + "L_f"), &G1 = d.a.at(p + "G1"), &PF = d.a.at(p + "PF"), &G2 = d.a.at(p + "G2");
    const float* scal = d.a.at(p + "scal").v.data();
    const double scale = scal[0], fwBase = scal[1], featScale = scal[2], jawk[4] = {scal[3], scal[4], scal[5], scal[6]};
    const double cu = (double)s.skinStrength * s.upperFaceStrength * s.strength, cl = (double)s.skinStrength * s.lowerFaceStrength * s.strength;
    std::vector<double> b1((size_t)F * NMOUTH, 0.0), b2((size_t)F * NSOLVE, 0.0), fm((size_t)F * NFEAT, 0.0);
    for (int t = 0; t < F; ++t) {
        double* B1 = &b1[(size_t)t * NMOUTH]; double* B2 = &b2[(size_t)t * NSOLVE]; double* FM = &fm[(size_t)t * NFEAT];
        for (int k = 0; k < NPCA; ++k) {
            double du = cu * (pU[(size_t)t * NPCA + k] - rU[k]), dl = cl * (pL[(size_t)t * NPCA + k] - rL[k]);
            const float *u1 = Ub1[k], *l1 = Lb1[k], *u2 = Ub2[k], *l2 = Lb2[k], *uf = Uf[k], *lf = Lf[k];
            for (int j = 0; j < NMOUTH; ++j) B1[j] += du * u1[j] + dl * l1[j];
            for (int j = 0; j < NSOLVE; ++j) B2[j] += du * u2[j] + dl * l2[j];
            for (int j = 0; j < NFEAT; ++j) FM[j] += du * uf[j] + dl * lf[j];
        }
        for (int j = 0; j < NFEAT; ++j) FM[j] *= featScale;
    }

    // --- stage 1: mouth channels on lower-face rows + mouth features ---
    const double fw = s.featureWeight * fwBase, fw2 = fw * fw, r = s.ridge;
    std::vector<double> W((size_t)F * NSOLVE, 0.0), X, Q, Bs;
    {
        std::vector<int> sel; for (int j = 0; j < NMOUTH; ++j) if (s.solve[j]) sel.push_back(j);
        int k = (int)sel.size();
        Q.assign((size_t)k * k, 0.0); Bs.assign((size_t)F * k, 0.0);
        std::vector<double> ridge2(k);
        for (int a = 0; a < k; ++a) {
            ridge2[a] = std::pow(r * scale * s.cost[sel[a]], 2);
            for (int b = 0; b < k; ++b) {
                double v = G1[sel[a]][sel[b]];
                for (int q = 0; q < NFEAT; ++q) v += fw2 * PF[q][sel[a]] * PF[q][sel[b]];
                Q[a * k + b] = v;
            }
        }
        for (int t = 0; t < F; ++t)
            for (int a = 0; a < k; ++a) {
                double v = b1[(size_t)t * NMOUTH + sel[a]];
                for (int q = 0; q < NFEAT; ++q) v += fw2 * fm[(size_t)t * NFEAT + q] * PF[q][sel[a]];
                Bs[(size_t)t * k + a] = v;
            }
        BoundedLS(Q, k, Bs, F, ridge2.data(), X);
        for (int t = 0; t < F; ++t) for (int a = 0; a < k; ++a) W[(size_t)t * NSOLVE + sel[a]] = X[(size_t)t * k + a];
    }
    // --- stage 2: modifiers on upper-face rows, fitting what the mouth channels leave over ---
    {
        std::vector<int> sel; for (int j = NMOUTH; j < NSOLVE; ++j) if (s.solve[j]) sel.push_back(j);
        int k = (int)sel.size();
        Q.assign((size_t)k * k, 0.0); Bs.assign((size_t)F * k, 0.0);
        std::vector<double> ridge2(k);
        for (int a = 0; a < k; ++a) {
            ridge2[a] = std::pow(r * scale * s.cost[sel[a]], 2);
            for (int b = 0; b < k; ++b) Q[a * k + b] = G2[sel[a]][sel[b]];
        }
        for (int t = 0; t < F; ++t)
            for (int a = 0; a < k; ++a) {
                double v = b2[(size_t)t * NSOLVE + sel[a]];
                for (int m = 0; m < NMOUTH; ++m) v -= W[(size_t)t * NSOLVE + m] * G2[m][sel[a]];
                Bs[(size_t)t * k + a] = v;
            }
        BoundedLS(Q, k, Bs, F, ridge2.data(), X);
        for (int t = 0; t < F; ++t) for (int a = 0; a < k; ++a) W[(size_t)t * NSOLVE + sel[a]] = X[(size_t)t * k + a];
    }

    // --- tongue layer: hand part of the jaw opening to DST / Th when A2F's tongue tip rises ---
    if (s.tongueOn) {
        const Array& tip = d.a.at("tip");
        for (int t = 0; t < F; ++t) {
            double td[3] = {};
            for (int k = 0; k < NTONGUE; ++k) {
                double dq = s.tongueStrength * (raw[(size_t)t * NRAW + NPCA + k] - rT[k]);
                for (int c = 0; c < 3; ++c) td[c] += dq * tip[k][c];
            }
            double* w = &W[(size_t)t * NSOLVE];
            double up = std::clamp((td[1] - s.tipUpMin) / (s.tipUpFull - s.tipUpMin), 0.0, 1.0);
            bool fwd = td[2] > s.thFwd;
            double J = w[AAH] * jawk[0] + w[BIGAAH] * jawk[1], share = up * s.tongueShare;
            for (int pass = 0; pass < 2; ++pass) {
                int j = pass == 0 ? TH : DST; bool mask = pass == 0 ? fwd : !fwd;
                double k = std::max(jawk[pass == 0 ? 3 : 2], 1e-3);
                double ww = std::clamp(share * (mask ? 1.0 : 0.0) * J / k, 0.0, 1.0), moved = ww * k;
                w[j] = std::max(w[j], ww);
                double takeA = std::min(w[AAH] * jawk[0], moved);
                w[AAH] -= takeA / jawk[0];
                w[BIGAAH] = std::clamp(w[BIGAAH] - (moved - takeA) / jawk[1], 0.0, 1.0);
            }
        }
    }

    // --- zero-phase smoothing, per-channel response, expression curve, quantize ---
    out.frames = F; out.data.assign((size_t)F * NCH, 0);
    std::vector<double> c(F);
    for (int j = 0; j < NSOLVE; ++j) {
        for (int t = 0; t < F; ++t) c[t] = W[(size_t)t * NSOLVE + j];
        ZeroPhase(c, s.smooth);
        for (int t = 0; t < F; ++t) {
            double v = std::clamp(c[t] * s.gain[j] + s.offset[j], 0.0, 1.0);
            v = std::clamp(std::pow(v, (double)s.gamma[j]), 0.0, (double)s.max[j]);
            out.data[(size_t)t * NCH + j] = (uint8_t)std::nearbyint(std::clamp(v, 0.0, 1.0) * 255);
        }
    }
    out.expr = (emotion >= 1 && emotion <= 7 && s.expressionOn) ? ENGINE_EXPR[emotion] : 255;
    if (out.expr != 255) {
        // expression.loudness: per-frame RMS in dB, normalised between the 20th and 97th percentile, slow smoothing
        std::vector<double> db(F);
        double hop = (double)SR / FPS;
        for (int i = 0; i < F; ++i) {
            size_t a = (size_t)(i * hop), b = std::min((size_t)((i + 1) * hop), n);
            double e = 0; for (size_t k = a; k < b; ++k) e += (double)pcm[k] * pcm[k];
            db[i] = 20 * std::log10(std::sqrt((b > a ? e / (b - a) : 0) + 1e-12));
        }
        double lo = Percentile(db, 20), hi = Percentile(db, 97);
        for (double& v : db) v = std::clamp((v - lo) / std::max(hi - lo, 1e-3), 0.0, 1.0);
        ZeroPhase(db, 0.82);
        double level = Intensity(value, s.exprFloor, s.exprGain, 1.0), dur = (double)F / FPS;
        for (int t = 0; t < F; ++t) {
            double v = level * Envelope((double)t / FPS, dur, s) * (1 - s.emphasis + s.emphasis * db[t]);
            out.data[(size_t)t * NCH + NSOLVE] = (uint8_t)std::nearbyint(std::clamp(v, 0.0, 1.0) * 255);
        }
    }
    if (timing) {
        timing->net = std::chrono::duration<double>(t1 - t0).count();
        timing->solve = std::chrono::duration<double>(clk::now() - t1).count();
    }
    return true;
}

}  // namespace a2f
