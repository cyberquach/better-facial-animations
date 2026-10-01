#include "ort_net.h"
#include <windows.h>
#include <cstring>
#include <algorithm>
#include "onnxruntime_c_api.h"
// dml_provider_factory.h needs DirectML.h (MSVC SAL annotations); only its first entry is used
struct OrtDmlApi { OrtStatus*(ORT_API_CALL* SessionOptionsAppendExecutionProvider_DML)(OrtSessionOptions*, int device_id); };

namespace a2f {
namespace {

const OrtApi* g_api;
OrtEnv* g_env;
std::string g_version;

bool Check(OrtStatus* s, std::string& err, const char* what) {
    if (!s) return true;
    err = std::string(what) + ": " + g_api->GetErrorMessage(s);
    g_api->ReleaseStatus(s);
    return false;
}

struct OrtNet : Net {
    OrtSession* sess = nullptr; OrtMemoryInfo* mem = nullptr; int batch;
    explicit OrtNet(int b) : batch(b) {}
    ~OrtNet() override {
        if (sess) g_api->ReleaseSession(sess);
        if (mem) g_api->ReleaseMemoryInfo(mem);
    }
    int Batch() const override { return batch; }
    bool Run(const float* x, const float* emo, int b, float* out, std::string& err) override {
        int64_t sx[3] = {b, 1, WIN}, se[3] = {b, 1, NEMO};
        OrtValue *vx = nullptr, *ve = nullptr, *vy = nullptr;
        bool ok = Check(g_api->CreateTensorWithDataAsOrtValue(mem, (void*)x, (size_t)b * WIN * 4, sx, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &vx), err, "input") &&
                  Check(g_api->CreateTensorWithDataAsOrtValue(mem, (void*)emo, (size_t)b * NEMO * 4, se, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &ve), err, "emotion");
        if (ok) {
            const char* in[] = {"input", "emotion"}; const char* outn[] = {"result"};
            const OrtValue* ins[] = {vx, ve};
            ok = Check(g_api->Run(sess, nullptr, in, ins, 2, outn, 1, &vy), err, "run");
        }
        if (ok) {
            float* p = nullptr;
            ok = Check(g_api->GetTensorMutableData(vy, (void**)&p), err, "output");
            if (ok) memcpy(out, p, (size_t)b * NOUT * 4);
        }
        if (vy) g_api->ReleaseValue(vy);
        if (vx) g_api->ReleaseValue(vx);
        if (ve) g_api->ReleaseValue(ve);
        return ok;
    }
};

}  // namespace

std::unique_ptr<Net> CreateOrtNet(const std::wstring& dllDir, const std::wstring& model, const OrtOptions& o, std::string& err, std::string& info) {
    if (!g_api) {
        // full path + altered search path: DirectML.dll / onnxruntime_providers_shared.dll resolve from the same folder
        HMODULE h = LoadLibraryExW((dllDir + L"\\onnxruntime.dll").c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
        if (!h) { err = "cannot load onnxruntime.dll (error " + std::to_string(GetLastError()) + ")"; return nullptr; }
        auto base = (const OrtApiBase* (*)())GetProcAddress(h, "OrtGetApiBase");
        if (!base || !(g_api = base()->GetApi(ORT_API_VERSION))) { err = "onnxruntime.dll: wrong version"; g_api = nullptr; return nullptr; }
        if (!Check(g_api->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "A2FLive", &g_env), err, "env")) return nullptr;
        g_version = base()->GetVersionString();
    }
    auto net = std::make_unique<OrtNet>(std::max(1, o.batch));
    OrtSessionOptions* so = nullptr;
    bool ok = Check(g_api->CreateSessionOptions(&so), err, "options");
    if (ok && o.gpu) {
        const OrtDmlApi* dml = nullptr;
        ok = Check(g_api->DisableMemPattern(so), err, "mem pattern") && Check(g_api->SetSessionExecutionMode(so, ORT_SEQUENTIAL), err, "mode") &&
             Check(g_api->GetExecutionProviderApi("DML", ORT_API_VERSION, (const void**)&dml), err, "DML api") &&
             Check(dml->SessionOptionsAppendExecutionProvider_DML(so, 0), err, "DirectML");
        // DML's arena rounds allocations up to powers of two: off saves ~16-32 MB at small batches
        if (ok) ok = Check(g_api->AddSessionConfigEntry(so, "ep.dml.disable_memory_arena", "1"), err, "arena");
        // the batch is always the same: let DirectML plan one static shape (also 4x faster at batch 1-2)
        for (const char* dim : {"input_dynamic_axes_1", "emotion_dynamic_axes_1"})
            if (ok) ok = Check(g_api->AddFreeDimensionOverrideByName(so, dim, net->batch), err, "batch override");
    } else if (ok) {
        ok = Check(g_api->SetIntraOpNumThreads(so, 2), err, "threads") && Check(g_api->SetInterOpNumThreads(so, 1), err, "threads");
    }
    if (ok) ok = Check(g_api->CreateSession(g_env, model.c_str(), so, &net->sess), err, "session");
    if (ok) ok = Check(g_api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &net->mem), err, "memory info");
    if (so) g_api->ReleaseSessionOptions(so);
    if (!ok) return nullptr;
    info = "onnxruntime " + g_version + (o.gpu ? ", DirectML device 0" : ", CPU 2 threads") + ", batch " + std::to_string(net->batch);
    return net;
}

}  // namespace a2f
