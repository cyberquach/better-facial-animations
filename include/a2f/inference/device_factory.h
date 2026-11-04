// Minimal SDK ABI declarations for MinGW headers predating independent devices.
// Method order and GUIDs: Microsoft DirectX-Headers, include/directx/d3d12.idl.
// https://github.com/microsoft/DirectX-Headers/blob/main/include/directx/d3d12.idl
#ifndef INCLUDE_A2F_INFERENCE_DEVICE_FACTORY_H_
#define INCLUDE_A2F_INFERENCE_DEVICE_FACTORY_H_

#include <d3d12.h>

namespace a2f::independent {
inline constexpr GUID kFactoryClass = {
    0x114863bf,
    0xc386,
    0x4aee,
    {0xb3, 0x9d, 0x8f, 0x0b, 0xbb, 0x06, 0x29, 0x55}};
inline constexpr GUID kFactoryInterface = {
    0x61f307d3,
    0xd34e,
    0x4e7c,
    {0x83, 0x74, 0x3b, 0xa4, 0xde, 0x23, 0xcc, 0xcb}};
inline constexpr GUID kDmlDevice = {
    0x6dbd6437,
    0x96fd,
    0x423f,
    {0xa9, 0x8c, 0xae, 0x5e, 0x7c, 0x2a, 0x57, 0x3f}};
// No ALLOW_RETURNING_EXISTING_DEVICE, and never install a new singleton.
inline constexpr UINT kStrictFlags = 0x4;
struct Factory : IUnknown {
  virtual HRESULT STDMETHODCALLTYPE InitializeFromGlobalState() = 0;
  virtual HRESULT STDMETHODCALLTYPE ApplyToGlobalState() = 0;
  virtual HRESULT STDMETHODCALLTYPE SetFlags(UINT flags) = 0;
  virtual UINT STDMETHODCALLTYPE GetFlags() = 0;
  virtual HRESULT STDMETHODCALLTYPE GetConfigurationInterface(REFCLSID clsid,
                                                              REFIID iid,
                                                              void** out) = 0;
  virtual HRESULT STDMETHODCALLTYPE EnableExperimentalFeatures(UINT count,
                                                               const IID* ids,
                                                               void* configs,
                                                               UINT* sizes) = 0;
  virtual HRESULT STDMETHODCALLTYPE CreateDevice(IUnknown* adapter,
                                                 D3D_FEATURE_LEVEL level,
                                                 REFIID iid, void** out) = 0;
};
// CLSID_D3D12SDKConfiguration / ID3D12SDKConfiguration1: a factory over the
// bundled Microsoft.Direct3D.D3D12 1.619.5 runtime (BetterFacialAnimations/
// Agility) when the default runtime has none.
inline constexpr UINT kAgilityVersion = 619;
inline constexpr GUID kConfigurationClass = {
    0x7cda6aca,
    0xa03e,
    0x49c8,
    {0x94, 0x58, 0x03, 0x34, 0xd2, 0x0e, 0x07, 0xce}};
inline constexpr GUID kConfigurationInterface = {
    0x8aaf9303,
    0xad25,
    0x48b9,
    {0x9a, 0x57, 0xd9, 0xc3, 0x7e, 0x00, 0x9d, 0x9f}};
struct Configuration : IUnknown {
  virtual HRESULT STDMETHODCALLTYPE SetSDKVersion(UINT version,
                                                  LPCSTR path) = 0;
  virtual HRESULT STDMETHODCALLTYPE CreateDeviceFactory(UINT version,
                                                        LPCSTR path, REFIID iid,
                                                        void** out) = 0;
  virtual void STDMETHODCALLTYPE FreeUnusedSDKs() = 0;
};
using GetInterface = HRESULT(WINAPI*)(REFCLSID, REFIID, void**);
using CreateDml = HRESULT(WINAPI*)(ID3D12Device*, UINT, REFIID, void**);
}  // namespace a2f::independent
#endif
