#!/usr/bin/env python3
"""Build the CommonLib-based Windows plugin with MSVC (native or under Wine)."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = Path(os.environ.get('COMMONLIB_DEPS', ROOT / 'deps/commonlib')).resolve()
BUNDLE, SUPPORT, FMT = (BASE / name for name in ('prebuilt', 'support', 'fmt'))
BUILD = ROOT / 'build'
SOURCES = sorted((ROOT / 'src').rglob('*.cc'))
FFMPEG = Path(os.environ.get('FFMPEG', ROOT / 'deps/ffmpeg/win64')).resolve()
ORT = Path(os.environ.get('ORT', ROOT / 'deps/dml/microsoft.ml.onnxruntime.directml')).resolve()
LIBRARIES = ['CommonLibSSE.lib', 'openvr_api.lib', 'spdlog.lib', 'DirectXTK.lib',
             'advapi32.lib', 'bcrypt.lib', 'D3D11.lib', 'd3dcompiler.lib', 'dbghelp.lib',
             'DXGI.lib', 'delayimp.lib', 'ole32.lib', 'version.lib', 'user32.lib', 'shell32.lib', 'uuid.lib']


def win(path):
    return 'Z:' + str(path).replace('/', '\\')


def compiler(tool, args):
    """Only C/POD crosses into the MinGW-built FFmpeg DLL; the plugin is MSVC."""
    env = dict(os.environ)
    if os.name == 'nt':
        command = [tool, *args]
    else:
        root = Path(os.environ.get('BFA_MSVC_ROOT', BASE / 'msvc')).resolve()
        vc = root / 'VC/Tools/MSVC/14.51.36231'
        sdk = root / 'Windows Kits/10'
        sdk_version = '10.0.26100.0'
        env.update(WINEPREFIX=str(Path(os.environ.get('BFA_BUILD_WINEPREFIX', BUILD / 'msvc-wine')).resolve()),
                   WINEDEBUG='-all', WINEDLLOVERRIDES='vcruntime140=n;vcruntime140_1=n')
        env['WINEPATH'] = win(vc / 'bin/Hostx64/x64')
        env['INCLUDE'] = ';'.join(win(p) for p in [vc / 'include', *[
            sdk / 'Include' / sdk_version / name for name in ('shared', 'ucrt', 'um', 'winrt')]])
        env['LIB'] = ';'.join(win(p) for p in [vc / 'lib/x64', *[
            sdk / 'Lib' / sdk_version / name / 'x64' for name in ('ucrt', 'um')]])
        converted = []
        for arg in map(str, args):
            if arg.startswith('/') and Path(arg).exists():
                arg = win(arg)
            for prefix in ('/I', '/Fo', '/Fe', '/Fp', '/FI', '/OUT:', '/DEF:', '/LIBPATH:', '/MAP:'):
                if arg.startswith(prefix + '/'):
                    arg = prefix + win(arg[len(prefix):])
                    break
            converted.append(arg)
        command = ['wine64', str(vc / 'bin/Hostx64/x64' / (tool + '.exe')), *converted]
    subprocess.run(command, cwd=ROOT, env=env, check=True, timeout=600)


def flags(test=False):
    version = os.environ.get('BFA_BUILD_VERSION', (ROOT / 'VERSION').read_text().strip())
    args = ['/nologo', '/EHsc', '/std:c++latest', '/MD', '/O2', '/utf-8', '/Zc:preprocessor',
            '/DNOMINMAX', '/DWIN32_LEAN_AND_MEAN', '/DSPDLOG_COMPILED_LIB',
            '/DSPDLOG_FMT_EXTERNAL', '/DENABLE_SKYRIM_SE=1', '/DENABLE_SKYRIM_AE=1',
            '/DENABLE_SKYRIM_VR=1', '/DHAS_SKYRIM_MULTI_TARGETING=1',
            '/DWINVER=0x0601', '/D_WIN32_WINNT=0x0601', '/DBFA_VERSION="' + version + '"',
            '/FIRE/Skyrim.h', '/wd4005']
    args += ['/UNDEBUG', '/DENABLE_COMMONLIBSSE_TESTING'] if test else ['/DNDEBUG']
    return args + ['/I' + str(p) for p in [FMT / 'include', BUNDLE / 'include',
        BUNDLE / 'extern/openvr/headers', SUPPORT / 'include', SUPPORT / 'include/directxtk',
        ROOT / 'include', FFMPEG / 'include', ORT / 'build/native/include']]


def link_args():
    return ['/LIBPATH:' + str(p) for p in [BUNDLE / 'lib', SUPPORT / 'lib', BUILD]] + LIBRARIES + [
        str(BUILD / 'format.obj'), str(BUILD / 'os.obj'), 'BFAFFmpeg.lib']


def compile_source(source, output, test=False):
    compiler('cl', flags(test) + ['/c', '/Fo' + str(output), str(source)])


def runtime_files(destination):
    """Tests use explicit local CRTs, never alter the user's Wine/game prefix."""
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(BUILD / 'BFAFFmpeg.dll', destination)
    if os.name != 'nt':
        vc = Path(os.environ.get('BFA_MSVC_ROOT', BASE / 'msvc')).resolve() / 'VC'
        crt = next((vc / 'Redist/MSVC/14.51.36231/x64').glob('Microsoft.VC*.CRT'))
        for dll in crt.glob('*.dll'):
            shutil.copy2(dll, destination)


def build():
    BUILD.mkdir(exist_ok=True)
    digest = hashlib.sha256((BUNDLE / 'lib/CommonLibSSE.lib').read_bytes()).hexdigest()
    if digest != '80ffa2a6f7d40483b4d9f92515a091bb939ddc07da202f5fc087a38b88e0632a':
        raise ValueError('CommonLib must be the verified v8.4.0 all-runtime MSVC bundle')
    symbols = sorted(set(re.findall(r'\b((?:av[a-z0-9_]+|swr_[a-z0-9_]+))\s*\(',
                                    (ROOT / 'src/inference/live_core.cc').read_text())))
    definition = BUILD / 'BFAFFmpeg.def'
    definition.write_text('LIBRARY BFAFFmpeg\nEXPORTS\n' + '\n'.join(symbols) + '\n')
    subprocess.run([os.environ.get('CC', 'x86_64-w64-mingw32-gcc'), '-shared', '-static-libgcc',
        str(definition), '-Wl,--start-group', *[str(FFMPEG / 'lib' / ('lib' + name + '.a'))
        for name in ('avformat', 'avcodec', 'swresample', 'avutil')], '-Wl,--end-group',
        '-lbcrypt', '-lole32', '-o', str(BUILD / 'BFAFFmpeg.dll')], check=True, timeout=120)
    compiler('lib', ['/nologo', '/DEF:' + str(definition), '/MACHINE:X64', '/OUT:' + str(BUILD / 'BFAFFmpeg.lib')])
    sources = SOURCES + [FMT / 'src/format.cc', FMT / 'src/os.cc']
    with concurrent.futures.ThreadPoolExecutor(int(os.environ.get('JOBS', 4))) as pool:
        list(pool.map(lambda p: compile_source(p, BUILD / (p.stem + '.obj')), sources))
    compiler('link', ['/NOLOGO', '/DLL', '/DELAYLOAD:BFAFFmpeg.dll', '/OUT:' + str(BUILD / 'BetterFacialAnimations.dll'),
        *[str(BUILD / (p.stem + '.obj')) for p in SOURCES], *link_args()])
    test_objects = BUILD / 'test-objects'
    test_objects.mkdir(exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(int(os.environ.get('JOBS', 4))) as pool:
        list(pool.map(lambda p: compile_source(p, test_objects / (p.stem + '.obj'), True), SOURCES))
    provenance = {'commonlib_revision': 'adb3e2c4dff61d151370a6777ed9e77f8e005c62',
        'commonlib_library_sha256': digest, 'objects': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in BUILD.glob('*.obj')}, 'dll_sha256': hashlib.sha256((BUILD / 'BetterFacialAnimations.dll').read_bytes()).hexdigest()}
    (BUILD / 'commonlib-build.json').write_text(json.dumps(provenance, indent=2) + '\n')


def build_case(case, executable):
    executable = Path(executable).resolve()
    objdir = executable.parent / (case['test'] + '-objects')
    objdir.mkdir(exist_ok=True)
    objects = []
    for source in case['sources']:
        obj = objdir / (Path(source).stem + '.obj')
        compile_source(ROOT / source, obj, True)
        objects.append(obj)
    if case.get('plugin_objects'):
        objects += [BUILD / 'test-objects' / (p.stem + '.obj') for p in SOURCES
                    if p.name != 'better_facial_animations.cc']
    objects += [BUILD / 'test-objects' / (name + '.obj') for name in case.get('objects', [])]
    compiler('link', ['/NOLOGO', '/MAP:' + str(executable.with_suffix('.map')), '/OUT:' + str(executable), *map(str, objects), *link_args()])


def extra_tests():
    folder = BUILD / 'test'
    runtime_files(folder)
    fixtures = folder / 'Data'
    fixtures.mkdir(exist_ok=True)
    skse = fixtures / 'SKSE'
    if not skse.exists():
        skse.symlink_to(Path(os.environ.get('ADDRLIB', ROOT / 'deps/address-library')).resolve() / 'SKSE', target_is_directory=True)
    asset = 'build/mod/SKSE/Plugins/BetterFacialAnimations'
    tests = [
        ('runtimes_test', ['tests/runtimes_test.cc', 'src/runtime/skse.cc'], [],
         [['build/BetterFacialAnimations.dll', 'build/test']]),
        ('emotion_test', ['tests/emotion_test.cc'], ['live_core', 'performance_features'], [[]]),
        ('direct_bins_test', ['tests/direct_bins_test.cc', 'src/animation/direct.cc'], [],
         [['dist/A2FDirect.bin', 'dist/A2FDirect3.bin']]),
        ('sentiment_model_test', ['tests/sentiment_model_test.cc'], ['ort_net', 'performance_features'],
         [[asset, os.environ.get('EMOTION_MODEL', 'models/a2e/network_fp16.onnx')]]),
        ('ort_net_test', ['tests/ort_net_test.cc'], ['ort_net', 'performance_features'],
         [[asset, asset + '/network_fp16.onnx', os.environ.get('MODEL_FP32', 'models/mark/network.onnx')],
          [asset, asset + '/a2f3_fp16.onnx', os.environ.get('MODEL3_FP32', 'models/v3/network_folded.onnx'), '3']])]
    for name, sources, objects, runs in tests:
        exe = folder / (name + '.exe')
        build_case({'test': name, 'sources': sources, 'objects': objects}, exe)
        for args in runs:
            subprocess.run([os.environ.get('WINE', 'wine64'), str(exe), *args], cwd=ROOT,
                           env={**os.environ, 'WINEDEBUG': '-all'}, check=True, timeout=600)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--compiler':
        compiler(sys.argv[2], sys.argv[3:])
    elif len(sys.argv) > 1 and sys.argv[1] == '--case':
        build_case(json.loads(sys.argv[2]), sys.argv[3])
    elif len(sys.argv) > 1 and sys.argv[1] == '--extra-tests':
        extra_tests()
    else:
        build()
