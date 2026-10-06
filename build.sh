#!/bin/sh
# Build BetterFacialAnimations.dll with mingw-w64 (static FFmpeg + libstdc++) and assemble the mod in build/mod.
#   ./build.sh                    build + assemble
#   ./build.sh install <dir>      also copy into <dir> (e.g. an MO2 mod folder)
#   ./build.sh release            also write release/Better Facial Animations-<ver>.<n>.7z
#                                 (feature branches append -<branch>; n counts existing archives)
#   ./build.sh compat             also run the compatibility contracts in tests/compatibility.json
#                                 (no Skyrim launch) and write their evidence to build/compatibility/
#   ./build.sh test               also run the contracts, check every supported runtime (tests/runtimes_test.cc) and the
#                                 Direct mesh placement, voice-start gating, losing and regaining the GPU
#                                 session (tests/net_keeper_test.cc) and the CPU session
#                                 (tests/ort_net_test.cc: the packaged networks load and run) and the
#                                 line's emotion and call sites shared with another plugin, under $WINE
# Dependency locations (see README) can be overridden with these variables:
cd "$(dirname "$0")" || exit 1
V=$(cat VERSION) || exit 1
ARCHIVE_SUFFIX=
# Select the release version before compiling so DLL, readme and archive agree.
if [ "$1" = release ]; then
    R=$PWD/release
    # Every release uses one more than the highest build for this version.
    N=0
    for f in "$R/Better Facial Animations-$V."*.7z; do
        n=${f##*/}; n=${n#"Better Facial Animations-$V."}; n=${n%.7z}; n=${n%%-*}
        case "$n" in ""|*[!0-9]*) ;; *) [ "$n" -gt "$N" ] && N=$n ;; esac
    done
    V=$V.$((N + 1))
    BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)
    case "$BRANCH" in
    main|HEAD|"") ;;
    *)  # Keep the numeric DLL version separate from the filename's branch label.
        ARCHIVE_SUFFIX=-$(printf '%s' "$BRANCH" | LC_ALL=C sed 's/[^A-Za-z0-9._-]/-/g') ;;
    esac
fi
FFMPEG=${FFMPEG:-deps/ffmpeg/win64}
ORT=${ORT:-deps/dml/microsoft.ml.onnxruntime.directml}
DML=${DML:-deps/dml/microsoft.ai.directml}
MODEL=${MODEL:-models/mark/network_fp16.onnx}  # Audio2Face-3D v2.3 Mark
MODEL_FP32=${MODEL_FP32:-$(dirname "$MODEL")/network.onnx}  # test only: reference for the FP16 network
MODEL3=${MODEL3:-models/v3/network_fp16.onnx}  # v3.0, folded (tools/a2f3_fold.py)
MODEL3_FP32=${MODEL3_FP32:-$(dirname "$MODEL3")/network_folded.onnx}  # test only
ADDRLIB=${ADDRLIB:-deps/address-library}  # test only
JOBS=${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)}
case "$JOBS" in ""|*[!0-9]*|0*) echo "JOBS must be a positive integer" >&2; exit 1 ;; esac

mkdir -p build
CXX="x86_64-w64-mingw32-g++ -std=c++17 -O2"
# a test that includes the plugin's main source links its other objects (also read by tools/compatibility.py)
PLUGIN_OBJS="build/player_speech.o build/settings.o build/pose.o build/net_keeper.o build/head_cache.o build/menu.o build/skse.o build/direct.o build/live_core.o build/ort_net.o"
PLUGIN_LIBS="-L$FFMPEG/lib -lavformat -lavcodec -lswresample -lavutil -lbcrypt -lshell32 -lole32 -luuid -ldxgi"
make -f tools/build.mk -B -j "$JOBS" CXX="$CXX" BFA_VERSION="$V" \
    FFMPEG="$FFMPEG" ORT="$ORT" build/better_facial_animations.o $PLUGIN_OBJS &&
$CXX -shared -s build/better_facial_animations.o build/player_speech.o build/settings.o build/pose.o build/net_keeper.o build/head_cache.o build/menu.o build/skse.o build/direct.o build/live_core.o build/ort_net.o -o build/BetterFacialAnimations.dll \
    -L$FFMPEG/lib -lavformat -lavcodec -lswresample -lavutil -lbcrypt -lshell32 -lole32 -luuid -ldxgi -static -static-libgcc -static-libstdc++ || exit 1

PKG=build/mod; P=$PKG/SKSE/Plugins; L=$P/BetterFacialAnimations/licenses
rm -rf $PKG; mkdir -p $L
cp build/BetterFacialAnimations.dll dist/BetterFacialAnimations.ini $P/
cp $ORT/runtimes/win-x64/native/onnxruntime.dll $ORT/runtimes/win-x64/native/onnxruntime_providers_shared.dll \
   $DML/bin/x64-win/DirectML.dll $MODEL dist/A2FDirect.bin dist/A2FDirect3.bin dist/NOTICE.txt $P/BetterFacialAnimations/ || exit 1
cp $MODEL3 $P/BetterFacialAnimations/a2f3_fp16.onnx || exit 1
sed "s/@VERSION@/$V/" dist/README.txt > $P/BetterFacialAnimations/README.txt
cp LICENSE $L/BetterFacialAnimations-GPL-3.0.txt
cp $FFMPEG/COPYING.LGPLv2.1 $L/FFmpeg-LGPL-2.1.txt
cp $ORT/LICENSE $L/onnxruntime-LICENSE.txt
cp $ORT/ThirdPartyNotices.txt $L/onnxruntime-ThirdPartyNotices.txt
cp $DML/LICENSE.txt $L/DirectML-LICENSE.txt
cp dist/licenses/* $L/
du -sh $PKG

case "$1" in
compat)
    python3 tools/compatibility.py --tier contracts ;;
install)
    [ -n "$2" ] || { echo "usage: $0 install <dir>"; exit 1; }
    mkdir -p "$2" && cp -r $PKG/. "$2/" && echo "installed to $2" ;;
release)
    R=$PWD/release; mkdir -p $R
    rm -f "$R/Better Facial Animations-$V$ARCHIVE_SUFFIX.7z"
    (cd $PKG && 7z a -t7z -mx=9 "$R/Better Facial Animations-$V$ARCHIVE_SUFFIX.7z" SKSE >/dev/null) &&
    ls -la $R ;;
test)
    T=build/test; mkdir -p $T/Data && ln -sfn "$(realpath "$ADDRLIB")/SKSE" $T/Data/SKSE &&
    python3 tools/compatibility.py --tier contracts &&
    $CXX -Wall -static src/skse.cc tests/runtimes_test.cc -o $T/runtimes_test.exe -lshell32 -lole32 -luuid &&
    $CXX -Wall -static tests/emotion_test.cc build/live_core.o -o $T/emotion_test.exe \
        -L$FFMPEG/lib -lavformat -lavcodec -lswresample -lavutil -lbcrypt -lole32 &&
    WINEDEBUG=-all ${WINE-wine64} $T/emotion_test.exe &&
    $CXX -Wall -static tests/reference_test.cc -o $T/reference_test.exe &&
    WINEDEBUG=-all ${WINE-wine64} $T/reference_test.exe &&
    $CXX -Wall -static tests/passive_report_test.cc build/ort_net.o -o $T/passive_report_test.exe -Wl,--wrap=__imp_CreateDXGIFactory1 -ldxgi &&
    WINEDEBUG=-all ${WINE-wine64} $T/passive_report_test.exe &&
    $CXX -Wall -static src/net_keeper.cc tests/net_keeper_test.cc -o $T/net_keeper_test.exe &&
    $CXX -Wall -static tests/ort_net_test.cc build/ort_net.o -o $T/ort_net_test.exe -ldxgi &&
    WINEDEBUG=-all ${WINE-wine64} $T/runtimes_test.exe build/BetterFacialAnimations.dll $T &&
    WINEDEBUG=-all ${WINE-wine64} $T/net_keeper_test.exe &&
    WINEDEBUG=-all ${WINE-wine64} $T/ort_net_test.exe $P/BetterFacialAnimations "$P/BetterFacialAnimations/$(basename "$MODEL")" "$MODEL_FP32" &&
    WINEDEBUG=-all ${WINE-wine64} $T/ort_net_test.exe $P/BetterFacialAnimations $P/BetterFacialAnimations/a2f3_fp16.onnx "$MODEL3_FP32" 3 &&
    $CXX -Wall -static src/direct.cc tests/direct_bins_test.cc -o $T/direct_bins_test.exe &&
    WINEDEBUG=-all ${WINE-wine64} $T/direct_bins_test.exe dist/A2FDirect.bin dist/A2FDirect3.bin ;;
esac
