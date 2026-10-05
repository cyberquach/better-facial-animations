#!/bin/sh
# Build BetterFacialAnimations.dll with mingw-w64 (static FFmpeg + libstdc++) and assemble the mod in build/mod.
#   ./build.sh                    build + assemble
#   ./build.sh install <dir>      also copy into <dir> (e.g. an MO2 mod folder)
#   ./build.sh release            also write release/Better Facial Animations-<ver>.7z (on a feature branch
#                                 <ver>.<n>, n counting up from the archives already there)
#   ./build.sh test               also check every supported runtime (tests/runtimes_test.cc) and the
#                                 Direct mesh placement, voice-start gating, losing and regaining the GPU
#                                 session (tests/net_keeper_test.cc) and the CPU session
#                                 (tests/ort_net_test.cc: the packaged networks load and run) and the
#                                 line's emotion and call sites shared with another plugin, under $WINE
# Dependency locations (see README) can be overridden with these variables:
cd "$(dirname "$0")" || exit 1
V=$(cat VERSION) || exit 1
FFMPEG=${FFMPEG:-deps/ffmpeg/win64}
ORT=${ORT:-deps/dml/microsoft.ml.onnxruntime.directml}
DML=${DML:-deps/dml/microsoft.ai.directml}
MODEL=${MODEL:-models/mark/network_fp16.onnx}  # Audio2Face-3D v2.3 Mark
MODEL_FP32=${MODEL_FP32:-$(dirname "$MODEL")/network.onnx}  # test only: reference for the FP16 network
MODEL3=${MODEL3:-models/v3/network_fp16.onnx}  # v3.0, folded (tools/a2f3_fold.py)
MODEL3_FP32=${MODEL3_FP32:-$(dirname "$MODEL3")/network_folded.onnx}  # test only
ADDRLIB=${ADDRLIB:-deps/address-library}  # test only

mkdir -p build
CXX="x86_64-w64-mingw32-g++ -std=c++17 -O2"
$CXX -Wall -DBFA_VERSION=\"$V\" -c src/better_facial_animations.cc -o build/better_facial_animations.o &&
$CXX -Wall -c src/player_speech.cc -o build/player_speech.o &&
$CXX -Wall -c src/pose.cc -o build/pose.o &&
$CXX -Wall -c src/net_keeper.cc -o build/net_keeper.o &&
$CXX -Wall -c src/settings.cc -o build/settings.o &&
$CXX -Wall -c src/head_cache.cc -o build/head_cache.o &&
$CXX -Wall -c src/menu.cc -o build/menu.o &&
$CXX -Wall -c src/skse.cc -o build/skse.o &&
$CXX -Wall -c src/direct.cc -o build/direct.o &&
$CXX -Wall -isystem "$FFMPEG/include" -c src/live_core.cc -o build/live_core.o &&
$CXX -Wall -isystem "$ORT/build/native/include" -c src/ort_net.cc -o build/ort_net.o &&
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
install)
    [ -n "$2" ] || { echo "usage: $0 install <dir>"; exit 1; }
    mkdir -p "$2" && cp -r $PKG/. "$2/" && echo "installed to $2" ;;
release)
    R=$PWD/release; mkdir -p $R
    # On a feature branch the archive is a numbered build, <version>.<n>: one more than the highest in release/.
    case "$(git rev-parse --abbrev-ref HEAD 2>/dev/null)" in
    main|HEAD|"") ;;
    *)  N=0
        for f in "$R/Better Facial Animations-$V."*.7z; do
            n=${f##*-"$V".}; n=${n%.7z}
            case "$n" in ""|*[!0-9]*) ;; *) [ "$n" -gt "$N" ] && N=$n ;; esac
        done
        V=$V.$((N + 1)) ;;
    esac
    rm -f "$R/Better Facial Animations-$V.7z"
    (cd $PKG && 7z a -t7z -mx=9 "$R/Better Facial Animations-$V.7z" SKSE >/dev/null) &&
    ls -la $R ;;
test)
    T=build/test; mkdir -p $T/Data && ln -sfn "$(realpath "$ADDRLIB")/SKSE" $T/Data/SKSE &&
    $CXX -Wall -static src/skse.cc tests/runtimes_test.cc -o $T/runtimes_test.exe -lshell32 -lole32 -luuid &&
    $CXX -Wall -static src/direct.cc tests/direct_test.cc -o $T/direct_test.exe &&
    $CXX -Wall -static tests/emotion_test.cc build/live_core.o -o $T/emotion_test.exe \
        -L$FFMPEG/lib -lavformat -lavcodec -lswresample -lavutil -lbcrypt -lole32 &&
    WINEDEBUG=-all ${WINE-wine64} $T/emotion_test.exe &&
    $CXX -Wall -static src/player_speech.cc tests/player_speech_test.cc -o $T/player_speech_test.exe &&
    WINEDEBUG=-all ${WINE-wine64} $T/player_speech_test.exe &&
    $CXX -Wall -static tests/voice_sync_test.cc -o $T/voice_sync_test.exe &&
    $CXX -Wall -static tests/call_sites_test.cc -o $T/call_sites_test.exe &&
    $CXX -Wall -static src/net_keeper.cc tests/net_keeper_test.cc -o $T/net_keeper_test.exe &&
    $CXX -Wall -static tests/ort_net_test.cc build/ort_net.o -o $T/ort_net_test.exe -ldxgi &&
    WINEDEBUG=-all ${WINE-wine64} $T/runtimes_test.exe build/BetterFacialAnimations.dll $T &&
    WINEDEBUG=-all ${WINE-wine64} $T/direct_test.exe &&
    WINEDEBUG=-all ${WINE-wine64} $T/voice_sync_test.exe &&
    WINEDEBUG=-all ${WINE-wine64} $T/call_sites_test.exe &&
    WINEDEBUG=-all ${WINE-wine64} $T/net_keeper_test.exe &&
    WINEDEBUG=-all ${WINE-wine64} $T/ort_net_test.exe $P/BetterFacialAnimations "$P/BetterFacialAnimations/$(basename "$MODEL")" "$MODEL_FP32" &&
    WINEDEBUG=-all ${WINE-wine64} $T/ort_net_test.exe $P/BetterFacialAnimations $P/BetterFacialAnimations/a2f3_fp16.onnx "$MODEL3_FP32" 3 &&
    $CXX -Wall -static src/direct.cc tests/direct_bins_test.cc -o $T/direct_bins_test.exe &&
    WINEDEBUG=-all ${WINE-wine64} $T/direct_bins_test.exe dist/A2FDirect.bin dist/A2FDirect3.bin ;;
esac
