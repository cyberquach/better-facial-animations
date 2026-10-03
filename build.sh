#!/bin/sh
# Build BetterFacialAnimations.dll with mingw-w64 (static FFmpeg + libstdc++) and assemble the mod in build/mod.
#   ./build.sh                    build + assemble
#   ./build.sh install <dir>      also copy into <dir> (e.g. an MO2 mod folder)
#   ./build.sh release            also write release/Better Facial Animations-<ver>.7z and the source zip
#   ./build.sh test               also check every supported runtime (tests/runtimes_test.cc) and the
#                                 Direct mesh placement (tests/direct_test.cc), under $WINE
# Dependency locations (see README) can be overridden with these variables:
cd "$(dirname "$0")" || exit 1
V=$(cat VERSION) || exit 1
FFMPEG=${FFMPEG:-deps/ffmpeg/win64}
ORT=${ORT:-deps/dml/microsoft.ml.onnxruntime.directml}
DML=${DML:-deps/dml/microsoft.ai.directml}
MODEL=${MODEL:-models/mark/network_fp16.onnx}
ADDRLIB=${ADDRLIB:-deps/address-library}  # test only

mkdir -p build
CXX="x86_64-w64-mingw32-g++ -std=c++17 -O2"
$CXX -Wall -DBFA_VERSION=\"$V\" -c src/better_facial_animations.cc -o build/better_facial_animations.o &&
$CXX -Wall -c src/skse.cc -o build/skse.o &&
$CXX -Wall -c src/direct.cc -o build/direct.o &&
$CXX -w -I$FFMPEG/include -c src/live_core.cc -o build/live_core.o &&
$CXX -w -I$ORT/build/native/include -c src/ort_net.cc -o build/ort_net.o &&
$CXX -shared -s build/better_facial_animations.o build/skse.o build/direct.o build/live_core.o build/ort_net.o -o build/BetterFacialAnimations.dll \
    -L$FFMPEG/lib -lavformat -lavcodec -lswresample -lavutil -lbcrypt -lshell32 -lole32 -luuid -static -static-libgcc -static-libstdc++ || exit 1

PKG=build/mod; P=$PKG/SKSE/Plugins; L=$P/BetterFacialAnimations/licenses
rm -rf $PKG; mkdir -p $L
cp build/BetterFacialAnimations.dll dist/BetterFacialAnimations.ini $P/
cp $ORT/runtimes/win-x64/native/onnxruntime.dll $ORT/runtimes/win-x64/native/onnxruntime_providers_shared.dll \
   $DML/bin/x64-win/DirectML.dll $MODEL dist/A2FDirect.bin dist/NOTICE.txt $P/BetterFacialAnimations/ || exit 1
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
    rm -f "$R/Better Facial Animations-$V.7z" "$R/Better Facial Animations - Source-$V.zip"
    (cd $PKG && 7z a -t7z -mx=9 "$R/Better Facial Animations-$V.7z" SKSE >/dev/null) &&
    # source only: no binaries (A2FDirect.bin ships in the mod archive), no commit id comment
    git archive --prefix="BetterFacialAnimations-$V-src/" -o "$R/Better Facial Animations - Source-$V.zip" "HEAD^{tree}" -- . ":!dist/A2FDirect.bin" &&
    ls -la $R ;;
test)
    T=build/test; mkdir -p $T/Data && ln -sfn "$(realpath "$ADDRLIB")/SKSE" $T/Data/SKSE &&
    $CXX -Wall -static src/skse.cc tests/runtimes_test.cc -o $T/runtimes_test.exe -lshell32 -lole32 -luuid &&
    $CXX -Wall -static src/direct.cc tests/direct_test.cc -o $T/direct_test.exe &&
    WINEDEBUG=-all ${WINE-wine64} $T/runtimes_test.exe build/BetterFacialAnimations.dll $T &&
    WINEDEBUG=-all ${WINE-wine64} $T/direct_test.exe ;;
esac
