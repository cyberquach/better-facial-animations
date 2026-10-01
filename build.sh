#!/bin/sh
# Build A2FLive.dll with mingw-w64 (static FFmpeg + libstdc++) and assemble the mod in build/mod.
#   ./build.sh                    build + assemble
#   ./build.sh install <dir>      also copy into <dir> (e.g. an MO2 mod folder)
#   ./build.sh release            also write release/A2F Live-<ver>.7z and the source zip
# Dependency locations (see README) can be overridden with these variables:
cd "$(dirname "$0")" || exit 1
FFMPEG=${FFMPEG:-deps/ffmpeg/win64}
ORT=${ORT:-deps/dml/microsoft.ml.onnxruntime.directml}
DML=${DML:-deps/dml/microsoft.ai.directml}
MODEL=${MODEL:-models/mark/network_fp16.onnx}

mkdir -p build
CXX="x86_64-w64-mingw32-g++ -std=c++17 -O2"
$CXX -Wall -Wno-unknown-pragmas -I$FFMPEG/include -c src/A2FLive.cpp -o build/A2FLive.o &&
$CXX -w -I$FFMPEG/include -c src/live_core.cpp -o build/live_core.o &&
$CXX -w -I$ORT/build/native/include -c src/ort_net.cpp -o build/ort_net.o &&
$CXX -shared -s build/A2FLive.o build/live_core.o build/ort_net.o -o build/A2FLive.dll \
    -L$FFMPEG/lib -lavformat -lavcodec -lswresample -lavutil -lbcrypt -lshell32 -lole32 -luuid -static -static-libgcc -static-libstdc++ || exit 1

PKG=build/mod; P=$PKG/SKSE/Plugins; L=$P/A2FLive/licenses
rm -rf $PKG; mkdir -p $L
cp build/A2FLive.dll dist/A2FLive.ini $P/
cp $ORT/runtimes/win-x64/native/onnxruntime.dll $ORT/runtimes/win-x64/native/onnxruntime_providers_shared.dll \
   $DML/bin/x64-win/DirectML.dll $MODEL dist/A2FLive.dat dist/README.txt dist/NOTICE.txt $P/A2FLive/ || exit 1
cp LICENSE $L/A2FLive-GPL-3.0.txt
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
    V=$(sed -n 's/^#define A2FLIVE_VERSION "\(.*\)"/\1/p' src/A2FLive.cpp); R=$PWD/release; mkdir -p $R
    rm -f "$R/A2F Live-$V.7z" "$R/A2F Live - Source-$V.zip"
    (cd $PKG && 7z a -t7z -mx=9 "$R/A2F Live-$V.7z" SKSE >/dev/null) &&
    git archive --prefix="A2FLive-$V-src/" -o "$R/A2F Live - Source-$V.zip" HEAD &&
    ls -la $R ;;
esac
