#!/bin/sh
# Build the MSVC/CommonLib plugin and FFmpeg C DLL; assemble the mod in build/mod.
#   ./build.sh                    build + assemble
#   ./build.sh install <dir>      also copy into <dir> (e.g. an MO2 mod folder)
#   ./build.sh release            write matching -with-sentiment and -without-sentiment archives
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
    if [ "$N" -ge 15 ]; then
        echo "Release build exceeds SKSE revision limit (15); version unchanged." >&2
        exit 1
    fi
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
AGILITY=${AGILITY:-deps/dml/microsoft.direct3d.d3d12}  # pinned by dist/Agility/SOURCE.txt, SHA256SUMS
MODEL=${MODEL:-models/mark/network_fp16.onnx}  # Audio2Face-3D v2.3 Mark
MODEL_FP32=${MODEL_FP32:-$(dirname "$MODEL")/network.onnx}  # test only: reference for the FP16 network
MODEL3=${MODEL3:-models/v3/network_fp16.onnx}  # v3.0, folded (tools/a2f3_fold.py)
MODEL3_FP32=${MODEL3_FP32:-$(dirname "$MODEL3")/network_folded.onnx}  # test only
EMOTION_MODEL=${EMOTION_MODEL:-models/a2e/network_fp16.onnx}
ADDRLIB=${ADDRLIB:-deps/address-library}  # test only
JOBS=${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)}
case "$JOBS" in ""|*[!0-9]*|0*) echo "JOBS must be a positive integer" >&2; exit 1 ;; esac

if [ "$1" = release ] && [ ! -f "$EMOTION_MODEL" ]; then
    echo "Both release archives require EMOTION_MODEL: $EMOTION_MODEL" >&2
    exit 1
fi

mkdir -p build
# CommonLib and the plugin share the MSVC C++ ABI; FFmpeg remains a C DLL.
BFA_BUILD_VERSION="$V" python3 tools/commonlib_build.py || exit 1

PKG=build/mod; P=$PKG/SKSE/Plugins; L=$P/BetterFacialAnimations/licenses
rm -rf $PKG; mkdir -p $L
cp build/BetterFacialAnimations.dll dist/BetterFacialAnimations.ini $P/
cp build/BFAFFmpeg.dll $P/BetterFacialAnimations/ || exit 1
cp $ORT/runtimes/win-x64/native/onnxruntime.dll $ORT/runtimes/win-x64/native/onnxruntime_providers_shared.dll \
   $DML/bin/x64-win/DirectML.dll $MODEL dist/A2FDirect.bin dist/A2FDirect3.bin dist/NOTICE.txt $P/BetterFacialAnimations/ || exit 1
cp $MODEL3 $P/BetterFacialAnimations/a2f3_fp16.onnx || exit 1
if [ -f "$EMOTION_MODEL" ]; then
    cp "$EMOTION_MODEL" "$P/BetterFacialAnimations/emotion_fp16.onnx" || exit 1
fi
cp include/a2f/speech/BFA_API.h $P/BetterFacialAnimations/ || exit 1
cp dist/performance.onnx dist/performance-model.json $P/BetterFacialAnimations/ || exit 1
cp dist/emotion_performance.onnx dist/emotion_performance.json $P/BetterFacialAnimations/ || exit 1
cp dist/GPU-SETUP.txt $P/BetterFacialAnimations/ || exit 1
sed "s/@VERSION@/$V/" dist/README.txt > $P/BetterFacialAnimations/README.txt
cp LICENSE $L/BetterFacialAnimations-GPL-3.0.txt
cp $FFMPEG/COPYING.LGPLv2.1 $L/FFmpeg-LGPL-2.1.txt
cp $ORT/LICENSE $L/onnxruntime-LICENSE.txt
cp $ORT/ThirdPartyNotices.txt $L/onnxruntime-ThirdPartyNotices.txt
cp $DML/LICENSE.txt $L/DirectML-LICENSE.txt
AGILITY_DLL=$AGILITY/build/native/bin/x64/D3D12Core.dll
echo "$(cut -d' ' -f1 dist/Agility/SHA256SUMS)  $AGILITY_DLL" | sha256sum -c - || exit 1
mkdir -p "$P/BetterFacialAnimations/Agility"
cp "$AGILITY_DLL" "$P/BetterFacialAnimations/Agility/" || exit 1
cp dist/licenses/* $L/
python3 tools/package_installer.py build/installer "$V" || exit 1
du -sh $PKG

case "$1" in
compat)
    python3 tools/compatibility.py --tier contracts ;;
install)
    [ -n "$2" ] || { echo "usage: $0 install <dir>"; exit 1; }
    mkdir -p "$2" && cp -r $PKG/. "$2/" && echo "installed to $2" ;;
release)
    R=$PWD/release; mkdir -p $R
    for variant in without-sentiment with-sentiment; do
        archive="$R/Better Facial Animations-$V$ARCHIVE_SUFFIX-$variant.7z"
        if [ "$variant" = without-sentiment ]; then
            python3 tools/package_installer.py build/installer "$V" --without-full-emotion || exit 1
            (cd "$PKG" && 7z a -t7z -mx=9 "$archive" SKSE \
                -x!SKSE/Plugins/BetterFacialAnimations/emotion_fp16.onnx >/dev/null) || exit 1
        else
            python3 tools/package_installer.py build/installer "$V" || exit 1
            (cd "$PKG" && 7z a -t7z -mx=9 "$archive" SKSE >/dev/null) || exit 1
        fi
        (cd build/installer && 7z a -t7z -mx=9 "$archive" fomod options >/dev/null) || exit 1
        echo "$archive"
    done ;;
test)
    python3 tools/compatibility.py --tier contracts &&
    python3 tools/commonlib_build.py --extra-tests ;;
esac
