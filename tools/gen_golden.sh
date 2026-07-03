#!/bin/bash
# Regenerate C-oracle golden files from ../koral_lite.
#
# Copies the koral_lite sources into oracle/build/src (never touches the
# original tree), patches PROBLEM -> 147 (PUFFY), compiles the compute
# objects (no silo, no MPI, no HDF5 -- all ifdef'd out) with clang + GSL,
# builds the harnesses, and runs them into tests/golden/.
#
# Usage: tools/gen_golden.sh            (KORAL_LITE=<path> to override)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${KORAL_LITE:-$ROOT/../koral_lite}"
BUILD="$ROOT/oracle/build"
GSL_PREFIX="$(brew --prefix gsl)"

[ -f "$SRC/ko.h" ] || { echo "koral_lite not found at $SRC" >&2; exit 1; }

echo "== copying sources"
rm -rf "$BUILD"
mkdir -p "$BUILD/src"
cp "$SRC"/*.c "$SRC"/*.h "$BUILD/src/"
cp -R "$SRC/PROBLEMS" "$BUILD/src/PROBLEMS"

echo "== patching PROBLEM -> 147 (PUFFY)"
sed -i '' 's/^#define PROBLEM .*/#define PROBLEM 147/' "$BUILD/src/problem.h"
grep -q "^#define PROBLEM 147" "$BUILD/src/problem.h"

cp "$ROOT"/oracle/harness_*.c "$BUILD/src/"

echo "== compiling (clang, serial, GSL only)"
cd "$BUILD/src"
OBJS="mpi u2prad magn postproc fileop misc physics finite problem metric relele rad opacities u2p u2p_ff frames p2u nonthermal"
CFLAGS="-O2 -fcommon -w -I$GSL_PREFIX/include"
for o in $OBJS; do
  cc $CFLAGS -c "$o.c" -o "$o.o" &
done
wait

build_harness() {
  local name="$1"
  cc $CFLAGS -c "$name.c" -o "$name.o"
  cc -o "$name" "$name.o" $(for o in $OBJS; do echo "$o.o"; done) \
     -L"$GSL_PREFIX/lib" -lgsl -lgslcblas -lm
}

echo "== building harnesses"
build_harness harness_metric

echo "== running harness_metric"
mkdir -p "$ROOT/tests/golden/metric"
./harness_metric "$ROOT/tests/golden/metric"

echo "== writing manifest"
SHA="$(git -C "$SRC" rev-parse --short HEAD 2>/dev/null || echo unknown)"
cat > "$ROOT/tests/golden/manifest.json" <<EOF
{
  "koral_lite_sha": "$SHA",
  "problem": 147,
  "compiler": "$(cc --version | head -1)",
  "generated": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "files": {
    "metric/metric_points.kgld": "coords,x0..x3 -> gg16 GG16 gdet dlgdet3 Kr64 gttpert",
    "metric/coco_dxdx.kgld": "co1,co2,x0..x3 -> xout4 dxdx16",
    "metric/krzysie_grid.kgld": "ix,iy -> x1 x2 Kr64 (MODYFIKUJKRZYSIE-corrected, PUFFY grid)"
  }
}
EOF

echo "== done"
