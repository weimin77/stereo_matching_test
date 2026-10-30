#!/usr/bin/env bash
# Runs every agreement check in one go.
#
#   tests/run_all.sh [build-dir]
#
# Exit code 0 means every stage of every case was byte-identical to
# cv::StereoSGBM. The Middlebury part is skipped when the dataset is absent.
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/.." && pwd)"
build="${1:-$root/build}"
testdir="$(cd "$root/.." && pwd)"

fail=0
pass() { printf '  \033[32mok\033[0m   %s\n' "$1"; }
bad()  { printf '  \033[31mFAIL\033[0m %s\n' "$1"; fail=1; }

echo "== 0. arithmetic / reference uniqueness =="
if "$build/probe_simd" >/tmp/sgbm_probe.log 2>&1; then
    pass "reference is independent of setUseOptimized and the oracle reproduces it"
    sed -n '/--- summary/,$p' /tmp/sgbm_probe.log | sed 's/^/       /'
else
    bad "probe_simd"
    tail -20 /tmp/sgbm_probe.log
fi

echo
echo "== 1. synthetic matrix, per stage =="
if "$build/compare" >/tmp/sgbm_compare.log 2>&1; then
    pass "all stages bit-exact"
    sed -n '/--- [0-9]* cases/,$p' /tmp/sgbm_compare.log | sed 's/^/       /'
else
    bad "compare"
    grep -E 'FAIL|ORACLE' /tmp/sgbm_compare.log | head -30
    sed -n '/--- [0-9]* cases/,$p' /tmp/sgbm_compare.log | sed 's/^/       /'
fi

echo
echo "== 2. oven simulation captures =="
# (left, right, minDisp, numDisp, block, p1Factor, p2Factor, pfc, ur, sws, sr, d12)
run_pair() {
    local l="$1" r="$2" min="$3" num="$4" bs="$5"
    local p1=$((8 * bs * bs)) p2=$((256 * bs * bs))
    local label
    label="$(basename "$(dirname "$l")")/$(basename "$l") min=$min num=$num bs=$bs"
    if [[ ! -f "$l" ]]; then
        printf '  skip %s (missing)\n' "$label"
        return
    fi
    if "$build/pairs" "$l" "$r" "$min" "$num" "$bs" "$p1" "$p2" 31 5 40 1 1 \
            >/tmp/sgbm_pair.log 2>&1; then
        pass "$label"
    else
        bad "$label"
        cat /tmp/sgbm_pair.log | sed 's/^/       /'
    fi
}

for case in on off; do
    run_pair "$testdir/left_$case.png" "$testdir/right_$case.png" 32 96 5
    run_pair "$testdir/left_$case.png" "$testdir/right_$case.png" 32 96 3
done

echo
echo "== 3. Middlebury 2006 =="
mroot="/home/min/projects/stereo/Middlebury Stereo Datasets/2006"
if [[ -d "$mroot" ]]; then
    for scene in Baby2 Bowling1 Cloth1 Lampshade1 Midd1 Monopoly Plastic Rocks2 Wood1 Aloe; do
        run_pair "$mroot/$scene/view1.png" "$mroot/$scene/view5.png" 0 64 5
    done
else
    printf '  skip (dataset not found at %s)\n' "$mroot"
fi

echo
if [[ $fail -eq 0 ]]; then
    printf '\033[32mALL CHECKS PASSED\033[0m\n'
else
    printf '\033[31mFAILURES PRESENT\033[0m\n'
fi
exit $fail
