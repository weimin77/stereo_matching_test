# sgbm_repro

A standalone reimplementation of OpenCV's `StereoSGBM` in `MODE_SGBM`, written so
that its `CV_16S` output is **byte-for-byte identical** to `cv::StereoSGBM`, and
written in plain scalar C++ so it can be ported to a target without OpenCV or
with a different SIMD width.

Only the pieces needed for that agreement are here:

| Supported | Not supported (on purpose) |
|---|---|
| `MODE_SGBM` (one pass, 5 directions) | `MODE_HH`, `MODE_HH4`, `MODE_SGBM_3WAY` |
| `CV_8UC1` input | multi-channel input, ROI processing |
| `medianBlur(3)` + `filterSpeckles` epilogue | `StereoBinarySGBM`, WLS, anything downstream of SGBM |

The status is `ALL STAGES BIT-EXACT`: see [Results](#results).

---

## Why "identical to OpenCV" needed a definition

The reference is not unique, which is worth knowing before copying this code:

* `cv::setUseOptimized(false)` does **not** change the SGBM result. The file is
  not a dispatched translation unit, so `CV_SIMD` is fixed at build time. This
  was verified: 0 differing pixels over 4.2 M across 51 cases.
* OpenCV's SGBM **does not agree with itself across backends**. For `v_int16`,
  `v_add`/`v_sub`/`v_mul` map to `_mm_adds_epi16` / `_mm_subs_epi16` / a widening
  saturating multiply, while the `#else` scalar fallback in the same file uses
  plain wrapping `short` arithmetic. The two diverge as soon as an intermediate
  leaves `[-32768, 32767]`, which happens at `blockSize 11` with
  `preFilterCap 31`: a horizontal window sum times `(SH2+1)` crosses 32767.

So this implementation pins the **saturating** semantics, i.e. what a normal x86
build ships. Every 16-bit operation is spelled out as `add16`/`sub16`/`mul16` in
[`src/internal.h`](src/internal.h) so the saturation points are auditable. Two
narrowing points are also reproduced deliberately: the `P2 + minLr` delta is
truncated to `short` before use, and the consistency maps store plain pixel
disparity rather than fixed point.

---

## Layout

```
include/sgbm_repro/sgbm.h   public API: Params, Geometry, DebugTrace, computeDisparity
src/internal.h              saturating arithmetic, clip table, buffer layout
src/workspace.cpp           buffer sizing and directional slot addressing
src/prefilter.cpp           two-layer Sobel/clip prefilter + Birchfield-Tomasi cost
src/cost_volume.cpp         horizontal then vertical box aggregation
src/dp_sgbm.cpp             four directional passes
src/select.cpp              fifth direction, label choice, sub-pixel fit, L/R check
src/postprocess.cpp         3x3 median, speckle filter
src/sgbm.cpp                orchestration, geometry resolution
tests/oracle/               scalar transcription of OpenCV's own SGBM, test-only
tests/probe_simd.cpp        proves the reference is unique and the oracle correct
tests/compare.cpp           per-stage comparison over a synthetic matrix
tests/pairs.cpp             comparison on a real image pair
tests/debug_case.cpp        ablation harness used while developing
tests/run_all.sh            everything above, with a pass/fail summary
tools/bench.cpp             single-frame time and peak RSS
```

`tests/oracle/` is a transcription of `opencv/modules/stereo/src/stereosgbm.cpp`
(Apache-2.0) with the universal-intrinsic branches dropped and the saturating
semantics made explicit. It exists because the two functions worth comparing
against are file-static, and because cv::StereoSGBM never exposes the per-stage
buffers (pixel cost, aggregated cost, directional partial costs) — without those,
a mismatch cannot be attributed to a stage. It is linked into the test binaries
only.

---

## Build and run

Requires OpenCV 5 with `core`, `imgproc`, `stereo`. `OPENCV_LOCAL_PREFIX`
defaults to `$HOME/opt/opencv5`.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

./build/probe_simd          # reference uniqueness + oracle fidelity
./build/compare             # per-stage agreement, 88 synthetic cases
./build/pairs <l> <r> <minDisp> <numDisp> <block> <P1> <P2> \
              <preFilterCap> <uniquenessRatio> <speckleWindowSize> <speckleRange> <disp12MaxDiff>
./build/bench [<l> <r> <minDisp> <numDisp> <block>]

tests/run_all.sh            # all of the above plus the real image pairs
```

`run_all.sh` reads the oven captures from the parent directory
(`../left_on.png`, `../right_on.png`, `../left_off.png`, `../right_off.png`) and a
Middlebury 2006 subset from
`/home/min/projects/stereo/Middlebury Stereo Datasets/2006`; missing inputs are
skipped rather than failed.

---

## Results

### Agreement

| Suite | Cases | Verdict |
|---|---|---|
| `probe_simd` — reference uniqueness, oracle fidelity | 51 cases / 4,210,455 px | 0 differences |
| `compare` — synthetic matrix, 6 stages each | 88 cases / 9.6 M output values | 0 differences |
| oven captures, `min=32 num=96`, `blockSize` 3 and 5 | 4 runs / 1.2 M px | 0 differences |
| Middlebury 2006, `min=0 num=64 bs=5` | 10 scenes / 1.6 M px | 0 differences |

The synthetic matrix covers both edge windows, `numDisparities` that are not a
multiple of the int16 lane count, negative and positive `minDisparity`,
`blockSize` 1 / 4 / 7 / 9 / 11, `preFilterCap` 0 / 63 / 200, `uniquenessRatio`
-1 / 0 / 40, `disp12MaxDiff` 0 / 64, speckle window 0 / 5000, and heights down to
one block.

### Speed and memory

640×480, `min=32 num=96 block=5`, single frame, best of five:

| | time | vs `cv::StereoSGBM` |
|---|---|---|
| `cv::StereoSGBM` (SSE2 intrinsics) | 47 ms | 1.0× |
| oracle (same scalar code) | 340 ms | 7.2× |
| `sgbm_repro` | 343 ms | 7.3× |

Peak RSS for the whole bench process: 18 MB. This implementation is not faster
than OpenCV and does not try to be — the point is a source-level description of
the arithmetic that can be retargeted. Being level with the scalar transcription
means nothing pathological crept into the buffer layout.

---

## Things that would silently break the agreement

Collected here because each one cost a debugging session:

1. **Layer 1 boundary.** The raw-intensity cost layer keeps the clip table's
   zero point in columns `0` and `width-1`, not the edge pixel. OpenCV
   initialises both layers to `tab[0]` and only overwrites `[1, width-1)`.
2. **Cost buffer starts at `P2`,** not zero. The recurrence adds `P2` on the
   first row and subtracts it per direction; with saturating arithmetic the
   starting value is observable.
3. **Operand order is not commutative** once saturation is involved. The row head
   of the aggregation is `sat(sat(C + hsum) - hsumSub)`, the sliding step is
   `sat(sat(hsum - pixSub) + pixAdd)`. They are not the same expression.
4. **The bottom `SH2` rows do not freeze the cost.** When `k >= height` the cost
   is rebuilt from the last real horizontal row
   (`sat(sat(Cprev - hsumSub) + hsum)`), which is a different code path from the
   normal one.
5. **The delta is truncated, not saturated**, when it is materialised as `int16`
   (`P2 + minLr` can exceed 32767).
6. **The sub-pixel fit is integer.** `d = d*16 + ((S[d-1]-S[d+1])*16 + denom) / (denom*2)`
   with `denom = max(S[d-1]+S[d+1]-2*S[d], 1)`. A floating-point version drifts by
   one unit of 1/16 px, which is enough to fail.
7. **`numDisparities` padding is harmless.** OpenCV pads the cost row stride to
   `alignSize(numDisparities, lanes)`; this implementation does not. Cases with
   `numDisparities` of 20, 36 and 44 confirm the padding never reaches the result.

---

## Not implemented

* `MODE_HH` / `MODE_HH4` — a second image pass plus a full-image cost volume. The
  cost, aggregation and epilogue stages here are already verified, so adding it
  means the extra pass in `dp_sgbm.cpp` and `costHeight = height` buffers.
* Negative-disparity mirroring for the right view (`rightMinDisparity`), which
  `test.cpp` handles outside the matcher.
