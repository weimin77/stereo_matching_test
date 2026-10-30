// Shared internal state of the SGBM pipeline.
//
// ARITHMETIC CONTRACT
// -------------------
// Everything below operates on 16-bit signed cost values, and every addition,
// subtraction and multiplication saturates. This is not a stylistic choice:
// OpenCV's SGBM reaches its results through the universal intrinsics, where
// v_int16 v_add/v_sub/v_mul map to _mm_adds_epi16/_mm_subs_epi16 and a widening
// saturating multiply. Its own scalar `#else` fallback wraps instead, so the two
// disagree as soon as an intermediate leaves [-32768, 32767] -- which happens
// with blockSize 11 and preFilterCap 31, where a horizontal window sum times
// (SH2+1) crosses 32767. Since the build we validate against takes the
// saturating path, so does this implementation. The helpers below make every
// saturation point explicit.
//
// Two further narrowing points matter and are reproduced deliberately:
//   * the P2 + minLr delta is narrowed with a truncating cast (vx_setall_s16)
//   * the pseudo-right disparity / cost maps store plain ints, not fixed point
#ifndef SGBM_REPRO_SRC_INTERNAL_H
#define SGBM_REPRO_SRC_INTERNAL_H

#include "sgbm_repro/sgbm.h"

#include <opencv2/core.hpp>

#include <algorithm>
#include <cstring>
#include <limits>
#include <vector>

namespace sgbm_repro {
namespace internal {

constexpr int kMaxCost = std::numeric_limits<short>::max();
constexpr int kDispShift = 4;
constexpr int kDispScale = 1 << kDispShift;

inline short saturate16(int v) {
    return (short)std::min(kMaxCost, std::max(-kMaxCost - 1, v));
}
inline short add16(short a, short b) { return saturate16((int)a + b); }
inline short sub16(short a, short b) { return saturate16((int)a - b); }
inline short mul16(short a, short b) { return saturate16((int)a * b); }
// vx_setall_s16((short)value): truncation, not saturation.
inline short narrow16(int v) { return (short)(v & 0xffff); }

// Lookup table that folds the clipped 3x3 derivative response used as the first
// cost layer. index = 1024 + (2*(r[x+1]-r[x-1]) + (up-right - up-left) +
// (down-right - down-left)), clipped to +-ftZero and biased back to be
// non-negative.
class ClipTable {
public:
    static constexpr int kOffset = 256 * 4;
    static constexpr int kSize = 256 + kOffset * 2;

    explicit ClipTable(int preFilterCap) {
        // OpenCV clamps the clip range to at least 15, and forces it odd with a
        // bitwise-or (not +1).
        const int ftZero = std::max(preFilterCap, 15) | 1;
        table_.resize(kSize);
        for (int i = 0; i < kSize; ++i)
            table_[i] = (uchar)(std::min(std::max(i - kOffset, -ftZero), ftZero) + ftZero);
    }

    const uchar* centered() const { return table_.data() + kOffset; }

private:
    std::vector<uchar> table_;
};

// All working memory for one image pair. Sized once, reused for every row.
class Workspace {
public:
    Workspace(const Geometry& geometry, int channels, int width);

    Geometry g;
    int channels;

    // Cost volume row: width1 * numDisp, one disparity block per column.
    std::vector<short> pixelCost;
    // Aggregated cost for the current row (single row: MODE_SGBM only).
    std::vector<short> cost;
    // Sum of the per-direction partial costs for the current row.
    std::vector<short> aggregate;

    // Horizontal window sums, kept as a ring so that the vertical window can be
    // advanced by adding one row and subtracting another.
    std::size_t ringRows;
    std::vector<short> horizontalRing;

    // Prefilter scratch: two layers of width*channels entries each plus the
    // running min/max pair.
    std::vector<uchar> prefilterScratch;

    // Current and previous row of directional partial costs. Index [0] of a
    // slot is d = 0; the slot is padded on both sides so d = -1 and d = numDisp
    // can be written as sentinels.
    std::vector<short> directional[2];
    std::vector<short> directionalMin[2];

    ClipTable clipTable;

    // Pseudo right-view disparity (plain pixels) and its cost, used by the
    // final left-right consistency test. Reset for every row.
    std::vector<short> pseudoRightDisp;
    std::vector<short> pseudoRightCost;

    // Optional per-stage capture; not used by the pipeline logic.
    DebugTrace* trace = nullptr;

    int slots() const { return g.width1 + 2; }          // x index -1 .. width1
    short* directionRow(int id, int x, int shift);
    const short* directionRow(int id, int x, int shift) const;
    short* directionMin(int id, int x, int shift);
    const short* directionMin(int id, int x, int shift) const;

    short* horizontal(int row) { return horizontalRing.data() + (row % ringRows) * rowStride(); }
    short* costRow() { return cost.data(); }
    short* aggregateRow() { return aggregate.data(); }

    void resetDirectional();
    void zeroAggregate();

    // Records the raw pixel cost of image row `row` if a trace asks for it.
    void maybeCapturePixelCost(int row);

private:
    std::size_t rowStride() const { return (std::size_t)g.width1 * g.numDisp; }
};

// Fills ws.pixelCost for one image row (prefilter.cpp).
void fillPixelCostRow(Workspace& ws, const cv::Mat& left, const cv::Mat& right, int y);

// Accumulates the box-aggregated cost of image row y into ws.cost
// (cost_volume.cpp). `firstRowOfImage` is true for y == 0.
void accumulateCostRow(Workspace& ws, const cv::Mat& left, const cv::Mat& right, int y,
                       bool firstRowOfImage);

// Four directional passes for image row y (dp_sgbm.cpp).
void forwardSweep(Workspace& ws, int previousId, int currentId, int y);

// Fifth direction, label selection and the consistency test (select.cpp).
void reverseSweepAndSelect(Workspace& ws, int currentId, int y, short* disparityRow);

}  // namespace internal
}  // namespace sgbm_repro

#endif  // SGBM_REPRO_SRC_INTERNAL_H
