// Box aggregation: horizontal window of blockSize, then vertical window of
// blockSize, both with edge replication.
//
// The two directions are fused because the horizontal sums are kept in a ring of
// rows: advancing one image row only requires adding the new horizontal row and
// subtracting the one that just fell out of the vertical window, so the cost is
// never re-summed over the whole block.
#include "internal.h"

#include <algorithm>
#include <cstring>

namespace sgbm_repro {
namespace internal {

namespace {

// Horizontal pass for one image row, writing the summed row into the ring slot
// for `row`. The window is symmetric with radius SW2 and duplicates the first /
// last column beyond the edges.
void horizontalPass(const Workspace& ws, const short* pixelCost, short* out) {
    const int D = ws.g.numDisp;
    const int width1 = ws.g.width1;
    const int SW2 = ws.g.SW2;
    const short hScale = (short)(SW2 + 1);

    // x = 0: the window covers columns 0..SW2 with column 0 counted SW2+1 times.
    for (int d = 0; d < D; ++d) {
        short acc = mul16(pixelCost[d], hScale);
        for (int j = 1; j <= SW2; ++j) acc = add16(acc, pixelCost[(std::size_t)j * D + d]);
        out[d] = acc;
    }
    // x > 0: slide by one column, clamping both ends.
    for (int x = 1; x < width1; ++x) {
        const short* add = pixelCost + (std::size_t)std::min(x + SW2, width1 - 1) * D;
        const short* sub = pixelCost + (std::size_t)std::max(x - SW2 - 1, 0) * D;
        const short* prev = out + (std::size_t)(x - 1) * D;
        short* dst = out + (std::size_t)x * D;
        for (int d = 0; d < D; ++d) dst[d] = add16(sub16(prev[d], sub[d]), add[d]);
    }
}

}  // namespace

// Accumulates the vertical window for image row `y` into ws.cost.
//
// `firstRowOfImage` tells whether this is row 0, where the top edge has to be
// replicated by weighting the first horizontal row. Rows below the image are
// handled by reusing the last real horizontal row: the caller clamps `k` to
// height-1, and the branch below then rebuilds the cost from that row instead of
// recomputing any pixel cost.
void accumulateCostRow(Workspace& ws, const cv::Mat& left, const cv::Mat& right, int y,
                       bool firstRowOfImage) {
    const Geometry& g = ws.g;
    const int D = g.numDisp;
    const int width1 = g.width1;
    const std::size_t stride = (std::size_t)width1 * D;

    short* cost = ws.costRow();

    const int kBegin = firstRowOfImage ? 0 : y + g.SH2;
    const int kEnd = firstRowOfImage ? g.SH2 : kBegin;

    for (int k = kBegin; k <= kEnd; ++k) {
        const int rowInImage = std::min(k, g.height - 1);
        short* hsum = ws.horizontal(rowInImage);

        if (k < g.height) {
            fillPixelCostRow(ws, left, right, k);
            ws.maybeCapturePixelCost(k);
            horizontalPass(ws, ws.pixelCost.data(), hsum);

            const short* hsumSub = ws.horizontal(std::max(y - g.SH2 - 1, 0));

            if (!firstRowOfImage) {
                // The previous row's cost lives in the same buffer (one row is
                // kept), and every read below happens before the write to the
                // same index, so this is the in-place form of
                // C[x] = Cprev[x] + hsum[x] - hsumSub[x].
                //
                // Row head: add first, then subtract.
                for (int d = 0; d < D; ++d)
                    cost[d] = sub16(add16(cost[d], hsum[d]), hsumSub[d]);
                for (int x = 1; x < width1; ++x) {
                    const short* add = ws.pixelCost.data() +
                                       (std::size_t)std::min(x + g.SW2, width1 - 1) * D;
                    const short* sub = ws.pixelCost.data() +
                                       (std::size_t)std::max(x - (g.SW2 + 1), 0) * D;
                    for (int d = 0; d < D; ++d) {
                        // Sliding step: subtract first, then add -- the opposite
                        // order to the row head. With saturating arithmetic the
                        // two orders are not interchangeable.
                        const short hv = add16(sub16(hsum[(std::size_t)(x - 1) * D + d], sub[d]),
                                               add[d]);
                        hsum[(std::size_t)x * D + d] = hv;
                        cost[(std::size_t)x * D + d] =
                            add16(sub16(cost[(std::size_t)x * D + d],
                                        hsumSub[(std::size_t)x * D + d]),
                                  hv);
                    }
                }
            } else {
                const short scale = (short)(k == 0 ? g.SH2 + 1 : 1);
                for (int d = 0; d < D; ++d) cost[d] = add16(cost[d], mul16(hsum[d], scale));
                for (int x = 1; x < width1; ++x) {
                    const short* add = ws.pixelCost.data() +
                                       (std::size_t)std::min(x + g.SW2, width1 - 1) * D;
                    const short* sub = ws.pixelCost.data() +
                                       (std::size_t)std::max(x - (g.SW2 + 1), 0) * D;
                    for (int d = 0; d < D; ++d) {
                        const short hv =
                            sub16(add16(hsum[(std::size_t)(x - 1) * D + d], add[d]), sub[d]);
                        hsum[(std::size_t)x * D + d] = hv;
                        cost[(std::size_t)x * D + d] =
                            add16(cost[(std::size_t)x * D + d], mul16(hv, scale));
                    }
                }
            }
        } else {
            // Below the last image row: `hsum` already holds the last real row.
            if (!firstRowOfImage) {
                const short* hsumSub = ws.horizontal(std::max(y - g.SH2 - 1, 0));
                for (std::size_t i = 0; i < stride; ++i)
                    cost[i] = add16(sub16(cost[i], hsumSub[i]), hsum[i]);
            } else {
                for (std::size_t i = 0; i < stride; ++i) cost[i] = add16(cost[i], hsum[i]);
            }
        }
    }
}

}  // namespace internal
}  // namespace sgbm_repro
