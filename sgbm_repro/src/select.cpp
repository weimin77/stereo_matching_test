// Reverse half of the semi-global optimisation: the fifth direction, the
// winning label, sub-pixel refinement and the left-right consistency test.
//
// The fifth direction is produced by walking x from right to left and writing
// into the same buffer slot the forward sweep used for direction 0. Reading the
// slot of x+1 therefore reads the previous iteration's result, and at
// x == width1-1 it reads the zero-initialised padding, which is exactly the
// "no cost yet" state the forward sweep has at x == 0.
#include "internal.h"

#include <algorithm>
#include <cstdlib>

namespace sgbm_repro {
namespace internal {

namespace {

// Quadratic fit through (d-1, S[d-1]), (d, S[d]), (d+1, S[d+1]). Integer
// arithmetic throughout: a floating-point version drifts by one unit in the
// 1/16 fixed-point result, which is enough to break exact agreement.
int subpixel(const short* S, int d, int D) {
    if (d > 0 && d < D - 1) {
        const int denom = std::max((int)S[d - 1] + (int)S[d + 1] - 2 * (int)S[d], 1);
        return d * kDispScale + (((int)S[d - 1] - (int)S[d + 1]) * kDispScale + denom) / (denom * 2);
    }
    return d * kDispScale;
}

}  // namespace

void reverseSweepAndSelect(Workspace& ws, int currentId, int y, short* disp1Row) {
    const Geometry& g = ws.g;
    const int D = g.numDisp;
    const int width1 = g.width1;
    const int width = g.width;
    const short P1 = narrow16(g.P1);

    short* aggregate = ws.aggregateRow();
    const short* cost = ws.costRow();

    for (int x = 0; x < width; ++x) {
        disp1Row[x] = (short)g.invalidDispScaled;
        ws.pseudoRightDisp[x] = (short)g.invalidDispScaled;
        ws.pseudoRightCost[x] = (short)kMaxCost;
    }

    for (int x = width1 - 1; x >= 0; --x) {
        short* in0 = ws.directionRow(currentId, x + 1, 0);
        short* out0 = ws.directionRow(currentId, x, 0);
        const short* C = cost + (std::size_t)x * D;
        short* S = aggregate + (std::size_t)x * D;

        in0[-1] = in0[D] = (short)kMaxCost;
        const short delta = narrow16(g.P2 + *ws.directionMin(currentId, x + 1, 0));

        short minL = (short)kMaxCost;
        short minS = (short)kMaxCost;
        int best = -1;

        for (int d = 0; d < D; ++d) {
            const short bestNeighbour = std::min(std::min(in0[d], add16(in0[d - 1], P1)),
                                                 std::min(add16(in0[d + 1], P1), delta));
            const short L = add16(sub16(bestNeighbour, delta), C[d]);
            out0[d] = L;
            minL = std::min(minL, L);

            const short total = add16(S[d], L);
            S[d] = total;
            if (total < minS) {
                minS = total;
                best = d;
            }
        }
        *ws.directionMin(currentId, x, 0) = minL;

        // Reject the match when a clearly better, non-adjacent label is close
        // enough that the choice is ambiguous.
        {
            int d = 0;
            for (; d < D; ++d) {
                if ((int)S[d] * (100 - g.uniquenessRatio) < (int)minS * 100 && std::abs(best - d) > 1)
                    break;
            }
            if (d < D) continue;
        }

        // Record the winner in the pseudo right-view map. Column x of the left
        // image claims column x - d of the right image; the cheapest claim wins.
        const int rx = x + g.minX1 - best - g.minDisp;
        if (ws.pseudoRightCost[rx] > minS) {
            ws.pseudoRightCost[rx] = minS;
            ws.pseudoRightDisp[rx] = (short)(best + g.minDisp);
        }

        disp1Row[x + g.minX1] = (short)(subpixel(S, best, D) + g.minDisp * kDispScale);
    }

    // Left-right consistency against the pseudo right-view map. Both the
    // truncation and the rounding of the fixed-point disparity are tried, so a
    // pixel is only rejected when neither can be confirmed.
    for (int x = g.minX1; x < g.maxX1; ++x) {
        const int v = disp1Row[x];
        if (v == g.invalidDispScaled) continue;
        const int floorD = v >> kDispShift;
        const int ceilD = (v + kDispScale - 1) >> kDispShift;
        const int a = x - floorD;
        const int b = x - ceilD;
        const bool rejectA = a >= 0 && a < width && ws.pseudoRightDisp[a] >= g.minDisp &&
                             std::abs(ws.pseudoRightDisp[a] - floorD) > g.disp12MaxDiff;
        const bool rejectB = b >= 0 && b < width && ws.pseudoRightDisp[b] >= g.minDisp &&
                             std::abs(ws.pseudoRightDisp[b] - ceilD) > g.disp12MaxDiff;
        if (rejectA && rejectB) disp1Row[x] = (short)g.invalidDispScaled;
    }
    (void)y;
}

}  // namespace internal
}  // namespace sgbm_repro
