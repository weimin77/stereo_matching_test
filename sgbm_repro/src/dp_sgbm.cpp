// Forward half of the semi-global optimisation.
//
// For MODE_SGBM one image pass runs four directions left-to-right; a fifth
// direction comes from the reverse sweep in select.cpp, which reuses the
// direction-0 buffer. That is why this file computes four partial costs per
// pixel and not eight.
//
// Recurrence, per direction r:
//
//   L(p, d) = C(p, d) + min( L(p-r, d),
//                            L(p-r, d-1) + P1,
//                            L(p-r, d+1) + P1,
//                            min_k L(p-r, k) + P2 ) - min_k L(p-r, k)
//
// with the last term folded into a per-direction delta so the inner loop is one
// min chain: L = C + sat( min(L[d], L[d-1]+P1, L[d+1]+P1, delta) - delta ).
#include "internal.h"

#include <algorithm>

namespace sgbm_repro {
namespace internal {

void forwardSweep(Workspace& ws, int previousId, int currentId, int y) {
    const Geometry& g = ws.g;
    const int D = g.numDisp;
    const int width1 = g.width1;
    const short P1 = narrow16(g.P1);

    short* aggregate = ws.aggregateRow();
    const short* cost = ws.costRow();

    for (int x = 0; x < width1; ++x) {
        // Each direction's delta is P2 plus the minimum partial cost of the
        // neighbour it propagates from. The narrowing to short is intentional:
        // the value is materialised as an int16 before use.
        const short delta0 = narrow16(g.P2 + *ws.directionMin(currentId, x - 1, 0));
        const short delta1 = narrow16(g.P2 + *ws.directionMin(previousId, x - 1, 1));
        const short delta2 = narrow16(g.P2 + *ws.directionMin(previousId, x, 2));
        const short delta3 = narrow16(g.P2 + *ws.directionMin(previousId, x + 1, 3));

        short* in0 = ws.directionRow(currentId, x - 1, 0);
        short* in1 = ws.directionRow(previousId, x - 1, 1);
        short* in2 = ws.directionRow(previousId, x, 2);
        short* in3 = ws.directionRow(previousId, x + 1, 3);
        // d = -1 and d = numDisp are outside the label range, so they must never
        // be selected.
        in0[-1] = in0[D] = (short)kMaxCost;
        in1[-1] = in1[D] = (short)kMaxCost;
        in2[-1] = in2[D] = (short)kMaxCost;
        in3[-1] = in3[D] = (short)kMaxCost;

        short* out0 = ws.directionRow(currentId, x, 0);
        short* out1 = ws.directionRow(currentId, x, 1);
        short* out2 = ws.directionRow(currentId, x, 2);
        short* out3 = ws.directionRow(currentId, x, 3);
        const short* C = cost + (std::size_t)x * D;
        short* S = aggregate + (std::size_t)x * D;

        // Kept in locals: the running minima are touched once per label, and
        // recomputing the four pointers inside the label loop costs more than the
        // recurrence itself.
        short minL0 = (short)kMaxCost;
        short minL1 = (short)kMaxCost;
        short minL2 = (short)kMaxCost;
        short minL3 = (short)kMaxCost;

        for (int d = 0; d < D; ++d) {
            const short Cd = C[d];
            short sum = S[d];

            auto step = [&](const short* in, short delta) -> short {
                const short best = std::min(std::min(in[d], add16(in[d - 1], P1)),
                                            std::min(add16(in[d + 1], P1), delta));
                return add16(sub16(best, delta), Cd);
            };

            const short L0 = step(in0, delta0);
            out0[d] = L0;
            minL0 = std::min(minL0, L0);
            sum = add16(sum, L0);

            const short L1 = step(in1, delta1);
            out1[d] = L1;
            minL1 = std::min(minL1, L1);
            sum = add16(sum, L1);

            const short L2 = step(in2, delta2);
            out2[d] = L2;
            minL2 = std::min(minL2, L2);
            sum = add16(sum, L2);

            const short L3 = step(in3, delta3);
            out3[d] = L3;
            minL3 = std::min(minL3, L3);
            sum = add16(sum, L3);

            S[d] = sum;
        }

        *ws.directionMin(currentId, x, 0) = minL0;
        *ws.directionMin(currentId, x, 1) = minL1;
        *ws.directionMin(currentId, x, 2) = minL2;
        *ws.directionMin(currentId, x, 3) = minL3;
    }
    (void)y;
}

}  // namespace internal
}  // namespace sgbm_repro
