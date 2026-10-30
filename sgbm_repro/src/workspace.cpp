// Buffer layout for the pipeline.
//
// Compared with OpenCV's BufferSGBM the layout differs deliberately: every
// (column, direction) pair gets its own slot, and each slot is padded by one
// entry on either side so that the d = -1 / d = numDisparities sentinels never
// alias real data. OpenCV packs the four directions of a column contiguously
// with a stride of Da + vlanes and lets the d = -1 sentinel of one direction land
// in the padding of the previous one; since neither layout is ever read outside
// [0, numDisp), the computed values are the same.
#include "internal.h"

#include <algorithm>

namespace sgbm_repro {
namespace internal {

Workspace::Workspace(const Geometry& geometry, int channels_, int width)
    : g(geometry), channels(channels_), clipTable(geometry.preFilterCap) {
    const std::size_t cols = (std::size_t)g.width1 * g.numDisp;
    pixelCost.assign(cols, 0);
    cost.assign(cols, 0);
    aggregate.assign(cols, 0);

    // Rows of horizontal sums that must coexist: the vertical window
    // (2*SH2+1 rows) plus the one being subtracted, plus one slot of slack so the
    // newest and oldest never collide.
    ringRows = (std::size_t)g.SH2 * 2 + 3;
    horizontalRing.assign(ringRows * cols, 0);

    // Two prefilter layers for both images, each width*channels, plus the
    // running 3-tap min and max of the right image.
    prefilterScratch.assign((std::size_t)width * (4 * channels + 2), 0);

    const std::size_t slotStride = (std::size_t)(g.numDisp + 2);
    for (int i = 0; i < 2; ++i) {
        directional[i].assign((std::size_t)slots() * 4 * slotStride, 0);
        directionalMin[i].assign((std::size_t)slots() * 4, 0);
    }

    pseudoRightDisp.assign(g.width, 0);
    pseudoRightCost.assign(g.width, 0);
}

short* Workspace::directionRow(int id, int x, int shift) {
    const std::size_t stride = (std::size_t)(g.numDisp + 2);
    return directional[id].data() + (((std::size_t)(x + 1) * 4 + shift) * stride) + 1;
}

const short* Workspace::directionRow(int id, int x, int shift) const {
    const std::size_t stride = (std::size_t)(g.numDisp + 2);
    return directional[id].data() + (((std::size_t)(x + 1) * 4 + shift) * stride) + 1;
}

short* Workspace::directionMin(int id, int x, int shift) {
    return directionalMin[id].data() + (std::size_t)(x + 1) * 4 + shift;
}

const short* Workspace::directionMin(int id, int x, int shift) const {
    return directionalMin[id].data() + (std::size_t)(x + 1) * 4 + shift;
}

void Workspace::resetDirectional() {
    for (int i = 0; i < 2; ++i) {
        std::fill(directional[i].begin(), directional[i].end(), (short)0);
        std::fill(directionalMin[i].begin(), directionalMin[i].end(), (short)0);
    }
}

void Workspace::zeroAggregate() { std::fill(aggregate.begin(), aggregate.end(), (short)0); }

}  // namespace internal
}  // namespace sgbm_repro
