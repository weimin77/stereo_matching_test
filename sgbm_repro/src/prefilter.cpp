// Cost prefilter and the Birchfield-Tomasi pixel cost.
//
// Two cost layers are produced per pixel, in this order:
//   layer 0  the clipped 3x3 derivative response, full weight
//   layer 1  the raw intensity, right-shifted by two bits
// Layer 0 is insensitive to gain; layer 1 keeps some absolute information. Both
// accumulate into the same cost volume.
//
// The horizontal window (of size blockSize) is *not* applied here; see
// cost_volume.cpp. This file only fills the per-pixel cost.
#include "internal.h"

#include <algorithm>
#include <cstring>

namespace sgbm_repro {
namespace internal {

namespace {

// Builds one prefilter layer for a single image row.
//
// `up`/`down` are the row offsets used by the derivative term; they collapse to
// zero at the top and bottom image edges, which replicates the edge row. The two
// end columns are forced to the table's zero point so that the 3-tap windows in
// the matching loop see a neutral boundary.
void buildLayer(const cv::Mat& image, int y, int cn, int width, const uchar* clip, uchar* dst) {
    dst[0] = dst[width - 1] = clip[0];

    const int step = (int)image.step;
    const int up = y > 0 ? -step : 0;
    const int down = y < image.rows - 1 ? step : 0;
    const uchar* row = image.ptr<uchar>(y);

    if (cn == 1) {
        for (int x = 1; x < width - 1; ++x) {
            // 2*horizontal + vertical-above + vertical-below, all as forward
            // differences; the raw pixel goes into the second layer instead.
            dst[x] = clip[(row[x + 1] - row[x - 1]) * 2 + row[x + up + 1] - row[x + up - 1] +
                          row[x + down + 1] - row[x + down - 1]];
        }
        return;
    }
    // Multi-channel input is rejected by computeDisparity(); the plane layout
    // here is only kept so the border handling stays explicit.
    for (int c = 0; c < cn; ++c) {
        for (int x = 1; x < width - 1; ++x) {
            const int xc = x * cn + c;
            dst[c * width + x] =
                clip[(row[xc + cn] - row[xc - cn]) * 2 + row[xc + up + cn] - row[xc + up - cn] +
                     row[xc + down + cn] - row[xc + down - cn]];
        }
    }
}

void mirrorInPlace(uchar* plane, int width) {
    for (int x = 0, y = width - 1; x < y; ++x, --y) std::swap(plane[x], plane[y]);
}

}  // namespace

// Fills pixelCost for image row `y`: for every column x in [minX1, maxX1) and
// every label d in [0, numDisp), the sum of the two BT cost layers between
// left(x, y) and right(x - minDisp - d, y).
//
// Gradients on the right image are taken in mirrored column order, which is how
// the matching loop is able to read `right` with a single increasing index. The
// min/max of a 3-tap window is invariant under that mirror, so the cost is not
// affected by doing it in either order.
void fillPixelCostRow(Workspace& ws, const cv::Mat& left, const cv::Mat& right, int y) {
    const Geometry& g = ws.g;
    const int width = g.width;
    const int cn = ws.channels;

    uchar* leftLayer0 = ws.prefilterScratch.data();
    uchar* leftLayer1 = leftLayer0 + width * cn;
    uchar* rightLayer0 = leftLayer1 + width * cn;
    uchar* rightLayer1 = rightLayer0 + width * cn;

    buildLayer(left, y, cn, width, ws.clipTable.centered(), leftLayer0);
    if (cn == 1) {
        const uchar* row = left.ptr<uchar>(y);
        for (int x = 1; x < width - 1; ++x) leftLayer1[x] = row[x];
        // The end columns keep the table's zero point, not the edge pixel: the
        // raw layer is initialised the same way as the derivative layer and only
        // [1, width-1) is overwritten.
        leftLayer1[0] = leftLayer1[width - 1] = ws.clipTable.centered()[0];
    }

    buildLayer(right, y, cn, width, ws.clipTable.centered(), rightLayer0);
    if (cn == 1) {
        const uchar* row = right.ptr<uchar>(y);
        for (int x = 1; x < width - 1; ++x) rightLayer1[x] = row[x];
        rightLayer1[0] = rightLayer1[width - 1] = ws.clipTable.centered()[0];
    }

    for (int c = 0; c < cn; ++c) {
        mirrorInPlace(rightLayer0 + c * width, width);
        mirrorInPlace(rightLayer1 + c * width, width);
    }

    std::memset(ws.pixelCost.data(), 0, ws.pixelCost.size() * sizeof(short));

    // Three-tap min/max of the mirrored right image, computed once per layer.
    uchar* rightMin = ws.prefilterScratch.data() + 4 * width * cn;
    uchar* rightMax = rightMin + width;

    for (int layer = 0; layer < 2; ++layer) {
        const int shift = layer == 0 ? 0 : 2;
        const uchar* lRef = (layer == 0 ? leftLayer0 : leftLayer1);
        const uchar* rRef = (layer == 0 ? rightLayer0 : rightLayer1);

        for (int x = 0; x < width; ++x) {
            const int v = rRef[x];
            const int vl = x > 0 ? (v + rRef[x - 1]) / 2 : v;
            const int vr = x < width - 1 ? (v + rRef[x + 1]) / 2 : v;
            rightMin[x] = (uchar)std::min(std::min(vl, vr), v);
            rightMax[x] = (uchar)std::max(std::max(vl, vr), v);
        }

        for (int x = g.minX1; x < g.maxX1; ++x) {
            const int u = lRef[x];
            const int ul = x > 0 ? (u + lRef[x - 1]) / 2 : u;
            const int ur = x < width - 1 ? (u + lRef[x + 1]) / 2 : u;
            const int uMin = std::min(std::min(ul, ur), u);
            const int uMax = std::max(std::max(ul, ur), u);

            short* dst = ws.pixelCost.data() + (std::size_t)(x - g.minX1) * g.numDisp;
            for (int d = 0; d < g.numDisp; ++d) {
                const int col = width - 1 - x + g.minDisp + d;
                const int v = rRef[col];
                int leftAbove = std::max(0, u - rightMax[col]);
                leftAbove = std::max(leftAbove, rightMin[col] - u);
                int rightAbove = std::max(0, v - uMax);
                rightAbove = std::max(rightAbove, uMin - v);
                dst[d] = add16(dst[d], (short)(std::min(leftAbove, rightAbove) >> shift));
            }
        }
    }
}

void Workspace::maybeCapturePixelCost(int row) {
    if (!trace || !trace->capturePixDiff) return;
    if (!trace->rows.empty()) {
        const bool wanted = std::any_of(trace->rows.begin(), trace->rows.end(),
                                        [row](const DebugRow& r) { return r.y == row; });
        if (!wanted) return;
    }
    std::size_t used = 0;
    for (const auto& v : trace->pixDiff) used += v.size() * sizeof(short);
    const std::size_t bytes = pixelCost.size() * sizeof(short);
    if (used + bytes > trace->budget) return;
    trace->pixDiffRows.push_back(row);
    trace->pixDiff.push_back(pixelCost);
}

}  // namespace internal
}  // namespace sgbm_repro
