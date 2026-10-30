// Pipeline orchestration.
#include "sgbm_repro/sgbm.h"

#include "internal.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace sgbm_repro {

using namespace internal;

Geometry resolveGeometry(int width, int height, const Params& params) {
    if (width <= 0 || height <= 0) throw std::invalid_argument("image is empty");
    if (params.numDisparities <= 0) throw std::invalid_argument("numDisparities must be positive");
    if (params.blockSize <= 0) throw std::invalid_argument("blockSize must be positive");

    Geometry g;
    g.width = width;
    g.height = height;
    g.minDisp = params.minDisparity;
    g.numDisp = params.numDisparities;
    g.maxDisp = g.minDisp + g.numDisp;
    g.minX1 = std::max(g.maxDisp, 0);
    g.maxX1 = width + std::min(g.minDisp, 0);
    g.width1 = g.maxX1 - g.minX1;

    g.SW2 = params.blockSize / 2;
    g.SH2 = params.blockSize / 2;

    g.P1 = params.P1 > 0 ? params.P1 : 2;
    g.P2 = std::max(params.P2 > 0 ? params.P2 : 5, g.P1 + 1);
    g.uniquenessRatio = params.uniquenessRatio >= 0 ? params.uniquenessRatio : 10;
    g.disp12MaxDiff = params.disp12MaxDiff > 0 ? params.disp12MaxDiff : 1;

    g.preFilterCap = params.preFilterCap;

    g.invalidDisp = g.minDisp - 1;
    g.invalidDispScaled = g.invalidDisp * kDispScale;

    // The window must fit inside the region that has a full disparity range.
    if (width - g.maxDisp <= g.SW2)
        throw std::invalid_argument(
            "image too narrow for the requested numDisparities and blockSize");
    if (g.width1 <= 0) throw std::invalid_argument("image does not leave any matchable column");

    return g;
}

void computeDisparity(const cv::Mat& left, const cv::Mat& right, cv::Mat& disparity,
                      const Params& params, DebugTrace* trace) {
    if (left.type() != CV_8UC1 || right.type() != CV_8UC1)
        throw std::invalid_argument("expected single-channel 8-bit images");
    if (left.size() != right.size()) throw std::invalid_argument("left and right sizes differ");

    const Geometry g = resolveGeometry(left.cols, left.rows, params);
    if (disparity.size() != left.size() || disparity.type() != CV_16S)
        disparity.create(left.size(), CV_16S);

    Workspace ws(g, 1, left.cols);

    if (trace) trace->geometry = g;
    ws.trace = trace;

    // Slot `currentId` receives this row; the previous row lives in the other
    // slot, except for direction 0 which is read from and written to the same
    // slot one column behind.
    int previousId = 1;
    int currentId = 0;
    ws.resetDirectional();
    // The cost buffer starts at P2 rather than zero: the recurrence below adds
    // P2 on the first row and subtracts it back per direction, which saves an
    // add in the inner loop. Reproducing the initial value is required, not
    // merely tidy -- the saturating arithmetic makes the starting point visible.
    std::fill(ws.cost.begin(), ws.cost.end(), narrow16(g.P2));

    for (int y = 0; y < g.height; ++y) {
        short* dispRow = disparity.ptr<short>(y);

        accumulateCostRow(ws, left, right, y, y == 0);

        if (trace) {
            for (auto& row : trace->rows) {
                if (row.y != y) continue;
                row.cost.assign(ws.costRow(), ws.costRow() + ws.cost.size());
            }
        }

        ws.zeroAggregate();
        forwardSweep(ws, previousId, currentId, y);

        if (trace) {
            for (auto& row : trace->rows) {
                if (row.y != y) continue;
                row.aggregateForward.assign(ws.aggregateRow(),
                                            ws.aggregateRow() + ws.aggregate.size());
            }
        }

        reverseSweepAndSelect(ws, currentId, y, dispRow);

        if (trace) {
            for (auto& row : trace->rows) {
                if (row.y != y) continue;
                row.aggregate.assign(ws.aggregateRow(), ws.aggregateRow() + ws.aggregate.size());
            }
        }

        std::swap(previousId, currentId);
    }

    if (trace) trace->disparityBeforeEpilogue = disparity.clone();

    medianBlur3(disparity);
    if (params.speckleWindowSize > 0)
        filterSpeckles(disparity, g.invalidDispScaled, params.speckleWindowSize,
                       kDispScale * params.speckleRange);

    if (trace) trace->disparity = disparity.clone();
}

}  // namespace sgbm_repro
