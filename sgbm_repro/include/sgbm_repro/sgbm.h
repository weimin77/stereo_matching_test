// Public interface of the SGBM reimplementation.
//
// The goal is byte-for-byte agreement with cv::StereoSGBM in MODE_SGBM on an
// x86 build, i.e. the same CV_16S x16 fixed-point disparity map for the same
// (minDisparity, numDisparities, blockSize, P1, P2, disp12MaxDiff,
// preFilterCap, uniquenessRatio, speckleWindowSize, speckleRange). See
// ../README.md for why the agreement is defined against that build and not
// against OpenCV's scalar fallback.
#ifndef SGBM_REPRO_INCLUDE_SGBM_REPRO_SGBM_H
#define SGBM_REPRO_INCLUDE_SGBM_REPRO_SGBM_H

#include <opencv2/core.hpp>

#include <cstddef>
#include <vector>

namespace sgbm_repro {

// Shape of the cost volume / shared buffers, resolved from the user parameters
// the same way OpenCV's computeDisparitySGBM() does.
struct Geometry {
    int width = 0;
    int height = 0;
    // Disparity range.
    int minDisp = 0;
    int numDisp = 0;
    int maxDisp = 0;
    // Columns of the cost volume that have a full disparity range.
    int width1 = 0;
    int minX1 = 0;
    int maxX1 = 0;
    // Half window.
    int SW2 = 0;
    int SH2 = 0;
    // Effective penalties (P1/P2 defaults applied, P2 >= P1+1).
    int P1 = 0;
    int P2 = 0;
    int uniquenessRatio = 10;
    int disp12MaxDiff = 1;
    // Only the prefilter needs this, but it belongs with the other resolved
    // parameters that shape the cost volume.
    int preFilterCap = 0;
    // Invalid marker written for pixels with no valid disparity.
    int invalidDisp = 0;
    int invalidDispScaled = 0;
};

struct Params {
    int minDisparity = 0;
    int numDisparities = 0;
    int blockSize = 5;
    int P1 = 0;   // 0 -> 2
    int P2 = 0;   // 0 -> max(5, P1+1)
    int disp12MaxDiff = 0;  // <= 0 -> 1
    int preFilterCap = 0;
    int uniquenessRatio = 0;  // < 0 -> 10
    int speckleWindowSize = 0;
    int speckleRange = 0;
};

// Throws std::invalid_argument when the parameters cannot describe a valid run.
Geometry resolveGeometry(int width, int height, const Params& params);

// Optional per-stage capture, used by the comparison harness. Mirrors the
// buffers the pipeline actually computes so a failure can be attributed to the
// cost, the aggregation, the DP or the epilogue.
struct DebugOptions {
    std::vector<int> rows;                        // rows to capture; empty = none
    bool capturePixDiff = false;                  // capture raw BT cost rows
    std::size_t pixDiffByteBudget = std::size_t(256) << 20;
};

struct DebugRow {
    int y = 0;
    // Row cost after box aggregation, width1 entries per disparity.
    std::vector<short> cost;
    // Aggregate after the four forward directions only.
    std::vector<short> aggregateForward;
    // Aggregate after the reverse direction too.
    std::vector<short> aggregate;
};

struct DebugTrace {
    Geometry geometry;
    bool capturePixDiff = false;
    std::size_t budget = 0;
    std::vector<int> pixDiffRows;
    std::vector<std::vector<short>> pixDiff;
    std::vector<DebugRow> rows;
    // Disparity before / after the median and speckle epilogue.
    cv::Mat disparityBeforeEpilogue;
    cv::Mat disparity;
};

// Computes disparity for `left` against `right`. Both must be CV_8UC1 and the
// same size; `disparity` becomes CV_16S holding x16 fixed point, with
// (minDisparity-1)*16 marking pixels that have no valid match.
void computeDisparity(const cv::Mat& left, const cv::Mat& right, cv::Mat& disparity,
                      const Params& params, DebugTrace* trace = nullptr);

// Exposed for direct testing of the epilogue.
void medianBlur3(cv::Mat& disparity);
void filterSpeckles(cv::Mat& disparity, int newValue, int maxSpeckleSize, int maxDiff);

}  // namespace sgbm_repro

#endif  // SGBM_REPRO_INCLUDE_SGBM_REPRO_SGBM_H
