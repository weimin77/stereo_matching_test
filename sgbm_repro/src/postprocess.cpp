// Epilogue: 3x3 median followed by speckle removal.
//
// Both operate on the CV_16S fixed-point disparity, so the ordering of the two
// stages matters and neither can be folded into the DP.
#include "sgbm_repro/sgbm.h"

#include <algorithm>
#include <cstdlib>
#include <vector>

namespace sgbm_repro {

// 3x3 median with edge replication. The median of nine values is unique, so any
// correct selection network agrees with any other; no particular sort order is
// being reproduced here.
void medianBlur3(cv::Mat& disparity) {
    if (disparity.empty()) return;
    const int width = disparity.cols;
    const int height = disparity.rows;
    cv::Mat src = disparity.clone();

    for (int y = 0; y < height; ++y) {
        short* dst = disparity.ptr<short>(y);
        const short* r0 = src.ptr<short>(std::max(y - 1, 0));
        const short* r1 = src.ptr<short>(y);
        const short* r2 = src.ptr<short>(std::min(y + 1, height - 1));
        for (int x = 0; x < width; ++x) {
            const int x0 = x > 0 ? x - 1 : 0;
            const int x2 = x + 1 < width ? x + 1 : x;
            short v[9] = {r0[x0], r0[x], r0[x2], r1[x0], r1[x], r1[x2], r2[x0], r2[x], r2[x2]};
            std::nth_element(v, v + 4, v + 9);
            dst[x] = v[4];
        }
    }
}

// Replaces every 4-connected region of at most maxSpeckleSize pixels whose
// neighbouring disparities differ by no more than maxDiff with newValue.
//
// Pixels already equal to newValue are the "invalid" marker and are never part
// of a region, which is also what keeps the result independent of the order the
// regions are visited in.
void filterSpeckles(cv::Mat& disparity, int newValue, int maxSpeckleSize, int maxDiff) {
    if (disparity.empty() || maxSpeckleSize <= 0) return;
    const int width = disparity.cols;
    const int height = disparity.rows;
    const int total = width * height;

    std::vector<int> labels((std::size_t)total, 0);
    std::vector<int> stack;
    stack.reserve((std::size_t)total);
    std::vector<int> members;
    members.reserve(1024);

    int nextLabel = 0;
    for (int y = 0; y < height; ++y) {
        short* row = disparity.ptr<short>(y);
        for (int x = 0; x < width; ++x) {
            const int seed = y * width + x;
            if (row[x] == newValue || labels[seed]) continue;

            ++nextLabel;
            members.clear();
            stack.clear();
            stack.push_back(seed);
            labels[seed] = nextLabel;

            while (!stack.empty()) {
                const int index = stack.back();
                stack.pop_back();
                members.push_back(index);
                const int cy = index / width;
                const int cx = index - cy * width;
                const short cv = disparity.ptr<short>(cy)[cx];

                auto visit = [&](int nx, int ny) {
                    if (nx < 0 || nx >= width || ny < 0 || ny >= height) return;
                    const int ni = ny * width + nx;
                    if (labels[ni]) return;
                    const short nv = disparity.ptr<short>(ny)[nx];
                    if (nv == newValue) return;
                    if (std::abs((int)cv - (int)nv) > maxDiff) return;
                    labels[ni] = nextLabel;
                    stack.push_back(ni);
                };
                visit(cx, cy + 1);
                visit(cx, cy - 1);
                visit(cx + 1, cy);
                visit(cx - 1, cy);
            }

            if ((int)members.size() <= maxSpeckleSize) {
                for (int index : members) {
                    const int my = index / width;
                    disparity.ptr<short>(my)[index - my * width] = (short)newValue;
                }
            }
        }
    }
}

}  // namespace sgbm_repro
