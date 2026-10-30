// Phase 0: decide whether "bit-exact with cv::StereoSGBM" is a well-posed
// target on this OpenCV build.
//
// Three questions, in order of how much they would cost us to get wrong:
//
//  1. Does cv::setUseOptimized(false) change the MODE_SGBM output? If yes, the
//     "reference" depends on a runtime toggle and we would have to pin it.
//  2. Does the oracle transcription reproduce cv::StereoSGBM byte for byte?
//     If no, the transcription is wrong and every later comparison is noise.
//  3. Does the Da = alignSize(D, vlanes) row padding leak into the result?
//     If yes, the reimplementation has to inherit an OpenCV-internal constant.
//
// The probe prints one line per case; the exit code is non-zero only when an
// answer invalidates the plan.
#include "sgbm_oracle.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/stereo.hpp>

#include <algorithm>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <random>

namespace {

struct Case {
    std::string name;
    int width;
    int height;
    int minDisparity;
    int numDisparities;
    int blockSize;
    int preFilterCap;
    int uniquenessRatio;
    int speckleWindowSize;
    int speckleRange;
    int disp12MaxDiff;
    int kind;  // 0 constant, 1 ramp, 2 stripes, 3 noise, 4 checker
};

std::vector<Case> cases() {
    std::vector<Case> v;
    const int sizes[][2] = {{64, 48}, {128, 96}, {206, 131}, {640, 480}, {97, 71}};
    const int disps[][2] = {{0, 16}, {0, 32}, {32, 96}, {0, 48}};
    const int blocks[] = {3, 5, 11};
    for (const auto& sz : sizes) {
        for (const auto& dp : disps) {
            for (int bs : blocks) {
                if (sz[0] - (dp[0] + dp[1]) <= bs / 2) continue;
                Case c;
                c.width = sz[0];
                c.height = sz[1];
                c.minDisparity = dp[0];
                c.numDisparities = dp[1];
                c.blockSize = bs;
                c.preFilterCap = 31;
                c.uniquenessRatio = 5;
                c.speckleWindowSize = 40;
                c.speckleRange = 1;
                c.disp12MaxDiff = 1;
                c.kind = (int)(v.size() % 5);
                char buf[128];
                std::snprintf(buf, sizeof buf, "%dx%d_d%d+%d_bs%d_k%d", c.width, c.height,
                              c.minDisparity, c.numDisparities, c.blockSize, c.kind);
                c.name = buf;
                v.push_back(c);
            }
        }
    }
    return v;
}

// Left/right pair with a known disparity, so the case exercises real matches
// rather than two unrelated images.
void makePair(const Case& c, cv::Mat& left, cv::Mat& right) {
    left.create(c.height, c.width, CV_8UC1);
    right.create(c.height, c.width, CV_8UC1);
    std::mt19937 rng(12345u + (unsigned)c.kind * 7u + (unsigned)c.width);

    // Right view = left view shifted left by a fixed disparity, so the pair is
    // actually related. Drawing both independently would make the pair
    // degenerate and the comparison meaningless.
    const int pad = c.minDisparity + c.numDisparities + 8;
    cv::Mat base(c.height, c.width + pad, CV_8UC1);
    for (int y = 0; y < base.rows; ++y) {
        uchar* b = base.ptr<uchar>(y);
        for (int x = 0; x < base.cols; ++x) {
            int v;
            switch (c.kind) {
                case 0: v = 128; break;
                case 1: v = (x * 255) / std::max(base.cols - 1, 1); break;
                case 2: v = (y / 4) % 2 ? 200 : 40; break;
                case 3: v = 128 + (int)(rng() % 61) - 30; break;
                case 4: v = ((x / 8) + (y / 8)) % 2 ? 220 : 30; break;
                default: v = 100; break;
            }
            b[x] = (uchar)std::min(255, std::max(0, v));
        }
    }
    const int shift = (c.minDisparity + c.numDisparities) / 2;
    for (int y = 0; y < c.height; ++y) {
        std::memcpy(left.ptr<uchar>(y), base.ptr<uchar>(y), c.width);
        std::memcpy(right.ptr<uchar>(y), base.ptr<uchar>(y) + shift, c.width);
    }
}

oracle::Params toOracle(const Case& c) {
    oracle::Params p;
    p.minDisparity = c.minDisparity;
    p.numDisparities = c.numDisparities;
    p.SADWindowSize = c.blockSize;
    p.P1 = 8 * c.blockSize * c.blockSize;
    p.P2 = 256 * c.blockSize * c.blockSize;
    p.disp12MaxDiff = c.disp12MaxDiff;
    p.preFilterCap = c.preFilterCap;
    p.uniquenessRatio = c.uniquenessRatio;
    p.speckleWindowSize = c.speckleWindowSize;
    p.speckleRange = c.speckleRange;
    p.mode = 0;
    return p;
}

struct Diff {
    long pixels = 0;
    long differing = 0;
    int maxAbs = 0;
    int firstX = -1;
    int firstY = -1;
};

Diff diffMat(const cv::Mat& a, const cv::Mat& b) {
    Diff d;
    if (a.size() != b.size() || a.type() != b.type()) {
        d.differing = -1;
        return d;
    }
    for (int y = 0; y < a.rows; ++y) {
        const short* pa = a.ptr<short>(y);
        const short* pb = b.ptr<short>(y);
        for (int x = 0; x < a.cols; ++x) {
            ++d.pixels;
            if (pa[x] != pb[x]) {
                ++d.differing;
                const int ad = std::abs((int)pa[x] - (int)pb[x]);
                if (ad > d.maxAbs) d.maxAbs = ad;
                if (d.firstX < 0) {
                    d.firstX = x;
                    d.firstY = y;
                }
            }
        }
    }
    return d;
}

cv::Mat runOpenCV(const cv::Mat& l, const cv::Mat& r, const Case& c) {
    cv::Ptr<cv::StereoSGBM> m = cv::StereoSGBM::create(
        c.minDisparity, c.numDisparities, c.blockSize, 8 * c.blockSize * c.blockSize,
        256 * c.blockSize * c.blockSize, c.disp12MaxDiff, c.preFilterCap, c.uniquenessRatio,
        c.speckleWindowSize, c.speckleRange, cv::StereoSGBM::MODE_SGBM);
    cv::Mat d;
    m->compute(l, r, d);
    return d;
}

}  // namespace

int main(int argc, char** argv) {
    const bool verbose = argc > 1 && std::string(argv[1]) == "-v";

    std::printf("OpenCV %s\n", CV_VERSION);
    std::printf("int16 lanes            : %d\n", oracle::simdInt16Lanes());
    std::printf("useOptimized default   : %s\n", cv::useOptimized() ? "true" : "false");
    std::printf("\n");

    const std::vector<Case> list = cases();
    std::printf("%-30s %10s %10s %10s %10s\n", "case", "opt!=noopt", "oracle!=cv", "pad16!=orr", "pad!=orr");
    std::printf("%-30s %10s %10s %10s %10s\n", "", "(pixels)", "(pixels)", "(pixels)", "(pixels)");

    int q1_bad = 0, q2_bad = 0, q3_bad = 0;
    long q1_px = 0, q2_px = 0, q3_px = 0, q1_tot = 0;

    for (const Case& c : list) {
        cv::Mat left, right;
        makePair(c, left, right);

        cv::setUseOptimized(true);
        const cv::Mat dOpt = runOpenCV(left, right, c);
        cv::setUseOptimized(false);
        const cv::Mat dNoOpt = runOpenCV(left, right, c);
        cv::setUseOptimized(true);

        const Diff q1 = diffMat(dOpt, dNoOpt);

        // The oracle now runs DP + epilogue, i.e. the whole of
        // cv::StereoSGBM::compute().
        oracle::StageDump full = oracle::makeStageDump(oracle::DumpConfig{}, c.height);
        cv::Mat dOracle;
        oracle::computeDisparitySGBM(left, right, dOracle, toOracle(c), &full);
        const Diff q2 = diffMat(dOracle, dOpt);

        // Q3: how much the epilogue moves the result, i.e. how much of the
        // bit-exactness work sits behind medianBlur/filterSpeckles.
        const Diff q3 = diffMat(full.disparityBeforeEpilogue, full.disparity);

        std::printf("%-30s %10ld %10ld %10ld %10ld\n", c.name.c_str(), q1.differing, q2.differing,
                    q3.differing, (long)full.disparityBeforeEpilogue.total());

        q1_tot += q1.pixels;
        q1_px += q1.differing;
        q2_px += q2.differing;
        q3_px += q3.differing;
        if (q1.differing != 0) ++q1_bad;
        if (q2.differing != 0) {
            ++q2_bad;
            if (verbose || q2_bad <= 5) {
                std::printf("    oracle mismatch: %ld px, max|d|=%d, first at (%d,%d)\n", q2.differing,
                            q2.maxAbs, q2.firstX, q2.firstY);
            }
        }
        if (q3.differing != 0) ++q3_bad;
    }

    std::printf("\n--- summary over %zu cases ---\n", list.size());
    std::printf("Q1 setUseOptimized(true) vs (false) : %d cases differ, %ld/%ld px\n", q1_bad, q1_px,
                q1_tot);
    std::printf("Q2 oracle vs cv::StereoSGBM         : %d cases differ, %ld px total\n", q2_bad,
                q2_px);
    std::printf("Q3 before vs after epilogue         : %d cases differ, %ld px total\n", q3_bad,
                q3_px);
    std::printf("\nDa alignment: numDisparities -> Da = alignSize(numDisparities, %d)\n",
                oracle::simdInt16Lanes());

    int rc = 0;
    if (q1_bad != 0) {
        std::printf("\nVERDICT: FAIL - reference depends on setUseOptimized; pin it before comparing.\n");
        rc = 1;
    } else if (q2_bad != 0) {
        std::printf("\nVERDICT: FAIL - oracle transcription does not reproduce cv::StereoSGBM.\n");
        rc = 1;
    } else {
        std::printf("\nVERDICT: PASS - reference is unique and the oracle reproduces it exactly.\n");
    }
    return rc;
}
