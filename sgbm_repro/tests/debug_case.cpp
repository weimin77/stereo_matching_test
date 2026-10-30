// Ablation harness for bringing the oracle transcription into agreement with
// cv::StereoSGBM. Kept in the tree because the same ablations are useful when
// the reimplementation (src/) misbehaves.
//
// Each ablation removes one stage from both sides, so a mismatch that survives
// an ablation cannot have been caused by the removed stage.
#include "sgbm_oracle.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/stereo.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

struct Spec {
    int width = 206, height = 131;
    int minDisparity = 0, numDisparities = 32;
    int blockSize = 11;
    int preFilterCap = 31, uniquenessRatio = 5;
    int speckleWindowSize = 40, speckleRange = 1, disp12MaxDiff = 1;
    int kind = 3;  // 0 constant, 1 ramp, 2 stripes, 3 noise, 4 checker
};

// Renders one base image, then derives the right view by shifting it, so the
// two views are actually related. Generating them independently (as an earlier
// version did) produces a degenerate pair where any cost aggregation is
// arbitrary and the comparison means nothing.
void makePair(const Spec& s, cv::Mat& left, cv::Mat& right) {
    const int pad = s.minDisparity + s.numDisparities + 8;
    cv::Mat base(s.height, s.width + pad, CV_8UC1);
    std::mt19937 rng(12345u + (unsigned)s.kind * 7u + (unsigned)s.width);
    for (int y = 0; y < base.rows; ++y) {
        uchar* b = base.ptr<uchar>(y);
        for (int x = 0; x < base.cols; ++x) {
            int v;
            switch (s.kind) {
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
    const int shift = (s.minDisparity + s.numDisparities) / 2;
    left.create(s.height, s.width, CV_8UC1);
    right.create(s.height, s.width, CV_8UC1);
    for (int y = 0; y < s.height; ++y) {
        std::memcpy(left.ptr<uchar>(y), base.ptr<uchar>(y), s.width);
        std::memcpy(right.ptr<uchar>(y), base.ptr<uchar>(y) + shift, s.width);
    }
}

oracle::Params toOracle(const Spec& s) {
    oracle::Params p;
    p.minDisparity = s.minDisparity;
    p.numDisparities = s.numDisparities;
    p.SADWindowSize = s.blockSize;
    p.P1 = 8 * s.blockSize * s.blockSize;
    p.P2 = 256 * s.blockSize * s.blockSize;
    p.disp12MaxDiff = s.disp12MaxDiff;
    p.preFilterCap = s.preFilterCap;
    p.uniquenessRatio = s.uniquenessRatio;
    p.speckleWindowSize = s.speckleWindowSize;
    p.speckleRange = s.speckleRange;
    return p;
}

cv::Mat runOpenCV(const cv::Mat& l, const cv::Mat& r, const Spec& s) {
    cv::Ptr<cv::StereoSGBM> m = cv::StereoSGBM::create(
        s.minDisparity, s.numDisparities, s.blockSize, 8 * s.blockSize * s.blockSize,
        256 * s.blockSize * s.blockSize, s.disp12MaxDiff, s.preFilterCap, s.uniquenessRatio,
        s.speckleWindowSize, s.speckleRange, cv::StereoSGBM::MODE_SGBM);
    cv::Mat d;
    m->compute(l, r, d);
    return d;
}

struct Stats {
    long differing = 0;
    long aInvalid = 0, bInvalid = 0, bothInvalid = 0;
    int maxAbs = 0;
    double meanAbs = 0;
    long hist[8] = {0, 0, 0, 0, 0, 0, 0, 0};
};

Stats compare(const cv::Mat& a, const cv::Mat& b, int invalid) {
    Stats s;
    long sum = 0;
    for (int y = 0; y < a.rows; ++y) {
        const short* pa = a.ptr<short>(y);
        const short* pb = b.ptr<short>(y);
        for (int x = 0; x < a.cols; ++x) {
            const bool ia = pa[x] == invalid, ib = pb[x] == invalid;
            s.aInvalid += ia;
            s.bInvalid += ib;
            if (ia && ib) ++s.bothInvalid;
            if (pa[x] == pb[x]) continue;
            ++s.differing;
            const int d = std::abs((int)pa[x] - (int)pb[x]);
            sum += d;
            s.maxAbs = std::max(s.maxAbs, d);
            const int bucket = d < 16 ? 0 : d < 64 ? 1 : d < 256 ? 2 : d < 1024 ? 3 : d < 4096 ? 4 : 5;
            ++s.hist[bucket];
        }
    }
    s.meanAbs = s.differing ? (double)sum / s.differing : 0.0;
    return s;
}

void report(const char* label, const Stats& s, long total) {
    std::printf("  %-38s differ %7ld/%ld (%5.2f%%)  max|d| %5d  mean %7.2f  "
                "invA %6ld invB %6ld\n",
                label, s.differing, total, 100.0 * s.differing / total, s.maxAbs, s.meanAbs,
                s.aInvalid, s.bInvalid);
    if (s.differing) {
        std::printf("  %-38s buckets <16:%ld <64:%ld <256:%ld <1k:%ld <4k:%ld >=4k:%ld\n", "",
                    s.hist[0], s.hist[1], s.hist[2], s.hist[3], s.hist[4], s.hist[5]);
    }
}

}  // namespace

int main(int argc, char** argv) {
    Spec s;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto num = [&](const char* key, int& out) {
            const std::string k = std::string("-") + key + "=";
            if (a.rfind(k, 0) == 0) {
                out = std::atoi(a.c_str() + k.size());
                return true;
            }
            return false;
        };
        if (num("w", s.width)) continue;
        if (num("h", s.height)) continue;
        if (num("min", s.minDisparity)) continue;
        if (num("num", s.numDisparities)) continue;
        if (num("bs", s.blockSize)) continue;
        if (num("pfc", s.preFilterCap)) continue;
        if (num("ur", s.uniquenessRatio)) continue;
        if (num("sws", s.speckleWindowSize)) continue;
        if (num("sr", s.speckleRange)) continue;
        if (num("d12", s.disp12MaxDiff)) continue;
        if (num("kind", s.kind)) continue;
    }

    std::printf("case %dx%d min=%d num=%d bs=%d pfc=%d ur=%d sws=%d sr=%d d12=%d kind=%d\n",
                s.width, s.height, s.minDisparity, s.numDisparities, s.blockSize, s.preFilterCap,
                s.uniquenessRatio, s.speckleWindowSize, s.speckleRange, s.disp12MaxDiff, s.kind);

    cv::Mat left, right;
    makePair(s, left, right);
    const long total = left.total();

    // 1. full pipeline
    {
        const cv::Mat cvOut = runOpenCV(left, right, s);
        oracle::StageDump dump = oracle::makeStageDump(oracle::DumpConfig{}, s.height);
        cv::Mat myOut;
        oracle::computeDisparitySGBM(left, right, myOut, toOracle(s), &dump);
        const int invalid = (s.minDisparity - 1) * 16;
        report("full pipeline", compare(myOut, cvOut, invalid), total);

        // Per-row difference profile: a block-size-dependent failure shows up as
        // whole rows differing, which points at the vertical aggregation rather
        // than at the DP.
        {
            std::printf("  per-row differing pixels (row: count), first 24 rows with any:\n");
            int shown = 0;
            for (int y = 0; y < myOut.rows && shown < 24; ++y) {
                const short* a = myOut.ptr<short>(y);
                const short* b = cvOut.ptr<short>(y);
                int n = 0;
                for (int x = 0; x < myOut.cols; ++x) n += (a[x] != b[x]);
                if (n) {
                    std::printf("    y=%4d  %5d / %d\n", y, n, myOut.cols);
                    ++shown;
                }
            }
            // and the tail
            shown = 0;
            std::printf("  last 8 rows with any difference:\n");
            for (int y = myOut.rows - 1; y >= 0 && shown < 8; --y) {
                const short* a = myOut.ptr<short>(y);
                const short* b = cvOut.ptr<short>(y);
                int n = 0;
                for (int x = 0; x < myOut.cols; ++x) n += (a[x] != b[x]);
                if (n) {
                    std::printf("    y=%4d  %5d / %d\n", y, n, myOut.cols);
                    ++shown;
                }
            }
        }

        // 2. epilogue only: feed the oracle's own pre-epilogue map through both
        //    epilogues. Isolates medianBlur/filterSpeckles from the DP.
        cv::Mat a = dump.disparityBeforeEpilogue.clone();
        cv::Mat b = a.clone();
        oracle::epilogue(a, toOracle(s));
        // Same epilogue the OpenCV path applies, but on the oracle's DP output:
        cv::medianBlur(b, b, 3);
        if (s.speckleWindowSize > 0) {
            cv::Mat buf;
            cv::filterSpeckles(b, (s.minDisparity - 1) * 16, s.speckleWindowSize, 16 * s.speckleRange,
                               buf);
        }
        report("epilogue(dump) vs epilogue(cv)", compare(a, b, invalid), total);

        // 3. no speckle: medianBlur only
        {
            Spec s2 = s;
            s2.speckleWindowSize = 0;
            const cv::Mat cvOut2 = runOpenCV(left, right, s2);
            oracle::StageDump d2 = oracle::makeStageDump(oracle::DumpConfig{}, s.height);
            cv::Mat myOut2;
            oracle::computeDisparitySGBM(left, right, myOut2, toOracle(s2), &d2);
            report("no speckle: full pipeline", compare(myOut2, cvOut2, invalid), total);
        }

        // 4. save for inspection
        cv::imwrite("/tmp/oracle_dbg_cv.png", cvOut);
        cv::imwrite("/tmp/oracle_dbg_my.png", myOut);
        cv::imwrite("/tmp/oracle_dbg_mine_dp.png", dump.disparityBeforeEpilogue);
    }

    std::printf("\nrows captured: %zu, pixDiff rows: %zu\n", size_t(0), size_t(0));
    return 0;
}
