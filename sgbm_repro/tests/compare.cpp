// Per-stage comparison of sgbm_repro against the OpenCV transcription and
// against cv::StereoSGBM itself.
//
// Stage granularity is what makes a mismatch actionable:
//   pixel cost      -> prefilter / BT cost
//   cost            -> horizontal + vertical box aggregation
//   aggregateForward-> the four directional passes
//   aggregate       -> the fifth direction
//   pre-epilogue    -> label selection, sub-pixel fit, consistency test
//   final           -> median / speckle epilogue
#include "sgbm_repro/sgbm.h"
#include "sgbm_oracle.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/stereo.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

struct Case {
    std::string name;
    int width = 0;
    int height = 0;
    int minDisparity = 0;
    int numDisparities = 0;
    int blockSize = 5;
    int preFilterCap = 31;
    int uniquenessRatio = 5;
    int speckleWindowSize = 40;
    int speckleRange = 1;
    int disp12MaxDiff = 1;
    int pattern = 0;  // 0 constant, 1 ramp, 2 stripes, 3 noise, 4 checker
    int noiseScale = 60;
};

void makePair(const Case& c, cv::Mat& left, cv::Mat& right) {
    const int pad = c.minDisparity + c.numDisparities + 8;
    cv::Mat base(c.height, c.width + pad, CV_8UC1);
    std::mt19937 rng(9001u + (unsigned)c.pattern * 31u + (unsigned)c.width);
    for (int y = 0; y < base.rows; ++y) {
        uchar* b = base.ptr<uchar>(y);
        for (int x = 0; x < base.cols; ++x) {
            int v;
            switch (c.pattern) {
                case 0: v = 128; break;
                case 1: v = (x * 255) / std::max(base.cols - 1, 1); break;
                case 2: v = (y / 4) % 2 ? 200 : 40; break;
                case 3: v = 128 + (int)(rng() % (unsigned)c.noiseScale) - c.noiseScale / 2; break;
                case 4: v = ((x / 8) + (y / 8)) % 2 ? 220 : 30; break;
                default: v = 100; break;
            }
            b[x] = (uchar)std::min(255, std::max(0, v));
        }
    }
    // A depth ramp on top of the pattern, so the true disparity varies with x and
    // the cases exercise more than one label.
    const int shift = (c.minDisparity + c.numDisparities) / 2;
    left.create(c.height, c.width, CV_8UC1);
    right.create(c.height, c.width, CV_8UC1);
    for (int y = 0; y < c.height; ++y) {
        for (int x = 0; x < c.width; ++x) {
            const int extra = (c.pattern == 3 || c.pattern == 4) ? 0 : (x / 64);
            const int lx = std::min(x + extra, base.cols - 1);
            const int rx = std::min(x + shift + extra, base.cols - 1);
            left.ptr<uchar>(y)[x] = base.ptr<uchar>(y)[lx];
            right.ptr<uchar>(y)[x] = base.ptr<uchar>(y)[rx];
        }
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
    return p;
}

sgbm_repro::Params toRepro(const Case& c) {
    sgbm_repro::Params p;
    p.minDisparity = c.minDisparity;
    p.numDisparities = c.numDisparities;
    p.blockSize = c.blockSize;
    p.P1 = 8 * c.blockSize * c.blockSize;
    p.P2 = 256 * c.blockSize * c.blockSize;
    p.disp12MaxDiff = c.disp12MaxDiff;
    p.preFilterCap = c.preFilterCap;
    p.uniquenessRatio = c.uniquenessRatio;
    p.speckleWindowSize = c.speckleWindowSize;
    p.speckleRange = c.speckleRange;
    return p;
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

// Number of differing entries plus the largest magnitude, for buffers of shorts.
struct BufferDiff {
    long total = 0;
    long differing = 0;
    int maxAbs = 0;
    int firstIndex = -1;
};

BufferDiff diffBuffer(const std::vector<short>& a, const std::vector<short>& b) {
    BufferDiff d;
    if (a.size() != b.size()) {
        d.total = (long)std::max(a.size(), b.size());
        d.differing = -1;
        return d;
    }
    d.total = (long)a.size();
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i] == b[i]) continue;
        ++d.differing;
        d.maxAbs = std::max(d.maxAbs, std::abs((int)a[i] - (int)b[i]));
        if (d.firstIndex < 0) d.firstIndex = (int)i;
    }
    return d;
}

// OpenCV pads the cost row stride to alignSize(numDisparities, int16 lanes);
// this implementation does not. Compare the overlapping labels column by
// column so the padding is invisible to the test.
BufferDiff diffBufferStrided(const std::vector<short>& mine, const std::vector<short>& theirs,
                             int numDisp, int stride) {
    BufferDiff d;
    const int width1 = (int)(mine.size() / std::max(numDisp, 1));
    d.total = (long)mine.size();
    for (int x = 0; x < width1; ++x) {
        for (int label = 0; label < numDisp; ++label) {
            const short a = mine[(std::size_t)x * numDisp + label];
            const short b = theirs[(std::size_t)x * stride + label];
            if (a == b) continue;
            ++d.differing;
            d.maxAbs = std::max(d.maxAbs, std::abs((int)a - (int)b));
            if (d.firstIndex < 0) d.firstIndex = x * numDisp + label;
        }
    }
    return d;
}

BufferDiff diffMat(const cv::Mat& a, const cv::Mat& b) {
    BufferDiff d;
    if (a.size() != b.size() || a.type() != b.type()) {
        d.differing = -1;
        return d;
    }
    d.total = (long)a.total();
    for (int y = 0; y < a.rows; ++y) {
        const short* pa = a.ptr<short>(y);
        const short* pb = b.ptr<short>(y);
        for (int x = 0; x < a.cols; ++x) {
            if (pa[x] == pb[x]) continue;
            ++d.differing;
            d.maxAbs = std::max(d.maxAbs, std::abs((int)pa[x] - (int)pb[x]));
            if (d.firstIndex < 0) d.firstIndex = y * a.cols + x;
        }
    }
    return d;
}

struct Totals {
    long stages = 0;
    long casesBad = 0;
    long pixels = 0;
};

bool report(const char* stage, const BufferDiff& d, Totals& t, const char* caseName) {
    ++t.stages;
    t.pixels += d.total;
    if (d.differing == 0) return true;
    ++t.casesBad;
    std::printf("  FAIL %-18s %-34s differing %8ld/%-8ld max|d| %6d first %d\n", stage, caseName,
                d.differing, d.total, d.maxAbs, d.firstIndex);
    return false;
}

std::vector<Case> buildCases() {
    std::vector<Case> v;
    struct Size {
        int w, h;
    };
    const Size sizes[] = {{64, 48}, {128, 96}, {208, 128}, {206, 131}, {640, 480}, {97, 71}};
    struct Disp {
        int min, num;
    };
    const Disp disps[] = {{0, 16}, {0, 32}, {32, 96}, {0, 48}};
    const int blocks[] = {3, 5, 11};

    for (const auto& s : sizes) {
        for (const auto& d : disps) {
            for (int b : blocks) {
                if (s.w - (d.min + d.num) <= b / 2) continue;
                Case c;
                c.width = s.w;
                c.height = s.h;
                c.minDisparity = d.min;
                c.numDisparities = d.num;
                c.blockSize = b;
                c.pattern = (int)(v.size() % 5);
                char buf[128];
                std::snprintf(buf, sizeof buf, "%dx%d d%d+%d bs%d p%d", c.width, c.height,
                              c.minDisparity, c.numDisparities, c.blockSize, c.pattern);
                c.name = buf;
                v.push_back(c);
            }
        }
    }

    // A low-contrast noise case, the hardest one for the tie-breaking paths.
    {
        Case c;
        c.width = 320;
        c.height = 240;
        c.minDisparity = 0;
        c.numDisparities = 32;
        c.blockSize = 5;
        c.pattern = 3;
        c.noiseScale = 6;
        c.name = "320x240 d0+32 bs5 lowcontrast-noise";
        v.push_back(c);
    }

    // Targeted probes for the parts of the geometry that the grid above never
    // leaves alone: the top and bottom edge windows, the label-count padding,
    // and sign/zero corners of the parameter space.
    auto add = [&v](const char* name, int w, int h, int minD, int numD, int bs, int pattern,
                    int pfc, int ur, int sws, int sr, int d12) {
        Case c;
        c.width = w;
        c.height = h;
        c.minDisparity = minD;
        c.numDisparities = numD;
        c.blockSize = bs;
        c.pattern = pattern;
        c.preFilterCap = pfc;
        c.uniquenessRatio = ur;
        c.speckleWindowSize = sws;
        c.speckleRange = sr;
        c.disp12MaxDiff = d12;
        c.name = name;
        v.push_back(c);
    };

    // numDisparities not a multiple of the int16 lane count: OpenCV pads the cost
    // row stride to alignSize(numDisparities, lanes), this implementation does
    // not. If the padding leaked into the result these would diverge.
    add("pad d0+20 bs5", 96, 64, 0, 20, 5, 4, 31, 5, 40, 1, 1);
    add("pad d0+36 bs3", 128, 72, 0, 36, 3, 3, 31, 5, 40, 1, 1);
    add("pad d8+44 bs7", 160, 80, 8, 44, 7, 4, 31, 5, 40, 1, 1);

    // Negative minDisparity.
    add("negmin d-32+64 bs5", 192, 96, -32, 64, 5, 3, 31, 5, 40, 1, 1);
    add("negmin d-16+32 bs3", 128, 72, -16, 32, 3, 4, 31, 5, 40, 1, 1);

    // Block size corners.
    add("bs1", 96, 64, 0, 32, 1, 4, 31, 5, 40, 1, 1);
    add("bs4-even", 96, 64, 0, 32, 4, 4, 31, 5, 40, 1, 1);
    add("bs9", 160, 96, 0, 32, 9, 3, 31, 5, 40, 1, 1);

    // Aggregation weight / clipping corners.
    add("pfc0", 128, 72, 0, 32, 5, 4, 0, 5, 40, 1, 1);
    add("pfc63", 128, 72, 0, 32, 5, 4, 63, 5, 40, 1, 1);
    add("pfc200-bs11", 128, 72, 0, 32, 11, 4, 200, 5, 40, 1, 1);

    // Selection / consistency corners.
    add("ur0", 128, 72, 0, 32, 5, 3, 31, 0, 40, 1, 1);
    add("ur-neg", 128, 72, 0, 32, 5, 3, 31, -1, 40, 1, 1);
    add("ur40", 128, 72, 0, 32, 5, 3, 31, 40, 40, 1, 1);
    add("d12-0", 128, 72, 0, 32, 5, 3, 31, 5, 40, 1, 0);
    add("d12-64", 128, 72, 0, 32, 5, 3, 31, 5, 40, 1, 64);

    // Epilogue corners.
    add("sws0", 128, 72, 0, 32, 5, 3, 31, 5, 0, 1, 1);
    add("sws-large", 128, 72, 0, 32, 5, 3, 31, 5, 5000, 1, 1);
    add("ssr8", 128, 72, 0, 32, 5, 3, 31, 5, 40, 8, 1);

    // Heights that make the bottom-edge window dominate.
    add("h=bs", 96, 3, 0, 32, 3, 1, 31, 5, 40, 1, 1);
    add("h=bs+1", 96, 4, 0, 32, 3, 1, 31, 5, 40, 1, 1);
    add("h=bs+5", 96, 8, 0, 32, 3, 3, 31, 5, 40, 1, 1);
    add("tiny-w", 48, 32, 0, 16, 3, 3, 31, 5, 40, 1, 1);
    add("odd-both", 129, 97, 0, 32, 5, 3, 31, 5, 40, 1, 1);
    return v;
}

}  // namespace

int main(int argc, char** argv) {
    bool verbose = argc > 1 && std::string(argv[1]) == "-v";
    const bool one = argc > 1 && std::string(argv[1]) == "-one";

    const std::vector<Case> cases = buildCases();
    Totals pixelCost, costStage, fwdStage, aggStage, preStage, finalStage;
    long oracleVsCv = 0;

    for (const Case& c : cases) {
        cv::Mat left, right;
        makePair(c, left, right);

        const cv::Mat reference = runOpenCV(left, right, c);

        oracle::DumpConfig dc;
        dc.capturePixDiff = true;
        dc.rows = {0, 1, c.height / 2, c.height - 2, c.height - 1};
        oracle::StageDump od = oracle::makeStageDump(dc, c.height);
        // makeStageDump() now drops duplicates, so align the requested list with
        // what it actually kept.
        cv::Mat oracleOut;
        oracle::computeDisparitySGBM(left, right, oracleOut, toOracle(c), &od);

        sgbm_repro::DebugTrace rt;
        rt.capturePixDiff = true;
        rt.budget = dc.pixDiffByteBudget;
        for (int y : dc.rows) {
            bool seen = false;
            for (const auto& existing : rt.rows) seen |= (existing.y == y);
            if (seen) continue;
            sgbm_repro::DebugRow r;
            r.y = y;
            rt.rows.push_back(r);
        }
        cv::Mat reproOut;
        sgbm_repro::computeDisparity(left, right, reproOut, toRepro(c), &rt);

        // The transcription must still agree with the library, otherwise a
        // failure below cannot be attributed.
        if (diffMat(oracleOut, reference).differing != 0) {
            ++oracleVsCv;
            std::printf("  ORACLE DRIFT on %s\n", c.name.c_str());
        }

        report("oracle vs cv final", diffMat(oracleOut, reference), finalStage, c.name.c_str());
        report("repro vs cv final", diffMat(reproOut, reference), finalStage, c.name.c_str());
        report("repro vs oracle pre", diffMat(rt.disparityBeforeEpilogue, od.disparityBeforeEpilogue),
               preStage, c.name.c_str());

        // Per-row buffers.
        const int stride = od.Da;  // >= numDisparities; equals it unless padded
        if (rt.pixDiffRows == od.pixDiffRows) {
            for (std::size_t i = 0; i < rt.pixDiffRows.size(); ++i) {
                report("pixel cost",
                       diffBufferStrided(rt.pixDiff[i], od.pixDiff[i], rt.geometry.numDisp, stride),
                       pixelCost, c.name.c_str());
            }
        } else {
            std::printf("  FAIL pixel cost row set mismatch on %s\n", c.name.c_str());
            ++pixelCost.casesBad;
        }

        if (rt.rows.size() == od.rows.size()) {
            for (std::size_t i = 0; i < rt.rows.size(); ++i) {
                report("cost",
                       diffBufferStrided(rt.rows[i].cost, od.rows[i].C, rt.geometry.numDisp, stride),
                       costStage, c.name.c_str());
                report("aggregateForward",
                       diffBufferStrided(rt.rows[i].aggregateForward, od.rows[i].SForward,
                                         rt.geometry.numDisp, stride),
                       fwdStage, c.name.c_str());
                report("aggregate",
                       diffBufferStrided(rt.rows[i].aggregate, od.rows[i].S, rt.geometry.numDisp,
                                         stride),
                       aggStage, c.name.c_str());
            }
        } else {
            std::printf("  FAIL row set mismatch on %s\n", c.name.c_str());
            ++costStage.casesBad;
        }

        if (one) {
            // Drill into the first captured pixel-cost row and print the leading
            // disagreements with their (column, label) coordinates.
            std::printf("case %s  width1=%d D=%d\n", c.name.c_str(), rt.geometry.width1,
                        rt.geometry.numDisp);
            if (!rt.pixDiff.empty() && !od.pixDiff.empty()) {
                const int row = rt.pixDiffRows[0];
                const int D = rt.geometry.numDisp;
                const int width1 = rt.geometry.width1;
                int shown = 0;
                for (int i = 0; i < width1 * D && shown < 12; ++i) {
                    if (rt.pixDiff[0][i] == od.pixDiff[0][i]) continue;
                    std::printf("  y=%d x=%d d=%d  repro %6d  oracle %6d\n", row, i / D, i % D,
                                rt.pixDiff[0][i], od.pixDiff[0][i]);
                    ++shown;
                }
                BufferDiff d0 = diffBuffer(rt.pixDiff[0], od.pixDiff[0]);
                std::printf("  row %d: %ld/%ld differ, max|d|=%d\n", row, d0.differing, d0.total,
                            d0.maxAbs);
            }
            return 0;
        }
        if (verbose) std::printf("  ok   %s\n", c.name.c_str());
    }

    std::printf("\n--- %zu cases ---\n", cases.size());
    auto summary = [](const char* label, const Totals& t) {
        std::printf("%-22s %s  (%ld failures / %ld comparisons, %ld values)\n", label,
                    t.casesBad == 0 ? "PASS" : "FAIL", t.casesBad, t.stages, t.pixels);
        return t.casesBad == 0;
    };
    bool ok = true;
    ok &= summary("pixel cost", pixelCost);
    ok &= summary("box aggregation", costStage);
    ok &= summary("4 directions", fwdStage);
    ok &= summary("5th direction", aggStage);
    ok &= summary("pre-epilogue", preStage);
    ok &= summary("final", finalStage);
    if (oracleVsCv) std::printf("oracle drifted from cv on %ld cases\n", oracleVsCv);

    std::printf("\n%s\n", ok && oracleVsCv == 0 ? "ALL STAGES BIT-EXACT"
                                                : "MISMATCH -- see per-stage lines above");
    return (ok && oracleVsCv == 0) ? 0 : 1;
}
