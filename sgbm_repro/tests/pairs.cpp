// Compares sgbm_repro against cv::StereoSGBM on real image pairs.
//
// usage:
//   pairs <left> <right> <minDisp> <numDisp> <blockSize> <P1> <P2>
//         <preFilterCap> <uniquenessRatio> <speckleWindowSize> <speckleRange>
//         <disp12MaxDiff>
//
// P1 and P2 are given raw (as cv::StereoSGBM::create takes them), so the factors
// from sgbm_config.json have to be expanded by the caller; run_all.sh does that.
#include "sgbm_repro/sgbm.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/stereo.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

int bad(const char* what) {
    std::fprintf(stderr, "pairs: %s\n", what);
    return 2;
}

struct Diff {
    long differing = 0;
    int maxAbs = 0;
    long invalidLeft = 0;
    long invalidRight = 0;
};

Diff compare(const cv::Mat& a, const cv::Mat& b, int invalidValue) {
    Diff d;
    for (int y = 0; y < a.rows; ++y) {
        const short* pa = a.ptr<short>(y);
        const short* pb = b.ptr<short>(y);
        for (int x = 0; x < a.cols; ++x) {
            d.invalidLeft += (pa[x] == invalidValue);
            d.invalidRight += (pb[x] == invalidValue);
            if (pa[x] == pb[x]) continue;
            ++d.differing;
            const int ad = std::abs((int)pa[x] - (int)pb[x]);
            if (ad > d.maxAbs) d.maxAbs = ad;
        }
    }
    return d;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 13) {
        std::fprintf(stderr,
                     "usage: pairs <left> <right> <minDisp> <numDisp> <blockSize> <P1> <P2>\n"
                     "             <preFilterCap> <uniquenessRatio> <speckleWindowSize>\n"
                     "             <speckleRange> <disp12MaxDiff>\n");
        return 2;
    }

    const cv::Mat left = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    const cv::Mat right = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
    if (left.empty()) return bad("could not read the left image");
    if (right.empty()) return bad("could not read the right image");
    if (left.size() != right.size()) return bad("left and right sizes differ");

    const int minDisp = std::atoi(argv[3]);
    const int numDisp = std::atoi(argv[4]);
    const int block = std::atoi(argv[5]);
    const int p1 = std::atoi(argv[6]);
    const int p2 = std::atoi(argv[7]);
    const int pfc = std::atoi(argv[8]);
    const int ur = std::atoi(argv[9]);
    const int sws = std::atoi(argv[10]);
    const int sr = std::atoi(argv[11]);
    const int d12 = std::atoi(argv[12]);

    cv::Mat reference;
    {
        cv::Ptr<cv::StereoSGBM> m = cv::StereoSGBM::create(
            minDisp, numDisp, block, p1, p2, d12, pfc, ur, sws, sr, cv::StereoSGBM::MODE_SGBM);
        m->compute(left, right, reference);
    }

    sgbm_repro::Params params;
    params.minDisparity = minDisp;
    params.numDisparities = numDisp;
    params.blockSize = block;
    params.P1 = p1;
    params.P2 = p2;
    params.disp12MaxDiff = d12;
    params.preFilterCap = pfc;
    params.uniquenessRatio = ur;
    params.speckleWindowSize = sws;
    params.speckleRange = sr;

    cv::Mat mine;
    sgbm_repro::computeDisparity(left, right, mine, params);

    const int invalidValue = (minDisp - 1) * 16;
    const Diff d = compare(mine, reference, invalidValue);

    std::printf("%-44s %dx%d min=%d num=%d bs=%d  differing %ld/%d  max|d|=%d  invalid %ld/%ld\n",
                std::string(argv[1]).c_str(), left.cols, left.rows, minDisp, numDisp, block,
                d.differing, left.rows * left.cols, d.maxAbs, d.invalidLeft, d.invalidRight);

    if (d.differing != 0) {
        std::printf("  MISMATCH\n");
        return 1;
    }
    std::printf("  bit-exact\n");
    return 0;
}
