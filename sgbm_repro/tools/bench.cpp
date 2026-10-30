// Single-frame time and peak RSS for cv::StereoSGBM, the transcription and this
// implementation, so the port can be judged on more than correctness.
#include "sgbm_repro/sgbm.h"
#include "sgbm_oracle.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/stereo.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>

namespace {

long peakRssKb() {
    FILE* f = std::fopen("/proc/self/status", "r");
    if (!f) return 0;
    char line[256];
    long kb = 0;
    while (std::fgets(line, sizeof line, f))
        if (!std::strncmp(line, "VmHWM:", 6)) kb = std::atol(line + 6);
    std::fclose(f);
    return kb;
}

double timeMs(int repeats, const std::function<void()>& fn) {
    // Warm up once, then take the best of `repeats` so scheduler noise does not
    // dominate.
    fn();
    double best = 1e30;
    for (int i = 0; i < repeats; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        fn();
        const auto t1 = std::chrono::steady_clock::now();
        best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    return best;
}

}  // namespace

int main(int argc, char** argv) {
    const char* leftPath = argc > 1 ? argv[1] : "/home/min/projects/test/left_off.png";
    const char* rightPath = argc > 2 ? argv[2] : "/home/min/projects/test/right_off.png";
    const int minDisp = argc > 3 ? std::atoi(argv[3]) : 32;
    const int numDisp = argc > 4 ? std::atoi(argv[4]) : 96;
    const int block = argc > 5 ? std::atoi(argv[5]) : 5;

    const cv::Mat left = cv::imread(leftPath, cv::IMREAD_GRAYSCALE);
    const cv::Mat right = cv::imread(rightPath, cv::IMREAD_GRAYSCALE);
    if (left.empty() || right.empty()) {
        std::fprintf(stderr, "bench: could not read the input pair\n");
        return 1;
    }

    const int p1 = 8 * block * block;
    const int p2 = 256 * block * block;

    cv::Mat out;

    cv::Ptr<cv::StereoSGBM> cvMatcher =
        cv::StereoSGBM::create(minDisp, numDisp, block, p1, p2, 1, 31, 5, 40, 1,
                               cv::StereoSGBM::MODE_SGBM);
    const double cvMs = timeMs(5, [&] { cvMatcher->compute(left, right, out); });

    oracle::Params op;
    op.minDisparity = minDisp;
    op.numDisparities = numDisp;
    op.SADWindowSize = block;
    op.P1 = p1;
    op.P2 = p2;
    op.disp12MaxDiff = 1;
    op.preFilterCap = 31;
    op.uniquenessRatio = 5;
    op.speckleWindowSize = 40;
    op.speckleRange = 1;
    const double oracleMs = timeMs(5, [&] { oracle::computeDisparitySGBM(left, right, out, op); });

    sgbm_repro::Params rp;
    rp.minDisparity = minDisp;
    rp.numDisparities = numDisp;
    rp.blockSize = block;
    rp.P1 = p1;
    rp.P2 = p2;
    rp.disp12MaxDiff = 1;
    rp.preFilterCap = 31;
    rp.uniquenessRatio = 5;
    rp.speckleWindowSize = 40;
    rp.speckleRange = 1;
    const double reproMs = timeMs(5, [&] { sgbm_repro::computeDisparity(left, right, out, rp); });

    std::printf("%dx%d min=%d num=%d block=%d\n", left.cols, left.rows, minDisp, numDisp, block);
    std::printf("  cv::StereoSGBM      %8.1f ms\n", cvMs);
    std::printf("  oracle (scalar)     %8.1f ms  (%.2fx)\n", oracleMs, oracleMs / cvMs);
    std::printf("  sgbm_repro          %8.1f ms  (%.2fx)\n", reproMs, reproMs / cvMs);
    std::printf("  peak RSS            %8.1f MB (whole process)\n", peakRssKb() / 1024.0);
    return 0;
}
