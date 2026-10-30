// Scalar transcription of OpenCV 5's StereoSGBM (MODE_SGBM only), used as the
// test oracle for sgbm_repro.
//
// Source: opencv/modules/stereo/src/stereosgbm.cpp (Apache-2.0)
//   calcPixelCostBT        -> oracle::calcPixelCostBT
//   struct BufferSGBM      -> oracle::Buffer
//   computeDisparitySGBM   -> oracle::computeDisparitySGBM
//
// The universal-intrinsic branches are dropped; only the scalar `#else` paths
// are kept. MODE_HH / MODE_HH4 / MODE_SGBM_3WAY are not transcribed.
//
// Why a transcription instead of calling cv::StereoSGBM: the two functions we
// need to compare against are file-static, and we need per-stage buffers
// (pixDiff, C, S) that cv::StereoSGBM never exposes.
#ifndef SGBM_REPRO_TESTS_ORACLE_SGBM_ORACLE_H
#define SGBM_REPRO_TESTS_ORACLE_SGBM_ORACLE_H

#include <opencv2/core.hpp>

#include <cstddef>
#include <vector>

namespace oracle {

// Mirrors cv::StereoSGBM's parameter set. P1/P2 etc. are stored raw; the
// defaults applied by computeDisparitySGBM() are the ones that matter.
struct Params {
    int minDisparity = 0;
    int numDisparities = 0;
    int SADWindowSize = 0;
    int P1 = 0;
    int P2 = 0;
    int disp12MaxDiff = 0;
    int preFilterCap = 0;
    int uniquenessRatio = 0;
    int speckleWindowSize = 0;
    int speckleRange = 0;
    // Only MODE_SGBM is supported; kept so callers can assert on it.
    int mode = 0;

    cv::Size calcSadWindowSize() const {
        const int dim = SADWindowSize > 0 ? SADWindowSize : 5;
        return cv::Size(dim, dim);
    }
};

struct DumpConfig {
    // Rows whose C / SForward / S are captured. Empty means "no row capture".
    std::vector<int> rows;
    bool capturePixDiff = true;
    // pixDiff is width1*Da shorts per row; stop capturing once this is exceeded.
    std::size_t pixDiffByteBudget = std::size_t(256) << 20;
};

struct RowCapture {
    int y = 0;
    // Row cost after box aggregation, before any direction sweep. width1*Da.
    std::vector<short> C;
    // Aggregate after the forward 4-direction sweep only. width1*Da.
    std::vector<short> SForward;
    // Aggregate after the reverse sweep that adds the 5th direction. width1*Da.
    std::vector<short> S;
};

struct StageDump {
    int width1 = 0;
    int Da = 0;
    int D = 0;
    int minX1 = 0;

    bool capturePixDiff = true;
    std::size_t budget = 0;
    std::vector<int> pixDiffRows;
    std::vector<std::vector<short>> pixDiff;

    std::vector<RowCapture> rows;

    // Disparity before the medianBlur(3) / filterSpeckles() epilogue.
    cv::Mat disparityBeforeEpilogue;
    // Disparity after the epilogue; what cv::StereoSGBM::compute() returns.
    cv::Mat disparity;
};

// Fills in rows / capturePixDiff / budget from cfg. Rows outside [0, height)
// are dropped.
StageDump makeStageDump(const DumpConfig& cfg, int height);

// Runs the transcribed pipeline. `left` and `right` must be CV_8UC1 and the
// same size. `disp` receives CV_16S x16 fixed point, exactly like
// cv::StereoSGBM::compute(). Pass dump=nullptr to skip capture.
void computeDisparitySGBM(const cv::Mat& left, const cv::Mat& right, cv::Mat& disp,
                          const Params& params, StageDump* dump = nullptr);

// Runs only the epilogue (medianBlur 3, then filterSpeckles when configured),
// matching the tail of StereoSGBMImpl::compute(). Exposed so the harness can
// check it independently of the DP.
void epilogue(cv::Mat& disp, const Params& params);

// Number of int16 lanes OpenCV was compiled for; the cost buffer row stride is
// alignSize(numDisparities, simdInt16Lanes()).
int simdInt16Lanes();

}  // namespace oracle

#endif  // SGBM_REPRO_TESTS_ORACLE_SGBM_ORACLE_H
