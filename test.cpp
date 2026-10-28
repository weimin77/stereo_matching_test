// ===========================================================================
//  Stereo matching — C++ port of stereo_oven_food_tray/sgbm_pipeline.py
//      profile     : "balanced"  (sgbm_config.json)
//      postprocess : "wls"
//
//  Every constant below mirrors sgbm_config.json / resolve_settings().
//  Stage order is the same as the Python pipeline:
//    1. grayscale, horizontal padding  minDisp + numDisp + block = 133 px
//    2. left + right SGBM (MODE_HH); the right matcher uses the mirrored range
//    3. subpixel left-right consistency check (1.5 px)
//    4. generic WLS filter (lambda 100, sigmaColor 0.8, LRC 24/16 px)
//    5. hole filling within 8 px of a trusted pixel, 3x3 median (span <= 3 px),
//       then speckle cleanup (20 px window / 2 px range)
// ===========================================================================
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc/disparity_filter.hpp>
#if CV_VERSION_MAJOR >= 5
// OpenCV 5 split calib3d into calib/stereo/geometry. cv::DIST_L2 now lives in
// the geometry module and is no longer reachable through <opencv2/opencv.hpp>.
#include <opencv2/geometry/2d.hpp>
#endif
// cv::stereo::StereoBinarySGBM lives in opencv_contrib's xstereo module. CMake
// defines the macro when that module is part of the OpenCV build.
#ifdef ALGORITHM_TEST_HAVE_XSTEREO
#include <opencv2/xstereo.hpp>
#endif

namespace sgbmpipe {

// ---------------------------------------------------------------------------
// sgbm_config.json — everything tunable lives here, nothing is hard-coded.
// ---------------------------------------------------------------------------
enum class SgbmMode { SGBM, HH, HH4, ThreeWay };
enum class Preprocess { None, Clahe };
enum class Postprocess { None, Wls };
// Initial disparity estimator. Everything downstream (LR consistency, WLS,
// refinement) is shared, so switching this is a fair A/B on identical
// postprocessing.
enum class Matcher { SGBM, BinarySGBM };
// cv::stereo::StereoBinarySGBM works on census descriptors instead of grayscale
// costs; the kernel decides which variant. CV_MEAN_VARIATION and CV_STAR_KERNEL
// are deliberately absent: the former asserts on an integral image xstereo never
// builds, the latter requires block_size >= 7.
enum class BinaryKernel { DenseCensus, SparseCensus, CenterSymmetric, ModifiedCenterSymmetric, ModifiedCensus };

struct Settings {
    // --- search window -----------------------------------------------------
    // Derived from working_depth_m unless min_disparity / num_disparities are
    // given explicitly in the JSON (0 = "derive it").
    double workingNearM           = 0.07;
    double workingFarM            = 0.25;
    int    explicitMinDisparity   = 0;
    int    explicitNumDisparities = 0;
    bool   padding                = true;   // horizontal search-range padding

    // --- matcher -----------------------------------------------------------
    Matcher matcher               = Matcher::SGBM;
    int    blockSize              = 5;
    int    p1Factor               = 8;      // P1 = p1Factor * block^2
    int    p2Factor               = 256;    // P2 = p2Factor * block^2
    SgbmMode mode                 = SgbmMode::HH;
    int    uniquenessRatio        = 5;
    int    preFilterCap           = 31;
    int    disp12MaxDiff          = 1;
    int    speckleWindowSize      = 40;     // SGBM's own speckle filter
    int    speckleRange           = 1;

    // --- matcher = "binarysgbm" ---------------------------------------------
    // Reuses block_size, p1/p2_factor, pre_filter_cap, uniqueness_ratio,
    // disp12_max_diff, speckle_window_size and speckle_range. `mode` is
    // honoured as well (hh -> MODE_HH, anything else -> MODE_SGBM).
    BinaryKernel binaryKernel     = BinaryKernel::ModifiedCensus;
    bool   binarySymmetricSubpixel = false;  // false = quadratic, true = symmetric
    bool   binaryAverageSpeckleRemoval = true;

    // --- preprocessing -----------------------------------------------------
    Preprocess preprocess         = Preprocess::None;
    double claheClipLimit         = 2.0;
    int    claheTileGrid          = 8;

    // --- postprocessing ----------------------------------------------------
    Postprocess postprocess       = Postprocess::Wls;
    double lrMaxDiffPx            = 1.5;
    double wlsLambda              = 100.0;
    double wlsSigmaColor          = 0.8;
    int    wlsDiscontinuityRadius = 1;
    double confidenceThreshold    = 0.0;
    double inferenceRadiusAt640   = 8.0;    // inference_radius_at_640_px
    int    medianSize             = 3;
    double medianMaxSpanPx        = 3.0;
    double preserveTrustedJumpPx  = 2.0;
    int    postSpeckleSize        = 20;
    double postSpeckleRangePx     = 2.0;
};

// ---------------------------------------------------------------------------
// Everything that is not a matcher parameter but still user-adjustable.
// ---------------------------------------------------------------------------
struct Config {
    Settings settings;
    // Camera. Either inline, or read from a camera.json (dataset calibration).
    double fx       = 140.0;
    double baseline = 0.06;
    std::string cameraFile;

    // Outputs.
    std::string outputDirectory;      // empty = directory of the left image
    std::string outputPrefix = "cpp_";
    bool   saveArtifacts     = true;
    bool   showWindow        = true;
    // Preview colour range in px. Both 0 = derive from the result percentiles.
    double previewNearPx     = 0.0;
    double previewFarPx      = 0.0;
};

// Reads sgbm_config.json. Defined at the end of this file so that the JSON
// plumbing stays out of the algorithm core.
Config loadConfig(const std::string& path);

// Result of resolve_settings(): the search window, either derived from the
// configured working depth range or taken as-is from the JSON.
struct Resolved {
    Settings    s;
    int    width             = 0;
    int    height            = 0;
    int    minDisparity      = 0;
    int    numDisparities    = 0;
    int    rightMinDisparity = 0;
    int    padding           = 0;
    double limitLow          = 0.0;
    double limitHigh         = 0.0;
    double inferenceRadiusPx = 0.0;
};

inline float NaN() { return std::numeric_limits<float>::quiet_NaN(); }

inline Resolved resolve(const int width, const int height, const double fx, const double baseline,
                        const Settings& s)
{
    if (width <= 0 || height <= 0) throw std::invalid_argument("empty input images");
    if (!(fx > 0) || !(baseline > 0))
        throw std::invalid_argument("camera fx and baseline must be positive");

    Resolved r;
    r.s = s;
    r.width = width;
    r.height = height;

    if (s.explicitNumDisparities > 0) {
        // Explicit search window from the JSON.
        if (s.explicitNumDisparities % 16 != 0 || s.explicitMinDisparity < 0)
            throw std::invalid_argument("num_disparities must be a multiple of 16 and "
                                        "min_disparity must be >= 0");
        r.minDisparity = s.explicitMinDisparity;
        r.numDisparities = s.explicitNumDisparities;
        r.limitLow = static_cast<double>(r.minDisparity);
        r.limitHigh = r.minDisparity + r.numDisparities - 1.0 / 16.0;
    } else {
        if (!(s.workingNearM > 0) || !(s.workingFarM > s.workingNearM))
            throw std::invalid_argument("working_depth_m must satisfy 0 < near < far");
        const double product = fx * baseline;
        const double low = product / s.workingFarM;
        const double high = product / s.workingNearM;
        r.minDisparity = static_cast<int>(std::floor(low / 16.0)) * 16;
        r.numDisparities = static_cast<int>(std::ceil((high - r.minDisparity + 1.0) / 16.0)) * 16;
        r.limitLow = std::max(low, static_cast<double>(r.minDisparity));
        r.limitHigh = std::min(high, r.minDisparity + r.numDisparities - 1.0 / 16.0);
        if (r.limitLow >= r.limitHigh)
            throw std::invalid_argument("working_depth_m yields an empty disparity range");
    }

    r.rightMinDisparity = -r.minDisparity - r.numDisparities + 1;
    r.padding = s.padding ? r.minDisparity + r.numDisparities + s.blockSize : 0;
    r.inferenceRadiusPx = s.inferenceRadiusAt640 * width / 640.0;
    if (r.padding >= width || r.padding >= height)
        throw std::runtime_error("disparity search window does not fit the image");
    if (r.numDisparities >= width)
        throw std::runtime_error("num_disparities must be smaller than the image width");
    return r;
}

inline bool finite(float v) { return std::isfinite(v); }

// sgbm_pipeline.correspondence_valid
cv::Mat correspondenceValid(const cv::Mat& disparity, const Resolved& r)
{
    cv::Mat valid(disparity.size(), CV_8U, cv::Scalar(0));
    for (int y = 0; y < disparity.rows; ++y) {
        const float* d = disparity.ptr<float>(y);
        unsigned char* v = valid.ptr<unsigned char>(y);
        for (int x = 0; x < disparity.cols; ++x) {
            const float value = d[x];
            if (!finite(value) || value < r.limitLow || value > r.limitHigh) continue;
            const float destination = static_cast<float>(x) - value;
            if (destination < 0.0f || destination > disparity.cols - 1) continue;
            v[x] = 255;
        }
    }
    return valid;
}

inline cv::Mat finiteMask(const cv::Mat& values)
{
    cv::Mat mask(values.size(), CV_8U, cv::Scalar(0));
    for (int y = 0; y < values.rows; ++y) {
        const float* in = values.ptr<float>(y);
        unsigned char* out = mask.ptr<unsigned char>(y);
        for (int x = 0; x < values.cols; ++x) out[x] = finite(in[x]) ? 255 : 0;
    }
    return mask;
}

// sgbm_pipeline.left_right_consistency: subpixel check, bilinear sample of the
// right map at x - d_left; the right disparity must be negative.
void leftRightConsistency(const cv::Mat& left, const cv::Mat& right, double maxDifference,
                          cv::Mat& consistent, cv::Mat& lrError)
{
    consistent = cv::Mat(left.size(), CV_8U, cv::Scalar(0));
    lrError = cv::Mat(left.size(), CV_32F, cv::Scalar(NaN()));
    const int width = left.cols;
    for (int y = 0; y < left.rows; ++y) {
        const float* l = left.ptr<float>(y);
        const float* rr = right.ptr<float>(y);
        unsigned char* ok = consistent.ptr<unsigned char>(y);
        float* err = lrError.ptr<float>(y);
        for (int x = 0; x < width; ++x) {
            const float value = l[x];
            if (!finite(value)) continue;
            const float destination = static_cast<float>(x) - value;
            if (!(destination >= 0.0f && destination <= width - 1)) continue;
            const int lower = std::min(std::max(static_cast<int>(std::floor(destination)), 0), width - 1);
            const int upper = std::min(lower + 1, width - 1);
            const float weight = destination - lower;
            const float a = rr[lower];
            const float b = rr[upper];
            // numpy's np.where(weight < 1e-6, a, ...) never touches b in that branch,
            // so an invalid right sample only matters when it is actually used.
            const float sample = (weight < 1e-6f) ? a : (1.0f - weight) * a + weight * b;
            if (!finite(sample)) continue;
            err[x] = std::abs(value + sample);
            if (sample < 0.0f && err[x] <= maxDifference) ok[x] = 255;
        }
    }
}

// sgbm_pipeline._matcher
inline int toCvMode(SgbmMode mode)
{
    switch (mode) {
    case SgbmMode::HH:       return cv::StereoSGBM::MODE_HH;
    case SgbmMode::HH4:      return cv::StereoSGBM::MODE_HH4;
    case SgbmMode::ThreeWay: return cv::StereoSGBM::MODE_SGBM_3WAY;
    default:                 return cv::StereoSGBM::MODE_SGBM;
    }
}

inline const char* toModeName(SgbmMode mode)
{
    switch (mode) {
    case SgbmMode::HH:       return "hh";
    case SgbmMode::HH4:      return "hh4";
    case SgbmMode::ThreeWay: return "3way";
    default:                 return "sgbm";
    }
}

cv::Ptr<cv::StereoSGBM> makeMatcher(const Resolved& r, int minDisparity)
{
    const Settings& s = r.s;
    const int block = s.blockSize;
    return cv::StereoSGBM::create(
        minDisparity, r.numDisparities, block,
        s.p1Factor * block * block, s.p2Factor * block * block,
        s.disp12MaxDiff, s.preFilterCap, s.uniquenessRatio,
        s.speckleWindowSize, s.speckleRange, toCvMode(s.mode));
}

// ---------------------------------------------------------------------------
// matcher = "binarysgbm": cv::stereo::StereoBinarySGBM (opencv_contrib::xstereo)
//
// Same energy formulation as StereoSGBM, but the pixel cost is a Hamming
// distance between census descriptors instead of a Birchfield-Tomasi grayscale
// difference. Census codes the ordering of neighbours, so it is insensitive to
// the monotonic gain/offset changes that projector light falloff produces --
// which is why it does better on the textureless tray.
//
// xstereo has three behaviours that a direct port has to work around:
//
//  1. The cost volume is indexed by ABSOLUTE disparity offset 0..numDisparities
//     (hammingDistanceBlockMatching fills maxDisparity+1 entries starting at 0),
//     so minDisparity does not shift the search range at all.
//  2. It then ADDS minDisparity to every output value:
//         disp1ptr[x] = (DispType)(d + minD * DISP_SCALE)
//     so asking for minDisparity = 32 yields (true disparity + 32) * 16.
//  3. setMinDisparity() asserts minDisparity >= 0, so the mirrored negative
//     range used for the right-view matcher is impossible.
//
// All three disappear by configuring minDisparity = 0 and numDisparities =
// minDisparity + numDisparities (i.e. covering the whole absolute range). That
// costs the same number of labels as the correct range would, and lets the
// right-view map be produced by running the same matcher on (right, left) and
// negating the (non-negative) magnitude.
// ---------------------------------------------------------------------------

// Absolute label count the binary matcher has to search: it always starts at 0.
inline int binaryNumDisparities(const Resolved& r)
{
    return ((r.minDisparity + r.numDisparities + 15) / 16) * 16;
}

// Largest block_size each census descriptor accepts. These are the assertions in
// xstereo's descriptor.cpp (kernelSize <= 5 / 11 / 7 / 7 / 9); checking them here
// turns a cryptic OpenCV assertion into a configuration error.
inline int binaryMaxBlockSize(BinaryKernel kernel)
{
    switch (kernel) {
    case BinaryKernel::DenseCensus:              return 5;
    case BinaryKernel::SparseCensus:             return 11;
    case BinaryKernel::CenterSymmetric:          return 7;
    case BinaryKernel::ModifiedCenterSymmetric:  return 7;
    default:                                     return 9;   // ModifiedCensus
    }
}

cv::Ptr<cv::StereoMatcher> makeBinaryMatcher(const Resolved& r)
{
#ifndef ALGORITHM_TEST_HAVE_XSTEREO
    (void)r;
    throw std::runtime_error("matcher \"binarysgbm\" needs the OpenCV xstereo module "
                             "(opencv_contrib); this build does not have it");
#else
    const Settings& s = r.s;
    const int block = s.blockSize;
    cv::Ptr<cv::stereo::StereoBinarySGBM> matcher = cv::stereo::StereoBinarySGBM::create(
        0, binaryNumDisparities(r), block,
        s.p1Factor * block * block, s.p2Factor * block * block,
        s.disp12MaxDiff, s.preFilterCap, s.uniquenessRatio,
        s.speckleWindowSize, s.speckleRange,
        s.mode == SgbmMode::HH ? cv::stereo::StereoBinarySGBM::MODE_HH
                               : cv::stereo::StereoBinarySGBM::MODE_SGBM);
    const int kernels[] = {cv::stereo::CV_DENSE_CENSUS, cv::stereo::CV_SPARSE_CENSUS,
                           cv::stereo::CV_CS_CENSUS, cv::stereo::CV_MODIFIED_CS_CENSUS,
                           cv::stereo::CV_MODIFIED_CENSUS_TRANSFORM};
    matcher->setBinaryKernelType(kernels[static_cast<int>(s.binaryKernel)]);
    matcher->setSubPixelInterpolationMethod(
        s.binarySymmetricSubpixel ? cv::stereo::CV_SIMETRICV_INTERPOLATION
                                  : cv::stereo::CV_QUADRATIC_INTERPOLATION);
    matcher->setSpekleRemovalTechnique(
        s.binaryAverageSpeckleRemoval ? cv::stereo::CV_SPECKLE_REMOVAL_AVG_ALGORITHM
                                      : cv::stereo::CV_SPECKLE_REMOVAL_ALGORITHM);
    return matcher;
#endif
}

// Runs the binary matcher on a padded pair and re-encodes the result into the
// same convention the rest of the pipeline expects: CV_16S x16 fixed point,
// invalid pixels at `leftInvalid` / `rightInvalid`. With minDisparity = 0 the
// library's own invalid marker is (0 - 1) * 16 = -16.
cv::Mat binaryDisparity(const cv::Mat& leftPadded, const cv::Mat& rightPadded,
                        const Resolved& r, bool rightSide, int& invalidMarker)
{
    cv::Ptr<cv::StereoMatcher> matcher = makeBinaryMatcher(r);
    cv::Mat raw;
    if (rightSide) {
        // The right-view map needs right(x) <-> left(x + d), i.e. a search in the
        // +x direction. xstereo only has cost labels for offsets 0..numD (always
        // 0..+x), unlike cv::StereoSGBM which accepts a negative minDisparity.
        // Flipping both images mirrors the search: in flipped coordinates
        // u = W-1-x the correspondence lands at u - d, so the same matcher finds
        // the magnitude, and the caller negates it back.
        cv::Mat leftFlipped, rightFlipped;
        cv::flip(leftPadded, leftFlipped, 1);
        cv::flip(rightPadded, rightFlipped, 1);
        matcher->compute(rightFlipped, leftFlipped, raw);
    } else {
        matcher->compute(leftPadded, rightPadded, raw);
    }

    const int libraryInvalid = -16;                       // (minDisparity - 1) * 16, minDisparity = 0
    // A valid right-view disparity reaches down to -numDisparities*16 = -2048,
    // which is why its invalid marker cannot be -16 like the left one.
    invalidMarker = rightSide ? -32768 : libraryInvalid;
    cv::Mat fixed(raw.size(), CV_16S, cv::Scalar(invalidMarker));
    const int width = raw.cols;
    for (int y = 0; y < raw.rows; ++y) {
        const short* in = raw.ptr<short>(y);
        short* out = fixed.ptr<short>(y);
        for (int x = 0; x < width; ++x) {
            // Undo the horizontal flip and the sign together.
            const short value = in[rightSide ? width - 1 - x : x];
            if (value <= libraryInvalid) continue;
            out[x] = rightSide ? static_cast<short>(-value) : value;
        }
    }
    return fixed;
}


struct Stages {
    cv::Mat raw;         // left SGBM, invalidated, before LR check
    cv::Mat checked;     // left-right consistent subset
    cv::Mat candidate;   // after generic WLS
    cv::Mat finalMap;    // after hole filling + median + speckle cleanup
    cv::Mat confidence;  // WLS confidence map (CV_32F)
    cv::Mat lrError;     // |d_left + d_right| in px
    cv::Mat inferred;    // final pixels that were not left-right consistent
    double  matchingMs = 0.0;
    double  totalMs = 0.0;
    // per-stage wall time, for profiling
    std::vector<std::pair<std::string, double>> timings;
};

class StageTimer
{
public:
    StageTimer(std::vector<std::pair<std::string, double>>& sink, const char* name)
        : sink_(sink), name_(name), start_(std::chrono::steady_clock::now()) {}
    ~StageTimer()
    {
        sink_.emplace_back(name_, std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start_).count());
    }
private:
    std::vector<std::pair<std::string, double>>& sink_;
    const char* name_;
    std::chrono::steady_clock::time_point start_;
};

#ifdef NDEBUG
constexpr const char* kBuildType = "Release (optimised)";
#else
constexpr const char* kBuildType = "Debug (-O0)";
#endif

// sgbm_pipeline.refine_disparity
cv::Mat refineDisparity(const cv::Mat& checked, const cv::Mat& candidate,
                        const cv::Mat& confidence, const Resolved& r)
{
    const Settings& s = r.s;
    const cv::Mat trusted = finiteMask(checked);

    cv::Mat distance;
    if (cv::countNonZero(trusted) > 0) {
        cv::Mat untrusted(trusted.size(), CV_8U);
        cv::bitwise_not(trusted, untrusted);                 // 0 where trusted
        cv::distanceTransform(untrusted, distance, cv::DIST_L2, cv::DIST_MASK_PRECISE);
    } else {
        distance = cv::Mat(checked.size(), CV_32F,
                           cv::Scalar(std::numeric_limits<float>::infinity()));
    }
    const cv::Mat trustedFlags = correspondenceValid(candidate, r);

    cv::Mat finalMap(checked.size(), CV_32F, cv::Scalar(NaN()));
    for (int y = 0; y < checked.rows; ++y) {
        const float* cand = candidate.ptr<float>(y);
        const float* ref = checked.ptr<float>(y);
        const float* conf = confidence.ptr<float>(y);
        const float* dist = distance.ptr<float>(y);
        const unsigned char* isTrusted = trusted.ptr<unsigned char>(y);
        const unsigned char* inside = trustedFlags.ptr<unsigned char>(y);
        float* out = finalMap.ptr<float>(y);
        for (int x = 0; x < checked.cols; ++x) {
            bool keep = inside[x] != 0 && conf[x] >= s.confidenceThreshold;
            keep = keep && (isTrusted[x] != 0 || dist[x] <= r.inferenceRadiusPx);
            // Keep a trusted discontinuity rather than letting the smoother move it.
            const bool preserve = isTrusted[x] != 0 &&
                (!finite(cand[x]) || std::abs(cand[x] - ref[x]) > s.preserveTrustedJumpPx);
            float value = cand[x];
            if (preserve) {
                value = ref[x];
                keep = conf[x] >= s.confidenceThreshold;
            }
            out[x] = keep ? value : NaN();
        }
    }

    const int size = s.medianSize;
    if (size > 0) {
        const cv::Mat kernel8 = cv::Mat::ones(size, size, CV_8U);
        const cv::Mat kernel32 = cv::Mat::ones(size, size, CV_32F);
        cv::Mat validFlags(checked.size(), CV_32F);
        cv::Mat values(checked.size(), CV_32F);
        cv::Mat lowInput(checked.size(), CV_32F);
        cv::Mat highInput(checked.size(), CV_32F);
        for (int y = 0; y < checked.rows; ++y) {
            const float* in = finalMap.ptr<float>(y);
            float* vf = validFlags.ptr<float>(y);
            float* vv = values.ptr<float>(y);
            float* lo = lowInput.ptr<float>(y);
            float* hi = highInput.ptr<float>(y);
            for (int x = 0; x < checked.cols; ++x) {
                const bool ok = finite(in[x]);
                vf[x] = ok ? 1.0f : 0.0f;
                vv[x] = ok ? in[x] : 0.0f;
                lo[x] = ok ? in[x] : 1e6f;
                hi[x] = ok ? in[x] : -1e6f;
            }
        }
        cv::Mat count;
        cv::filter2D(validFlags, count, CV_32F, kernel32, cv::Point(-1, -1), 0.0, cv::BORDER_CONSTANT);
        cv::Mat lowImage, highImage, median;
        cv::erode(lowInput, lowImage, kernel8);
        cv::dilate(highInput, highImage, kernel8);
        cv::medianBlur(values, median, size);
        const double majority = size * size / 2.0;
        for (int y = 0; y < checked.rows; ++y) {
            float* out = finalMap.ptr<float>(y);
            const float* c = count.ptr<float>(y);
            const float* lo = lowImage.ptr<float>(y);
            const float* hi = highImage.ptr<float>(y);
            const float* med = median.ptr<float>(y);
            for (int x = 0; x < checked.cols; ++x) {
                if (!finite(out[x])) continue;
                if (c[x] > majority && (hi[x] - lo[x]) <= s.medianMaxSpanPx && med[x] > 0.0f)
                    out[x] = med[x];
            }
        }
    }

    if (s.postSpeckleSize > 0) {
        cv::Mat fixed16(checked.size(), CV_16S, cv::Scalar(0));
        for (int y = 0; y < checked.rows; ++y) {
            const float* in = finalMap.ptr<float>(y);
            short* out = fixed16.ptr<short>(y);
            for (int x = 0; x < checked.cols; ++x) {
                if (!finite(in[x])) continue;
                // numpy.rint == round-half-to-even -> std::nearbyint
                long long q = static_cast<long long>(std::nearbyint(static_cast<double>(in[x]) * 16.0));
                out[x] = static_cast<short>(std::min<long long>(std::max<long long>(q, -32768), 32767));
            }
        }
        cv::filterSpeckles(fixed16, 0.0, s.postSpeckleSize,
                           std::nearbyint(s.postSpeckleRangePx * 16.0));
        for (int y = 0; y < checked.rows; ++y) {
            float* out = finalMap.ptr<float>(y);
            const short* fixed = fixed16.ptr<short>(y);
            for (int x = 0; x < checked.cols; ++x)
                if (fixed[x] == 0) out[x] = NaN();
        }
    }

    const cv::Mat stillValid = correspondenceValid(finalMap, r);
    for (int y = 0; y < checked.rows; ++y) {
        float* out = finalMap.ptr<float>(y);
        const unsigned char* ok = stillValid.ptr<unsigned char>(y);
        for (int x = 0; x < checked.cols; ++x)
            if (ok[x] == 0) out[x] = NaN();
    }
    return finalMap;
}

// sgbm_pipeline.match_stereo (WLS backend). rgb inputs are HxWx3 uint8 in RGB order.
Stages matchStereo(const cv::Mat& leftRgb, const cv::Mat& rightRgb, const Resolved& r)
{
    const Settings& s = r.s;
    const int width = r.width;
    const int height = r.height;
    if (leftRgb.size() != cv::Size(width, height) || rightRgb.size() != leftRgb.size() ||
        leftRgb.type() != CV_8UC3 || rightRgb.type() != CV_8UC3)
        throw std::invalid_argument("expected two equal HxWx3 uint8 RGB images");

    Stages stages;
    const auto start = std::chrono::steady_clock::now();
    auto& t = stages.timings;

    cv::Mat leftGray, rightGray;
    {
        StageTimer timer(t, "cvtColor x2");
        cv::cvtColor(leftRgb, leftGray, cv::COLOR_RGB2GRAY);      // RGB order, like the pipeline
        cv::cvtColor(rightRgb, rightGray, cv::COLOR_RGB2GRAY);
    }
    if (s.preprocess == Preprocess::Clahe) {
        StageTimer timer(t, "CLAHE x2");
        const cv::Size tile(s.claheTileGrid, s.claheTileGrid);
        cv::createCLAHE(s.claheClipLimit, tile)->apply(leftGray, leftGray);
        cv::createCLAHE(s.claheClipLimit, tile)->apply(rightGray, rightGray);
    }
    const int padding = r.padding;
    cv::Mat leftPadded, rightPadded;
    {
        StageTimer timer(t, "copyMakeBorder x2");
        cv::copyMakeBorder(leftGray, leftPadded, 0, 0, padding, padding, cv::BORDER_CONSTANT, cv::Scalar(0));
        cv::copyMakeBorder(rightGray, rightPadded, 0, 0, padding, padding, cv::BORDER_CONSTANT, cv::Scalar(0));
    }

    cv::Mat leftFixed, rightFixed;   // CV_16S x16, padded size, invalid = sentinel
    // SGBM reports "invalid" as (minDisparity - 1) * 16; binaryDisparity() picks
    // its own sentinels because the binary matcher's valid right-view values are
    // negative and would collide with -16.
    int leftInvalid  = (r.minDisparity - 1) * 16;
    int rightInvalid = (r.rightMinDisparity - 1) * 16;
    if (s.matcher == Matcher::BinarySGBM) {
        {
            StageTimer timer(t, "binary SGBM left");
            leftFixed = binaryDisparity(leftPadded, rightPadded, r, false, leftInvalid);
        }
        {
            StageTimer timer(t, "binary SGBM right");
            // Arguments stay in true left/right order: binaryDisparity() does the
            // flip and the role swap itself (both are needed).
            rightFixed = binaryDisparity(leftPadded, rightPadded, r, true, rightInvalid);
        }
    } else {
        cv::Ptr<cv::StereoSGBM> leftMatcher = makeMatcher(r, r.minDisparity);
        cv::Ptr<cv::StereoSGBM> rightMatcher = makeMatcher(r, r.rightMinDisparity);
        {
            StageTimer timer(t, "SGBM left");
            leftMatcher->compute(leftPadded, rightPadded, leftFixed);
        }
        {
            StageTimer timer(t, "SGBM right");
            rightMatcher->compute(rightPadded, leftPadded, rightFixed);
        }
    }

    const cv::Rect crop(padding, 0, width, height);
    cv::Mat leftCropped = leftFixed(crop);
    cv::Mat rightCropped = rightFixed(crop);
    cv::Mat right = cv::Mat(height, width, CV_32F);
    cv::Mat rawValid;
    {
        StageTimer timer(t, "unpack + invalidate");
        stages.raw = cv::Mat(height, width, CV_32F);
        for (int y = 0; y < height; ++y) {
            const short* in = leftCropped.ptr<short>(y);
            float* out = stages.raw.ptr<float>(y);
            for (int x = 0; x < width; ++x) out[x] = static_cast<float>(in[x]) / 16.0f;
        }
        for (int y = 0; y < height; ++y) {
            const short* in = rightCropped.ptr<short>(y);
            float* out = right.ptr<float>(y);
            for (int x = 0; x < width; ++x) out[x] = static_cast<float>(in[x]) / 16.0f;
        }
        rawValid = correspondenceValid(stages.raw, r);
        for (int y = 0; y < height; ++y) {
            float* raw = stages.raw.ptr<float>(y);
            float* rgt = right.ptr<float>(y);
            const short* lraw = leftCropped.ptr<short>(y);
            const short* rraw = rightCropped.ptr<short>(y);
            const unsigned char* ok = rawValid.ptr<unsigned char>(y);
            for (int x = 0; x < width; ++x) {
                if (lraw[x] <= leftInvalid || ok[x] == 0) raw[x] = NaN();
                if (rraw[x] <= rightInvalid) rgt[x] = NaN();
            }
        }
    }

    cv::Mat consistent;
    {
        StageTimer timer(t, "LR consistency");
        leftRightConsistency(stages.raw, right, s.lrMaxDiffPx, consistent, stages.lrError);
    }
    {
        StageTimer timer(t, "checked mask");
        stages.checked = cv::Mat(height, width, CV_32F);
        for (int y = 0; y < height; ++y) {
            const float* in = stages.raw.ptr<float>(y);
            const unsigned char* ok = consistent.ptr<unsigned char>(y);
            float* out = stages.checked.ptr<float>(y);
            for (int x = 0; x < width; ++x) out[x] = ok[x] ? in[x] : NaN();
        }
    }
    stages.matchingMs =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();

    if (s.postprocess == Postprocess::None) {
        // No WLS: report the LR-consistent result as the final map. The
        // candidate/inferred stages do not exist in this mode.
        StageTimer timer(t, "postprocess skipped");
        stages.candidate = stages.checked.clone();
        stages.confidence = cv::Mat(height, width, CV_32F, cv::Scalar(0));
        stages.finalMap = stages.checked.clone();
        stages.inferred = cv::Mat(height, width, CV_8U, cv::Scalar(0));
        stages.totalMs =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        return stages;
    }

    // --- generic WLS filter -------------------------------------------------
    // The guide keeps the RGB channel order of the Python pipeline: OpenCV
    // converts it internally with COLOR_BGR2GRAY, so feeding RGB reproduces
    // exactly the same luma the Python code sees.
    cv::Mat guide;
    cv::Mat filterLeft = leftFixed.clone();
    cv::Mat filterRight = rightFixed.clone();
    {
        StageTimer timer(t, "WLS prep");
        cv::copyMakeBorder(leftRgb, guide, 0, 0, padding, padding, cv::BORDER_CONSTANT, cv::Scalar(0));
        for (int y = 0; y < filterLeft.rows; ++y) {
            short* row = filterLeft.ptr<short>(y);
            for (int x = 0; x < filterLeft.cols; ++x)
                if (row[x] <= leftInvalid) row[x] = -16;          // WLS' own invalid marker
        }
        for (int y = 0; y < filterRight.rows; ++y) {
            short* row = filterRight.ptr<short>(y);
            for (int x = 0; x < filterRight.cols; ++x)
                if (row[x] <= rightInvalid) row[x] = -32768;
        }
    }
    cv::Ptr<cv::ximgproc::DisparityWLSFilter> filter =
        cv::ximgproc::createDisparityWLSFilterGeneric(true);
    filter->setLambda(s.wlsLambda);
    filter->setSigmaColor(s.wlsSigmaColor);
    filter->setLRCthresh(static_cast<int>(std::nearbyint(s.lrMaxDiffPx * 16.0)));
    filter->setDepthDiscontinuityRadius(s.wlsDiscontinuityRadius);
    // CV_16S in -> CV_16S out (still x16 fixed point), full padded size.
    cv::Mat filtered16;
    {
        StageTimer timer(t, "WLS filter");
        filter->filter(filterLeft, guide, filtered16, filterRight, cv::Rect(padding, 0, width, height));
    }
    cv::Mat filteredCropped = filtered16(crop);
    cv::Mat candidateValid;
    {
        StageTimer timer(t, "unpack WLS + invalidate");
        stages.candidate = cv::Mat(height, width, CV_32F);
        for (int y = 0; y < height; ++y) {
            const short* in = filteredCropped.ptr<short>(y);
            float* out = stages.candidate.ptr<float>(y);
            for (int x = 0; x < width; ++x) out[x] = static_cast<float>(in[x]) / 16.0f;
        }
        candidateValid = correspondenceValid(stages.candidate, r);
        for (int y = 0; y < height; ++y) {
            float* out = stages.candidate.ptr<float>(y);
            const unsigned char* ok = candidateValid.ptr<unsigned char>(y);
            for (int x = 0; x < width; ++x)
                if (ok[x] == 0) out[x] = NaN();
        }
        stages.confidence = filter->getConfidenceMap()(crop).clone();
        if (stages.confidence.type() != CV_32F) stages.confidence.convertTo(stages.confidence, CV_32F);
    }

    {
        StageTimer timer(t, "refine (fill/median/speckle)");
        stages.finalMap = refineDisparity(stages.checked, stages.candidate, stages.confidence, r);
    }
    {
        StageTimer timer(t, "inferred mask");
        stages.inferred = cv::Mat(height, width, CV_8U, cv::Scalar(0));
        const cv::Mat trustedMask = finiteMask(stages.checked);
        for (int y = 0; y < height; ++y) {
            const float* fin = stages.finalMap.ptr<float>(y);
            const unsigned char* trusted = trustedMask.ptr<unsigned char>(y);
            unsigned char* out = stages.inferred.ptr<unsigned char>(y);
            for (int x = 0; x < width; ++x)
                out[x] = (finite(fin[x]) && trusted[x] == 0) ? 255 : 0;
        }
    }
    stages.totalMs =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    return stages;
}

// Python save_uint16(): rint(v * scale), 0 = invalid / not representable.
void saveUint16(const cv::Mat& values, double scale, const std::string& path)
{
    cv::Mat encoded(values.size(), CV_16U, cv::Scalar(0));
    for (int y = 0; y < values.rows; ++y) {
        const float* in = values.ptr<float>(y);
        unsigned short* out = encoded.ptr<unsigned short>(y);
        for (int x = 0; x < values.cols; ++x) {
            if (!finite(in[x]) || in[x] <= 0.0f) continue;
            const double scaled = std::nearbyint(static_cast<double>(in[x]) * scale);
            if (scaled < 1.0 || scaled > 65535.0) continue;
            out[x] = static_cast<unsigned short>(scaled);
        }
    }
    cv::imwrite(path, encoded);
}

void saveMask(const cv::Mat& mask, const std::string& path) { cv::imwrite(path, mask); }

// Same palette and colour range as evaluation_sgbm/disparity_pred_left_vis.png
void saveDisparityPreview(const cv::Mat& disparity, double nearPx, double farPx,
                          const std::string& path)
{
    static const int palette[5][3] = {{220, 50, 47}, {245, 178, 55}, {90, 200, 140},
                                      {44, 140, 210}, {65, 62, 155}};
    cv::Mat rgb(disparity.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    const double span = std::max(farPx - nearPx, 1e-6);
    for (int y = 0; y < disparity.rows; ++y) {
        const float* in = disparity.ptr<float>(y);
        cv::Vec3b* out = rgb.ptr<cv::Vec3b>(y);
        for (int x = 0; x < disparity.cols; ++x) {
            if (!finite(in[x]) || in[x] <= 0.0f) continue;         // black = invalid
            const double t = std::min(std::max((in[x] - nearPx) / span, 0.0), 1.0);
            const double position = t * 4.0;
            const int lower = std::min(static_cast<int>(position), 3);
            const double weight = position - lower;
            double color[3];
            for (int k = 0; k < 3; ++k)
                color[k] = palette[lower][k] * (1.0 - weight) + palette[lower + 1][k] * weight;
            // numpy's .astype(np.uint8) truncates; reproduce that instead of rounding
            out[x] = cv::Vec3b(static_cast<uchar>(std::min(std::max(color[0], 0.0), 255.0)),
                               static_cast<uchar>(std::min(std::max(color[1], 0.0), 255.0)),
                               static_cast<uchar>(std::min(std::max(color[2], 0.0), 255.0)));
        }
    }
    cv::Mat bgr;
    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
    cv::imwrite(path, bgr);
}

// ---------------------------------------------------------------------------
// sgbm_config.json loading
//
// Read with cv::FileStorage (JSON is supported natively, no extra dependency).
// Every key is optional: anything missing keeps the default from Settings /
// Config, so a partial config is valid.
//
// Precedence follows sgbm_pipeline.resolve_settings():
//     built-in defaults  <  top-level keys  <  profiles[profile]
// i.e. a key that a profile defines wins over the same top-level key.
// ---------------------------------------------------------------------------
namespace detail {

// A missing key yields an empty node; leave the default untouched then.
inline bool present(const cv::FileNode& node) { return !node.empty(); }

inline void readInt(const cv::FileNode& n, int& out)
{
    if (!present(n)) return;
    if (!n.isInt() && !n.isReal())
        throw std::invalid_argument("expected a number for key '" + n.name() + "'");
    out = static_cast<int>(n.real());
}

inline void readDouble(const cv::FileNode& n, double& out)
{
    if (!present(n)) return;
    if (!n.isReal() && !n.isInt())
        throw std::invalid_argument("expected a number for key '" + n.name() + "'");
    out = n.real();
}

inline void readBool(const cv::FileNode& n, bool& out)
{
    if (!present(n)) return;
    if (!n.isInt() && !n.isReal())
        throw std::invalid_argument("expected a boolean for key '" + n.name() + "'");
    out = n.real() != 0.0;
}

inline void readString(const cv::FileNode& n, std::string& out)
{
    if (!present(n)) return;
    if (!n.isString())
        throw std::invalid_argument("expected a string for key '" + n.name() + "'");
    out = static_cast<std::string>(n);
}

inline void readEnum(const cv::FileNode& n, SgbmMode& out)
{
    std::string value;
    readString(n, value);
    if (value.empty()) return;
    if (value == "hh")         out = SgbmMode::HH;
    else if (value == "hh4")   out = SgbmMode::HH4;
    else if (value == "3way")  out = SgbmMode::ThreeWay;
    else if (value == "sgbm")  out = SgbmMode::SGBM;
    else throw std::invalid_argument("mode must be one of hh / hh4 / 3way / sgbm, got '" + value + "'");
}

inline void readEnum(const cv::FileNode& n, Preprocess& out)
{
    std::string value;
    readString(n, value);
    if (value.empty()) return;
    if (value == "none")       out = Preprocess::None;
    else if (value == "clahe") out = Preprocess::Clahe;
    else throw std::invalid_argument("preprocess must be none or clahe, got '" + value + "'");
}

inline void readEnum(const cv::FileNode& n, Postprocess& out)
{
    std::string value;
    readString(n, value);
    if (value.empty()) return;
    if (value == "none")      out = Postprocess::None;
    else if (value == "wls")  out = Postprocess::Wls;
    else throw std::invalid_argument("postprocess must be none or wls, got '" + value + "' "
                                     "(the guided backend is not ported)");
}

// Flat matcher/postprocess keys shared by `profiles[name]` and the top level.
inline void readEnum(const cv::FileNode& n, Matcher& out)
{
    std::string value;
    readString(n, value);
    if (value.empty()) return;
    if (value == "sgbm")            out = Matcher::SGBM;
    else if (value == "binarysgbm") out = Matcher::BinarySGBM;
    else throw std::invalid_argument("matcher must be sgbm or binarysgbm, got '" + value + "'");
}

inline void readEnum(const cv::FileNode& n, BinaryKernel& out)
{
    std::string value;
    readString(n, value);
    if (value.empty()) return;
    if (value == "dense")        out = BinaryKernel::DenseCensus;
    else if (value == "sparse")  out = BinaryKernel::SparseCensus;
    else if (value == "cs")      out = BinaryKernel::CenterSymmetric;
    else if (value == "mod_cs")  out = BinaryKernel::ModifiedCenterSymmetric;
    else if (value == "mod_census") out = BinaryKernel::ModifiedCensus;
    else throw std::invalid_argument("binary_kernel must be one of dense / sparse / cs / mod_cs / "
                                     "mod_census, got '" + value + "'");
}

inline void readMatcherSettings(const cv::FileNode& n, Settings& s)
{
    readEnum(n["matcher"], s.matcher);
    readInt(n["block_size"], s.blockSize);
    readInt(n["p1_factor"], s.p1Factor);
    readInt(n["p2_factor"], s.p2Factor);
    readEnum(n["mode"], s.mode);
    readInt(n["uniqueness_ratio"], s.uniquenessRatio);
    readInt(n["pre_filter_cap"], s.preFilterCap);
    readInt(n["disp12_max_diff"], s.disp12MaxDiff);
    readInt(n["speckle_window_size"], s.speckleWindowSize);
    readInt(n["speckle_range"], s.speckleRange);
    readEnum(n["preprocess"], s.preprocess);
    readDouble(n["clahe_clip_limit"], s.claheClipLimit);
    readInt(n["clahe_tile_grid"], s.claheTileGrid);
    readEnum(n["postprocess"], s.postprocess);
    readDouble(n["lr_max_diff_px"], s.lrMaxDiffPx);
    readDouble(n["wls_lambda"], s.wlsLambda);
    readDouble(n["wls_sigma_color"], s.wlsSigmaColor);
    readInt(n["wls_discontinuity_radius"], s.wlsDiscontinuityRadius);
    readDouble(n["confidence_threshold"], s.confidenceThreshold);
    readDouble(n["inference_radius_at_640_px"], s.inferenceRadiusAt640);
    readInt(n["median_size"], s.medianSize);
    readDouble(n["median_max_span_px"], s.medianMaxSpanPx);
    readDouble(n["preserve_trusted_jump_px"], s.preserveTrustedJumpPx);
    readInt(n["post_speckle_size"], s.postSpeckleSize);
    readDouble(n["post_speckle_range_px"], s.postSpeckleRangePx);
    readEnum(n["binary_kernel"], s.binaryKernel);
    readBool(n["binary_symmetric_subpixel"], s.binarySymmetricSubpixel);
    readBool(n["binary_average_speckle_removal"], s.binaryAverageSpeckleRemoval);
}

inline void validate(const Settings& s)
{
    if (s.blockSize < 1 || s.blockSize % 2 == 0)
        throw std::invalid_argument("block_size must be a positive odd number");
    if (s.p1Factor <= 0 || s.p2Factor <= s.p1Factor)
        throw std::invalid_argument("p2_factor must be greater than p1_factor, both positive");
    if (s.speckleWindowSize < 0 || s.speckleRange < 0)
        throw std::invalid_argument("speckle_window_size / speckle_range must not be negative");
    if (!(s.lrMaxDiffPx > 0) || !(s.wlsLambda > 0) || !(s.wlsSigmaColor > 0))
        throw std::invalid_argument("lr_max_diff_px, wls_lambda and wls_sigma_color must be positive");
    if (s.confidenceThreshold < 0 || s.confidenceThreshold > 255)
        throw std::invalid_argument("confidence_threshold must be within [0, 255]");
    if (s.inferenceRadiusAt640 < 0)
        throw std::invalid_argument("inference_radius_at_640_px must not be negative");
    if (s.medianSize != 0 && s.medianSize != 3 && s.medianSize != 5 && s.medianSize != 7)
        throw std::invalid_argument("median_size must be 0, 3, 5 or 7");
    if (s.postSpeckleSize < 0 || s.postSpeckleRangePx < 0)
        throw std::invalid_argument("post_speckle_size / post_speckle_range_px must not be negative");
    if (s.claheTileGrid < 1)
        throw std::invalid_argument("clahe_tile_grid must be positive");
    if (s.explicitNumDisparities > 0 && s.explicitNumDisparities % 16 != 0)
        throw std::invalid_argument("num_disparities must be a multiple of 16");
    if (s.matcher == Matcher::BinarySGBM) {
        // StereoBinarySGBM only defines MODE_SGBM and MODE_HH. Refuse the other
        // two instead of silently running MODE_SGBM: in the SGBM matcher they
        // mean something (4-way / 3-way), so a quiet downgrade would make the
        // config lie about what was measured.
        if (s.mode == SgbmMode::HH4 || s.mode == SgbmMode::ThreeWay)
            throw std::invalid_argument(std::string("mode must be sgbm or hh when matcher is "
                                        "\"binarysgbm\" (StereoBinarySGBM has only those two); got ") +
                                        toModeName(s.mode));
        // The census descriptors are defined on small kernels: xstereo asserts
        // kernelSize <= {5,11,7,7,9} depending on the variant. block_size 3 is
        // legal for all of them; larger values are only legal for some.
        static const char* kKernelNames[] = {"dense", "sparse", "cs", "mod_cs", "mod_census"};
        const int maxBlock = binaryMaxBlockSize(s.binaryKernel);
        if (s.blockSize < 3 || s.blockSize > maxBlock)
            throw std::invalid_argument(std::string("block_size must be an odd number in [3, ") +
                                        std::to_string(maxBlock) + "] for binary_kernel \"" +
                                        kKernelNames[static_cast<int>(s.binaryKernel)] +
                                        "\"; xstereo asserts kernelSize <= " +
                                        std::to_string(maxBlock));
        if (s.disp12MaxDiff <= 0 || s.preFilterCap <= 0)
            throw std::invalid_argument("disp12_max_diff and pre_filter_cap must be positive for "
                                        "matcher \"binarysgbm\"");
    }
}

}  // namespace detail

// Read a camera.json (the format written by the capture scripts) for fx/baseline.
inline void applyCameraFile(const std::string& path, Config& config)
{
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) throw std::runtime_error("cannot open camera file: " + path);
    const cv::FileNode root = fs.root();
    if (!detail::present(root["fx"]) || !detail::present(root["baseline_m"]))
        throw std::runtime_error("camera file needs 'fx' and 'baseline_m': " + path);
    detail::readDouble(root["fx"], config.fx);
    detail::readDouble(root["baseline_m"], config.baseline);
    config.cameraFile = path;
}

inline Config loadConfig(const std::string& path)
{
    // OpenCV reports a malformed JSON with a generic persistence error, so add
    // the file name and a hint before letting the message through.
    // NOTE: a FileNode borrows the FileStorage's memory, so the storage must
    // stay alive for as long as `root` is used.
    Config config;
    cv::FileStorage fs;
    try {
        fs.open(path, cv::FileStorage::READ);
        if (!fs.isOpened()) throw std::runtime_error("cannot open config file: " + path);
    } catch (const cv::Exception& error) {
        throw std::runtime_error("cannot parse config file '" + path +
                                 "' as JSON (comments are not allowed, and numbers need "
                                 "a decimal point): " + error.what());
    }
    const cv::FileNode root = fs.root();
    if (root.empty()) throw std::runtime_error("config file is empty: " + path);

    Settings& s = config.settings;

    // --- 1) top-level keys are the base -----------------------------------
    detail::readMatcherSettings(root, s);

    // --- 2) a named profile overrides the keys it defines -----------------
    // Same order as sgbm_pipeline.resolve_settings(): base < profile.
    std::string profile;
    detail::readString(root["profile"], profile);
    if (!profile.empty()) {
        const cv::FileNode entry = root["profiles"][profile];
        if (entry.empty())
            throw std::invalid_argument("profile '" + profile + "' is not defined in " + path);
        detail::readMatcherSettings(entry, s);
    }

    // --- search window ----------------------------------------------------
    const cv::FileNode depth = root["working_depth_m"];
    if (detail::present(depth)) {
        if (!depth.isSeq() || depth.size() != 2)
            throw std::invalid_argument("working_depth_m must be [near_m, far_m]");
        s.workingNearM = depth[0].real();
        s.workingFarM = depth[1].real();
    }
    detail::readInt(root["min_disparity"], s.explicitMinDisparity);
    detail::readInt(root["num_disparities"], s.explicitNumDisparities);
    detail::readBool(root["padding"], s.padding);

    // --- camera ------------------------------------------------------------
    const cv::FileNode camera = root["camera"];
    if (detail::present(camera)) {
        detail::readDouble(camera["fx"], config.fx);
        detail::readDouble(camera["baseline_m"], config.baseline);
    }
    detail::readString(root["camera_file"], config.cameraFile);

    const cv::FileNode output = root["output"];
    if (detail::present(output)) {
        detail::readString(output["directory"], config.outputDirectory);
        detail::readString(output["prefix"], config.outputPrefix);
        detail::readBool(output["save_artifacts"], config.saveArtifacts);
        detail::readBool(output["show_window"], config.showWindow);
        const cv::FileNode range = output["preview_disparity_range_px"];
        if (detail::present(range)) {
            if (!range.isSeq() || range.size() != 2)
                throw std::invalid_argument("output.preview_disparity_range_px must be [near_px, far_px]");
            config.previewNearPx = range[0].real();
            config.previewFarPx = range[1].real();
        }
    }

    detail::validate(s);
    if (config.cameraFile.empty()) {
        if (!(config.fx > 0) || !(config.baseline > 0))
            throw std::invalid_argument("fx and baseline_m must be positive (set them in "
                                        "the config or point camera_file at a camera.json)");
    } else {
        applyCameraFile(config.cameraFile, config);
    }
    return config;
}

// ---------------------------------------------------------------------------
// Small helpers used by main() only.
// ---------------------------------------------------------------------------

// Percentile over the finite, positive entries. Used when the config leaves
// output.preview_disparity_range_px at [0, 0] ("derive from the result").
inline bool disparityPercentiles(const cv::Mat& values, double lowPct, double highPct,
                                 double& low, double& high)
{
    std::vector<float> pool;
    pool.reserve(static_cast<std::size_t>(values.total()));
    for (int y = 0; y < values.rows; ++y) {
        const float* row = values.ptr<float>(y);
        for (int x = 0; x < values.cols; ++x)
            if (finite(row[x]) && row[x] > 0.0f) pool.push_back(row[x]);
    }
    if (pool.empty()) return false;
    const auto pick = [&pool](double pct) {
        const std::size_t index = static_cast<std::size_t>(
            std::min(std::max(pct / 100.0, 0.0), 1.0) * static_cast<double>(pool.size() - 1));
        std::nth_element(pool.begin(), pool.begin() + static_cast<std::ptrdiff_t>(index), pool.end());
        return static_cast<double>(pool[index]);
    };
    low = pick(lowPct);
    high = pick(highPct);
    if (!(high > low)) high = low + 1.0;      // keep the colour ramp non-degenerate
    return true;
}

inline double validFraction(const cv::Mat& map)
{
    return cv::countNonZero(finiteMask(map)) * 100.0 / static_cast<double>(map.total());
}

inline std::string directoryOf(const std::string& path)
{
    const std::size_t slash = path.find_last_of("/\\");
    return slash == std::string::npos ? std::string(".") : path.substr(0, slash);
}

inline std::string joinPath(const std::string& directory, const std::string& name)
{
    if (directory.empty()) return name;
    const char last = directory.back();
    return (last == '/' || last == '\\') ? directory + name : directory + "/" + name;
}

}  // namespace sgbmpipe

// ---------------------------------------------------------------------------
// Command line:  algorithm_test <left.png> <right.png> [sgbm_config.json]
//
// Every tunable parameter lives in the JSON; the CLI only names the inputs.
// Use "-" or omit the config argument to run with the built-in defaults.
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    using namespace sgbmpipe;

    if (argc < 3 || argc > 4) {
        std::cerr << "usage: " << (argc > 0 ? argv[0] : "algorithm_test")
                  << " <left.png> <right.png> [sgbm_config.json]\n"
                     "\n"
                     "  <left.png>            left image (rectified)\n"
                     "  <right.png>           right image, same size, same row alignment\n"
                     "  [sgbm_config.json]    optional; omit for built-in defaults\n"
                     "\n"
                     "All disparity parameters come from the JSON. Paths inside the\n"
                     "JSON (output.directory, camera_file) are resolved relative to the\n"
                     "working directory, not to the JSON file.\n";
        return 1;
    }

    const std::string leftPath = argv[1];
    const std::string rightPath = argv[2];
    const std::string configPath = (argc == 4) ? argv[3] : std::string();

    try {
        const Config config =
            (configPath.empty() || configPath == "-") ? Config() : loadConfig(configPath);

        cv::Mat leftBgr = cv::imread(leftPath, cv::IMREAD_COLOR);
        cv::Mat rightBgr = cv::imread(rightPath, cv::IMREAD_COLOR);
        if (leftBgr.empty()) throw std::runtime_error("could not read left image: " + leftPath);
        if (rightBgr.empty()) throw std::runtime_error("could not read right image: " + rightPath);
        if (leftBgr.size() != rightBgr.size())
            throw std::runtime_error("left and right images differ in size");
        if (leftBgr.channels() != 3 || rightBgr.channels() != 3)
            throw std::runtime_error("expected 3-channel images");

        // The pipeline hands RGB-ordered arrays to OpenCV; match that byte order.
        cv::Mat leftRgb, rightRgb;
        cv::cvtColor(leftBgr, leftRgb, cv::COLOR_BGR2RGB);
        cv::cvtColor(rightBgr, rightRgb, cv::COLOR_BGR2RGB);

        const Resolved resolved =
            resolve(leftBgr.cols, leftBgr.rows, config.fx, config.baseline, config.settings);

        std::printf("config: %s\n", configPath.empty() || configPath == "-"
                                        ? "<built-in defaults>" : configPath.c_str());
        std::printf("input : %s + %s  (%dx%d)\n", leftPath.c_str(), rightPath.c_str(),
                    leftBgr.cols, leftBgr.rows);
        std::printf("camera: fx=%.4f  baseline=%.6f m  (fx*baseline=%.4f)\n",
                    config.fx, config.baseline, config.fx * config.baseline);
        std::printf("search: minDisparity=%d numDisparities=%d padding=%d "
                    "limits=[%.2f, %.2f]px\n",
                    resolved.minDisparity, resolved.numDisparities, resolved.padding,
                    resolved.limitLow, resolved.limitHigh);
        static const char* kKernelNames[] = {"dense", "sparse", "cs", "mod_cs", "mod_census"};
        std::printf("matcher: %s block=%d P1=%d P2=%d mode=%s pfc=%d ur=%d disp12=%d speckle=%d/%d\n",
                    resolved.s.matcher == Matcher::BinarySGBM ? "binarysgbm" : "sgbm",
                    resolved.s.blockSize,
                    resolved.s.p1Factor * resolved.s.blockSize * resolved.s.blockSize,
                    resolved.s.p2Factor * resolved.s.blockSize * resolved.s.blockSize,
                    toModeName(resolved.s.mode), resolved.s.preFilterCap,
                    resolved.s.uniquenessRatio, resolved.s.disp12MaxDiff,
                    resolved.s.speckleWindowSize, resolved.s.speckleRange);
        if (resolved.s.matcher == Matcher::BinarySGBM) {
            std::printf("binary : kernel=%s subpixel=%s speckle_removal=%s "
                        "search=[0, %d]px (xstereo always starts at 0)\n",
                        kKernelNames[static_cast<int>(resolved.s.binaryKernel)],
                        resolved.s.binarySymmetricSubpixel ? "symmetric" : "quadratic",
                        resolved.s.binaryAverageSpeckleRemoval ? "avg" : "strict",
                        binaryNumDisparities(resolved));
        }
        std::printf("pre/post: preprocess=%s postprocess=%s lr=%.2fpx lambda=%.1f sigma=%.2f "
                    "infer=%.2fpx median=%d speckle=%d/%.2f\n",
                    resolved.s.preprocess == Preprocess::Clahe ? "clahe" : "none",
                    resolved.s.postprocess == Postprocess::Wls ? "wls" : "none",
                    resolved.s.lrMaxDiffPx, resolved.s.wlsLambda, resolved.s.wlsSigmaColor,
                    resolved.inferenceRadiusPx, resolved.s.medianSize,
                    resolved.s.postSpeckleSize, resolved.s.postSpeckleRangePx);

        const Stages stages = matchStereo(leftRgb, rightRgb, resolved);

        std::printf("\ncoverage: raw %.2f%%  checked %.2f%%  candidate %.2f%%  final %.2f%%"
                    "  inferred %d px\n",
                    validFraction(stages.raw), validFraction(stages.checked),
                    validFraction(stages.candidate), validFraction(stages.finalMap),
                    cv::countNonZero(stages.inferred));
        std::printf("timing  : matching %.1f ms, total %.1f ms\n",
                    stages.matchingMs, stages.totalMs);

        double accounted = 0.0;
        std::printf("\n--- stage timings (%s build) ---\n", kBuildType);
        for (const auto& stage : stages.timings) {
            accounted += stage.second;
            std::printf("  %-30s %8.2f ms  %5.1f%%\n", stage.first.c_str(), stage.second,
                        stage.second * 100.0 / stages.totalMs);
        }
        std::printf("  %-30s %8.2f ms  %5.1f%%\n", "accounted", accounted,
                    accounted * 100.0 / stages.totalMs);
        std::printf("  %-30s %8.2f ms\n", "total", stages.totalMs);

        // --- artifacts -------------------------------------------------------
        std::string outputDirectory = config.outputDirectory;
        if (outputDirectory.empty()) outputDirectory = directoryOf(leftPath);
        const std::string prefix = joinPath(outputDirectory, config.outputPrefix);
        std::string finalPreview;

        if (config.saveArtifacts) {
            double previewNearPx = config.previewNearPx;
            double previewFarPx = config.previewFarPx;
            if (!(previewFarPx > previewNearPx)) {
                // Not configured (or invalid): derive a stable range from the result.
                if (!disparityPercentiles(stages.finalMap, 1.0, 99.0, previewNearPx, previewFarPx))
                    throw std::runtime_error("the result has no valid disparity; nothing to preview");
                std::printf("\npreview range: derived [%.2f, %.2f]px (1-99%% of the result)\n",
                            previewNearPx, previewFarPx);
            } else {
                std::printf("\npreview range: configured [%.2f, %.2f]px\n",
                            previewNearPx, previewFarPx);
            }

            finalPreview = prefix + "disparity_final_left_vis.png";
            saveUint16(stages.raw, 256.0, prefix + "disparity_raw_left_16.png");
            saveUint16(stages.finalMap, 256.0, prefix + "disparity_final_left_16.png");
            saveDisparityPreview(stages.finalMap, previewNearPx, previewFarPx, finalPreview);
            saveMask(finiteMask(stages.finalMap), prefix + "prediction_valid_mask_left.png");
            saveMask(stages.inferred, prefix + "postprocess_inferred_mask_left.png");
            std::printf("wrote %s (and raw/final 16-bit + masks)\n", finalPreview.c_str());
        } else {
            std::printf("\noutput.save_artifacts is false; nothing written to disk\n");
        }

        // --- display ---------------------------------------------------------
        if (!config.showWindow || finalPreview.empty()) return 0;
        if (!std::getenv("DISPLAY") && !std::getenv("WAYLAND_DISPLAY")) {
            std::printf("No display detected; skipping imshow.\n");
            return 0;
        }
        const cv::Mat colored = cv::imread(finalPreview, cv::IMREAD_COLOR);
        if (colored.empty()) {
            std::cerr << "Error: could not re-read the preview image for display.\n";
            return 3;
        }
        try {
            const std::string windowName = "Disparity (final)";
            cv::namedWindow(windowName, cv::WINDOW_NORMAL);
            cv::imshow(windowName, colored);
            cv::waitKey(0);              // any key closes the window
            cv::destroyAllWindows();
        } catch (const cv::Exception& error) {
            // A GUI backend can be missing even with DISPLAY set; the image is
            // already on disk, so report and continue.
            std::cerr << "Display failed (" << error.what() << "); see "
                      << finalPreview << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 2;
    }
}