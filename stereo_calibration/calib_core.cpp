// ===========================================================================
//  Stereo calibration for OpenCV 5 — pinhole model, planar chessboard target.
//
//  Pipeline
//    1. discover left/right image pairs by numeric suffix
//    2. detect the board with findChessboardCorners, refine with cornerSubPix
//    3. calibrate each camera on its own            -> K1,D1 / K2,D2
//    4. stereoCalibrate with the intrinsics frozen  -> R,T,E,F
//    5. stereoRectify + initUndistortRectifyMap     -> R1,R2,P1,P2,Q
//    6. verify the result — a small RMS alone does not prove the extrinsics are
//       usable, so two independent checks are run:
//         a. epipolar error after rectification: corresponding corners must end
//            up on the same row, so |dy| -> 0
//         b. triangulate the rectified corners: the board must reconstruct as a
//            plane, and its square edge must match the configured size, which
//            catches a wrong baseline or a wrong --square-size
//
//  OpenCV 5 note: calib3d has been split across modules. The functions used
//  here live in
//      calib.hpp        calibrateCamera, stereoCalibrate
//      stereo.hpp       stereoRectify, STEREO_ZERO_DISPARITY
//      objdetect.hpp    findChessboardCorners, drawChessboardCorners
//      imgproc.hpp      cornerSubPix, initUndistortRectifyMap, remap
//      geometry/3d.hpp  undistortPoints, triangulatePoints, projectPoints
//  <opencv2/opencv.hpp> does not pull in geometry, hence the explicit include.
// ===========================================================================
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdarg>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/calib.hpp>
#include <opencv2/stereo.hpp>
#include <opencv2/geometry/3d.hpp>
#include "calib_core.hpp"


namespace calib {

// ---------------------------------------------------------------------------
// Logging. Everything the pipeline reports goes through print() so the GUI can
// show the same text the CLI writes to stdout. With no sink installed the bytes
// go to stdout unchanged, which is exactly what the CLI relies on.
// ---------------------------------------------------------------------------
namespace {

LogSink& sink_ref()
{
    static LogSink sink;
    return sink;
}

bool& enabled_ref()
{
    static bool enabled = true;
    return enabled;
}

}  // namespace

void setLogSink(LogSink sink)
{
    sink_ref() = std::move(sink);
}

void clearLogSink()
{
    sink_ref() = nullptr;
}

void setLogEnabled(bool enabled)
{
    enabled_ref() = enabled;
}

void print(const char* format, ...)
{
    if (!enabled_ref()) return;

    va_list args;
    va_start(args, format);
    char stack[1024];
    va_list copy;
    va_copy(copy, args);
    const int needed = std::vsnprintf(stack, sizeof(stack), format, copy);
    va_end(copy);

    std::string message;
    if (needed < 0) {
        message = "log formatting failed";
    } else if (needed < static_cast<int>(sizeof(stack))) {
        message.assign(stack, static_cast<std::size_t>(needed));
    } else {
        message.resize(static_cast<std::size_t>(needed) + 1);
        std::vsnprintf(&message[0], message.size(), format, args);
        message.resize(static_cast<std::size_t>(needed));
    }
    va_end(args);

    const LogSink& sink = sink_ref();
    if (sink) {
        sink(message);
    } else {
        std::fwrite(message.data(), 1, message.size(), stdout);
    }
}

// ---------------------------------------------------------------------------
// Stage timer: one line per stage in the summary.
// ---------------------------------------------------------------------------
// Wall-clock timer, one line per stage in the summary.
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

// ---------------------------------------------------------------------------
// 1. Image discovery
// ---------------------------------------------------------------------------
// Trailing digit group of a filename stem, so left01.jpg -> "01". Files without
// a numeric tail are ignored: they are not part of a numbered pair.
inline std::string numericSuffix(const std::string& stem, const std::string& prefix)
{
    if (stem.rfind(prefix, 0) != 0) return {};
    std::size_t pos = prefix.size();
    while (pos < stem.size() && std::isdigit(static_cast<unsigned char>(stem[pos]))) ++pos;
    if (pos == prefix.size()) return {};
    return stem.substr(prefix.size(), pos - prefix.size());
}

std::vector<ImagePair> discoverPairs(const Settings& s)
{
    if (!fs::is_directory(s.imageDirectory))
        throw std::runtime_error("image directory does not exist: " + s.imageDirectory);

    std::vector<std::pair<std::string, fs::path>> left, right;
    for (const auto& entry : fs::directory_iterator(s.imageDirectory)) {
        if (!entry.is_regular_file()) continue;
        const fs::path& path = entry.path();
        if (path.extension() != s.extension) continue;
        const std::string stem = path.stem().string();
        if (const std::string key = numericSuffix(stem, s.leftPrefix); !key.empty())
            left.emplace_back(key, path);
        else if (const std::string key2 = numericSuffix(stem, s.rightPrefix); !key2.empty())
            right.emplace_back(key2, path);
    }
    if (left.empty() || right.empty())
        throw std::runtime_error("no " + s.leftPrefix + "*/" + s.rightPrefix + "*" + s.extension +
                                 " files found in " + s.imageDirectory);

    const auto byKey = [](const auto& a, const auto& b) { return a.first < b.first; };
    std::sort(left.begin(), left.end(), byKey);
    std::sort(right.begin(), right.end(), byKey);

    // Pair by suffix, and report asymmetric indices instead of silently dropping
    // them: this dataset has no index 10 on either side.
    std::vector<ImagePair> pairs;
    std::vector<std::string> onlyLeft, onlyRight;
    std::size_t i = 0, j = 0;
    while (i < left.size() || j < right.size()) {
        if (i < left.size() && j < right.size() && left[i].first == right[j].first) {
            pairs.push_back({left[i].first, left[i].second, right[j].second});
            ++i; ++j;
        } else if (j >= right.size() || (i < left.size() && left[i].first < right[j].first)) {
            onlyLeft.push_back(left[i].first); ++i;
        } else {
            onlyRight.push_back(right[j].first); ++j;
        }
    }
    if (!onlyLeft.empty() || !onlyRight.empty()) {
        print("  warning: unpaired images — left only:");
        for (const auto& k : onlyLeft) print(" %s", k.c_str());
        print("  right only:");
        for (const auto& k : onlyRight) print(" %s", k.c_str());
        print("\n");
    }

    // Views the caller asked to drop (GUI: "exclude and re-run"). The CLI leaves
    // the list empty, so this block never prints there.
    if (!s.excludedKeys.empty()) {
        std::vector<ImagePair> remaining;
        for (const auto& pair : pairs) {
            if (std::find(s.excludedKeys.begin(), s.excludedKeys.end(), pair.key) !=
                s.excludedKeys.end()) {
                print("  excluded on request: %s\n", pair.key.c_str());
            } else {
                remaining.push_back(pair);
            }
        }
        pairs = remaining;
    }
    return pairs;
}

// ---------------------------------------------------------------------------
// 2. Corner detection
// ---------------------------------------------------------------------------
std::vector<cv::Point2f> detectCorners(const cv::Mat& gray, const Settings& s, bool& usedSb)
{
    const cv::Size pattern(s.boardWidth, s.boardHeight);
    std::vector<cv::Point2f> corners;
    usedSb = false;

    if (s.useSbDetector) {
        // Slower but far more robust on blurry or low-contrast targets, and it
        // already returns subpixel-accurate corners.
        if (cv::findChessboardCornersSB(gray, pattern, corners,
                                        cv::CALIB_CB_EXHAUSTIVE | cv::CALIB_CB_ACCURACY)) {
            usedSb = true;
            return corners;
        }
        return {};
    }

    const int flags = cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE;
    if (!cv::findChessboardCorners(gray, pattern, corners, flags)) return {};
    cv::cornerSubPix(gray, corners,
                     cv::Size(s.subpixWindow, s.subpixWindow), cv::Size(-1, -1),
                     cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT,
                                      s.subpixIterations, s.subpixEpsilon));
    return corners;
}

// Detects the board in every pair. Reports the corner count per image so a
// partially visible board shows up in the log instead of quietly reducing the
// number of calibration views.
std::vector<std::vector<cv::Point2f>> detectAll(const std::vector<ImagePair>& pairs,
                                                const Settings& s,
                                                std::vector<std::vector<cv::Point2f>>& rightCorners,
                                                std::vector<ImagePair>& kept,
                                                bool verbose, const RunHooks& hooks,
                                                std::vector<PairStatus>* status)
{
    std::vector<std::vector<cv::Point2f>> leftCorners;
    const int expected = s.boardWidth * s.boardHeight;
    const int total = static_cast<int>(pairs.size());
    int done = 0;
    for (const auto& pair : pairs) {
        if (hooks.cancel && hooks.cancel->load()) throw Cancelled();
        if (hooks.progress) hooks.progress("detect", done, total);
        ++done;

        const cv::Mat leftImage  = cv::imread(pair.left.string(), cv::IMREAD_GRAYSCALE);
        const cv::Mat rightImage = cv::imread(pair.right.string(), cv::IMREAD_GRAYSCALE);
        if (leftImage.empty() || rightImage.empty()) {
            print("  %-4s skipped: could not read an image\n", pair.key.c_str());
            if (status) status->push_back({pair.key, false, "could not read an image"});
            continue;
        }
        bool sbLeft = false, sbRight = false;
        std::vector<cv::Point2f> lc = detectCorners(leftImage, s, sbLeft);
        std::vector<cv::Point2f> rc = detectCorners(rightImage, s, sbRight);
        if (static_cast<int>(lc.size()) != expected || static_cast<int>(rc.size()) != expected) {
            print("  %-4s skipped: %zu/%zu corners, expected %d\n",
                        pair.key.c_str(), lc.size(), rc.size(), expected);
            if (status) status->push_back({pair.key, false,
                std::to_string(lc.size()) + "/" + std::to_string(rc.size()) +
                " corners, expected " + std::to_string(expected)});
            continue;
        }
        if (verbose)
            print("  %-4s %d corners per eye%s\n", pair.key.c_str(), expected,
                        (sbLeft || sbRight) ? "  (SB detector)" : "");
        leftCorners.push_back(std::move(lc));
        rightCorners.push_back(std::move(rc));
        kept.push_back(pair);
        if (status) status->push_back({pair.key, true, ""});
    }
    if (hooks.progress) hooks.progress("detect", total, total);
    return leftCorners;
}

// Restores the correspondence when only one camera saw the target "mirrored":
// the detector can latch onto a different starting corner, so every pair would
// violate the epipolar constraint and the optimiser would return a 180° R.
// The test compares the summed grid-edge direction, which is stable across
// rotations but flips sign under mirroring.
bool reverseIfMirrored(const std::vector<cv::Point2f>& left,
                       std::vector<cv::Point2f>& right, int boardWidth)
{
    cv::Point2f leftEdge(0.f, 0.f), rightEdge(0.f, 0.f);
    const int rows = static_cast<int>(left.size()) / boardWidth;
    for (int y = 0; y < rows; ++y) {
        const int rowStart = y * boardWidth;
        leftEdge  += left[rowStart + boardWidth - 1]  - left[rowStart];
        rightEdge += right[rowStart + boardWidth - 1] - right[rowStart];
    }
    if (leftEdge.dot(rightEdge) >= 0.f) return false;
    std::reverse(right.begin(), right.end());
    return true;
}

// ---------------------------------------------------------------------------
// 3./4. Calibration
// ---------------------------------------------------------------------------
std::vector<cv::Point3f> boardObjectPoints(const Settings& s)
{
    std::vector<cv::Point3f> points;
    points.reserve(static_cast<std::size_t>(s.boardWidth) * s.boardHeight);
    for (int y = 0; y < s.boardHeight; ++y)
        for (int x = 0; x < s.boardWidth; ++x)
            points.emplace_back(static_cast<float>(x * s.squareSize),
                                static_cast<float>(y * s.squareSize), 0.0f);
    return points;
}

CameraModel calibrateMono(const std::vector<std::vector<cv::Point3f>>& objectPoints,
                          const std::vector<std::vector<cv::Point2f>>& imagePoints,
                          const cv::Size& imageSize, const char* label)
{
    CameraModel model;
    std::vector<cv::Mat> rvecs, tvecs;
    const cv::TermCriteria criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1e-9);
    model.rms = cv::calibrateCamera(objectPoints, imagePoints, imageSize,
                                    model.K, model.D, rvecs, tvecs, 0, criteria);
    if (model.D.total() < 5)
        throw std::runtime_error(std::string(label) +
                                 ": expected at least 5 distortion coefficients, got " +
                                 std::to_string(model.D.total()));

    // Recompute the per-view error from projectPoints rather than trusting
    // calibrateCamera's own bookkeeping, so the numbers in the report are
    // directly comparable with the stereo stage.
    model.perViewErrors.reserve(imagePoints.size());
    for (std::size_t view = 0; view < imagePoints.size(); ++view) {
        std::vector<cv::Point2f> projected;
        cv::projectPoints(objectPoints[view], rvecs[view], tvecs[view],
                          model.K, model.D, projected);
        double sum = 0.0;
        for (std::size_t k = 0; k < projected.size(); ++k) {
            const cv::Point2f d = projected[k] - imagePoints[view][k];
            sum += d.x * d.x + d.y * d.y;
        }
        model.perViewErrors.push_back(std::sqrt(sum / static_cast<double>(projected.size())));
    }
    return model;
}

StereoModel calibrateStereoPair(const std::vector<std::vector<cv::Point3f>>& objectPoints,
                                const std::vector<std::vector<cv::Point2f>>& left,
                                const std::vector<std::vector<cv::Point2f>>& right,
                                const cv::Size& imageSize, CameraModel& camera1,
                                CameraModel& camera2, const Settings& s)
{
    StereoModel model;
    // Two-step recipe: per-camera intrinsics first, then the extrinsics with the
    // intrinsics frozen. This is what the OpenCV sample does and it converges far
    // more reliably than optimising all 9+5+9+5+6 parameters in one shot.
    int flags = s.refineIntrinsics ? cv::CALIB_USE_INTRINSIC_GUESS : cv::CALIB_FIX_INTRINSIC;
    if (s.sameFocalLength) flags |= cv::CALIB_SAME_FOCAL_LENGTH;
    const cv::TermCriteria criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1e-9);

    model.rms = cv::stereoCalibrate(objectPoints, left, right,
                                    camera1.K, camera1.D, camera2.K, camera2.D, imageSize,
                                    model.R, model.T, model.E, model.F,
                                    model.perViewErrors, flags, criteria);
    if (model.perViewErrors.rows != static_cast<int>(left.size()) ||
        model.perViewErrors.cols != 2)
        throw std::runtime_error("stereoCalibrate returned an unexpected perViewErrors shape");

    cv::stereoRectify(camera1.K, camera1.D, camera2.K, camera2.D, imageSize,
                      model.R, model.T, model.R1, model.R2, model.P1, model.P2, model.Q,
                      cv::STEREO_ZERO_DISPARITY, s.alpha, imageSize,
                      &model.roi1, &model.roi2);
    return model;
}

// ---------------------------------------------------------------------------
// 5. Verification
// ---------------------------------------------------------------------------
// Maps the detected corners through undistort + rectification. With
// STEREO_ZERO_DISPARITY the two rows must coincide, so |y_left - y_right| is a
// direct measure of how usable the rectification is for dense stereo.
void appendEpipolarErrors(const std::vector<std::vector<cv::Point2f>>& left,
                          const std::vector<std::vector<cv::Point2f>>& right,
                          const CameraModel& camera1, const CameraModel& camera2,
                          const StereoModel& stereo, std::vector<double>& dy,
                          std::vector<cv::Point2f>* rectifiedLeft,
                          std::vector<cv::Point2f>* rectifiedRight)
{
    for (std::size_t view = 0; view < left.size(); ++view) {
        std::vector<cv::Point2f> rl, rr;
        cv::undistortPoints(left[view],  rl, camera1.K, camera1.D, stereo.R1, stereo.P1);
        cv::undistortPoints(right[view], rr, camera2.K, camera2.D, stereo.R2, stereo.P2);
        for (std::size_t k = 0; k < rl.size(); ++k)
            dy.push_back(std::abs(static_cast<double>(rl[k].y) - rr[k].y));
        if (rectifiedLeft && rectifiedRight) {
            rectifiedLeft->insert(rectifiedLeft->end(), rl.begin(), rl.end());
            rectifiedRight->insert(rectifiedRight->end(), rr.begin(), rr.end());
        }
    }
}

// RMS distance of the points to the plane that best fits them. The normal is the
// eigenvector of the smallest singular value of the centred point set.
double planarityRms(const std::vector<cv::Point3d>& points)
{
    if (points.size() < 3) return std::numeric_limits<double>::quiet_NaN();

    cv::Point3d centroid(0, 0, 0);
    for (const auto& p : points) centroid += p;
    centroid *= 1.0 / static_cast<double>(points.size());

    cv::Mat samples(static_cast<int>(points.size()), 3, CV_64F);
    for (std::size_t i = 0; i < points.size(); ++i) {
        samples.at<double>(static_cast<int>(i), 0) = points[i].x - centroid.x;
        samples.at<double>(static_cast<int>(i), 1) = points[i].y - centroid.y;
        samples.at<double>(static_cast<int>(i), 2) = points[i].z - centroid.z;
    }
    cv::Mat w, u, vt;
    cv::SVD::compute(samples, w, u, vt, cv::SVD::FULL_UV);
    const cv::Point3d normal(vt.at<double>(2, 0), vt.at<double>(2, 1), vt.at<double>(2, 2));

    double squared = 0.0;
    for (const auto& p : points) {
        const double d = normal.dot(p - centroid);
        squared += d * d;
    }
    return std::sqrt(squared / static_cast<double>(points.size()));
}

// 2xN CV_64F matrix, the layout triangulatePoints documents.
cv::Mat toPointMatrix(const std::vector<cv::Point2f>& points)
{
    cv::Mat matrix(2, static_cast<int>(points.size()), CV_64F);
    for (int i = 0; i < static_cast<int>(points.size()); ++i) {
        matrix.at<double>(0, i) = points[i].x;
        matrix.at<double>(1, i) = points[i].y;
    }
    return matrix;
}

Verification verify(const std::vector<std::vector<cv::Point2f>>& left,
                    const std::vector<std::vector<cv::Point2f>>& right,
                    const CameraModel& camera1, const CameraModel& camera2,
                    const StereoModel& stereo, const Settings& s)
{
    Verification result;

    // --- a. epipolar alignment ---------------------------------------------
    std::vector<double> dy;
    std::vector<cv::Point2f> rectLeft, rectRight;
    appendEpipolarErrors(left, right, camera1, camera2, stereo, dy, &rectLeft, &rectRight);
    if (!dy.empty()) {
        std::vector<double> sorted = dy;
        std::sort(sorted.begin(), sorted.end());
        const double sum = std::accumulate(sorted.begin(), sorted.end(), 0.0);
        const std::size_t p95Index = std::min(sorted.size() - 1,
            static_cast<std::size_t>(0.95 * static_cast<double>(sorted.size())));
        result.epipolarMean = sum / static_cast<double>(sorted.size());
        result.epipolarP95  = sorted[p95Index];
        result.epipolarMax  = sorted.back();
        result.comparedCorners = static_cast<int>(sorted.size());
    }

    // --- b. triangulated geometry ------------------------------------------
    // A planar target must reconstruct as a plane, and the grid spacing must
    // equal squareSize in metres -- which only holds when the baseline and the
    // square size agree. The plane is fitted per view: the 13 views have
    // different poses, so a single plane across all of them is meaningless.
    const int perView = s.boardWidth * s.boardHeight;
    double depthSum = 0.0;
    int depthCount = 0;
    double edgeSum = 0.0;
    int edgeCount = 0;
    double planaritySum = 0.0;
    int planarityViews = 0;
    double planarityWorst = 0.0;
    std::size_t validPoints = 0;
    const std::size_t expectedPoints = static_cast<std::size_t>(perView) * left.size();

    for (std::size_t view = 0; view < left.size(); ++view) {
        std::vector<cv::Point2f> rl, rr;
        cv::undistortPoints(left[view],  rl, camera1.K, camera1.D, stereo.R1, stereo.P1);
        cv::undistortPoints(right[view], rr, camera2.K, camera2.D, stereo.R2, stereo.P2);

        // triangulatePoints allocates its output with the depth of the input
        // points, so hand it CV_64F explicitly; feeding a vector<Point2f> would
        // silently produce a CV_32F 4xN matrix.
        cv::Mat points4d;
        cv::triangulatePoints(stereo.P1, stereo.P2, toPointMatrix(rl), toPointMatrix(rr),
                              points4d);

        std::vector<cv::Point3d> viewPoints;
        viewPoints.reserve(static_cast<std::size_t>(perView));
        for (int k = 0; k < points4d.cols; ++k) {
            const double w = points4d.at<double>(3, k);
            if (std::abs(w) < 1e-12) continue;
            const cv::Point3d p(points4d.at<double>(0, k) / w,
                                points4d.at<double>(1, k) / w,
                                points4d.at<double>(2, k) / w);
            // A corner that lands behind the camera means the pair was matched
            // wrongly; counting it would corrupt both statistics below.
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) || p.z <= 0)
                continue;
            viewPoints.push_back(p);
            depthSum += p.z;
            ++depthCount;
        }
        validPoints += viewPoints.size();
        if (static_cast<int>(viewPoints.size()) != perView) continue;

        for (int y = 0; y < s.boardHeight; ++y) {
            for (int x = 0; x + 1 < s.boardWidth; ++x) {
                edgeSum += cv::norm(viewPoints[y * s.boardWidth + x + 1] -
                                    viewPoints[y * s.boardWidth + x]);
                ++edgeCount;
            }
        }

        const double planarity = planarityRms(viewPoints);
        if (std::isfinite(planarity)) {
            planaritySum += planarity;
            planarityWorst = std::max(planarityWorst, planarity);
            ++planarityViews;
        }
    }

    if (validPoints < expectedPoints)
        print("  三角化：%zu/%zu 个角点得到正深度（其余被剔除）\n",
                    validPoints, expectedPoints);
    if (depthCount > 0) {
        result.triangulated = true;
        result.meanDepth = depthSum / depthCount;
    }
    if (planarityViews > 0) {
        result.planarityRms = planaritySum / planarityViews;
        result.planarityWorst = planarityWorst;
    }
    if (edgeCount > 0) {
        result.meanEdgeLength  = edgeSum / edgeCount;
        result.edgeLengthError = (result.meanEdgeLength - s.squareSize) / s.squareSize;
    }
    return result;
}

// ---------------------------------------------------------------------------
// 6. Output
// ---------------------------------------------------------------------------
void saveIntrinsics(const fs::path& path, const CameraModel& camera1, const CameraModel& camera2)
{
    cv::FileStorage fs(path.string(), cv::FileStorage::WRITE);
    if (!fs.isOpened()) throw std::runtime_error("cannot write " + path.string());
    fs << "M1" << camera1.K << "D1" << camera1.D
       << "M2" << camera2.K << "D2" << camera2.D
       << "rms1" << camera1.rms << "rms2" << camera2.rms;
}

void saveStereo(const fs::path& path, const StereoModel& stereo, const CameraModel& camera1,
                const CameraModel& camera2)
{
    cv::FileStorage fs(path.string(), cv::FileStorage::WRITE);
    if (!fs.isOpened()) throw std::runtime_error("cannot write " + path.string());
    fs << "M1" << camera1.K << "D1" << camera1.D
       << "M2" << camera2.K << "D2" << camera2.D
       << "R" << stereo.R << "T" << stereo.T
       << "E" << stereo.E << "F" << stereo.F
       << "R1" << stereo.R1 << "R2" << stereo.R2
       << "P1" << stereo.P1 << "P2" << stereo.P2
       << "Q" << stereo.Q
       << "baseline_m" << stereo.baseline()
       << "rms" << stereo.rms;
}

// Optional sanity check against a known-good monocular calibration: the sample
// dataset ships left_intrinsics.yml, so the absolute numbers can be compared
// instead of only their internal consistency.
bool compareWithReference(const fs::path& path, const CameraModel& camera, std::string& summary)
{
    cv::FileStorage fs(path.string(), cv::FileStorage::READ);
    if (!fs.isOpened()) return false;
    cv::Mat k, d;
    fs["camera_matrix"] >> k;
    fs["distortion_coefficients"] >> d;
    if (k.empty() || k.rows != 3 || k.cols != 3) return false;

    std::ostringstream os;
    const char* names[4] = {"fx", "fy", "cx", "cy"};
    const double mine[4] = {camera.K.at<double>(0, 0), camera.K.at<double>(1, 1),
                            camera.K.at<double>(0, 2), camera.K.at<double>(1, 2)};
    const double ref[4]  = {k.at<double>(0, 0), k.at<double>(1, 1),
                            k.at<double>(0, 2), k.at<double>(1, 2)};
    for (int i = 0; i < 4; ++i) {
        os << "  " << names[i] << "  " << std::fixed << std::setprecision(3) << mine[i]
           << "  vs  " << ref[i] << "   (" << std::showpos << std::setprecision(4)
           << (mine[i] - ref[i]) / ref[i] * 100.0 << std::noshowpos << " %)\n";
    }
    summary = os.str();
    return true;
}

// Side-by-side rectified pair with horizontal guides: a row mismatch, i.e. a bad
// rectification, is immediately visible.
cv::Mat rectifiedPreview(const ImagePair& pair, const CameraModel& camera1,
                         const CameraModel& camera2, const StereoModel& stereo,
                         const cv::Size& imageSize, cv::Mat& rectLeft, cv::Mat& rectRight)
{
    const cv::Mat leftImage  = cv::imread(pair.left.string(), cv::IMREAD_COLOR);
    const cv::Mat rightImage = cv::imread(pair.right.string(), cv::IMREAD_COLOR);
    if (leftImage.empty() || rightImage.empty())
        throw std::runtime_error("cannot read the pair used for the rectified preview");

    cv::Mat map1x, map1y, map2x, map2y;
    cv::initUndistortRectifyMap(camera1.K, camera1.D, stereo.R1, stereo.P1, imageSize,
                                CV_16SC2, map1x, map1y);
    cv::initUndistortRectifyMap(camera2.K, camera2.D, stereo.R2, stereo.P2, imageSize,
                                CV_16SC2, map2x, map2y);
    cv::remap(leftImage,  rectLeft,  map1x, map1y, cv::INTER_LINEAR);
    cv::remap(rightImage, rectRight, map2x, map2y, cv::INTER_LINEAR);

    cv::Mat canvas(imageSize.height, imageSize.width * 2, CV_8UC3);
    rectLeft.copyTo(canvas(cv::Rect(0, 0, imageSize.width, imageSize.height)));
    rectRight.copyTo(canvas(cv::Rect(imageSize.width, 0, imageSize.width, imageSize.height)));
    for (int y = 0; y < imageSize.height; y += 24) {
        const cv::Scalar color = (y % 120 == 0) ? cv::Scalar(0, 220, 0)
                                                : cv::Scalar(70, 120, 70);
        cv::line(canvas, cv::Point(0, y), cv::Point(canvas.cols, y), color, 1, cv::LINE_AA);
    }
    cv::putText(canvas, "left / right after rectification: matching corners must share a row",
                cv::Point(8, imageSize.height - 10), cv::FONT_HERSHEY_SIMPLEX, 0.42,
                cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    return canvas;
}

void printMatrix(const char* label, const cv::Mat& m)
{
    print("  %-4s", label);
    for (int r = 0; r < m.rows; ++r) {
        if (r) print("      ");
        print("[");
        for (int c = 0; c < m.cols; ++c) print(" %12.6f", m.at<double>(r, c));
        print(" ]\n");
    }
}

std::string buildMarkdown(const Settings& s, const std::vector<ImagePair>& pairs,
                          const cv::Size& imageSize, const CameraModel& camera1,
                          const CameraModel& camera2, const StereoModel& stereo,
                          const Verification& verification, const std::string& referenceSummary)
{
    std::ostringstream md;
    md << "# 双目标定报告\n\n## 输入\n\n| 项 | 值 |\n| --- | --- |\n";
    md << "| 图像目录 | `" << s.imageDirectory << "` |\n";
    md << "| 有效图像对 | " << pairs.size() << " |\n";
    md << "| 棋盘格内角点 | " << s.boardWidth << " × " << s.boardHeight << " |\n";
    md << "| 方格边长 | " << s.squareSize * 1000.0 << " mm |\n";
    md << "| 图像尺寸 | " << imageSize.width << " × " << imageSize.height << " |\n";
    md << "| 内参是否联合优化 | " << (s.refineIntrinsics ? "是" : "否（CALIB_FIX_INTRINSIC）") << " |\n";
    md << "| stereoRectify alpha | " << s.alpha << " |\n\n";

    md << "## 内参\n\n| 相机 | RMS (px) | fx | fy | cx | cy |\n"
          "| --- | --- | --- | --- | --- | --- |\n";
    for (int i = 0; i < 2; ++i) {
        const CameraModel& c = (i == 0) ? camera1 : camera2;
        md << "| " << (i == 0 ? "left" : "right") << " | " << std::fixed << std::setprecision(4)
           << c.rms << " | " << c.K.at<double>(0, 0) << " | " << c.K.at<double>(1, 1) << " | "
           << c.K.at<double>(0, 2) << " | " << c.K.at<double>(1, 2) << " |\n";
    }
    md << "\n畸变系数（k1, k2, p1, p2, k3）:\n\n";
    for (int i = 0; i < 2; ++i) {
        const CameraModel& c = (i == 0) ? camera1 : camera2;
        md << "- " << (i == 0 ? "left " : "right") << ": ";
        for (int k = 0; k < c.D.total(); ++k) {
            if (k) md << ", ";
            md << std::setprecision(6) << c.D.at<double>(k);
        }
        md << "\n";
    }

    md << "\n## 外参\n\n";
    md << "- 基线 `|T|` = " << std::fixed << std::setprecision(6) << stereo.baseline() << " m\n";
    md << "- 立体标定 RMS = " << std::setprecision(4) << stereo.rms << " px\n";
    md << "- R 与单位矩阵的偏差 = " << std::setprecision(4)
       << cv::norm(stereo.R - cv::Mat::eye(3, 3, CV_64F)) << "\n";

    md << "\n## 验证\n\n| 指标 | 值 | 判据 |\n| --- | --- | --- |\n";
    md << "| 校正后极线误差 均值 | " << std::setprecision(4) << verification.epipolarMean
       << " px | < 0.5 px |\n";
    md << "| 校正后极线误差 P95 | " << verification.epipolarP95 << " px | < 1 px |\n";
    md << "| 校正后极线误差 最大 | " << verification.epipolarMax << " px | — |\n";
    md << "| 参与统计的角点数 | " << verification.comparedCorners << " | — |\n";
    if (verification.triangulated) {
        md << "| 三角化平面性 RMS（各视图均值） | " << verification.planarityRms * 1000.0
           << " mm | ≪ 方格边长 |\n";
        md << "| 三角化平面性 RMS（最差视图） | " << verification.planarityWorst * 1000.0
           << " mm | ≪ 方格边长 |\n";
        md << "| 重建方格边长 | " << verification.meanEdgeLength * 1000.0
           << " mm | ≈ " << s.squareSize * 1000.0 << " mm |\n";
        md << "| 边长相对误差 | " << verification.edgeLengthError * 100.0 << " % | < 1 % |\n";
        md << "| 棋盘格平均深度 | " << verification.meanDepth << " m | — |\n";
    }
    md << "\n判据说明：极线误差衡量 R/T 与畸变模型是否能真正把对应点拉到同一行（这是"
          "稠密匹配的前提），它与重投影误差是两个独立的量；平面性独立于两者，反映畸变"
          "残差与靶标平整度。\n\n"
          "注意边长检验的适用范围：三角化坐标正比于基线，而基线正比于 `--square-size`，"
          "两者同比缩放，所以这条检验只能证明基线与焦距相互自洽（裁剪错位、坐标系不匹配"
          "会让它变大），不能证明绝对尺度。`--square-size` 本身填错时，重投影误差、极线"
          "误差、边长误差全部保持不变，只能用外部尺寸交叉验证。\n";
    if (!referenceSummary.empty()) {
        md << "\n### 与参考标定 `" << s.referenceIntrinsics << "` 对比\n\n```\n"
           << referenceSummary << "```\n";
    }
    return md.str();
}

// ---------------------------------------------------------------------------
// Whole run. The log text and its order are exactly what the CLI used to print
// from main(), so both front ends share one code path.
// ---------------------------------------------------------------------------
namespace {

void checkCancelled(const RunHooks& hooks)
{
    if (hooks.cancel && hooks.cancel->load()) throw Cancelled();
}

void reportStage(const RunHooks& hooks, const char* stage, int done, int total)
{
    if (hooks.progress) hooks.progress(stage, done, total);
}

}  // namespace

DetectResult detectPairs(const Settings& s, const RunHooks& hooks)
{
    DetectResult result;

    print("\n=== 1. 图像对 ===\n");
    result.pairs = discoverPairs(s);
    print("  找到 %zu 对图像\n", result.pairs.size());
    if (result.pairs.size() < 3)
        throw std::runtime_error("need at least 3 stereo pairs to calibrate");

    print("\n=== 2. 角点检测 ===\n");
    {
        StageTimer timer(result.timings, "detect corners");
        result.leftCorners = detectAll(result.pairs, s, result.rightCorners, result.kept,
                                       s.verbose, hooks, &result.status);
    }
    if (result.leftCorners.size() < 3)
        throw std::runtime_error("fewer than 3 pairs yielded a complete board");

    // Restore the correspondence wherever only one camera mirrored the order.
    for (std::size_t view = 0; view < result.leftCorners.size(); ++view) {
        if (reverseIfMirrored(result.leftCorners[view], result.rightCorners[view], s.boardWidth))
            print("  view %zu: corner order was mirrored, right reversed\n", view);
    }

    const cv::Mat probe = cv::imread(result.kept.front().left.string(), cv::IMREAD_GRAYSCALE);
    if (probe.empty())
        throw std::runtime_error("cannot read " + result.kept.front().left.string());
    result.imageSize = probe.size();
    print("  图像尺寸 %d × %d，有效视图 %zu\n",
          result.imageSize.width, result.imageSize.height, result.leftCorners.size());
    return result;
}

RunResult run(const Settings& s, const RunHooks& hooks)
{
    RunResult result;
    const DetectResult detected = detectPairs(s, hooks);
    result.pairs             = detected.kept;
    result.leftCorners       = detected.leftCorners;
    result.rightCorners      = detected.rightCorners;
    result.imageSize         = detected.imageSize;
    result.timings           = detected.timings;

    const std::vector<std::vector<cv::Point3f>> objectPoints(
        result.leftCorners.size(), boardObjectPoints(s));

    checkCancelled(hooks);
    print("\n=== 3. 单目标定 ===\n");
    reportStage(hooks, "calibrate", 0, 3);
    {
        StageTimer timer(result.timings, "calibrate left");
        result.camera1 = calibrateMono(objectPoints, result.leftCorners, result.imageSize, "left");
    }
    reportStage(hooks, "calibrate", 1, 3);
    {
        StageTimer timer(result.timings, "calibrate right");
        result.camera2 = calibrateMono(objectPoints, result.rightCorners, result.imageSize, "right");
    }
    reportStage(hooks, "calibrate", 2, 3);
    for (int i = 0; i < 2; ++i) {
        const CameraModel& c = (i == 0) ? result.camera1 : result.camera2;
        print("  %-5s rms %.4f px  fx %.3f  fy %.3f  cx %.3f  cy %.3f  k1 %+.5f  k2 %+.5f\n",
              i == 0 ? "left" : "right", c.rms,
              c.K.at<double>(0, 0), c.K.at<double>(1, 1),
              c.K.at<double>(0, 2), c.K.at<double>(1, 2),
              c.D.at<double>(0, 0), c.D.at<double>(0, 1));
    }

    checkCancelled(hooks);
    print("\n=== 4. 双目标定 ===\n");
    {
        StageTimer timer(result.timings, "stereo calibrate");
        result.stereo = calibrateStereoPair(objectPoints, result.leftCorners, result.rightCorners,
                                            result.imageSize, result.camera1, result.camera2, s);
    }
    reportStage(hooks, "calibrate", 3, 3);
    print("  rms %.4f px   baseline |T| = %.6f m\n", result.stereo.rms, result.stereo.baseline());
    printMatrix("R", result.stereo.R);
    printMatrix("T", result.stereo.T);
    print("  roi1 %dx%d+%d+%d   roi2 %dx%d+%d+%d\n",
          result.stereo.roi1.width, result.stereo.roi1.height,
          result.stereo.roi1.x, result.stereo.roi1.y,
          result.stereo.roi2.width, result.stereo.roi2.height,
          result.stereo.roi2.x, result.stereo.roi2.y);

    // Per-view error: a single bad frame is the usual reason for a mediocre
    // calibration, and the mean alone hides it.
    if (s.verbose && result.stereo.perViewErrors.rows > 0) {
        print("\n  每视图重投影误差（left / right，px）:\n");
        for (int view = 0; view < result.stereo.perViewErrors.rows; ++view) {
            print("    %-4s %7.4f %7.4f%s\n",
                  result.pairs[static_cast<std::size_t>(view)].key.c_str(),
                  result.stereo.perViewErrors.at<double>(view, 0),
                  result.stereo.perViewErrors.at<double>(view, 1),
                  std::max(result.stereo.perViewErrors.at<double>(view, 0),
                           result.stereo.perViewErrors.at<double>(view, 1)) > 1.0
                      ? "   <- 偏差偏大" : "");
        }
    }

    checkCancelled(hooks);
    print("\n=== 5. 验证 ===\n");
    {
        StageTimer timer(result.timings, "verify");
        result.verification = verify(result.leftCorners, result.rightCorners, result.camera1,
                                     result.camera2, result.stereo, s);
    }
    print("  极线误差（%d 角点）：均值 %.4f px，P95 %.4f px，最大 %.4f px\n",
          result.verification.comparedCorners, result.verification.epipolarMean,
          result.verification.epipolarP95, result.verification.epipolarMax);
    if (result.verification.triangulated) {
        print("  三角化：平面性 RMS %.4f mm（最差视图 %.4f mm），"
              "方格边长 %.4f mm（%+.3f %%），平均深度 %.4f m\n",
              result.verification.planarityRms * 1000.0,
              result.verification.planarityWorst * 1000.0,
              result.verification.meanEdgeLength * 1000.0,
              result.verification.edgeLengthError * 100.0,
              result.verification.meanDepth);
    }

    if (!s.referenceIntrinsics.empty()) {
        if (compareWithReference(s.referenceIntrinsics, result.camera1, result.referenceSummary))
            print("\n  与参考标定对比（left）:\n%s", result.referenceSummary.c_str());
        else
            print("\n  警告：无法从 %s 读取 camera_matrix\n", s.referenceIntrinsics.c_str());
    }

    result.epipolarOk = result.verification.epipolarMean < 0.5 &&
                        result.verification.epipolarP95 < 1.0;
    // The edge length is compared against --square-size, and both scale with it,
    // so this only proves that the baseline and the focal length agree with each
    // other. A wrong --square-size leaves it at zero.
    result.edgeOk = !result.verification.triangulated ||
                    std::abs(result.verification.edgeLengthError) < 0.01;
    return result;
}

void writeOutputs(RunResult& result, const Settings& s)
{
    const fs::path output = s.outputDirectory;
    fs::create_directories(output);
    saveIntrinsics(output / "intrinsics.yml", result.camera1, result.camera2);
    saveStereo(output / "stereo.yml", result.stereo, result.camera1, result.camera2);

    print("\n=== 6. 输出 ===\n");
    print("  %s\n  %s\n", (output / "intrinsics.yml").c_str(),
          (output / "stereo.yml").c_str());

    if (s.saveDebugImages) {
        {
            StageTimer timer(result.timings, "rectify preview");
            result.preview = rectifiedPreview(result.pairs.front(), result.camera1, result.camera2,
                                              result.stereo, result.imageSize,
                                              result.rectLeft, result.rectRight);
            cv::imwrite((output / "rectified_left.png").string(), result.rectLeft);
            cv::imwrite((output / "rectified_right.png").string(), result.rectRight);
            cv::imwrite((output / "epipolar_check.png").string(), result.preview);
        }
        print("  %s\n  %s\n  %s\n",
              (output / "rectified_left.png").c_str(),
              (output / "rectified_right.png").c_str(),
              (output / "epipolar_check.png").c_str());
    }

    {
        std::ofstream report(output / "calibration_report.md");
        if (!report)
            throw std::runtime_error("cannot write " +
                                     (output / "calibration_report.md").string());
        report << buildMarkdown(s, result.pairs, result.imageSize, result.camera1, result.camera2,
                                result.stereo, result.verification, result.referenceSummary);
    }
    print("  %s\n", (output / "calibration_report.md").c_str());
    result.outputDirectory = output.string();
}

void printConclusion(const RunResult& result)
{
    print("\n--- 阶段耗时 ---\n");
    double total = 0.0;
    for (const auto& stage : result.timings) {
        total += stage.second;
        print("  %-22s %8.1f ms\n", stage.first.c_str(), stage.second);
    }
    print("  %-22s %8.1f ms\n", "total", total);

    print("\n结论: 极线误差%s；三角化边长%s\n",
          result.epipolarOk ? "合格（均值 < 0.5 px）" : "偏大，检查角点质量或畸变模型",
          result.edgeOk ? "自洽（误差 < 1%，绝对尺度仍需外部确认）"
                        : "不自洽，检查校正/裁剪是否与 P 一致");
}

std::string matrixToString(const cv::Mat& m)
{
    std::ostringstream os;
    os << std::fixed << std::setprecision(6);
    for (int r = 0; r < m.rows; ++r) {
        os << '[';
        for (int c = 0; c < m.cols; ++c) os << ' ' << std::setw(12) << m.at<double>(r, c);
        os << " ]\n";
    }
    return os.str();
}

cv::Mat drawDetectedCorners(const cv::Mat& bgr, const std::vector<cv::Point2f>& corners,
                            const cv::Size& pattern, bool showIds)
{
    if (bgr.empty()) return bgr.clone();
    cv::Mat out = bgr.clone();
    const cv::Scalar green(0, 220, 0);
    const cv::Scalar yellow(0, 210, 255);
    for (std::size_t k = 0; k < corners.size(); ++k) {
        cv::circle(out, corners[k], 4, green, -1, cv::LINE_AA);
        if (showIds)
            cv::putText(out, std::to_string(k), corners[k] + cv::Point2f(5.f, -5.f),
                        cv::FONT_HERSHEY_SIMPLEX, 0.32, yellow, 1, cv::LINE_AA);
    }
    if (pattern.width > 1 && pattern.height > 1 &&
        static_cast<int>(corners.size()) == pattern.width * pattern.height) {
        for (int y = 0; y < pattern.height; ++y) {
            for (int x = 0; x + 1 < pattern.width; ++x)
                cv::line(out, corners[y * pattern.width + x],
                         corners[y * pattern.width + x + 1], green, 1, cv::LINE_AA);
        }
        for (int x = 0; x < pattern.width; ++x) {
            for (int y = 0; y + 1 < pattern.height; ++y)
                cv::line(out, corners[y * pattern.width + x],
                         corners[(y + 1) * pattern.width + x], green, 1, cv::LINE_AA);
        }
    }
    return out;
}

}  // namespace calib
