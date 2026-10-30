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
#include <opencv2/highgui.hpp>

namespace calib {

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Configuration. The defaults match the OpenCV sample dataset: 640x480,
// 9x6 inner corners, 25 mm squares.
// ---------------------------------------------------------------------------
struct Settings {
    std::string imageDirectory = ".";
    std::string leftPrefix     = "left";
    std::string rightPrefix    = "right";
    std::string extension      = ".jpg";

    int    boardWidth   = 9;       // inner corners along a row
    int    boardHeight  = 6;       // inner corners along a column
    double squareSize   = 0.025;   // metres; sets the scale of T

    // cornerSubPix
    int    subpixWindow     = 11;
    int    subpixIterations = 30;
    double subpixEpsilon    = 0.001;
    bool   useSbDetector    = false;

    // stereoCalibrate
    bool refineIntrinsics = false;  // false => CALIB_FIX_INTRINSIC
    bool sameFocalLength  = false;  // CALIB_SAME_FOCAL_LENGTH

    // stereoRectify
    double alpha = 0.0;             // 0 = crop to valid pixels, 1 = keep everything

    std::string outputDirectory     = "output";
    std::string referenceIntrinsics;    // optional left_intrinsics.yml to compare against
    bool saveDebugImages = true;
    bool showWindow      = true;
    bool verbose         = true;
};

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

struct ImagePair {
    std::string key;      // shared numeric suffix, e.g. "01"
    fs::path left;
    fs::path right;
};

struct CameraModel {
    cv::Mat K;                    // 3x3 CV_64F
    cv::Mat D;                    // 1xN CV_64F
    double  rms = 0.0;
    std::vector<double> perViewErrors;
};

struct StereoModel {
    cv::Mat R, T, E, F;
    cv::Mat R1, R2, P1, P2, Q;
    cv::Rect roi1, roi2;
    double  rms = 0.0;
    // stereoCalibrate reports N x 2: column 0 belongs to the first camera,
    // column 1 to the second. A flat vector would be rejected by the
    // OutputArray, which insists on that shape.
    cv::Mat perViewErrors;
    double  baseline() const { return cv::norm(T); }
};

struct Verification {
    // Epipolar alignment after rectification.
    double epipolarMean = 0.0;
    double epipolarP95  = 0.0;
    double epipolarMax  = 0.0;
    int    comparedCorners = 0;

    // Triangulated board geometry.
    bool   triangulated    = false;
    double planarityRms    = 0.0;   // metres, mean over views of the RMS distance to the fitted plane
    double planarityWorst  = 0.0;   // metres, worst view
    double meanEdgeLength  = 0.0;   // metres, measured grid spacing
    double edgeLengthError = 0.0;   // fraction, relative to squareSize
    double meanDepth       = 0.0;   // metres, board distance from the left camera
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
        std::printf("  warning: unpaired images — left only:");
        for (const auto& k : onlyLeft) std::printf(" %s", k.c_str());
        std::printf("  right only:");
        for (const auto& k : onlyRight) std::printf(" %s", k.c_str());
        std::printf("\n");
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
                                                bool verbose = true)
{
    std::vector<std::vector<cv::Point2f>> leftCorners;
    const int expected = s.boardWidth * s.boardHeight;
    for (const auto& pair : pairs) {
        const cv::Mat leftImage  = cv::imread(pair.left.string(), cv::IMREAD_GRAYSCALE);
        const cv::Mat rightImage = cv::imread(pair.right.string(), cv::IMREAD_GRAYSCALE);
        if (leftImage.empty() || rightImage.empty()) {
            std::printf("  %-4s skipped: could not read an image\n", pair.key.c_str());
            continue;
        }
        bool sbLeft = false, sbRight = false;
        std::vector<cv::Point2f> lc = detectCorners(leftImage, s, sbLeft);
        std::vector<cv::Point2f> rc = detectCorners(rightImage, s, sbRight);
        if (static_cast<int>(lc.size()) != expected || static_cast<int>(rc.size()) != expected) {
            std::printf("  %-4s skipped: %zu/%zu corners, expected %d\n",
                        pair.key.c_str(), lc.size(), rc.size(), expected);
            continue;
        }
        if (verbose)
            std::printf("  %-4s %d corners per eye%s\n", pair.key.c_str(), expected,
                        (sbLeft || sbRight) ? "  (SB detector)" : "");
        leftCorners.push_back(std::move(lc));
        rightCorners.push_back(std::move(rc));
        kept.push_back(pair);
    }
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
        std::printf("  三角化：%zu/%zu 个角点得到正深度（其余被剔除）\n",
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
    std::printf("  %-4s", label);
    for (int r = 0; r < m.rows; ++r) {
        if (r) std::printf("      ");
        std::printf("[");
        for (int c = 0; c < m.cols; ++c) std::printf(" %12.6f", m.at<double>(r, c));
        std::printf(" ]\n");
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
// Command line
// ---------------------------------------------------------------------------
void printUsage(const char* program)
{
    std::printf(
        "usage: %s [options]\n\n"
        "  --images <dir>        image directory                        (default: .)\n"
        "  --left-prefix <str>   left file prefix                       (default: left)\n"
        "  --right-prefix <str>  right file prefix                      (default: right)\n"
        "  --ext <.jpg>          file extension                         (default: .jpg)\n"
        "  --pattern <WxH>       inner corners, e.g. 9x6                (default: 9x6)\n"
        "  --square-size <m>     side of one square in metres           (default: 0.025)\n"
        "  --alpha <0..1>        stereoRectify balance, 0 crops, 1 keeps (default: 0)\n"
        "  --output <dir>        where the results are written          (default: output)\n"
        "  --reference <yml>     compare the left intrinsics with a known calibration\n"
        "  --refine-intrinsics   optimise intrinsics jointly (default: keep them fixed)\n"
        "  --same-focal-length   constrain fx == fy of both cameras\n"
        "  --sb                  use findChessboardCornersSB instead of the classic detector\n"
        "  --no-debug-images     do not write the rectified previews\n"
        "  --no-window           never open a window\n"
        "  --quiet               skip the per-image and per-view detail\n\n",
        program);
}

Settings parseArgs(int argc, char** argv)
{
    Settings s;
    const auto next = [&](int& i, const char* name) -> std::string {
        if (i + 1 >= argc) throw std::invalid_argument(std::string(name) + " needs a value");
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--images")                 s.imageDirectory = next(i, "--images");
        else if (arg == "--left-prefix")       s.leftPrefix = next(i, "--left-prefix");
        else if (arg == "--right-prefix")      s.rightPrefix = next(i, "--right-prefix");
        else if (arg == "--ext")               s.extension = next(i, "--ext");
        else if (arg == "--square-size")       s.squareSize = std::stod(next(i, "--square-size"));
        else if (arg == "--alpha")             s.alpha = std::stod(next(i, "--alpha"));
        else if (arg == "--output")            s.outputDirectory = next(i, "--output");
        else if (arg == "--reference")         s.referenceIntrinsics = next(i, "--reference");
        else if (arg == "--refine-intrinsics") s.refineIntrinsics = true;
        else if (arg == "--same-focal-length") s.sameFocalLength = true;
        else if (arg == "--sb")                s.useSbDetector = true;
        else if (arg == "--no-debug-images")   s.saveDebugImages = false;
        else if (arg == "--no-window")         s.showWindow = false;
        else if (arg == "--quiet")             s.verbose = false;
        else if (arg == "--pattern") {
            const std::string value = next(i, "--pattern");
            const std::size_t sep = value.find('x');
            if (sep == std::string::npos)
                throw std::invalid_argument("--pattern expects WxH, e.g. 9x6");
            s.boardWidth  = std::stoi(value.substr(0, sep));
            s.boardHeight = std::stoi(value.substr(sep + 1));
        } else if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown option: " + arg);
        }
    }
    if (s.boardWidth < 3 || s.boardHeight < 3)
        throw std::invalid_argument("--pattern needs at least 3x3 inner corners");
    if (!(s.squareSize > 0))
        throw std::invalid_argument("--square-size must be positive");
    if (s.alpha < 0 || s.alpha > 1)
        throw std::invalid_argument("--alpha must be within [0, 1]");
    return s;
}

}  // namespace calib

int main(int argc, char** argv)
{
    using namespace calib;

    try {
        const Settings settings = parseArgs(argc, argv);
        std::vector<std::pair<std::string, double>> timings;

        std::printf("\n=== 1. 图像对 ===\n");
        const std::vector<ImagePair> pairs = discoverPairs(settings);
        std::printf("  找到 %zu 对图像\n", pairs.size());
        if (pairs.size() < 3)
            throw std::runtime_error("need at least 3 stereo pairs to calibrate");

        std::printf("\n=== 2. 角点检测 ===\n");
        std::vector<std::vector<cv::Point2f>> rightCorners;
        std::vector<ImagePair> kept;
        std::vector<std::vector<cv::Point2f>> leftCorners;
        {
            StageTimer timer(timings, "detect corners");
            leftCorners = detectAll(pairs, settings, rightCorners, kept, settings.verbose);
        }
        if (leftCorners.size() < 3)
            throw std::runtime_error("fewer than 3 pairs yielded a complete board");

        // Restore the correspondence wherever only one camera mirrored the order.
        for (std::size_t view = 0; view < leftCorners.size(); ++view) {
            if (reverseIfMirrored(leftCorners[view], rightCorners[view], settings.boardWidth))
                std::printf("  view %zu: corner order was mirrored, right reversed\n", view);
        }

        cv::Size imageSize;
        {
            const cv::Mat probe = cv::imread(kept.front().left.string(), cv::IMREAD_GRAYSCALE);
            if (probe.empty()) throw std::runtime_error("cannot read " + kept.front().left.string());
            imageSize = probe.size();
        }
        std::printf("  图像尺寸 %d × %d，有效视图 %zu\n",
                    imageSize.width, imageSize.height, leftCorners.size());

        const std::vector<std::vector<cv::Point3f>> objectPoints(
            leftCorners.size(), boardObjectPoints(settings));

        std::printf("\n=== 3. 单目标定 ===\n");
        CameraModel camera1, camera2;
        {
            StageTimer timer(timings, "calibrate left");
            camera1 = calibrateMono(objectPoints, leftCorners, imageSize, "left");
        }
        {
            StageTimer timer(timings, "calibrate right");
            camera2 = calibrateMono(objectPoints, rightCorners, imageSize, "right");
        }
        for (int i = 0; i < 2; ++i) {
            const CameraModel& c = (i == 0) ? camera1 : camera2;
            std::printf("  %-5s rms %.4f px  fx %.3f  fy %.3f  cx %.3f  cy %.3f  k1 %+.5f  k2 %+.5f\n",
                        i == 0 ? "left" : "right", c.rms,
                        c.K.at<double>(0, 0), c.K.at<double>(1, 1),
                        c.K.at<double>(0, 2), c.K.at<double>(1, 2),
                        c.D.at<double>(0, 0), c.D.at<double>(0, 1));
        }

        std::printf("\n=== 4. 双目标定 ===\n");
        StereoModel stereo;
        {
            StageTimer timer(timings, "stereo calibrate");
            stereo = calibrateStereoPair(objectPoints, leftCorners, rightCorners, imageSize,
                                         camera1, camera2, settings);
        }
        std::printf("  rms %.4f px   baseline |T| = %.6f m\n", stereo.rms, stereo.baseline());
        printMatrix("R", stereo.R);
        printMatrix("T", stereo.T);
        std::printf("  roi1 %dx%d+%d+%d   roi2 %dx%d+%d+%d\n",
                    stereo.roi1.width, stereo.roi1.height, stereo.roi1.x, stereo.roi1.y,
                    stereo.roi2.width, stereo.roi2.height, stereo.roi2.x, stereo.roi2.y);

        // Per-view error: a single bad frame is the usual reason for a
        // mediocre calibration, and the mean alone hides it.
        if (settings.verbose && stereo.perViewErrors.rows > 0) {
            std::printf("\n  每视图重投影误差（left / right，px）:\n");
            for (int view = 0; view < stereo.perViewErrors.rows; ++view) {
                std::printf("    %-4s %7.4f %7.4f%s\n",
                            kept[static_cast<std::size_t>(view)].key.c_str(),
                            stereo.perViewErrors.at<double>(view, 0),
                            stereo.perViewErrors.at<double>(view, 1),
                            std::max(stereo.perViewErrors.at<double>(view, 0),
                                     stereo.perViewErrors.at<double>(view, 1)) > 1.0
                                ? "   <- 偏差偏大" : "");
            }
        }

        std::printf("\n=== 5. 验证 ===\n");
        Verification verification;
        {
            StageTimer timer(timings, "verify");
            verification = verify(leftCorners, rightCorners, camera1, camera2, stereo, settings);
        }
        std::printf("  极线误差（%d 角点）：均值 %.4f px，P95 %.4f px，最大 %.4f px\n",
                    verification.comparedCorners, verification.epipolarMean,
                    verification.epipolarP95, verification.epipolarMax);
        if (verification.triangulated) {
            std::printf("  三角化：平面性 RMS %.4f mm（最差视图 %.4f mm），"
                        "方格边长 %.4f mm（%+.3f %%），平均深度 %.4f m\n",
                        verification.planarityRms * 1000.0,
                        verification.planarityWorst * 1000.0,
                        verification.meanEdgeLength * 1000.0,
                        verification.edgeLengthError * 100.0,
                        verification.meanDepth);
        }

        std::string referenceSummary;
        if (!settings.referenceIntrinsics.empty()) {
            if (compareWithReference(settings.referenceIntrinsics, camera1, referenceSummary))
                std::printf("\n  与参考标定对比（left）:\n%s", referenceSummary.c_str());
            else
                std::printf("\n  警告：无法从 %s 读取 camera_matrix\n",
                            settings.referenceIntrinsics.c_str());
        }

        const fs::path output = settings.outputDirectory;
        fs::create_directories(output);
        saveIntrinsics(output / "intrinsics.yml", camera1, camera2);
        saveStereo(output / "stereo.yml", stereo, camera1, camera2);

        std::printf("\n=== 6. 输出 ===\n");
        std::printf("  %s\n  %s\n", (output / "intrinsics.yml").c_str(),
                    (output / "stereo.yml").c_str());

        if (settings.saveDebugImages) {
            cv::Mat rectLeft, rectRight;
            cv::Mat preview;
            {
                StageTimer timer(timings, "rectify preview");
                preview = rectifiedPreview(kept.front(), camera1, camera2, stereo,
                                           imageSize, rectLeft, rectRight);
                cv::imwrite((output / "rectified_left.png").string(), rectLeft);
                cv::imwrite((output / "rectified_right.png").string(), rectRight);
                cv::imwrite((output / "epipolar_check.png").string(), preview);
            }
            std::printf("  %s\n  %s\n  %s\n",
                        (output / "rectified_left.png").c_str(),
                        (output / "rectified_right.png").c_str(),
                        (output / "epipolar_check.png").c_str());

            if (settings.showWindow && (std::getenv("DISPLAY") || std::getenv("WAYLAND_DISPLAY"))) {
                try {
                    cv::imshow("epipolar check (any key to close)", preview);
                    cv::waitKey(0);
                    cv::destroyAllWindows();
                } catch (const cv::Exception& error) {
                    std::cerr << "display failed (" << error.what() << "); see "
                              << (output / "epipolar_check.png") << '\n';
                }
            }
        }

        {
            std::ofstream report(output / "calibration_report.md");
            if (!report) throw std::runtime_error("cannot write " + (output / "calibration_report.md").string());
            report << buildMarkdown(settings, kept, imageSize, camera1, camera2, stereo,
                                    verification, referenceSummary);
        }
        std::printf("  %s\n", (output / "calibration_report.md").c_str());

        std::printf("\n--- 阶段耗时 ---\n");
        double total = 0.0;
        for (const auto& stage : timings) {
            total += stage.second;
            std::printf("  %-22s %8.1f ms\n", stage.first.c_str(), stage.second);
        }
        std::printf("  %-22s %8.1f ms\n", "total", total);

        const bool epipolarOk = verification.epipolarMean < 0.5 && verification.epipolarP95 < 1.0;
        // The edge length is compared against --square-size, and both scale with
        // it, so this only proves that the baseline and the focal length agree
        // with each other. A wrong --square-size leaves it at zero.
        const bool edgeOk = !verification.triangulated ||
                            std::abs(verification.edgeLengthError) < 0.01;
        std::printf("\n结论: 极线误差%s；三角化边长%s\n",
                    epipolarOk ? "合格（均值 < 0.5 px）" : "偏大，检查角点质量或畸变模型",
                    edgeOk ? "自洽（误差 < 1%，绝对尺度仍需外部确认）"
                           : "不自洽，检查校正/裁剪是否与 P 一致");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "\nError: " << error.what() << '\n';
        return 1;
    }
}
