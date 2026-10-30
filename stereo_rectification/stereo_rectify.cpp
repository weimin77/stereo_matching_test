// ===========================================================================
//  Epipolar rectification for OpenCV 5 — turns a calibration into images that a
//  stereo matcher can consume without further geometric work.
//
//  Pipeline
//    1. read a calibration (stereo.yml from stereo_calibration, or any file that
//       carries M1/D1/M2/D2 plus R/T or a ready-made R1/P1 set)
//    2. stereoRectify                               -> R1,R2,P1,P2,Q,roi1,roi2
//    3. work out what the rectified images actually look like: the new focal
//       length, the principal points, the disparity offset doffs, the baseline,
//       and the common valid region of the two views
//    4. build the maps once with initUndistortRectifyMap, cropped to that region
//    5. remap every pair, and remap a white image to get the validity mask
//    6. write camera.json / rectification.yml / a report, plus the rectified pairs
//
//  Why this is not just "stereoRectify then remap"
//    a. alpha < 1 shrinks the focal length to keep every pixel valid. Feeding the
//       original fx to a depth formula then biases every distance by that factor
//       (2.9 % on the OpenCV sample data), so the new fx must be carried along.
//    b. Remapping leaves the outside of the distorted image at 0. A matcher
//       happily matches that black border against image content and produces a
//       wide band of garbage. The mask produced here marks the pixels that came
//       from real data.
//    c. Q is expressed in the coordinates of the uncropped rectified image. After
//       cropping, its principal-point terms must shift by the crop origin or the
//       reprojected point cloud is translated sideways.
//    d. minDisparity/numDisparities have to be derived from f'*B and the depth
//       range of interest, not from the raw fx.
//
//  OpenCV 5 note: stereoRectify lives in stereo.hpp, initUndistortRectifyMap and
//  remap in imgproc.hpp. <opencv2/opencv.hpp> does not pull the split modules in.
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
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/stereo.hpp>

namespace rectify {

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
struct Settings {
    std::string calibrationPath;      // stereo.yml / intrinsics.yml with R,T
    std::string imageDirectory = "."; // batch mode
    std::string leftPath;             // single-pair mode
    std::string rightPath;

    std::string leftPrefix  = "left";
    std::string rightPrefix = "right";
    std::string extension   = ".jpg";

    std::string outputDirectory = "output";

    // stereoRectify
    double alpha = 0.0;               // 0 = crop to valid pixels, 1 = keep everything
    cv::Size newImageSize;            // empty => same as the input
    bool keepStoredRectification = false;

    // Cropping
    bool cropToCommonRoi = true;
    int  padColumns       = 0;        // extra columns kept around the common ROI

    // Disparity window, only used to derive minDisparity/numDisparities.
    double depthNear = 0.0;           // metres
    double depthFar  = 0.0;           // metres
    int    explicitMinDisparity   = -1;
    int    explicitNumDisparities = 0;

    // Remap
    int  mapType        = CV_16SC2;   // CV_16SC2 (fixed point, ~2x faster) or CV_32FC1
    bool useCubic       = false;
    std::string outputExtension = ".png";
    // Optional closed-loop check: detect a chessboard in the *rectified* pair and
    // measure the row mismatch. This is the only check that exercises the actual
    // output pixels rather than the matrices that produced them.
    cv::Size verifyPattern;
    bool   verifyWithSb = false;

    bool writeMask      = true;
    bool writePreview   = true;
    bool verbose        = true;
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

struct ImagePair {
    std::string key;
    fs::path left;
    fs::path right;
};

// A calibration is "usable" in two ways. The preferred one recreates the
// rectification from M/D/R/T so alpha and the output size stay adjustable; the
// fallback takes the R1/R2/P1/P2 that were stored at calibration time and
// therefore cannot be re-tuned.
struct Calibration {
    cv::Mat M1, D1, M2, D2;   // 3x3, 1xN (or empty for a distortion-free camera)
    cv::Mat R, T;             // 3x3, 3x1
    cv::Mat R1, R2, P1, P2, Q;
    bool hasExtrinsics = false;
    bool hasRectification = false;
    double baseline = 0.0;
};

// Everything a matcher needs to know about the rectified pair, plus the numbers
// the report prints.
struct Geometry {
    cv::Size sourceSize;        // calibration image size
    cv::Size fullSize;          // rectified size before cropping
    cv::Rect crop;              // valid region inside fullSize
    cv::Size outputSize;        // crop size

    cv::Mat R1, R2;
    cv::Mat P1, P2, Q;          // already adjusted for the crop

    double fx = 0.0, fy = 0.0;
    double cx1 = 0.0, cy1 = 0.0;
    double cx2 = 0.0;
    double doffs = 0.0;         // cx1 - cx2, the disparity at infinity
    double baseline = 0.0;      // metres, |T|
    double focalBaseline = 0.0; // fx * baseline, the disparity-constant (positive)
    bool   disparitySignFlipped = false;  // Tx > 0: the right view is the reference

    // Suggested search window, only filled when a depth range was given.
    bool   disparityRangeKnown = false;
    int    minDisparity   = 0;
    int    numDisparities = 0;
    double disparityNear  = 0.0;   // largest disparity, at depthNear
    double disparityFar   = 0.0;   // smallest disparity, at depthFar
    double depthNear      = 0.0;
    double depthFar       = 0.0;
    bool   rangeClipped   = false; // requested depth range exceeded the image width
};

struct Maps {
    cv::Mat map1x, map1y, map2x, map2y;
};

// ---------------------------------------------------------------------------
// Small I/O helpers
// ---------------------------------------------------------------------------
inline std::string joinPath(const fs::path& directory, const std::string& name)
{
    return (directory / name).string();
}

// cv::FileNode returns an empty matrix for a missing key, so this distinguishes
// "absent" from "present". OpenCV 5 renamed the old `empty()` on FileNode.
inline bool present(const cv::FileNode& node)
{
    return !node.empty() && node.isNone() == false;
}

inline cv::Mat readMatrix(const cv::FileNode& node)
{
    cv::Mat value;
    if (present(node)) node >> value;
    return value;
}

inline std::string numericSuffix(const std::string& stem, const std::string& prefix)
{
    if (stem.rfind(prefix, 0) != 0) return {};
    std::size_t pos = prefix.size();
    while (pos < stem.size() && std::isdigit(static_cast<unsigned char>(stem[pos]))) ++pos;
    if (pos == prefix.size()) return {};
    return stem.substr(prefix.size(), pos - prefix.size());
}

// ---------------------------------------------------------------------------
// 1. Argument-free image discovery: leftNN<ext> paired with rightNN<ext>.
// ---------------------------------------------------------------------------
std::vector<ImagePair> discoverPairs(const Settings& s)
{
    if (!s.leftPath.empty() || !s.rightPath.empty()) {
        if (s.leftPath.empty() || s.rightPath.empty())
            throw std::invalid_argument("--left and --right must be given together");
        if (!fs::exists(s.leftPath)) throw std::runtime_error("no such file: " + s.leftPath);
        if (!fs::exists(s.rightPath)) throw std::runtime_error("no such file: " + s.rightPath);
        return {{"pair", s.leftPath, s.rightPath}};
    }

    if (!fs::is_directory(s.imageDirectory))
        throw std::runtime_error("not a directory: " + s.imageDirectory);

    std::map<std::string, std::string> lefts;
    for (const auto& entry : fs::directory_iterator(s.imageDirectory)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() != s.extension) continue;
        const std::string suffix = numericSuffix(entry.path().stem().string(), s.leftPrefix);
        if (!suffix.empty()) lefts[suffix] = entry.path().string();
    }

    std::vector<ImagePair> pairs;
    for (const auto& entry : fs::directory_iterator(s.imageDirectory)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() != s.extension) continue;
        const std::string suffix = numericSuffix(entry.path().stem().string(), s.rightPrefix);
        if (suffix.empty()) continue;
        const auto match = lefts.find(suffix);
        if (match == lefts.end()) continue;
        pairs.push_back({suffix, match->second, entry.path().string()});
    }

    std::sort(pairs.begin(), pairs.end(), [](const ImagePair& a, const ImagePair& b) {
        if (a.key.size() != b.key.size()) return a.key.size() < b.key.size();
        return a.key < b.key;
    });
    return pairs;
}

// ---------------------------------------------------------------------------
// 2. Calibration input
// ---------------------------------------------------------------------------
Calibration loadCalibration(const Settings& s)
{
    cv::FileStorage storage(s.calibrationPath, cv::FileStorage::READ);
    if (!storage.isOpened())
        throw std::runtime_error("cannot open calibration: " + s.calibrationPath);

    const cv::FileNode root = storage.root();
    Calibration c;
    c.M1 = readMatrix(root["M1"]);
    c.D1 = readMatrix(root["D1"]);
    c.M2 = readMatrix(root["M2"]);
    c.D2 = readMatrix(root["D2"]);
    if (c.M1.empty() || c.M2.empty())
        throw std::runtime_error("calibration needs M1 and M2 (camera matrices): " +
                                 s.calibrationPath);
    if (c.M1.rows != 3 || c.M1.cols != 3 || c.M2.rows != 3 || c.M2.cols != 3)
        throw std::runtime_error("M1 and M2 must be 3x3");

    c.R = readMatrix(root["R"]);
    c.T = readMatrix(root["T"]);
    c.hasExtrinsics = !c.R.empty() && !c.T.empty();
    if (c.hasExtrinsics && (c.R.rows != 3 || c.R.cols != 3 || c.T.total() != 3))
        throw std::runtime_error("R must be 3x3 and T must have 3 elements");

    c.R1 = readMatrix(root["R1"]);
    c.R2 = readMatrix(root["R2"]);
    c.P1 = readMatrix(root["P1"]);
    c.P2 = readMatrix(root["P2"]);
    c.Q  = readMatrix(root["Q"]);
    c.hasRectification = !c.R1.empty() && !c.P1.empty() && !c.R2.empty() && !c.P2.empty();

    if (present(root["baseline_m"])) root["baseline_m"] >> c.baseline;
    if (c.baseline <= 0 && !c.T.empty()) c.baseline = cv::norm(c.T);

    if (!c.hasExtrinsics && !c.hasRectification)
        throw std::runtime_error(
            "calibration carries neither R/T nor R1/P1; run stereo_calibration first");

    // OpenCV 5 returns the rotation/translation as CV_64F but a hand-edited file
    // may hold CV_32F, and stereoRectify insists on the former.
    const auto as64 = [](cv::Mat m) { if (!m.empty() && m.depth() != CV_64F) m.convertTo(m, CV_64F); return m; };
    c.M1 = as64(c.M1); c.M2 = as64(c.M2);
    c.D1 = as64(c.D1); c.D2 = as64(c.D2);
    c.R  = as64(c.R);  c.T  = as64(c.T);
    c.R1 = as64(c.R1); c.R2 = as64(c.R2);
    c.P1 = as64(c.P1); c.P2 = as64(c.P2);
    c.Q  = as64(c.Q);
    return c;
}

// ---------------------------------------------------------------------------
// 3. Rectification geometry
// ---------------------------------------------------------------------------
// Intersection of the two "fully valid" rectangles. The rectified views share one
// coordinate frame (stereoRectify gives both the same fy and the same cy), so
// intersecting the two rectangles is meaningful and keeps the pair the same size.
inline cv::Rect intersectRoi(const cv::Rect& a, const cv::Rect& b)
{
    if (a.empty()) return b;
    if (b.empty()) return a;
    const int x0 = std::max(a.x, b.x);
    const int y0 = std::max(a.y, b.y);
    const int x1 = std::min(a.x + a.width,  b.x + b.width);
    const int y1 = std::min(a.y + a.height, b.y + b.height);
    if (x1 <= x0 || y1 <= y0) return {};
    return {x0, y0, x1 - x0, y1 - y0};
}

Geometry computeGeometry(const Calibration& c, const cv::Size& imageSize, const Settings& s)
{
    Geometry g;
    g.sourceSize = imageSize;
    g.fullSize   = s.newImageSize.area() > 0 ? s.newImageSize : imageSize;
    g.baseline   = c.baseline > 0 ? c.baseline : cv::norm(c.T);

    cv::Rect roi1, roi2;
    if (s.keepStoredRectification && c.hasRectification) {
        // The stored matrices were produced for the calibration image size; a
        // different output size would silently stretch the result.
        if (g.fullSize != imageSize)
            throw std::invalid_argument(
                "--keep-rectification reuses the stored P1/P2, which are tied to the "
                "calibration image size; drop --new-size or drop --keep-rectification");
        g.R1 = c.R1.clone();
        g.R2 = c.R2.clone();
        g.P1 = c.P1.clone();
        g.P2 = c.P2.clone();
        g.Q  = c.Q.empty() ? cv::Mat() : c.Q.clone();
    } else {
        if (!c.hasExtrinsics)
            throw std::runtime_error(
                "calibration has no R/T, so the rectification cannot be recomputed; "
                "pass --keep-rectification to use the stored R1/R2/P1/P2");
        cv::stereoRectify(c.M1, c.D1, c.M2, c.D2, imageSize, c.R, c.T,
                          g.R1, g.R2, g.P1, g.P2, g.Q,
                          cv::STEREO_ZERO_DISPARITY, s.alpha, g.fullSize, &roi1, &roi2);
    }

    // The new focal length and principal points. P2 differs from P1 only in the
    // baseline column, so the horizontal principal points usually coincide.
    g.fx  = g.P1.at<double>(0, 0);
    g.fy  = g.P1.at<double>(1, 1);
    g.cx1 = g.P1.at<double>(0, 2);
    g.cy1 = g.P1.at<double>(1, 2);
    g.cx2 = g.P2.at<double>(0, 2);
    g.doffs = g.cx1 - g.cx2;

    // The disparity of a point at depth Z is doffs - P2(0,3)/Z, and P2(0,3) equals
    // fx * Tx. A positive Tx means the second camera sits on the +x side of the
    // first, so the disparity comes out negative. StereoBM/SGBM can only search a
    // non-negative range in algorithm_test's configuration, so flag it.
    const double tx = g.P2.at<double>(0, 3) / g.fx;
    g.disparitySignFlipped = tx > 0.0;
    g.focalBaseline = g.fx * g.baseline;

    if (!(g.fx > 0) || !(g.baseline > 0))
        throw std::runtime_error("rectification produced a non-positive focal length or baseline");

    // Common valid region.
    if (s.cropToCommonRoi && !roi1.empty() && !roi2.empty()) {
        cv::Rect common = intersectRoi(roi1, roi2);
        if (common.empty())
            throw std::runtime_error("the two views have no common valid region at alpha=" +
                                     std::to_string(s.alpha) + "; raise --alpha");
        if (s.padColumns > 0) {
            // Padding is only honoured where the padded rectangle still fits.
            const int x0 = std::max(0, common.x - s.padColumns);
            const int x1 = std::min(g.fullSize.width, common.x + common.width + s.padColumns);
            common.x = x0;
            common.width = x1 - x0;
        }
        g.crop = common;
    } else {
        g.crop = cv::Rect(0, 0, g.fullSize.width, g.fullSize.height);
    }
    g.outputSize = g.crop.size();

    // Rebase P and Q onto the cropped coordinate system. A crop only moves the
    // origin, so the focal length and the baseline column stay untouched while
    // the principal points lose the crop offset. Q carries the principal point as
    // its fourth column with the opposite sign, hence the addition.
    g.P1.at<double>(0, 2) -= g.crop.x;
    g.P1.at<double>(1, 2) -= g.crop.y;
    g.P2.at<double>(0, 2) -= g.crop.x;
    g.P2.at<double>(1, 2) -= g.crop.y;
    g.cx1 = g.P1.at<double>(0, 2);
    g.cy1 = g.P1.at<double>(1, 2);
    g.cx2 = g.P2.at<double>(0, 2);
    if (!g.Q.empty() && g.Q.rows == 4 && g.Q.cols == 4) {
        g.Q.at<double>(0, 3) += g.crop.x;
        g.Q.at<double>(1, 3) += g.crop.y;
    } else {
        // Rebuild the standard Q from the rectified geometry.
        g.Q = cv::Mat::zeros(4, 4, CV_64F);
        g.Q.at<double>(0, 0) = 1.0;
        g.Q.at<double>(1, 1) = 1.0;
        g.Q.at<double>(0, 3) = -g.cx1;
        g.Q.at<double>(1, 3) = -g.cy1;
        g.Q.at<double>(2, 3) = g.fx;
        const double tx = g.P2.at<double>(0, 3) / g.fx;   // negative for a left-to-right rig
        if (tx != 0.0) {
            g.Q.at<double>(3, 2) = -1.0 / tx;
            g.Q.at<double>(3, 3) = g.doffs / tx;
        }
    }

    // Suggested search window.
    if (s.explicitNumDisparities > 0) {
        g.disparityRangeKnown = true;
        g.minDisparity   = std::max(0, s.explicitMinDisparity);
        g.numDisparities = ((s.explicitNumDisparities + 15) / 16) * 16;
        // Report the covered disparity range and the depths it maps to, so the
        // report is meaningful even when the window came from the command line.
        g.disparityFar  = g.minDisparity;
        g.disparityNear = g.minDisparity + g.numDisparities - 1;
        if (g.disparityNear > g.doffs) g.depthNear = g.focalBaseline / (g.disparityNear - g.doffs);
        if (g.disparityFar  > g.doffs) g.depthFar  = g.focalBaseline / (g.disparityFar  - g.doffs);
    } else if (s.depthNear > 0 && s.depthFar > s.depthNear) {
        g.disparityRangeKnown = true;
        g.depthNear = s.depthNear;
        g.depthFar  = s.depthFar;
        g.disparityNear = g.focalBaseline / s.depthNear + g.doffs;
        g.disparityFar  = g.focalBaseline / s.depthFar  + g.doffs;

        // MinDisparity must be a multiple of 16 for StereoBM/SGBM, and it is
        // rounded down so the requested far plane stays inside the window.
        const int mn = std::max(0, static_cast<int>(std::floor(g.disparityFar / 16.0)) * 16);
        const int mx = static_cast<int>(std::ceil((g.disparityNear + 1.0) / 16.0)) * 16;
        g.minDisparity   = mn;
        g.numDisparities = std::max(16, mx - mn);
        g.disparityFar   = std::max(g.disparityFar, 0.0);
        if (g.minDisparity + g.numDisparities >= g.outputSize.width) {
            g.rangeClipped = true;
            g.numDisparities = ((g.outputSize.width - 1) / 16) * 16;
            if (g.numDisparities < 16)
                throw std::runtime_error("the rectified image is too narrow for any disparity window");
        }
    }
    return g;
}

// ---------------------------------------------------------------------------
// 3b. Q / P consistency
// ---------------------------------------------------------------------------
// Q and P are written by the same routine but stored separately, and Q is the one
// that silently breaks when the crop is forgotten. Projecting a 3D point through P
// and taking it back through Q catches any drift between them, including a Q that
// was rebuilt for the uncropped frame.
struct SelfCheck {
    bool   ok = false;
    double error = 0.0;      // metres, |reconstructed - original|
    double depth = 0.0;      // metres, the reconstructed Z
    double disparity = 0.0;  // pixels
};

SelfCheck checkQAgainstP(const Geometry& g)
{
    SelfCheck result;
    if (g.Q.rows != 4 || g.Q.cols != 4) return result;

    // A point 1 m in front of the left camera, off-axis enough that a principal
    // point error cannot cancel out.
    const double point[4] = {0.05, -0.03, 1.0, 1.0};
    const auto project = [&point](const cv::Mat& P, double& x, double& y) {
        double w = 0.0;
        x = 0.0; y = 0.0;
        for (int c = 0; c < 4; ++c) {
            x += P.at<double>(0, c) * point[c];
            y += P.at<double>(1, c) * point[c];
            w += P.at<double>(2, c) * point[c];
        }
        if (w == 0.0) return false;
        x /= w; y /= w;
        return true;
    };

    double x1 = 0.0, y1 = 0.0, x2 = 0.0, y2 = 0.0;
    if (!project(g.P1, x1, y1) || !project(g.P2, x2, y2)) return result;

    const double disparity = x1 - x2;
    const double vector[4] = {x1, y1, disparity, 1.0};
    double homogeneous[4] = {0.0, 0.0, 0.0, 0.0};
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            homogeneous[r] += g.Q.at<double>(r, c) * vector[c];
    if (homogeneous[3] == 0.0) return result;

    const double X = homogeneous[0] / homogeneous[3];
    const double Y = homogeneous[1] / homogeneous[3];
    const double Z = homogeneous[2] / homogeneous[3];
    result.error = std::sqrt((X - point[0]) * (X - point[0]) +
                             (Y - point[1]) * (Y - point[1]) +
                             (Z - point[2]) * (Z - point[2]));
    result.depth      = Z;
    result.disparity  = disparity;
    result.ok         = true;
    return result;
}

// ---------------------------------------------------------------------------
// 4. Maps
// ---------------------------------------------------------------------------
// Built for the *uncropped* rectified size and then sliced, so remap writes
// straight into a crop-sized destination with the offset already applied.
Maps buildMaps(const Calibration& c, const Geometry& g, const Settings& s)
{
    cv::Mat full1x, full1y, full2x, full2y;
    cv::initUndistortRectifyMap(c.M1, c.D1, g.R1, g.P1, g.fullSize, s.mapType, full1x, full1y);
    cv::initUndistortRectifyMap(c.M2, c.D2, g.R2, g.P2, g.fullSize, s.mapType, full2x, full2y);

    Maps maps;
    if (g.crop == cv::Rect(0, 0, g.fullSize.width, g.fullSize.height)) {
        maps.map1x = full1x; maps.map1y = full1y;
        maps.map2x = full2x; maps.map2y = full2y;
    } else {
        maps.map1x = full1x(g.crop).clone();
        maps.map1y = full1y(g.crop).clone();
        maps.map2x = full2x(g.crop).clone();
        maps.map2y = full2y(g.crop).clone();
    }
    return maps;
}

// ---------------------------------------------------------------------------
// 5. Remap
// ---------------------------------------------------------------------------
struct RectifiedPair {
    cv::Mat left, right, valid;
};

RectifiedPair rectifyPair(const ImagePair& pair, const Maps& maps, const Geometry& g,
                          const Settings& s)
{
    const cv::Mat left  = cv::imread(pair.left.string(),  cv::IMREAD_COLOR);
    const cv::Mat right = cv::imread(pair.right.string(), cv::IMREAD_COLOR);
    if (left.empty())  throw std::runtime_error("cannot read " + pair.left.string());
    if (right.empty()) throw std::runtime_error("cannot read " + pair.right.string());
    if (left.size() != g.sourceSize || right.size() != g.sourceSize) {
        std::ostringstream os;
        os << "size mismatch for pair " << pair.key << ": calibration was made for "
           << g.sourceSize.width << "x" << g.sourceSize.height << " but the images are "
           << left.size().width << "x" << left.size().height;
        throw std::runtime_error(os.str());
    }

    const int interpolation = s.useCubic ? cv::INTER_CUBIC : cv::INTER_LINEAR;

    RectifiedPair out;
    cv::remap(left,  out.left,  maps.map1x, maps.map1y, interpolation, cv::BORDER_CONSTANT);
    cv::remap(right, out.right, maps.map2x, maps.map2y, interpolation, cv::BORDER_CONSTANT);

    if (s.writeMask) {
        // Remapping a white image leaves 0 exactly where the source had no data,
        // which is the only reliable way to find the border: image content itself
        // can be black. Erode by one pixel because the interpolation at the
        // boundary blends the constant border colour into the result.
        const cv::Mat white(g.sourceSize, CV_8UC1, cv::Scalar(255));
        cv::remap(white, out.valid, maps.map1x, maps.map1y, cv::INTER_NEAREST,
                  cv::BORDER_CONSTANT, cv::Scalar(0));
        cv::Mat mask2;
        cv::remap(white, mask2, maps.map2x, maps.map2y, cv::INTER_NEAREST,
                  cv::BORDER_CONSTANT, cv::Scalar(0));
        cv::bitwise_and(out.valid, mask2, out.valid);
        const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, {3, 3});
        cv::erode(out.valid, out.valid, kernel);
    }
    return out;
}

// ---------------------------------------------------------------------------
// 5b. Closed-loop check on the rectified pixels
// ---------------------------------------------------------------------------
// Rectification is only useful if the *output* pair is row-aligned. Everything
// up to here verified the matrices; this detects the board again in the remapped
// images, so a missing crop, a wrong map size or an unadjusted Q all show up.
struct RowCheck {
    bool   measured = false;
    int    pairs    = 0;
    int    corners  = 0;
    double meanDy   = 0.0;
    double p95Dy    = 0.0;
    double maxDy    = 0.0;
    int    cornersOutsideMask = 0;
};

std::vector<cv::Point2f> findBoard(const cv::Mat& gray, const cv::Size& pattern, bool useSb)
{
    std::vector<cv::Point2f> corners;
    if (useSb) {
        if (cv::findChessboardCornersSB(gray, pattern, corners,
                                        cv::CALIB_CB_EXHAUSTIVE | cv::CALIB_CB_ACCURACY))
            return corners;
        return {};
    }
    const int flags = cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE;
    if (!cv::findChessboardCorners(gray, pattern, corners, flags)) return {};
    cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
                     cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.001));
    return corners;
}

// The detector walks the board in its own order, which can start from the
// opposite corner in the two views. Both orderings are tried and the one with the
// smaller row mismatch wins; otherwise a mirrored board looks like a 100 px error.
// The detector walks the board in its own order, which can start from the
// opposite corner in the two views. Both orderings are tried and the one with the
// smaller mean row mismatch wins; otherwise a mirrored board looks like a 100 px
// error and hides a good rectification.
std::vector<double> matchRowErrors(const std::vector<cv::Point2f>& left,
                                   const std::vector<cv::Point2f>& right)
{
    if (left.size() != right.size() || left.empty()) return {};

    std::vector<double> straight, reversed;
    straight.reserve(left.size());
    reversed.reserve(left.size());
    for (std::size_t i = 0; i < left.size(); ++i) {
        straight.push_back(std::abs(left[i].y - right[i].y));
        reversed.push_back(std::abs(left[i].y - right[left.size() - 1 - i].y));
    }
    const auto mean = [](const std::vector<double>& v) {
        double sum = 0.0;
        for (double x : v) sum += x;
        return sum / static_cast<double>(v.size());
    };
    return mean(reversed) < mean(straight) ? reversed : straight;
}

int countOutsideMask(const std::vector<cv::Point2f>& left, const std::vector<cv::Point2f>& right,
                     const std::vector<double>& dy, const cv::Mat& valid)
{
    if (valid.empty() || dy.empty()) return 0;
    int outside = 0;
    const auto isOutside = [&valid](const cv::Point2f& point) {
        const int x = cvRound(point.x);
        const int y = cvRound(point.y);
        return x < 0 || y < 0 || x >= valid.cols || y >= valid.rows ||
               valid.at<unsigned char>(y, x) == 0;
    };
    for (std::size_t i = 0; i < left.size(); ++i) {
        if (isOutside(left[i]))  ++outside;
        if (isOutside(right[i])) ++outside;
    }
    return outside;
}

RowCheck verifyRows(const std::vector<ImagePair>& pairs, const Maps& maps, const Geometry& g,
                    const Settings& s)
{
    RowCheck total;
    if (s.verifyPattern.area() <= 0) return total;

    // Pool every corner of every view: the report quotes one distribution, the
    // same way the calibration report does, rather than the mean of per-view means.
    std::vector<double> errors;
    int views = 0;

    for (const ImagePair& pair : pairs) {
        const RectifiedPair rect = rectifyPair(pair, maps, g, s);
        cv::Mat leftGray, rightGray;
        cv::cvtColor(rect.left,  leftGray,  cv::COLOR_BGR2GRAY);
        cv::cvtColor(rect.right, rightGray, cv::COLOR_BGR2GRAY);
        const std::vector<cv::Point2f> leftCorners  = findBoard(leftGray,  s.verifyPattern, s.verifyWithSb);
        const std::vector<cv::Point2f> rightCorners = findBoard(rightGray, s.verifyPattern, s.verifyWithSb);
        const std::vector<double> dy = matchRowErrors(leftCorners, rightCorners);
        if (dy.empty()) continue;

        ++views;
        errors.insert(errors.end(), dy.begin(), dy.end());
        total.cornersOutsideMask += countOutsideMask(leftCorners, rightCorners, dy, rect.valid);
    }

    if (views > 0) {
        std::sort(errors.begin(), errors.end());
        const auto mean = [](const std::vector<double>& v) {
            double sum = 0.0;
            for (double x : v) sum += x;
            return sum / static_cast<double>(v.size());
        };
        total.measured = true;
        total.pairs    = views;
        total.corners  = static_cast<int>(errors.size());
        total.meanDy   = mean(errors);
        total.p95Dy    = errors[static_cast<std::size_t>(0.95 * static_cast<double>(errors.size() - 1))];
        total.maxDy    = errors.back();
    }
    return total;
}

// ---------------------------------------------------------------------------
// 6. Output
// ---------------------------------------------------------------------------
void writeCameraJson(const fs::path& path, const Geometry& g)
{
    std::ofstream file(path);
    if (!file) throw std::runtime_error("cannot write " + path.string());
    file << std::setprecision(10);
    file << "{\n"
         << "    \"fx\": " << g.fx << ",\n"
         << "    \"baseline_m\": " << g.baseline << ",\n"
         << "    \"fy\": " << g.fy << ",\n"
         << "    \"cx\": " << g.cx1 << ",\n"
         << "    \"cy\": " << g.cy1 << ",\n"
         << "    \"doffs\": " << g.doffs << ",\n"
         << "    \"focal_baseline_px_m\": " << g.focalBaseline << ",\n"
         << "    \"image_width\": " << g.outputSize.width << ",\n"
         << "    \"image_height\": " << g.outputSize.height << ",\n"
         << "    \"min_disparity\": " << g.minDisparity << ",\n"
         << "    \"num_disparities\": " << g.numDisparities << ",\n"
         << "    \"note\": \"fx and baseline_m make this usable as algorithm_test's "
            "camera_file; the other fields are informational\"\n"
         << "}\n";
}

void writeRectification(const fs::path& path, const Calibration& c, const Geometry& g,
                        const Settings& s)
{
    cv::FileStorage fsOut(path.string(), cv::FileStorage::WRITE);
    if (!fsOut.isOpened()) throw std::runtime_error("cannot write " + path.string());
    fsOut << "M1" << c.M1 << "D1" << c.D1 << "M2" << c.M2 << "D2" << c.D2
          << "R1" << g.R1 << "R2" << g.R2
          << "P1" << g.P1 << "P2" << g.P2 << "Q" << g.Q
          << "image_width" << g.outputSize.width
          << "image_height" << g.outputSize.height
          << "crop_x" << g.crop.x << "crop_y" << g.crop.y
          << "fx" << g.fx << "fy" << g.fy
          << "cx1" << g.cx1 << "cy1" << g.cy1
          << "baseline_m" << g.baseline
          << "doffs" << g.doffs
          << "alpha" << s.alpha
          << "min_disparity" << g.minDisparity
          << "num_disparities" << g.numDisparities;
}

// Side-by-side rectified pair with horizontal guides, plus the mask overlay. A
// row mismatch from a broken rectification is visible at a glance.
cv::Mat buildPreview(const ImagePair& pair, const RectifiedPair& rect, const Geometry& g)
{
    cv::Mat canvas(g.outputSize.height, g.outputSize.width * 2, CV_8UC3);
    rect.left.copyTo(canvas(cv::Rect(0, 0, g.outputSize.width, g.outputSize.height)));
    rect.right.copyTo(canvas(cv::Rect(g.outputSize.width, 0, g.outputSize.width, g.outputSize.height)));

    for (int y = 0; y < g.outputSize.height; y += 24) {
        const cv::Scalar color = (y % 120 == 0) ? cv::Scalar(0, 220, 0) : cv::Scalar(70, 120, 70);
        cv::line(canvas, cv::Point(0, y), cv::Point(canvas.cols, y), color, 1, cv::LINE_AA);
    }

    if (!rect.valid.empty()) {
        // Tint the invalid pixels so an over-aggressive crop or a wrong alpha is
        // obvious rather than silently feeding zeros to the matcher.
        cv::Mat invalid;
        cv::bitwise_not(rect.valid, invalid);
        const cv::Mat tint(g.outputSize, CV_8UC3, cv::Scalar(60, 0, 90));
        for (int half = 0; half < 2; ++half) {
            cv::Mat region = canvas(cv::Rect(half * g.outputSize.width, 0,
                                             g.outputSize.width, g.outputSize.height));
            cv::Mat blended;
            cv::addWeighted(region, 0.45, tint, 0.55, 0.0, blended);
            blended.copyTo(region, invalid);
        }
    }

    char caption[256];
    std::snprintf(caption, sizeof caption, "%s   %dx%d   fx %.3f   B %.6f m   %zu of %zu px valid",
                  pair.key.c_str(), g.outputSize.width, g.outputSize.height, g.fx, g.baseline,
                  rect.valid.empty() ? static_cast<std::size_t>(g.outputSize.area())
                                     : static_cast<std::size_t>(cv::countNonZero(rect.valid)),
                  static_cast<std::size_t>(g.outputSize.area()));
    cv::putText(canvas, caption, cv::Point(8, g.outputSize.height - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    return canvas;
}

std::string buildMarkdown(const Settings& s, const Geometry& g, const std::vector<ImagePair>& pairs,
                          double validFraction, const RowCheck& rows)
{
    std::ostringstream md;
    md << "# 极线校正结果\n\n";
    md << "标定文件：`" << s.calibrationPath << "`\n\n";
    md << "## 尺寸\n\n";
    md << "| 项 | 值 |\n| --- | --- |\n";
    md << "| 校正前 | " << g.sourceSize.width << " × " << g.sourceSize.height << " |\n";
    md << "| 校正后（未裁） | " << g.fullSize.width << " × " << g.fullSize.height << " |\n";
    md << "| 裁剪原点 | (" << g.crop.x << ", " << g.crop.y << ") |\n";
    md << "| 输出 | " << g.outputSize.width << " × " << g.outputSize.height << " |\n";
    md << "| 有效像素 | " << std::fixed << std::setprecision(2) << validFraction * 100.0 << " % |\n";
    md << "| 图像对 | " << pairs.size() << " |\n\n";

    md << "## 几何\n\n";
    md << "| 项 | 值 |\n| --- | --- |\n";
    md << "| fx' | " << std::fixed << std::setprecision(4) << g.fx << " px |\n";
    md << "| fy' | " << g.fy << " px |\n";
    md << "| cx1' / cy1' | " << g.cx1 << " / " << g.cy1 << " px |\n";
    md << "| doffs (cx1'-cx2') | " << g.doffs << " px |\n";
    md << "| 基线 | " << std::setprecision(6) << g.baseline << " m |\n";
    md << "| f'·B | " << std::setprecision(4) << g.focalBaseline << " px·m |\n\n";
    md << "深度换算：`Z = f'·B / (d - doffs) = " << std::setprecision(4) << g.focalBaseline
       << " / (d - " << g.doffs << ")`，`d` 取校正后图像的视差。\n\n";

    if (g.disparityRangeKnown) {
        md << "## 视差窗口\n\n";
        if (g.depthNear > 0 && g.depthFar > 0)
            md << "对应深度 " << std::fixed << std::setprecision(3) << g.depthFar << " … "
               << g.depthNear << " m（近处视差大）：\n\n";
        else
            md << "由命令行指定：\n\n";
        md << "| 项 | 值 |\n| --- | --- |\n";
        md << "| 视差范围 | " << std::setprecision(2) << g.disparityFar << " … "
           << g.disparityNear << " px |\n";
        md << "| minDisparity | " << g.minDisparity << " |\n";
        md << "| numDisparities | " << g.numDisparities << " |\n\n";
        if (g.rangeClipped)
            md << "> 注意：完整视差范围超出输出宽度，窗口已被截断到 "
               << g.numDisparities << "。要保留完整范围，请减小 `--alpha` 的裁剪量或增大输出尺寸。\n\n";
    }

    if (rows.measured) {
        md << "## 行对齐验证\n\n";
        md << "在校正后的图像上重新检测棋盘格，按索引配对后统计行坐标差：\n\n";
        md << "| 项 | 值 |\n| --- | --- |\n";
        md << "| 视图 | " << rows.pairs << " |\n";
        md << "| 角点 | " << rows.corners << " |\n";
        md << "| \\|dy\\| 均值 | " << std::fixed << std::setprecision(4) << rows.meanDy << " px |\n";
        md << "| \\|dy\\| P95 | " << rows.p95Dy << " px |\n";
        md << "| \\|dy\\| 最大 | " << rows.maxDy << " px |\n";
        md << "| 掩膜外角点 | " << rows.cornersOutsideMask << " |\n\n";
    }

    md << "## 输出文件\n\n";
    md << "| 文件 | 内容 |\n| --- | --- |\n";
    md << "| `left/` `right/` | 校正后的图像对，尺寸一致、行对齐 |\n";
    if (s.writeMask)
        md << "| `mask/` | 每对的左右共同有效区（255 有效） |\n";
    md << "| `camera.json` | `fx` / `baseline_m`，可直接作为 `algorithm_test` 的 `camera_file` |\n";
    md << "| `rectification.yml` | `R1 R2 P1 P2 Q`（已按裁剪修正）与全部几何量 |\n";
    if (s.writePreview)
        md << "| `preview_<key>.png` | 左右拼接 + 水平参考线 + 无效区着色 |\n";
    md << "\n> 视差图的像素坐标与裁剪后的图像一致，`Q` 已相应修正，"
          "可直接传给 `cv::reprojectImageTo3D`。\n";
    return md.str();
}

// ---------------------------------------------------------------------------
// Command line
// ---------------------------------------------------------------------------
void printUsage(const char* program)
{
    std::printf(
        "usage: %s --calibration <stereo.yml> [options]\n\n"
        "  --calibration <yml>    stereo.yml from stereo_calibration   (required)\n"
        "  --images <dir>         batch: rectify every leftNN/rightNN pair (default: .)\n"
        "  --left <path> --right <path>   single pair instead of --images\n"
        "  --left-prefix <str>    left file prefix                     (default: left)\n"
        "  --right-prefix <str>   right file prefix                    (default: right)\n"
        "  --ext <.jpg>           input extension                      (default: .jpg)\n"
        "  --output <dir>         output directory                     (default: output)\n"
        "  --out-ext <.png>       output image extension               (default: .png)\n"
        "  --alpha <0..1>         stereoRectify balance, 0 crops, 1 keeps (default: 0)\n"
        "  --new-size <WxH>       rectified size, default same as the input\n"
        "  --no-crop              keep the full rectified frame instead of the common ROI\n"
        "  --pad <n>              keep n extra columns around the common ROI (default: 0)\n"
        "  --depth-range <near>,<far>  metres; derives minDisparity/numDisparities\n"
        "  --min-disparity <n>    explicit minDisparity, overrides --depth-range\n"
        "  --num-disparities <n>  explicit numDisparities, rounded up to a multiple of 16\n"
        "  --keep-rectification   reuse the stored R1/R2/P1/P2 (alpha would be ignored)\n"
        "  --float-maps           CV_32FC1 maps instead of the fixed-point CV_16SC2\n"
        "  --cubic                INTER_CUBIC instead of INTER_LINEAR\n"
        "  --verify-pattern <WxH> detect a chessboard in the rectified pair and report\n"
        "                         the row mismatch (needs inner-corner counts)\n"
        "  --verify-sb            use findChessboardCornersSB for --verify-pattern\n"
        "  --no-mask              skip the validity mask\n"
        "  --no-preview           skip the preview images\n"
        "  --quiet                summary only\n\n",
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
        if (arg == "--calibration")            s.calibrationPath = next(i, "--calibration");
        else if (arg == "--images")            s.imageDirectory = next(i, "--images");
        else if (arg == "--left")              s.leftPath = next(i, "--left");
        else if (arg == "--right")             s.rightPath = next(i, "--right");
        else if (arg == "--left-prefix")       s.leftPrefix = next(i, "--left-prefix");
        else if (arg == "--right-prefix")      s.rightPrefix = next(i, "--right-prefix");
        else if (arg == "--ext")               s.extension = next(i, "--ext");
        else if (arg == "--output")            s.outputDirectory = next(i, "--output");
        else if (arg == "--out-ext")           s.outputExtension = next(i, "--out-ext");
        else if (arg == "--alpha")             s.alpha = std::stod(next(i, "--alpha"));
        else if (arg == "--pad")               s.padColumns = std::stoi(next(i, "--pad"));
        else if (arg == "--min-disparity")     s.explicitMinDisparity = std::stoi(next(i, "--min-disparity"));
        else if (arg == "--num-disparities")   s.explicitNumDisparities = std::stoi(next(i, "--num-disparities"));
        else if (arg == "--no-crop")           s.cropToCommonRoi = false;
        else if (arg == "--keep-rectification") s.keepStoredRectification = true;
        else if (arg == "--float-maps")        s.mapType = CV_32FC1;
        else if (arg == "--cubic")             s.useCubic = true;
        else if (arg == "--verify-sb")         s.verifyWithSb = true;
        else if (arg == "--no-mask")           s.writeMask = false;
        else if (arg == "--no-preview")        s.writePreview = false;
        else if (arg == "--quiet")             s.verbose = false;
        else if (arg == "--new-size") {
            const std::string value = next(i, "--new-size");
            const std::size_t sep = value.find('x');
            if (sep == std::string::npos)
                throw std::invalid_argument("--new-size expects WxH, e.g. 640x480");
            s.newImageSize = cv::Size(std::stoi(value.substr(0, sep)),
                                      std::stoi(value.substr(sep + 1)));
        } else if (arg == "--verify-pattern") {
            const std::string value = next(i, "--verify-pattern");
            const std::size_t sep = value.find('x');
            if (sep == std::string::npos)
                throw std::invalid_argument("--verify-pattern expects WxH, e.g. 9x6");
            s.verifyPattern = cv::Size(std::stoi(value.substr(0, sep)),
                                       std::stoi(value.substr(sep + 1)));
        } else if (arg == "--depth-range") {
            const std::string value = next(i, "--depth-range");
            const std::size_t sep = value.find(',');
            if (sep == std::string::npos)
                throw std::invalid_argument("--depth-range expects near,far, e.g. 0.15,1.5");
            s.depthNear = std::stod(value.substr(0, sep));
            s.depthFar  = std::stod(value.substr(sep + 1));
        } else if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown option: " + arg);
        }
    }

    if (s.calibrationPath.empty())
        throw std::invalid_argument("--calibration is required (stereo.yml from stereo_calibration)");
    if (s.alpha < 0.0 || s.alpha > 1.0)
        throw std::invalid_argument("--alpha must be within [0, 1]");
    if (s.padColumns < 0)
        throw std::invalid_argument("--pad must not be negative");
    if (s.depthFar > 0 && !(s.depthFar > s.depthNear))
        throw std::invalid_argument("--depth-range needs far > near");
    if (s.depthNear > 0 && s.explicitNumDisparities > 0)
        throw std::invalid_argument("--depth-range and --num-disparities are mutually exclusive");
    return s;
}

}  // namespace rectify

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    using namespace rectify;

    try {
        const Settings settings = parseArgs(argc, argv);
        std::vector<std::pair<std::string, double>> timings;

        std::printf("\n=== 1. 标定 ===\n");
        Calibration calibration;
        {
            StageTimer timer(timings, "load calibration");
            calibration = loadCalibration(settings);
        }
        std::printf("  %s\n", settings.calibrationPath.c_str());
        std::printf("  %s，%s\n",
                    calibration.hasExtrinsics ? "含 R/T，可重算校正" : "无 R/T",
                    calibration.hasRectification ? "含已存 R1/P1" : "无已存校正矩阵");

        std::printf("\n=== 2. 图像对 ===\n");
        const std::vector<ImagePair> pairs = discoverPairs(settings);
        if (pairs.empty()) throw std::runtime_error("no image pairs found");
        std::printf("  找到 %zu 对图像\n", pairs.size());

        const cv::Mat firstLeft = cv::imread(pairs.front().left.string(), cv::IMREAD_COLOR);
        if (firstLeft.empty()) throw std::runtime_error("cannot read " + pairs.front().left.string());
        const cv::Size imageSize = firstLeft.size();

        std::printf("\n=== 3. 校正几何 ===\n");
        Geometry geometry;
        {
            StageTimer timer(timings, "compute geometry");
            geometry = computeGeometry(calibration, imageSize, settings);
        }
        std::printf("  fx' %.4f  fy' %.4f  cx1' %.4f  cy1' %.4f\n",
                    geometry.fx, geometry.fy, geometry.cx1, geometry.cy1);
        std::printf("  doffs %.4f px   基线 %.6f m   f'·B %.4f px·m\n",
                    geometry.doffs, geometry.baseline, geometry.focalBaseline);
        std::printf("  裁剪 (%d, %d)，输出 %dx%d\n",
                    geometry.crop.x, geometry.crop.y,
                    geometry.outputSize.width, geometry.outputSize.height);

        const SelfCheck selfCheck = checkQAgainstP(geometry);
        if (selfCheck.ok)
            std::printf("  Q/P 自检：深度 %.4f m，视差 %.2f px，重构误差 %.3e m\n",
                        selfCheck.depth, selfCheck.disparity, selfCheck.error);
        else
            std::printf("  Q/P 自检：跳过（Q 不可用）\n");

        if (geometry.disparitySignFlipped)
            std::printf("  警告：基线方向使视差为负（Tx > 0），算法要求非负视差，"
                        "请交换 --left / --right 后重新标定\n");
        if (geometry.disparityRangeKnown) {
            std::printf("  视差 %.2f … %.2f px  ->  minDisparity %d, numDisparities %d",
                        geometry.disparityFar, geometry.disparityNear,
                        geometry.minDisparity, geometry.numDisparities);
            if (geometry.depthFar > 0 && geometry.depthNear > 0)
                std::printf("  (深度 %.3f … %.3f m)", geometry.depthFar, geometry.depthNear);
            std::printf("\n");
            if (geometry.rangeClipped)
                std::printf("  警告：完整视差范围超出图像宽度，窗口已截断\n");
        } else {
            std::printf("  提示：用 --depth-range <near>,<far> 可推导视差窗口\n");
        }

        Maps maps;
        {
            StageTimer timer(timings, "build maps");
            maps = buildMaps(calibration, geometry, settings);
        }

        const fs::path outputDirectory(settings.outputDirectory);
        const fs::path leftDirectory  = outputDirectory / "left";
        const fs::path rightDirectory = outputDirectory / "right";
        const fs::path maskDirectory  = outputDirectory / "mask";
        fs::create_directories(leftDirectory);
        fs::create_directories(rightDirectory);
        if (settings.writeMask) fs::create_directories(maskDirectory);

        std::printf("\n=== 4. 重采样 ===\n");
        double validSum = 0.0;
        int    validCount = 0;
        {
            StageTimer timer(timings, "rectify");
            for (std::size_t index = 0; index < pairs.size(); ++index) {
                const ImagePair& pair = pairs[index];
                const RectifiedPair rect = rectifyPair(pair, maps, geometry, settings);
                const std::string name = pair.key + settings.outputExtension;
                cv::imwrite(joinPath(leftDirectory, name), rect.left);
                cv::imwrite(joinPath(rightDirectory, name), rect.right);

                double fraction = 1.0;
                if (!rect.valid.empty()) {
                    const int valid = cv::countNonZero(rect.valid);
                    fraction = static_cast<double>(valid) /
                               static_cast<double>(rect.valid.total());
                    validSum += fraction;
                    ++validCount;
                    if (settings.writeMask) cv::imwrite(joinPath(maskDirectory, name), rect.valid);
                }
                if (settings.writePreview && index == 0) {
                    cv::imwrite(joinPath(outputDirectory, "preview_" + pair.key + ".png"),
                                buildPreview(pair, rect, geometry));
                }
                if (settings.verbose)
                    std::printf("  %-6s 有效 %5.1f %%\n", pair.key.c_str(), fraction * 100.0);
            }
        }
        const double meanValid = validCount > 0 ? validSum / validCount : 1.0;

        RowCheck rows;
        if (settings.verifyPattern.area() > 0) {
            std::printf("\n=== 5. 行对齐验证 ===\n");
            StageTimer timer(timings, "verify rows");
            rows = verifyRows(pairs, maps, geometry, settings);
            if (rows.measured) {
                std::printf("  %d 对 / %d 角点：|dy| 均值 %.4f px，P95 %.4f px，最大 %.4f px\n",
                            rows.pairs, rows.corners, rows.meanDy, rows.p95Dy, rows.maxDy);
                if (rows.cornersOutsideMask > 0)
                    std::printf("  警告：%d 个角点落在有效掩膜之外\n", rows.cornersOutsideMask);
                std::printf("  %s\n", rows.meanDy < 0.5 ? "结论: 行对齐合格" : "结论: 行对齐不合格，检查 alpha 与标定");
            } else {
                std::printf("  未检测到棋盘，跳过（图案或图像不匹配）\n");
            }
        }

        std::printf("\n=== 6. 输出 ===\n");
        writeCameraJson(outputDirectory / "camera.json", geometry);
        writeRectification(outputDirectory / "rectification.yml", calibration, geometry, settings);
        {
            std::ofstream report(outputDirectory / "rectification_report.md");
            if (!report) throw std::runtime_error("cannot write the report");
            report << buildMarkdown(settings, geometry, pairs, meanValid, rows);
        }
        std::printf("  %s\n", (outputDirectory / "camera.json").string().c_str());
        std::printf("  %s\n", (outputDirectory / "rectification.yml").string().c_str());
        std::printf("  %s\n", (outputDirectory / "rectification_report.md").string().c_str());
        std::printf("  图像 %zu 对 -> %s / %s\n", pairs.size(),
                    leftDirectory.string().c_str(), rightDirectory.string().c_str());
        if (settings.writeMask)
            std::printf("  有效掩膜 -> %s\n", maskDirectory.string().c_str());
        std::printf("  平均有效像素 %.2f %%\n", meanValid * 100.0);

        std::printf("\n--- 阶段耗时 ---\n");
        double total = 0.0;
        for (const auto& [name, ms] : timings) {
            std::printf("  %-22s %8.1f ms\n", name.c_str(), ms);
            total += ms;
        }
        std::printf("  %-22s %8.1f ms\n", "total", total);

        if (meanValid < 0.6)
            std::printf("\n警告：有效像素不足 60 %%，考虑提高 --alpha 或改用 --no-crop\n");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "\nError: %s\n", error.what());
        return 2;
    }
}
