// ===========================================================================
//  calib_core.hpp — the stereo calibration pipeline shared by the CLI and the
//  Qt5 GUI.
//
//  The implementation is the original stereo_calib.cpp logic, unchanged; the
//  only addition is an injectable log sink, so the GUI can show the same text
//  the CLI writes to stdout.
// ===========================================================================
#ifndef STEREO_CALIB_CORE_HPP
#define STEREO_CALIB_CORE_HPP

#include <atomic>
#include <filesystem>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

#if defined(__GNUC__) || defined(__clang__)
#define CALIB_PRINTF_FORMAT(f, a) __attribute__((format(printf, f, a)))
#else
#define CALIB_PRINTF_FORMAT(f, a)
#endif

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

    // Views (by numeric suffix, e.g. "05") the caller decided to drop after
    // inspecting the per-view error table. The CLI never sets this, so the
    // command-line behaviour is unchanged.
    std::vector<std::string> excludedKeys;
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
// Logging. With no sink installed print() writes to stdout byte for byte what
// the original std::printf() calls wrote.
// ---------------------------------------------------------------------------
using LogSink = std::function<void(const std::string&)>;

void setLogSink(LogSink sink);
void clearLogSink();
void setLogEnabled(bool enabled);
void print(const char* format, ...) CALIB_PRINTF_FORMAT(1, 2);

// ---------------------------------------------------------------------------
// Run control: progress reporting and cooperative cancellation, both optional.
// ---------------------------------------------------------------------------
/// Thrown by the pipeline when RunHooks::cancel is set. The GUI catches it and
/// leaves the output directory untouched.
class Cancelled : public std::runtime_error {
public:
    Cancelled() : std::runtime_error("cancelled") {}
};

struct RunHooks {
    std::function<void(const std::string& stage, int done, int total)> progress;
    std::atomic<bool>* cancel = nullptr;
};

struct PairStatus {
    std::string key;
    bool ok = false;
    std::string detail;   // why the pair was dropped, empty when ok
};

struct DetectResult {
    std::vector<ImagePair> pairs;                  // every discovered pair
    std::vector<ImagePair> kept;                   // full board in both eyes
    std::vector<std::vector<cv::Point2f>> leftCorners;
    std::vector<std::vector<cv::Point2f>> rightCorners;
    std::vector<PairStatus> status;
    cv::Size imageSize;
    std::vector<std::pair<std::string, double>> timings;
};

struct RunResult {
    std::vector<ImagePair> pairs;                  // the views that were used
    std::vector<std::vector<cv::Point2f>> leftCorners;
    std::vector<std::vector<cv::Point2f>> rightCorners;
    cv::Size imageSize;
    CameraModel camera1;
    CameraModel camera2;
    StereoModel stereo;
    Verification verification;
    std::string referenceSummary;
    std::vector<std::pair<std::string, double>> timings;

    cv::Mat preview;                               // epipolar_check canvas
    cv::Mat rectLeft;
    cv::Mat rectRight;

    bool epipolarOk = false;
    bool edgeOk = false;
    std::string outputDirectory;
};

// ---------------------------------------------------------------------------
// 1.-2. Discovery and corner detection
// ---------------------------------------------------------------------------
std::vector<ImagePair> discoverPairs(const Settings& s);

std::vector<cv::Point2f> detectCorners(const cv::Mat& gray, const Settings& s, bool& usedSb);

std::vector<std::vector<cv::Point2f>> detectAll(
    const std::vector<ImagePair>& pairs, const Settings& s,
    std::vector<std::vector<cv::Point2f>>& rightCorners, std::vector<ImagePair>& kept,
    bool verbose = true, const RunHooks& hooks = RunHooks(),
    std::vector<PairStatus>* status = nullptr);

bool reverseIfMirrored(const std::vector<cv::Point2f>& left, std::vector<cv::Point2f>& right,
                       int boardWidth);

/// Stages 1 and 2 only: what the GUI's "Detect corners" button runs.
DetectResult detectPairs(const Settings& s, const RunHooks& hooks = RunHooks());

// ---------------------------------------------------------------------------
// 3.-5. Calibration and verification
// ---------------------------------------------------------------------------
std::vector<cv::Point3f> boardObjectPoints(const Settings& s);

CameraModel calibrateMono(const std::vector<std::vector<cv::Point3f>>& objectPoints,
                          const std::vector<std::vector<cv::Point2f>>& imagePoints,
                          const cv::Size& imageSize, const char* label);

StereoModel calibrateStereoPair(const std::vector<std::vector<cv::Point3f>>& objectPoints,
                                const std::vector<std::vector<cv::Point2f>>& left,
                                const std::vector<std::vector<cv::Point2f>>& right,
                                const cv::Size& imageSize, CameraModel& camera1,
                                CameraModel& camera2, const Settings& s);

void appendEpipolarErrors(const std::vector<std::vector<cv::Point2f>>& left,
                          const std::vector<std::vector<cv::Point2f>>& right,
                          const CameraModel& camera1, const CameraModel& camera2,
                          const StereoModel& stereo, std::vector<double>& dy,
                          std::vector<cv::Point2f>* rectifiedLeft,
                          std::vector<cv::Point2f>* rectifiedRight);

double planarityRms(const std::vector<cv::Point3d>& points);

cv::Mat toPointMatrix(const std::vector<cv::Point2f>& points);

Verification verify(const std::vector<std::vector<cv::Point2f>>& left,
                    const std::vector<std::vector<cv::Point2f>>& right,
                    const CameraModel& camera1, const CameraModel& camera2,
                    const StereoModel& stereo, const Settings& s);

// ---------------------------------------------------------------------------
// 6. Output
// ---------------------------------------------------------------------------
void saveIntrinsics(const fs::path& path, const CameraModel& camera1, const CameraModel& camera2);

void saveStereo(const fs::path& path, const StereoModel& stereo, const CameraModel& camera1,
                const CameraModel& camera2);

bool compareWithReference(const fs::path& path, const CameraModel& camera, std::string& summary);

cv::Mat rectifiedPreview(const ImagePair& pair, const CameraModel& camera1,
                         const CameraModel& camera2, const StereoModel& stereo,
                         const cv::Size& imageSize, cv::Mat& rectLeft, cv::Mat& rectRight);

void printMatrix(const char* label, const cv::Mat& m);

/// Dense, one row per matrix row; used by the GUI matrix tab.
std::string matrixToString(const cv::Mat& m);

std::string buildMarkdown(const Settings& s, const std::vector<ImagePair>& pairs,
                          const cv::Size& imageSize, const CameraModel& camera1,
                          const CameraModel& camera2, const StereoModel& stereo,
                          const Verification& verification, const std::string& referenceSummary);

/// Colour image with the detected corners drawn on top, optionally numbered.
cv::Mat drawDetectedCorners(const cv::Mat& bgr, const std::vector<cv::Point2f>& corners,
                            const cv::Size& pattern, bool showIds = false);

// ---------------------------------------------------------------------------
// Whole run, in the same order and with the same log text as the CLI prints.
// ---------------------------------------------------------------------------
RunResult run(const Settings& s, const RunHooks& hooks = RunHooks());

/// Writes intrinsics.yml, stereo.yml, the debug images and the report, and
/// fills result.preview / rectLeft / rectRight.
void writeOutputs(RunResult& result, const Settings& s);

/// Stage timings and the closing verdict, exactly as the CLI prints them.
void printConclusion(const RunResult& result);

}  // namespace calib

#endif  // STEREO_CALIB_CORE_HPP
