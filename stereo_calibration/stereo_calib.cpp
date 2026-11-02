// ===========================================================================
//  Command line front end for the stereo calibration pipeline.
//
//  The pipeline itself lives in calib_core.cpp and is shared with the Qt5 GUI
//  (stereo_calib_gui.cpp). This file only owns argument parsing, the optional
//  preview window and the process exit code.
// ===========================================================================
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>

#include "calib_core.hpp"

namespace {

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
        "  --quiet               skip the per-image and per-view detail\n\n"
        "The GUI (stereo_calib_gui) exposes the same options.\n\n",
        program);
}

calib::Settings parseArgs(int argc, char** argv)
{
    calib::Settings s;
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

}  // namespace

int main(int argc, char** argv)
{
    using namespace calib;

    try {
        const Settings settings = parseArgs(argc, argv);

        RunResult result = run(settings);
        writeOutputs(result, settings);

        // The preview window is a CLI-only convenience: the GUI has its own
        // viewer, and mixing cv::waitKey() with a Qt event loop deadlocks.
        if (settings.saveDebugImages && settings.showWindow && !result.preview.empty() &&
            (std::getenv("DISPLAY") || std::getenv("WAYLAND_DISPLAY"))) {
            try {
                cv::imshow("epipolar check (any key to close)", result.preview);
                cv::waitKey(0);
                cv::destroyAllWindows();
            } catch (const cv::Exception& error) {
                std::cerr << "display failed (" << error.what() << "); see "
                          << (fs::path(settings.outputDirectory) / "epipolar_check.png") << '\n';
            }
        }

        printConclusion(result);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "\nError: " << error.what() << '\n';
        return 1;
    }
}
