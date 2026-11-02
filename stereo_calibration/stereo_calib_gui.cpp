// ===========================================================================
//  Qt5 front end for the stereo calibration pipeline.
//
//  The window is a thin shell around calib_core: it collects Settings, runs one
//  job on a QThread, and shows the log, the metrics and the rectified images.
//
//  --selftest [--images DIR] [--output DIR] [--pattern WxH] [--square-size MM]
//             [--screenshot FILE] [--screenshot-tab log|results]
//             [--exclude 05,06] [--detect-only]
//  runs one job without user interaction and exits with 0 on success.
//  Combine with QT_QPA_PLATFORM=offscreen on a machine without a display.
// ===========================================================================
#include <QApplication>
#include <QCoreApplication>
#include <QDir>
#include <QSettings>
#include <QStringList>
#include <QTimer>

#include <cstdio>

#include "gui/fonts.hpp"
#include "gui/main_window.hpp"

namespace {

void printUsage()
{
    std::printf(
        "usage: stereo_calib_gui [--reset-settings]\n"
        "       stereo_calib_gui --selftest [options]\n"
        "\n"
        "  --selftest            run one calibration without interaction and exit\n"
        "  --images <dir>        image directory for --selftest        (default: .)\n"
        "  --output <dir>        output directory for --selftest       (default: output)\n"
        "  --pattern <WxH>       inner corners for --selftest          (default: 9x6)\n"
        "  --square-size <mm>    square size for --selftest            (default: 25)\n"
        "  --screenshot <file>   save a window screenshot after --selftest\n"
        "  --screenshot-tab <t>  log (default) or results, which tab to capture\n"
        "  --exclude <list>      comma separated view keys to drop, e.g. 05,06\n"
        "  --detect-only         stop after corner detection (no calibration)\n"
        "  --reset-settings      forget the stored parameters before opening\n");
}

QString valueOf(const QStringList& args, const QString& name)
{
    const int index = args.indexOf(name);
    if (index < 0 || index + 1 >= args.size()) return QString();
    return args[index + 1];
}

}  // namespace

int main(int argc, char** argv)
{
#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
    QApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
    QApplication::setAttribute(Qt::AA_UseHighDpiPixmaps);
#endif
    QApplication app(argc, argv);
    QCoreApplication::setOrganizationName(QStringLiteral("stereo_calib"));
    QCoreApplication::setApplicationName(QStringLiteral("stereo_calib_gui"));
    QCoreApplication::setApplicationVersion(QStringLiteral("1.0"));
    ensureCjkFont();

    const QStringList args = QCoreApplication::arguments();
    if (args.contains(QStringLiteral("--help")) || args.contains(QStringLiteral("-h"))) {
        printUsage();
        return 0;
    }
    if (args.contains(QStringLiteral("--reset-settings"))) {
        QSettings().clear();
    }

    MainWindow window;
    const bool selftest = args.contains(QStringLiteral("--selftest"));

    if (!selftest) {
        window.show();
        return app.exec();
    }

    const QString images = valueOf(args, QStringLiteral("--images"));
    const QString output = valueOf(args, QStringLiteral("--output"));
    const QString screenshot = valueOf(args, QStringLiteral("--screenshot"));
    const QString pattern = valueOf(args, QStringLiteral("--pattern"));
    const QString square = valueOf(args, QStringLiteral("--square-size"));

    int boardWidth = 9;
    int boardHeight = 6;
    if (!pattern.isEmpty()) {
        const QStringList parts = pattern.split(QLatin1Char('x'));
        if (parts.size() != 2) {
            std::fprintf(stderr, "--pattern expects WxH, e.g. 9x6\n");
            return 2;
        }
        boardWidth = parts[0].toInt();
        boardHeight = parts[1].toInt();
    }
    const double squareMm = square.isEmpty() ? 25.0 : square.toDouble();

    window.applyTestParams(images.isEmpty() ? QStringLiteral(".") : images,
                           output.isEmpty() ? QStringLiteral("output") : output,
                           boardWidth, boardHeight, squareMm);

    const bool detectOnly = args.contains(QStringLiteral("--detect-only"));
    const QString tab = valueOf(args, QStringLiteral("--screenshot-tab"));
    const QString excludes = valueOf(args, QStringLiteral("--exclude"));
    if (!excludes.isEmpty()) window.setExcludedViews(excludes.split(QLatin1Char(',')));

    QObject::connect(&window, &MainWindow::jobFinished, &app,
                     [&window, &screenshot, detectOnly, tab](bool ok) {
        if (detectOnly) {
            if (auto detected = window.detectResult()) {
                std::printf("selftest-detect: usable=%zu of %zu pairs\n",
                            detected->kept.size(), detected->pairs.size());
            }
            if (!screenshot.isEmpty() && !window.grab().save(screenshot))
                std::fprintf(stderr, "could not save %s\n", qPrintable(screenshot));
            QCoreApplication::exit(ok ? 0 : 3);
            return;
        }
        auto result = window.runResult();
        if (result) {
            std::printf("selftest: views=%zu stereo_rms=%.4f baseline_m=%.6f "
                        "epipolar_mean=%.4f planarity_rms_mm=%.4f\n",
                        result->pairs.size(), result->stereo.rms, result->stereo.baseline(),
                        result->verification.epipolarMean,
                        result->verification.planarityRms * 1000.0);
        }
        if (tab == QLatin1String("results")) window.showResultsTab();
        if (!screenshot.isEmpty() && !window.grab().save(screenshot))
            std::fprintf(stderr, "could not save %s\n", qPrintable(screenshot));
        QCoreApplication::exit(ok && result ? 0 : 3);
    });

    // The window has to be shown before the run so grab() has a laid-out widget,
    // which also works with QT_QPA_PLATFORM=offscreen.
    window.show();
    QTimer::singleShot(0, &window, [&window, detectOnly]() {
        if (detectOnly) window.startDetection();
        else             window.startCalibration();
    });
    QTimer::singleShot(600000, &app, []() {
        std::fprintf(stderr, "selftest timed out\n");
        QCoreApplication::exit(4);
    });
    return app.exec();
}
