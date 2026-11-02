#ifndef STEREO_CALIB_GUI_PARAMS_HPP
#define STEREO_CALIB_GUI_PARAMS_HPP

#include <QString>

#include "calib_core.hpp"

/// The subset of calib::Settings the form exposes, plus the values that only
/// matter to the GUI (the reference yml path is part of Settings, the window
/// geometry is not).
struct GuiParams {
    QString imageDir = ".";
    QString leftPrefix = "left";
    QString rightPrefix = "right";
    QString extension = ".jpg";

    int boardWidth = 9;
    int boardHeight = 6;
    double squareMm = 25.0;
    bool useSbDetector = false;

    double alpha = 0.0;
    bool refineIntrinsics = false;
    bool sameFocalLength = false;

    QString outputDir = "output";
    QString reference;
    bool saveDebugImages = true;

    /// Square size is edited in millimetres and handed to the pipeline in
    /// metres, which is what Settings documents.
    calib::Settings toSettings() const;

    static GuiParams load();
    void save() const;
};

#endif  // STEREO_CALIB_GUI_PARAMS_HPP
