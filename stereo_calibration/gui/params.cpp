#include "params.hpp"

#include <QSettings>

namespace {

QString readString(QSettings& s, const char* key, const QString& fallback)
{
    return s.value(key, fallback).toString();
}

}  // namespace

calib::Settings GuiParams::toSettings() const
{
    calib::Settings s;
    s.imageDirectory      = imageDir.toStdString();
    s.leftPrefix          = leftPrefix.toStdString();
    s.rightPrefix         = rightPrefix.toStdString();
    s.extension           = extension.toStdString();
    s.boardWidth          = boardWidth;
    s.boardHeight         = boardHeight;
    s.squareSize          = squareMm / 1000.0;
    s.useSbDetector       = useSbDetector;
    s.alpha               = alpha;
    s.refineIntrinsics    = refineIntrinsics;
    s.sameFocalLength     = sameFocalLength;
    s.outputDirectory     = outputDir.toStdString();
    s.referenceIntrinsics = reference.toStdString();
    s.saveDebugImages     = saveDebugImages;
    // The GUI has its own viewer, never cv::imshow().
    s.showWindow          = false;
    s.verbose             = true;
    return s;
}

GuiParams GuiParams::load()
{
    GuiParams p;
    QSettings s;
    p.imageDir         = readString(s, "input/imageDir", p.imageDir);
    p.leftPrefix       = readString(s, "input/leftPrefix", p.leftPrefix);
    p.rightPrefix      = readString(s, "input/rightPrefix", p.rightPrefix);
    p.extension        = readString(s, "input/extension", p.extension);
    p.reference        = readString(s, "input/reference", p.reference);
    p.boardWidth       = s.value("board/width", p.boardWidth).toInt();
    p.boardHeight      = s.value("board/height", p.boardHeight).toInt();
    p.squareMm         = s.value("board/squareMm", p.squareMm).toDouble();
    p.useSbDetector    = s.value("board/useSb", p.useSbDetector).toBool();
    p.alpha            = s.value("rectify/alpha", p.alpha).toDouble();
    p.refineIntrinsics = s.value("rectify/refineIntrinsics", p.refineIntrinsics).toBool();
    p.sameFocalLength  = s.value("rectify/sameFocalLength", p.sameFocalLength).toBool();
    p.outputDir        = readString(s, "output/dir", p.outputDir);
    p.saveDebugImages  = s.value("output/debugImages", p.saveDebugImages).toBool();
    return p;
}

void GuiParams::save() const
{
    QSettings s;
    s.setValue("input/imageDir", imageDir);
    s.setValue("input/leftPrefix", leftPrefix);
    s.setValue("input/rightPrefix", rightPrefix);
    s.setValue("input/extension", extension);
    s.setValue("input/reference", reference);
    s.setValue("board/width", boardWidth);
    s.setValue("board/height", boardHeight);
    s.setValue("board/squareMm", squareMm);
    s.setValue("board/useSb", useSbDetector);
    s.setValue("rectify/alpha", alpha);
    s.setValue("rectify/refineIntrinsics", refineIntrinsics);
    s.setValue("rectify/sameFocalLength", sameFocalLength);
    s.setValue("output/dir", outputDir);
    s.setValue("output/debugImages", saveDebugImages);
}
