#ifndef STEREO_CALIB_GUI_FONTS_HPP
#define STEREO_CALIB_GUI_FONTS_HPP

#include <QFont>

/// The pipeline log is Chinese (it is the same text the CLI prints), but Qt does
/// not use application fonts for glyph fallback in Qt 5, and a bare WSL install
/// ships no CJK font at all — every log line would render as boxes. If the
/// default font cannot draw 中, this loads a font that can and installs it as
/// the application font.
void ensureCjkFont();

/// Font for the log view: monospaced when the default fixed font covers CJK,
/// otherwise the CJK-capable family found by ensureCjkFont().
QFont logFont();

#endif  // STEREO_CALIB_GUI_FONTS_HPP
