#include "fonts.hpp"

#include <QApplication>
#include <QFileInfo>
#include <QFontDatabase>
#include <QFontMetrics>
#include <QStringList>

#include <cstdio>

namespace {

QString g_cjkFamily;

bool canDrawChinese(const QFont& font)
{
    return QFontMetrics(font).inFont(QChar(0x4E2D));
}

}  // namespace

void ensureCjkFont()
{
    if (canDrawChinese(QApplication::font())) return;

    const QStringList candidates = {
        QStringLiteral("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        QStringLiteral("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        QStringLiteral("/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf"),
        QStringLiteral("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
        QStringLiteral("/usr/share/fonts/truetype/arphic/uming.ttc"),
        QStringLiteral("/mnt/c/Windows/Fonts/msyh.ttc"),
        QStringLiteral("/mnt/c/Windows/Fonts/simhei.ttf"),
    };

    for (const QString& path : candidates) {
        if (!QFileInfo::exists(path)) continue;
        const int id = QFontDatabase::addApplicationFont(path);
        if (id < 0) continue;
        const QStringList families = QFontDatabase::applicationFontFamilies(id);
        if (families.isEmpty()) continue;

        QFont font(families.front());
        const QFont current = QApplication::font();
        if (current.pointSizeF() > 0) font.setPointSizeF(current.pointSizeF());
        font.setStyleHint(QFont::SansSerif);
        if (!canDrawChinese(font)) continue;

        QApplication::setFont(font);
        g_cjkFamily = families.front();
        return;
    }

    std::fprintf(stderr,
                 "warning: no CJK font found; the pipeline log is Chinese and will render "
                 "as boxes. Install fonts-noto-cjk (Debian/Ubuntu) to fix it.\n");
}

QFont logFont()
{
    if (!g_cjkFamily.isEmpty()) {
        QFont font(g_cjkFamily);
        font.setStyleHint(QFont::Monospace);
        return font;
    }
    return QFontDatabase::systemFont(QFontDatabase::FixedFont);
}
