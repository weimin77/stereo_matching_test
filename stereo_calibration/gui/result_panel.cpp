#include "result_panel.hpp"

#include <QColor>
#include <QComboBox>
#include <QFontDatabase>
#include <QHeaderView>
#include <QItemSelectionModel>
#include <QLabel>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QSplitter>
#include <QTableWidget>
#include <QTabWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>

#include <algorithm>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "image_view.hpp"

namespace {

QTableWidgetItem* cell(const QString& text)
{
    auto* item = new QTableWidgetItem(text);
    item->setFlags(item->flags() & ~Qt::ItemIsEditable);
    return item;
}

/// "a | b" with a caption strip above each half. Used for the before/after view
/// so the pair is comparable without two windows.
cv::Mat sideBySide(const cv::Mat& left, const cv::Mat& right,
                   const QString& leftCaption, const QString& rightCaption)
{
    const int height = std::max(left.rows, right.rows);
    const int width = left.cols + right.cols;
    cv::Mat canvas(height + 22, width, CV_8UC3, cv::Scalar(28, 28, 28));
    left.copyTo(canvas(cv::Rect(0, 22, left.cols, left.rows)));
    right.copyTo(canvas(cv::Rect(left.cols, 22, right.cols, right.rows)));
    cv::putText(canvas, leftCaption.toStdString(), cv::Point(6, 16),
                cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(220, 220, 220), 1, cv::LINE_AA);
    cv::putText(canvas, rightCaption.toStdString(), cv::Point(left.cols + 6, 16),
                cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(220, 220, 220), 1, cv::LINE_AA);
    return canvas;
}

QString number(double value, int decimals)
{
    return QString::number(value, 'f', decimals);
}

}  // namespace

ResultPanel::ResultPanel(QWidget* parent)
    : QWidget(parent)
{
    auto* tabs = new QTabWidget(this);

    // --- Summary -----------------------------------------------------------
    summary_ = new QTableWidget(0, 2, this);
    summary_->setHorizontalHeaderLabels({QStringLiteral("Metric"), QStringLiteral("Value")});
    summary_->horizontalHeader()->setStretchLastSection(true);
    summary_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    summary_->verticalHeader()->setVisible(false);
    summary_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    summary_->setSelectionMode(QAbstractItemView::NoSelection);
    summary_->setAlternatingRowColors(true);
    tabs->addTab(summary_, QStringLiteral("Summary"));

    // --- Matrices ----------------------------------------------------------
    auto* matrixPage = new QWidget(this);
    auto* matrixLayout = new QVBoxLayout(matrixPage);
    matrixLayout->setContentsMargins(4, 4, 4, 4);
    matrixSelect_ = new QComboBox(matrixPage);
    matrixText_ = new QPlainTextEdit(matrixPage);
    matrixText_->setReadOnly(true);
    matrixText_->setFont(QFontDatabase::systemFont(QFontDatabase::FixedFont));
    matrixText_->setLineWrapMode(QPlainTextEdit::NoWrap);
    matrixLayout->addWidget(matrixSelect_);
    matrixLayout->addWidget(matrixText_, 1);
    connect(matrixSelect_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int) {
        if (!matrixSelect_->currentIndex()) matrixText_->clear();
        const int index = matrixSelect_->currentIndex() - 1;
        matrixText_->setPlainText(index >= 0 && index < static_cast<int>(matrixValues_.size())
                                      ? QString::fromStdString(
                                            calib::matrixToString(matrixValues_[index]))
                                      : QString());
    });
    tabs->addTab(matrixPage, QStringLiteral("Matrices"));

    // --- Per-view errors ---------------------------------------------------
    auto* perViewPage = new QWidget(this);
    auto* perViewLayout = new QVBoxLayout(perViewPage);
    perViewLayout->setContentsMargins(4, 4, 4, 4);
    perView_ = new QTableWidget(0, 3, perViewPage);
    perView_->setHorizontalHeaderLabels(
        {QStringLiteral("View"), QStringLiteral("Left (px)"), QStringLiteral("Right (px)")});
    perView_->horizontalHeader()->setStretchLastSection(true);
    perView_->verticalHeader()->setVisible(false);
    perView_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    perView_->setSelectionBehavior(QAbstractItemView::SelectRows);
    perView_->setSelectionMode(QAbstractItemView::ExtendedSelection);
    perView_->setAlternatingRowColors(true);
    auto* perViewFooter = new QHBoxLayout();
    auto* hint = new QLabel(
        QStringLiteral("A view above 1.0 px is the usual cause of a mediocre calibration. "
                       "Select rows and re-run without them to compare."),
        perViewPage);
    hint->setWordWrap(true);
    excludeButton_ = new QPushButton(QStringLiteral("Exclude selected and re-run"), perViewPage);
    excludeButton_->setEnabled(false);
    perViewFooter->addWidget(hint, 1);
    perViewFooter->addWidget(excludeButton_);
    perViewLayout->addWidget(perView_, 1);
    perViewLayout->addLayout(perViewFooter);
    connect(perView_->selectionModel(), &QItemSelectionModel::selectionChanged, this,
            [this](const QItemSelection&, const QItemSelection&) { updateExcludeButton(); });
    connect(excludeButton_, &QPushButton::clicked, this, [this]() {
        QStringList keys;
        const auto rows = perView_->selectionModel()->selectedRows();
        for (const QModelIndex& index : rows) {
            if (auto* item = perView_->item(index.row(), 0))
                keys << item->text();
        }
        if (!keys.isEmpty()) emit rerunWithoutViews(keys);
    });
    tabs->addTab(perViewPage, QStringLiteral("Per-view"));

    // --- Images ------------------------------------------------------------
    auto* imagePage = new QWidget(this);
    auto* imageLayout = new QVBoxLayout(imagePage);
    imageLayout->setContentsMargins(4, 4, 4, 4);
    imageSelect_ = new QComboBox(imagePage);
    imageView_ = new ImageView(imagePage);
    imageLayout->addWidget(imageSelect_);
    imageLayout->addWidget(imageView_, 1);
    connect(imageSelect_, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            [this](int) { updateImage(); });
    tabs->addTab(imagePage, QStringLiteral("Images"));

    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->addWidget(tabs);

    matrixSelect_->addItem(QStringLiteral("(select a matrix)"));
}

void ResultPanel::clearResult()
{
    result_.reset();
    summary_->setRowCount(0);
    matrixSelect_->blockSignals(true);
    matrixSelect_->clear();
    matrixSelect_->addItem(QStringLiteral("(select a matrix)"));
    matrixSelect_->blockSignals(false);
    matrixText_->clear();
    perView_->setRowCount(0);
    imageSelect_->clear();
    imageView_->clearImage();
    matrixNames_.clear();
    matrixValues_.clear();
    imageNames_.clear();
    imageValues_.clear();
    updateExcludeButton();
}

void ResultPanel::setResult(const std::shared_ptr<calib::RunResult>& result)
{
    result_ = result;
    if (!result_) {
        clearResult();
        return;
    }
    fillSummary(*result);
    fillMatrices(*result);
    fillPerView(*result);
    fillImages(*result);
}

void ResultPanel::fillSummary(const calib::RunResult& r)
{
    struct Row { QString name; QString value; };
    const std::vector<Row> rows = {
        {QStringLiteral("Views used"), QString::number(r.pairs.size())},
        {QStringLiteral("Image size"),
         QStringLiteral("%1 x %2").arg(r.imageSize.width).arg(r.imageSize.height)},
        {QStringLiteral("Mono RMS left"), number(r.camera1.rms, 4) + QStringLiteral(" px")},
        {QStringLiteral("Mono RMS right"), number(r.camera2.rms, 4) + QStringLiteral(" px")},
        {QStringLiteral("Stereo RMS"), number(r.stereo.rms, 4) + QStringLiteral(" px")},
        {QStringLiteral("Baseline |T|"), number(r.stereo.baseline(), 6) + QStringLiteral(" m")},
        {QStringLiteral("R deviation from identity"),
         number(cv::norm(r.stereo.R - cv::Mat::eye(3, 3, CV_64F)), 6)},
        {QStringLiteral("Epipolar error mean"),
         number(r.verification.epipolarMean, 4) + QStringLiteral(" px")},
        {QStringLiteral("Epipolar error P95"),
         number(r.verification.epipolarP95, 4) + QStringLiteral(" px")},
        {QStringLiteral("Epipolar error max"),
         number(r.verification.epipolarMax, 4) + QStringLiteral(" px")},
        {QStringLiteral("Corners compared"), QString::number(r.verification.comparedCorners)},
    };
    std::vector<Row> all = rows;
    if (r.verification.triangulated) {
        all.push_back({QStringLiteral("Planarity RMS (mean)"),
                       number(r.verification.planarityRms * 1000.0, 4) + QStringLiteral(" mm")});
        all.push_back({QStringLiteral("Planarity RMS (worst)"),
                       number(r.verification.planarityWorst * 1000.0, 4) + QStringLiteral(" mm")});
        all.push_back({QStringLiteral("Reconstructed square edge"),
                       number(r.verification.meanEdgeLength * 1000.0, 4) + QStringLiteral(" mm")});
        all.push_back({QStringLiteral("Edge length error"),
                       number(r.verification.edgeLengthError * 100.0, 3) + QStringLiteral(" %")});
        all.push_back({QStringLiteral("Mean board depth"),
                       number(r.verification.meanDepth, 4) + QStringLiteral(" m")});
    }
    all.push_back({QStringLiteral("Epipolar verdict"),
                   r.epipolarOk ? QStringLiteral("pass (mean < 0.5 px)") : QStringLiteral("too large")});
    all.push_back({QStringLiteral("Edge length verdict"),
                   r.edgeOk ? QStringLiteral("consistent (< 1 %)")
                            : QStringLiteral("inconsistent, check P vs rectification")});
    all.push_back({QStringLiteral("Output"), QString::fromStdString(r.outputDirectory)});

    summary_->setRowCount(static_cast<int>(all.size()));
    for (int i = 0; i < static_cast<int>(all.size()); ++i) {
        summary_->setItem(i, 0, cell(all[i].name));
        summary_->setItem(i, 1, cell(all[i].value));
        if (all[i].name == QLatin1String("Epipolar verdict") && !r.epipolarOk)
            summary_->item(i, 1)->setForeground(QColor(190, 30, 30));
        if (all[i].name == QLatin1String("Edge length verdict") && !r.edgeOk)
            summary_->item(i, 1)->setForeground(QColor(190, 30, 30));
    }
    summary_->resizeRowsToContents();
}

void ResultPanel::fillMatrices(const calib::RunResult& r)
{
    matrixNames_ = {QStringLiteral("K1  (left camera matrix)"),
                    QStringLiteral("D1  (left distortion)"),
                    QStringLiteral("K2  (right camera matrix)"),
                    QStringLiteral("D2  (right distortion)"),
                    QStringLiteral("R   (rotation)"),
                    QStringLiteral("T   (translation, metres)"),
                    QStringLiteral("E   (essential)"),
                    QStringLiteral("F   (fundamental)"),
                    QStringLiteral("R1  (left rectification)"),
                    QStringLiteral("R2  (right rectification)"),
                    QStringLiteral("P1  (left projection)"),
                    QStringLiteral("P2  (right projection)"),
                    QStringLiteral("Q   (disparity-to-depth)")};
    matrixValues_ = {r.camera1.K, r.camera1.D, r.camera2.K, r.camera2.D,
                     r.stereo.R, r.stereo.T, r.stereo.E, r.stereo.F,
                     r.stereo.R1, r.stereo.R2, r.stereo.P1, r.stereo.P2, r.stereo.Q};

    matrixSelect_->blockSignals(true);
    matrixSelect_->clear();
    matrixSelect_->addItem(QStringLiteral("(select a matrix)"));
    for (const QString& name : matrixNames_) matrixSelect_->addItem(name);
    matrixSelect_->setCurrentIndex(0);
    matrixSelect_->blockSignals(false);
    matrixText_->clear();
}

void ResultPanel::fillPerView(const calib::RunResult& r)
{
    const int rows = r.stereo.perViewErrors.rows;
    perView_->setRowCount(rows);
    for (int i = 0; i < rows; ++i) {
        const double left = r.stereo.perViewErrors.at<double>(i, 0);
        const double right = r.stereo.perViewErrors.at<double>(i, 1);
        const QString key = i < static_cast<int>(r.pairs.size())
                                ? QString::fromStdString(r.pairs[static_cast<std::size_t>(i)].key)
                                : QString::number(i);
        perView_->setItem(i, 0, cell(key));
        perView_->setItem(i, 1, cell(number(left, 4)));
        perView_->setItem(i, 2, cell(number(right, 4)));
        if (std::max(left, right) > 1.0) {
            const QColor bad(255, 226, 226);
            for (int c = 0; c < 3; ++c) perView_->item(i, c)->setBackground(bad);
        }
    }
    perView_->resizeColumnsToContents();
    perView_->clearSelection();
    updateExcludeButton();
}

void ResultPanel::fillImages(const calib::RunResult& r)
{
    imageNames_.clear();
    imageValues_.clear();
    if (!r.rectLeft.empty()) {
        imageNames_.push_back(QStringLiteral("Rectified left"));
        imageValues_.push_back(r.rectLeft);
    }
    if (!r.rectRight.empty()) {
        imageNames_.push_back(QStringLiteral("Rectified right"));
        imageValues_.push_back(r.rectRight);
    }
    if (!r.preview.empty()) {
        imageNames_.push_back(QStringLiteral("Epipolar check (left | right)"));
        imageValues_.push_back(r.preview);
    }
    if (!r.rectLeft.empty() && !r.pairs.empty()) {
        const cv::Mat original = cv::imread(r.pairs.front().left.string(), cv::IMREAD_COLOR);
        if (!original.empty()) {
            imageNames_.push_back(QStringLiteral("Left: original vs rectified"));
            imageValues_.push_back(sideBySide(original, r.rectLeft, QStringLiteral("original"),
                                              QStringLiteral("rectified")));
        }
    }

    imageSelect_->blockSignals(true);
    imageSelect_->clear();
    for (const QString& name : imageNames_) imageSelect_->addItem(name);
    imageSelect_->blockSignals(false);
    updateImage();
}

void ResultPanel::updateImage()
{
    const int index = imageSelect_->currentIndex();
    if (index < 0 || index >= static_cast<int>(imageValues_.size())) {
        imageView_->clearImage();
        return;
    }
    imageView_->setMat(imageValues_[static_cast<std::size_t>(index)]);
}

void ResultPanel::updateExcludeButton()
{
    const bool hasRows = perView_->rowCount() > 0;
    const bool hasSelection = perView_->selectionModel() &&
                              !perView_->selectionModel()->selectedRows().isEmpty();
    excludeButton_->setEnabled(result_ != nullptr && hasRows && hasSelection);
}
