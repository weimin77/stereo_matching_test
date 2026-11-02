#include "main_window.hpp"

#include <QAction>
#include <QApplication>
#include <QCheckBox>
#include <QColor>
#include <QCloseEvent>
#include <QComboBox>
#include <QDesktopServices>
#include <QDir>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFontDatabase>
#include <QFormLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QHash>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMenuBar>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QProgressBar>
#include <QPushButton>
#include <QScrollArea>
#include <QSettings>
#include <QSpinBox>
#include <QSplitter>
#include <QStatusBar>
#include <QTabWidget>
#include <QUrl>
#include <QVBoxLayout>

#include <opencv2/imgcodecs.hpp>

#include <algorithm>

#include "gui/fonts.hpp"
#include "gui/image_view.hpp"
#include "gui/params.hpp"
#include "gui/result_panel.hpp"

namespace {

QWidget* pathRow(QLineEdit* edit, QPushButton* button)
{
    auto* row = new QWidget();
    auto* layout = new QHBoxLayout(row);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(4);
    layout->addWidget(edit, 1);
    layout->addWidget(button);
    return row;
}

QPlainTextEdit* makeLogView(QWidget* parent)
{
    auto* view = new QPlainTextEdit(parent);
    view->setReadOnly(true);
    view->setFont(logFont());
    view->setMaximumBlockCount(5000);
    view->setLineWrapMode(QPlainTextEdit::NoWrap);
    return view;
}

}  // namespace

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
{
    setWindowTitle(tr("Stereo calibration"));
    buildUi();
    loadParams();

    QSettings settings;
    const QByteArray geometry = settings.value("window/geometry").toByteArray();
    if (!geometry.isEmpty()) restoreGeometry(geometry);
    else                     resize(1280, 820);

    const QByteArray splitter = settings.value("window/splitter").toByteArray();
    if (!splitter.isEmpty()) {
        if (auto* split = findChild<QSplitter*>(QStringLiteral("mainSplitter")))
            split->restoreState(splitter);
    }
    statusBar()->showMessage(
        tr("Choose an image directory, then Detect corners or Run calibration."));
}

MainWindow::~MainWindow()
{
    if (worker_) {
        worker_->requestCancel();
        worker_->wait();
        delete worker_;
        worker_ = nullptr;
    }
}

void MainWindow::buildUi()
{
    auto* splitter = new QSplitter(Qt::Horizontal, this);
    splitter->setObjectName(QStringLiteral("mainSplitter"));

    parameterPanel_ = buildParameterPanel();
    auto* scroll = new QScrollArea(this);
    scroll->setWidget(parameterPanel_);
    scroll->setWidgetResizable(true);
    scroll->setMinimumWidth(330);
    scroll->setMaximumWidth(430);
    splitter->addWidget(scroll);

    auto* rightSplitter = new QSplitter(Qt::Vertical, this);

    auto* pairGroup = new QGroupBox(tr("Image pairs"), this);
    auto* pairLayout = new QHBoxLayout(pairGroup);
    pairList_ = new QListWidget(pairGroup);
    pairList_->setMaximumWidth(220);
    pairList_->setMinimumWidth(150);
    pairLayout->addWidget(pairList_);

    auto* previewPanel = new QWidget(pairGroup);
    auto* previewLayout = new QVBoxLayout(previewPanel);
    previewLayout->setContentsMargins(0, 0, 0, 0);
    auto* previewControls = new QHBoxLayout();
    previewControls->addWidget(new QLabel(tr("Eye:"), previewPanel));
    eyeSelect_ = new QComboBox(previewPanel);
    eyeSelect_->addItems({tr("Left"), tr("Right")});
    showCorners_ = new QCheckBox(tr("Show corners"), previewPanel);
    showCorners_->setChecked(true);
    showIds_ = new QCheckBox(tr("Show indices"), previewPanel);
    previewControls->addWidget(eyeSelect_);
    previewControls->addWidget(showCorners_);
    previewControls->addWidget(showIds_);
    previewControls->addStretch(1);
    previewLayout->addLayout(previewControls);
    preview_ = new ImageView(previewPanel);
    previewLayout->addWidget(preview_, 1);
    pairLayout->addWidget(previewPanel, 1);
    rightSplitter->addWidget(pairGroup);

    tabs_ = new QTabWidget(this);
    log_ = makeLogView(tabs_);
    tabs_->addTab(log_, tr("Log"));
    results_ = new ResultPanel(tabs_);
    tabs_->addTab(results_, tr("Results"));
    rightSplitter->addWidget(tabs_);
    rightSplitter->setStretchFactor(0, 3);
    rightSplitter->setStretchFactor(1, 2);

    splitter->addWidget(rightSplitter);
    splitter->setStretchFactor(0, 0);
    splitter->setStretchFactor(1, 1);
    setCentralWidget(splitter);

    progress_ = new QProgressBar(this);
    progress_->setMaximumWidth(220);
    progress_->setTextVisible(true);
    progress_->setRange(0, 1);
    progress_->setValue(0);
    statusBar()->addPermanentWidget(progress_);

    // --- menus -------------------------------------------------------------
    auto* fileMenu = menuBar()->addMenu(tr("&File"));
    auto* openAction = fileMenu->addAction(tr("Open image folder..."));
    connect(openAction, &QAction::triggered, this, [this]() {
        const QString dir = QFileDialog::getExistingDirectory(this, tr("Image directory"),
                                                              imageDir_->text());
        if (!dir.isEmpty()) imageDir_->setText(dir);
    });
    auto* outputAction = fileMenu->addAction(tr("Choose output folder..."));
    connect(outputAction, &QAction::triggered, this, [this]() {
        const QString dir = QFileDialog::getExistingDirectory(this, tr("Output directory"),
                                                              outputDir_->text());
        if (!dir.isEmpty()) outputDir_->setText(dir);
    });
    fileMenu->addSeparator();
    auto* quitAction = fileMenu->addAction(tr("Quit"));
    connect(quitAction, &QAction::triggered, this, &QWidget::close);

    // --- connections -------------------------------------------------------
    connect(pairList_, &QListWidget::currentItemChanged, this,
            [this](QListWidgetItem*, QListWidgetItem*) { updatePreview(); });
    connect(eyeSelect_, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            [this](int) { updatePreview(); });
    connect(showCorners_, &QCheckBox::toggled, this, [this](bool) { updatePreview(); });
    connect(showIds_, &QCheckBox::toggled, this, [this](bool) { updatePreview(); });
    connect(results_, &ResultPanel::rerunWithoutViews, this, [this](const QStringList& keys) {
        if (busy_) return;
        for (const QString& key : keys) excluded_.insert(key);
        appendLog(tr("Excluding views: %1").arg(keys.join(QStringLiteral(", "))));
        updateExclusionLabel();
        startJob(CalibWorker::Job::Calibrate);
    });
}

QWidget* MainWindow::buildParameterPanel()
{
    auto* panel = new QWidget(this);
    auto* layout = new QVBoxLayout(panel);
    layout->setContentsMargins(0, 0, 4, 0);
    layout->setSpacing(8);

    // --- Input -------------------------------------------------------------
    auto* inputGroup = new QGroupBox(tr("Input"), panel);
    auto* inputForm = new QFormLayout(inputGroup);
    imageDir_ = new QLineEdit(inputGroup);
    auto* imageBrowse = new QPushButton(tr("Browse"), inputGroup);
    connect(imageBrowse, &QPushButton::clicked, this, [this]() {
        const QString dir = QFileDialog::getExistingDirectory(this, tr("Image directory"),
                                                              imageDir_->text());
        if (!dir.isEmpty()) imageDir_->setText(dir);
    });
    inputForm->addRow(tr("Image directory"), pathRow(imageDir_, imageBrowse));

    leftPrefix_ = new QLineEdit(inputGroup);
    rightPrefix_ = new QLineEdit(inputGroup);
    extension_ = new QLineEdit(inputGroup);
    inputForm->addRow(tr("Left prefix"), leftPrefix_);
    inputForm->addRow(tr("Right prefix"), rightPrefix_);
    inputForm->addRow(tr("Extension"), extension_);

    reference_ = new QLineEdit(inputGroup);
    auto* referenceBrowse = new QPushButton(tr("Browse"), inputGroup);
    connect(referenceBrowse, &QPushButton::clicked, this, [this]() {
        const QString file = QFileDialog::getOpenFileName(
            this, tr("Reference intrinsics"), reference_->text(),
            tr("OpenCV YAML (*.yml *.yaml);;All files (*)"));
        if (!file.isEmpty()) reference_->setText(file);
    });
    inputForm->addRow(tr("Reference yml"), pathRow(reference_, referenceBrowse));
    layout->addWidget(inputGroup);

    // --- Board -------------------------------------------------------------
    auto* boardGroup = new QGroupBox(tr("Chessboard"), panel);
    auto* boardForm = new QFormLayout(boardGroup);
    boardWidth_ = new QSpinBox(boardGroup);
    boardWidth_->setRange(3, 30);
    boardHeight_ = new QSpinBox(boardGroup);
    boardHeight_->setRange(3, 30);
    auto* sizeRow = new QWidget(boardGroup);
    auto* sizeLayout = new QHBoxLayout(sizeRow);
    sizeLayout->setContentsMargins(0, 0, 0, 0);
    sizeLayout->addWidget(boardWidth_);
    sizeLayout->addWidget(new QLabel(QStringLiteral("x"), sizeRow));
    sizeLayout->addWidget(boardHeight_);
    sizeLayout->addStretch(1);
    boardForm->addRow(tr("Inner corners"), sizeRow);

    squareMm_ = new QDoubleSpinBox(boardGroup);
    squareMm_->setRange(0.1, 10000.0);
    squareMm_->setDecimals(3);
    squareMm_->setSingleStep(1.0);
    squareMm_->setSuffix(tr(" mm"));
    boardForm->addRow(tr("Square size"), squareMm_);

    useSb_ = new QCheckBox(tr("findChessboardCornersSB"), boardGroup);
    useSb_->setToolTip(tr("Typically lower RMS, but its corner convention differs from the "
                          "classic detector: on the OpenCV sample data fx drifts by -0.67 %."));
    boardForm->addRow(QString(), useSb_);
    layout->addWidget(boardGroup);

    // --- Rectification ------------------------------------------------------
    auto* rectifyGroup = new QGroupBox(tr("Rectification"), panel);
    auto* rectifyForm = new QFormLayout(rectifyGroup);
    alpha_ = new QDoubleSpinBox(rectifyGroup);
    alpha_->setRange(0.0, 1.0);
    alpha_->setDecimals(2);
    alpha_->setSingleStep(0.05);
    alpha_->setToolTip(tr("0 keeps only valid pixels, 1 keeps everything."));
    rectifyForm->addRow(tr("alpha"), alpha_);
    refineIntrinsics_ = new QCheckBox(tr("Refine intrinsics jointly"), rectifyGroup);
    sameFocalLength_ = new QCheckBox(tr("Force fx == fy"), rectifyGroup);
    rectifyForm->addRow(refineIntrinsics_);
    rectifyForm->addRow(sameFocalLength_);
    layout->addWidget(rectifyGroup);

    // --- Output -------------------------------------------------------------
    auto* outputGroup = new QGroupBox(tr("Output"), panel);
    auto* outputForm = new QFormLayout(outputGroup);
    outputDir_ = new QLineEdit(outputGroup);
    auto* outputBrowse = new QPushButton(tr("Browse"), outputGroup);
    connect(outputBrowse, &QPushButton::clicked, this, [this]() {
        const QString dir = QFileDialog::getExistingDirectory(this, tr("Output directory"),
                                                              outputDir_->text());
        if (!dir.isEmpty()) outputDir_->setText(dir);
    });
    outputForm->addRow(tr("Directory"), pathRow(outputDir_, outputBrowse));
    saveDebugImages_ = new QCheckBox(tr("Write rectified previews"), outputGroup);
    outputForm->addRow(saveDebugImages_);

    exclusionLabel_ = new QLabel(outputGroup);
    exclusionLabel_->setWordWrap(true);
    clearExclusionsButton_ = new QPushButton(tr("Clear"), outputGroup);
    clearExclusionsButton_->setEnabled(false);
    auto* exclusionRow = new QWidget(outputGroup);
    auto* exclusionLayout = new QHBoxLayout(exclusionRow);
    exclusionLayout->setContentsMargins(0, 0, 0, 0);
    exclusionLayout->addWidget(exclusionLabel_, 1);
    exclusionLayout->addWidget(clearExclusionsButton_);
    outputForm->addRow(tr("Excluded"), exclusionRow);
    connect(clearExclusionsButton_, &QPushButton::clicked, this, &MainWindow::clearExclusions);
    layout->addWidget(outputGroup);

    // --- Actions ------------------------------------------------------------
    detectButton_ = new QPushButton(tr("Detect corners"), panel);
    runButton_ = new QPushButton(tr("Run calibration"), panel);
    cancelButton_ = new QPushButton(tr("Cancel"), panel);
    openOutputButton_ = new QPushButton(tr("Open output folder"), panel);
    cancelButton_->setEnabled(false);
    runButton_->setDefault(true);
    connect(detectButton_, &QPushButton::clicked, this,
            [this]() { startJob(CalibWorker::Job::Detect); });
    connect(runButton_, &QPushButton::clicked, this,
            [this]() { startJob(CalibWorker::Job::Calibrate); });
    connect(cancelButton_, &QPushButton::clicked, this, [this]() {
        if (!worker_) return;
        worker_->requestCancel();
        cancelButton_->setEnabled(false);
        statusBar()->showMessage(tr("Cancelling..."));
    });
    connect(openOutputButton_, &QPushButton::clicked, this, &MainWindow::openOutputDir);

    layout->addWidget(detectButton_);
    layout->addWidget(runButton_);
    layout->addWidget(cancelButton_);
    layout->addWidget(openOutputButton_);
    layout->addStretch(1);
    return panel;
}

void MainWindow::loadParams()
{
    const GuiParams p = GuiParams::load();
    imageDir_->setText(p.imageDir);
    leftPrefix_->setText(p.leftPrefix);
    rightPrefix_->setText(p.rightPrefix);
    extension_->setText(p.extension);
    reference_->setText(p.reference);
    boardWidth_->setValue(p.boardWidth);
    boardHeight_->setValue(p.boardHeight);
    squareMm_->setValue(p.squareMm);
    useSb_->setChecked(p.useSbDetector);
    alpha_->setValue(p.alpha);
    refineIntrinsics_->setChecked(p.refineIntrinsics);
    sameFocalLength_->setChecked(p.sameFocalLength);
    outputDir_->setText(p.outputDir);
    saveDebugImages_->setChecked(p.saveDebugImages);
    updateExclusionLabel();
}

GuiParams MainWindow::collectParams() const
{
    GuiParams p;
    p.imageDir = imageDir_->text();
    p.leftPrefix = leftPrefix_->text();
    p.rightPrefix = rightPrefix_->text();
    p.extension = extension_->text();
    p.reference = reference_->text();
    p.boardWidth = boardWidth_->value();
    p.boardHeight = boardHeight_->value();
    p.squareMm = squareMm_->value();
    p.useSbDetector = useSb_->isChecked();
    p.alpha = alpha_->value();
    p.refineIntrinsics = refineIntrinsics_->isChecked();
    p.sameFocalLength = sameFocalLength_->isChecked();
    p.outputDir = outputDir_->text();
    p.saveDebugImages = saveDebugImages_->isChecked();
    return p;
}

void MainWindow::saveParams()
{
    collectParams().save();
    QSettings settings;
    settings.setValue("window/geometry", saveGeometry());
    if (auto* split = findChild<QSplitter*>(QStringLiteral("mainSplitter")))
        settings.setValue("window/splitter", split->saveState());
}

void MainWindow::applyTestParams(const QString& images, const QString& output, int boardWidth,
                                 int boardHeight, double squareMm)
{
    testMode_ = true;
    imageDir_->setText(images);
    if (!output.isEmpty()) outputDir_->setText(output);
    if (boardWidth > 0) boardWidth_->setValue(boardWidth);
    if (boardHeight > 0) boardHeight_->setValue(boardHeight);
    if (squareMm > 0.0) squareMm_->setValue(squareMm);
}

bool MainWindow::startCalibration()
{
    if (busy_) return false;
    startJob(CalibWorker::Job::Calibrate);
    return true;
}

bool MainWindow::startDetection()
{
    if (busy_) return false;
    startJob(CalibWorker::Job::Detect);
    return true;
}

void MainWindow::setExcludedViews(const QStringList& keys)
{
    for (const QString& key : keys) {
        const QString trimmed = key.trimmed();
        if (!trimmed.isEmpty()) excluded_.insert(trimmed);
    }
    updateExclusionLabel();
}

void MainWindow::showResultsTab()
{
    tabs_->setCurrentWidget(results_);
}

void MainWindow::startJob(CalibWorker::Job job)
{
    if (busy_) return;

    const calib::Settings settings = collectParams().toSettings();
    if (job == CalibWorker::Job::Calibrate && settings.outputDirectory.empty()) {
        QMessageBox::warning(this, tr("Missing output directory"),
                             tr("Choose an output directory before running the calibration."));
        return;
    }

    calib::Settings effective = settings;
    effective.excludedKeys.clear();
    for (const QString& key : excluded_) effective.excludedKeys.push_back(key.toStdString());

    busy_ = true;
    errorMessage_.clear();
    setBusy(true);
    log_->clear();
    if (job == CalibWorker::Job::Calibrate) results_->clearResult();
    progress_->setRange(0, 0);
    statusBar()->showMessage(job == CalibWorker::Job::Detect ? tr("Detecting corners...")
                                                            : tr("Calibrating..."));
    appendLog(job == CalibWorker::Job::Detect ? tr("--- detect corners ---")
                                              : tr("--- calibration ---"));

    worker_ = new CalibWorker(effective, job, this);
    connect(worker_, &CalibWorker::logLine, this, &MainWindow::appendLog);
    connect(worker_, &CalibWorker::progress, this,
            [this](const QString& stage, int done, int total) {
                progress_->setRange(0, std::max(total, 1));
                progress_->setValue(done);
                if (stage == QLatin1String("detect"))
                    statusBar()->showMessage(tr("Detecting corners %1/%2").arg(done).arg(total));
                else
                    statusBar()->showMessage(tr("Calibrating %1/%2").arg(done).arg(total));
            });
    connect(worker_, &CalibWorker::failed, this, [this](const QString& message) {
        errorMessage_ = message;
        appendLog(tr("ERROR: %1").arg(message));
    });
    connect(worker_, &QThread::finished, this, &MainWindow::onJobFinished);
    worker_->start();
}

void MainWindow::onJobFinished()
{
    if (!worker_) return;
    worker_->wait();

    const bool cancelled = worker_->wasCancelled();
    const auto detected = worker_->detectResult();
    const auto runResult = worker_->runResult();
    worker_->deleteLater();
    worker_ = nullptr;

    busy_ = false;
    setBusy(false);
    progress_->setRange(0, 1);
    progress_->setValue(1);

    if (cancelled) {
        appendLog(tr("--- cancelled by user ---"));
        statusBar()->showMessage(tr("Cancelled."));
        emit jobFinished(false);
        return;
    }
    if (!errorMessage_.isEmpty()) {
        statusBar()->showMessage(tr("Failed: %1").arg(errorMessage_));
        // A modal dialog would block --selftest on a headless machine.
        if (!testMode_) QMessageBox::critical(this, tr("Calibration failed"), errorMessage_);
        emit jobFinished(false);
        return;
    }
    if (detected) {
        detected_ = detected;
        const int ok = static_cast<int>(detected_->kept.size());
        const int total = static_cast<int>(detected_->pairs.size());
        appendLog(tr("--- %1 of %2 pairs ready ---").arg(ok).arg(total));
        refreshPairList();
        statusBar()->showMessage(
            tr("Corner detection: %1 of %2 pairs usable.").arg(ok).arg(total));
    }
    if (runResult) {
        result_ = runResult;
        results_->setResult(result_);
        statusBar()->showMessage(
            tr("Calibration finished: stereo RMS %1 px, baseline %2 m.")
                .arg(result_->stereo.rms, 0, 'f', 4)
                .arg(result_->stereo.baseline(), 0, 'f', 6));
        appendLog(tr("--- done: results written to %1 ---")
                      .arg(QString::fromStdString(result_->outputDirectory)));
    }
    emit jobFinished(true);
}

void MainWindow::setBusy(bool busy)
{
    parameterPanel_->setEnabled(!busy);
    detectButton_->setEnabled(!busy);
    runButton_->setEnabled(!busy);
    cancelButton_->setEnabled(busy);
    openOutputButton_->setEnabled(!busy);
    if (busy) {
        results_->setEnabled(false);
    } else {
        results_->setEnabled(true);
    }
}

void MainWindow::appendLog(const QString& line)
{
    log_->appendPlainText(line);
}

void MainWindow::refreshPairList()
{
    const QString previous = pairList_->currentItem()
                                 ? pairList_->currentItem()->data(Qt::UserRole).toString()
                                 : QString();

    pairList_->blockSignals(true);
    pairList_->clear();
    if (!detected_) {
        pairList_->blockSignals(false);
        return;
    }

    QHash<QString, calib::PairStatus> status;
    for (const auto& entry : detected_->status) status.insert(QString::fromStdString(entry.key), entry);

    int firstOk = -1;
    for (const auto& pair : detected_->pairs) {
        const QString key = QString::fromStdString(pair.key);
        const auto it = status.constFind(key);
        const bool ok = it != status.constEnd() && it->ok;
        const bool excluded = excluded_.contains(key);
        QString text = key;
        text += ok ? QStringLiteral("   ok") : QStringLiteral("   no board");
        if (excluded) text += QStringLiteral("   (excluded)");
        auto* item = new QListWidgetItem(text, pairList_);
        item->setData(Qt::UserRole, key);
        if (excluded)
            item->setForeground(QColor(130, 130, 130));
        else if (!ok)
            item->setForeground(QColor(190, 40, 40));
        item->setToolTip(it != status.constEnd() && !it->ok
                             ? QString::fromStdString(it->detail)
                             : (ok ? QStringLiteral("full board in both eyes") : QString()));
        if (ok && !excluded && firstOk < 0) firstOk = pairList_->count() - 1;
    }
    pairList_->blockSignals(false);

    int restore = -1;
    if (!previous.isEmpty()) {
        for (int row = 0; row < pairList_->count(); ++row) {
            if (pairList_->item(row)->data(Qt::UserRole).toString() == previous) {
                restore = row;
                break;
            }
        }
    }
    if (restore >= 0) {
        pairList_->setCurrentItem(pairList_->item(restore));
    } else if (firstOk >= 0) {
        pairList_->setCurrentItem(pairList_->item(firstOk));
    }
    updatePreview();
}

void MainWindow::updatePreview()
{
    if (!detected_ || !pairList_->currentItem()) {
        preview_->clearImage();
        return;
    }
    const QString key = pairList_->currentItem()->data(Qt::UserRole).toString();

    int index = -1;
    for (std::size_t i = 0; i < detected_->kept.size(); ++i) {
        if (QString::fromStdString(detected_->kept[i].key) == key) {
            index = static_cast<int>(i);
            break;
        }
    }
    if (index < 0) {
        preview_->clearImage();
        return;
    }
    const calib::ImagePair& pair = detected_->kept[static_cast<std::size_t>(index)];
    const bool rightEye = eyeSelect_->currentIndex() == 1;
    cv::Mat image = cv::imread(rightEye ? pair.right.string() : pair.left.string(),
                               cv::IMREAD_COLOR);
    if (image.empty()) {
        preview_->clearImage();
        return;
    }
    if (showCorners_->isChecked()) {
        const auto& corners = rightEye ? detected_->rightCorners[static_cast<std::size_t>(index)]
                                       : detected_->leftCorners[static_cast<std::size_t>(index)];
        image = calib::drawDetectedCorners(image, corners,
                                           cv::Size(boardWidth_->value(), boardHeight_->value()),
                                           showIds_->isChecked());
    }
    preview_->setMat(image);
}

void MainWindow::openOutputDir()
{
    const QString dir = outputDir_->text().isEmpty() ? QStringLiteral("output")
                                                     : outputDir_->text();
    QDir().mkpath(dir);
    if (!QDesktopServices::openUrl(QUrl::fromLocalFile(QDir(dir).absolutePath())))
        QMessageBox::information(this, tr("Open output folder"),
                                 tr("Could not open %1 with the desktop's file manager.")
                                     .arg(QDir(dir).absolutePath()));
}

void MainWindow::updateExclusionLabel()
{
    if (excluded_.isEmpty()) {
        exclusionLabel_->setText(tr("none"));
        clearExclusionsButton_->setEnabled(false);
        return;
    }
    QStringList keys;
    for (const QString& key : excluded_) keys << key;
    keys.sort();
    exclusionLabel_->setText(keys.join(QStringLiteral(", ")));
    clearExclusionsButton_->setEnabled(true);
}

void MainWindow::clearExclusions()
{
    excluded_.clear();
    updateExclusionLabel();
    refreshPairList();
    appendLog(tr("Excluded views cleared; the next run uses every pair."));
}

void MainWindow::closeEvent(QCloseEvent* event)
{
    saveParams();
    if (worker_) {
        worker_->requestCancel();
        worker_->wait();
        delete worker_;
        worker_ = nullptr;
    }
    saveParams();
    QMainWindow::closeEvent(event);
}
