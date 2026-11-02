#ifndef STEREO_CALIB_GUI_MAIN_WINDOW_HPP
#define STEREO_CALIB_GUI_MAIN_WINDOW_HPP

#include <QMainWindow>
#include <QSet>
#include <QString>

#include <memory>

#include "calib_core.hpp"
#include "gui/calib_worker.hpp"

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QLabel;
class QLineEdit;
class QListWidget;
class QPlainTextEdit;
class QProgressBar;
class QPushButton;
class QSpinBox;
class ImageView;
class ResultPanel;
struct GuiParams;

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow() override;

    /// Fills the form and starts a calibration without user interaction. Used
    /// by --selftest so the GUI path can be exercised on a headless machine.
    void applyTestParams(const QString& images, const QString& output, int boardWidth,
                         int boardHeight, double squareMm);
    bool startCalibration();
    bool startDetection();
    void setExcludedViews(const QStringList& keys);
    void showResultsTab();
    std::shared_ptr<calib::RunResult> runResult() const { return result_; }
    std::shared_ptr<calib::DetectResult> detectResult() const { return detected_; }

signals:
    void jobFinished(bool ok);

protected:
    void closeEvent(QCloseEvent* event) override;

private:
    QWidget* buildParameterPanel();
    void buildUi();
    GuiParams collectParams() const;
    void loadParams();
    void saveParams();

    void startJob(CalibWorker::Job job);
    void onJobFinished();
    void setBusy(bool busy);

    void appendLog(const QString& line);
    void refreshPairList();
    void updatePreview();
    void updateExclusionLabel();
    void openOutputDir();
    void clearExclusions();

    // Input
    QLineEdit* imageDir_ = nullptr;
    QLineEdit* leftPrefix_ = nullptr;
    QLineEdit* rightPrefix_ = nullptr;
    QLineEdit* extension_ = nullptr;
    QLineEdit* reference_ = nullptr;
    // Board
    QSpinBox* boardWidth_ = nullptr;
    QSpinBox* boardHeight_ = nullptr;
    QDoubleSpinBox* squareMm_ = nullptr;
    QCheckBox* useSb_ = nullptr;
    // Rectification
    QDoubleSpinBox* alpha_ = nullptr;
    QCheckBox* refineIntrinsics_ = nullptr;
    QCheckBox* sameFocalLength_ = nullptr;
    // Output
    QLineEdit* outputDir_ = nullptr;
    QCheckBox* saveDebugImages_ = nullptr;
    QPushButton* clearExclusionsButton_ = nullptr;
    QLabel* exclusionLabel_ = nullptr;

    QPushButton* detectButton_ = nullptr;
    QPushButton* runButton_ = nullptr;
    QPushButton* cancelButton_ = nullptr;
    QPushButton* openOutputButton_ = nullptr;
    QWidget* parameterPanel_ = nullptr;

    QListWidget* pairList_ = nullptr;
    QComboBox* eyeSelect_ = nullptr;
    QCheckBox* showCorners_ = nullptr;
    QCheckBox* showIds_ = nullptr;
    ImageView* preview_ = nullptr;

    QPlainTextEdit* log_ = nullptr;
    ResultPanel* results_ = nullptr;
    QTabWidget* tabs_ = nullptr;
    QProgressBar* progress_ = nullptr;

    CalibWorker* worker_ = nullptr;
    bool busy_ = false;
    QString errorMessage_;

    std::shared_ptr<calib::DetectResult> detected_;
    std::shared_ptr<calib::RunResult> result_;
    QSet<QString> excluded_;
    bool testMode_ = false;
};

#endif  // STEREO_CALIB_GUI_MAIN_WINDOW_HPP
