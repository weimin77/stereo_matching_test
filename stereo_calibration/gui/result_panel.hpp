#ifndef STEREO_CALIB_GUI_RESULT_PANEL_HPP
#define STEREO_CALIB_GUI_RESULT_PANEL_HPP

#include <QWidget>

#include <memory>
#include <vector>

#include <opencv2/core.hpp>

#include "calib_core.hpp"

class QComboBox;
class QLabel;
class QPlainTextEdit;
class QPushButton;
class QTableWidget;
class ImageView;

/// Everything the pipeline produced, in four tabs: the numbers worth reading at
/// a glance, the raw matrices, the per-view error table that tells you which
/// frame to drop, and the rectified images.
class ResultPanel : public QWidget {
    Q_OBJECT
public:
    explicit ResultPanel(QWidget* parent = nullptr);

    void clearResult();
    void setResult(const std::shared_ptr<calib::RunResult>& result);

signals:
    void rerunWithoutViews(const QStringList& keys);

private:
    void fillSummary(const calib::RunResult& result);
    void fillMatrices(const calib::RunResult& result);
    void fillPerView(const calib::RunResult& result);
    void fillImages(const calib::RunResult& result);
    void updateImage();
    void updateExcludeButton();

    std::shared_ptr<calib::RunResult> result_;

    QTableWidget* summary_ = nullptr;
    QComboBox* matrixSelect_ = nullptr;
    QPlainTextEdit* matrixText_ = nullptr;
    QTableWidget* perView_ = nullptr;
    QPushButton* excludeButton_ = nullptr;
    QComboBox* imageSelect_ = nullptr;
    ImageView* imageView_ = nullptr;

    std::vector<QString> matrixNames_;
    std::vector<cv::Mat> matrixValues_;
    std::vector<QString> imageNames_;
    std::vector<cv::Mat> imageValues_;
};

#endif  // STEREO_CALIB_GUI_RESULT_PANEL_HPP
