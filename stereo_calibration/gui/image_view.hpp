#ifndef STEREO_CALIB_GUI_IMAGE_VIEW_HPP
#define STEREO_CALIB_GUI_IMAGE_VIEW_HPP

#include <QGraphicsView>

#include <opencv2/core.hpp>

class QGraphicsPixmapItem;
class QGraphicsScene;

/// Zoomable, pannable image viewer. A plain QLabel cannot show a 1280x480
/// epipolar canvas at a readable size on every screen, so the widget is a
/// QGraphicsView with a single pixmap item.
class ImageView : public QGraphicsView {
    Q_OBJECT
public:
    explicit ImageView(QWidget* parent = nullptr);

    void setMat(const cv::Mat& mat);
    void clearImage();
    void fitToWindow();
    bool hasImage() const;

protected:
    void wheelEvent(QWheelEvent* event) override;
    void mouseDoubleClickEvent(QMouseEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;

private:
    void applyFit();

    QGraphicsScene* scene_ = nullptr;
    QGraphicsPixmapItem* item_ = nullptr;
    bool fitPending_ = false;
};

#endif  // STEREO_CALIB_GUI_IMAGE_VIEW_HPP
