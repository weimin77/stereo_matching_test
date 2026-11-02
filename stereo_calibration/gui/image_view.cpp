#include "image_view.hpp"

#include <QColor>
#include <QGraphicsPixmapItem>
#include <QGraphicsScene>
#include <QImage>
#include <QMouseEvent>
#include <QResizeEvent>
#include <QWheelEvent>

#include <cmath>

#include <opencv2/imgproc.hpp>

namespace {

QImage toQImage(const cv::Mat& mat)
{
    if (mat.empty()) return QImage();
    if (mat.type() == CV_8UC3) {
        return QImage(mat.data, mat.cols, mat.rows, static_cast<int>(mat.step),
                      QImage::Format_BGR888).copy();
    }
    if (mat.type() == CV_8UC1) {
        return QImage(mat.data, mat.cols, mat.rows, static_cast<int>(mat.step),
                      QImage::Format_Grayscale8).copy();
    }
    if (mat.type() == CV_8UC4) {
        return QImage(mat.data, mat.cols, mat.rows, static_cast<int>(mat.step),
                      QImage::Format_ARGB32).copy();
    }
    cv::Mat eightBit;
    if (mat.depth() != CV_8U) {
        mat.convertTo(eightBit, CV_8U);
    } else {
        eightBit = mat;
    }
    cv::Mat converted;
    if (eightBit.channels() == 1) {
        cv::cvtColor(eightBit, converted, cv::COLOR_GRAY2BGR);
    } else if (eightBit.channels() == 3) {
        converted = eightBit;
    } else {
        return QImage();
    }
    return QImage(converted.data, converted.cols, converted.rows,
                  static_cast<int>(converted.step), QImage::Format_BGR888).copy();
}

}  // namespace

ImageView::ImageView(QWidget* parent)
    : QGraphicsView(parent)
{
    scene_ = new QGraphicsScene(this);
    setScene(scene_);
    setRenderHint(QPainter::SmoothPixmapTransform, true);
    setDragMode(QGraphicsView::ScrollHandDrag);
    setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
    setResizeAnchor(QGraphicsView::AnchorViewCenter);
    setBackgroundBrush(QColor(32, 32, 32));
    setFrameShape(QFrame::StyledPanel);
    setMinimumSize(240, 180);
}

void ImageView::setMat(const cv::Mat& mat)
{
    const QImage image = toQImage(mat);
    if (image.isNull()) {
        clearImage();
        return;
    }
    scene_->clear();
    item_ = scene_->addPixmap(QPixmap::fromImage(image));
    item_->setTransformationMode(Qt::SmoothTransformation);
    scene_->setSceneRect(item_->boundingRect());
    fitPending_ = true;
    applyFit();
}

void ImageView::clearImage()
{
    scene_->clear();
    item_ = nullptr;
    scene_->setSceneRect(QRectF());
}

void ImageView::fitToWindow()
{
    if (!item_) return;
    resetTransform();
    fitInView(item_, Qt::KeepAspectRatio);
}

bool ImageView::hasImage() const
{
    return item_ != nullptr;
}

void ImageView::applyFit()
{
    if (!fitPending_ || !item_ || !isVisible()) return;
    // Defer until the widget has a real size, otherwise the first fit is
    // computed against the 100x30 default viewport and zooms absurdly in.
    if (viewport()->width() < 50 || viewport()->height() < 50) return;
    fitToWindow();
    fitPending_ = false;
}

void ImageView::wheelEvent(QWheelEvent* event)
{
    if (!item_) {
        QGraphicsView::wheelEvent(event);
        return;
    }
    const double steps = event->angleDelta().y() / 120.0;
    if (steps == 0.0) {
        event->ignore();
        return;
    }
    const double factor = std::pow(1.25, steps);
    scale(factor, factor);
    fitPending_ = false;
    event->accept();
}

void ImageView::mouseDoubleClickEvent(QMouseEvent* event)
{
    if (item_ && event->button() == Qt::LeftButton) {
        fitToWindow();
        fitPending_ = false;
        event->accept();
        return;
    }
    QGraphicsView::mouseDoubleClickEvent(event);
}

void ImageView::resizeEvent(QResizeEvent* event)
{
    QGraphicsView::resizeEvent(event);
    if (fitPending_) applyFit();
}
