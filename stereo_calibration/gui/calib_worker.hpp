#ifndef STEREO_CALIB_GUI_CALIB_WORKER_HPP
#define STEREO_CALIB_GUI_CALIB_WORKER_HPP

#include <QThread>

#include <atomic>
#include <memory>
#include <string>

#include "calib_core.hpp"

/// Runs one pipeline job off the GUI thread.
///
/// QThread's run() override rather than worker-object + moveToThread: the job
/// is a single blocking call with no event loop of its own, so there is nothing
/// for a queued-invocation rewrite to buy, and object lifetime stays obvious.
class CalibWorker : public QThread {
    Q_OBJECT
public:
    enum class Job { Detect, Calibrate };

    CalibWorker(calib::Settings settings, Job job, QObject* parent = nullptr);
    ~CalibWorker() override;

    void requestCancel();

    /// Valid only after the thread has finished (the caller uses wait()).
    std::shared_ptr<calib::DetectResult> detectResult() const { return detectResult_; }
    std::shared_ptr<calib::RunResult> runResult() const { return runResult_; }
    bool wasCancelled() const { return cancelled_; }

signals:
    void logLine(const QString& line);
    void progress(const QString& stage, int done, int total);
    void failed(const QString& message);

protected:
    void run() override;

private:
    void flushLog();

    calib::Settings settings_;
    Job job_;
    std::atomic<bool> cancel_{false};
    bool cancelled_ = false;

    std::shared_ptr<calib::DetectResult> detectResult_;
    std::shared_ptr<calib::RunResult> runResult_;

    std::string buffer_;
};

#endif  // STEREO_CALIB_GUI_CALIB_WORKER_HPP
