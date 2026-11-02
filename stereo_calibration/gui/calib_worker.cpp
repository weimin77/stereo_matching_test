#include "calib_worker.hpp"

#include <QString>

#include <exception>

CalibWorker::CalibWorker(calib::Settings settings, Job job, QObject* parent)
    : QThread(parent), settings_(std::move(settings)), job_(job)
{
}

CalibWorker::~CalibWorker()
{
    // The sink captures `this`; leaving it installed after the object dies would
    // crash the next pipeline call (the GUI is single-job, but be explicit).
    calib::clearLogSink();
}

void CalibWorker::requestCancel()
{
    cancel_.store(true);
}

void CalibWorker::flushLog()
{
    if (buffer_.empty()) return;
    emit logLine(QString::fromStdString(buffer_));
    buffer_.clear();
}

void CalibWorker::run()
{
    calib::setLogSink([this](const std::string& chunk) {
        buffer_ += chunk;
        std::size_t pos = 0;
        while ((pos = buffer_.find('\n')) != std::string::npos) {
            const std::string line = buffer_.substr(0, pos);
            buffer_.erase(0, pos + 1);
            emit logLine(QString::fromStdString(line));
        }
        // printMatrix() writes several fragments per line; anything larger than
        // this without a newline is not a progress line and should not be held.
        if (buffer_.size() > 8192) {
            emit logLine(QString::fromStdString(buffer_));
            buffer_.clear();
        }
    });

    calib::RunHooks hooks;
    hooks.cancel = &cancel_;
    hooks.progress = [this](const std::string& stage, int done, int total) {
        emit progress(QString::fromStdString(stage), done, total);
    };

    try {
        if (job_ == Job::Detect) {
            detectResult_ = std::make_shared<calib::DetectResult>(
                calib::detectPairs(settings_, hooks));
        } else {
            auto result = std::make_shared<calib::RunResult>(calib::run(settings_, hooks));
            if (cancel_.load()) throw calib::Cancelled();
            calib::writeOutputs(*result, settings_);
            calib::printConclusion(*result);
            runResult_ = std::move(result);
        }
    } catch (const calib::Cancelled&) {
        cancelled_ = true;
    } catch (const std::exception& error) {
        flushLog();
        calib::clearLogSink();
        emit failed(QString::fromUtf8(error.what()));
        return;
    } catch (...) {
        flushLog();
        calib::clearLogSink();
        emit failed(QStringLiteral("unknown error"));
        return;
    }

    flushLog();
    calib::clearLogSink();
}
