#include "cost_volume.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

static stereo::GrayImage read_gray(const std::string& path) {
    const cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (image.empty()) throw std::runtime_error("Cannot open " + path);
    stereo::GrayImage out;
    out.width = image.cols; out.height = image.rows;
    out.pixels.resize(static_cast<size_t>(out.width) * out.height);
    for (int y = 0; y < out.height; ++y) {
        const auto* src = image.ptr<uint8_t>(y);
        std::copy(src, src + out.width,
                  out.pixels.begin() + static_cast<size_t>(y) * out.width);
    }
    return out;
}

static void evaluate(const stereo::BMResult& pred, const std::string& gtFile,
                     const std::string& maskFile) {
    const cv::Mat gt = cv::imread(gtFile, cv::IMREAD_UNCHANGED);
    if (gt.empty() || gt.type() != CV_16UC1 || gt.cols != pred.width || gt.rows != pred.height)
        throw std::runtime_error("GT must be 16-bit PNG disparity*256 matching image dimensions");
    cv::Mat mask;
    if (!maskFile.empty()) {
        mask = cv::imread(maskFile, cv::IMREAD_GRAYSCALE);
        if (mask.empty() || mask.size() != gt.size()) throw std::runtime_error("Invalid GT mask");
    }
    int total = 0, matched = 0, bad1 = 0;
    double absSum = 0, sqSum = 0;
    for (int y = 0; y < pred.height; ++y)
        for (int x = 0; x < pred.width; ++x) {
            const uint16_t raw = gt.at<uint16_t>(y, x);
            if (raw == 0 || (!mask.empty() && mask.at<uint8_t>(y,x) == 0)) continue;
            ++total;
            const int d = pred.disparity[static_cast<size_t>(y) * pred.width + x];
            if (d < 0) continue;
            ++matched;
            const double err = std::abs(d - raw / 256.0);
            absSum += err; sqSum += err * err;
            bad1 += err > 1.0;
        }
    std::cout << "GT valid=" << total << ", prediction valid=" << matched
              << ", coverage=" << (total ? 100.0 * matched / total : 0.0) << "%\n";
    if (matched)
        std::cout << "EPE=" << absSum / matched << "px, RMSE="
                  << std::sqrt(sqSum / matched) << "px, Bad1="
                  << 100.0 * bad1 / matched << "% (valid intersection)\n";
}

int main(int argc, char** argv) {
    if (argc < 7 || argc > 9) {
        std::cerr << "Usage: handmade_cost_volume left.png right.png outdir minDisp numDisp blockSize [gt_disp16.png] [gt_mask.png]\n";
        return 2;
    }
    try {
        auto left = read_gray(argv[1]);
        auto right = read_gray(argv[2]);
        stereo::BMParams p;
        p.min_disparity = std::stoi(argv[4]);
        p.num_disparities = std::stoi(argv[5]);
        p.block_size = std::stoi(argv[6]);
        const std::filesystem::path outdir(argv[3]);
        std::filesystem::create_directories(outdir);

        const auto begin = std::chrono::steady_clock::now();
        const auto costVolume = stereo::compute_cost_volume_sad_integral(left, right, p);
        const auto end = std::chrono::steady_clock::now();
        const auto disparity = stereo::winner_takes_all(costVolume);
        const double ms = std::chrono::duration<double, std::milli>(end - begin).count();
        std::cout << "Integral Cost Volume: " << ms << " ms\n";
        std::cout << "Elements: " << costVolume.costs.size() << ", memory: "
                  << (costVolume.costs.size() * sizeof(int32_t)) / 1048576.0 << " MiB\n";

        // 保存 WTA 输出；与 Stage01 使用相同格式（真实视差 x256）。
        cv::Mat disp16(left.height, left.width, CV_16UC1, cv::Scalar(0));
        cv::Mat valid(left.height, left.width, CV_8UC1, cv::Scalar(0));
        cv::Mat visGray(left.height, left.width, CV_8UC1, cv::Scalar(0));
        for (int y = 0; y < left.height; ++y)
            for (int x = 0; x < left.width; ++x) {
                const int d = disparity.disparity[static_cast<size_t>(y) * left.width + x];
                if (d < 0) continue;
                if (d >= 256) throw std::runtime_error("Disparity cannot fit disparity*256 uint16 PNG; use float output for values >=256");
                valid.at<uint8_t>(y,x) = 255;
                disp16.at<uint16_t>(y,x) = static_cast<uint16_t>(d * 256);
                visGray.at<uint8_t>(y,x) = static_cast<uint8_t>(
                    std::clamp((d - p.min_disparity) * 255 / std::max(1, p.num_disparities), 0, 255));
            }
        cv::Mat color;
        cv::applyColorMap(visGray, color, cv::COLORMAP_TURBO);
        color.setTo(cv::Scalar(0,0,0), valid == 0);
        cv::imwrite((outdir / "cost_wta_disp16.png").string(), disp16);
        cv::imwrite((outdir / "cost_wta_valid.png").string(), valid);
        cv::imwrite((outdir / "cost_wta_vis.png").string(), color);

        // 输出某个视差切片供检查。显示时归一化只是可视化，不可用于精度计算。
        const int di = p.num_disparities / 2;
        const int d = costVolume.actual_disparity(di);
        cv::Mat slice(left.height, left.width, CV_8UC1, cv::Scalar(0));
        int32_t minCost = stereo::kInvalidCost, maxCost = 0;
        for (int y = 0; y < left.height; ++y)
            for (int x = 0; x < left.width; ++x) {
                auto c = costVolume.at(y,x,di);
                if (c == stereo::kInvalidCost) continue;
                minCost = std::min(minCost, c);
                maxCost = std::max(maxCost, c);
            }
        if (minCost != stereo::kInvalidCost) {
            for (int y = 0; y < left.height; ++y)
                for (int x = 0; x < left.width; ++x) {
                    const auto c = costVolume.at(y,x,di);
                    if (c == stereo::kInvalidCost) continue;
                    slice.at<uint8_t>(y,x) = static_cast<uint8_t>(
                        (static_cast<int64_t>(c - minCost) * 254)
                        / std::max<int32_t>(1, maxCost - minCost) + 1);
                }
        }
        cv::imwrite((outdir / ("cost_slice_d" + std::to_string(d) + ".png")).string(), slice);
        const int cx = left.width/2, cy = left.height/2;
        std::cout << "Cost curve at center (" << cx << "," << cy << "):\n";
        for (int k = 0; k < p.num_disparities; ++k)
            std::cout << "  d=" << costVolume.actual_disparity(k)
                      << ", SAD=" << costVolume.at(cy,cx,k) << '\n';
        if (argc >= 8) evaluate(disparity, argv[7], argc == 9 ? argv[8] : "");
        std::cout << "Saved Stage02 outputs to " << outdir << '\n';
    } catch (const std::exception& ex) {
        std::cerr << "ERROR: " << ex.what() << '\n';
        return 1;
    }
}
