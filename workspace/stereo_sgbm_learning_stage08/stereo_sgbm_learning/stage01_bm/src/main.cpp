#include "handmade_bm.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

static stereo::GrayImage read_gray(const std::string& path) {
    cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (image.empty()) throw std::runtime_error("Cannot open " + path);
    stereo::GrayImage out;
    out.width = image.cols;
    out.height = image.rows;
    out.pixels.resize(static_cast<size_t>(out.width) * out.height);
    for (int y = 0; y < out.height; ++y) {
        const auto* row = image.ptr<uint8_t>(y);
        std::copy(row, row + out.width, out.pixels.begin() + static_cast<size_t>(y)*out.width);
    }
    return out;
}

static void evaluate(const stereo::BMResult& result,
                     const std::string& gt_path, const std::string& mask_path) {
    const cv::Mat gt = cv::imread(gt_path, cv::IMREAD_UNCHANGED);
    if (gt.empty() || gt.type() != CV_16UC1 ||
        gt.cols != result.width || gt.rows != result.height) {
        throw std::runtime_error("Expected matching CV_16UC1 GT file (disparity x 256)");
    }
    cv::Mat mask;
    if (!mask_path.empty()) {
        mask = cv::imread(mask_path, cv::IMREAD_GRAYSCALE);
        if (mask.empty() || mask.size() != gt.size()) {
            throw std::runtime_error("Invalid GT valid mask");
        }
    }
    int valid_gt = 0, matched = 0, bad1 = 0;
    double sum_abs = 0.0, sum_sq = 0.0;
    for (int y = 0; y < result.height; ++y) {
        for (int x = 0; x < result.width; ++x) {
            const auto raw = gt.at<uint16_t>(y,x);
            if (!raw || (!mask.empty() && mask.at<uint8_t>(y,x)==0)) continue;
            ++valid_gt;
            const int pred = result.disparity[static_cast<size_t>(y)*result.width+x];
            if (pred < 0) continue;
            ++matched;
            double error = std::abs(pred - raw / 256.0);
            sum_abs += error;
            sum_sq += error*error;
            bad1 += error > 1.0;
        }
    }
    std::cout << "GT valid=" << valid_gt << ", BM valid on GT=" << matched << '\n';
    std::cout << "Coverage=" << (valid_gt ? 100.0*matched/valid_gt : 0.0) << "%\n";
    if (matched) {
        std::cout << "EPE=" << sum_abs/matched << "px, RMSE="
                  << std::sqrt(sum_sq/matched) << "px, Bad1="
                  << 100.0*bad1/matched << "% (only on valid intersection)\n";
    }
}

int main(int argc, char** argv) {
    if (argc < 7 || argc > 9) {
        std::cerr << "Usage: handmade_bm left.png right.png outdir minDisp numDisp blockSize [gt_disp16.png] [gt_valid_mask.png]\n";
        return 2;
    }
    try {
        const stereo::GrayImage left = read_gray(argv[1]);
        const stereo::GrayImage right = read_gray(argv[2]);
        const std::filesystem::path outdir = argv[3];
        stereo::BMParams p;
        p.min_disparity = std::stoi(argv[4]);
        p.num_disparities = std::stoi(argv[5]);
        p.block_size = std::stoi(argv[6]);
        const auto result = stereo::compute_bm_sad(left, right, p);
        std::filesystem::create_directories(outdir);

        // Ground-truth convention: d_fixed = round(d_px * 256), zero means invalid.
        cv::Mat disp16(left.height, left.width, CV_16UC1, cv::Scalar(0));
        cv::Mat valid(left.height, left.width, CV_8UC1, cv::Scalar(0));
        cv::Mat normalized(left.height, left.width, CV_8UC1, cv::Scalar(0));
        for (int y = 0; y < left.height; ++y) {
            for (int x = 0; x < left.width; ++x) {
                int d = result.disparity[static_cast<size_t>(y)*left.width + x];
                if (d < 0) continue;
                valid.at<uint8_t>(y,x) = 255;
                disp16.at<uint16_t>(y,x) = static_cast<uint16_t>(d*256);
                normalized.at<uint8_t>(y,x) = static_cast<uint8_t>(
                    std::min(255, std::max(0, (d - p.min_disparity) * 255 / std::max(1, p.num_disparities))));
            }
        }
        cv::Mat colored;
        cv::applyColorMap(normalized, colored, cv::COLORMAP_TURBO);
        colored.setTo(cv::Scalar(0,0,0), valid == 0);
        cv::imwrite((outdir / "bm_disparity_16.png").string(), disp16);
        cv::imwrite((outdir / "bm_valid.png").string(), valid);
        cv::imwrite((outdir / "bm_vis.png").string(), colored);
        if (argc >= 8) evaluate(result, argv[7], argc == 9 ? argv[8] : "");
        std::cout << "Saved BM outputs to " << outdir << '\n';
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
}
