#include "lr_consistency.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static stereo::GrayImage read_gray(const std::string& file) {
    const cv::Mat img = cv::imread(file, cv::IMREAD_GRAYSCALE);
    if (img.empty()) throw std::runtime_error("cannot read image: " + file);
    stereo::GrayImage out{img.cols, img.rows,
        std::vector<uint8_t>(static_cast<size_t>(img.cols) * img.rows)};
    for (int y=0; y<img.rows; ++y)
        std::copy_n(img.ptr<uint8_t>(y),img.cols,
                    out.pixels.begin()+static_cast<size_t>(y)*img.cols);
    return out;
}

static void write_pfm(const fs::path& filename, const stereo::SubpixelResult& src) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("PFM open failed: " + filename.string());
    const uint16_t endian = 1;
    const bool little = *reinterpret_cast<const uint8_t*>(&endian) == 1;
    file << "Pf\n" << src.width << ' ' << src.height << '\n'
         << (little ? "-1.0\n" : "1.0\n");
    for (int y=src.height-1; y>=0; --y) {
        const float* row=src.disparity.data()+static_cast<size_t>(y)*src.width;
        file.write(reinterpret_cast<const char*>(row),
                   static_cast<std::streamsize>(src.width*sizeof(float)));
    }
    if (!file) throw std::runtime_error("PFM write failed: " + filename.string());
}

static void write_checked(const fs::path& prefix, const stereo::SubpixelResult& src,
                          int min_d, int num_d) {
    stereo::validate_subpixel_map(src);
    cv::Mat disp16(src.height,src.width,CV_16UC1,cv::Scalar(0));
    cv::Mat valid(src.height,src.width,CV_8UC1,cv::Scalar(0));
    cv::Mat refined(src.height,src.width,CV_8UC1,cv::Scalar(0));
    cv::Mat gray(src.height,src.width,CV_8UC1,cv::Scalar(0));
    for (int y=0; y<src.height; ++y)for(int x=0; x<src.width; ++x) {
        const size_t i=static_cast<size_t>(y)*src.width+x;
        if (!src.valid[i]) continue;
        const float d = src.disparity[i];
        if (!std::isfinite(d) || d<0.0f || d*256.0f>65535.0f)
            throw std::runtime_error("16-bit x256 PNG supports [0,255.996] px only");
        disp16.at<uint16_t>(y,x)=static_cast<uint16_t>(std::lround(d*256.0f));
        valid.at<uint8_t>(y,x)=255;
        refined.at<uint8_t>(y,x)=src.refined[i]?255:0;
        const float f=std::clamp((d-min_d)/std::max(1,num_d-1),0.0f,1.0f);
        gray.at<uint8_t>(y,x)=static_cast<uint8_t>(std::lround(255.0f*f));
    }
    cv::Mat color;
    cv::applyColorMap(gray,color,cv::COLORMAP_TURBO);
    color.setTo(cv::Scalar(0,0,0),valid==0);
    const auto save=[&](const std::string& ext,const cv::Mat& mat){
        if(!cv::imwrite(prefix.string()+ext,mat))
            throw std::runtime_error("imwrite failed: "+prefix.string()+ext);
    };
    save("_disp16.png",disp16); // float disparity *256, NOT OpenCV SGBM *16
    save("_valid.png",valid);
    save("_vis.png",color);
    save("_refined.png",refined);
    write_pfm(prefix.string()+".pfm",src);
}

static void write_status(const fs::path& filename, const stereo::LRResult& result) {
    const int W = result.filtered.width, H = result.filtered.height;
    cv::Mat status(H,W,CV_8UC1,cv::Scalar(0));
    for(int y=0;y<H;++y)for(int x=0;x<W;++x) {
        const auto v = result.status[static_cast<size_t>(y)*W+x];
        // Distinct gray labels: LeftInvalid=0, Consistent=255,
        // RightOutside=64, RightInvalid=128, Disagree=192.
        const uint8_t pixel = v==stereo::LRStatus::Consistent?255:
            v==stereo::LRStatus::RightOutside?64:
            v==stereo::LRStatus::RightInvalid?128:
            v==stereo::LRStatus::Disagree?192:0;
        status.at<uint8_t>(y,x)=pixel;
    }
    if (!cv::imwrite(filename.string(),status))
        throw std::runtime_error("cannot write LRC status image");
}

static void report_lrc(const stereo::LRResult& res) {
    size_t counts[5]={};
    for (const auto status: res.status) ++counts[static_cast<uint8_t>(status)];
    const size_t valid_before = counts[1]+counts[2]+counts[3]+counts[4];
    std::cout << "LRC status: left invalid=" << counts[0]
              << ", consistent=" << counts[1]
              << ", right outside=" << counts[2]
              << ", right invalid=" << counts[3]
              << ", disparity disagreement=" << counts[4] << '\n';
    std::cout << "LRC retention among initially valid: " << std::fixed
              << std::setprecision(2) << (valid_before?100.0*counts[1]/valid_before:0.0)
              << "%\n";
}

static void compare_gt(const char* label, const stereo::SubpixelResult& pred,
                       const cv::Mat& gt, const cv::Mat& mask) {
    int n_gt=0,n_valid=0,bad1=0;
    double sae=0,sse=0;
    for(int y=0;y<pred.height;++y)for(int x=0;x<pred.width;++x) {
        const uint16_t raw=gt.at<uint16_t>(y,x);
        if ((!mask.empty() && !mask.at<uint8_t>(y,x)) ||
            (mask.empty() && raw==0)) continue;
        ++n_gt;
        const size_t i=static_cast<size_t>(y)*pred.width+x;
        if (!pred.valid[i] || !std::isfinite(pred.disparity[i]))continue;
        const double error=std::abs(pred.disparity[i]-raw/256.0);
        ++n_valid; sae+=error;sse+=error*error;bad1+=(error>1.0);
    }
    std::cout << label << ": GT pixels=" << n_gt
              << ", evaluated=" << n_valid
              << ", coverage=" << (n_gt?100.0*n_valid/n_gt:0.0) << "%";
    if(n_valid)std::cout << ", EPE=" << sae/n_valid
                        << ", RMSE=" << std::sqrt(sse/n_valid)
                        << ", Bad1=" << 100.0*bad1/n_valid << "%";
    std::cout << '\n';
}

int main(int argc, char** argv) {
    if (argc < 10 || argc > 12) {
        std::cerr << "Usage: handmade_sgm_lr left.png right.png outdir "
                  << "minDisp numDisp blockSize P1 P2 lrThreshold "
                  << "[gt_disp16.png] [gt_valid_mask.png]\n";
        return 2;
    }
    try {
        const auto left=read_gray(argv[1]);
        const auto right=read_gray(argv[2]);
        const fs::path out=argv[3];
        fs::create_directories(out);
        const stereo::BMParams bm{std::stoi(argv[4]),std::stoi(argv[5]),std::stoi(argv[6])};
        const stereo::SGMParams sg{std::stoi(argv[7]),std::stoi(argv[8])};
        const stereo::LRParams lr{std::stof(argv[9])};
        const auto t0=std::chrono::steady_clock::now();
        const auto Cleft=stereo::compute_cost_volume_sad_integral(left,right,bm);
        const auto Sleft=stereo::aggregate_eight_directions(Cleft,sg);
        const auto Dleft=stereo::refine_subpixel(Sleft);
        const auto t1=std::chrono::steady_clock::now();
        const auto Dright=stereo::compute_right_disparity_flipped(left,right,bm,sg);
        const auto t2=std::chrono::steady_clock::now();
        const auto checked=stereo::left_right_consistency_check(Dleft,Dright,lr);
        const auto t3=std::chrono::steady_clock::now();
        const auto ms=[](auto a,auto b){return std::chrono::duration<double,std::milli>(b-a).count();};
        std::cout << "left disparity=" << ms(t0,t1) << " ms; right disparity="
                  << ms(t1,t2) << " ms; LRC=" << ms(t2,t3) << " ms\n";
        write_checked(out/"left_before_lrc",Dleft,bm.min_disparity,bm.num_disparities);
        write_checked(out/"right_disparity",Dright,bm.min_disparity,bm.num_disparities);
        write_checked(out/"left_after_lrc",checked.filtered,bm.min_disparity,bm.num_disparities);
        write_status(out/"lrc_status.png",checked);
        report_lrc(checked);
        if (argc >= 11) {
            const cv::Mat gt=cv::imread(argv[10],cv::IMREAD_UNCHANGED);
            if(gt.empty()||gt.type()!=CV_16UC1||gt.cols!=left.width||gt.rows!=left.height)
                throw std::runtime_error("GT must be same-size CV_16UC1 disparity x256");
            cv::Mat mask;
            if(argc==12) {
                mask=cv::imread(argv[11],cv::IMREAD_GRAYSCALE);
                if(mask.empty()||mask.size()!=gt.size())
                    throw std::runtime_error("GT mask size mismatch");
            }
            compare_gt("before LRC",Dleft,gt,mask);
            compare_gt("after  LRC",checked.filtered,gt,mask);
        }
        std::cout << "Output: " << fs::absolute(out) << '\n';
    } catch(const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
