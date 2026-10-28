#include "sgm_single_path.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

static stereo::GrayImage read_gray(const std::string& path) {
    cv::Mat src = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (src.empty()) throw std::runtime_error("cannot read image: " + path);
    stereo::GrayImage im{src.cols,src.rows,
        std::vector<uint8_t>(static_cast<size_t>(src.cols)*src.rows)};
    for(int y=0;y<src.rows;++y)
        std::copy_n(src.ptr<uint8_t>(y),src.cols,im.pixels.begin()+static_cast<size_t>(y)*src.cols);
    return im;
}

static void write_disparity(const fs::path& output, const stereo::BMResult& r,
                            int minD, int numD) {
    cv::Mat d16(r.height,r.width,CV_16UC1,cv::Scalar(0));
    cv::Mat mask(r.height,r.width,CV_8UC1,cv::Scalar(0));
    cv::Mat gray(r.height,r.width,CV_8UC1,cv::Scalar(0));
    for(int y=0;y<r.height;++y) for(int x=0;x<r.width;++x) {
        const int d=r.disparity[static_cast<size_t>(y)*r.width+x];
        if(d<0) continue;
        if(d>=256) throw std::runtime_error("16-bit PNG scale 256 supports only disparities <256");
        d16.at<uint16_t>(y,x)=static_cast<uint16_t>(d*256);
        mask.at<uint8_t>(y,x)=255;
        gray.at<uint8_t>(y,x)=static_cast<uint8_t>(
            std::clamp((d-minD)*255/std::max(1,numD-1),0,255));
    }
    cv::Mat color;
    cv::applyColorMap(gray,color,cv::COLORMAP_TURBO);
    color.setTo(cv::Scalar(0,0,0),mask==0);
    cv::imwrite((output.string()+"_disp16.png"),d16);
    cv::imwrite((output.string()+"_valid.png"),mask);
    cv::imwrite((output.string()+"_vis.png"),color);
}

static void print_stats(const char* name, const stereo::BMResult& pred,
                        const cv::Mat& gt, const cv::Mat& gtMask) {
    int validGT=0, validPred=0, bad1=0;
    double sumErr=0,sumSq=0;
    for(int y=0;y<pred.height;++y) for(int x=0;x<pred.width;++x) {
        const uint16_t g=gt.at<uint16_t>(y,x);
        if(g==0 || (!gtMask.empty() && gtMask.at<uint8_t>(y,x)==0)) continue;
        ++validGT;
        const int d=pred.disparity[static_cast<size_t>(y)*pred.width+x];
        if(d<0) continue;
        ++validPred;
        const double err=std::abs(d-g/256.0);
        sumErr+=err;
        sumSq+=err*err;
        bad1+=err>1.0;
    }
    std::cout<<name<<": GT="<<validGT<<", matched="<<validPred
             <<", coverage="<<(validGT?100.0*validPred/validGT:0)<<"%";
    if(validPred>0)
        std::cout<<", EPE="<<sumErr/validPred<<" px, RMSE="
                 <<std::sqrt(sumSq/validPred)<<" px, Bad1="
                 <<100.0*bad1/validPred<<"%";
    std::cout<<"\n";
}

int main(int argc,char** argv) {
    if(argc<9 || argc>11) {
        std::cerr<<"Usage: handmade_sgm_1path left.png right.png outdir minDisp numDisp blockSize P1 P2 [gt_disp16.png] [gt_mask.png]\n";
        return 2;
    }
    try {
        const auto left=read_gray(argv[1]);
        const auto right=read_gray(argv[2]);
        const fs::path output=argv[3];
        fs::create_directories(output);
        const stereo::BMParams bp{std::stoi(argv[4]),std::stoi(argv[5]),std::stoi(argv[6])};
        const stereo::SGMParams sp{std::stoi(argv[7]),std::stoi(argv[8])};

        const auto t0=std::chrono::steady_clock::now();
        const auto C=stereo::compute_cost_volume_sad_integral(left,right,bp);
        const auto t1=std::chrono::steady_clock::now();
        const auto L=stereo::aggregate_left_to_right(C,sp);
        const auto t2=std::chrono::steady_clock::now();
        const auto bm=stereo::winner_takes_all(C);
        const auto one=stereo::winner_takes_all(L);
        const auto t3=std::chrono::steady_clock::now();
        auto ms=[](auto a,auto b){return std::chrono::duration<double,std::milli>(b-a).count();};
        std::cout<<"SAD volume="<<ms(t0,t1)<<"ms, path="<<ms(t1,t2)
                 <<"ms, WTA x2="<<ms(t2,t3)<<"ms\n";
        std::cout<<"C memory="<<C.costs.size()*sizeof(int32_t)/1048576.0
                 <<" MiB, L memory="<<L.costs.size()*sizeof(int32_t)/1048576.0<<" MiB\n";

        write_disparity(output/"bm",bm,bp.min_disparity,bp.num_disparities);
        write_disparity(output/"sgm_1path",one,bp.min_disparity,bp.num_disparities);

        cv::Mat diff(left.height,left.width,CV_8UC1,cv::Scalar(0));
        int compared=0,changed=0;
        for(int y=0;y<left.height;++y) for(int x=0;x<left.width;++x) {
            const size_t i=static_cast<size_t>(y)*left.width+x;
            const int a=bm.disparity[i],b=one.disparity[i];
            if(a<0||b<0) continue;
            ++compared; changed+=a!=b;
            diff.at<uint8_t>(y,x)=static_cast<uint8_t>(std::min(255,std::abs(a-b)*8));
        }
        cv::imwrite((output/"bm_vs_1path_difference.png").string(),diff);
        std::cout<<"BM vs one-path: valid overlap="<<compared<<", changed="<<changed<<"\n";

        const int cy=left.height/2, cx=left.width/2;
        std::ofstream csv(output/"center_cost_curve.csv");
        csv<<"disparity,SAD_C,aggregated_L\n";
        std::cout<<"Center ("<<cx<<","<<cy<<"): disparity, C, L\n";
        for(int di=0;di<C.num_disparities;++di){
            const auto c=C.at(cy,cx,di),l=L.at(cy,cx,di);
            csv<<C.actual_disparity(di)<<","<<c<<","<<l<<"\n";
            if(di<12 || di==C.num_disparities-1)
                std::cout<<"  "<<C.actual_disparity(di)<<", "<<c<<", "<<l<<"\n";
        }

        if(argc>=10) {
            cv::Mat gt=cv::imread(argv[9],cv::IMREAD_UNCHANGED);
            if(gt.empty()||gt.type()!=CV_16UC1||gt.rows!=left.height||gt.cols!=left.width)
                throw std::runtime_error("GT must be uint16 PNG with disparity scale 256");
            cv::Mat gtMask;
            if(argc==11) {
                gtMask=cv::imread(argv[10],cv::IMREAD_GRAYSCALE);
                if(gtMask.empty()||gtMask.size()!=gt.size())
                    throw std::runtime_error("bad GT mask");
            }
            print_stats("SAD + WTA",bm,gt,gtMask);
            print_stats("SAD + 1-path + WTA",one,gt,gtMask);
        }
        std::cout<<"Done: "<<fs::absolute(output)<<"\n";
    } catch(const std::exception& e) {
        std::cerr<<"ERROR: "<<e.what()<<"\n";
        return 1;
    }
}
