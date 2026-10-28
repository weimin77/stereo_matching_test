#include "sgm_eight_directions.hpp"
#include "sgm_four_directions.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

static stereo::GrayImage read_gray(const std::string& path) {
    cv::Mat src=cv::imread(path,cv::IMREAD_GRAYSCALE);
    if(src.empty()) throw std::runtime_error("cannot read: " + path);
    stereo::GrayImage dst{src.cols,src.rows,
        std::vector<uint8_t>(static_cast<size_t>(src.cols)*src.rows)};
    for(int y=0;y<src.rows;++y)
        std::copy_n(src.ptr<uint8_t>(y),src.cols,
                    dst.pixels.begin()+static_cast<size_t>(y)*src.cols);
    return dst;
}

static void write_disparity(const fs::path& filename,
                            const stereo::BMResult& pred, int minD,int D) {
    cv::Mat d16(pred.height,pred.width,CV_16UC1,cv::Scalar(0));
    cv::Mat mask(pred.height,pred.width,CV_8UC1,cv::Scalar(0));
    cv::Mat gray(pred.height,pred.width,CV_8UC1,cv::Scalar(0));
    for(int y=0;y<pred.height;++y)for(int x=0;x<pred.width;++x) {
        const int d=pred.disparity[static_cast<size_t>(y)*pred.width+x];
        if(d<0)continue;
        if(d>=256)throw std::runtime_error("PNG16 disparity x256 needs d<256");
        d16.at<uint16_t>(y,x)=static_cast<uint16_t>(d*256);
        mask.at<uint8_t>(y,x)=255;
        gray.at<uint8_t>(y,x)=static_cast<uint8_t>(
            std::clamp((d-minD)*255/std::max(1,D-1),0,255));
    }
    cv::Mat colors;
    cv::applyColorMap(gray,colors,cv::COLORMAP_TURBO);
    colors.setTo(cv::Scalar(0,0,0),mask==0);
    if(!cv::imwrite(filename.string()+"_disp16.png",d16) ||
       !cv::imwrite(filename.string()+"_valid.png",mask) ||
       !cv::imwrite(filename.string()+"_vis.png",colors))
        throw std::runtime_error("failed to write disparity output images");
}

static void print_stats(const char* label,const stereo::BMResult& prediction,
                        const cv::Mat& gt,const cv::Mat& gt_mask) {
    int validGT=0,validPred=0,bad1=0;
    double sumError=0,sumErrorSquared=0;
    for(int y=0;y<prediction.height;++y)for(int x=0;x<prediction.width;++x) {
        const uint16_t raw=gt.at<uint16_t>(y,x);
        if(raw==0||(!gt_mask.empty()&&gt_mask.at<uint8_t>(y,x)==0))continue;
        ++validGT;
        const int d=prediction.disparity[static_cast<size_t>(y)*prediction.width+x];
        if(d<0)continue;
        ++validPred;
        const double error=std::abs(d-raw/256.0);
        sumError+=error;
        sumErrorSquared+=error*error;
        bad1+=(error>1.0);
    }
    std::cout<<label<<": validGT="<<validGT<<", matched="<<validPred
             <<", coverage="<<(validGT?100.0*validPred/validGT:0)<<"%";
    if(validPred)std::cout<<", EPE="<<sumError/validPred<<"px, RMSE="
                          <<std::sqrt(sumErrorSquared/validPred)<<"px, Bad1="
                          <<100.0*bad1/validPred<<"%";
    std::cout<<"\n";
}

int main(int argc,char** argv) {
    if(argc<9||argc>11) {
        std::cerr<<"Usage: handmade_sgm_8path left.png right.png outdir minDisp numDisp blockSize P1 P2 [gt_disp16.png] [gt_mask.png]\n";
        return 2;
    }
    try {
        const auto left=read_gray(argv[1]);
        const auto right=read_gray(argv[2]);
        const fs::path out=argv[3];
        fs::create_directories(out);
        const stereo::BMParams bmparams{std::stoi(argv[4]),std::stoi(argv[5]),std::stoi(argv[6])};
        const stereo::SGMParams sgparams{std::stoi(argv[7]),std::stoi(argv[8])};
        const auto t0=std::chrono::steady_clock::now();
        const auto C=stereo::compute_cost_volume_sad_integral(left,right,bmparams);
        const auto t1=std::chrono::steady_clock::now();
        const auto L=stereo::aggregate_left_to_right(C,sgparams);
        const auto t2=std::chrono::steady_clock::now();
        const auto S4=stereo::aggregate_four_directions(C,sgparams);
        const auto t3=std::chrono::steady_clock::now();
        const auto S8=stereo::aggregate_eight_directions(C,sgparams);
        const auto t4=std::chrono::steady_clock::now();
        const auto bm=stereo::winner_takes_all(C);
        const auto one=stereo::winner_takes_all(L);
        const auto four=stereo::winner_takes_all(S4);
        const auto eight=stereo::winner_takes_all(S8);
        const auto t5=std::chrono::steady_clock::now();
        const auto ms=[](auto a,auto b){return std::chrono::duration<double,std::milli>(b-a).count();};
        std::cout<<"SAD volume="<<ms(t0,t1)<<"ms, one-path="<<ms(t1,t2)
                 <<"ms, 4-path="<<ms(t2,t3)<<"ms, 8-path="<<ms(t3,t4)
                 <<"ms, 4xWTA="<<ms(t4,t5)<<"ms\n";
        std::cout<<"Each full int32 cost volume="<<C.costs.size()*sizeof(int32_t)/1048576.0<<" MiB\n";

        write_disparity(out/"bm",bm,bmparams.min_disparity,bmparams.num_disparities);
        write_disparity(out/"sgm_1path",one,bmparams.min_disparity,bmparams.num_disparities);
        write_disparity(out/"sgm_4path",four,bmparams.min_disparity,bmparams.num_disparities);
        write_disparity(out/"sgm_8path",eight,bmparams.min_disparity,bmparams.num_disparities);

        cv::Mat diff(left.height,left.width,CV_8UC1,cv::Scalar(0));
        int compared=0,changed=0;
        for(int y=0;y<left.height;++y)for(int x=0;x<left.width;++x) {
            const size_t i=static_cast<size_t>(y)*left.width+x;
            const int a=four.disparity[i],b=eight.disparity[i];
            if(a<0||b<0)continue;
            ++compared; changed+=(a!=b);
            diff.at<uint8_t>(y,x)=static_cast<uint8_t>(std::min(255,std::abs(a-b)*8));
        }
        cv::imwrite((out/"4path_vs_8path_difference.png").string(),diff);
        std::cout<<"4path vs 8path valid comparison pixels="<<compared<<", changed="<<changed<<"\n";

        const int cx=left.width/2, cy=left.height/2;
        std::ofstream csv(out/"center_cost_curve.csv");
        if(!csv)throw std::runtime_error("cannot write center_cost_curve.csv");
        csv<<"disparity,original_SAD_C,left_right_L,four_direction_S,eight_direction_S\n";
        for(int di=0;di<C.num_disparities;++di)
            csv<<C.actual_disparity(di)<<","<<C.at(cy,cx,di)<<","<<L.at(cy,cx,di)
               <<","<<S4.at(cy,cx,di)<<","<<S8.at(cy,cx,di)<<"\n";

        if(argc>=10) {
            const cv::Mat gt=cv::imread(argv[9],cv::IMREAD_UNCHANGED);
            if(gt.empty()||gt.type()!=CV_16UC1||gt.cols!=left.width||gt.rows!=left.height)
                throw std::runtime_error("GT must be matching-size 16bit PNG with disparity x256");
            cv::Mat gtMask;
            if(argc==11) {
                gtMask=cv::imread(argv[10],cv::IMREAD_GRAYSCALE);
                if(gtMask.empty()||gtMask.size()!=gt.size())
                    throw std::runtime_error("GT mask must match image size");
            }
            print_stats("SAD + WTA",bm,gt,gtMask);
            print_stats("SAD + 1path + WTA",one,gt,gtMask);
            print_stats("SAD + 4path + WTA",four,gt,gtMask);
            print_stats("SAD + 8path + WTA",eight,gt,gtMask);
        }
        std::cout<<"Output: "<<fs::absolute(out)<<"\n";
    } catch(const std::exception& e) {
        std::cerr<<"ERROR: "<<e.what()<<"\n";
        return 1;
    }
    return 0;
}
