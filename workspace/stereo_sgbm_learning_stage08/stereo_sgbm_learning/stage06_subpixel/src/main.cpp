#include "subpixel_refinement.hpp"
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
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static stereo::GrayImage read_gray(const std::string& file) {
    const cv::Mat src=cv::imread(file,cv::IMREAD_GRAYSCALE);
    if(src.empty()) throw std::runtime_error("cannot read image: " + file);
    stereo::GrayImage dst{src.cols,src.rows,
        std::vector<uint8_t>(static_cast<size_t>(src.cols)*src.rows)};
    for(int y=0;y<src.rows;++y)
        std::copy_n(src.ptr<uint8_t>(y),src.cols,
                    dst.pixels.begin()+static_cast<size_t>(y)*src.cols);
    return dst;
}

static void write_pfm(const fs::path& file, const stereo::SubpixelResult& disp) {
    std::ofstream out(file,std::ios::binary);
    if(!out) throw std::runtime_error("cannot write PFM: " + file.string());
    const uint16_t endian_test=1;
    const bool little_endian=*reinterpret_cast<const uint8_t*>(&endian_test)==1;
    out<<"Pf\n"<<disp.width<<" "<<disp.height<<"\n"
       <<(little_endian?"-1.0\n":"1.0\n");
    // PFM 第一行是图像底部。
    for(int y=disp.height-1;y>=0;--y) {
        const float* row=disp.disparity.data()+static_cast<size_t>(y)*disp.width;
        out.write(reinterpret_cast<const char*>(row),
                  static_cast<std::streamsize>(disp.width*sizeof(float)));
    }
    if(!out) throw std::runtime_error("PFM write error: " + file.string());
}

static void save_disparity(const fs::path& prefix,
                           const stereo::SubpixelResult& value,
                           int min_d, int num_d, bool write_float) {
    cv::Mat encoded(value.height,value.width,CV_16UC1,cv::Scalar(0));
    cv::Mat mask(value.height,value.width,CV_8UC1,cv::Scalar(0));
    cv::Mat refined(value.height,value.width,CV_8UC1,cv::Scalar(0));
    cv::Mat gray(value.height,value.width,CV_8UC1,cv::Scalar(0));
    cv::Mat diff(value.height,value.width,CV_8UC1,cv::Scalar(0));
    for(int y=0;y<value.height;++y)for(int x=0;x<value.width;++x) {
        const size_t i=static_cast<size_t>(y)*value.width+x;
        if(!value.valid[i])continue;
        const float d=value.disparity[i];
        if(!std::isfinite(d) || d<0.0f || d*256.0f>65535.0f)
            throw std::runtime_error("x256 uint16 PNG requires valid disparity in [0,255.996]");
        encoded.at<uint16_t>(y,x)=static_cast<uint16_t>(std::lround(d*256.0f));
        mask.at<uint8_t>(y,x)=255;
        refined.at<uint8_t>(y,x)=value.refined[i]?255:0;
        const float rel=std::clamp((d-min_d)/std::max(1,num_d-1),0.0f,1.0f);
        gray.at<uint8_t>(y,x)=static_cast<uint8_t>(std::lround(rel*255.0f));
        const float delta=d-std::round(d);
        diff.at<uint8_t>(y,x)=static_cast<uint8_t>(
            std::lround(std::clamp(delta+0.5f,0.0f,1.0f)*255.0f));
    }
    cv::Mat colored;
    cv::applyColorMap(gray,colored,cv::COLORMAP_TURBO);
    colored.setTo(cv::Scalar(0,0,0),mask==0);
    const auto save=[&](const std::string& suffix,const cv::Mat& img) {
        if(!cv::imwrite(prefix.string()+suffix,img))
            throw std::runtime_error("cv::imwrite failed: "+prefix.string()+suffix);
    };
    save("_disp16.png",encoded);
    save("_valid.png",mask);
    save("_vis.png",colored);
    if(write_float) {
        save("_refined.png",refined);
        save("_minus_integer.png",diff);
        write_pfm(prefix.string()+".pfm",value);
    }
}

static stereo::SubpixelResult integer_to_float(const stereo::BMResult& src) {
    stereo::SubpixelResult out;
    out.width=src.width;out.height=src.height;
    const size_t n=src.disparity.size();
    out.disparity.assign(n,std::numeric_limits<float>::quiet_NaN());
    out.valid.assign(n,0);out.refined.assign(n,0);
    for(size_t i=0;i<n;++i)if(src.disparity[i]>=0){
        out.valid[i]=1;out.disparity[i]=src.disparity[i];
    }
    return out;
}

static void compare_gt(const char* title,
                       const stereo::SubpixelResult& pred,
                       const cv::Mat& gt,const cv::Mat& gt_mask) {
    int gt_valid=0,pred_valid=0,bad1=0;
    double sum=0,sum2=0;
    for(int y=0;y<pred.height;++y)for(int x=0;x<pred.width;++x) {
        const uint16_t raw=gt.at<uint16_t>(y,x);
        if(raw==0 || (!gt_mask.empty()&&gt_mask.at<uint8_t>(y,x)==0))continue;
        ++gt_valid;
        const size_t idx=static_cast<size_t>(y)*pred.width+x;
        if(!pred.valid[idx])continue;
        const double err=std::abs(static_cast<double>(pred.disparity[idx])-raw/256.0);
        ++pred_valid;sum+=err;sum2+=err*err;bad1+=(err>1.0);
    }
    std::cout<<title<<": gt="<<gt_valid<<", valid="<<pred_valid
             <<", coverage="<<(gt_valid?100.0*pred_valid/gt_valid:0.0)<<"%";
    if(pred_valid)std::cout<<", EPE="<<sum/pred_valid<<" px, RMSE="
                          <<std::sqrt(sum2/pred_valid)<<" px, Bad1="
                          <<100.0*bad1/pred_valid<<"%";
    std::cout<<'\n';
}

int main(int argc,char** argv) {
    if(argc<9 || argc>11) {
        std::cerr<<"Usage: handmade_sgm_subpixel left.png right.png outdir "
                 <<"minDisp numDisp blockSize P1 P2 [gt_disp16.png] [gt_mask.png]\n";
        return 2;
    }
    try {
        const auto L=read_gray(argv[1]);
        const auto R=read_gray(argv[2]);
        const fs::path out=argv[3];fs::create_directories(out);
        const stereo::BMParams bp{std::stoi(argv[4]),std::stoi(argv[5]),std::stoi(argv[6])};
        const stereo::SGMParams sp{std::stoi(argv[7]),std::stoi(argv[8])};
        const auto t0=std::chrono::steady_clock::now();
        const auto C=stereo::compute_cost_volume_sad_integral(L,R,bp);
        const auto t1=std::chrono::steady_clock::now();
        const auto S=stereo::aggregate_eight_directions(C,sp);
        const auto t2=std::chrono::steady_clock::now();
        const auto integer=integer_to_float(stereo::winner_takes_all(S));
        const auto sub=stereo::refine_subpixel(S);
        const auto t3=std::chrono::steady_clock::now();
        const auto ms=[](auto a,auto b){return std::chrono::duration<double,std::milli>(b-a).count();};
        std::cout<<"SAD cost="<<ms(t0,t1)<<" ms, eight paths="<<ms(t1,t2)
                 <<" ms, WTA+subpixel="<<ms(t2,t3)<<" ms\n";
        save_disparity(out/"sgm_8path_integer",integer,bp.min_disparity,bp.num_disparities,false);
        save_disparity(out/"sgm_8path_subpixel",sub,bp.min_disparity,bp.num_disparities,true);
        int valid=0,refined=0;
        for(size_t i=0;i<sub.valid.size();++i){valid+=sub.valid[i];refined+=sub.refined[i];}
        std::cout<<"valid="<<valid<<", genuinely subpixel-shifted="<<refined<<"\n";
        const int cx=L.width/2,cy=L.height/2;
        std::ofstream csv(out/"center_cost_curve.csv");
        if(!csv)throw std::runtime_error("cannot save CSV");
        csv<<"disparity,eight_direction_S\n";
        for(int di=0;di<S.num_disparities;++di)
            csv<<S.actual_disparity(di)<<','<<S.at(cy,cx,di)<<'\n';
        if(argc>=10) {
            const cv::Mat gt=cv::imread(argv[9],cv::IMREAD_UNCHANGED);
            if(gt.empty()||gt.type()!=CV_16UC1||gt.cols!=L.width||gt.rows!=L.height)
                throw std::runtime_error("GT must be same-sized uint16 PNG with disparity x256");
            cv::Mat mask;
            if(argc==11){mask=cv::imread(argv[10],cv::IMREAD_GRAYSCALE);
                if(mask.empty()||mask.size()!=gt.size())
                    throw std::runtime_error("GT mask size mismatch");}
            compare_gt("integer WTA",integer,gt,mask);
            compare_gt("subpixel    ",sub,gt,mask);
        }
        std::cout<<"Output: "<<fs::absolute(out)<<'\n';
    } catch(const std::exception& e) {
        std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;
    }
    return 0;
}
