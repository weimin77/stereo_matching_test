#include "sgm_single_path.hpp"
#ifdef NDEBUG
#undef NDEBUG  // Release 构建中仍然强制执行单元测试的断言
#endif
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <vector>

using stereo::CostVolume;
using stereo::kInvalidCost;
using stereo::SGMParams;

static CostVolume from_rows(int width, int height, int minDisp, int D,
                            const std::vector<int32_t>& values) {
    CostVolume v;
    v.width=width; v.height=height;
    v.min_disparity=minDisp; v.num_disparities=D;
    v.costs=values;
    return v;
}

static void test_numeric_recursion() {
    // 手算教材例子：C0=[10,0,12], C1=[0,3,4], C2=[1,5,3]
    // P1=5, P2=20; L0=[10,0,12], L1=[5,3,9], L2=[3,5,8]
    const CostVolume C = from_rows(3,1,10,3,{10,0,12, 0,3,4, 1,5,3});
    const CostVolume L = stereo::aggregate_left_to_right(C, {5,20});
    assert((L.costs == std::vector<int32_t>{10,0,12, 5,3,9, 3,5,8}));
    assert((C.costs == std::vector<int32_t>{10,0,12, 0,3,4, 1,5,3}));
    auto disp = stereo::winner_takes_all(L);
    assert((disp.disparity == std::vector<int16_t>{11,11,10}));
    std::cout << "[PASS] hand-calculated normalized recursion and WTA\n";
}

static void test_large_jump_uses_p2() {
    // 最便宜的前驱 di=0，当前 di=3 只能跨大视差跳变，付 P2 而非 3*P1。
    auto C = from_rows(2,1,0,4,{0,60,60,60, 50,60,60,0});
    auto L = stereo::aggregate_left_to_right(C,{5,12});
    assert(L.at(0,1,3) == 12);
    assert(L.at(0,1,0) == 50);
    std::cout << "[PASS] jump across >1 disparity uses P2\n";
}

static void test_invalid_states_and_row_reset() {
    const int X = kInvalidCost;
    // 2行、4像素、3视差。每行重置；完全无效列后，后面的有效列重新起路径。
    auto C = from_rows(4,2,7,3,{
        X,X,X, 4,X,X, X,X,X, X,2,X,
        3,1,2, 5,2,8, 6,3,1, 4,7,2
    });
    auto L = stereo::aggregate_left_to_right(C,{5,20});
    for (int d=0; d<3; ++d) assert(L.at(0,0,d)==X);
    assert(L.at(0,1,0)==4 && L.at(0,1,1)==X);
    for (int d=0; d<3; ++d) assert(L.at(0,2,d)==X);
    assert(L.at(0,3,1)==2);
    assert(L.at(1,0,0)==3 && L.at(1,0,1)==1 && L.at(1,0,2)==2);
    std::cout << "[PASS] invalid disparity remains invalid; each row resets\n";
}

static void test_shifted_images() {
    const int w=96,h=64,gt=7;
    stereo::GrayImage left{w,h,std::vector<uint8_t>(w*h)};
    stereo::GrayImage right{w,h,std::vector<uint8_t>(w*h,0)};
    for(int y=0;y<h;++y) for(int x=0;x<w;++x)
        left.pixels[y*w+x]=static_cast<uint8_t>((x*37+y*17+x*y*13+x*x*3)%251);
    for(int y=0;y<h;++y) for(int x=0;x<w-gt;++x)
        right.pixels[y*w+x]=left.pixels[y*w+x+gt];
    const stereo::BMParams bp{0,16,5};
    auto C=stereo::compute_cost_volume_sad_integral(left,right,bp);
    auto L=stereo::aggregate_left_to_right(C,{20,80});
    auto pred=stereo::winner_takes_all(L);
    int n=0,wrong=0;
    for(int y=4;y<h-4;++y) for(int x=24;x<w-4;++x) {
        ++n;
        wrong += pred.disparity[y*w+x]!=gt;
    }
    std::cout<<"[INFO] synthetic shift 7px: inspected="<<n<<", wrong="<<wrong<<"\n";
    assert(wrong==0);
    std::cout<<"[PASS] Stage02 SAD volume -> Stage03 1-path -> WTA\n";
}

static void test_invalid_config() {
    auto C=from_rows(1,1,0,2,{3,4});
    bool thrown=false;
    try { stereo::aggregate_left_to_right(C,{20,5}); }
    catch(const std::invalid_argument&) { thrown=true; }
    assert(thrown);
}


static void test_zero_penalties() {
    // P1=P2=0 时，从任意有效前驱转移都是不计惩罚的；归一化后 L=C。
    const auto C=from_rows(4,1,0,3,{3,1,7, 6,2,4, 5,8,1, 4,6,9});
    const auto L=stereo::aggregate_left_to_right(C,{0,0});
    assert(C.costs==L.costs);
    std::cout<<"[PASS] P1=P2=0 implies L=C for fully valid path\n";
}

static void test_normalized_cost_vs_raw_dp() {
    // 不做归一化的数学 DP 可当作校验基准。归一化改变每个像素全部视差的公共偏移，
    // 但对同一像素，所有候选视差两两的代价差必须与未归一化 DP 完全一致。
    const int W=5,D=4;
    const auto C=from_rows(W,1,0,D,{
        17,9,30,23,
        12,22,6,14,
        5,17,10,26,
        11,19,1,27,
        12,0,15,7
    });
    const SGMParams p{4,13};
    const auto L=stereo::aggregate_left_to_right(C,p);
    std::vector<int64_t> prev(D),next(D);
    for(int d=0;d<D;++d) prev[d]=C.at(0,0,d);
    for(int x=0;x<W;++x) {
        if(x>0) {
            int64_t prevMin=*std::min_element(prev.begin(),prev.end());
            for(int d=0;d<D;++d) {
                int64_t best=std::min(prev[d],prevMin+p.P2);
                if(d>0) best=std::min(best,prev[d-1]+p.P1);
                if(d+1<D) best=std::min(best,prev[d+1]+p.P1);
                next[d]=C.at(0,x,d)+best;
            }
            prev.swap(next);
        }
        for(int d=0;d<D;++d)
            assert(prev[d]-prev[0]==L.at(0,x,d)-L.at(0,x,0));
    }
    std::cout<<"[PASS] normalized DP differs from raw DP only by per-pixel constant\n";
}

int main() {
    test_numeric_recursion();
    test_large_jump_uses_p2();
    test_invalid_states_and_row_reset();
    test_shifted_images();
    test_invalid_config();
    test_zero_penalties();
    test_normalized_cost_vs_raw_dp();
    std::cout<<"ALL STAGE 03 TESTS PASS\n";
}
