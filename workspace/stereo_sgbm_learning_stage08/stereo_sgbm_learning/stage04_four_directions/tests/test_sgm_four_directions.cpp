#include "sgm_four_directions.hpp"
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using stereo::CostVolume;
using stereo::Direction;

static void check(bool condition, const char* description) {
    if (!condition) throw std::runtime_error(description);
}

static CostVolume volume(int w, int h, int min_disp, int d,
                         std::vector<int32_t> costs) {
    CostVolume C;
    C.width=w; C.height=h; C.min_disparity=min_disp; C.num_disparities=d;
    C.costs=std::move(costs);
    return C;
}

static void check_vectors(const std::vector<int32_t>& got,
                          const std::vector<int32_t>& expected,
                          const char* description) {
    if (got != expected) {
        std::cerr << description << " failed.\nGot: ";
        for (auto v:got) std::cerr << v << " ";
        std::cerr << "\nExpected: ";
        for (auto v:expected) std::cerr << v << " ";
        std::cerr << "\n";
        throw std::runtime_error(description);
    }
}

static void test_hand_calculated_four_paths() {
    // H=2, W=3, D=3. Same 1st row as Stage03's hand example.
    // Each of 4 path directions is checked independently.
    const auto C = volume(3,2,10,3,{
        10,0,12,  0,3,4,  1,5,3,
         4,0,6,   2,3,5,  3,1,4
    });
    const auto p = stereo::SGMParams{5,20};
    const auto Ltr=stereo::aggregate_one_direction(C,p,Direction::LeftToRight);
    const auto Rtl=stereo::aggregate_one_direction(C,p,Direction::RightToLeft);
    const auto Ttb=stereo::aggregate_one_direction(C,p,Direction::TopToBottom);
    const auto Btt=stereo::aggregate_one_direction(C,p,Direction::BottomToTop);
    const auto Sum=stereo::aggregate_four_directions(C,p);
    check_vectors(Ltr.costs,{
        10,0,12, 5,3,9, 3,5,8,
        4,0,6, 6,3,10, 6,1,9
    },"left-right paths");
    check_vectors(Rtl.costs,{
        10,5,18, 0,7,6, 1,5,3,
        5,0,11, 4,3,8, 3,1,4
    },"right-left paths");
    check_vectors(Ttb.costs,{
        10,0,12, 0,3,4, 1,5,3,
        9,0,11, 2,6,9, 3,5,6
    },"top-bottom paths");
    check_vectors(Btt.costs,{
        14,0,17, 0,4,7, 3,5,6,
        4,0,6, 2,3,5, 3,1,4
    },"bottom-top paths");
    check_vectors(Sum.costs,{
        44,5,59, 5,17,26, 8,20,20,
        22,0,34, 14,15,32, 15,8,23
    },"4-direction Eq.14 sums");
    check_vectors(C.costs,{10,0,12,0,3,4,1,5,3,4,0,6,2,3,5,3,1,4},"original C intact");
    const auto disp=stereo::winner_takes_all(Sum);
    check((disp.disparity==std::vector<int16_t>{11,10,10,11,10,11}),"expected WTA after four directions");
    std::cout << "[PASS] handwritten C -> four directional L -> S -> WTA\n";
}

static void test_consistency_with_stage03_and_manual_sum() {
    const auto C=volume(4,3,1,4,{
        15,10,4,2, 5,8,4,10, 7,9,1,14, 4,5,8,1,
         2,15,11,4, 9,10,15,1, 6,9,8,14, 7,1,9,15,
        15,8,7,11, 9,3,1,12, 8,11,4,10, 2,4,15,3
    });
    const auto p=stereo::SGMParams{3,9};
    const auto stage03=stereo::aggregate_left_to_right(C,p);
    const auto ltr=stereo::aggregate_one_direction(C,p,Direction::LeftToRight);
    check(stage03.costs==ltr.costs,"Stage03 left-right != Stage04 left-right");
    std::vector<int64_t> independently_summed(C.costs.size(),0);
    for(auto dir:{Direction::LeftToRight,Direction::RightToLeft,
                  Direction::TopToBottom,Direction::BottomToTop}){
        const auto L=stereo::aggregate_one_direction(C,p,dir);
        for(size_t i=0;i<L.costs.size();++i) independently_summed[i]+=L.costs[i];
    }
    const auto sum=stereo::aggregate_four_directions(C,p);
    for(size_t i=0;i<sum.costs.size();++i)
        check(sum.costs[i]==independently_summed[i],"streamed sum != sum of 4 full L volumes");
    std::cout << "[PASS] Stage03 left-right == Stage04 left-right; streaming == explicit sum\n";
}

static void test_zero_penalties_and_invalid() {
    const int X=stereo::kInvalidCost;
    const auto C=volume(3,2,4,3,{
        5,2,X, 4,8,1, X,X,X,
        9,5,3, 1,X,2, 7,0,4
    });
    const auto S=stereo::aggregate_four_directions(C,{0,0});
    for(size_t i=0;i<S.costs.size();++i) {
        check(S.costs[i]==(C.costs[i]==X ? X : 4*C.costs[i]),
              "P1=P2=0 should yield S=4C, preserving invalid candidates");
    }
    std::cout << "[PASS] zero penalties: S=4C; invalid candidates stay invalid\n";
}

static void test_synthetic_stereo() {
    const int w=96,h=64,gt=7;
    stereo::GrayImage left{w,h,std::vector<uint8_t>(w*h)};
    stereo::GrayImage right{w,h,std::vector<uint8_t>(w*h,0)};
    for(int y=0;y<h;++y)for(int x=0;x<w;++x)
        left.pixels[y*w+x]=static_cast<uint8_t>((x*37+y*17+x*y*13+x*x*3)%251);
    for(int y=0;y<h;++y)for(int x=0;x<w-gt;++x)
        right.pixels[y*w+x]=left.pixels[y*w+x+gt];
    const auto C=stereo::compute_cost_volume_sad_integral(left,right,{0,16,5});
    const auto S=stereo::aggregate_four_directions(C,{20,80});
    const auto disp=stereo::winner_takes_all(S);
    int count=0,wrong=0;
    for(int y=4;y<h-4;++y)for(int x=24;x<w-4;++x){
        ++count;wrong+=disp.disparity[static_cast<size_t>(y)*w+x]!=gt;
    }
    std::cout<<"[INFO] 7-px synthetic stereo: tested="<<count<<", wrong="<<wrong<<"\n";
    check(wrong==0,"Four direction SGM did not recover known ground truth");
    std::cout<<"[PASS] CostVolume -> 4-path -> WTA synthetic image\n";
}

int main() {
    try {
        test_hand_calculated_four_paths();
        test_consistency_with_stage03_and_manual_sum();
        test_zero_penalties_and_invalid();
        test_synthetic_stereo();
        std::cout << "ALL STAGE 04 TESTS PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] " << e.what() << "\n";
        return 1;
    }
}
