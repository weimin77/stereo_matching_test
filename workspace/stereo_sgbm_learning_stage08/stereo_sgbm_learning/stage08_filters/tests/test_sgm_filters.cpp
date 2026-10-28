#include "sgm_filters.hpp"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace stereo;

static SubpixelResult blank_map(int W,int H) {
    SubpixelResult m;
    m.width=W; m.height=H;
    m.disparity.assign(static_cast<size_t>(W)*H,std::numeric_limits<float>::quiet_NaN());
    m.valid.assign(static_cast<size_t>(W)*H,0);
    m.refined.assign(static_cast<size_t>(W)*H,0);
    return m;
}
static void set_point(SubpixelResult& m,int x,int y,float d,bool refined=true) {
    const size_t i=static_cast<size_t>(y)*m.width+x;
    m.disparity[i]=d; m.valid[i]=1; m.refined[i]=refined;
}

static void test_uniqueness_hand() {
    CostVolume S{5,1,10,6,std::vector<int32_t>(5*6,kInvalidCost)};
    // p0: best at di0=100; competitor di2=150 => clearly unique.
    const int32_t p0[]={100,105,150,190,220,250};
    // p1: best di0=100; competitor di2=105 => ambiguity at 10%.
    const int32_t p1[]={100,101,105,110,130,180};
    // p2: di2=100, di3=101 (adjacent); best other di0/4=200 => unique.
    const int32_t p2[]={200,150,100,101,200,300};
    // p3: no valid disparity, entire row INF.
    // p4: tied zero at distant candidates; rejected.
    const int32_t p4[]={0,1,0,50,60,70};
    for(int di=0;di<6;++di) {
        S.at(0,0,di)=p0[di]; S.at(0,1,di)=p1[di];
        S.at(0,2,di)=p2[di]; S.at(0,4,di)=p4[di];
    }
    const auto d=refine_subpixel(S);
    const auto u=uniqueness_filter(S,d,UniquenessParams{10,1});
    assert(u.status[0]==UniqueStatus::Unique && u.filtered.valid[0]);
    assert(u.status[1]==UniqueStatus::Ambiguous && !u.filtered.valid[1]);
    assert(std::isnan(u.filtered.disparity[1]) && !u.filtered.refined[1]);
    assert(u.status[2]==UniqueStatus::Unique && u.filtered.valid[2]);
    assert(u.status[3]==UniqueStatus::InputInvalid && !u.filtered.valid[3]);
    assert(u.status[4]==UniqueStatus::Ambiguous && !u.filtered.valid[4]);
    assert(u.best_cost[2]==100 && u.other_cost[2]==200);
    const auto disabled=uniqueness_filter(S,d,UniquenessParams{0,1});
    assert(disabled.filtered.valid[1] && disabled.filtered.valid[4]);

    // When the only other candidate is adjacent, report no nonlocal competitor.
    CostVolume short_s{1,1,0,2,{10,11}};
    const auto small = uniqueness_filter(short_s, refine_subpixel(short_s));
    assert(small.status[0]==UniqueStatus::NoNonlocalCompetitor);
    assert(small.filtered.valid[0]);

    std::cout<<"[PASS] uniqueness: nonlocal rival, neighbor exclusion, zero tie, invalid, disabled\n";
}

static void test_speckle_components() {
    constexpr int W=8,H=6;
    auto m=blank_map(W,H);
    for(int y=0;y<H;++y)for(int x=0;x<W;++x)set_point(m,x,y,70.0f);
    // Four-pixel island that is NOT 4-connected to background (disparity jump).
    for(int y=2;y<=3;++y)for(int x=3;x<=4;++x)set_point(m,x,y,95.0f);
    const auto o=remove_small_speckles(m,SpeckleParams{4,1.0f});
    assert(o.components_total==2 && o.components_removed==1 && o.pixels_removed==4);
    for(int y=2;y<=3;++y)for(int x=3;x<=4;++x) {
        const size_t i=static_cast<size_t>(y)*W+x;
        assert(!o.filtered.valid[i] && std::isnan(o.filtered.disparity[i]));
        assert(!o.filtered.refined[i] && o.removed[i]);
    }
    assert(o.filtered.valid[0] && !o.removed[0]);
    const auto kept=remove_small_speckles(m,SpeckleParams{3,1.0f});
    assert(kept.pixels_removed==0 && kept.filtered.valid[3+2*W]);
    const auto no_filter=remove_small_speckles(m,SpeckleParams{0,0});
    assert(no_filter.pixels_removed==0 && no_filter.filtered.valid[3+2*W]);
    std::cout<<"[PASS] speckle: 4-connected island removal, size boundary, disable\n";
}

static void test_edge_difference_and_diagonal() {
    // Chained local differences, even if end-to-end range exceeds threshold.
    auto chain=blank_map(4,1);
    for(int x=0;x<4;++x)set_point(chain,x,0,70.0f+0.8f*x);
    auto r=remove_small_speckles(chain,SpeckleParams{3,1.0f});
    assert(r.components_total==1 && r.pixels_removed==0);
    r=remove_small_speckles(chain,SpeckleParams{3,0.5f});
    assert(r.components_total==4 && r.pixels_removed==4);

    // Diagonal-only contact is NOT connected with 4-neighbor rule.
    auto diag=blank_map(3,3);
    set_point(diag,0,0,10.0f);
    set_point(diag,1,1,10.0f);
    r=remove_small_speckles(diag,SpeckleParams{1,1.0f});
    assert(r.components_total==2 && r.pixels_removed==2);
    std::cout<<"[PASS] speckle: chained neighbor rule and 4-connectivity\n";
}

static void test_integration() {
    constexpr int W=55,H=22,shift=7;
    GrayImage L{W,H,std::vector<uint8_t>(W*H)},R=L;
    uint32_t seed=0x12345678u;
    for(int y=0;y<H;++y)for(int x=0;x<W;++x) {
        seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;
        L.pixels[static_cast<size_t>(y)*W+x]=static_cast<uint8_t>(seed&255);
    }
    for(int y=0;y<H;++y)for(int x=0;x<W;++x) {
        R.pixels[static_cast<size_t>(y)*W+x] = x+shift<W?
            L.pixels[static_cast<size_t>(y)*W+x+shift]:0;
    }
    const BMParams bp{0,13,3};
    const SGMParams sp{5,25};
    const auto C=compute_cost_volume_sad_integral(L,R,bp);
    const auto S=aggregate_eight_directions(C,sp);
    const auto d=refine_subpixel(S);
    const auto unique=uniqueness_filter(S,d,UniquenessParams{10,1});
    const auto right=compute_right_disparity_flipped(L,R,bp,sp);
    const auto lr=left_right_consistency_check(unique.filtered,right,LRParams{1.0f});
    const auto speckle=remove_small_speckles(lr.filtered,SpeckleParams{4,1.0f});
    int evaluated=0,wrong=0;
    for(int y=3;y<H-3;++y)for(int x=shift+4;x<W-4;++x) {
        const size_t i=static_cast<size_t>(y)*W+x;
        if(!speckle.filtered.valid[i])continue;
        ++evaluated;
        wrong+=(std::abs(speckle.filtered.disparity[i]-shift)>0.9f);
    }
    assert(evaluated>300 && wrong==0);
    std::cout<<"[PASS] SAD -> 8-path -> subpixel -> uniqueness -> LRC -> speckle; "
             <<"valid_center="<<evaluated<<", wrong="<<wrong<<"\n";
}

static void test_invalid_params() {
    CostVolume S{1,1,0,3,{3,1,4}};
    auto d=refine_subpixel(S);
    bool threw=false;
    try { (void)uniqueness_filter(S,d,UniquenessParams{100,1}); }
    catch (const std::invalid_argument&) {threw=true;}
    assert(threw);
    threw=false;
    try {(void)remove_small_speckles(d,SpeckleParams{5,-1});}
    catch (const std::invalid_argument&) {threw=true;}
    assert(threw);
    std::cout<<"[PASS] parameter validation\n";
}

int main() {
    test_uniqueness_hand();
    test_speckle_components();
    test_edge_difference_and_diagonal();
    test_integration();
    test_invalid_params();
    std::cout<<"ALL STAGE 08 TESTS PASS\n";
}
