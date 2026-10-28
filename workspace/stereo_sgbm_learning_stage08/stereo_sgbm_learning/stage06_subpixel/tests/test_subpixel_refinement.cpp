#include "subpixel_refinement.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cstdint>

using stereo::CostVolume;

static void check(bool ok, const char* msg) {
    if (!ok) throw std::runtime_error(msg);
}
static void near(float got, float expected, float eps, const char* why) {
    if (!(std::abs(got - expected) < eps)) {
        std::cerr << why << ": got=" << got << ", expected=" << expected << '\n';
        throw std::runtime_error(why);
    }
}
static CostVolume one_pixel(int min_disparity, std::vector<int32_t> costs) {
    CostVolume S;
    S.width=1; S.height=1;
    S.min_disparity=min_disparity;
    S.num_disparities=static_cast<int>(costs.size());
    S.costs=std::move(costs);
    return S;
}

static void test_hand_calculated_plus_quarter() {
    const auto S = one_pixel(68, {45,25,10,15,35});
    const auto out = stereo::refine_subpixel(S);
    near(out.disparity[0],70.25f,1e-6f,"hand-calculated parabola should yield d=70.25");
    check(out.valid[0] == 1 && out.refined[0] == 1,"quadratic fitting flags");
    const auto integer = stereo::winner_takes_all(S);
    check(integer.disparity[0]==70,"integer WTA should be 70");
    std::cout << "[PASS] hand example: WTA 70 -> parabola 70.25\n";
}

static void test_sign_and_offset() {
    const auto S = one_pixel(68, {45,15,10,25,35});
    const auto out=stereo::refine_subpixel(S);
    near(out.disparity[0],69.75f,1e-6f,"negative subpixel offset must be -0.25");
    const auto T=one_pixel(65,{100,18,1,4,27});
    near(stereo::refine_subpixel(T).disparity[0],67.35f,1e-5f,
         "asymmetric parabola should yield +0.35 (include minDisparity)");
    std::cout << "[PASS] subpixel negative shift and nonzero minDisparity\n";
}

static void test_boundaries_and_invalids() {
    const int32_t INF=stereo::kInvalidCost;
    near(stereo::refine_subpixel(one_pixel(70,{2,3,9})).disparity[0],70.0f,1e-6f,
         "left boundary falls back to integer");
    near(stereo::refine_subpixel(one_pixel(70,{9,3,2})).disparity[0],72.0f,1e-6f,
         "right boundary falls back to integer");
    near(stereo::refine_subpixel(one_pixel(70,{INF,2,INF})).disparity[0],71.0f,1e-6f,
         "invalid neighbor should fall back to integer");
    const auto no_candidate=stereo::refine_subpixel(one_pixel(70,{INF,INF,INF}));
    check(no_candidate.valid[0]==0 && std::isnan(no_candidate.disparity[0]),
          "all invalid -> NaN and valid mask=0");
    const auto sym=stereo::refine_subpixel(one_pixel(70,{20,10,3,10,20}));
    near(sym.disparity[0],72.0f,1e-6f,"symmetric parabola must produce zero delta");
    check(sym.refined[0]==0,"zero delta should not be marked changed");
    const auto clamp=stereo::refine_subpixel(one_pixel(70,{20,10,10,20}));
    near(clamp.disparity[0],71.5f,1e-6f,"tie & 0.5 case consistent with first min");
    std::cout << "[PASS] integer fallbacks, masks, invalid values, symmetric curve and tie\n";
}

static void test_curvature_gate() {
    auto S=one_pixel(68,{45,25,10,15,35});
    const auto out=stereo::refine_subpixel(S,{1000.0});
    near(out.disparity[0],70.0f,1e-6f,"small curvature should fall back to integer");
    check(out.refined[0]==0,"fall-back should not be marked refined");
    std::cout << "[PASS] curvature threshold disables unstable fitting\n";
}

static void test_scale_invariance() {
    auto C = one_pixel(68,{45,25,10,15,35});
    auto scaled = C;
    for (auto& v: scaled.costs) v *= 8;
    near(stereo::refine_subpixel(C).disparity[0],
         stereo::refine_subpixel(scaled).disparity[0],1e-6f,
         "positive cost scaling should not move parabola vertex");
    std::cout << "[PASS] parabola vertex invariant under positive cost scaling\n";
}

static void test_full_pipeline_and_integer_agreement() {
    const int W=96, H=64, shift=7;
    stereo::GrayImage left{W,H,std::vector<uint8_t>(W*H)};
    stereo::GrayImage right{W,H,std::vector<uint8_t>(W*H,0)};
    for (int y=0; y<H; ++y)
        for (int x=0; x<W; ++x)
            left.pixels[y*W+x]=static_cast<uint8_t>((x*37+y*17+x*y*13+x*x*3)%251);
    for (int y=0;y<H;++y)
        for (int x=0;x<W-shift;++x)
            right.pixels[y*W+x]=left.pixels[y*W+x+shift];
    const auto C=stereo::compute_cost_volume_sad_integral(left,right,{0,16,5});
    const auto S=stereo::aggregate_eight_directions(C,{20,80});
    const auto integer=stereo::winner_takes_all(S);
    const auto sub=stereo::refine_subpixel(S);
    int count=0,wrong=0;
    for (int y=4;y<H-4;++y)
        for (int x=24;x<W-4;++x) {
            const size_t i=static_cast<size_t>(y)*W+x;
            ++count;
            wrong += (!sub.valid[i] || sub.disparity[i]<shift-0.5f ||
                      sub.disparity[i]>shift+0.5f ||integer.disparity[i]!=shift);
        }
    std::cout << "[INFO] full SAD->8path->subpixel test pixels="<<count<<", wrong="<<wrong<<"\n";
    check(wrong==0,"synthetic shift 7 failed");
    std::cout << "[PASS] Stage02 -> Stage05 -> Stage06 complete pipeline\n";
}

int main() {
    try {
        test_hand_calculated_plus_quarter();
        test_sign_and_offset();
        test_boundaries_and_invalids();
        test_curvature_gate();
        test_scale_invariance();
        test_full_pipeline_and_integer_agreement();
        std::cout << "ALL STAGE 06 TESTS PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] " << e.what() << '\n';
        return 1;
    }
}
