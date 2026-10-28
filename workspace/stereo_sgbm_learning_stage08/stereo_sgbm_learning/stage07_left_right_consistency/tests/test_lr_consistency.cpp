#include "lr_consistency.hpp"
// Keep assertions active even for cmake -DCMAKE_BUILD_TYPE=Release.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace stereo;

static SubpixelResult empty_map(int width, int height) {
    SubpixelResult out;
    out.width = width;
    out.height = height;
    const size_t n = static_cast<size_t>(width) * height;
    out.disparity.assign(n, std::numeric_limits<float>::quiet_NaN());
    out.valid.assign(n, 0);
    out.refined.assign(n, 0);
    return out;
}

static void set(SubpixelResult& out, int x, int y, float d, bool refined = false) {
    const size_t i = static_cast<size_t>(y) * out.width + x;
    out.disparity[i] = d;
    out.valid[i] = 1;
    out.refined[i] = refined;
}

static void test_hand_lrc() {
    auto L = empty_map(10, 1);
    auto R = empty_map(10, 1);
    set(L, 1, 0, 2.0f);     // x_right=-1, outside
    set(L, 5, 0, 2.25f, true); // x_right=2.75 => nearest 3
    set(R, 3, 0, 2.1f);     // |2.25-2.1|=0.15 => consistent at 0.3
    set(L, 7, 0, 2.0f);    // x_right=5, R(5)=4 => mismatch
    set(R, 5, 0, 4.0f);
    set(L, 9, 0, 2.0f);    // x_right=7, right invalid
    const auto out = left_right_consistency_check(L, R, LRParams{0.3f});
    assert(out.status[0] == LRStatus::LeftInvalid);
    assert(out.status[1] == LRStatus::RightOutside);
    assert(out.status[5] == LRStatus::Consistent);
    assert(out.status[7] == LRStatus::Disagree);
    assert(out.status[9] == LRStatus::RightInvalid);
    assert(out.filtered.valid[5] == 1);
    assert(out.filtered.refined[5] == 1);
    assert(std::abs(out.filtered.disparity[5] - 2.25f) < 1e-6f);
    for (int x : {0,1,7,9}) {
        assert(out.filtered.valid[x] == 0);
        assert(std::isnan(out.filtered.disparity[x]));
        assert(out.filtered.refined[x] == 0);
    }
    // At the paper default threshold 1.0, the same consistent sample passes.
    assert(left_right_consistency_check(L, R).status[5] == LRStatus::Consistent);
    std::cout << "[PASS] hand LRC: valid/outside/right-invalid/disagree/left-invalid\n";
}

static void test_geometry_and_flip() {
    // A unique non-symmetric image checks right matching +d rather than -d.
    constexpr int W = 55, H = 22, shift = 7;
    GrayImage left{W, H, std::vector<uint8_t>(W * H)};
    GrayImage right = left;
    uint32_t seed = 0x12345678u;
    for (int y = 0; y < H; ++y)for(int x = 0; x < W; ++x) {
        seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
        left.pixels[static_cast<size_t>(y) * W + x] = static_cast<uint8_t>(seed & 255);
    }
    for(int y = 0; y < H; ++y)for(int x = 0; x < W; ++x) {
        const int src_x = x + shift;
        right.pixels[static_cast<size_t>(y) * W + x] =
            src_x < W ? left.pixels[static_cast<size_t>(y) * W + src_x] : 0;
    }
    const auto twice = flip_horizontal(flip_horizontal(left));
    assert(twice.pixels == left.pixels);
    const BMParams bm{0, 13, 3};
    const SGMParams sg{5, 25};
    const auto Cl = compute_cost_volume_sad_integral(left, right, bm);
    const auto Cright_mirror = compute_cost_volume_sad_integral(
        flip_horizontal(right), flip_horizontal(left), bm);

    // Pixel in right view at xR=19 corresponds to original left xL=26
    // and flipped right-base uR=W-1-xR.
    const int xr = 19, y = 9, di = shift;
    const int u = W - 1 - xr;
    assert(Cl.at(y, xr+shift, di) == 0);
    assert(Cright_mirror.at(y, u, di) == 0);
    // Check the reverse cost is exactly a direct right-centered window SAD.
    int direct_sad = 0;
    for (int oy=-1; oy<=1; ++oy) for (int ox=-1; ox<=1; ++ox) {
        const int rv = right.at(y+oy, xr+ox);
        const int lv = left.at(y+oy, xr+ox+shift);
        direct_sad += std::abs(rv-lv);
    }
    assert(direct_sad == Cright_mirror.at(y,u,di));

    const auto Sl = aggregate_eight_directions(Cl,sg);
    const auto Dl = refine_subpixel(Sl);
    const auto Dr = compute_right_disparity_flipped(left,right,bm,sg);
    const auto checked = left_right_consistency_check(Dl,Dr,LRParams{1.0f});

    int inspected=0, wrong=0, passed=0, right_correct=0;
    for(int yy=3; yy<H-3; ++yy)for(int xx=shift+4; xx<W-4; ++xx) {
        const size_t i=static_cast<size_t>(yy)*W+xx;
        if(!Dl.valid[i]) continue;
        ++inspected;
        wrong += (std::abs(Dl.disparity[i] - shift) > 0.9f);
        passed += checked.filtered.valid[i];
        const size_t j=static_cast<size_t>(yy)*W+(xx-shift);
        if(Dr.valid[j] && std::abs(Dr.disparity[j]-shift)<0.9f)++right_correct;
    }
    std::cout << "[INFO] synthetic disparity=" << shift << ": inspected=" << inspected
              << " wrong_left=" << wrong << " passed_LRC=" << passed
              << " correct_right=" << right_correct << "\n";
    assert(inspected > 400);
    assert(wrong == 0);
    assert(passed == inspected);
    assert(right_correct == inspected);
    std::cout << "[PASS] flipped right matching + full stage02->05->06->07 pipeline\n";
}

static void test_bad_input() {
    auto L = empty_map(5,1), R = empty_map(4,1);
    bool thrown = false;
    try { (void)left_right_consistency_check(L,R); }
    catch(const std::invalid_argument&) {thrown=true;}
    assert(thrown);
    R = empty_map(5,1);
    thrown = false;
    try { (void)left_right_consistency_check(L,R,LRParams{-1.0f}); }
    catch(const std::invalid_argument&) {thrown=true;}
    assert(thrown);
    // qFloat outside before rounding: x=0, d=.25 must be invalid.
    set(L,0,0,.25f);
    set(R,0,0,.25f);
    assert(left_right_consistency_check(L,R).status[0] == LRStatus::RightOutside);
    std::cout << "[PASS] invalid inputs and fractional out-of-bounds\n";
}

int main() {
    test_hand_lrc();
    test_geometry_and_flip();
    test_bad_input();
    std::cout << "ALL STAGE 07 TESTS PASS\n";
}
