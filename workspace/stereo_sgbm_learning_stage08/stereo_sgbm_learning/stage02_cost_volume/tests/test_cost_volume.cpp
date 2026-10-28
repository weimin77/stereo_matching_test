#include "cost_volume.hpp"
#include <cassert>
#include <iostream>
#include <stdexcept>

static stereo::GrayImage create_left(int w, int h) {
    stereo::GrayImage L{w, h, std::vector<uint8_t>(static_cast<size_t>(w) * h)};
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            L.pixels[y*w+x] = static_cast<uint8_t>((x*37+y*17+x*y*13+x*x*3) % 251);
    return L;
}
static stereo::GrayImage shift_to_right_image(const stereo::GrayImage& L, int d) {
    stereo::GrayImage R{L.width, L.height,
                        std::vector<uint8_t>(static_cast<size_t>(L.width) * L.height, 0)};
    for (int y = 0; y < L.height; ++y)
        for (int x = 0; x < L.width-d; ++x)
            R.pixels[y*L.width+x] = L.pixels[y*L.width+x+d];
    return R;
}

int main() {
    // Test 1: 两种计算方法得到逐元素完全一致的 Cost Volume。
    const auto L = create_left(96, 64);
    const auto R = shift_to_right_image(L, 7);
    stereo::BMParams p{0, 16, 5};
    const auto naive = stereo::compute_cost_volume_sad_naive(L, R, p);
    const auto fast = stereo::compute_cost_volume_sad_integral(L, R, p);
    assert(naive.costs.size() == fast.costs.size());
    int mismatches = 0;
    for (size_t i = 0; i < naive.costs.size(); ++i)
        mismatches += naive.costs[i] != fast.costs[i];
    std::cout << "Naive vs Integral CostVolume mismatches: " << mismatches << '\n';
    assert(mismatches == 0);

    // Test 2: Cost Volume -> WTA 与 Stage01 BM 的所有像素结果完全相同。
    const auto bm = stereo::compute_bm_sad(L, R, p);
    const auto pred = stereo::winner_takes_all(fast);
    assert(bm.disparity == pred.disparity);

    // Test 3: 内部纹理区域恢复人为设置的真实视差 7px。
    int inspected = 0, wrong = 0;
    for (int y = 4; y < L.height-4; ++y)
        for (int x = 22; x < L.width-4; ++x) {
            ++inspected;
            wrong += pred.disparity[y*L.width+x] != 7;
        }
    std::cout << "Truth=7 on interior: tested=" << inspected
              << ", wrong=" << wrong << '\n';
    assert(wrong == 0);

    // Test 4: 搜索视差过大导致所有右窗口越界时，应返回无效而不是 0。
    const auto SmallL = create_left(9, 7);
    const auto SmallR = SmallL;
    const stereo::BMParams impossible{10, 2, 3};
    auto emptyVol = stereo::compute_cost_volume_sad_integral(SmallL, SmallR, impossible);
    for (int32_t c : emptyVol.costs) assert(c == stereo::kInvalidCost);
    auto emptyWta = stereo::winner_takes_all(emptyVol);
    for (auto v : emptyWta.disparity) assert(v == -1);

    // Test 5: 平坦图像各视差代价相同，WTA 选第一个有效视差。
    stereo::GrayImage FlatL{32, 20, std::vector<uint8_t>(32*20, 100)};
    auto FlatR = FlatL;
    stereo::BMParams flatP{2, 4, 3};
    auto flatVol = stereo::compute_cost_volume_sad_integral(FlatL, FlatR, flatP);
    auto flatWta = stereo::winner_takes_all(flatVol);
    assert(flatWta.disparity[10*32+15] == 2);

    std::cout << "ALL STAGE 02 TESTS PASS\n";
}
