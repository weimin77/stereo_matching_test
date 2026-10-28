#include "sgm_eight_directions.hpp"
#include "sgm_four_directions.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

using stereo::CostVolume;
using stereo::Direction8;

static void check(bool ok, const char* why) {
    if (!ok) throw std::runtime_error(why);
}

static CostVolume make_volume(int w, int h, int min_d, int d,
                              std::vector<int32_t> values) {
    CostVolume C;
    C.width = w;
    C.height = h;
    C.min_disparity = min_d;
    C.num_disparities = d;
    C.costs = std::move(values);
    return C;
}

static void check_vector(const std::vector<int32_t>& got,
                         const std::vector<int32_t>& expected,
                         const char* why) {
    if (got == expected) return;
    std::cerr << why << "\nGot: ";
    for (auto v : got) std::cerr << v << ' ';
    std::cerr << "\nExpected: ";
    for (auto v : expected) std::cerr << v << ' ';
    std::cerr << "\n";
    throw std::runtime_error(why);
}

static constexpr std::array<Direction8, 8> directions = {
    Direction8::LeftToRight, Direction8::RightToLeft,
    Direction8::TopToBottom, Direction8::BottomToTop,
    Direction8::TopLeftToBottomRight, Direction8::BottomRightToTopLeft,
    Direction8::TopRightToBottomLeft, Direction8::BottomLeftToTopRight
};

// 验证：每个方向每个像素恰好访问一次；路径数量符合边界数学规律。
static void test_scan_coverage_and_predecessor() {
    constexpr int W = 5, H = 4, D = 3;
    auto C = make_volume(W, H, 0, D,
                         std::vector<int32_t>(W * H * D, 5));
    for (const auto r : directions) {
        const auto [dx, dy] = stereo::direction_vector(r);
        std::vector<int> visit_count(W * H, 0);
        int starts = 0;
        int total = 0;
        stereo::scan_direction8(C, {2, 10}, r,
            [&](int y, int x, size_t base, const std::vector<int32_t>& current) {
                check(base == C.index(y, x, 0), "callback base index wrong");
                check(current.size() == D, "callback disparity count wrong");
                const size_t idx = static_cast<size_t>(y) * W + x;
                ++visit_count[idx];
                ++total;
                if (!stereo::in_image(x - dx, y - dy, W, H)) ++starts;
            });
        check(total == W * H, "some pixels are missed or visited twice");
        check(std::all_of(visit_count.begin(), visit_count.end(),
                          [](int v) { return v == 1; }),
              "each pixel must be visited exactly once");
        const int expected_paths = dy == 0 ? H : (dx == 0 ? W : W + H - 1);
        check(starts == expected_paths, "path start count is incorrect");
    }
    std::cout << "[PASS] eight directions cover each pixel once; diagonal starts=W+H-1\n";
}

// 独立计算的 3x2x3 数字算例：八方向聚合；也是 Stage04 同一套算例。
static void test_small_hand_example() {
    const auto C = make_volume(3, 2, 10, 3, {
        10, 0, 12,    0, 3, 4,    1, 5, 3,
         4, 0,  6,    2, 3, 5,    3, 1, 4
    });
    const stereo::SGMParams p{5, 20};
    const auto Ldiag = stereo::aggregate_one_direction8(
        C, p, Direction8::BottomRightToTopLeft);
    check_vector({Ldiag.at(0, 1, 0), Ldiag.at(0, 1, 1), Ldiag.at(0, 1, 2)},
                 {2, 3, 7}, "B pixel reverse diagonal should use F predecessor");
    const auto S = stereo::aggregate_eight_directions(C, p);
    check_vector(S.costs, {
        84, 6, 110,  11, 29, 50,  12, 41, 35,
        38, 3,  62,  27, 31, 59,  27, 15, 43
    }, "8-path S differs from hand example");
    const auto disp = stereo::winner_takes_all(S);
    check((disp.disparity == std::vector<int16_t>{11, 10, 10, 11, 10, 11}),
          "hand example WTA incorrect");
    std::cout << "[PASS] hand example diagonal costs and full eight-direction S/WTA\n";
}

// 前四个方向应逐元素与旧 Stage04 完全一致。
static void test_compatibility_and_streaming_sum() {
    std::vector<int32_t> raw;
    for (int y = 0; y < 5; ++y)
        for (int x = 0; x < 6; ++x)
            for (int d = 0; d < 4; ++d)
                raw.push_back((x * 17 + y * 11 + d * 7 + x * y * 5) % 29);
    const auto C = make_volume(6, 5, 20, 4, raw);
    const stereo::SGMParams p{3, 11};

    const std::array<stereo::Direction, 4> directions4 = {
        stereo::Direction::LeftToRight, stereo::Direction::RightToLeft,
        stereo::Direction::TopToBottom, stereo::Direction::BottomToTop
    };
    for (size_t i = 0; i < directions4.size(); ++i) {
        const auto old_L = stereo::aggregate_one_direction(C, p, directions4[i]);
        const auto new_L = stereo::aggregate_one_direction8(C, p, directions[i]);
        check(old_L.costs == new_L.costs, "new 8-dir scanner mismatches Stage04 cardinal direction");
    }

    std::vector<int64_t> independent(C.costs.size(), 0);
    for (const auto r : directions) {
        const auto L = stereo::aggregate_one_direction8(C, p, r);
        for (size_t i = 0; i < C.costs.size(); ++i) independent[i] += L.costs[i];
    }
    const auto S = stereo::aggregate_eight_directions(C, p);
    for (size_t i = 0; i < C.costs.size(); ++i)
        check(S.costs[i] == independent[i], "streamed sum != independent 8 full volumes");
    check(C.costs == raw, "input CostVolume should not change");
    std::cout << "[PASS] Stage04-compatible cardinal scans; 8-path streaming == explicit sum\n";
}

// 对形状、无效值以及 P1=P2=0 的不变量进行测试。
static void test_boundaries_and_invalid() {
    const int X = stereo::kInvalidCost;
    for (const auto [w, h] : {std::pair<int,int>{1, 6}, {6, 1}, {1, 1}, {3, 2}}) {
        const int D = 4;
        auto C = make_volume(w, h, 0, D,
                             std::vector<int32_t>(static_cast<size_t>(w) * h * D));
        for (size_t i = 0; i < C.costs.size(); ++i)
            C.costs[i] = (i % 13 == 0 ? X : static_cast<int32_t>(i % 9));
        const auto S = stereo::aggregate_eight_directions(C, {0, 0});
        for (size_t i = 0; i < C.costs.size(); ++i) {
            if (C.costs[i] == X)
                check(S.costs[i] == X, "invalid disparity should stay invalid");
            else
                check(S.costs[i] == 8 * C.costs[i], "P1=P2=0 must imply S=8*C");
        }
    }
    std::cout << "[PASS] 1xN / Nx1 / 1x1 boundaries; invalid candidates; P1=P2=0\n";
}

static void test_synthetic_stereo() {
    const int w = 96, h = 64, gt = 7;
    stereo::GrayImage left{w, h, std::vector<uint8_t>(w*h)};
    stereo::GrayImage right{w, h, std::vector<uint8_t>(w*h, 0)};
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            left.pixels[y*w+x] = static_cast<uint8_t>(
                (x*37 + y*17 + x*y*13 + x*x*3) % 251);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w-gt; ++x)
            right.pixels[y*w+x] = left.pixels[y*w+x+gt];

    const auto C = stereo::compute_cost_volume_sad_integral(left, right, {0,16,5});
    const auto S4 = stereo::aggregate_four_directions(C, {20,80});
    const auto S8 = stereo::aggregate_eight_directions(C, {20,80});
    const auto d4 = stereo::winner_takes_all(S4);
    const auto d8 = stereo::winner_takes_all(S8);
    int count = 0, wrong4 = 0, wrong8 = 0;
    for (int y = 4; y < h-4; ++y)
        for (int x = 24; x < w-4; ++x) {
            const size_t i = static_cast<size_t>(y)*w+x;
            ++count;
            wrong4 += (d4.disparity[i] != gt);
            wrong8 += (d8.disparity[i] != gt);
        }
    std::cout << "[INFO] synthetic 7px stereo: checked=" << count
              << ", wrong4=" << wrong4 << ", wrong8=" << wrong8 << '\n';
    check(wrong4 == 0 && wrong8 == 0, "synthetic known disparity test failed");
    std::cout << "[PASS] Stage02 SAD volume -> 4/8 direction SGM -> WTA\n";
}

int main() {
    try {
        test_scan_coverage_and_predecessor();
        test_small_hand_example();
        test_compatibility_and_streaming_sum();
        test_boundaries_and_invalid();
        test_synthetic_stereo();
        std::cout << "ALL STAGE 05 TESTS PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] " << e.what() << '\n';
        return 1;
    }
}
