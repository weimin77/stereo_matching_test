#pragma once

#include "handmade_bm.hpp"  // 复用 Stage 01 中的 GrayImage / BMParams / BMResult
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace stereo {

// 大惩罚值：防止越界候选视差被 WTA 当作有效匹配。
constexpr int32_t kInvalidCost = std::numeric_limits<int32_t>::max() / 4;

// 内存布局： [H][W][D]，每个像素的全部候选视差相邻。
struct CostVolume {
    int width = 0;
    int height = 0;
    int min_disparity = 0;
    int num_disparities = 0;
    std::vector<int32_t> costs;

    size_t index(int y, int x, int di) const {
        return (static_cast<size_t>(y) * width + x) * num_disparities + di;
    }
    int32_t& at(int y, int x, int di) { return costs[index(y, x, di)]; }
    int32_t at(int y, int x, int di) const { return costs[index(y, x, di)]; }
    int actual_disparity(int di) const { return min_disparity + di; }
};

inline void validate_inputs(const GrayImage& L, const GrayImage& R,
                            const BMParams& p) {
    if (L.width <= 0 || L.height <= 0 || L.width != R.width ||
        L.height != R.height ||
        L.pixels.size() != static_cast<size_t>(L.width) * L.height ||
        R.pixels.size() != L.pixels.size())
        throw std::invalid_argument("invalid stereo image shape");
    if (p.min_disparity < 0 || p.num_disparities <= 0 ||
        static_cast<int64_t>(p.min_disparity) + p.num_disparities - 1 > 32767 ||
        p.block_size <= 0 || !(p.block_size & 1) ||
        p.block_size > L.width || p.block_size > L.height ||
        255LL * p.block_size * p.block_size >= kInvalidCost)
        throw std::invalid_argument("bad disparity or block size");
}

inline CostVolume make_volume(const GrayImage& L, const BMParams& p) {
    CostVolume cv;
    cv.width = L.width;
    cv.height = L.height;
    cv.min_disparity = p.min_disparity;
    cv.num_disparities = p.num_disparities;
    cv.costs.assign(static_cast<size_t>(L.width) * L.height * p.num_disparities,
                    kInvalidCost);
    return cv;
}

// 检查以 (x,y) 和 (x-d,y) 为中心的左右两个方形窗口是否同时在图内。
inline bool valid_window(int x, int y, int d, int radius, int w, int h) {
    const int xr = x - d;
    return x - radius >= 0 && x + radius < w &&
           y - radius >= 0 && y + radius < h &&
           xr - radius >= 0 && xr + radius < w;
}

// 参考版本：直接用两个窗口循环累加 SAD；复杂度 O(W*H*D*B^2)。
inline CostVolume compute_cost_volume_sad_naive(
    const GrayImage& L, const GrayImage& R, const BMParams& p) {
    validate_inputs(L, R, p);
    CostVolume vol = make_volume(L, p);
    const int r = p.block_size / 2;
    for (int y = 0; y < L.height; ++y)
        for (int x = 0; x < L.width; ++x)
            for (int di = 0; di < p.num_disparities; ++di) {
                const int d = vol.actual_disparity(di);
                if (!valid_window(x, y, d, r, L.width, L.height)) continue;
                int32_t sad = 0;
                for (int dy = -r; dy <= r; ++dy)
                    for (int dx = -r; dx <= r; ++dx) {
                        const int a = L.at(y + dy, x + dx);
                        const int b = R.at(y + dy, x + dx - d);
                        sad += a > b ? a - b : b - a;
                    }
                vol.at(y, x, di) = sad;
            }
    return vol;
}

// 加速版本：每个视差先计算 abs-diff 图，再建立 SAT（积分图）。
// 一个 BxB 窗口 SAD 可以 O(1) 查询，整体 O(W*H*D)。
inline CostVolume compute_cost_volume_sad_integral(
    const GrayImage& L, const GrayImage& R, const BMParams& p) {
    validate_inputs(L, R, p);
    CostVolume vol = make_volume(L, p);
    const int w = L.width, h = L.height;
    const int r = p.block_size / 2;
    const size_t stride = static_cast<size_t>(w) + 1;
    std::vector<int64_t> sat(static_cast<size_t>(w + 1) * (h + 1));
    const auto idx = [stride](int y, int x) {
        return static_cast<size_t>(y) * stride + x;
    };

    // 更换一个候选视差，右图整体水平移位，SAD 差值图也跟着变化。
    for (int di = 0; di < p.num_disparities; ++di) {
        const int d = vol.actual_disparity(di);
        std::fill(sat.begin(), sat.end(), 0);

        // 构建 abs-diff 的积分图；越界点写 0，但不允许查询包含越界点的窗口。
        for (int y = 1; y <= h; ++y) {
            for (int x = 1; x <= w; ++x) {
                const int xr = (x - 1) - d;
                int32_t diff = 0;
                if (xr >= 0 && xr < w) {
                    const int a = L.at(y - 1, x - 1);
                    const int b = R.at(y - 1, xr);
                    diff = a > b ? a - b : b - a;
                }
                sat[idx(y, x)] = diff + sat[idx(y - 1, x)]
                                    + sat[idx(y, x - 1)] - sat[idx(y - 1, x - 1)];
            }
        }

        // 将 BxB 子矩形和写入该视差的代价切片。
        for (int y = r; y < h - r; ++y) {
            for (int x = r; x < w - r; ++x) {
                if (!valid_window(x, y, d, r, w, h)) continue;
                const int x0 = x - r, x1 = x + r + 1;
                const int y0 = y - r, y1 = y + r + 1;
                const int64_t sum = sat[idx(y1, x1)] - sat[idx(y0, x1)]
                                  - sat[idx(y1, x0)] + sat[idx(y0, x0)];
                vol.at(y, x, di) = static_cast<int32_t>(sum);
            }
        }
    }
    return vol;
}

// 依旧执行最简单的 WTA；目的是确认代价体计算正确。
inline BMResult winner_takes_all(const CostVolume& vol) {
    BMResult out{vol.width, vol.height,
                 std::vector<int16_t>(static_cast<size_t>(vol.width) * vol.height, -1)};
    for (int y = 0; y < vol.height; ++y)
        for (int x = 0; x < vol.width; ++x) {
            int32_t best = kInvalidCost;
            int bestD = -1;
            for (int di = 0; di < vol.num_disparities; ++di) {
                const int32_t c = vol.at(y, x, di);
                if (c < best) {
                    best = c;
                    bestD = vol.actual_disparity(di);
                }
            }
            if (bestD >= 0)
                out.disparity[static_cast<size_t>(y) * vol.width + x] =
                    static_cast<int16_t>(bestD);
        }
    return out;
}

} // namespace stereo
