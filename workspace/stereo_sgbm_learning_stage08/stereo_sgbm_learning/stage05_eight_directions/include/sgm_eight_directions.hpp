#pragma once

// Stage 05: SGM 八方向路径聚合。
// 论文 Hirschmueller (2008), Section 2.2, Eq. (13) + Eq. (14).
// 直接复用 Stage03 的 aggregate_one_step()，不修改它的 DP 递推。
// 与 Stage04 一样，所有方向的 L_r 在线累加到 S；额外路径状态仅 O(D)。
#include "sgm_single_path.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace stereo {

enum class Direction8 {
    LeftToRight,            //  r=(+1,  0)  →
    RightToLeft,            //  r=(-1,  0)  ←
    TopToBottom,            //  r=( 0, +1)  ↓
    BottomToTop,            //  r=( 0, -1)  ↑
    TopLeftToBottomRight,   //  r=(+1, +1)  ↘
    BottomRightToTopLeft,   //  r=(-1, -1)  ↖
    TopRightToBottomLeft,   //  r=(-1, +1)  ↙
    BottomLeftToTopRight    //  r=(+1, -1)  ↗
};

struct PathStep { int dx; int dy; };

inline PathStep direction_vector(Direction8 direction) {
    switch (direction) {
        case Direction8::LeftToRight:          return {+1,  0};
        case Direction8::RightToLeft:          return {-1,  0};
        case Direction8::TopToBottom:          return { 0, +1};
        case Direction8::BottomToTop:          return { 0, -1};
        case Direction8::TopLeftToBottomRight: return {+1, +1};
        case Direction8::BottomRightToTopLeft: return {-1, -1};
        case Direction8::TopRightToBottomLeft: return {-1, +1};
        case Direction8::BottomLeftToTopRight: return {+1, -1};
    }
    throw std::invalid_argument("unknown Direction8");
}

inline bool in_image(int x, int y, int W, int H) {
    return x >= 0 && x < W && y >= 0 && y < H;
}

// 核心路径扫描：
//   1) 遍历像素，若其前驱 (x-dx,y-dy) 越界，该像素就是一条路径的起点；
//   2) 清空 previous：每条路径必须独立；
//   3) 沿 (dx,dy) 前进，直到越过图像边界。
// callback 参数：on_pixel(y, x, base, current)，current 是刚计算完成的 L_r(p,:)。
// 每个方向，每个像素恰好计算一次（无遗漏、无重复）。
template <class OnPixel>
inline void scan_direction8(const CostVolume& C,
                            const SGMParams& p,
                            Direction8 direction,
                            OnPixel&& on_pixel) {
    validate_volume_for_sgm(C);
    if (p.P1 < 0 || p.P2 < p.P1 || p.P2 >= kInvalidCost / 4)
        throw std::invalid_argument("bad P1/P2");

    const int W = C.width;
    const int H = C.height;
    const int D = C.num_disparities;
    const auto [dx, dy] = direction_vector(direction);

    std::vector<int32_t> previous(D, kInvalidCost);
    std::vector<int32_t> current(D, kInvalidCost);
    std::vector<int32_t> point_cost(D, kInvalidCost);

    // 这里的两层循环是枚举“可能的路径起点”，不是动态规划本身。
    for (int sy = 0; sy < H; ++sy) {
        for (int sx = 0; sx < W; ++sx) {
            // 只有前驱越界的点才是路径起点。
            if (in_image(sx - dx, sy - dy, W, H)) continue;

            // 每条路径独立初始化，避免上一条对角线的结果串入当前路径。
            std::fill(previous.begin(), previous.end(), kInvalidCost);

            // 真正的路径遍历：两个坐标按 (dx,dy) 同时变化。
            for (int x = sx, y = sy; in_image(x, y, W, H); x += dx, y += dy) {
                const size_t base = C.index(y, x, 0);
                std::copy_n(C.costs.begin() + base, D, point_cost.begin());

                // Stage03 论文公式(13)。路径起点的 previous 全无效，于是 L=C。
                aggregate_one_step(point_cost, previous, current, p);

                // 当前像素计算完成，将 D 个路径代价交给调用者。
                on_pixel(y, x, base, current);

                // 下一像素的 previous 就是刚才的 current。
                previous.swap(current);
            }
        }
    }
}

// 调试用：单方向完整路径代价 L_r。方便与 Stage04 对比。
inline CostVolume aggregate_one_direction8(const CostVolume& C,
                                           const SGMParams& p,
                                           Direction8 direction) {
    CostVolume L;
    L.width = C.width;
    L.height = C.height;
    L.min_disparity = C.min_disparity;
    L.num_disparities = C.num_disparities;
    L.costs.assign(C.costs.size(), kInvalidCost);

    scan_direction8(C, p, direction,
                    [&](int /*y*/, int /*x*/, size_t base,
                        const std::vector<int32_t>& current) {
        std::copy(current.begin(), current.end(), L.costs.begin() + base);
    });
    return L;
}

// 生产路径：直接累加八方向路径代价（论文公式 14）。
// 注意：保留无效像素；使用 int64 做临时求和，防止溢出。
inline CostVolume aggregate_eight_directions(const CostVolume& C,
                                             const SGMParams& p) {
    validate_volume_for_sgm(C);
    if (p.P1 < 0 || p.P2 < p.P1 || p.P2 >= kInvalidCost / 4)
        throw std::invalid_argument("bad P1/P2");

    CostVolume S;
    S.width = C.width;
    S.height = C.height;
    S.min_disparity = C.min_disparity;
    S.num_disparities = C.num_disparities;
    S.costs.assign(C.costs.size(), 0);
    for (size_t i = 0; i < C.costs.size(); ++i)
        if (!is_valid_cost(C.costs[i])) S.costs[i] = kInvalidCost;

    constexpr std::array<Direction8, 8> directions = {
        Direction8::LeftToRight,
        Direction8::RightToLeft,
        Direction8::TopToBottom,
        Direction8::BottomToTop,
        Direction8::TopLeftToBottomRight,
        Direction8::BottomRightToTopLeft,
        Direction8::TopRightToBottomLeft,
        Direction8::BottomLeftToTopRight
    };

    for (const auto direction : directions) {
        scan_direction8(C, p, direction,
                        [&](int /*y*/, int /*x*/, size_t base,
                            const std::vector<int32_t>& current) {
            for (int di = 0; di < C.num_disparities; ++di) {
                const size_t i = base + static_cast<size_t>(di);
                if (!is_valid_cost(C.costs[i])) continue;
                if (!is_valid_cost(current[di]))
                    throw std::logic_error("valid C became invalid in path aggregation");
                const int64_t sum = static_cast<int64_t>(S.costs[i]) + current[di];
                if (sum >= kInvalidCost)
                    throw std::overflow_error("8-path sum exceeds int32 valid range");
                S.costs[i] = static_cast<int32_t>(sum);
            }
        });
    }
    return S;
}

}  // namespace stereo
