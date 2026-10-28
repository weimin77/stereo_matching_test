#pragma once

// Stage 03: 单方向（左->右）SGM 聚合。
// 只依赖 Stage 02 的 CostVolume；不调用 OpenCV 立体匹配接口。
#include "cost_volume.hpp"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace stereo {

struct SGMParams {
    int32_t P1 = 5;    // 视差变化 1 的惩罚
    int32_t P2 = 20;   // 视差变化超过 1 的惩罚
};

inline bool is_valid_cost(int32_t value) {
    return value >= 0 && value < kInvalidCost;
}

// 对应论文 Eq.(13)：沿一条路径，从前一个像素的 D 个状态计算当前像素。
// previous/current/point_cost 长度必须一致，且 previous 和 current 不能别名。
// previous 全无效时，把当前有效代价当成新路径的起点 L=C。
// 无效候选始终无效，不会通过动态规划“凭空创造”匹配。
inline void aggregate_one_step(
    const std::vector<int32_t>& point_cost,
    const std::vector<int32_t>& previous,
    std::vector<int32_t>& current,
    const SGMParams& p)
{
    if (point_cost.empty() || previous.size() != point_cost.size() ||
        &previous == &current || &point_cost == &current)
        throw std::invalid_argument("invalid or aliased path state buffers");
    if (p.P1 < 0 || p.P2 < p.P1 || p.P2 >= kInvalidCost / 4)
        throw std::invalid_argument("P1/P2 must satisfy 0 <= P1 <= P2");

    const size_t D = point_cost.size();
    current.assign(D, kInvalidCost);
    const int32_t prev_min = *std::min_element(previous.begin(), previous.end());

    // 路径的起点（或一段完全无效后重新进入有效区域）。
    if (!is_valid_cost(prev_min)) {
        for (size_t di = 0; di < D; ++di) {
            if (is_valid_cost(point_cost[di])) current[di] = point_cost[di];
        }
        return;
    }

    for (size_t di = 0; di < D; ++di) {
        const int32_t C = point_cost[di];
        if (!is_valid_cost(C)) continue;

        // 四种转移：同一视差、来自 di-1、来自 di+1、任意跳变。
        // 无效前驱直接保留 INF；不要对 INF 加 P1。
        int64_t best = is_valid_cost(previous[di])
            ? static_cast<int64_t>(previous[di]) : kInvalidCost;

        if (di > 0 && is_valid_cost(previous[di - 1]))
            best = std::min(best, static_cast<int64_t>(previous[di - 1]) + p.P1);
        if (di + 1 < D && is_valid_cost(previous[di + 1]))
            best = std::min(best, static_cast<int64_t>(previous[di + 1]) + p.P1);

        best = std::min(best, static_cast<int64_t>(prev_min) + p.P2);

        // 减去上一像素所有视差代价的最小值：防止路径累加溢出。
        const int64_t value = static_cast<int64_t>(C) + best - prev_min;
        if (value >= kInvalidCost)
            throw std::overflow_error("aggregated path cost exceeds valid range");
        current[di] = static_cast<int32_t>(value);
    }
}

inline void validate_volume_for_sgm(const CostVolume& v) {
    if (v.width <= 0 || v.height <= 0 || v.num_disparities <= 0 ||
        v.costs.size() != static_cast<size_t>(v.width) * v.height * v.num_disparities)
        throw std::invalid_argument("bad CostVolume geometry");
    for (int32_t c : v.costs)
        if (c < 0 || c > kInvalidCost)
            throw std::invalid_argument("CostVolume requires nonnegative costs and kInvalidCost sentinel");
}

// 左 -> 右：每一行是一条独立路径。
// 输入 C(y,x,di)，输出 L_right(y,x,di)，不跨行、不修改原始 C。
// 本阶段只做 1 个方向，因此还不是完整的 SGM，更不是 OpenCV StereoSGBM。
inline CostVolume aggregate_left_to_right(
    const CostVolume& cost_volume, const SGMParams& p)
{
    validate_volume_for_sgm(cost_volume);
    if (p.P1 < 0 || p.P2 < p.P1 || p.P2 >= kInvalidCost / 4)
        throw std::invalid_argument("bad P1/P2");

    CostVolume aggregated = cost_volume;
    const int W = cost_volume.width;
    const int H = cost_volume.height;
    const int D = cost_volume.num_disparities;

    std::vector<int32_t> previous(D, kInvalidCost);
    std::vector<int32_t> current(D, kInvalidCost);
    std::vector<int32_t> cost(D, kInvalidCost);

    for (int y = 0; y < H; ++y) {
        std::fill(previous.begin(), previous.end(), kInvalidCost);
        for (int x = 0; x < W; ++x) {
            const size_t base = cost_volume.index(y, x, 0);
            std::copy_n(cost_volume.costs.begin() + base, D, cost.begin());
            aggregate_one_step(cost, previous, current, p);
            std::copy(current.begin(), current.end(), aggregated.costs.begin() + base);
            previous.swap(current);   // 下一个像素使用刚刚求得的 Lr(p,d)
        }
    }
    return aggregated;
}

} // namespace stereo
