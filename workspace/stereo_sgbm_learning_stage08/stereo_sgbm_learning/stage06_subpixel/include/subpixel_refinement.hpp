#pragma once

// Stage 06: Section 2.3 (Disparity Computation) - subpixel refinement.
// 输入：Stage 05 的八方向聚合代价 S(y,x,di)，而非原始灰度值。
// 对 WTA 最优整数视差 d0 和左右相邻的代价，使用三点抛物线插值：
// delta = (S[d0-1]-S[d0+1]) / (2*(S[d0-1]-2*S[d0]+S[d0+1])).
// 输出浮点视差，NaN 表示无效；可用 valid 独立判断。
#include "sgm_eight_directions.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace stereo {

struct SubpixelResult {
    int width = 0;
    int height = 0;
    std::vector<float> disparity;   // actual pixel disparity, not disparity index
    std::vector<uint8_t> valid;     // 1 = valid WTA, 0 = no valid candidate
    std::vector<uint8_t> refined;   // 1 = quadratic fit changed integer result
};

struct SubpixelParams {
    // 只在分母（离散二阶差分）超过阈值时执行拟合。
    // 0 意味着所有正曲率均允许；增加阈值可以拒绝过于平坦的代价曲线。
    double min_curvature = 0.0;
};

inline SubpixelResult refine_subpixel(const CostVolume& aggregated,
                                     const SubpixelParams& p = {}) {
    validate_volume_for_sgm(aggregated);
    if (!std::isfinite(p.min_curvature) || p.min_curvature < 0)
        throw std::invalid_argument("min_curvature must be finite and nonnegative");

    SubpixelResult out;
    out.width = aggregated.width;
    out.height = aggregated.height;
    const size_t n = static_cast<size_t>(out.width) * out.height;
    out.disparity.assign(n, std::numeric_limits<float>::quiet_NaN());
    out.valid.assign(n, 0);
    out.refined.assign(n, 0);

    const int D = aggregated.num_disparities;
    for (int y = 0; y < out.height; ++y) {
        for (int x = 0; x < out.width; ++x) {
            const size_t pixel = static_cast<size_t>(y) * out.width + x;
            const size_t base = aggregated.index(y, x, 0);

            // Step 1. WTA: 和 Stage 02 一样，严格用 < 处理平局。
            int32_t best_cost = kInvalidCost;
            int best_di = -1;
            for (int di = 0; di < D; ++di) {
                const int32_t c = aggregated.costs[base + di];
                if (is_valid_cost(c) && c < best_cost) {
                    best_cost = c;
                    best_di = di;
                }
            }
            if (best_di < 0) continue;  // 所有候选视差无效
            out.valid[pixel] = 1;

            // 先保存整数 WTA：拟合不稳定时直接回退。
            const int integer_disparity = aggregated.actual_disparity(best_di);
            out.disparity[pixel] = static_cast<float>(integer_disparity);

            // Step 2. 左右邻居越界 => 没有 3 点，退回整数。
            if (best_di == 0 || best_di == D - 1) continue;

            const int32_t c_left = aggregated.costs[base + best_di - 1];
            const int32_t c0 = aggregated.costs[base + best_di];
            const int32_t c_right = aggregated.costs[base + best_di + 1];
            // 任一邻居的候选代价无效 => 不把 INF 用于抛物线拟合。
            if (!is_valid_cost(c_left) || !is_valid_cost(c_right)) continue;

            // Step 3. 使用 double/int64 避免 2*c0 整数运算溢出。
            const double curvature = static_cast<double>(c_left)
                                   - 2.0 * c0 + c_right;
            if (!std::isfinite(curvature) || curvature <= p.min_curvature)
                continue;
            double delta = (static_cast<double>(c_left) - c_right)
                         / (2.0 * curvature);
            if (!std::isfinite(delta)) continue;

            // 理论上当 c0 是严格局部最小值时位于 [-0.5, 0.5]；
            // clamp 是稳健防护，不是额外的准确性保证。
            delta = std::clamp(delta, -0.5, 0.5);
            out.disparity[pixel] = static_cast<float>(integer_disparity + delta);
            if (std::abs(delta) > 1e-8) out.refined[pixel] = 1;
        }
    }
    return out;
}

} // namespace stereo
