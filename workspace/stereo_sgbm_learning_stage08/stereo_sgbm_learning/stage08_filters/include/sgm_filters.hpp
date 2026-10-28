#pragma once

// Stage 08 -- confidence and spatial filtering for educational SGM.
// 1) uniqueness_filter(): inspect S(x,y,d), reject ambiguous NONLOCAL costs.
//    The candidate set excludes |di-best_di|<=1 as in classic OpenCV SGBM.
//    Here a strict percentage margin is used: 100*(second-best) > (100+u)*best.
//    This differs in strict tie/boundary details from OpenCV's inequality.
// 2) remove_small_speckles(): 4-connected disparity components, adjacent
//    disparity difference <= max_diff_px (paper Section 2.5.1).
//    Components of size <= max_component_size are invalidated.
// Both functions preserve dimensions and use valid mask + NaN for rejection.

#include "lr_consistency.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace stereo {

struct UniquenessParams {
    // 0 disables uniqueness filtering. For enabled filtering: 1..99.
    int ratio_percent = 10;
    // Exclude best index and its immediate neighboring disparities from competition.
    int excluded_radius = 1;
};

enum class UniqueStatus : uint8_t {
    InputInvalid = 0,
    Unique = 1,
    Ambiguous = 2,
    NoNonlocalCompetitor = 3
};

struct UniquenessResult {
    SubpixelResult filtered;
    std::vector<UniqueStatus> status;
    std::vector<int32_t> best_cost;
    std::vector<int32_t> other_cost;
};

inline void invalidate_pixel(SubpixelResult& m, size_t i) {
    m.valid[i] = 0;
    m.refined[i] = 0;
    m.disparity[i] = std::numeric_limits<float>::quiet_NaN();
}

// Apply after WTA/refine_subpixel(S), before LR check.
// The WTA index is re-found in S, not rounded from the refined float disparity.
// This deliberately follows exactly the same tie handling as Stage06 WTA.
inline UniquenessResult uniqueness_filter(
    const CostVolume& S,
    const SubpixelResult& disparity,
    const UniquenessParams& p = {})
{
    validate_volume_for_sgm(S);
    validate_subpixel_map(disparity);
    if (S.width != disparity.width || S.height != disparity.height)
        throw std::invalid_argument("S and disparity have different image sizes");
    if (p.ratio_percent < 0 || p.ratio_percent >= 100 || p.excluded_radius < 0)
        throw std::invalid_argument("ratio must be 0..99, excluded radius >= 0");

    UniquenessResult result;
    result.filtered = disparity;
    const size_t n = static_cast<size_t>(S.width) * S.height;
    result.status.assign(n, UniqueStatus::InputInvalid);
    result.best_cost.assign(n, kInvalidCost);
    result.other_cost.assign(n, kInvalidCost);
    const int D = S.num_disparities;

    for (int y = 0; y < S.height; ++y) {
        for (int x = 0; x < S.width; ++x) {
            const size_t pixel = static_cast<size_t>(y) * S.width + x;
            if (!disparity.valid[pixel] || !std::isfinite(disparity.disparity[pixel])) {
                invalidate_pixel(result.filtered, pixel);
                continue;
            }

            const size_t base = S.index(y, x, 0);
            int best_di = -1;
            int32_t best = kInvalidCost;
            for (int di=0; di<D; ++di) {
                const int32_t c = S.costs[base+di];
                if (is_valid_cost(c) && c < best) {
                    best = c;
                    best_di = di;
                }
            }
            if (best_di < 0) {
                invalidate_pixel(result.filtered, pixel);
                continue;
            }
            result.best_cost[pixel] = best;
            if (p.ratio_percent == 0) {
                result.status[pixel] = UniqueStatus::Unique;
                continue;
            }

            int32_t second = kInvalidCost;
            for (int di=0; di<D; ++di) {
                if (std::abs(di - best_di) <= p.excluded_radius) continue;
                const int32_t c = S.costs[base+di];
                if (is_valid_cost(c) && c < second) second = c;
            }
            result.other_cost[pixel] = second;
            if (!is_valid_cost(second)) {
                result.status[pixel] = UniqueStatus::NoNonlocalCompetitor;
                continue;
            }
            // Compare in int64; no division by zero (best can equal zero).
            // Require a STRICT margin; best==second==0 is ambiguous.
            const bool unique = 100LL * (static_cast<int64_t>(second) - best)
                              > static_cast<int64_t>(p.ratio_percent) * best;
            result.status[pixel] = unique ? UniqueStatus::Unique : UniqueStatus::Ambiguous;
            if (!unique) invalidate_pixel(result.filtered, pixel);
        }
    }
    return result;
}

struct SpeckleParams {
    // <=0 disables filtering; sizes are PIXEL COUNTS, not square-window width.
    int max_component_size = 50;
    // Adjacent finite disparities may connect if abs(d1-d2)<=max_diff_px.
    float max_diff_px = 1.0f;
};

struct SpeckleResult {
    SubpixelResult filtered;
    std::vector<uint8_t> removed; // 1 only for valid input discarded as small component
    int components_total = 0;
    int components_removed = 0;
    size_t pixels_removed = 0;
};

// Runs BFS on INPUT disparities and masks; no in-place changes while walking.
// Complexity O(W*H), auxiliary memory O(W*H).
inline SpeckleResult remove_small_speckles(
    const SubpixelResult& input,
    const SpeckleParams& p = {})
{
    validate_subpixel_map(input);
    if (p.max_component_size < 0 || !std::isfinite(p.max_diff_px) || p.max_diff_px < 0)
        throw std::invalid_argument("speckle params must be nonnegative and finite");

    SpeckleResult out;
    out.filtered = input;
    const int W = input.width, H = input.height;
    const size_t n = static_cast<size_t>(W)*H;
    out.removed.assign(n, 0);
    if (p.max_component_size == 0) return out;

    std::vector<uint8_t> seen(n,0);
    std::vector<size_t> queue;
    queue.reserve(std::min(n, static_cast<size_t>(4096)));

    for(size_t seed=0; seed<n; ++seed) {
        if (seen[seed] || !input.valid[seed] ||
            !std::isfinite(input.disparity[seed])) continue;
        ++out.components_total;
        queue.clear();
        queue.push_back(seed);
        seen[seed] = 1;
        size_t head = 0;
        while(head < queue.size()) {
            const size_t cur = queue[head++];
            const int x = static_cast<int>(cur % W);
            const int y = static_cast<int>(cur / W);
            const float d = input.disparity[cur];
            const auto try_neighbor = [&](size_t ni) {
                if (seen[ni] || !input.valid[ni] ||
                    !std::isfinite(input.disparity[ni])) return;
                if (std::abs(d - input.disparity[ni]) > p.max_diff_px) return;
                seen[ni]=1;
                queue.push_back(ni);
            };
            if(x>0) try_neighbor(cur-1);
            if(x+1<W) try_neighbor(cur+1);
            if(y>0) try_neighbor(cur-W);
            if(y+1<H) try_neighbor(cur+W);
        }
        if (queue.size() <= static_cast<size_t>(p.max_component_size)) {
            ++out.components_removed;
            out.pixels_removed += queue.size();
            for (const size_t i:queue) {
                out.removed[i] = 1;
                invalidate_pixel(out.filtered, i);
            }
        }
    }
    return out;
}

} // namespace stereo
