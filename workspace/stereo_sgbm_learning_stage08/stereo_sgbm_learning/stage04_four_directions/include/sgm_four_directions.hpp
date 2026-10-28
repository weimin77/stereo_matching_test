#pragma once

// Stage 04: four independent directional SGM path costs, summed according to
// Hirschmueller 2008 Eq. (14). Keep Stage03 Eq. (13) exactly as-is.
// Main memory: input C + output S + O(D) path state (no 4 full L volumes).
#include "sgm_single_path.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace stereo {

enum class Direction {
    LeftToRight,
    RightToLeft,
    TopToBottom,
    BottomToTop
};

// Runs one direction. The callback receives the flat [H][W][D] base index
// and that pixel's newly calculated L_r(p,:) before swapping buffers.
// For horizontal scans, every ROW is an independent path. For vertical scans,
// every COLUMN is an independent path. Reset previous before each path.
template<class OnPixel>
inline void scan_direction(const CostVolume& C, const SGMParams& p,
                           Direction direction, OnPixel&& on_pixel) {
    const int W = C.width;
    const int H = C.height;
    const int D = C.num_disparities;

    std::vector<int32_t> previous(D, kInvalidCost);
    std::vector<int32_t> current(D, kInvalidCost);
    std::vector<int32_t> point_cost(D, kInvalidCost);

    const auto step = [&](int y, int x) {
        const size_t base = C.index(y, x, 0);
        std::copy_n(C.costs.begin() + base, D, point_cost.begin());
        aggregate_one_step(point_cost, previous, current, p); // Eq. (13)
        on_pixel(base, current);                             // output this L_r
        previous.swap(current);                              // next predecessor
    };

    switch (direction) {
    case Direction::LeftToRight:
        for (int y = 0; y < H; ++y) {
            std::fill(previous.begin(), previous.end(), kInvalidCost);
            for (int x = 0; x < W; ++x) step(y, x);
        }
        break;
    case Direction::RightToLeft:
        for (int y = 0; y < H; ++y) {
            std::fill(previous.begin(), previous.end(), kInvalidCost);
            for (int x = W - 1; x >= 0; --x) step(y, x);
        }
        break;
    case Direction::TopToBottom:
        for (int x = 0; x < W; ++x) {
            std::fill(previous.begin(), previous.end(), kInvalidCost);
            for (int y = 0; y < H; ++y) step(y, x);
        }
        break;
    case Direction::BottomToTop:
        for (int x = 0; x < W; ++x) {
            std::fill(previous.begin(), previous.end(), kInvalidCost);
            for (int y = H - 1; y >= 0; --y) step(y, x);
        }
        break;
    default:
        throw std::invalid_argument("unknown aggregation direction");
    }
}

// Debug API: returns ONE complete L_r volume (requires another W*H*D buffer).
// It is helpful for inspecting one direction against hand-calculated results.
inline CostVolume aggregate_one_direction(const CostVolume& C,
                                          const SGMParams& p,
                                          Direction direction) {
    validate_volume_for_sgm(C);
    CostVolume L;
    L.width = C.width;
    L.height = C.height;
    L.min_disparity = C.min_disparity;
    L.num_disparities = C.num_disparities;
    L.costs.assign(C.costs.size(), kInvalidCost);

    scan_direction(C, p, direction, [&](size_t base, const auto& current) {
        std::copy(current.begin(), current.end(), L.costs.begin() + base);
    });
    return L;
}

// Main API: Eq. (14), S(p,d) = sum_r L_r(p,d).
// Stream each direction's D-vector directly into S without storing 4 large
// directional volumes. Invalid candidate disparities stay invalid.
// Deliberately DOES NOT subtract (R-1)*C: the source paper's Eq. (14) is a sum.
inline CostVolume aggregate_four_directions(const CostVolume& C,
                                            const SGMParams& p) {
    validate_volume_for_sgm(C);
    if (p.P1 < 0 || p.P2 < p.P1 || p.P2 >= kInvalidCost / 4)
        throw std::invalid_argument("bad P1/P2");

    CostVolume S;
    S.width = C.width;
    S.height = C.height;
    S.min_disparity = C.min_disparity;
    S.num_disparities = C.num_disparities;
    S.costs.resize(C.costs.size(), 0);
    for (size_t i = 0; i < S.costs.size(); ++i) {
        if (!is_valid_cost(C.costs[i])) S.costs[i] = kInvalidCost;
    }

    constexpr std::array<Direction, 4> directions{
        Direction::LeftToRight,
        Direction::RightToLeft,
        Direction::TopToBottom,
        Direction::BottomToTop
    };
    for (Direction direction : directions) {
        scan_direction(C, p, direction, [&](size_t base, const auto& current) {
            for (int di = 0; di < C.num_disparities; ++di) {
                const size_t index = base + static_cast<size_t>(di);
                if (!is_valid_cost(C.costs[index])) continue;
                if (!is_valid_cost(current[di]))
                    throw std::logic_error("valid original cost became invalid");
                const int64_t total = static_cast<int64_t>(S.costs[index]) + current[di];
                if (total >= kInvalidCost)
                    throw std::overflow_error("4-path summed cost exceeds valid int32 range");
                S.costs[index] = static_cast<int32_t>(total);
            }
        });
    }
    return S;
}

} // namespace stereo
