#pragma once

// Stage 07: Hirschmueller 2008, Section 2.3 Eq. (15): left-right consistency.
// Both D_L and D_R store positive disparity magnitudes:
//   x_right = x_left - D_L(x_left,y)
//   x_left  = x_right + D_R(x_right,y)
// For the nearest-neighbor right-map lookup, a fractional x_right is rounded.
// IMPORTANT: Stage02 only searches x_match=x_base-d and min_disparity>=0;
// right-view disparity is computed on flipped(right) / flipped(left), then
// flipped back. Merely swapping the original images is geometrically wrong.
#include "subpixel_refinement.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace stereo {

enum class LRStatus : uint8_t {
    LeftInvalid = 0,
    Consistent = 1,
    RightOutside = 2,
    RightInvalid = 3,
    Disagree = 4
};

struct LRParams {
    float max_disparity_difference = 1.0f; // paper Eq. (15): tolerance <= 1px
};

struct LRResult {
    SubpixelResult filtered;             // invalid samples -> valid=0, disparity=NaN
    std::vector<LRStatus> status;         // reason for accepting/rejecting each pixel
};

inline void validate_subpixel_map(const SubpixelResult& map) {
    if (map.width <= 0 || map.height <= 0)
        throw std::invalid_argument("disparity map must have positive dimensions");
    const size_t n = static_cast<size_t>(map.width) * map.height;
    if (map.disparity.size() != n || map.valid.size() != n || map.refined.size() != n)
        throw std::invalid_argument("disparity map arrays have wrong length");
}

inline GrayImage flip_horizontal(const GrayImage& input) {
    if (input.width <= 0 || input.height <= 0 ||
        input.pixels.size() != static_cast<size_t>(input.width) * input.height)
        throw std::invalid_argument("invalid GrayImage for flip_horizontal");
    GrayImage output = input;
    for (int y = 0; y < input.height; ++y)
        for (int x = 0; x < input.width; ++x)
            output.pixels[static_cast<size_t>(y) * input.width + x] =
                input.pixels[static_cast<size_t>(y) * input.width + (input.width - 1 - x)];
    return output;
}

inline SubpixelResult flip_horizontal(const SubpixelResult& input) {
    validate_subpixel_map(input);
    SubpixelResult output = input;
    const int W = input.width;
    for (int y = 0; y < input.height; ++y) {
        for (int x = 0; x < W; ++x) {
            const size_t dst = static_cast<size_t>(y) * W + x;
            const size_t src = static_cast<size_t>(y) * W + (W - 1 - x);
            output.disparity[dst] = input.disparity[src];
            output.valid[dst] = input.valid[src];
            output.refined[dst] = input.refined[src];
        }
    }
    return output;
}

// Reuse exactly the same left-view SAD + 8-path SGM + subpixel pipeline.
// In flipped coordinates the required x_left=x_right+d becomes u_left=u_right-d.
inline SubpixelResult compute_right_disparity_flipped(
    const GrayImage& original_left,
    const GrayImage& original_right,
    const BMParams& matching,
    const SGMParams& aggregation,
    const SubpixelParams& subpixel = {})
{
    const GrayImage reversed_base = flip_horizontal(original_right);
    const GrayImage reversed_match = flip_horizontal(original_left);
    const CostVolume C = compute_cost_volume_sad_integral(
        reversed_base, reversed_match, matching);
    const CostVolume S = aggregate_eight_directions(C, aggregation);
    return flip_horizontal(refine_subpixel(S, subpixel));
}

// Eq. (15). Reject invalid left, out of right image, invalid right and mismatch.
// Fractional x_right is rounded to nearest right-image pixel in this teaching
// implementation. A future refinement may bilinearly interpolate D_R.
inline LRResult left_right_consistency_check(
    const SubpixelResult& left,
    const SubpixelResult& right,
    const LRParams& params = {})
{
    validate_subpixel_map(left);
    validate_subpixel_map(right);
    if (left.width != right.width || left.height != right.height)
        throw std::invalid_argument("left and right disparity dimensions differ");
    if (!std::isfinite(params.max_disparity_difference) ||
        params.max_disparity_difference < 0)
        throw std::invalid_argument("LRC tolerance must be finite and >= 0");

    LRResult result;
    result.filtered = left; // preserve valid disparities and their refined mask
    const int W = left.width;
    const int H = left.height;
    const size_t n = static_cast<size_t>(W) * H;
    result.status.assign(n, LRStatus::LeftInvalid);
    const float nan = std::numeric_limits<float>::quiet_NaN();

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const size_t i = static_cast<size_t>(y) * W + x;
            LRStatus reason = LRStatus::LeftInvalid;
            const float dl = left.disparity[i];
            if (left.valid[i] && std::isfinite(dl) && dl >= 0.0f) {
                const double right_x = static_cast<double>(x) - dl;
                if (right_x < 0.0 || right_x > static_cast<double>(W - 1)) {
                    reason = LRStatus::RightOutside;
                } else {
                    const int xr = static_cast<int>(std::lround(right_x));
                    const size_t j = static_cast<size_t>(y) * W + xr;
                    const float dr = right.disparity[j];
                    if (!right.valid[j] || !std::isfinite(dr) || dr < 0.0f)
                        reason = LRStatus::RightInvalid;
                    else if (std::abs(static_cast<double>(dl) - dr) >
                             params.max_disparity_difference)
                        reason = LRStatus::Disagree;
                    else
                        reason = LRStatus::Consistent;
                }
            }
            result.status[i] = reason;
            if (reason != LRStatus::Consistent) {
                result.filtered.valid[i] = 0;
                result.filtered.refined[i] = 0;
                result.filtered.disparity[i] = nan;
            }
        }
    }
    return result;
}

} // namespace stereo
