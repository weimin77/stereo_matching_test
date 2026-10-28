#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace stereo {

struct GrayImage {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> pixels;

    uint8_t at(int y, int x) const {
        return pixels[static_cast<size_t>(y) * width + x];
    }
};

struct BMParams {
    int min_disparity = 0;
    int num_disparities = 64;
    int block_size = 5;
};

struct BMResult {
    int width = 0;
    int height = 0;
    // Integer disparity in pixels; -1 means invalid.
    std::vector<int16_t> disparity;
};

// Stage 1: pure C++ SAD block matching and Winner-Takes-All.
// Input images must already be rectified, grayscale, identical size.
// Intentionally simple (O(W*H*D*blockSize^2)); no OpenCV matcher used.
inline BMResult compute_bm_sad(const GrayImage& left, const GrayImage& right,
                               const BMParams& p) {
    if (left.width <= 0 || left.height <= 0 ||
        left.width != right.width || left.height != right.height ||
        left.pixels.size() != static_cast<size_t>(left.width) * left.height ||
        right.pixels.size() != left.pixels.size()) {
        throw std::invalid_argument("invalid stereo image shape");
    }
    if (p.num_disparities <= 0 || p.block_size <= 0 || p.block_size % 2 != 1 ||
        p.block_size > left.width || p.block_size > left.height) {
        throw std::invalid_argument("bad block size or disparity range");
    }
    const int w = left.width;
    const int h = left.height;
    const int radius = p.block_size / 2;
    BMResult result{w, h, std::vector<int16_t>(static_cast<size_t>(w) * h, -1)};

    for (int y = radius; y < h - radius; ++y) {
        for (int x = radius; x < w - radius; ++x) {
            int best_cost = std::numeric_limits<int>::max();
            int best_disp = -1;
            for (int k = 0; k < p.num_disparities; ++k) {
                const int disp = p.min_disparity + k;
                const int xr = x - disp;
                if (xr - radius < 0 || xr + radius >= w) continue;

                int sad = 0;
                for (int dy = -radius; dy <= radius; ++dy) {
                    for (int dx = -radius; dx <= radius; ++dx) {
                        const int a = left.at(y + dy, x + dx);
                        const int b = right.at(y + dy, xr + dx);
                        sad += a > b ? a - b : b - a;
                    }
                }
                if (sad < best_cost) {
                    best_cost = sad;
                    best_disp = disp;
                }
            }
            // int16_t is enough for the intentionally small disparity ranges here.
            if (best_disp >= 0 && best_disp <= 32767) {
                result.disparity[static_cast<size_t>(y) * w + x] =
                    static_cast<int16_t>(best_disp);
            }
        }
    }
    return result;
}

}  // namespace stereo
