#include "handmade_bm.hpp"
#include <cassert>
#include <iostream>

int main() {
    constexpr int w = 96, h = 64, true_disp = 7;
    stereo::GrayImage L{w, h, std::vector<uint8_t>(w * h)};
    stereo::GrayImage R{w, h, std::vector<uint8_t>(w * h)};
    // Deterministic textured pattern (no RNG dependency).
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            L.pixels[y*w+x] = static_cast<uint8_t>((x*37 + y*17 + x*y*13 + x*x*3) % 251);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w - true_disp; ++x)
            R.pixels[y*w+x] = L.pixels[y*w+x+true_disp];

    stereo::BMParams p;
    p.min_disparity = 0;
    p.num_disparities = 16;
    p.block_size = 5;
    auto result = stereo::compute_bm_sad(L, R, p);
    int matched = 0;
    int wrong = 0;
    for (int y = 4; y < h - 4; ++y) {
        for (int x = 22; x < w - 4; ++x) {
            ++matched;
            if (result.disparity[y*w+x] != true_disp) ++wrong;
        }
    }
    std::cout << "BM synthetic test: " << matched << " pixels, "
              << wrong << " wrong\n";
    assert(wrong == 0);
    std::cout << "PASS\n";
}
