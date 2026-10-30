// Transcription of OpenCV 5's StereoSGBM MODE_SGBM pipeline.
// See sgbm_oracle.h for provenance and intent.
//
// ARITHMETIC: OpenCV's SGBM does not agree with itself across backends. For
// v_int16 the universal intrinsics used by the shipped build are
//
//     v_add  -> _mm_adds_epi16   (saturating)
//     v_sub  -> _mm_subs_epi16   (saturating)
//     v_mul  -> widening multiply + pack (saturating)
//
// while the `#else` fallback in the same file is plain wrapping `short`
// arithmetic. The two only diverge once an intermediate leaves [-32768,32767],
// which happens for large block sizes: hsum*(SH2+1) crosses 32767 at
// blockSize 11 with preFilterCap 31. This transcription follows the SIMD
// semantics, because that is what a normal x86 build ships. Every 16-bit
// operation below is therefore spelled out as add16/sub16/mul16 so the
// saturation points are explicit and auditable.
#include "sgbm_oracle.h"

#include <opencv2/core/hal/intrin.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/stereo.hpp>

#include <algorithm>
#include <climits>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace oracle {
namespace {

typedef uchar PixType;
typedef short CostType;
typedef short DispType;

constexpr int kDispShift = 4;
constexpr int kDispScale = 1 << kDispShift;
constexpr int kMaxCost = SHRT_MAX;

// Saturating 16-bit arithmetic, matching the universal intrinsics OpenCV
// actually executes.
inline short sat16(int v) { return (short)std::min(32767, std::max(-32768, v)); }
inline short add16(short a, short b) { return sat16((int)a + b); }
inline short sub16(short a, short b) { return sat16((int)a - b); }
inline short mul16(short a, short b) { return sat16((int)a * b); }
// vx_setall_s16((short)x): truncation, not saturation.
inline short trunc16(int v) { return (short)(v & 0xffff); }

// Mirrors VTraits<v_int16>::vlanes() so Da matches whatever OpenCV this is
// built against.
int alignLanes() { return cv::VTraits<cv::v_int16>::vlanes(); }

int alignSizeTo(int value, int n) { return (value + n - 1) / n * n; }

// ---------------------------------------------------------------------------
// calcPixelCostBT, scalar path only.
//
// Two cost layers are produced (c = 0 and c = 1):
//   layer 0  clipped derivative response, full weight
//   layer 1  raw intensity, shifted right by 2
// The derivative layer uses a 3x3 neighbourhood folded into one lookup:
//   tab[(r[x+1]-r[x-1])*2 + r[x+1+n]-r[x-1+n] + r[x+1+s]-r[x-1+s]]
// where n/s are the previous/next row offsets (zero at the top/bottom edge).
// ---------------------------------------------------------------------------
void calcPixelCostBT(const cv::Mat& img1, const cv::Mat& img2, int y, int minD, int maxD,
                     CostType* pixDiff, PixType* tempBuf, const PixType* tab) {
    const int width = img1.cols;
    const int cn = img1.channels();
    const int lane = alignLanes();

    const int minX1 = std::max(maxD, 0);
    const int maxX1 = width + std::min(minD, 0);
    const int D = alignSizeTo(maxD - minD, lane);
    const int width1 = maxX1 - minX1;

    const int minX2 = std::max(minX1 - maxD, 0);
    const int maxX2 = std::min(maxX1 - minD, width);
    const int width2 = maxX2 - minX2;

    const PixType* row1 = img1.ptr<PixType>(y);
    const PixType* row2 = img2.ptr<PixType>(y);

    PixType* prow1 = tempBuf + width2 * 2;
    PixType* prow2 = prow1 + width * cn * 2;

    for (int c = 0; c < cn * 2; c++) {
        prow1[width * c] = prow1[width * c + width - 1] = tab[0];
        prow2[width * c] = prow2[width * c + width - 1] = tab[0];
    }

    const int n1 = y > 0 ? -(int)img1.step : 0;
    const int s1 = y < img1.rows - 1 ? (int)img1.step : 0;
    const int n2 = y > 0 ? -(int)img2.step : 0;
    const int s2 = y < img2.rows - 1 ? (int)img2.step : 0;

    int minX_cmn = std::min(minX1, minX2) - 1;
    int maxX_cmn = std::max(maxX1, maxX2) + 1;
    minX_cmn = std::max(minX_cmn, 1);
    maxX_cmn = std::min(maxX_cmn, width - 1);

    if (cn == 1) {
        for (int x = minX_cmn; x < maxX_cmn; x++) {
            prow1[x] = tab[(row1[x + 1] - row1[x - 1]) * 2 + row1[x + n1 + 1] - row1[x + n1 - 1] +
                           row1[x + s1 + 1] - row1[x + s1 - 1]];
            prow2[width - 1 - x] = tab[(row2[x + 1] - row2[x - 1]) * 2 + row2[x + n2 + 1] -
                                       row2[x + n2 - 1] + row2[x + s2 + 1] - row2[x + s2 - 1]];
            prow1[x + width] = row1[x];
            prow2[width - 1 - x + width] = row2[x];
        }
    } else {
        for (int x = minX_cmn; x < maxX_cmn; x++) {
            prow1[x] = tab[(row1[x * 3 + 3] - row1[x * 3 - 3]) * 2 + row1[x * 3 + n1 + 3] -
                           row1[x * 3 + n1 - 3] + row1[x * 3 + s1 + 3] - row1[x * 3 + s1 - 3]];
            prow1[x + width] = tab[(row1[x * 3 + 4] - row1[x * 3 - 2]) * 2 + row1[x * 3 + n1 + 4] -
                                   row1[x * 3 + n1 - 2] + row1[x * 3 + s1 + 4] - row1[x * 3 + s1 - 2]];
            prow1[x + width * 2] = tab[(row1[x * 3 + 5] - row1[x * 3 - 1]) * 2 + row1[x * 3 + n1 + 5] -
                                       row1[x * 3 + n1 - 1] + row1[x * 3 + s1 + 5] - row1[x * 3 + s1 - 1]];

            prow2[width - 1 - x] = tab[(row2[x * 3 + 3] - row2[x * 3 - 3]) * 2 + row2[x * 3 + n2 + 3] -
                                       row2[x * 3 + n2 - 3] + row2[x * 3 + s2 + 3] - row2[x * 3 + s2 - 3]];
            prow2[width - 1 - x + width] =
                tab[(row2[x * 3 + 4] - row2[x * 3 - 2]) * 2 + row2[x * 3 + n2 + 4] -
                     row2[x * 3 + n2 - 2] + row2[x * 3 + s2 + 4] - row2[x * 3 + s2 - 2]];
            prow2[width - 1 - x + width * 2] =
                tab[(row2[x * 3 + 5] - row2[x * 3 - 1]) * 2 + row2[x * 3 + n2 + 5] -
                     row2[x * 3 + n2 - 1] + row2[x * 3 + s2 + 5] - row2[x * 3 + s2 - 1]];

            prow1[x + width * 3] = row1[x * 3];
            prow1[x + width * 4] = row1[x * 3 + 1];
            prow1[x + width * 5] = row1[x * 3 + 2];

            prow2[width - 1 - x + width * 3] = row2[x * 3];
            prow2[width - 1 - x + width * 4] = row2[x * 3 + 1];
            prow2[width - 1 - x + width * 5] = row2[x * 3 + 2];
        }
    }

    memset(pixDiff, 0, static_cast<size_t>(width1) * D * sizeof(CostType));

    // OpenCV rebases both pointers so the inner loops can write `cost[x*D + d]`
    // and `buffer[x]` while iterating absolute coordinates. Keeping the rebase
    // as an integer offset avoids forming pointers outside the allocation.
    const long costOfs = -(long)minX1 * D - minD;
    const long bufOfs = (long)maxX2 - width;

    for (int c = 0; c < cn * 2; c++, prow1 += width, prow2 += width) {
        const int diff_scale = c < cn ? 0 : 2;

        for (int x = width - maxX2; x < width - minX2; x++) {
            const int v = prow2[x];
            const int vl = x > 0 ? (v + prow2[x - 1]) / 2 : v;
            const int vr = x < width - 1 ? (v + prow2[x + 1]) / 2 : v;
            int v0 = std::min(vl, vr);
            v0 = std::min(v0, v);
            int v1 = std::max(vl, vr);
            v1 = std::max(v1, v);
            tempBuf[bufOfs + x] = (PixType)v0;
            tempBuf[bufOfs + x + width2] = (PixType)v1;
        }

        for (int x = minX1; x < maxX1; x++) {
            const int u = prow1[x];
            const int ul = x > 0 ? (u + prow1[x - 1]) / 2 : u;
            const int ur = x < width - 1 ? (u + prow1[x + 1]) / 2 : u;
            int u0 = std::min(ul, ur);
            u0 = std::min(u0, u);
            int u1 = std::max(ul, ur);
            u1 = std::max(u1, u);

            for (int d = minD; d < maxD; d++) {
                const int v = prow2[width - x - 1 + d];
                const int v0 = tempBuf[bufOfs + width - x - 1 + d];
                const int v1 = tempBuf[bufOfs + width - x - 1 + d + width2];
                int c0 = std::max(0, u - v1);
                c0 = std::max(c0, v0 - u);
                int c1 = std::max(0, v - u1);
                c1 = std::max(c1, u0 - v);
                CostType& cell = pixDiff[costOfs + (long)x * D + d];
                // v_add(_c0, shifted diff) saturates.
                cell = add16(cell, (CostType)(std::min(c0, c1) >> diff_scale));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// BufferSGBM, MODE_SGBM instantiation only (fullDP == false).
// ---------------------------------------------------------------------------
class Buffer {
public:
    static constexpr int kNR = 8;
    static constexpr int kNR2 = kNR / 2;

    Buffer(int width1_, int Da_, int Dlra_, int cn, int width, int height, const Params& params) {
        width1 = width1_;
        Da = Da_;
        Dlra = Dlra_;
        fullDP = false;
        const int lane = alignLanes();
        dirs = (PixType)kNR;
        dirs2 = (PixType)kNR2;

        costWidth = (size_t)width1 * Da;
        costHeight = 1;  // !fullDP
        hsumRows = (size_t)params.calcSadWindowSize().height + 2;

        Cbuf.assign(costWidth * costHeight, 0);
        Sbuf.assign(costWidth * costHeight, 0);
        hsumBuf.assign(costWidth * hsumRows, 0);
        pixDiff.assign(costWidth, 0);
        disp2cost.assign(width, 0);
        disp2ptr.assign(width, 0);
        tempBuf.assign((size_t)width * (4 * cn + 2), 0);

        for (int i = 0; i < 2; ++i) {
            Lr[i].assign(lrCount() * Dlra, 0);
            minLr[i].assign(lrCount(), 0);
        }

        const size_t tabSize = 256 + (size_t)256 * 4 * 2;
        clipTab.assign(tabSize, 0);
        const int ftzero = std::max(params.preFilterCap, 15) | 1;
        for (int i = 0; i < (int)tabSize; i++)
            clipTab[i] = (PixType)(std::min(std::max(i - 256 * 4, -ftzero), ftzero) + ftzero);
        (void)lane;
    }

    const PixType* getClipTab() const { return clipTab.data() + 256 * 4; }

    void initCBuf(CostType val) { std::fill(Cbuf.begin(), Cbuf.end(), val); }

    void clearLr() {
        for (int i = 0; i < 2; ++i) {
            std::fill(Lr[i].begin(), Lr[i].end(), (CostType)0);
            std::fill(minLr[i].begin(), minLr[i].end(), (CostType)0);
        }
    }

    size_t lrCount() const { return (size_t)width1 * dirs2 + 2 * dirs; }

    void swapLr() {
        std::swap(Lr[0], Lr[1]);
        std::swap(minLr[0], minLr[1]);
    }

    CostType* getHSumBuf(int row) { return hsumBuf.data() + (row % hsumRows) * costWidth; }
    CostType* getCBuf(int /*row*/) { return Cbuf.data(); }
    CostType* getSBuf(int /*row*/) { return Sbuf.data(); }
    void clearSBuf(int /*row*/) { std::fill(Sbuf.begin(), Sbuf.end(), (CostType)0); }

    // Index 0 of the returned block is d == 0; d == -1 and d == D are the
    // sentinels the DP writes before each sweep.
    CostType* getLr(int id, int idx, int shift = 0) {
        return Lr[id].data() + (size_t)dirs * Dlra + ((size_t)idx * dirs2 + shift) * (size_t)Dlra;
    }
    CostType* getMinLr(int id, int idx, int shift = 0) {
        return minLr[id].data() + dirs + ((size_t)idx * dirs2 + shift);
    }

    size_t width1, Da, Dlra;
    size_t costWidth, costHeight, hsumRows;
    bool fullDP;
    PixType dirs, dirs2;

    std::vector<CostType> Cbuf, Sbuf, hsumBuf, pixDiff, disp2cost;
    std::vector<DispType> disp2ptr;
    std::vector<PixType> tempBuf;
    std::vector<CostType> Lr[2], minLr[2];
    std::vector<PixType> clipTab;
};

}  // namespace

int simdInt16Lanes() { return alignLanes(); }

StageDump makeStageDump(const DumpConfig& cfg, int height) {
    StageDump dump;
    dump.capturePixDiff = cfg.capturePixDiff;
    dump.budget = cfg.pixDiffByteBudget;
    for (int y : cfg.rows) {
        if (y < 0 || y >= height) continue;
        // Keep one capture per row: the pipeline callback below matches by row
        // index, and duplicates would leave all but the last copy empty.
        if (std::any_of(dump.rows.begin(), dump.rows.end(),
                        [y](const RowCapture& rc) { return rc.y == y; }))
            continue;
        RowCapture rc;
        rc.y = y;
        dump.rows.push_back(rc);
    }
    return dump;
}

// ---------------------------------------------------------------------------
// computeDisparitySGBM, MODE_SGBM only (one pass, 4 forward + 1 reverse
// direction).
// ---------------------------------------------------------------------------
void computeDisparitySGBM(const cv::Mat& img1, const cv::Mat& img2, cv::Mat& disp1,
                         const Params& params, StageDump* dump) {
    CV_Assert(img1.type() == CV_8UC1 && img2.type() == CV_8UC1 && img1.size() == img2.size());
    // cv::StereoSGBM::compute() creates the output before dispatching; mirror it
    // so the width/height below are meaningful.
    if (disp1.size() != img1.size() || disp1.type() != CV_16S)
        disp1.create(img1.size(), CV_16S);

    int minD = params.minDisparity;
    const int maxD = minD + params.numDisparities;
    const int uniquenessRatio = params.uniquenessRatio >= 0 ? params.uniquenessRatio : 10;
    const int disp12MaxDiff = params.disp12MaxDiff > 0 ? params.disp12MaxDiff : 1;
    const int P1 = params.P1 > 0 ? params.P1 : 2;
    const int P2 = std::max(params.P2 > 0 ? params.P2 : 5, P1 + 1);
    const int width = disp1.cols;
    const int height = disp1.rows;

    const int lane = alignLanes();
    const int minX1 = std::max(maxD, 0);
    const int maxX1 = width + std::min(minD, 0);
    const int D = params.numDisparities;
    int width1 = maxX1 - minX1;
    const int Da = alignSizeTo(D, lane);
    const int Dlra = Da + lane;
    const int INVALID_DISP = minD - 1;
    const int INVALID_DISP_SCALED = INVALID_DISP * kDispScale;

    const int SW2 = params.calcSadWindowSize().width / 2;
    const int SH2 = params.calcSadWindowSize().height / 2;

    if (width - (params.minDisparity + params.numDisparities) <=
        params.calcSadWindowSize().width / 2) {
        throw std::runtime_error(
            "image too small for the given window size and disparity range");
    }

    if (minX1 >= maxX1) {
        disp1 = cv::Scalar::all(INVALID_DISP_SCALED);
        return;
    }

    Buffer mem(width1, Da, Dlra, img1.channels(), width, height, params);
    mem.initCBuf((CostType)P2);

    auto captureRow = [&](int y, bool forwardOnly) {
        if (!dump) return;
        RowCapture* cap = nullptr;
        for (auto& rc : dump->rows)
            if (rc.y == y) cap = &rc;
        if (!cap) return;
        const size_t n = mem.costWidth;
        cap->C.assign(mem.getCBuf(y), mem.getCBuf(y) + n);
        if (forwardOnly) {
            cap->SForward.assign(mem.getSBuf(y), mem.getSBuf(y) + n);
        } else {
            cap->S.assign(mem.getSBuf(y), mem.getSBuf(y) + n);
        }
    };

    {
        // pass == 1
        const int y1 = 0, y2 = height, dy = 1;
        const int x1 = 0, x2 = width1, dx = 1;

        int lrID = 0;
        mem.clearLr();

        for (int y = y1; y != y2; y += dy) {
            DispType* disp1ptr = disp1.ptr<DispType>(y);
            CostType* const C = mem.getCBuf(y);
            CostType* const S = mem.getSBuf(y);

            const int dy1 = y == 0 ? 0 : y + SH2;
            const int dy2 = y == 0 ? SH2 : dy1;

            for (int k = dy1; k <= dy2; k++) {
                CostType* hsumAdd = mem.getHSumBuf(std::min(k, height - 1));

                if (k < height) {
                    calcPixelCostBT(img1, img2, k, minD, maxD, mem.pixDiff.data(),
                                    mem.tempBuf.data(), mem.getClipTab());

                    if (dump && dump->capturePixDiff) {
                        const size_t bytes = mem.costWidth * sizeof(CostType);
                        size_t used = 0;
                        for (const auto& v : dump->pixDiff) used += v.size() * sizeof(CostType);
                        const bool wanted =
                            dump->rows.empty() ||
                            std::find_if(dump->rows.begin(), dump->rows.end(), [&](const RowCapture& rc) {
                                return rc.y == k;
                            }) != dump->rows.end();
                        if (wanted && used + bytes <= dump->budget) {
                            dump->pixDiffRows.push_back(k);
                            dump->pixDiff.emplace_back(mem.pixDiff.begin(),
                                                       mem.pixDiff.begin() + mem.costWidth);
                        }
                    }

                    memset(hsumAdd, 0, Da * sizeof(CostType));
                    {
                        // v_mul then repeated v_add, all saturating.
                        const short hScale = (short)(SW2 + 1);
                        for (int d = 0; d < D; d++) {
                            short acc = mul16(mem.pixDiff[d], hScale);
                            for (int x = Da; x <= SW2 * Da; x += Da)
                                acc = add16(acc, mem.pixDiff[x + d]);
                            hsumAdd[d] = acc;
                        }
                    }

                    if (y > 0) {
                        const CostType* hsumSub = mem.getHSumBuf(std::max(y - SH2 - 1, 0));
                        const CostType* Cprev = mem.getCBuf(y - 1);

                        // v_add(Cprev, hsumAdd) then v_sub(.., hsumSub), both saturating.
                        for (int d = 0; d < D; d++)
                            C[d] = sub16(add16(Cprev[d], hsumAdd[d]), hsumSub[d]);

                        for (int x = Da; x < width1 * Da; x += Da) {
                            const CostType* pixAdd =
                                mem.pixDiff.data() + std::min(x + SW2 * Da, (width1 - 1) * Da);
                            const CostType* pixSub =
                                mem.pixDiff.data() + std::max(x - (SW2 + 1) * Da, 0);
                            for (int d = 0; d < D; d++) {
                                const short hv = add16(sub16(hsumAdd[x - Da + d], pixSub[d]), pixAdd[d]);
                                hsumAdd[x + d] = hv;
                                C[x + d] = add16(sub16(Cprev[x + d], hsumSub[x + d]), hv);
                            }
                        }
                    } else {
                        // Note the operand order differs from the y > 0 branch:
                        // here it is v_sub(v_add(hsumAdd, pixAdd), pixSub).
                        const short scale = (short)(k == 0 ? SH2 + 1 : 1);
                        for (int d = 0; d < D; d++)
                            C[d] = add16(C[d], mul16(hsumAdd[d], scale));

                        for (int x = Da; x < width1 * Da; x += Da) {
                            const CostType* pixAdd =
                                mem.pixDiff.data() + std::min(x + SW2 * Da, (width1 - 1) * Da);
                            const CostType* pixSub =
                                mem.pixDiff.data() + std::max(x - (SW2 + 1) * Da, 0);
                            for (int d = 0; d < D; d++) {
                                const short hv = sub16(add16(hsumAdd[x - Da + d], pixAdd[d]), pixSub[d]);
                                hsumAdd[x + d] = hv;
                                C[x + d] = add16(C[x + d], mul16(hv, scale));
                            }
                        }
                    }
                } else {
                    // Bottom edge (k >= height): hsumAdd already points at the
                    // last real row because getHSumBuf() clamps k, so C is
                    // rebuilt from it without recomputing any cost. Note the
                    // operand order matches the sliding step above
                    // (v_sub then v_add), not the row-head step.
                    if (y > 0) {
                        const CostType* hsumSub = mem.getHSumBuf(std::max(y - SH2 - 1, 0));
                        const CostType* Cprev = mem.getCBuf(y - 1);
                        for (int x = 0; x < width1 * Da; x++)
                            C[x] = add16(sub16(Cprev[x], hsumSub[x]), hsumAdd[x]);
                    } else {
                        for (int x = 0; x < width1 * Da; x++)
                            C[x] = add16(C[x], hsumAdd[x]);
                    }
                }
            }

            mem.clearSBuf(y);

            // Forward sweep: directions 0..3.
            //
            // The intrinsics path materialises each delta with
            // vx_setall_s16((short)delta), i.e. a truncating narrowing, and does
            // the recurrence in saturating 16-bit. Both details are reproduced
            // here; delta can legitimately exceed SHRT_MAX when P2 is large.
            for (int x = x1; x != x2; x += dx) {
                const short delta0 = trunc16(P2 + *mem.getMinLr(lrID, x - dx));
                const short delta1 = trunc16(P2 + *mem.getMinLr(1 - lrID, x - 1, 1));
                const short delta2 = trunc16(P2 + *mem.getMinLr(1 - lrID, x, 2));
                const short delta3 = trunc16(P2 + *mem.getMinLr(1 - lrID, x + 1, 3));

                CostType* Lr_p0 = mem.getLr(lrID, x - dx);
                CostType* Lr_p1 = mem.getLr(1 - lrID, x - 1, 1);
                CostType* Lr_p2 = mem.getLr(1 - lrID, x, 2);
                CostType* Lr_p3 = mem.getLr(1 - lrID, x + 1, 3);

                Lr_p0[-1] = Lr_p0[D] = kMaxCost;
                Lr_p1[-1] = Lr_p1[D] = kMaxCost;
                Lr_p2[-1] = Lr_p2[D] = kMaxCost;
                Lr_p3[-1] = Lr_p3[D] = kMaxCost;

                CostType* Lr_p = mem.getLr(lrID, x);
                const CostType* Cp = C + x * Da;
                CostType* Sp = S + x * Da;

                CostType* minL = mem.getMinLr(lrID, x);
                minL[0] = minL[1] = minL[2] = minL[3] = kMaxCost;

                const short P1s = trunc16(P1);
                for (int d = 0; d < D; d++) {
                    const short Cpd = Cp[d];
                    short Spd = Sp[d];

                    // v_add(v_sub(v_min(v_min(v_min(Lr[d], Lr[d-1]+P1),
                    //                              Lr[d+1]+P1), delta), delta), C)
                    auto step = [&](const CostType* Lp, short delta) -> short {
                        const short t = std::min(std::min(Lp[d], add16(Lp[d - 1], P1s)),
                                                 std::min(add16(Lp[d + 1], P1s), delta));
                        return add16(sub16(t, delta), Cpd);
                    };

                    const short L0 = step(Lr_p0, delta0);
                    Lr_p[d] = L0;
                    minL[0] = std::min(minL[0], L0);
                    Spd = add16(Spd, L0);

                    const short L1 = step(Lr_p1, delta1);
                    Lr_p[d + Dlra] = L1;
                    minL[1] = std::min(minL[1], L1);
                    Spd = add16(Spd, L1);

                    const short L2 = step(Lr_p2, delta2);
                    Lr_p[d + Dlra * 2] = L2;
                    minL[2] = std::min(minL[2], L2);
                    Spd = add16(Spd, L2);

                    const short L3 = step(Lr_p3, delta3);
                    Lr_p[d + Dlra * 3] = L3;
                    minL[3] = std::min(minL[3], L3);
                    Spd = add16(Spd, L3);

                    Sp[d] = Spd;
                }
            }

            captureRow(y, /*forwardOnly=*/true);

            // pass == npasses (npasses == 1): reverse sweep supplies direction 4
            // and picks the winner in the same walk.
            {
                for (int x = 0; x < width; x++) {
                    disp1ptr[x] = mem.disp2ptr[x] = (DispType)INVALID_DISP_SCALED;
                    mem.disp2cost[x] = kMaxCost;
                }

                for (int x = width1 - 1; x >= 0; x--) {
                    CostType* Sp = S + x * Da;
                    CostType minS = kMaxCost;
                    short bestDisp = -1;

                    {
                        CostType* Lr_p0 = mem.getLr(lrID, x + 1);
                        Lr_p0[-1] = Lr_p0[D] = kMaxCost;
                        CostType* Lr_p = mem.getLr(lrID, x);

                        const CostType* Cp = C + x * Da;

                        int d = 0;
                        const short delta0 = trunc16(P2 + *mem.getMinLr(lrID, x + 1));
                        const short P1s = trunc16(P1);
                        short minL0 = kMaxCost;

                        for (; d < D; d++) {
                            const short t = std::min(std::min(Lr_p0[d], add16(Lr_p0[d - 1], P1s)),
                                                     std::min(add16(Lr_p0[d + 1], P1s), delta0));
                            const short L0 = add16(sub16(t, delta0), Cp[d]);

                            Lr_p[d] = L0;
                            minL0 = std::min(minL0, L0);

                            const short Sval = add16(Sp[d], L0);
                            Sp[d] = Sval;
                            if (Sval < minS) {
                                minS = Sval;
                                bestDisp = (short)d;
                            }
                        }
                        *mem.getMinLr(lrID, x) = minL0;
                    }

                    int d = 0;
                    for (; d < D; d++) {
                        if (Sp[d] * (100 - uniquenessRatio) < minS * 100 &&
                            std::abs(bestDisp - d) > 1)
                            break;
                    }
                    if (d < D) continue;

                    d = bestDisp;
                    const int _x2 = x + minX1 - d - minD;
                    if (_x2 < 0 || _x2 >= width) continue;  // guard; OpenCV indexes disp2cost
                    if (mem.disp2cost[_x2] > minS) {
                        mem.disp2cost[_x2] = (CostType)minS;
                        mem.disp2ptr[_x2] = (DispType)(d + minD);
                    }

                    if (0 < d && d < D - 1) {
                        const int denom2 = std::max(Sp[d - 1] + Sp[d + 1] - 2 * Sp[d], 1);
                        d = d * kDispScale +
                            ((Sp[d - 1] - Sp[d + 1]) * kDispScale + denom2) / (denom2 * 2);
                    } else {
                        d *= kDispScale;
                    }
                    disp1ptr[x + minX1] = (DispType)(d + minD * kDispScale);
                }

                for (int x = minX1; x < maxX1; x++) {
                    const int d1 = disp1ptr[x];
                    if (d1 == INVALID_DISP_SCALED) continue;
                    const int _d = d1 >> kDispShift;
                    const int d_ = (d1 + kDispScale - 1) >> kDispShift;
                    const int _x = x - _d;
                    const int x_ = x - d_;
                    if (0 <= _x && _x < width && mem.disp2ptr[_x] >= minD &&
                        std::abs(mem.disp2ptr[_x] - _d) > disp12MaxDiff && 0 <= x_ && x_ < width &&
                        mem.disp2ptr[x_] >= minD && std::abs(mem.disp2ptr[x_] - d_) > disp12MaxDiff)
                        disp1ptr[x] = (DispType)INVALID_DISP_SCALED;
                }
            }

            captureRow(y, /*forwardOnly=*/false);
            lrID = 1 - lrID;
        }
    }

    if (dump) {
        dump->width1 = (int)mem.width1;
        dump->Da = (int)mem.Da;
        dump->D = D;
        dump->minX1 = minX1;
        dump->disparityBeforeEpilogue = disp1.clone();
    }

    // StereoSGBMImpl::compute() applies the epilogue unconditionally.
    epilogue(disp1, params);

    if (dump) dump->disparity = disp1.clone();
}

void epilogue(cv::Mat& disp, const Params& params) {
    cv::medianBlur(disp, disp, 3);
    if (params.speckleWindowSize > 0) {
        cv::Mat buf;
        cv::filterSpeckles(disp, (params.minDisparity - 1) * kDispScale, params.speckleWindowSize,
                           kDispScale * params.speckleRange, buf);
    }
}

}  // namespace oracle
