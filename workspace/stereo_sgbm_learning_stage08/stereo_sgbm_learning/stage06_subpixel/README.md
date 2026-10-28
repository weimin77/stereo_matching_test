# Stage 06 — Subpixel Refinement（亚像素视差）

论文：Hirschmüller (2008), Section 2.3（PDF 第 4 页下部，论文正文页码 331）。

本节在 Stage05 的八方向聚合代价体 `S` 后使用三点抛物线插值：

```text
S(p,d) -> argmin -> integer d0 -> S(p,d0-1),S(p,d0),S(p,d0+1)
       -> delta = (Cminus-Cplus)/(2*(Cminus-2*C0+Cplus)) -> d0+delta
```

`d0` 必须使用实际视差（`min_disparity + best_di`），不能把数组索引直接当作视差。

## 实现文件

* `include/subpixel_refinement.hpp`：纯 C++，依赖前面阶段的头文件。
* `tests/test_subpixel_refinement.cpp`：手算算例、负方向偏移、最小视差偏移、无效代价、边界、正比例缩放、完整合成双目流程。
* `src/main.cpp`：OpenCV 读写示例，对照整数 WTA 与亚像素，评估 x256 视差 GT。

## 编译

```bash
cd stereo_sgbm_learning/stage06_subpixel
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

## 运行（若找到了 OpenCV 开发包）

```bash
./build/handmade_sgm_subpixel left.png right.png out 48 80 5 200 800 [gt_disp16.png] [valid_stereo_mask.png]
```

输入：左右灰度或 RGB 图像（程序读成灰度），要求已完成极线校正。`P1=200` 和 `P2=800` 是实验初值，不代表最佳配置。

输出：

* `sgm_8path_integer_disp16.png`：整数 WTA 视差 ×256，保存为 16 位 PNG。
* `sgm_8path_subpixel_disp16.png`：亚像素视差 ×256 四舍五入，保存为 16 位 PNG。**该文件并非 OpenCV StereoSGBM 标准 ×16 格式**。
* `sgm_8path_subpixel.pfm`：完整的 float32 视差（像素为单位），PFM 支持 NaN 无效值，逐行由下往上储存。
* `sgm_8path_subpixel_valid.png`：有效掩码（白色有效，黑色无效）。
* `sgm_8path_subpixel_refined.png`：本阶段实际产生非零浮点偏移的像素。
* `sgm_8path_subpixel_vis.png`：伪彩色可视化。
* `sgm_8path_subpixel_minus_integer.png`：以灰度 128 表示 0 偏移，0 表示 -0.5，255 表示 +0.5，仅有效像素。
* `center_cost_curve.csv`：中心像素候选视差对应的八方向聚合代价。

说明：本工程是**教学版标准抛物线拟合**，并未复现 OpenCV 的所有定点细节、边界/唯一性筛选。本阶段尚无左右一致性检查，后续 Stage 07 学习。

### 默认回退到整数视差的情况

* 最优视差是搜索范围的第一个或最后一个候选。
* 最优视差左右任一个代价为无效代价。
* 二阶差分为 0、负值，或不超过设置的 `min_curvature`。
* 所有候选都无效时，`valid=0` 且浮点视差为 NaN。

论文中的二次拟合是对通用匹配代价的工程近似；不要把亚像素输出视为真实精度保证。与 GT 的评估要同时报告 EPE 和有效覆盖率。
