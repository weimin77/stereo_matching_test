# ADCensusBM 参数调优报告

日期：2026-10-10
对象：`test.cpp` 中 `ADCENSUS` 区域 + `ADCensusBM/`（Mei et al. ICCV2011 ADCensus）
配置场景：`Backpack-perfect`（2940x2016, ndisp=260），S=4 → 735×504，dMax=65

## 1. 结论（先看这个）

修改了 `test.cpp` 里 `namespace adc` 的 7 个参数（见第 5 节）。效果：

| 指标（Backpack, S=4） | 修改前 | 修改后 |
|---|---|---|
| **bad1all**（含缺失像素的 >1px 错误率） | 22.64 % | **14.52 %** |
| bad1val（只统计有效像素） | 22.26 % | 14.14 % |
| bad2all | 15.25 % | 11.04 % |
| coverage | 99.50 % | 99.56 % |
| RMSE | 3.916 | 3.681 |
| 中位绝对误差 | 0.302 px | **0.228 px** |
| 匹配耗时 | 58 s | 45 s |

跨 8 个 Middlebury 2014 场景的均值：bad1all **26.05 → 23.11**；留出场景 Cable（未参与调参）
**23.21 → 19.50**。

## 2. 问题诊断

原始输出的错误**高度结构化**，不是随机噪声：中位误差仅 0.30 px，但仍有 22 % 像素错 >1 px。
按区域统计（Backpack, S=8）：

- 列方向：左 60 % 列 bad1all 17–30 %，右 38 % 列 <1.5 %
- 行方向：上 3 行带 4–6 %，下 3 行带 31–39 %
- 按 GT 视差：视差 10–20 px 区间最差（26–37 %）

这种「中位很好但尾部很重」的模式是**视差在物体边界上被抹开（bleeding）**的典型特征。

同时做了参照：OpenCV `StereoSGBM`(MODE_HH) 在同一对上 S=8 bad1all 23.5 %／覆盖 89 %，
S=4 bad1all 23.2 %。**原 ADCensus 只是和 SGBM 打平**，并没有发挥 ADCensus 应有的优势。

## 3. 方法

写了一个临时 harness（链接 ADCensusBM 的 5 个 .cpp，不走 test.cpp，用 argv 覆盖参数），
在 S=8（367×252，~1.2 s/次）上扫参，再在 S=4 上确认。用完已删除。

单参数扫描的关键结果（S=8, Backpack）：

| 参数 | 影响 | 最佳方向 |
|---|---|---|
| **colorThreshold1/2** | **20/6 → 8/3 把 bad1all 从 22.7 降到 16.8** | **变小（最重要）** |
| maxLength1/2 | 34/17 → 10/5 → 16.3 | 变小 |
| aggregatingIterations | 4 → 1 → 20.4 | 变小 |
| lambdaAD | 10 → 3 → 22.0 | 略变小 |
| dispTolerance | 0 → 3 → 21.5 | 略变大 |
| censusWin / pi / lambdaCensus / 投票 / Canny / dMax | 影响 <0.3 pt | — |

「变小」的两类参数（`colorThreshold`、`maxLength`、迭代次数）都在**缩小代价聚合支撑域**，
与第 2 节的 bleeding 诊断一致。

## 4. 为什么最终没有选最激进的那组

`ct=20/6, ml=10/5, it=1` 能把 Backpack 压到 13.4（比最终结果还好 1 个百分点），但**跨场景不成立**：

| 场景 | 默认 | 激进组 | 最终组 |
|---|---|---|---|
| Adirondack | 15.31 | 11.72 | 11.59 |
| Backpack | 22.73 | 14.99 | 14.64 |
| Bicycle1 | 25.85 | 24.36 | 24.00 |
| **Classroom1** | **8.47** | **17.19** | 9.70 |
| Couch | 37.87 | 29.48 | 30.85 |
| Flowers | 36.63 | 40.34 | 38.47 |
| Mask | 28.61 | 21.03 | 22.66 |
| mean | 26.05 | 25.33 | **23.11** |

**根本矛盾**：聚合窗口大小需求取决于场景纹理
- Classroom1（大面积平坦墙面）需要**大支撑**（it=4, ml=34, ct=20）
- Couch / Flowers / Jadeplant / Mask（枝叶、细结构）需要**小支撑**（ct=6–8, ml≤14, it=1–2）

因此不存在「一组通吃」的参数。最终方案选择了**唯一的单参数 `colorThreshold` 变小**（跨场景均值最优），
再叠加若干在所有场景都一致变好、且对留出的 Cable 也有效的小改动。

另外验证：聚合窗口的**像素值不需要随降采样倍数 S 缩放**（S=2/4/8 上最优 `maxLength` 相同）。

## 5. 最终参数（`test.cpp` 的 `namespace adc`）

```cpp
censusWinH = 7, censusWinW = 7;              // 原 9, 7
defaultBorderCost = 0.8f;                    // 原 0.999
colorThreshold1 = 8, colorThreshold2 = 3;    // 原 20, 6   <-- 主要改动
colorDifference = 35;                        // 原 15
votingThreshold = 10;                        // 原 20
votingRatioThreshold = 0.3f;                 // 原 0.4
maxSearchDepth = 40;                         // 原 20
blurKernelSize = 5;                          // 原 3
// 未改动：lambdaAD=10, lambdaCensus=30, aggregatingIterations=4,
//         maxLength1/2=34/17, pi1/pi2=0.1/0.3, dispTolerance=0, canny=20/60/3
```

约束：必须 `colorThreshold2 < colorThreshold1`、`maxLength2 < maxLength1`，否则 `init()` 拒绝配置。

## 6. 遗留问题与后续建议

1. **`dMin` 必须保持 0**。实测 `dMin=1` → bad1all 57 %，`dMin=3` → 98 %。
   原因：`costInitialization()` 里 `wR = w - d`（按 0 基下标偏移），而 `cost2disparity()`
   把标签写成 `d + dMin`，两者在 `dMin>0` 时不自洽。若要用 `vmin` 收窄搜索范围，需先修这里。
2. **`subpixelEnhancement()` 里的 `medianBlur(dispTemp, dispTemp, 3)`** 会把遮蔽/误匹配哨兵值
   （`dMin-1` / `dMin-2`）与有效视差混合，抹掉细结构——正是误差集中的地方。去掉或改成
   只对有效像素做中值滤波，值得单独试一次（属于代码改动，本次未做）。
3. **按场景自适应**是提升天花板的正路：可考虑根据纹理强度/梯度自适应选择
   「大支撑」与「小支撑」两组参数，而不是全局固定。
4. Flowers / Jadeplant / Couch 的 bad1all 仍在 30–38 %：这些场景视差范围极大
   （vmax 600–619，ndisp 630–640），是经典方法在 Middlebury 2014 上的普遍难点，
   靠调参收益有限。

## 7. 复现

```bash
cd /home/min/project/test && cmake --build build -j4
cd build && ./algorithm_test
# 输出 adc_disparity.png / adc_error.png / adc_cloud_m.ply 等，并打印
#   adcensus  cov=.. bad1all=.. bad1val=.. bad2all=.. rmse=.. med=..
```
