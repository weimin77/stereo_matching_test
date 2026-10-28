# Stage 08 — Uniqueness Check + Speckle Filtering

继承 Stage 01～07，不修改已有算法函数。新增 `include/sgm_filters.hpp`，包含两个独立的过滤步骤。**纯 C++ 核心不依赖 OpenCV**；OpenCV 只用于演示程序图像 I/O。

## 知识对应关系

- Hirschmüller (2008), *Stereo Processing by Semiglobal Matching and Mutual Information*, §2.5.1 Removal of Peaks：四邻域 + 相邻视差差 <=1 pixel 的连通域，过滤过小区域。
- OpenCV `StereoSGBM::uniquenessRatio`：用于检查其他候选视差是否有与最优视差接近的聚合代价。本教程实现显式排除最优候选两侧 ±1 index 后的相对裕量；**不承诺与 OpenCV 在阈值的所有边界、整数舍入上完全一致**。
- OpenCV `speckleWindowSize`, `speckleRange`：前者是要剔除的小连通分量的最大像素数（不是窗口宽度），后者是相邻像素视差容差（单位 pixel）。OpenCV CV_16S/Q4 视差会在调用过滤函数时乘以 16；本教程 float 视差不用。

## 处理流水线

```
Stereo rectified images
    └── Stage02 SAD CostVolume C
            └── Stage05 Eight-direction aggregated costs S
                    ├── Stage06 WTA + subpixel -> D_L
                    ├── Stage08 uniqueness_filter(S, D_L)  [读 S]
                    └── Stage07 independently computed D_R
                            └── Stage07 LRC         [读 D_L, D_R]
                                    └── Stage08 Speckle Filter [读 D]
```

- 唯一性检查依赖完整的候选视差曲线 `S(x,y,d)`，**不能从只有一个 float 视差的结果中恢复**。
- Speckle Filtering 依赖视差图和有效掩码，不需要 S。
- 唯一性检查选择在 LRC 之前；Speckle Filtering 在 LRC 之后。
- 此演示仍沿用 Stage07 独立计算右视差的方式，额外的唯一性及 Speckle 处理本身不会重做一遍 SGM。可在后续性能优化阶段复用左侧 S 计算快速右视差。

## 数学与实现细节

### 1. Uniqueness

找到最优整数视差索引 `di_best` 及其代价 `best`，从 `abs(di-di_best)>excluded_radius` 的候选视差中寻找最小有效竞争代价 `other`。默认 `excluded_radius=1`。

如果 `ratio_percent=0`，此阶段停用（但保持输入中本来无效的像素无效）。

如果有竞争代价，要求：

```
100 * (other - best) > ratio_percent * best
```

否则剔除该像素。所有计算使用 int64 防止溢出；`best==0, other==0` 判为不唯一（不会除零）。如果没有任何非局部有效候选，状态为 `NoNonlocalCompetitor`，保留它——这不是证明它可靠，只是本检查无法提供反证。

`ratio_percent=10` 代表希望非局部候选比最优代价高出 **10% 以上**；它与 OpenCV 具体整数阈值/等号行为略有差异。

### 2. Speckle

以四邻域 BFS 进行连通分量分析。

邻接条件：

```
valid(a) && valid(b) && abs(disparity[a]-disparity[b]) <= max_diff_px
```

这里是逐条邻边比较，因此连通分量整体的最大值与最小值可以相差 > max_diff_px。

如果某个连通分量包含的像素数 `<= max_component_size`，全部置为无效：`valid=0, disparity=NaN, refined=0`。`max_component_size=0` 禁用。

复杂度 `O(W*H)`，辅助空间 `O(W*H)`。对于真实小食材、细长物体、细碎蔬菜块，要谨慎调节阈值，否则会把真实目标也剔除。

## 编译

```bash
cd stereo_sgbm_learning/stage08_filters
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

没有安装 OpenCV 时仍会编译 `test_sgm_filters`，图像演示程序不构建。如果需要实际图像输入/输出，先安装系统 `libopencv-dev`，并重新配置 CMake。

## BlenderProc 例子

```bash
./build/handmade_sgm_filters \
    ~/project/stereo_oven_sim/output/left.png \
    ~/project/stereo_oven_sim/output/right.png \
    ./out \
    48 80 5 200 800 1.0 10 50 1.0 \
    ~/project/stereo_oven_sim/output/disparity_gt_left_16.png \
    ~/project/stereo_oven_sim/output/valid_stereo_mask.png
```

参数顺序：

```
left.png right.png outdir
minDisparity numDisparities blockSize P1 P2
lrThreshold uniquenessRatio speckleMaxSize speckleMaxDiffPx
[gt_disp16.png] [gt_valid_mask.png]
```

取值 `48,80,5,200,800,1.0,10,50,1.0` 是**起点，不保证最优**。可将 `uniquenessRatio=0` 和 `speckleMaxSize=0` 关闭两类过滤，比较前后差异。

## 主要输出文件

- `left_before_filters_vis.png`：Stage06 计算出的左视差。
- `left_after_uniqueness_vis.png`：经过唯一性检查。
- `right_disparity_vis.png`：Stage07 独立右视差。
- `left_after_lrc_vis.png`：经过 LRC。
- `left_after_speckle_vis.png`：最终输出。
- `uniqueness_status.png`：灰度状态图：0 输入无效；64 匹配不明确；128 无非局部候选；255 通过唯一性。
- `speckle_removed.png`：255 为本步删除的像素，0 为其他。
- 每步还输出 `.pfm` (float32 disparity pixel, NaN 表示无效)、`*_disp16.png` (uint16 disparity ×256)、`*_valid.png`、`*_refined.png`。

程序会输出各阶段的有效像素数；给定 GT 后，输出各阶段 EPE、Bad1、coverage。**过滤后 EPE 下降可能是剔除了难点，须结合 coverage 一起判断。**

## 注意事项

1. 教程使用的八方向全代价体算法 **不是** OpenCV `MODE_SGBM` 默认方向与内存优化方案的逐行复制。
2. 强纹理区域不一定通过唯一性；两个严重错误但互相一致的视差也可能通过 LRC，过滤是启发式可靠性判断。
3. 对需要测量体积的食材，过滤产生的空洞应计入后续表面补全或缺失数据处理流程，不能直接赋零深度。
4. 当 `minDisparity` 使视差 >255 时，`*_disp16.png` 的 ×256 表示会超出 16 位范围；`.pfm` 浮点更通用。与 OpenCV `CV_16S` 的 ×16 不是同一比例。

## 建议练习

- 手算候选代价 [100,105,150,190] 与 [100,101,105,150] 在 10% 阈值下的唯一性结果。
- 构建 5×5 视差图，手工标出所有四连通区域，验证移除前后的掩码。
- 用 BlenderProc GT 绘制 `uniquenessRatio = 0, 5, 10, 15` 时 EPE-coverage 曲线；尝试调节 `speckleMaxSize = 0, 20, 50, 100`。

下一阶段：Birchfield–Tomasi (BT) 匹配代价及与 OpenCV SGBM 源码对照。
