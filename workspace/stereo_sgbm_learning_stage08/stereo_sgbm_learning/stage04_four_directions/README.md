# Stage 04 — 四方向 SGM 代价聚合

本阶段使用上一阶段的 `aggregate_one_step()`（论文公式 13）实现左→右、右→左、上→下、下→上四条路径方向，并依据公式 (14) 聚合：

\[
S(p,d) = L_{\rightarrow}(p,d)+L_{\leftarrow}(p,d)+L_{\downarrow}(p,d)+L_{\uparrow}(p,d)
\]

## 快速运行

```bash
cd stereo_sgbm_learning/stage04_four_directions
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

如果系统安装 OpenCV 开发库（Ubuntu: `sudo apt install libopencv-dev`），还可以运行影像演示：

```bash
./build/handmade_sgm_4path \
  ~/project/stereo_oven_sim/output/left.png \
  ~/project/stereo_oven_sim/output/right.png \
  ./out \
  48 80 5 200 800 \
  ~/project/stereo_oven_sim/output/disparity_gt_left_16.png \
  ~/project/stereo_oven_sim/output/valid_stereo_mask.png
```

参数依次为 `minDisp numDisp blockSize P1 P2`。示例值属于演示初始值，并非最优参数。图像要求为经过极线校正、相同分辨率的左右图像。

输出文件：`bm_disp16.png`, `sgm_1path_disp16.png`, `sgm_4path_disp16.png`，以及 `_valid.png`、`_vis.png`；`bm_vs_4path_difference.png` 和 `center_cost_curve.csv`（原始代价/单路径/四路径）。

## 逐行对应源码

1. `scan_direction()` 负责遍历方向，注意水平方向每行 reset，垂直方向每列 reset。
2. `aggregate_one_step()` 负责完成一个像素的 D 个状态更新，保持 Stage03 原样。
3. `aggregate_one_direction()` 输出一个完整的方向代价体 `L_r`，方便调试。
4. `aggregate_four_directions()` 逐方向执行，同时累加到单个 `S`（不会同时保存四份完整的 `L_r`）。
5. `winner_takes_all(S)` 从每个像素的 `D` 个聚合代价中选择最小者作为整数视差。

### 小例子

像素排列（W=3,H=2,D=3；候选视差 10,11,12；P1=5,P2=20）：

```text
       x=0      x=1      x=2
y=0   [10,0,12] [0,3,4] [1,5,3]
y=1   [4,0,6]  [2,3,5] [3,1,4]
```

例如位于 `(x=1,y=0)` 的 B 像素，各方向结果为：

```text
L_left_to_right(B) = [5, 3, 9]
L_right_to_left(B)= [0, 7, 6]
L_top_to_bottom(B)= [0, 3, 4]   // 第一行，没有上方前驱
L_bottom_to_top(B)= [0, 4, 7]
S(B)               = [5,17,26]
```

所以四方向聚合后 `WTA(B)=d=10`。单路径时 `WTA(B)=d=11`，说明路径方向会改变选择。所有这些期望值都经过单元测试逐元素校验。

## 关键说明

- **论文公式 (14) 对四方向的 L 直接求和**，因此当前版本并不去掉重复计算的 C。所有有效像素都在四条路径各包含一次原始 C。
- 输入候选无效（`kInvalidCost`）时，输出 S 仍然无效，绝不会把不可匹配位置当成最优视差。
- 如果 `P1=P2=0`，每条路径 `L=C`，所以 `S=4C`；WTA 与原始 SAD 完全相同。
- 代码保留上一阶段 `[H][W][D]` 内存布局。常规四方向聚合额外只申请一个 S 全量缓冲区和 O(D) 的单路径状态。
- 这只是教学用四方向 **SGM**，尚非 OpenCV StereoSGBM（BT、窗口代价、五方向扫描、亚像素和左右一致性后处理等尚未实现）。
- 四方向结果并不保证在所有图像上都比 SAD-BM 更准确，参数及弱纹理区域特性会产生明显影响。

## 练习

1. 尝试删除 `previous.swap(current)` 看看结果为何错误。
2. 删除每条路径开始的 reset，观察两个相邻行/列互相串扰。
3. 将 `P1=P2=0`，检查四方向总代价应恰好等于 `4*C`。
4. 不看函数的实现，独立补全 `BottomToTop` 方向遍历代码。
