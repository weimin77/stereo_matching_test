# Stage 05 · 八方向 SGM（论文 2008 年 Section 2.2）

本阶段在 Stage04 的 **4 方向动态规划** 上扩展至 **8 方向**，对应 Hirschmüller (2008) 公式 (13) 和 (14)。

**学习目标**：不修改 `aggregate_one_step()`，学会以任意 `(dx,dy)` 找到所有路径起点，并保证一个方向上的每个像素都只遍历一次，最后累加 `S(p,d) = Σ_r L_r(p,d)`。

## 1. 本工程的结构

```text
stereo_sgbm_learning/
├── stage01_bm/
├── stage02_cost_volume/
├── stage03_single_path/
├── stage04_four_directions/
└── stage05_eight_directions/
    ├── CMakeLists.txt
    ├── README.md
    ├── include/sgm_eight_directions.hpp
    ├── src/main.cpp
    └── tests/test_sgm_eight_directions.cpp
```

前四个阶段保持原样。`include/sgm_eight_directions.hpp` 是本节最重要的文件。

## 2. 先读懂路径起点判定

令 `r=(dx,dy)` 为本方向的传播增量。某个像素 `p=(x,y)` 的前驱是 `(x-dx,y-dy)`。

只有当前驱位于图像外时，`p` 才是一条新路径的起点：

```cpp
if (in_image(sx - dx, sy - dy, W, H)) continue;
```

从起点开始：

```cpp
for (int x=sx, y=sy; in_image(x,y,W,H); x+=dx, y+=dy) {
    aggregate_one_step(...);    // 公式 (13)
    previous.swap(current);
}
```

外层 `sy/sx` 两层循环用于**寻找路径起点**，内层 `(x,y)` 循环才是**沿一条路径做 DP**。三个循环不是三维 Cost Volume 的三维遍历；`aggregate_one_step` 内部还有候选视差 `d` 的循环。

方向表：

| 方向 | (dx,dy) | 起点边界 |
|---|---|---|
| → | (+1,0) | 左边 |
| ← | (-1,0) | 右边 |
| ↓ | (0,+1) | 上边 |
| ↑ | (0,-1) | 下边 |
| ↘ | (+1,+1) | 上边、左边 |
| ↖ | (-1,-1) | 下边、右边 |
| ↙ | (-1,+1) | 上边、右边 |
| ↗ | (+1,-1) | 下边、左边 |

每个对角方向的独立路径数量为 `W + H - 1`，因为公共角点只计一次。

## 3. 三个需要弄懂的接口

```cpp
stereo::scan_direction8(C, params, direction, callback);
```

沿一个方向扫描全部路径，`callback(y,x,base,current)` 得到当前像素的 `D` 个路径代价。

```cpp
auto L = stereo::aggregate_one_direction8(C, params, direction);
```

仅用于调试：输出一个方向的完整 `L_r`，方便检查单条路径的数值。

```cpp
auto S = stereo::aggregate_eight_directions(C, params);
```

正式八方向代价聚合：输出 `S`。每算完一个像素的 `L_r` 就直接累加到 `S`，不保留八份完整路径代价体，因此工作内存为：输入 C + 输出 S + O(D) 状态。

> 提醒：`S` 是聚合后的 Cost Volume，不是视差图；还需要 `winner_takes_all(S)` 才得到整数视差。

## 4. 用 Stage04 的同一个小例子手算

H=2, W=3, D=3；候选视差：10、11、12；P1=5、P2=20。

```text
A  B  C
D  E  F
```

像素 B 的初始匹配代价 `C(B,:)=[0,3,4]`。

Stage04 的四方向累计为：

```text
S4(B,:) = [5,17,26]
```

新增四个对角方向的路径代价（都针对像素 B）：

```text
↘： [0,3,4]    // B 在最上行，是该路径的起点
↖： [2,3,7]    // 来自右下角 F
↙： [0,3,4]    // B 在最上行，是该路径的起点
↗： [4,3,9]    // 来自左下角 D
```

所以：

```text
新增对角方向之和 = [6,12,24]
S8(B,:) = [5,17,26] + [6,12,24] = [11,29,50]
```

八方向 WTA 选择最小代价对应的实际视差 `10`。

请亲自计算 `↖` 从 F 到 B 的状态转移，验证 `[2,3,7]`。

## 5. Ubuntu 构建与测试

```bash
cd stereo_sgbm_learning/stage05_eight_directions
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

核心 C++ 测试**不需要 OpenCV**。当前测试包含：

- 八个方向是否分别访问每个像素一次；对角路径起点数是否为 `W+H-1`。
- 三维 3×2×3 数值例子、对角路径代价和 8 路聚合值。
- 前四个方向是否与 Stage04 完全一致。
- 八份 `L_r` 显式求和是否等于在线累加的 `S`。
- 图像只有一行或一列时的边界行为；无效代价保持无效。
- `P1=P2=0` 时，`S=8C`（有效候选处）。
- 平移 7 像素合成双目图像的 4/8 路视差回归测试。

## 6. BlenderProc 仿真图像

图像读写演示需要 Ubuntu 安装 OpenCV 开发包：

```bash
sudo apt install -y libopencv-dev
```

重新执行构建配置，再运行：

```bash
./build/handmade_sgm_8path \
  ~/project/stereo_oven_sim/output/left.png \
  ~/project/stereo_oven_sim/output/right.png \
  ./out \
  48 80 5 200 800 \
  ~/project/stereo_oven_sim/output/disparity_gt_left_16.png \
  ~/project/stereo_oven_sim/output/valid_stereo_mask.png
```

参数：`minDisp numDisp blockSize P1 P2`。上面惩罚值只是演示起点，不保证最优。

输出包括：

```text
out/bm_disp16.png          out/bm_vis.png
out/sgm_1path_disp16.png   out/sgm_1path_vis.png
out/sgm_4path_disp16.png   out/sgm_4path_vis.png
out/sgm_8path_disp16.png   out/sgm_8path_vis.png
out/4path_vs_8path_difference.png
out/center_cost_curve.csv
```

其中 `*_disp16.png` 采用整数视差 ×256 编码；是教学实验中的整数视差，并没有进行亚像素优化。

CLI 演示的 PNG 输出目前支持非负且小于 256 的视差；超出范围会报错。GT 的编码应与 `×256` 一致。

本环境已编译运行**核心算法单元测试**；由于当前环境没有 OpenCV 开发包，OpenCV 图片读写演示程序尚未在本环境端到端编译。

## 7. 不要误解八方向的意义

8 方向比 4 方向多考虑了一组斜向空间约束，常常有助于减少方向性条纹，但**不保证每张图上都比 4 方向精确**。实际误差仍由纹理、遮挡、匹配代价、P1/P2、视差范围等共同决定。

原论文中的多方向 SGM 与 OpenCV 默认 MODE_SGBM 的五方向策略并不完全相同；Stage05 也仍然使用 SAD 匹配代价。目标是先掌握完整多方向优化，再逐步复现 OpenCV 的 BT 代价及实现细节。

**下一节 Stage06：亚像素视差估计。** 将在八方向聚合代价曲线上拟合抛物线，得到浮点视差。然后进入左右一致性检查与具体 OpenCV 算法复现。
