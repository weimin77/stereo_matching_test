# Stage 02 — Cost Volume：从 BM 走向 SGM

本工程直接复用 stage01_bm/include/handmade_bm.hpp 中的 `GrayImage`、`BMParams`、`BMResult`。

## 学习目标

1. 区分 `di`（从 0 开始的视差索引）与真实 `d = minDisp + di`。
2. 用一维数组保存 `[H][W][D]`：`index=(y*W+x)*D+di`。
3. 使用 SAD 生成完整 `C(x,y,di)`；无效窗口用 `kInvalidCost`，不能填零。
4. 使用 WTA 验证 Stage02 视差与 Stage01 完全一致。
5. 理解每视差构造积分图，把 `O(W*H*D*B²)` 降为 `O(W*H*D)`。

本阶段还 **不是** SGM。Stage03 开始实现论文式 (13) 的路径递推；Stage04 进行多方向聚合。

## 编译

```bash
sudo apt update
sudo apt install -y build-essential cmake libopencv-dev
cd stereo_sgbm_learning/stage02_cost_volume
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

如果系统未安装 OpenCV，CMake 仍可构建 `test_cost_volume`（不涉及文件读写）。

纯 C++ 直接编译单测：

```bash
g++ -std=c++17 -O2 -Iinclude -I../stage01_bm/include \
    tests/test_cost_volume.cpp -o /tmp/test_cost_volume
/tmp/test_cost_volume
```

## 在 BlenderProc 图像上运行

```bash
./build/handmade_cost_volume \
    ~/project/stereo_oven_sim/output/left.png \
    ~/project/stereo_oven_sim/output/right.png \
    ./out 48 80 5 \
    ~/project/stereo_oven_sim/output/disparity_gt_left_16.png \
    ~/project/stereo_oven_sim/output/valid_stereo_mask.png
```

输出 `cost_wta_disp16.png`（uint16 = 视差 px x256）、`cost_wta_valid.png`、`cost_wta_vis.png`、`cost_slice_d88.png` 等，并打印中心点所有候选视差的 SAD 代价、计算耗时和 GT 指标。`cost_slice_d88.png` 为候选视差为 88 px 的代价切片（48+80/2）。

## 对照 Stage01 结果

运行 Stage01：

```bash
cd ../stage01_bm
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
./build/handmade_bm \
    ~/project/stereo_oven_sim/output/left.png \
    ~/project/stereo_oven_sim/output/right.png \
    ./out 48 80 5
```

对比 `stage01_bm/out/bm_disparity_16.png` 与 `stage02_cost_volume/out/cost_wta_disp16.png`，逐像素应完全相同（都还只是 SAD+WTA）。

## 为什么积分图成立？

对特定视差 d，定义 `A_d(x,y) = |L(x,y)-R(x-d,y)|`。
积分图 `SAT(x,y)` 是左上角至 (x-1,y-1) 的所有 `A_d` 像素和。
对于 `[x0,x1) × [y0,y1)`，矩形和：

`sum = SAT(y1,x1)-SAT(y0,x1)-SAT(y1,x0)+SAT(y0,x0)`。

每个候选视差都需要单独生成 SAT；每次构建 O(W*H)，每窗口查询 O(1)。无效区域的差值图填 0 仅用于积分图构造，**绝不**把包含越界点的窗口当作有效候选。

## 单元测试内容

- 朴素 SAD 与积分图 SAD 的完整代价体逐元素一致（包括无效边界）。
- `Cost Volume → WTA` 与 Stage01 BM 视差图逐像素一致。
- 合成纹理已知真实视差 7px，内部像素全正确。
- 完全越界的候选视差均被标记无效。
- 平坦图像发生匹配歧义时 WTA 选择最靠前的有效候选，这**不**代表真实视差。

## 思考题

1. 为什么一个像素有多个候选视差时，SGM 不应提前做 WTA？
2. `minDisp=48, D=80` 时数组 `di=40` 对应实际视差多少？
3. 为什么无效 Cost 不可以用 0 表示？
4. 为什么积分图加速了窗口 SAD，但不减少 Cost Volume 的存储空间？
5. 无纹理图像为什么会产生大量代价相同的视差候选？
