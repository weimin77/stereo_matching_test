# Stage 03 — SGM 单方向动态规划 (Left → Right)

本阶段承接 Stage 02 的 `CostVolume`，实现 Hirschmüller (2008) Section 2.2 的路径递推公式 (12)/(13)。

**注意：只有左→右一条路径，并不是完整多方向 SGM，更不是 OpenCV StereoSGBM。** 单路径 WTA 的精度不保证优于 BM，甚至可能产生水平条纹；Stage 04 才会加入四方向并累加代价，Stage 05 再扩展为八方向。

## 公式与数据流

`C(y,x,di)`：Stage 02 的 SAD 代价，`di=0..D-1`，实际视差 `d=minDisp+di`。


SGM 路径递推（论文式 (13)）：

```
L_r(p,d) = C(p,d) + min(
    L_r(p-r,d),
    L_r(p-r,d-1) + P1,
    L_r(p-r,d+1) + P1,
    min_k L_r(p-r,k) + P2
) - min_k L_r(p-r,k)
```

左→右时，`r=(1,0)`，`p-r=(x-1,y)`，对每个 `y` 独立从 `x=0` 到 `W-1` 执行递推。起点 `L=C`。

**数组**：
- 原始 `C`: `[H][W][D]`
- 单方向 `L`: `[H][W][D]`
- 临时 `previous` / `current`: 两个 `[D]` 行向量
- `kInvalidCost`: 无效视差候选，既不能参与最小值选择，也不能当作正常的聚合值。

**为什么减去上一像素最小值？** 对所有当前候选 `di` 减去相同常数，不改变它们的优劣次序，又可以阻止代价沿路径长度无限增长。

## 手算检验（单行三个像素、三个视差）

```
P1=5, P2=20

           d=10  d=11  d=12
C(x=0):      10     0    12
C(x=1):       0     3     4
C(x=2):       1     5     3

L(x=0):      10     0    12  (路径起点 = C)
L(x=1):       5     3     9
L(x=2):       3     5     8
```

例如 `x=1, di=0`：
`prevMin=0, min(10,INF,0+5,0+20)=5`，因此 `L=0+5-0=5`。
`x=2, di=0`：`prevMin=3, min(5,INF,3+5,3+20)=5`，因此 `L=1+5-3=3`。

本例可直接对照 `tests/test_sgm_single_path.cpp`。

## 文件

- `include/sgm_single_path.hpp`: 公式 (13) 的 `aggregate_one_step` 与全图左→右 `aggregate_left_to_right`。
- `tests/test_sgm_single_path.cpp`: 手算数值、P2 跳变、边界/整行重置、模拟左右图、参数异常测试。
- `src/main.cpp`: OpenCV 只用来读取/保存 PNG，不调用任何 OpenCV 立体匹配算法；输出 SAD+WTA 和 SAD+1path+WTA 的对比结果。
- Stage 01 / 02 一起保留在 zip 中，头文件可直接引用。

## Ubuntu 编译

```bash
sudo apt update
sudo apt install -y build-essential cmake libopencv-dev
cd stereo_sgbm_learning/stage03_single_path
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

不安装 OpenCV 也可以编译纯 C++ 单元测试：

```bash
g++ -O2 -std=c++17 -Iinclude -I../stage02_cost_volume/include -I../stage01_bm/include \
    tests/test_sgm_single_path.cpp -o /tmp/test_sgm_single_path
/tmp/test_sgm_single_path
```

## BlenderProc 图像测试

```bash
./build/handmade_sgm_1path \
  ~/project/stereo_oven_sim/output/left.png \
  ~/project/stereo_oven_sim/output/right.png \
  ./out 48 80 5 200 800 \
  ~/project/stereo_oven_sim/output/disparity_gt_left_16.png \
  ~/project/stereo_oven_sim/output/valid_stereo_mask.png
```

这里 P1=200,P2=800 来自常用的灰度 5x5 块匹配经验起点 `8*1*5²` 和 `32*1*5²`；**并不保证是该场景的最优参数**。

可生成：

- `bm_disp16.png`, `bm_vis.png`, `bm_valid.png`：Stage 02 `C → WTA`。
- `sgm_1path_disp16.png`, `sgm_1path_vis.png`, `sgm_1path_valid.png`：Stage 03 `C → L_→ → WTA`。
- `bm_vs_1path_difference.png`：相同有效区域上的视差差异，灰度越亮差异越大。
- `center_cost_curve.csv`：中心像素候选视差的 `C` 和 `L`，适合绘制双曲线对比。

视差 16 位 PNG 使用实际视差 `px × 256`，0 表示无效（要求真实视差非负且小于 256）。

## 特别注意的边界规则

1. 路径起点没有前一个像素：`L=C`。
2. 任一 `C(x,y,di)` 为 `kInvalidCost`：对应 `L` 必须仍然无效。
3. 前一个像素所有 `di` 都无效：当前非无效候选按新路径起点初始化。
4. 下一个 `y` 行必须把 `previous` 重置为无效，绝不能沿上一行传播。
5. 不应直接把 `kInvalidCost` 加上 P1/P2，否则容易溢出或污染有效路径状态。
6. 正常有效候选从视差 `di` 向 `di±1` 传播可受 P1 惩罚，跨多个视差受 P2 惩罚。

## 练习题

1. 将 `P1=P2=0`，验证公式 (13) 是否退化为 `L=C`？
2. 修改为 `P1=10,P2=80`，观察中心像素最优视差如何变化。
3. 为什么单路径的输出可能有条纹？
4. 为什么每个 `x` 都只需要两个长度为 `D` 的状态缓存？
5. 下一阶段如何从左→右推广到右→左和上→下？

建议阅读上传论文 PDF 第 3～4 页（期刊页码 330～331）Section 2.2，尤其公式 (11)～(14)、Fig. 2。
