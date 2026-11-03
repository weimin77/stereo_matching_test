# sgbm_repro 实现记录

> 从零写一份与 `cv::StereoSGBM` 逐字节一致的 SGBM 实现，是怎么一步步做出来的。

| 项 | 值 |
|---|---|
| 目标 | `MODE_SGBM` 的 `CV_16S` 输出与 `cv::StereoSGBM` 逐字节相同 |
| 源码 | [`src/`](src/)、[`include/sgbm_repro/sgbm.h`](include/sgbm_repro/sgbm.h) |
| 判据 | [`tests/oracle/`](tests/oracle/)，OpenCV `stereosgbm.cpp` 的转写 |
| 验证 | [`tests/run_all.sh`](tests/run_all.sh) |
| 规模 | 核心 1072 行 + 测试 1898 行 |
| 修订 | 2026-10-30 |

## 阅读路径

| 目的 | 章节 |
|---|---|
| 知道为什么要先定义"一致" | [1](#1-目标与验收定义)、[2](#2-第一步确定参考语义) |
| 理解判据的作用 | [3](#3-第二步先造判据) |
| 查几何推导与缓冲布局 | [4](#4-第三步几何与缓冲布局) |
| 看逐步实现与关键代码 | [5](#5-第四步分五个阶段实现) |
| 查必须照抄的语义约束 | [6](#6-第五步必须精确复刻的语义约束) |
| 看验证结果 | [7](#7-第六步验证) |
| 看性能边界 | [8](#8-性能与边界) |

---

## 1. 目标与验收定义

"与 OpenCV 的结果一致"有强弱两种，必须先选定，否则后面的每一步都无法判断对错：

| 等级 | 定义 | 可验证性 |
|---|---|---|
| **A 位精确** | `CV_16S` 输出逐字节相同 | 机器可判定，无歧义 |
| B 数值等价 | `\|Δd\| ≤ 1/16 px`，差异像素 < 0.01% | 需要人为设定阈值 |
| C 视觉等价 | 覆盖率 / EPE 指标持平 | 不构成对实现的约束 |

本项目采用 **A**。理由是 A 可以在每一步给出二值结论，而 B 的阈值会掩盖小范围系统性错误——一个只影响底边几行的 bug 在 B 下可以存活很久。

配套的范围裁剪（与验收标准同等重要）：

| 维度 | 决策 | 依据 |
|---|---|---|
| `mode` | 只做 `MODE_SGBM` | 行递归、内存最低，是嵌入式目标唯一现实的路径 |
| 输入 | 只支持 `CV_8UC1` | 上游已经转过灰度，多通道会引入额外的 `diff_scale` 分支 |
| 后处理 | 只复刻 `medianBlur(3)` + `filterSpeckles` | 这两步在 `StereoSGBMImpl::compute()` 内部无条件执行，属于算法的一部分 |

---

## 2. 第一步：确定参考语义

在写任何算法代码之前，先要确认"参考"是唯一的。这一步做了两个实验。

### 2.1 运行时优化开关是否影响结果

`cv::setUseOptimized(false)` 是常见的怀疑对象。实测用同一份输入分别在开关两种状态下跑 `cv::StereoSGBM`：

```
Q1 setUseOptimized(true) vs (false) : 0 cases differ, 0/4210455 px
```

51 个用例、421 万像素，零差异。原因是 `stereosgbm.cpp` 不是分派编译单元（没有 `CV_CPU_DISPATCH`），`CV_SIMD` 在编译期就固定了。**这条结论让参考变成唯一值，后续不需要固定任何运行时状态。**

### 2.2 OpenCV 的内部一致性

接着检查 OpenCV 自己的两条路径是否给出相同结果：`stereosgbm.cpp` 里同一段逻辑有 SIMD 分支和 `#else` 标量分支。对照 `intrin_sse.hpp`：

| 运算 | SIMD 分支 | 标量分支 |
|---|---|---|
| `v_add` (`v_int16`) | `_mm_adds_epi16` — **饱和** | `(CostType)(a + b)` — **环绕** |
| `v_sub` (`v_int16`) | `_mm_subs_epi16` — **饱和** | 环绕 |
| `v_mul` (`v_int16`) | 加宽乘 + 饱和打包 | 环绕 |

两条路径只在中间值超出 `[-32768, 32767]` 时分歧。这个条件并不罕见：`blockSize = 11`、`preFilterCap = 31` 时，水平窗口和乘以 `(SH2+1)` 就越过 32767。

**结论：参考必须是"某一套具体语义"，而不是笼统的"OpenCV"。** 选择正常 x86 构建实际执行的那一套，即**饱和语义**。

这个决定贯穿整个实现：所有 16 位算术都用显式的饱和原语（[`src/internal.h`](src/internal.h)）：

```cpp
inline short saturate16(int v) {
    return (short)std::min(kMaxCost, std::max(-kMaxCost - 1, v));
}
inline short add16(short a, short b) { return saturate16((int)a + b); }
inline short sub16(short a, short b) { return saturate16((int)a - b); }
inline short mul16(short a, short b) { return saturate16((int)a * b); }
// vx_setall_s16((short)value): truncation, not saturation.
inline short narrow16(int v) { return (short)(v & 0xffff); }
```

`narrow16` 与 `add16` 的区别是刻意的：`narrow16` 对应 `vx_setall_s16((short)x)` 的**截断**语义，用于把 `P2 + minLr` 广播成常量时；`add16` 是**饱和**，用于数据通路。

---

## 3. 第二步：先造判据

### 3.1 为什么不能直接调 `cv::StereoSGBM`

如果只有一个最终视差图可比，出问题时只能知道"某个像素差了"，无法判断是代价算错、聚合算错、还是 DP 算错。而 `cv::StereoSGBM` 完全不暴露中间缓冲。

`calcPixelCostBT` 和 `computeDisparitySGBM` 在 OpenCV 里都是 `static` 函数，也无法从外部链接。

### 3.2 判据的构成

[`tests/oracle/`](tests/oracle/) 逐字转写 `stereosgbm.cpp` 的以下三部分，去掉 SIMD 分支、把饱和语义显式化：

| OpenCV | 转写 |
|---|---|
| `calcPixelCostBT` | `oracle::calcPixelCostBT` |
| `struct BufferSGBM` | `oracle::Buffer` |
| `computeDisparitySGBM` | `oracle::computeDisparitySGBM` |

并注入逐阶段落盘钩子。判据本身必须先被证明正确——它给出 6 个可比的阶段：

```
pixel cost   -> 预滤波 + BT 代价
cost         -> 水平 + 垂直盒聚合
aggregateFwd -> 前向 4 个方向
aggregate    -> 加上第 5 方向
pre-epilogue -> 选优 + 亚像素 + 一致性检查
final        -> 中值 + 斑点
```

有了这 6 个断点，一次失败就能直接落到具体模块。

### 3.3 判据自身的验收

判据不是"相信我"，而是被独立验证过：**51 个用例、4,210,455 像素与 `cv::StereoSGBM` 零差异**。

```
Q2 oracle vs cv::StereoSGBM         : 0 cases differ, 0 px total
```

判据与库一致之后，它就成了可信的中间层标尺。这一条是本项目最重要的一次性投入——没有它，后面 7 个语义细节里的任何一个都会变成无解的调试。

---

## 4. 第三步：几何与缓冲布局

### 4.1 几何量的推导

`resolveGeometry()` 把用户参数解析成所有循环边界（[`src/sgbm.cpp`](src/sgbm.cpp)）：

```cpp
g.minDisp = params.minDisparity;
g.numDisp = params.numDisparities;
g.maxDisp = g.minDisp + g.numDisp;
g.minX1   = std::max(g.maxDisp, 0);
g.maxX1   = width + std::min(g.minDisp, 0);
g.width1  = g.maxX1 - g.minX1;
g.SW2 = g.SH2 = params.blockSize / 2;
g.invalidDisp       = g.minDisp - 1;
g.invalidDispScaled = g.invalidDisp * kDispScale;
```

含义：

- **`minX1`** —— 能拥有完整视差范围的起始列。左图小于 `maxDisp` 的列找不到对应的右图列，因为匹配需要向右看 `maxDisp` 个像素。
- **`maxX1`** —— 同理，末端列在 `minDisp` 为负时会收缩。
- **`width1 = maxX1 - minX1`** —— 有效列数，也是所有行缓冲的列数。
- **无效标记** —— `(minDisp - 1) * 16`，比任何合法视差（`minDisp * 16`）小 16，因此不会与真值撞车。

### 4.2 视差标签的对齐填充

OpenCV 用 `Da = alignSize(numDisparities, vlanes)` 作为代价行的步长；本实现用 `Da = numDisparities`。

这两种布局算出的值相同，前提是填充区永不参与运算——而 `d` 循环的边界始终是 `numDisparities`。用 `numDisparities` 取 20 / 36 / 44（均非 8 的倍数）的用例验证过：

```bash
add("pad d0+20 bs5", 96, 64, 0, 20, 5, ...);
add("pad d0+36 bs3", 128, 72, 0, 36, 3, ...);
add("pad d8+44 bs7", 160, 80, 8, 44, 7, ...);
```

三个用例在所有阶段均零差异，因此实现不继承这个平台相关常数。

### 4.3 方向槽位与哨兵

DP 递推需要读 `in[d-1]` 和 `in[d+1]`，边界处必须看到"极差的代价"，所以 `d = -1` 和 `d = numDisp` 需要可写的哨兵位。

本实现的布局（[`src/workspace.cpp`](src/workspace.cpp)）给每个 `(列, 方向)` 一个独立槽位，槽位两侧各留 1 个 `short`：

```cpp
short* Workspace::directionRow(int id, int x, int shift) {
    const std::size_t stride = (std::size_t)(g.numDisp + 2);
    return directional[id].data() + (((std::size_t)(x + 1) * 4 + shift) * stride) + 1;
}
```

`+ 1` 就是左侧哨兵。`x + 1` 让 `x = -1`（第 0 方向的起点）也有合法槽位。

OpenCV 用更紧凑的打包：4 个方向共用一个 `Dlra = Da + vlanes` 步长，让某个方向的 `d = -1` 落进上一个方向的填充区。两种布局始终只在 `[0, numDisp)` 内读写，所以结果相同；独立槽位换来的是可读性和调试时能单独 dump 某个方向。

### 4.4 工作区大小

实测（`min=32 num=96 bs=5`）：

| 分辨率 | 代价 / 聚合 | 水平环 | 方向缓冲 ×2 | 合计 | 输出 |
|---|---|---|---|---|---|
| 640×480 | 288 KB | 672 KB | 787 KB | **1.72 MB** | 0.59 MB |
| 1280×720 | 648 KB | 1512 KB | 1767 KB | **3.86 MB** | 1.76 MB |

全部在构造时一次性分配、逐行复用，运行期无动态分配。这是为嵌入式目标保留的性质。

### 4.5 流水线数据流

```
        left, right (CV_8U)
               |
      [1] 预滤波 + BT 代价        per-pixel cost, width1 x numDisp
               |
      [2] 水平盒 (半径 SW2)        环状行缓冲
      [2] 垂直盒 (半径 SH2)        滑动增量
               |
          cost C
               |
      [3] 前向 4 方向              累加进 aggregate S
               |
      [4] 反向第 5 方向            复用方向 0 的槽位，选优
               |
      亚像素 + 伪右视图一致性
               |
          disparity (CV_16S)
               |
      [5] medianBlur(3) + filterSpeckles
```

---

## 5. 第四步：分五个阶段实现

顺序不是随手排的：每一阶段只依赖上一阶段已锁定的不变量，这样任何新出现的差异必然落在刚写的代码里。

### 5.1 预滤波与 BT 代价

**预滤波**产生两层：

| 层 | 内容 | 权重 |
|---|---|---|
| 0 | 截断的 3×3 导数响应 | ×1 |
| 1 | 原始强度 | `>> 2` |

层 0 对增益不敏感（是导数），层 1 保留绝对信息，两者累加进同一个代价体。

导数层的查表把整个 3×3 邻域折进一个索引，省掉乘加：

```cpp
dst[x] = clip[(row[x + 1] - row[x - 1]) * 2 + row[x + up + 1] - row[x + up - 1] +
              row[x + down + 1] - row[x + down - 1]];
```

`up` / `down` 在图像首末行塌缩为 0，等价于复制边缘行。

查表本身的定义（[`src/internal.h`](src/internal.h)）是一个 clamp：

```cpp
const int ftZero = std::max(preFilterCap, 15) | 1;
table_[i] = (uchar)(std::min(std::max(i - kOffset, -ftZero), ftZero) + ftZero);
```

注意 `| 1` 而非 `+ 1`，且下限是 15 —— 所以 `preFilterCap = 0` 时实际取值仍是 15。

**BT 代价**在 `d` 方向遍历。右图被水平镜像，使得匹配循环可以用递增索引读它：

```cpp
const int col = width - 1 - x + g.minDisp + d;
int leftAbove  = std::max(0, u - rightMax[col]);
leftAbove      = std::max(leftAbove, rightMin[col] - u);
int rightAbove = std::max(0, v - uMax);
rightAbove     = std::max(rightAbove, uMin - v);
dst[d] = add16(dst[d], (short)(std::min(leftAbove, rightAbove) >> shift));
```

`rightMin` / `rightMax` 是右图 3 抽头的运行 min/max，每层每行算一次；`uMin` / `uMax` 是左图该像素的 3 抽头 min/max，每个 `x` 算一次。BT 代价取"左值高出右值区间"与"右值高出左值区间"的较小者。

3 抽头的中点用整数除法 `(v + rRef[x-1]) / 2`，两端点在边界处退化为中心值 —— 不能改用对称 padding。

### 5.2 盒聚合

水平窗口（半径 `SW2`）先做，结果进环状行缓冲；垂直窗口（半径 `SH2`）以滑动增量的方式推进，只加减两行，从不重算整块。

水平方向的 `x = 0` 特殊：窗口覆盖 `0..SW2`，且第 0 列被计 `SW2 + 1` 次（边界复制）：

```cpp
for (int d = 0; d < D; ++d) {
    short acc = mul16(pixelCost[d], hScale);          // hScale = SW2 + 1
    for (int j = 1; j <= SW2; ++j) acc = add16(acc, pixelCost[(std::size_t)j * D + d]);
    out[d] = acc;
}
```

`x > 0` 时滑动，两端都用夹紧索引：

```cpp
const short* add = pixelCost + (std::size_t)std::min(x + SW2, width1 - 1) * D;
const short* sub = pixelCost + (std::size_t)std::max(x - SW2 - 1, 0) * D;
for (int d = 0; d < D; ++d) dst[d] = add16(sub16(prev[d], sub[d]), add[d]);
```

垂直方向分两支。首行（`y == 0`）用权重复制顶部边缘；其余行走增量：

```cpp
// 行首：先加后减
for (int d = 0; d < D; ++d)
    cost[d] = sub16(add16(cost[d], hsum[d]), hsumSub[d]);

// 滑动：先减后加
const short hv = add16(sub16(hsum[(std::size_t)(x - 1) * D + d], sub[d]), add[d]);
hsum[(std::size_t)x * D + d] = hv;
cost[(std::size_t)x * D + d] = add16(sub16(cost[(std::size_t)x * D + d],
                                          hsumSub[(std::size_t)x * D + d]), hv);
```

两条注释里的"先加后减"和"先减后加"不是笔误：饱和算术下 `sat(sat(a+b)-c)` 与 `sat(sat(a-c)+b)` **不相等**。这个次序必须照抄。

还有第三支：`k >= height` 时（图像底边之外的窗口），用夹紧到最后一行的 `hsum` 重建代价，不重算任何像素代价。

代价缓冲的初值是 `P2` 而不是 0：

```cpp
std::fill(ws.cost.begin(), ws.cost.end(), narrow16(g.P2));
```

递推式在首行加 `P2`、每个方向减 `P2` 来抵消，这是省一次加法的写法。饱和算术让这个起点变成可观测的。

### 5.3 前向 4 个方向

`MODE_SGBM` 只有一趟图像扫描，前向贡献 4 个方向。递推式（[`src/dp_sgbm.cpp`](src/dp_sgbm.cpp)）：

```cpp
auto step = [&](const short* in, short delta) -> short {
    const short best = std::min(std::min(in[d], add16(in[d - 1], P1)),
                                std::min(add16(in[d + 1], P1), delta));
    return add16(sub16(best, delta), Cd);
};
```

`delta = P2 + minLr`，把"全方向最小 + P2"折进一个常量，内层就只剩一条 min 链：

```
L(p,d) = sat( min( L(p-r,d), L(p-r,d±1)+P1, delta ) - delta + C(p,d) )
```

四个方向共享同一份 `Cd` 与 `S[d]` 加载，逐个累加：

```cpp
const short L0 = step(in0, delta0); out0[d] = L0; sum = add16(sum, L0);
const short L1 = step(in1, delta1); out1[d] = L1; sum = add16(sum, L1);
const short L2 = step(in2, delta2); out2[d] = L2; sum = add16(sum, L2);
const short L3 = step(in3, delta3); out3[d] = L3; sum = add16(sum, L3);
S[d] = sum;
```

哨兵在每次写入前设置，保证 `d = -1` 和 `d = D` 永不被选中：

```cpp
in0[-1] = in0[D] = (short)kMaxCost;
```

### 5.4 反向第 5 方向与选优

第 5 个方向由 `x` 从右向左扫描得到，**写入方向 0 的同一个槽位**。因此读 `x + 1` 的槽位就是读上一轮的结果，`x = width1 - 1` 时读到零初始化的填充 —— 恰好与前向在 `x = 0` 处的初态一致。

同一趟里完成选优：

```cpp
const short total = add16(S[d], L);
S[d] = total;
if (total < minS) { minS = total; best = d; }
```

严格小于 → 并列时取最小的 `d`。

唯一性检验从 `d = 0` 向上扫，命中即拒：

```cpp
int d = 0;
for (; d < D; ++d) {
    if ((int)S[d] * (100 - g.uniquenessRatio) < (int)minS * 100 && std::abs(best - d) > 1)
        break;
}
if (d < D) continue;
```

亚像素拟合用**整数**运算：

```cpp
const int denom = std::max((int)S[d - 1] + (int)S[d + 1] - 2 * (int)S[d], 1);
return d * kDispScale + (((int)S[d - 1] - (int)S[d + 1]) * kDispScale + denom) / (denom * 2);
```

浮点版本会在 1/16 定点上偏 1 个单位，足以破坏一致性。

选优的同时填充伪右视图映射（`pseudoRightDisp` / `pseudoRightCost`）：左图第 `x` 列认领右图第 `x - d` 列，代价最低者胜。最后用这张图做左右一致性：

```cpp
const int floorD = v >> kDispShift;
const int ceilD  = (v + kDispScale - 1) >> kDispShift;
// 两个候选列都判定不一致才拒绝
if (rejectA && rejectB) disp1Row[x] = (short)g.invalidDispScaled;
```

截断和舍入两个候选都试，是为了让"在两种读法下都矛盾"才算无效 —— 这给了有效视差一次被接受的机会。

### 5.5 中值与斑点

两步都在 `CV_16S` 定点上做，顺序不可交换。

3×3 中值用 `nth_element`，边缘复制：

```cpp
short v[9] = {r0[x0], r0[x], r0[x2], r1[x0], r1[x], r1[x2], r2[x0], r2[x], r2[x2]};
std::nth_element(v, v + 4, v + 9);
dst[x] = v[4];
```

斑点滤除是 4 连通洪泛，区域像素数不超过 `maxSpeckleSize` 且邻域差不超过 `maxDiff` 时整块替换为无效值：

```cpp
if ((int)members.size() <= maxSpeckleSize) {
    for (int index : members) {
        const int my = index / width;
        disparity.ptr<short>(my)[index - my * width] = (short)newValue;
    }
}
```

已等于 `newValue` 的像素不参与区域生长 —— 这既是"无效值"的语义，也让结果与区域访问顺序无关。

---

## 6. 第五步：必须精确复刻的语义约束

以下七条都是"看起来可以化简、实际不能"的地方。每条都给出了违反后的表现，便于将来改动时自查。

### 6.1 原始强度层的两端列取查表零点

**约束**：层 1 的 `x = 0` 与 `x = width - 1` 填 `tab[0]`，不是边缘像素值。

**原因**：两层都初始化为 `tab[0]`，只覆写 `[1, width-1)`。

**违反后**：最右列及其附近出现常数偏差，表现为若干 `d` 值上固定差 12（`ftZero` 相关量）。

```cpp
leftLayer1[0] = leftLayer1[width - 1] = ws.clipTable.centered()[0];
```

### 6.2 代价缓冲初值为 `P2`

**约束**：`std::fill(cost, narrow16(P2))`。

**违反后**：整行 `cost` 系统性偏移，首行尤其明显（分支权重最大处）。

### 6.3 饱和运算下加减次序不可交换

**约束**：

- 行首 `sat(sat(C + hsum) - hsumSub)`
- 滑动 `sat(sat(hsum - pixSub) + pixAdd)`
- 底边 `sat(sat(Cprev - hsumSub) + hsum)`

**违反后**：整幅图每个像素都差，`max|d|` 上千，属最容易发现的一类。

### 6.4 底边窗口需要重建而非冻结

**约束**：`k >= height` 时用夹紧到最后一行的 `hsum` 重建 `cost`。

**违反后**：差异只在最后 `SH2 + 1` 行。用逐行差异剖面可以立刻看出来：

```
y= 129      8 / 206
y= 130     24 / 206
```

### 6.5 `delta` 用截断而非饱和

**约束**：`narrow16(P2 + minLr)`。

**违反后**：`minLr` 接近 `SHRT_MAX` 时 `P2 + minLr` 溢出，若改成饱和则该方向的最优路径被系统性抬高。

### 6.6 亚像素用整数除法

**约束**：`(S[d-1] - S[d+1]) * 16 + denom) / (denom * 2)`，向零截断。

**违反后**：差异表现为零散像素差 1（1/16 px），数量少但确实存在。

### 6.7 唯一性判定用 32 位中间值

**约束**：`(int)S[d] * (100 - ur) < (int)minS * 100`。

**违反后**：`S` 可达 32767，乘 100 溢出 `int16`，判定结果与输入分布相关，极难定位。

---

## 7. 第六步：验证

### 7.1 四层测试

| 层 | 内容 | 目的 |
|---|---|---|
| 算子与参考自检 | `probe_simd` | 确认参考唯一、判据正确 |
| 合成矩阵 | `compare` | 逐阶段比对，覆盖参数角落 |
| 真实抓图 | `pairs` | 烤箱仿真的两组工况 |
| 公开数据集 | `pairs` | Middlebury 2006 十个场景 |

合成数据由"基础图 + 按视差平移"生成，保证左右图真正相关；图案包括常量、斜坡、条纹、噪声、棋盘，另加一个低对比度噪声用例专门压并列值的取舍路径。

### 7.2 参数覆盖

| 维度 | 取值 |
|---|---|
| 尺寸 | 48×32 至 640×480，含 97×71 / 129×97 / 206×131 等奇数 |
| `minDisparity` | 0、8、32、**-16、-32** |
| `numDisparities` | 16、20、32、36、44、48、64、96 |
| `blockSize` | 1、3、4、5、7、9、11 |
| `preFilterCap` | 0、31、63、200 |
| `uniquenessRatio` | -1、0、5、40 |
| `disp12MaxDiff` | 0、1、64 |
| `speckleWindowSize` | 0、40、5000 |
| `speckleRange` | 1、8 |
| 高度 | 等于 `blockSize`、`blockSize + 1`、`blockSize + 5` |

### 7.3 结果

```
pixel cost             PASS  (0 failures / 437 comparisons, 2979056 values)
box aggregation        PASS  (0 failures / 437 comparisons, 2979056 values)
4 directions           PASS  (0 failures / 437 comparisons, 2979056 values)
5th direction          PASS  (0 failures / 437 comparisons, 2979056 values)
pre-epilogue           PASS  (0 failures / 88 comparisons, 4807064 values)
final                  PASS  (0 failures / 176 comparisons, 9614128 values)

ALL STAGES BIT-EXACT
```

| 套件 | 规模 | 结论 |
|---|---|---|
| `probe_simd` | 51 用例 / 4,210,455 px | 0 差异 |
| `compare` | 88 用例 / 9,614,128 输出值 | 0 差异 |
| 烤箱抓图，`min=32 num=96`，bs 3/5 | 4 次 / 1.2 M px | 0 差异 |
| Middlebury 2006，`min=0 num=64 bs=5` | 10 场景 / 1.6 M px | 0 差异 |

### 7.4 验证工具

```bash
./build/probe_simd      # 参考唯一性 + 判据保真度
./build/compare         # 逐阶段，88 个合成用例
./build/compare -one    # 只跑首个用例并逐点打印差异坐标
./build/pairs <l> <r> <minDisp> <numDisp> <block> <P1> <P2> \
              <preFilterCap> <uniquenessRatio> <speckleWindowSize> <speckleRange> <disp12MaxDiff>
./build/bench           # 时间与峰值内存
tests/run_all.sh        # 全部
```

`compare -one` 的逐点输出形式：

```
case 64x48 d0+16 bs3 p0  width1=48 D=16
  y=0 x=47 d=1  repro      0  oracle     12
```

`(x, d)` 坐标直接指向语义问题，而不只是"某处不同"。

### 7.5 调试用的消融开关

[`tests/debug_case.cpp`](tests/debug_case.cpp) 保留在仓库里，可以单独关掉收尾、只比 DP、打印逐行差异剖面。差分定位的常规顺序是：

1. 先看差异是全局、成片还是集中在少数行
2. 全局 → 代价层；成片 → 聚合或 DP；少数行 → 边界窗口
3. 定位到阶段后用 `compare -one` 拿到 `(x, d)` 坐标

---

## 8. 性能与边界

### 8.1 各阶段实测占比

640×480，`min=32 num=96 bs=5`，总计 360 ms：

| 阶段 | 耗时 | 占比 |
|---|---|---|
| 前向 4 方向 | 181.2 ms | 50.3% |
| 反向第 5 方向 + 选优 | 72.1 ms | 20.0% |
| 垂直盒累加 | ~29.7 ms | 8.2% |
| BT 代价 | 45.5 ms | 12.6% |
| 水平盒 | 18.7 ms | 5.2% |
| 中值滤波 | 6.8 ms | 1.9% |
| 斑点滤除 | 2.0 ms | 0.6% |

DP 占七成，是唯一值得优先优化的目标。

### 8.2 与 OpenCV 的差距来自哪里

| | 耗时 | 相对 |
|---|---|---|
| `cv::StereoSGBM`（SSE2，8 lane） | 46.6 ms | 1.0× |
| 判据（同一份标量代码） | 337 ms | 7.2× |
| `sgbm_repro` | 343 ms | 7.4× |

两者**算法步数完全相同**，差距全部来自标量与 8 通道的区别。本实现与判据持平，说明缓冲布局没有引入额外开销。

### 8.3 位精确允许与不允许的优化

| 允许 | 不允许 |
|---|---|
| 沿 `d` 向量化（`d` 相互独立） | 改方向数或扫描顺序 |
| 删去未被读取的中间结果 | 换非饱和算术 |
| 换更快的选择算法（中值） | 按行或按列多线程（分别破坏递推与边界累加） |
| 减少冗余内存访问 | 调整 `S[d]` 四项累加次序 |

沿 `d` 向量化之所以安全，是因为递推式里所有 `d` 互不依赖 —— 这正是 OpenCV 能向量化的原因。任何向量化改动都会同样被 6 阶段比对捕获。

不可并行的具体原因：

- **按行切分** → 方向 0 沿 `x` 递推、其余 3 个方向依赖上一行，行间串行
- **按列切分** → 分块边界的累加被截断，接缝处必不相同

### 8.4 已知不足

| 项 | 现状 |
|---|---|
| 速度 | 慢 7.4 倍；未做 SIMD |
| 中值滤波 | `nth_element` 6.8 ms，`cv::medianBlur` 0.09 ms（75 倍） |
| 水平盒 | 对 `x >= 1` 的输出无消费者，属冗余计算 |
| `MODE_HH` | 未实现 |

---

## 9. 遗留事项

### 9.1 补 `MODE_HH`

代价层、聚合层、收尾层都已验证，剩余工作是：

1. `costHeight = height`：代价体按整图高度分配，而非单行
2. 第二趟图像扫描：`y` 与 `x` 都反向，再贡献 4 个方向
3. `Buffer` 的 `getCBuf(row)` / `getSBuf(row)` 从恒返回同一行改为按行偏移

内存会从 1.72 MB 涨到约 96 MB（640×480, D=96），因为全尺寸代价体是 `2 × width1 × height × D × 2 B`。

### 9.2 多通道输入

`fillPixelCostRow` 已经保留了 `cn > 1` 的平面布局与边界处理，主要缺 `diff_scale` 分支的完整验证。需要注意的是 `P1` / `P2` 不随通道数缩放，这一点在彩色代价上已有实测记录：覆盖率提升但 EPE 恶化。

### 9.3 剥离 `cv::Mat`

核心库对 OpenCV 的依赖只剩 6 个符号：

```
cv::Mat::clone()   cv::Mat::create()   cv::Mat::empty()
cv::Mat::operator=(cv::Mat&&)          cv::Mat::~Mat()
cv::error()
```

没有 `imgproc`、`stereo`、`calib` —— 中值与斑点都是自实现的。把 `cv::Mat` 换成

```cpp
struct Image { uchar* data; int width, height, stride; };
```

即可让核心不依赖 OpenCV 编译。这一步与 SIMD 无关，且可在 PC 上完整验证。

### 9.4 SIMD 抽象层

目标平台（WQ5008 / RK3562）的向量指令集与 x86 不同，因此需要一层只暴露算子的接口，使后端代码只写一份。

从现有代码推出来的算子清单：

| 算子 | 出处 |
|---|---|
| `add_sat` / `sub_sat` | DP、盒累加 |
| `vmin` / `vmax` | DP 的 min 链、BT 的 max |
| `load` / `loadu` / `store` | 全部内层循环 |
| `set1` | `P1`、`delta`、`scale` 广播 |
| `reduce_min` | `minL` |
| `widen(u8) → i16` | BT 累加 |

`kLanes = 1` 的标量后端就是当前代码，性能应持平；`kLanes = 8/16` 的后端才是真向量。

哨兵位在向量化后仍然够用：循环写成 `for (; d + L <= D; d += L)` 时，`in[d+1]` 的向量读最大触到索引 `D`，正好是 `d == D` 的哨兵，无需加宽填充。

一个必须保留的语义：`minS` / `bestDisp` 的更新必须是"严格小于、从 `d = 0` 递增"，向量化时用 `cmp` + `movemask` + `ctz` 实现，不能重排。
