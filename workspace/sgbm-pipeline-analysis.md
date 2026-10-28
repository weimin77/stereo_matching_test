# `test.cpp`：双目立体匹配流水线

> SGBM 出初始视差 → 亚像素左右一致性检查 → WLS 引导滤波补洞 → 自定义「信任域」后处理。

| 项 | 值 |
| --- | --- |
| 源码 | [`test.cpp`](../test.cpp)（1085 行，单文件流水线） |
| 配置 | [`sgbm_config.json`](../sgbm_config.json)，profile `balanced`，postprocess `wls` |
| 对应实现 | `BlenderProc-main/examples/advanced/stereo_oven_{empty,food}_tray/sgbm_pipeline.py` |
| 依赖 | OpenCV 5.0（**必须含 `ximgproc`**；`binarysgbm` 另需 `xstereo`）；4.6 亦可，需 `-DREQUIRE_OPENCV5=OFF` |
| 修订 | 2026-10-28 |

## 阅读路径

| 你的目的 | 去这里 |
| --- | --- |
| 搞懂它为什么这样设计、每步在防什么 | [第一部分 理解](#第一部分理解) |
| 查默认值、产物格式、配置键、函数位置 | [第二部分 查阅](#第二部分查阅) |
| 编译、运行、调参 | [第三部分 操作](#第三部分操作) |

## 目录

**第一部分 理解**

1. [它要解决什么问题](#1-它要解决什么问题)
2. [几何前提：视差、深度与搜索窗口](#2-几何前提视差深度与搜索窗口)
3. [阶段一：SGBM 半全局匹配](#3-阶段一sgbm-半全局匹配)
4. [阶段二：亚像素左右一致性检查](#4-阶段二亚像素左右一致性检查)
5. [阶段三：WLS 引导滤波](#5-阶段三wls-引导滤波)
6. [阶段四：`refineDisparity()` 信任域细化](#6-阶段四refinedisparity-信任域细化)
7. [数据流总图](#7-数据流总图)
8. [设计取舍与已知局限](#8-设计取舍与已知局限)

**第二部分 查阅**

9. [术语与符号](#9-术语与符号)
10. [配置键](#10-配置键)
11. [产物与编码约定](#11-产物与编码约定)
12. [命令行、退出码与日志](#12-命令行退出码与日志)
13. [与 `sgbm_pipeline.py` 的对照](#13-与-sgbm_pipelinepy-的对照)
14. [概念与函数的对应](#14-概念与函数的对应)

**第三部分 操作**

15. [构建](#15-构建)
16. [运行](#16-运行)
17. [无 GUI 环境](#17-无-gui-环境)
18. [调参入门：按症状找参数](#18-调参入门按症状找参数)
19. [`binarysgbm` 与 `sgbm` 的实测对比](#19-binarysgbm-与-sgbm-的实测对比)

---

# 第一部分：理解

## 1. 它要解决什么问题

这条流水线的每个设计决定，都能从场景约束反推出来：

| 场景约束 | 直接后果 | 代码里的回应 |
| --- | --- | --- |
| 工作距离 0.07–0.25 m | 视差跨度 33.6–120 px（3.6 倍） | 由深度范围反推 `min/num_disparities`，把代价体限制在物理可能区间 |
| 近处视差大（120 px） | 图像左边界没有搜索空间 | 左右各补 `padding = 133` px 再匹配 |
| 金属内腔近似无纹理 | 局部块匹配的 argmin 由噪声决定 | SGBM 的全局平滑项 + WLS 补洞 |
| 已校正平行双目 | 极线水平，只需一维搜索 | 不做极线校正，只校验 `fx`/`baseline` 为正 |
| 下游要区分「测到的」与「猜的」 | 需要像素级可信度 | 无效统一用 `NaN`；额外输出 `inferred` 掩膜 |

**设计目标不是「理论最优视差图」**，而是：覆盖率尽可能高、结果可复现、且每个像素的来源可解释。

### 1.1 与 Python 实现的关系

`test.cpp` 是 `sgbm_pipeline.py` 中 **WLS 后端**的逐像素移植，为对齐 Python 的数值行为保留了几处刻意的「怪癖」（见 [§5.3](#53-三个必须遵守的工程约定)）。Python 侧还有一个 `guided` 后备后端（用于没有 opencv-contrib 的环境），**C++ 未移植**。

### 1.2 `ADCensusBM/` 不参与这条流水线

`CMakeLists.txt` 把 `ADCensusBM/*.cpp` 编进同一个可执行文件，但 `test.cpp` **没有任何调用**（无 `StereoProcessor`，无 `adc` 命名空间）。它是并行存在的另一套算法（ADCensus + cross-based 聚合），当前不产生任何输出。详见 [`adcensus-tuning.md`](./adcensus-tuning.md)。

## 2. 几何前提：视差、深度与搜索窗口

### 2.1 视差 ↔ 深度

平行双目、极线已对齐（同一行）时：

$$d = \frac{f_x \cdot B}{Z}, \qquad Z = \frac{f_x \cdot B}{d}$$

代码只用到两个相机标量——`fx` 与 `baseline`（米），来自 JSON 的 `camera{fx, baseline_m}`，或 `camera_file` 指向的 `camera.json`（即 BlenderProc 示例写出的格式）。默认 `fx = 140`、`B = 0.06 m`，故 `fx·B = 8.4 m·px`。

### 2.2 搜索窗口由工作深度反推

不写死 `num_disparities`，而是从 `working_depth_m = [near, far]` 反算：

```
low  = fx·B / far  = 8.4 / 0.25 = 33.6 px     ← 最远处的最小可能视差
high = fx·B / near = 8.4 / 0.07 = 120.0 px    ← 最近处的最大可能视差

minDisparity   = floor(low / 16) · 16
numDisparities = ceil((high − minDisparity + 1) / 16) · 16
limits         = [max(low, min), min(high, min + num − 1/16)]
```

代入默认值：`min = 32`、`num = 96`、`limits = [33.6, 120)`。

**为什么要取整到 16 的倍数**：SGBM 的 `numDisparities` 必须是 16 的倍数（内部按 16 分块做 SIMD 与代价归一化），所以窗口只能向外扩到网格上。`limits` 则保留**未被取整的物理边界**，供后续「超出物理范围即无效」的判定使用——这样既不浪费搜索空间，也不把取整扩出来的那一段当成可信结果。

显式写 `min_disparity` / `num_disparities`（都 > 0）时直接采用，跳过上述推导。

### 2.3 为什么左右各补 133 px 黑边

```
padding = minDisparity + numDisparities + blockSize = 32 + 96 + 5 = 133
```

SGBM 沿 −x 方向搜索视差。对真实图像**左边界**附近的像素，搜索窗口会越出图像，OpenCV 直接把那一带标为无效。把左右各补 133 px 黑边后，真实内容在补边坐标系里拥有了完整的搜索范围；算完用 `Rect(padding, 0, width, height)` 裁回。

这是覆盖率的关键：不加 padding，右侧会出现整条无效带。补边也统一了后续所有步骤的坐标系——SGBM 输出、WLS 的 guide 与 ROI、置信度图都按补边尺寸对齐，最后一起裁。

`padding: false` 可以关掉，但只在「视差范围远小于图像宽度」时才有意义。

### 2.4 右图 matcher 用镜像视差范围

右视图的视差约定为**负数**（同一物理点在右图中 x 更大）：

```
rightMinDisparity = −minDisparity − numDisparities + 1 = −32 − 96 + 1 = −127
```

右图 matcher 覆盖 `[−127, −32]`，与左图 `[32, 127]` 严格镜像。**这不是可选项**：阶段二的 LRC 依赖 `d_left + d_right ≈ 0`；若右图也用正视差范围，它会把背景匹配到错误一侧，LRC 就失去意义。

## 3. 阶段一：SGBM 半全局匹配

### 3.1 能量函数

局部块匹配（BM）的问题是：在无纹理区域，所有候选视差的代价几乎相同，argmin 由噪声决定，于是出现大片「白噪声视差」和「视差抹开（bleeding）」。

SGM（Hirschmüller 2008）把问题改成**全局能量最小化**：

$$
E(D) = \sum_{p} C(p, D_p) + \sum_{p}\sum_{q \in N_p}
\Big[\, P_1 \cdot \mathbb{1}\big[|D_p - D_q| = 1\big]
+ P_2 \cdot \mathbb{1}\big[|D_p - D_q| > 1\big] \,\Big]
$$

| 项 | 含义 | 本工程取值 |
| --- | --- | --- |
| $C(p, D_p)$ | 像素级匹配代价。OpenCV 实现是 **Birchfield–Tomasi**（`calcPixelCostBT`），对光照与噪声比 SAD 稳健；差值再被 `preFilterCap` 截断 | `pre_filter_cap = 31` |
| $P_1$ | **小惩罚**：允许视差连续变化（斜面、斜坡），但不免费 | `p1_factor · block² = 8 · 25 = 200` |
| $P_2$ | **大惩罚**：视差跳变（遮挡边界、物体边缘）代价更高，抑制 bleeding | `p2_factor · block² = 256 · 25 = 6400` |

`P2/P1 = 32` 相当激进，等价于**强烈偏好分段常数（fronto-parallel）表面**——正合「金属托盘 / 平板」场景；代价是对斜面会呈现轻微阶梯。`validate()` 强制 `P2 > P1 > 0`。

### 3.2 路径聚合

全局最小化是 NP-hard。SGM 用**沿多条一维路径独立做动态规划再求和**来近似：

$$S(p,d) = \sum_r L_r(p,d)$$

$$
L_r(p,d) = C(p,d) + \min \begin{cases}
L_r(p - \mathbf{d}_r,\, d) & \text{同视差，不罚}\\
L_r(p - \mathbf{d}_r,\, d \pm 1) + P_1 & \text{视差 ±1}\\
\displaystyle\min_{k} L_r(p - \mathbf{d}_r,\, k) + P_2 & \text{任意跳变}
\end{cases}
\;-\; \min_k L_r(p - \mathbf{d}_r,\, k)
$$

最后减去路径历史最小值：它是个与 `d` 无关的常数偏移，去掉后既防止累积溢出，也不改变 argmin。

### 3.3 `mode` 的实际含义

`mode` 是这条流水线里最容易被误读的参数。以 OpenCV 5 源码为准（`stereosgbm.cpp`，`NR = 8`）：

| 配置值 | 底层实现 | 方向数 | 遍数 | 特性 |
| --- | --- | --- | --- | --- |
| `hh`（默认） | `computeDisparitySGBM` + `isFullDP()` | 8 | 2 | 两遍 full DP，精度最高、最慢、内存最大 |
| `sgbm` | `computeDisparitySGBM` | 8 | 1 | 单遍 8 方向 |
| `hh4` | `computeDisparitySGBM_HH4` | **2**（水平 + 垂直） | 1 | 只做正交两组扫描，内存最省，细结构最弱 |
| `3way` | `computeDisparity3WaySGBM<4>` | 3（每带） | 4 条带 | 分块扫描，最快；条带接缝处可能残留 |

`profile` 的差异就落在这些参数上：`balanced` = `hh` + `block 5` + `P2 factor 256`；`detail` = `hh` + `block 3` + `P2 factor 64`（更小窗口、更弱跳变惩罚，保细节）；`fast` = `3way`。

### 3.4 OpenCV 实现里三个容易被忽略的行为

这三条不在 API 文档里，但直接影响你对输出的解读：

1. **`preFilterCap` 的截断对象**：内部令 `ftzero = max(preFilterCap, 15) | 1`，然后把所有差值钳到 `±ftzero`。默认 31 → `ftzero = 31`。作用是在强边缘/高光处限制单个像素的代价，避免一个坏点主导整条路径。
2. **`compute()` 末尾无条件做 `medianBlur(disp, 3)`**，随后才按 `speckleWindowSize` 跑 `filterSpeckles`。所以 `Stages::raw` **不是**纯粹的代价 argmin，而是「SGBM 内部中值 + 内部斑点滤波」之后的结果；这也是为什么外部 LRC 之前 `raw` 的覆盖率仍然不低（默认数据上 78.8%）。
3. **`uniquenessRatio` 在代价空间内判定**：最优代价必须比次优好 `(1 + 5%)` 以上，否则该像素直接标无效。这是抑制**重复纹理**（金属栅格、周期图案）误匹配的主要手段。

### 3.5 输出编码与三个哨兵值

SGBM 输出 `CV_16S`，数值是**视差的 1/16 定点**（内部用 4 bit 小数做亚像素抛物线拟合）。无效像素写 `(minDisparity − 1) · 16`。

这条流水线在解包时做三件事：

```cpp
raw[x] = in[x] / 16.0f;                                  // ×16 定点 → 浮点像素
if (lraw[x] <= leftInvalid  || !correspondenceValid) …   // → NaN
if (rraw[x] <= rightInvalid)                     …       // 右图同样处理
```

其中 `leftInvalid = (minDisparity − 1)·16 = 496`，`rightInvalid = (rightMinDisparity − 1)·16 = −2048`。

**整条流水线统一用 `NaN` 表示无效**，而不是 0 或 −1。这样中值、`distanceTransform`、`filter2D`、`countNonZero(finiteMask(…))` 全都能直接用 `isfinite` 判断，不需要额外传递 mask；`Stages` 里也因此省掉了成对的 mask 成员。

### 3.6 可选的初始匹配器：`cv::stereo::StereoBinarySGBM`

`matcher: "binarysgbm"` 把阶段一换成 opencv_contrib **xstereo** 的
`cv::stereo::StereoBinarySGBM`。它不是新算法，而是 SGBM 的**换代价版本**：能量函数、
P1/P2 正则化、8 方向聚合完全一样，唯一区别是像素代价从 Birchfield–Tomasi 灰度差换成
**Census 描述子的 Hamming 距离**。

| | `sgbm` | `binarysgbm` |
| --- | --- | --- |
| 像素代价 | BT 灰度差，`pre_filter_cap` 截断 | Census 编码的 Hamming 距离 |
| 对亮度变化的敏感性 | 敏感（增益变了代价就变） | **不敏感**（Census 只编码邻域大小关系） |
| 视差搜索 | `minDisparity` 可为负，天然支持右视图 | 只有 `0..numDisparities` 的正偏移 |
| 输出编码 | ×16 定点，无效 = `(minDisparity−1)·16` | 同左，但输出多加了 `minDisparity` |

这条流水线接它时绕开了 xstereo 的三个实现问题（都在源码里核实过）：

1. **代价体只按绝对偏移 `0..numDisparities` 建立**（`hammingDistanceBlockMatching` 填
   `maxDisparity+1` 项，从 0 开始），`minDisparity` 并不移动搜索范围。
2. **它又把 `minDisparity` 加到每个输出值上**：
   `disp1ptr[x] = (DispType)(d + minD * DISP_SCALE)`。于是「`minDisparity=32` + 正确视差 75」
   会输出 `107×16`。
3. **`setMinDisparity()` 断言 `minDisparity >= 0`**，因此核心 SGBM 用来生成右视图的
   负视差范围（`−(min+num)+1`）在这里不可用。

三个问题用同一个配置化解：**`minDisparity = 0`、`numDisparities = 标定范围的上界`
（`minDisparity + numDisparities`，向 16 取整）**。代价体大小与正确配置相同，偏移量恒为 0。

右视图改用**水平翻转**：右图需要 `right(x) ↔ left(x + d)`，即 +x 方向的搜索。把两幅图都翻转
后，对应点在翻转坐标里落到 `u − d`，于是同一个 matcher 就能找到幅值；结果再翻回来并取负：

```
right(x) = −flippedMap[W − 1 − x]
```

> 注意：`compute(flip R, flip L)` 才是可用方向，`compute(flip L, flip R)` 和
> `compute(R, L)` 都找不到匹配（实测中位数 5 px vs 72 px）。另外右图的有效值是负的
> （最低到 `−2048`），所以它不能用 `−16` 当无效哨兵，改用 `−32768`。

**`block_size` 的合法范围取决于 kernel。** 每个 census 描述子在 `descriptor.cpp` 里都有
`kernelSize <= N` 的断言：`dense` 5、`sparse` 11、`cs` / `mod_cs` 7、`mod_census` 9；下限都是 3
（`block_size = 3` 对五种 kernel 全部合法）。`validate()` 按 kernel 精确校验，越界时给出配置
错误而不是 OpenCV 的断言文本。

**`profile` 与 matcher 是正交的两件事**，但 profile 会覆盖 `block_size` 和 `mode`，因此
profile 能不能用取决于它写的是什么：

| profile | 内容 | `sgbm` | `binarysgbm` |
| --- | --- | --- | --- |
| `balanced` | block 5, P2 256, mode hh | ✅ | ✅ |
| `detail` | block 3, P2 64, mode hh | ✅ | ✅（block 3 合法） |
| `fast` | block 5, P2 256, **mode 3way** | ✅ | ❌ 报错 |

`fast` 不能用，原因是 `mode` 而不是 block size。binarysgbm 里「快」对应的是
`MODE_SGBM`（一遍），所以等价的写法是 `"profile": ""` + `"mode": "sgbm"`。

**`mode` 只有两个取值**：`StereoBinarySGBM` 的枚举里只有 `MODE_SGBM` 和 `MODE_HH`，没有
`MODE_HH4` / `MODE_SGBM_3WAY`。因此 `matcher: "binarysgbm"` 时写 `hh4` 或 `3way` 会直接报错
（而不是静默降级成 `MODE_SGBM`——那样配置就与实际测量不符了）。

留下的两种含义还与核心 SGBM **相反**：`hh`（两遍）精度更好，`sgbm`（一遍）覆盖率更高。
实测见 [§19](#19-binarysgbm-与-sgbm-的实测对比)。

## 4. 阶段二：亚像素左右一致性检查

SGBM 自带的 `disp12MaxDiff` 只在整数级比较，且过于粗糙。这里做显式亚像素检查（`leftRightConsistency()`）：

```
对每个 d_L 有限的像素 (x, y)：
    x' = x − d_L                        // 该点在右图中的位置，一般为小数
    若 x' 越界 → 无效
    d_R = 在 (y, x') 处对右图双线性插值
    err = |d_L + d_R|                   // 注意是"加"：d_R 应为负
    有效 ⟺ d_R < 0 且 err ≤ lr_max_diff_px (1.5)
```

三个要点：

1. **`d_L + d_R ≈ 0` 是遮挡的最强过滤器。** 被遮挡的区域在另一视图根本找不到对应点；SGBM 仍会给出一个「看起来合理」的值，但反查回去对不上。`lr_error` 图保留在 `Stages` 里便于诊断。
2. **必须亚像素。** 只对 `x'` 取整会引入 ±0.5 px 量化误差，直接吃掉 1.5 px 阈值预算，系统性地把好点判坏。插值时若 `weight < 1e-6`，直接用左样本而不去引用右邻点——这是为与 numpy 逐位一致而保留的分支。
3. **`d_R < 0` 是硬性要求。** 右图若给出正视差，说明匹配到了错误的一侧，直接否决。

产出 `checked` = `raw` 中通过 LRC 的部分，其余 `NaN`。**这是整条流水线最可信的一层**，后续所有推断都以它为锚（`trusted = finiteMask(checked)`）。

代价是覆盖率骤降——弱纹理区里 SGBM 靠 `P2` 填出的平滑表面大多过不了 LRC。这正是下一阶段存在的理由。

## 5. 阶段三：WLS 引导滤波

### 5.1 目标函数

$$
\min_{a} \sum_{p} \Big[ (a_p - d_p)^2 + \lambda \sum_{q \in N_8(p)} w_{pq}\,(a_p - a_q)^2 \Big],
\qquad w_{pq} = \exp\!\Big(-\frac{(I_p - I_q)^2}{\sigma_c}\Big)
$$

- 第一项**保真**：结果要接近原始视差。
- 第二项**平滑**：权重由引导图（左图灰度 `I`）决定。颜色相近才强平滑；跨物体边界（颜色突变）权重趋 0 → **平滑不跨越边缘**。
- `λ = 100`：保真与平滑的平衡。λ 大 → 补洞更狠，但更容易抹平真边缘。
- `σ_c = 0.8`：对颜色差极敏感，等价于「只信几乎同色的邻域」。
- 8 邻域、每方向一个方程 → 稀疏线性系统，用共轭梯度求解。

### 5.2 为什么必须用 `Generic` 版本

```cpp
cv::ximgproc::createDisparityWLSFilterGeneric(true);   // true = 输出置信度图
```

`createDisparityWLSFilter(leftMatcher)` 会从 matcher 读 `minDisparity`，据此推断无效值约定并自动配平左右 ROI。这里用的是**手工补边 + 镜像右视差**，语义与它假设的完全不同，所以只能走 generic 路径，并**自己按 generic 的约定写无效哨兵**。

哨兵值不是随手取的：xicmgproc 源码里 `dst_full_size = Scalar(16 * (min_disp − 1))`，generic 路径 `min_disp = 0`，于是左图无效值 = **−16**；右图沿用标准约定 **−32768**。写错这两个值，WLS 会把无效区当真实视差参与平滑，整片糊掉。

### 5.3 三个必须遵守的工程约定

| 约定 | 原因 |
| --- | --- |
| 用 `createDisparityWLSFilterGeneric` | 见 [§5.2](#52-为什么必须用-generic-版本) |
| 左无效写 `−16`、右无效写 `−32768` | generic 约定固定，不随 `minDisparity` 偏移 |
| `guide` 传 **RGB**（不是 BGR） | OpenCV 内部固定用 `COLOR_BGR2GRAY`；本流水线从 Python（PIL 读入 RGB）移植，为逐像素对齐而刻意反着传 |

第三条是 **bug-compatibility，不是笔误**。要改就得两边一起改并重新对拍，否则 C++ 与 Python 的结果会系统性偏离。

其余参数：`setLRCthresh(round(lr_max_diff_px · 16)) = 24`（源码按 ×16 定点比较）、`setDepthDiscontinuityRadius(1)`。`filter()` 的 ROI 传 `Rect(padding, 0, width, height)`，输出仍是 ×16 定点。

### 5.4 两个副产品

- **`candidate`**：WLS 结果。注意它**不一定**比 `checked` 覆盖率高——WLS 自己也做一次 LRC（`LRCthresh`），是「补一部分、删一部分」。
- **`confidence`**：`getConfidenceMap()`，`CV_32F` 0–255，表示每个像素被多少原始有效像素支持。供 `confidence_threshold`（默认 0 = 不筛）与离线分析使用。

## 6. 阶段四：`refineDisparity()` 信任域细化

输入 `checked`（锚）、`candidate`（WLS 结果）、`confidence`。核心思想一句话：**只在离可靠点足够近的地方，才采用平滑的结果**。

| 规则 | 判据 | 作用 |
| --- | --- | --- |
| 1. 距离推断 | `distanceTransform` 到最近 trusted 点的欧氏距离 ≤ `inference_radius_px`（8 × width/640） | 限制外推范围：越远越不可信，宁可留洞 |
| 2. 保护真实跳变 | trusted 且 `abs(cand − ref) > preserve_trusted_jump_px`（2 px） | 强边缘处 WLS 会拖 1–2 px，此时回退到 LRC 结果 |
| 3. 受控中值 | 邻域有效像素 > size²/2 **且** `hi − lo ≤ median_max_span_px`（3 px）**且** `median > 0` | 只在「平滑表面内部」平滑，避免普通中值在台阶处混合前后景 |
| 4. 二次斑点清理 | `filterSpeckles(win = 20, range = 32)` | 清除 WLS 后残留的孤立小连通块 |
| 5. 合法性收尾 | 再跑一次 `correspondenceValid` | 保证输出里没有「对应点落在图外」的像素 |

几个设计细节：

- **规则 1 的半径随分辨率线性缩放**（`8 × width/640`），保证 320×240 与 1280×960 上的物理推断范围一致。
- **规则 2 体现整条流水线的价值观**：`checked` 是经过几何验证的，`candidate` 只是统计平滑的。冲突时相信前者。
- **规则 3 不是普通 `medianBlur`**：普通 3×3 中值会在深度不连续处把前景/背景混在一起，产生典型的「边缘 1 px 台阶」。三个门限里 `hi − lo ≤ 3 px` 是保边的核心——邻域视差跨度大就说明这是边缘，跳过。
- **规则 4 的舍入**：浮点 ×16 用 `nearbyint`（round-half-to-even），精确复刻 numpy `np.rint`。

最终产出 `finalMap`，以及 `inferred = finite(final) && !trusted`——即**「经后处理推断而非 LRC 直接验证」的像素**。这个掩膜的价值在于把「测到的」和「猜的」分开统计。

## 7. 数据流总图

```text
left.png / right.png (BGR, cv::imread)
  │  cvtColor(BGR2RGB)                          ← 对齐 Python 字节序
  ▼
RGB ──cvtColor(RGB2GRAY)──▶ gray ──[CLAHE 可选]──▶ copyMakeBorder(133,133) ──▶ padded
                                                                    │
                         ┌──────────────────────────────────────────┤
                         ▼                                          ▼
              SGBM left  (min= 32, n=96, MODE_HH)        SGBM right (min=-127, n=96)
                         │  CV_16S（×16 定点）                        │
                         ▼                                          ▼
                 raw（float，NaN=无效）  ◀────────────  right（负视差）
                         │
              ┌──────────┴───────────────────────────────┐
              ▼                                          ▼
   leftRightConsistency(1.5 px)                  WLS(generic, λ=100, σ=0.8, LRC=24/16)
              │                                          │
        checked ─▶ trusted                    candidate + confidence
              │                                          │
              └──────────────┬───────────────────────────┘
                             ▼
              refineDisparity(checked, candidate, confidence)
                ① dist ≤ 8 px 才允许推断
                ② |cand − ref| > 2 px → 回退 ref
                ③ 受控中值（count > size²/2 且 span ≤ 3 px）
                ④ filterSpeckles(20, 32)
                ⑤ correspondenceValid 收尾
                             ▼
                         finalMap
                             │
        ┌────────────────────┴─────────────────────┐
        ▼                                          ▼
   *_16.png (×256)                        *_mask.png
   *_vis.png                               inferred 掩膜
```

`postprocess: "none"` 时跳过阶段三、四：`finalMap = checked`，`candidate` 同值，`confidence` 全 0，`inferred` 全 0。

## 8. 设计取舍与已知局限

**刻意做的取舍**

1. **padding 覆盖整个搜索范围**，而不是接受边界无效——覆盖率优先。
2. **三层独立错误过滤**（SGBM 唯一性 → 亚像素 LRC → WLS 内部 LRC），宁可少给像素，也不给错值。
3. **「可靠优先」的细化策略**：`checked` 为锚，`candidate` 仅在距离与跳变两条规则内补充。
4. **无效值统一 `NaN`**：后处理全部「mask 自由」，代码短且不易漏判。
5. **与 Python 逐像素对齐**：RGB 通道序、`nearbyint`、`weight < 1e-6` 分支、preview 截断而非四舍五入。跨实现一致性优先于「写得更自然」。

**已知局限（也是后续可做的方向）**

1. **`P2` 不自适应。** SGBM 原版的 `P2` 应随局部梯度 `|I_p − I_q|` 变化；这里固定 6400，在「高光金属 + 强反射」的镜面区会过度平滑。
2. **单一尺度。** 视差跨度 32→127（约 4 倍），单一 `block_size` 无法同时最优。
3. **WLS 的 guide 只有灰度**，丢失颜色信息；对「同亮度不同色」（不锈钢 vs 食物）的边界不敏感。
4. **推断半径是各向同性圆**，而实际的洞沿极线方向拉长。改用沿扫描线方向的各向异性推断会更合理。
5. **`ADCensusBM` 完全未被使用**：cross-based 自适应支撑窗口 + AD/Census 组合代价，理论上在弱纹理与边缘上优于固定窗口 SGBM，是现成的对比基线。
6. **参数全局固定。** [`adcensus-tuning.md`](./adcensus-tuning.md) 已证明「聚合窗口需求取决于场景纹理」；此处 `block_size` / `p2_factor` 同样在「托盘 / 食物 / 背景」之间存在最优冲突——来自 `comparison.md` 的证据：同一组参数下 foreground EPE 0.27 px、tray 0.22 px、background 1.34 px。

**这些局限的实际量级**（`BlenderProc-main/.../output/*/evaluation_sgbm/metrics.json`，640×480）：

| 场景 | 投影 OFF 覆盖率 / EPE | 投影 ON 覆盖率 / EPE |
| --- | --- | --- |
| 空托盘 | 56.0% / 9.22 px | 98.8% / 0.25 px |
| 带食物 | 80.2% / 2.24 px | 98.5% / 0.38 px |

也就是说：**开启投影图案后，无纹理区的问题基本被数据端解决**；剩余算法空间集中在投影关闭时的补洞策略，以及投影开启时的边缘精度。

---

# 第二部分：查阅

## 9. 术语与符号

| 符号 / 词 | 含义 |
| --- | --- |
| `d_L`, `d_R` | 左图 / 右图的视差。约定 `d_R < 0`（右视图中对应点在右侧） |
| `fb` | `fx · baseline`，单位 m·px；默认 8.4 |
| `trusted` | 通过 LRC 的像素集合（`finiteMask(checked)`） |
| `checked` | LRC 通过后的视差图，未通过为 `NaN` |
| `candidate` | WLS 输出的视差图 |
| `finalMap` / `final` | 细化后的最终视差图 |
| `inferred` | `finite(final) && !trusted`：推断（而非直接验证）得到的像素 |
| `padding` | 左右补充的黑边宽度，默认 133 |
| `limits` | 物理有效的视差区间 `[low, high)` |
| `×16 定点` | OpenCV 视差内部表示：整数值 = 像素值 × 16 |
| BPR | bad pixel rate；此处 `bad_or_invalid_rate` 额外把缺失也算错 |

## 10. 配置键

读取顺序：**内置默认 < 顶层键 < `profiles[profile]`**。以 `_` 开头的键会被忽略（用于写注释）。

### 10.1 搜索窗口与相机

| 键 | 默认 | 作用 | 何时改 |
| --- | --- | --- | --- |
| `working_depth_m` | `[0.07, 0.25]` | 由深度反推视差范围 | 场景工作距离变了才改 |
| `min_disparity` / `num_disparities` | 0 / 0 | 0 = 自动推导；>0 则显式采用 | 已知精确范围时手动收紧 |
| `padding` | `true` | 左右补 `min+num+block` 宽黑边 | 视差范围远小于图宽时可关 |
| `camera.fx` / `camera.baseline_m` | 140 / 0.06 | 内联相机参数 | 标定变更 |
| `camera_file` | `""` | 指向 `camera.json`，覆盖上面的内联值 | 用仿真数据自带的标定 |

### 10.2 Matcher

| 键 | 默认 | 影响（调大 / 调小） | 何时动 |
| --- | --- | --- | --- |
| `matcher` | `sgbm` | `sgbm` / `binarysgbm`，见 [§3.6](#36-可选的初始匹配器cvstereostereobinarysgbm) | 弱纹理场景（投影关闭）优先试 `binarysgbm` |
| `profile` | `balanced` | 预设组合，会覆盖 `block_size` / `p1/p2_factor` / `mode` / `uniqueness_ratio`。`fast` 含 `mode: 3way`，因此只适用于 `sgbm` | 与 matcher 搭配见 [§3.6](#36-可选的初始匹配器cvstereostereobinarysgbm) |
| `block_size` | 5 | 大 → 更平滑、细结构丢失；小 → 保细节、噪声多 | 细结构重要时试 3 |
| `p1_factor` | 8 | 大 → 更抗拒 ±1 视差变化（斜面变阶梯） | 通常不动 |
| `p2_factor` | 256 | 大 → 更偏向分段常数面；小 → 允许更多跳变 | 镜面/曲面场景调小 |
| `mode` | `hh` | 见 [§3.3](#33-mode-的实际含义)。`hh4` / `3way` 仅 `sgbm` 可用 | 需要速度时试 `3way` |
| `uniqueness_ratio` | 5 | 大 → 更严格，覆盖率降；小 → 误匹配增多 | 重复纹理多时调大 |
| `pre_filter_cap` | 31 | 实际截断值 `max(v,15)\|1`；大 → 抗高光但弱纹理更糊 | 高光严重时试 45–63 |
| `disp12_max_diff` | 1 | SGBM 内部整数级 LRC 阈值 | 一般不动 |
| `speckle_window_size` / `speckle_range` | 40 / 1 | SGBM 内部斑点滤波：窗口 / 视差跨度 | 想保留小结构时调小窗口 |

`matcher: "binarysgbm"` 额外读这三个键（其余 matcher 参数照旧共用）：

| 键 | 默认 | 作用 |
| --- | --- | --- |
| `binary_kernel` | `mod_census` | Census 描述子变体：`dense` / `sparse` / `cs` / `mod_cs` / `mod_census`。默认值在实测中最好 |
| `binary_symmetric_subpixel` | `false` | `false` = 抛物线插值，`true` = symmetric 插值 |
| `binary_average_speckle_removal` | `true` | `true` = 邻域平均去斑，`false` = 严格阈值去斑 |

> `mean_variation` 与 `star` 两种 kernel **故意没有开放**：前者要求传入 xstereo 自己从不
> 构造的积分图（`descriptor.cpp:149` 直接 assert），后者要求 `block_size >= 7`。


### 10.3 前后处理

| 键 | 默认 | 影响 | 何时动 |
| --- | --- | --- | --- |
| `preprocess` | `none` | `none` / `clahe`；CLAHE 提升局部对比，但可能让左右视图响应不一致 | 弱光、低对比时试 |
| `clahe_clip_limit` / `clahe_tile_grid` | 2.0 / 8 | CLAHE 参数 | 仅在开启 CLAHE 时 |
| `postprocess` | `wls` | `wls` / `none` | 需要对比基线时设 `none` |
| `lr_max_diff_px` | 1.5 | 亚像素 LRC 阈值：大 → 覆盖↑误匹配↑ | 投影关闭时可试 1.0 收紧 |
| `wls_lambda` | 100 | 平滑强度：大 → 补洞多、边缘易被抹 | 洞多但边缘糊时在 50–200 间找 |
| `wls_sigma_color` | 0.8 | 颜色相似度尺度：大 → 补洞少 | 边界渗色时调大 |
| `wls_discontinuity_radius` | 1 | 边缘保护半径 | 一般不动 |
| `confidence_threshold` | 0 | WLS 置信度下限（0–255） | 精度优先、可接受低覆盖时设 10–50 |
| `inference_radius_at_640_px` | 8 | 推断最大距离（随分辨率缩放） | 洞大且贴近可信区时调大 |
| `preserve_trusted_jump_px` | 2 | 与 trusted 值偏离超过此值就回退 | 边缘出现「被拖走」时调小 |
| `median_size` | 3 | 受控中值窗口；C++ 允许 0/3/5/7，Python 只允许 0/3/5 | 一般不动 |
| `median_max_span_px` | 3 | 中值生效的平滑度上限 | 边缘失真时调小 |
| `post_speckle_size` / `post_speckle_range_px` | 20 / 2 | 二次斑点清除 | 小物体被删时调小 |

### 10.4 输出

| 键 | 默认 | 说明 |
| --- | --- | --- |
| `output.directory` | `""` | **空 = 左图所在目录**，不是进程工作目录 |
| `output.prefix` | `cpp_` | 产物文件名前缀 |
| `output.save_artifacts` | `true` | false 则只打印统计 |
| `output.show_window` | `true` | true 且存在 `DISPLAY`/`WAYLAND_DISPLAY` 时**阻塞**在 `waitKey(0)` |
| `output.preview_disparity_range_px` | `[0, 0]` | `[0,0]` = 由结果 1–99 百分位推导 |

## 11. 产物与编码约定

| 文件 | 编码 | 说明 |
| --- | --- | --- |
| `*disparity_raw_left_16.png` | `uint16`，`rint(d × 256)` | SGBM 输出（已含 OpenCV 内部中值与斑点滤波）；0 = 无效 |
| `*disparity_final_left_16.png` | 同上 | 最终结果 |
| `*disparity_final_left_vis.png` | `uint8` RGB | 5 色调色板；范围 1–99 百分位或配置指定；**颜色截断而非四舍五入**（复刻 numpy `astype(uint8)`） |
| `*prediction_valid_mask_left.png` | `uint8` 0/255 | `isfinite(final)` |
| `*postprocess_inferred_mask_left.png` | `uint8` 0/255 | 推断像素（见 [§6](#6-阶段四refinedisparity-信任域细化)） |

`saveUint16()` 会把「有效但无法用 uint16 表示」的像素写成 0，因此 PNG 只适合预览；需要无损值请走 Python 侧的 `.npy`。

## 12. 命令行、退出码与日志

```
algorithm_test <left.png> <right.png> [sgbm_config.json]
```

| 参数 | 说明 |
| --- | --- |
| `<left.png>` / `<right.png>` | 已校正、同尺寸、同一行对齐的 8 位三通道图 |
| `[sgbm_config.json]` | 省略或传 `-` 时使用内置默认 |

| 退出码 | 含义 |
| --- | --- |
| 0 | 成功（包括「无显示环境跳过 imshow」「imshow 失败但已落盘」） |
| 1 | 命令行用法错误 |
| 2 | 运行期异常（读图失败、配置非法、搜索窗口放不下等） |

启动时打印的日志字段：`config` / `input` / `camera`（含 `fx·b`）/ `search`（`min`、`num`、`padding`、`limits`）/ `matcher`（`block`、`P1`、`P2`、`mode`、`pfc`、`ur`、`disp12`、`speckle`）/ `pre/post`；运行后打印各阶段覆盖率与 `StageTimer` 计时（`StageTimer` 在每阶段析构时记录，`main()` 汇总占比）。

## 13. 与 `sgbm_pipeline.py` 的对照

| Python | C++ | 一致性 |
| --- | --- | --- |
| `resolve_settings()` | `resolve()` | 搜索窗口推导一致 |
| `correspondence_valid()` | `correspondenceValid()` | 一致 |
| `left_right_consistency()` | `leftRightConsistency()` | 一致（含 `weight < 1e-6` 分支） |
| `_matcher()` | `makeMatcher()` | 一致 |
| `match_stereo(..., "wls")` | `matchStereo()` | 一致（generic WLS + 相同哨兵） |
| `refine_disparity()` | `refineDisparity()` | 一致 |
| `postprocess="auto"` → `guided` | 无 | C++ 只实现 WLS |

### 三处必须知道的不一致

1. **`0` 的语义相反。** C++ 把 `num_disparities: 0` 当「自动推导」；Python 用 `setdefault`，键存在即采用 0，随后 `count <= 0` 直接 `ValueError`。因此 `test/sgbm_config.json` **不能**直接喂给 `sgbm_pipeline.py`（烤箱示例自带的配置没写这两个键，才没暴露）。
2. **profile 覆盖范围不同。** Python 的 `settings.update(profiles[profile])` 能覆盖 `working_depth_m`、`padding`、`min/num_disparities`；C++ 的 profile 只覆盖 matcher 与后处理那批键，搜索窗口键只从根节点读取。
3. **取值/校验差异。** `median_size`：Python 允许 0/3/5，C++ 另允许 7；`disp12_max_diff`：Python 的 `_matcher()` 硬编码 1，C++ 读配置。

## 14. 概念与函数的对应

按阅读顺序排列，便于在源码里定位（`sgbmpipe` 命名空间内）：

| 概念 | 符号 | 备注 |
| --- | --- | --- |
| 配置结构 | `Settings` / `Config` / `Resolved` | `Resolved` 是 `resolve()` 的产物，携带搜索窗口 |
| 搜索窗口推导 | `resolve()` | 见 [§2.2](#22-搜索窗口由工作深度反推) |
| 无效值判据 | `finite()` / `finiteMask()` | 全流水线唯一判据 |
| 有效对应判定 | `correspondenceValid()` | 视差在 `limits` 内且对应点在图内 |
| 亚像素 LRC | `leftRightConsistency()` | 同时产出 `lrError` |
| matcher 装配 | `makeMatcher()` / `toCvMode()` / `toModeName()` | 左、右各建一个 |
| 二进制 matcher | `makeBinaryMatcher()` / `binaryDisparity()` / `binaryNumDisparities()` | 见 [§3.6](#36-可选的初始匹配器cvstereostereobinarysgbm) |
| 阶段产物容器 | `Stages` | `raw` / `checked` / `candidate` / `finalMap` / `confidence` / `lrError` / `inferred` + 计时 |
| 阶段计时 | `StageTimer` | RAII，析构时写入 `Stages::timings` |
| 主流水线 | `matchStereo()` | 串起阶段一到四 |
| 细化 | `refineDisparity()` | 见 [§6](#6-阶段四refinedisparity-信任域细化) |
| 落盘 | `saveUint16()` / `saveMask()` / `saveDisparityPreview()` | 编码约定见 [§11](#11-产物与编码约定) |
| 配置读取 | `namespace detail`（`readInt`/`readDouble`/`readEnum`/`validate` 等）、`loadConfig()`、`applyCameraFile()` | 用 `cv::FileStorage` 读 JSON |
| 预览范围 | `disparityPercentiles()` / `validFraction()` | 1–99 百分位、有效率 |
| 入口 | `main()` | CLI、日志、落盘、显示 |

---

# 第三部分：操作

## 15. 构建

### 15.1 依赖

- OpenCV **5.0**，编译时必须带 `ximgproc`（`createDisparityWLSFilterGeneric` 所在模块）。
- OpenMP。
- PCL **可选**：源码未使用，找不到就跳过。
- CMake ≥ 3.16、支持 C++17 的编译器。

`CMakeLists.txt` 默认从 `$ENV{HOME}/opt/opencv5` 找 OpenCV，并做两道校验：版本必须 ≥ 5
（除非显式放宽）、`OpenCV_LIB_COMPONENTS` 必须含 `opencv_ximgproc`。`opencv_xstereo` 是**可选**
的：有则 `matcher: "binarysgbm"` 可用，没有则该选项编译期禁用并在运行期给出明确错误。

### 15.2 常规构建

```bash
cd /home/min/projects/test
cmake -S . -B build          # 输出: OpenCV 5.0.0 from ~/opt/opencv5/lib/cmake/opencv5
cmake --build build -j$(nproc)
```

### 15.3 OpenCV 5 带来的唯一源码差异

OpenCV 5 把 `calib3d` 拆成 `calib` / `stereo` / `geometry`。`cv::DIST_L2` 随之移入 `geometry`，不再由 `<opencv2/opencv.hpp>` 带出，因此 `test.cpp` 里有一段版本保护：

```cpp
#if CV_VERSION_MAJOR >= 5
#include <opencv2/geometry/2d.hpp>
#endif
```

其余 API 在 5.0 中保持不变：`StereoSGBM::create` 的十一参数签名、`MODE_*` 常量、`createDisparityWLSFilterGeneric`、`filter(..., ROI)`、`FileStorage`。

### 15.4 可覆盖的构建开关

| 用途 | 参数 |
| --- | --- |
| 指定另一套 OpenCV | `-DOpenCV_DIR=<prefix>/lib/cmake/opencv5` |
| 本地前缀不在 `~/opt/opencv5` | `-DOPENCV_LOCAL_PREFIX=<prefix>` |
| 临时退回 4.x | `-DREQUIRE_OPENCV5=OFF` |

### 15.5 常见报错

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `Found OpenCV 4.6.0, but OpenCV 5 is required` | `CMakeCache.txt` 把 `OpenCV_DIR` 钉在旧版本上——`CMAKE_PREFIX_PATH` 无法覆盖 | 删掉整个 `build/` 再 configure |
| `... was built without the ximgproc module` | OpenCV 编译时没带 `opencv_contrib` | 用 `-DOPENCV_EXTRA_MODULES_PATH=<opencv_contrib>/modules` 重编 |
| `'DIST_L2' is not a member of 'cv'` | 用了 OpenCV 5 但缺 `geometry` 头 | 确认 `#if CV_VERSION_MAJOR >= 5` 那段存在，且该 OpenCV 编了 `geometry` 模块 |

## 16. 运行

### 16.1 跑一对图

```bash
cd /home/min/projects/test
./build/algorithm_test left.png right.png sgbm_config.json
```

产物写到 `output.directory`（留空则是**左图所在目录**），前缀 `cpp_`。

### 16.2 跑仿真数据（带真值，可与 `evaluation_sgbm` 对拍）

```bash
./build/algorithm_test \
  ../BlenderProc-main/examples/advanced/stereo_oven_food_tray/output/on/left.png \
  ../BlenderProc-main/examples/advanced/stereo_oven_food_tray/output/on/right.png \
  sgbm_config.json
```

`camera_file` 留空时用内联的 `fx = 140 / baseline = 0.06`，与烤箱示例的 `camera.json` 一致；要精确对齐就把 `camera_file` 指向 `output/on/camera.json`。

### 16.3 读输出

关注三行：

```
coverage: raw 78.83%  checked 74.88%  candidate 87.82%  final 83.04%  inferred 25071 px
timing  : matching 322.3 ms, total 360.9 ms
--- stage timings (Debug (-O0) build) ---
```

- `raw → checked` 掉多少，就是 LRC 剔除的比例（遮挡 + 误匹配）；
- `checked → candidate` 上升，说明 WLS 补洞有效；若反而下降，说明 WLS 的 LRC 删得更多；
- `inferred` 越大，最终结果中「推断」占比越高，应结合掩膜判断可信度。

## 17. 无 GUI 环境

`sgbm_config.json` 默认 `show_window: true`。存在 `DISPLAY` 或 `WAYLAND_DISPLAY`（WSLg 下通常是 `:0`）时会真的弹窗并**阻塞在 `waitKey(0)`**。

批处理时用临时配置关掉：

```bash
sed 's#"show_window": true#"show_window": false#' sgbm_config.json > /tmp/headless.json
./build/algorithm_test left.png right.png /tmp/headless.json
```

完全没有显示环境时程序会自行跳过 `imshow` 并返回 0。

## 18. 调参入门：按症状找参数

先确认症状出现在哪个阶段（看 `coverage` 那一行），再动对应参数——改错阶段的参数往往只是把问题挪了个位置。

| 症状 | 先看 | 可调 |
| --- | --- | --- |
| 大面积无有效值（覆盖率低），且集中在平滑区 | `raw` 覆盖率是否已经很低 | 提高 `pre_filter_cap`；试 `preprocess: clahe`；放宽 `uniqueness_ratio` |
| `raw` 高但 `checked` 骤降 | LRC 太严或 SGBM 结果不稳 | `lr_max_diff_px` 1.5 → 2.0；`speckle_window_size` 调小 |
| 边界处视差被「抹开」1–3 px | `error_preview` / 掩膜 | 降低 `wls_lambda`；降低 `wls_sigma_color`；降低 `median_max_span_px`；降低 `preserve_trusted_jump_px` |
| 洞多且离可信区较远 | `inferred` 占比 | 提高 `inference_radius_at_640_px`；提高 `wls_lambda`（注意付边缘代价） |
| 小物体/细结构被整块删除 | `postprocess_inferred_mask` | 降低 `post_speckle_size`（20 → 10）；降低 `speckle_window_size` |
| 重复纹理区整片错配 | 原始图 | 提高 `uniqueness_ratio`（5 → 10/15） |
| 高光处成片无效 | 原始图 | 提高 `pre_filter_cap`；考虑开启投影图案（数据端） |
| 结果整体偏慢 | 计时表 | `mode` 改 `3way`；`block_size` 不动的前提下先把 `postprocess` 设 `none` 做基线 |

> 调参的边界条件：`num_disparities` 必须是 16 的倍数；`P2 > P1 > 0`；`block_size` 必须是正奇数；`median_size ∈ {0,3,5,7}`（C++）。`validate()` 会在 configure 阶段之外、运行期启动时检查这些。

## 19. `binarysgbm` 与 `sgbm` 的实测对比

同一套后处理（`lr_max_diff_px 1.5` + WLS + 细化）、同一标定、同一批 640×480 仿真数据，
只有 `matcher` 不同。预测取 `*_disparity_final_left_16.png ÷ 256`，指标由
`stereo_utils.evaluate_disparity` 在 `valid_stereo_mask_left` 上计算。

| 场景 | 投影 | matcher | 覆盖率 | EPE | bad-or-invalid | 前景 EPE | 托盘 EPE | 背景 EPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 空托盘 | OFF | `sgbm` | 56.00% | 9.222 px | 65.13% | — | 11.573 px | 1.137 px |
| 空托盘 | OFF | `binarysgbm` | **85.06%** | **0.937 px** | **31.14%** | — | **0.992 px** | **0.562 px** |
| 空托盘 | ON | `sgbm` | 98.81% | **0.248 px** | **2.77%** | — | **0.085 px** | 1.339 px |
| 空托盘 | ON | `binarysgbm` | **98.83%** | 0.274 px | 3.72% | — | 0.168 px | **0.974 px** |
| 带食物 | OFF | `sgbm` | 80.17% | 2.237 px | 41.96% | **0.331 px** | 3.222 px | 1.143 px |
| 带食物 | OFF | `binarysgbm` | **87.67%** | **1.000 px** | **28.81%** | 0.390 px | **1.251 px** | **0.676 px** |
| 带食物 | ON | `sgbm` | **98.49%** | **0.377 px** | **4.73%** | **0.273 px** | **0.216 px** | 1.341 px |
| 带食物 | ON | `binarysgbm` | 98.16% | 0.524 px | 6.83% | 1.116 px | 0.274 px | **1.003 px** |

结论：

1. **投影关闭时 `binarysgbm` 是压倒性的**：空托盘的覆盖率 56% → 85%，EPE 9.22 → 0.94 px，
   `bad_or_invalid` 65% → 31%；带食物同样 80% → 88%、2.24 → 1.00 px。托盘区域的 EPE 从
   11.57 px 降到 0.99 px，正是 Census 代价不受投影亮度渐变影响的直接体现。
2. **投影开启时两者基本持平**：覆盖率相同（98.5% vs 98.8%），`sgbm` 的 EPE 更好
   （0.25/0.38 vs 0.27/0.52）。此时纹理已经由投影补齐，Census 的优势失效，而它的整像素
   量化 + 更粗糙的代价曲面让精度略逊。
3. **背景区域 `binarysgbm` 反而更稳**（1.0 vs 1.34 px）：腔体壁是弱纹理，属于同一个机制。

### 参数敏感性（投影关闭）

**`binary_kernel`**（`final` 覆盖率，两个 OFF 数据集）：

| kernel | 空托盘 | 带食物 |
| --- | --- | --- |
| `mod_census`（默认） | **68.70%** | **70.17%** |
| `dense` | 65.38% | 69.31% |
| `mod_cs` | 61.64% | 63.61% |
| `cs` | 60.31% | 66.79% |
| `sparse` | 30.54% | 59.82% |

默认值恰好最好，不需要调。

**`mode`** —— 与核心 SGBM 的结论相反：

| mode | 空托盘 覆盖 / EPE | 带食物 覆盖 / EPE |
| --- | --- | --- |
| `hh`（默认） | 85.06% / **0.937 px** | 87.67% / **1.000 px** |
| `sgbm` | **92.25%** / 1.463 px | **91.53%** / 1.808 px |

一遍（`sgbm`）的覆盖率高出 5–6 个百分点、`bad_or_invalid` 更低（25% vs 31%），但 EPE 接近翻倍。
**要覆盖率选 `sgbm`，要精度选 `hh`。**

**耗时**（Debug 构建、640×480）：`binarysgbm` 双向匹配约 680 ms，`sgbm` 约 310 ms。

### 适用判断

| 工况 | 建议 |
| --- | --- |
| 投影关闭（无图案，纯环境光） | **`binarysgbm`** —— 覆盖率与精度同时更好 |
| 投影开启（有散斑图案） | `sgbm` —— 精度更好，覆盖率相当 |
| 两种工况都有 | 按 `matcher` 切换，或在数据端固定开启投影 |

也就是说：**投影图案和 `binarysgbm` 解决的是同一个问题**（弱纹理下的可匹配性），
二者叠加收益递减；只保留其一时，`binarysgbm` 是更好的算法侧答案。
