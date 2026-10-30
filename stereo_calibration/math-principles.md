# 双目标定的数学原理

> 从像素坐标到三维坐标需要哪些量，这些量怎么从标定板图像里解出来，以及为什么某些错误在重投影误差里完全不可见。

| 项 | 值 |
| --- | --- |
| 对应实现 | [`stereo_calib.cpp`](./stereo_calib.cpp) |
| 使用说明 | [`README.md`](./README.md) |
| 依赖 | OpenCV 5（`calib` / `stereo` / `geometry` / `objdetect`） |

## 阅读路径

| 你想知道 | 去这里 |
| --- | --- |
| 每个公式是怎么来的 | 第 1–8 节 |
| 为什么标定做好了还会出错 | [第 6.4 节](#64-尺度是不可观测的)、[第 9 节](#9-校验指标的数学含义) |
| 哪些位姿会让标定退化 | [第 5.8 节](#58-位姿数量与退化)、[第 10 节](#10-数值条件与退化) |
| 公式对应代码里哪个函数 | [第 11 节](#11-与实现的对应) |

---

## 1. 待求量的自由度

标定的目标是解出一组参数，使「三维点 → 两幅图像上的像素」这个映射能够被精确预测。

**单目**：内参 $K$ 有 4 个自由度（$f_x, f_y, c_x, c_y$；若考虑像素不正交再加偏斜 $\gamma$，共 5 个），畸变系数通常是 5 个（$k_1, k_2, p_1, p_2, k_3$）。本仓库在 OpenCV 样例数据上解出的 $D$ 就是 5 维。

**双目**：在两台相机各自的内参之外，再加入它们之间的相对位姿——旋转 $R$ 有 3 个自由度，平移 $T$ 有 3 个，共 6 个。

这些量不能从一幅图像里读出来，因为投影本身是「三维降二维」，信息有损。可用的额外信息来自**已知几何的靶标**：一个方格边长为 $s$ 的棋盘格，内角点的三维坐标在靶标坐标系里是精确已知的。

于是一幅 $9\times6$ 的靶标图像提供 $54$ 个角点、$108$ 个标量观测，代价是引入该位姿的 6 个外参。净余 102 个约束，足以确定几个内参——这正是标定可行的原因。

---

## 2. 坐标系与符号

| 符号 | 含义 |
| --- | --- |
| $\mathbf{X}$ | 世界坐标，齐次形式 $\tilde{\mathbf{X}} = (X, Y, Z, 1)^{\mathsf T}$ |
| $\mathbf{X}_1, \mathbf{X}_2$ | 左、右相机坐标 |
| $\mathbf{x} = (u, v, 1)^{\mathsf T}$ | 像素坐标，齐次 |
| $\mathbf{y} = (x, y, 1)^{\mathsf T}$ | 归一化坐标，$\mathbf{y} \propto K^{-1}\mathbf{x}$ |
| $K, D$ | 内参矩阵、畸变系数 |
| $R_i, \mathbf{t}_i$ | 第 $i$ 个位姿的靶标 → 相机变换 |
| $R, T$ | 相机 1 → 相机 2 的相对位姿 |
| $\mathbf{x}'$ | 加撇表示校正后（rectified）的量 |

坐标链：

$$
\mathbf{X} \xrightarrow{\;R_i,\;\mathbf{t}_i\;} \mathbf{X}_c \xrightarrow{\;\text{投影}\;/\;Z_c\;} \mathbf{y} \xrightarrow{\;D\;} \mathbf{y}_d \xrightarrow{\;K\;} \mathbf{x}
$$

反方向（像素 → 射线）没有闭式解：$D$ 只在正方向是多项式，反向需要数值迭代（§4.2）。

---

## 3. 针孔投影

### 3.1 投影方程

$$
\mathbf{X}_c = R\,\mathbf{X} + \mathbf{t}, \qquad
\mathbf{x} = K\,\mathbf{X}_c \big/ Z_c
$$

展开为

$$
u = f_x\frac{X_c}{Z_c} + \gamma\frac{Y_c}{Z_c} + c_x, \qquad
v = f_y\frac{Y_c}{Z_c} + c_y
$$

$$
K = \begin{bmatrix} f_x & \gamma & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}
$$

各参数的物理含义：

- $f_x = F \cdot s_x$，$F$ 是物理焦距，$s_x$ 是每毫米的像素数，所以**焦距的单位是像素**。这正是「视差以像素计、深度以米计」时中间要乘一个 $f$ 的原因。
- $f_x \ne f_y$ 反映像素不是正方形（或者感光单元的填充方式不对称）。本数据集 $f_x = 536.07$、$f_y = 536.02$，差 0.01 %，基本相等。
- $(c_x, c_y)$ 是主点，即光轴与像平面的交点，通常接近图像中心但很少精确等于中心。
- $\gamma$ 是像素坐标轴的不正交量（skew）。现代 CMOS 传感器几乎总是 $\gamma = 0$，OpenCV 默认也不估计它。

### 3.2 归一化坐标

令 $\mathbf{y} = K^{-1}\mathbf{x}$，则 $x = X_c/Z_c$、$y = Y_c/Z_c$。几何上这是把点投影到 $Z_c = 1$ 的平面上。归一化坐标的好处是把「相机内部结构」和「相机姿态」解耦：

$$
\mathbf{y} = \big[\,R \mid \mathbf{t}\,\big]\tilde{\mathbf{X}} \quad\text{(相差一个尺度)}
$$

畸变也定义在归一化坐标上，与分辨率无关。

### 3.3 一个像素约定的差异

不同实现的像素原点约定差 0.5 像素：OpenCV 把左上角**像素的中心**定为 $(0,0)$，而部分文献与工具箱把左上角**像素的角点**定为原点。两者只让 $c_x, c_y$ 差 $0.5$，$f$ 不受影响。跨实现对比主点时必须先对齐这个约定，否则会看到恒定 0.5 px 的「系统性偏差」并被误判成实现错误。

---

## 4. 畸变模型

### 4.1 多项式模型

记归一化坐标的半径 $r^2 = x^2 + y^2$，OpenCV 的默认模型是

$$
\begin{aligned}
x_d &= x\,\frac{1 + k_1 r^2 + k_2 r^4 + k_3 r^6}{1 + k_4 r^2 + k_5 r^4 + k_6 r^6}
      + 2p_1 xy + p_2\left(r^2 + 2x^2\right) + s_1 r^2 + s_2 r^4 \\[2pt]
y_d &= y\,\frac{1 + k_1 r^2 + k_2 r^4 + k_3 r^6}{1 + k_4 r^2 + k_5 r^4 + k_6 r^6}
      + p_1\left(r^2 + 2y^2\right) + 2p_2 xy + s_3 r^2 + s_4 r^4
\end{aligned}
$$

各项的来源不同：

| 项 | 名称 | 物理成因 |
| --- | --- | --- |
| $k_1, k_2, k_3$ | 径向（多项式） | 透镜曲率使放大率随入射角变化，直边变弯 |
| $k_4, k_5, k_6$ | 径向（有理） | 同上，用分式逼近，对大畸变更稳 |
| $p_1, p_2$ | 切向 | 镜片与传感器不平行 |
| $s_1 \dots s_4$ | 薄棱镜 | 镜片倾斜引入的楔形效应 |
| $\tau_x, \tau_y$ | 倾斜传感器 | 像平面与光轴不垂直（`CALIB_TILTED_MODEL`） |

只用 2 个径向系数（$k_1, k_2$）通常在归一化半径 $r \approx 0.6$ 之外开始欠拟合；残差不会消失，而是被外参和主点吸收掉，表现为**同一靶标在不同画面位置给出不一致的位姿**。加上 $k_3$（5 参数）对普通工业镜头足够，这是本仓库的配置。

参数越多，过拟合的风险越大。风险来源不是「参数个数超过观测数」（角点总是远多于畸变参数），而是**畸变参数在图面上的空间自由度超过角点的覆盖范围**：只有画面中心有角点时，$k_4 \dots k_6$ 这些高阶项几乎没有约束，它们会去拟合局部残差，代价是外推到边缘时发散。经验判据是**每个畸变参数至少配 3 个位姿**，且角点必须铺满画面。

### 4.2 为什么反投影要迭代

给定畸变后的 $\mathbf{y}_d$，要求 $\mathbf{y}$，需要解

$$
\mathbf{y}_d = f(\mathbf{y})
$$

这个方程没有闭式逆解（$f$ 是高次多项式）。OpenCV 的 `undistortPoints` 用不动点迭代：把分式部分当成一个作用于 $\mathbf{y}$ 的标量修正因子 $\rho$，用上一步的估计值算出 $\rho$ 与切向项，再从 $\mathbf{y}_d$ 里扣掉并乘回：

$$
\begin{aligned}
r^2 &= x^2 + y^2 \\
\rho &= \frac{1 + k_4 r^2 + k_5 r^4 + k_6 r^6}{1 + k_1 r^2 + k_2 r^4 + k_3 r^6} \\
\Delta_x &= 2p_1 xy + p_2\left(r^2 + 2x^2\right), \qquad
\Delta_y = p_1\left(r^2 + 2y^2\right) + 2p_2 xy \\
x_{\text{new}} &= \left(x_d - \Delta_x\right)\rho, \qquad
y_{\text{new}} = \left(y_d - \Delta_y\right)\rho
\end{aligned}
$$

右侧的 $r^2, \rho, \Delta$ 全部用上一步的 $(x, y)$ 计算，$(x, y) \leftarrow (x_{\text{new}}, y_{\text{new}})$ 迭代。初值取 $\mathbf{y}^{(0)} = \mathbf{y}_d$。

通常 5 次左右就能收敛；OpenCV 5 的 `undistortPoints` 默认判据是 `MAX_ITER + EPS`、10 次、$10^{-8}$。这一步在极线误差校验里会用到：校验不是比较像素，而是把角点投到校正后的归一化平面再比较行坐标。

---

## 5. 单目标定：张正友方法

### 5.1 平面靶标给出单应

令靶标平面为 $Z = 0$，则 $\tilde{\mathbf{X}} = (X, Y, 0, 1)^{\mathsf T}$：

$$
\mathbf{x} = K\big[\mathbf{r}_1\;\mathbf{r}_2\;\mathbf{r}_3\;\mathbf{t}\big]
\begin{bmatrix}X\\Y\\0\\1\end{bmatrix}
= K\big[\mathbf{r}_1\;\mathbf{r}_2\;\mathbf{t}\big]
\begin{bmatrix}X\\Y\\1\end{bmatrix}
= H\begin{bmatrix}X\\Y\\1\end{bmatrix}
$$

$H$ 是 $3\times3$ 单应，把靶标平面映到像平面。它同时吸收了 $K$、靶标的旋转和位移，**这是关键**：一幅图像只能给出 $H$（8 个自由度），而 $K$ 和该位姿的外参是 10 个未知量。所以单一视角永远不足以分离内参，必须用多个不同朝向的位姿联合求解。

### 5.2 单应的 DLT 求解

由 $\mathbf{x} \sim H\mathbf{X}$，写成叉积形式 $\mathbf{x} \times H\mathbf{X} = \mathbf{0}$，展开后每个点给 2 个独立方程（第 3 个是前两个的线性组合）。$n$ 个点堆成

$$
A\,\mathbf{h} = \mathbf{0}, \qquad A \in \mathbb{R}^{2n \times 9}
$$

$\mathbf{h}$ 是 $H$ 按行展开的 9 维向量。$A$ 的秩最多 8，所以 $\mathbf{h}$ 取 $A$ 的最小奇异值对应的右奇异向量（$A = U\Sigma V^{\mathsf T}$，$\mathbf{h} = V$ 的最后一列）。这就是「最小代数误差」解，也等价于 $\|A\mathbf{h}\|^2$ 在 $\|\mathbf{h}\| = 1$ 下的最小化。

**必须做归一化。** 像素坐标量级是 $10^2$–$10^3$，而 $H$ 的元素跨越 $10^4$ 量级；直接堆 $A$ 会让 $A^{\mathsf T}A$ 的条件数达到 $10^{8}$ 以上，双精度也掉精度。标准做法是先把两组点各自平移到质心、缩放到平均距离 $\sqrt{2}$，解完再把变换乘回 $H$。这不是优化技巧，而是不归一化就算不出结果。

### 5.3 两个约束

把 $H$ 按列写为 $[\mathbf{h}_1\;\mathbf{h}_2\;\mathbf{h}_3]$，与 $H = \lambda K[\mathbf{r}_1\;\mathbf{r}_2\;\mathbf{t}]$ 对比：

$$
\mathbf{r}_1 = \frac{1}{\lambda}K^{-1}\mathbf{h}_1, \qquad
\mathbf{r}_2 = \frac{1}{\lambda}K^{-1}\mathbf{h}_2
$$

旋转矩阵的列向量是单位正交的，于是 $\mathbf{r}_1^{\mathsf T}\mathbf{r}_2 = 0$、$\|\mathbf{r}_1\| = \|\mathbf{r}_2\|$：

$$
\boxed{\;\mathbf{h}_1^{\mathsf T}K^{-\mathsf T}K^{-1}\mathbf{h}_2 = 0\;}
\qquad
\boxed{\;\mathbf{h}_1^{\mathsf T}K^{-\mathsf T}K^{-1}\mathbf{h}_1 = \mathbf{h}_2^{\mathsf T}K^{-\mathsf T}K^{-1}\mathbf{h}_2\;}
$$

这两个方程里已经没有外参了，只剩 $K$。每幅图像给出 2 个方程。

### 5.4 绝对二次曲线像与线性解

记

$$
B = K^{-\mathsf T}K^{-1} =
\begin{bmatrix}
B_{11} & B_{12} & B_{13}\\
B_{12} & B_{22} & B_{23}\\
B_{13} & B_{23} & B_{33}
\end{bmatrix}
$$

$B$ 是对称阵，差一个尺度共 5 个未知量。几何上它是「绝对二次曲线」在像平面上的像，只与内参有关。

把 $B$ 的元素代入，可得 $B$ 与 $K$ 的显式关系（$\alpha = f_x$、$\beta = f_y$、$\gamma$ 为偏斜）：

$$
\begin{aligned}
B_{11} &= \frac{1}{\alpha^2}, &
B_{12} &= -\frac{\gamma}{\alpha^2\beta}, &
B_{13} &= \frac{\gamma v_0 - u_0\beta}{\alpha^2\beta}\\[3pt]
B_{22} &= \frac{\gamma^2}{\alpha^2\beta^2} + \frac{1}{\beta^2}, &
B_{23} &= -\frac{\gamma(\gamma v_0 - u_0\beta)}{\alpha^2\beta^2} - \frac{v_0}{\beta^2}, &
B_{33} &= \frac{(\gamma v_0 - u_0\beta)^2}{\alpha^2\beta^2} + \frac{v_0^2}{\beta^2} + 1
\end{aligned}
$$

定义 6 维向量 $\mathbf{v}_{ij}$ 使 $\mathbf{v}_{ij}^{\mathsf T}\mathbf{b} = \mathbf{h}_i^{\mathsf T}B\mathbf{h}_j$，其中 $\mathbf{b} = (B_{11}, B_{12}, B_{22}, B_{13}, B_{23}, B_{33})^{\mathsf T}$：

$$
\mathbf{v}_{ij} =
\begin{bmatrix}
h_{i1}h_{j1}\\
h_{i1}h_{j2} + h_{i2}h_{j1}\\
h_{i2}h_{j2}\\
h_{i3}h_{j1} + h_{i1}h_{j3}\\
h_{i3}h_{j2} + h_{i2}h_{j3}\\
h_{i3}h_{j3}
\end{bmatrix}
$$

5.3 的两个约束变成 $\mathbf{v}_{12}^{\mathsf T}\mathbf{b} = 0$ 与 $(\mathbf{v}_{11} - \mathbf{v}_{22})^{\mathsf T}\mathbf{b} = 0$。$n$ 幅图像堆成

$$
V\,\mathbf{b} = \mathbf{0}, \qquad V \in \mathbb{R}^{2n\times 6}
$$

$\mathbf{b}$ 取 $V$ 的最小奇异向量。**这就是「至少 3 个位姿」的来源**：3 幅给出 6 个方程，恰好确定 6 个未知量（含尺度）；若偏斜已知为 0，未知量降到 4 个，2 幅理论上够，但实际需要冗余来压制噪声，所以实现里要求至少 3 对。

### 5.5 内参的闭式恢复

$\mathbf{b}$ 解出后直接代数求解（$\gamma = 0$ 时可化简）：

$$
\begin{aligned}
v_0 &= \frac{B_{12}B_{13} - B_{11}B_{23}}{B_{11}B_{22} - B_{12}^2}, &
\lambda &= B_{33} - \frac{B_{13}^2 + v_0\left(B_{12}B_{13} - B_{11}B_{23}\right)}{B_{11}}\\[4pt]
\alpha &= \sqrt{\frac{\lambda}{B_{11}}}, &
\beta &= \sqrt{\frac{\lambda B_{11}}{B_{11}B_{22} - B_{12}^2}}\\[4pt]
\gamma &= -\frac{B_{12}\alpha^2\beta}{\lambda}, &
u_0 &= \frac{\gamma v_0}{\beta} - \frac{B_{13}\alpha^2}{\lambda}
\end{aligned}
$$

这是线性估计，没有用到任何非线性优化。它的作用是给下一步提供一个「足够好的初值」——不是最终答案。

### 5.6 外参恢复与正交化

$$
\mathbf{r}_1 = \frac{K^{-1}\mathbf{h}_1}{\|K^{-1}\mathbf{h}_1\|}, \quad
\mathbf{r}_2 = \frac{K^{-1}\mathbf{h}_2}{\|K^{-1}\mathbf{h}_2\|}, \quad
\mathbf{r}_3 = \mathbf{r}_1\times\mathbf{r}_2, \quad
\mathbf{t} = \frac{K^{-1}\mathbf{h}_3}{\|K^{-1}\mathbf{h}_1\|}
$$

问题是估计出的 $[\mathbf{r}_1\;\mathbf{r}_2\;\mathbf{r}_3]$ **不是严格正交阵**（噪声破坏了正交性）。把它投影到 $SO(3)$ 的标准做法是 SVD：设 $M = U\Sigma V^{\mathsf T}$，取 $R = UV^{\mathsf T}$，然后修正行列式符号

$$
R = U\,\mathrm{diag}\!\left(1,\,1,\,\det(UV^{\mathsf T})\right)V^{\mathsf T}
$$

不做这一步，初始 $R$ 不是旋转矩阵，对应的投影与 $\mathbf{t}$ 的尺度都不自洽，LM 要先花若干次迭代去修正初值误差，且容易收敛到局部极小。

### 5.7 极大似然细化

线性解只最小化代数误差，与「像素上的距离」不是一回事。真正的目标函数是重投影误差：

$$
\min_{K,\,D,\,\{R_i,\mathbf{t}_i\}} \sum_{i=1}^{n}\sum_{j=1}^{m}
\left\| \mathbf{x}_{ij} - \hat{\mathbf{x}}\!\left(K, D, R_i, \mathbf{t}_i, \mathbf{X}_j\right) \right\|^2
$$

注意 $\mathbf{X}_j$ 是**已知且固定**的（靶标坐标系里的格点），所以这不是完整的 bundle adjustment，而是「内参 + 各位姿外参」的联合调整。

用 Levenberg–Marquardt 求解。记残差向量 $\mathbf{r}$，雅可比 $J = \partial\mathbf{r}/\partial\boldsymbol{\theta}$，迭代式为

$$
\left(J^{\mathsf T}J + \lambda\,\mathrm{diag}\!\left(J^{\mathsf T}J\right)\right)\delta\boldsymbol{\theta} = -J^{\mathsf T}\mathbf{r},
\qquad \boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \delta\boldsymbol{\theta}
$$

$\lambda$ 大时退化为梯度下降（保证下降），小时接近高斯-牛顿（收敛快）。OpenCV 用稀疏实现：$J$ 是块稀疏的——每个观测只依赖「全局内参」和「该位姿的 6 个外参」两组变量，$J^{\mathsf T}J$ 因此是箭头型结构，可以用 Schur 补把外参块消掉，只解内参那块稠密子问题。

参数规模：$4$（内参，$\gamma = 0$）+ $5$（畸变）+ $6n$（外参）。本数据集 $n = 13$，共 87 个未知量，观测 702 个点（1404 个标量方程）。

### 5.8 位姿数量与退化

回到 5.4：每幅图像给 2 个方程。若所有靶标位姿都是**正对镜头、只做平移**，则 $\mathbf{r}_1 = (1,0,0)$、$\mathbf{r}_2 = (0,1,0)$，单应退化成

$$
H = K\begin{bmatrix}1&0&t_x\\0&1&t_y\\0&0&t_z\end{bmatrix}
\;\;\Longrightarrow\;\;
\mathbf{h}_1 = \lambda(f_x, 0, 0)^{\mathsf T},\;\; \mathbf{h}_2 = \lambda(0, f_y, 0)^{\mathsf T}
$$

代入 5.3 会发现两条约束都没有给出有效信息：

$$
\mathbf{h}_1^{\mathsf T}B\,\mathbf{h}_2 = \lambda^2 f_x f_y B_{12} = 0 \;\Rightarrow\; \gamma = 0,
\qquad
\mathbf{h}_1^{\mathsf T}B\,\mathbf{h}_1 = \lambda^2 f_x^2\cdot\frac{1}{f_x^2} = \lambda^2 = \mathbf{h}_2^{\mathsf T}B\,\mathbf{h}_2
$$

第一式只剩 $\gamma = 0$（在 $\gamma$ 已固定为 0 时等于没有约束）；第二式**退化成恒等式**——右端代入 $B_{11} = 1/f_x^2$、$B_{22} = 1/f_y^2$ 后两边同为 $\lambda^2$，不含任何信息。于是整组正对镜头的位姿确定不了任何内参：$c_x, c_y$ 与 $f_x, f_y$ 全部不可观。

物理上很好理解：靶标只有平移时，主点偏移可以被「靶标位置也有偏移」完全补偿，焦距偏差可以被「靶标距离也有偏差」完全补偿，三者在投影方程里是同一个自由度。

因此有效位姿必须包含**倾斜**。经验上：绕两个轴的倾斜各覆盖 $\pm 30^\circ$ 以上，靶标在画面四个角和中心都出现，才谈得上标定可靠。

另一种退化是纯旋转（$|\mathbf{t}| \approx 0$）：外参不可分离，$T$ 无意义。

---

## 6. 双目外参

### 6.1 从单目到双目

同一个靶标在世界坐标系里位置相同，两个相机各自给出

$$
\mathbf{X}_1 = R_1\mathbf{X} + \mathbf{t}_1, \qquad \mathbf{X}_2 = R_2\mathbf{X} + \mathbf{t}_2
$$

消去 $\mathbf{X}$：

$$
\boxed{\;\mathbf{X}_2 = \underbrace{R_2R_1^{\mathsf T}}_{R}\,\mathbf{X}_1 + \underbrace{\mathbf{t}_2 - R\,\mathbf{t}_1}_{T}\;}
$$

这就是 OpenCV 的约定 $\mathbf{X}_2 = R\mathbf{X}_1 + T$。$R$ 是 $3\times3$ 旋转，$T$ 是 $3\times1$ 平移，$|T|$ 即基线长度。

### 6.2 本质矩阵与基础矩阵

$$
E = [T]_{\times}R, \qquad F = K_2^{-\mathsf T}E\,K_1^{-1}
$$

其中

$$
[T]_{\times} = \begin{bmatrix}0&-T_z&T_y\\T_z&0&-T_x\\-T_y&T_x&0\end{bmatrix}
$$

$E$ 把归一化坐标联系起来：$\mathbf{y}_2^{\mathsf T}E\,\mathbf{y}_1 = 0$；$F$ 则直接作用于像素：$\mathbf{x}_2^{\mathsf T}F\,\mathbf{x}_1 = 0$。

两者各有 5 个自由度（$R$ 的 3 个 + $T$ 的方向 2 个，**不含尺度**），都是秩 2 的 $3\times3$ 矩阵。

秩 2 的代数含义是存在一条右零空间：$E\mathbf{v} = \mathbf{0}$ 的非零解满足 $[T]_{\times}R\mathbf{v} = \mathbf{0}$，即 $R\mathbf{v} \parallel T$，所以 $\mathbf{v} \propto R^{\mathsf T}T$——这正是第二个相机光心在第一个相机坐标系里的方向，也就是极点方向。

自由度缺失的物理后果是：**只靠点对应**时，$E$ 只约束 $T$ 的**方向**而不约束 $|T|$，从对应关系恢复位姿只能得到「相差一个尺度」的解。标定之所以能给出以米计的 $T$，是因为额外引入了靶标的物理尺寸——而这个尺寸是外部输入的，不是从图像里解出来的（§6.4）。

### 6.3 联合优化

有了初值之后，`stereoCalibrate` 最小化两路的重投影误差之和：

$$
\min_{R,\,T}\;\sum_{i=1}^{n}\sum_{j=1}^{m}
\left( \left\|\mathbf{x}_{ij}^{(1)} - \hat{\mathbf{x}}^{(1)}\right\|^2
     + \left\|\mathbf{x}_{ij}^{(2)} - \hat{\mathbf{x}}^{(2)}\right\|^2 \right)
$$

两种模式：

| 模式 | 优化变量 | 参数个数 | 适用场景 |
| --- | --- | --- | --- |
| `CALIB_FIX_INTRINSIC`（默认） | 仅 $R, T$ | 6 | 单目已标好且质量可信 |
| 联合优化（`--refine-intrinsics`） | $K_1, D_1, K_2, D_2, R, T$ | $6 + 2(4+5) = 24$ | 单目标定不够准，愿意用外参约束反过来修内参 |

一次优化 24 个参数比 6 个难收敛得多，而且内参误差和外参误差在目标函数里强耦合：模型不足（比如畸变阶数不够）时，两者会互相补偿出一个「看起来很好」的解。本仓库默认冻结内参，就是避免这种补偿。

`perViewErrors` 的形状是 $n\times2$：第 0 列属于相机 1，第 1 列属于相机 2。传一维 `vector<double>` 会被 `OutputArray` 拒绝（它要求这个形状）。

### 6.4 尺度是不可观测的

这是本节最重要的一条，也是「重投影误差小 ≠ 外参可用」的严格理由。

设把靶标的三维格点整体缩放 $k$ 倍，同时把两个相机的位置 $\mathbf{t}_1, \mathbf{t}_2$ 也缩放 $k$ 倍。投影

$$
\mathbf{x} = K\frac{R\,\mathbf{X} + \mathbf{t}}{Z}
$$

的分子分母同乘 $k$，**结果完全不变**。于是：

- 重投影误差一字不变；
- 基线 $|T| = |\mathbf{t}_2 - R\mathbf{t}_1|$ 变成 $k|T|$；
- 深度 $Z = fB/d$ 也变成 $kZ$。

而 $\mathbf{X}$ 的尺度和 $\mathbf{t}$ 的尺度**都由 `--square-size` 决定**。所以方格边长填错时，标定结果在数学上完美自洽：RMS 不变、极线误差不变、连「三角化出的方格边长与 `--square-size` 是否一致」都不变（因为那个检查的分子分母也一起缩放了）。

在本数据集上实测（OpenCV 样例数据，13 对图像）：

| `--square-size` | 单目 RMS (L/R) | 双目 RMS | 基线 | 三角化边长 | 边长相对误差 |
| --- | --- | --- | --- | --- | --- |
| 0.020 | 0.4087 / 0.4586 px | 0.4478 px | 0.066899 m | 20.0307 mm | +0.153 % |
| 0.025 | 0.4087 / 0.4586 px | 0.4478 px | 0.083623 m | 25.0384 mm | +0.154 % |
| 0.030 | 0.4087 / 0.4586 px | 0.4478 px | 0.100348 m | 30.0461 mm | +0.154 % |

三个配置的 RMS 完全一致，边长误差也一致。**没有任何图像内的检验能发现它**。绝对尺度只能来自外部：用卡尺量方格，或者用一个尺寸已知的物体交叉验证。

推论：三角化的边长检验真正测的是「基线与焦距的相互一致性」（例如裁剪错位、$Q$ 修正写反、视差图坐标系不对都会让它变大），它**不验证**绝对尺度。这一点在 §9.3 展开。

---

## 7. 极线校正

### 7.1 极线几何

对左图上一点 $\mathbf{x}_1$，它在右图上的对应点必定落在一条直线——**极线**——上：

$$
\mathbf{l}_2 = F\,\mathbf{x}_1
$$

因为 $\mathbf{x}_2^{\mathsf T}\mathbf{l}_2 = \mathbf{x}_2^{\mathsf T}F\mathbf{x}_1 = 0$。所有极线交于**极点** $\mathbf{e}_2$（$F^{\mathsf T}\mathbf{e}_2 = \mathbf{0}$ 的非零解）。

一般情形下极线是斜的，搜索必须二维。校正的目的就是把极线变成水平线，让搜索退化为一维。

### 7.2 目标：让基础矩阵的第一行为零

校正后的相机对满足 $P_1' = K'[I\mid\mathbf{0}]$、$P_2' = K'[I\mid T']$，其中 $T' = (T_x', 0, 0)^{\mathsf T}$。于是

$$
F' = K'^{-\mathsf T}\,[T']_{\times}R'\,K'^{-1} = K'^{-\mathsf T}[T']_{\times}K'^{-1}
$$

$[T']_{\times}$ 的第一行全为零，而 $K'^{-\mathsf T}$ 是下三角阵（$K'^{-1}$ 上三角，转置后下三角），其第一行只有对角元非零，所以

$$
F' = \begin{bmatrix}0&0&0\\ \ast&\ast&\ast\\ \ast&\ast&\ast\end{bmatrix}
$$

即 $\mathbf{l}_2 = F'\mathbf{x}_1$ 的水平分量恒为 0，极线全是水平线。等价地：任意两点满足 $v_2 = v_1$。

校正的结果全部压在这个结构里。检查实现是否正确，最小判据是「$F'$ 的第一行是否为零」，不必逐点验证。

### 7.3 Bouguet 构造

要显式构造 $R_1', R_2'$。思路分两步：先把两相机各自转一半，让它们的光轴平行；再整体旋转，让新的 $x$ 轴对齐基线。

**第一步：半角旋转。** 设 $R = \mathrm{Rodrigues}(\mathbf{r})$，令

$$
R_{1/2} = \mathrm{Rodrigues}\!\left(\tfrac{1}{2}\mathbf{r}\right)
\;\;\Longrightarrow\;\;
R_{1/2}^{2} = R
$$

左相机转 $R_{1/2}$，右相机转 $R_{1/2}^{\mathsf T}$（即反向转同样角度）。这样两个光轴变成平行，且各自只承担一半的调整量，重采样的畸变最小。

**第二步：对齐基线。** 构造新的基向量

$$
\mathbf{e}_1 = \frac{T}{|T|}, \qquad
\mathbf{e}_2 = \frac{1}{\sqrt{T_x^2 + T_y^2}}\begin{bmatrix}-T_y\\T_x\\0\end{bmatrix}, \qquad
\mathbf{e}_3 = \mathbf{e}_1\times\mathbf{e}_2
$$

$\mathbf{e}_2$ 由 $\mathbf{e}_1$ 与光轴 $z$ 叉乘归一化而来（与两者都垂直），$\mathbf{e}_3$ 补成右手系。旋转矩阵按行堆叠：

$$
R_{\text{rect}} = \begin{bmatrix}\mathbf{e}_1^{\mathsf T}\\ \mathbf{e}_2^{\mathsf T}\\ \mathbf{e}_3^{\mathsf T}\end{bmatrix}
$$

最终

$$
R_1' = R_{\text{rect}}\,R_{1/2}, \qquad R_2' = R_{\text{rect}}\,R_{1/2}^{\mathsf T}
$$

OpenCV 的 `stereoRectify` 实现的就是这一构造。$R_1'$ 与 $R_2'$ 的作用是把「未校正相机坐标」变成「校正后相机坐标」（基变换，不是点变换）。

**符号约定。** $\mathbf{e}_1 = T/|T|$ 会让新 $x$ 轴顺着 $T$ 的方向。对本数据集的左右相机，$T_x < 0$，若直接取 $T/|T|$ 会翻转 $x$ 轴使图像镜像。实际取的是反方向，使校正后的平移 $T' = R_{\text{rect}}T$ 满足 $T_x' = -|T|$，从而 $P_2$ 的焦距为正、视差为正。判断符号是否正确的实用标准是：$f' > 0$ 且 $d > 0$。

### 7.4 新的投影矩阵

$$
P_1' = \begin{bmatrix} f'&0&c_{x1}'&0\\ 0&f'&c_y'&0\\ 0&0&1&0\end{bmatrix}, \qquad
P_2' = \begin{bmatrix} f'&0&c_{x2}'&T_x'f'\\ 0&f'&c_y'&0\\ 0&0&1&0\end{bmatrix}
$$

三个要点：

1. **两路焦距相同**（$f'$），且焦距在 $x, y$ 方向相同。校正后相机是「同一个虚拟相机平移两次」，所以 $f_x' = f_y' = f'$ 是构造出来的，不是拟合出来的。
2. **行对齐体现在 $c_y'$ 相同**；`STEREO_ZERO_DISPARITY` 进一步强制 $c_{x1}' = c_{x2}'$，即 $\text{doffs} = 0$。本数据集实测 $c_{x1}' = c_{x2}' = 350.6118$，$\text{doffs} = 0$。
3. $P_2'(0,3) = T_x' f'$，本数据集为 $-43.5506 = -0.083623 \times 520.796$。这个数可以当自检用：它必须等于 $f' \cdot T_x'$，对不上说明 $T$ 的量纲与 $P$ 的焦距不是同一套。

### 7.5 `alpha` 的几何含义

$f'$ 不是自由的。它有一个可行区间：

- $f'$ 取到**最大** $f'_{\max}$ 时，校正后的每个像素都能在原图上找到数据（无黑边），但视野最小；
- $f'$ 取到**最小** $f'_{\min}$ 时，原图的所有像素都被保留，但四周出现黑边。

`alpha` 在这个区间里线性插值：

$$
f'(\alpha) = (1-\alpha)\,f'_{\max} + \alpha\,f'_{\min}, \qquad f'_{\max} = f'(\alpha{=}0)
$$

本数据集实测（原始 $f_x = 536.07$）：

| `--alpha` | $f'$ | 输出有效像素 |
| --- | --- | --- |
| 0.0 | 520.7956 | 100.00 % |
| 0.5 | 473.3244 | 94.62 % |
| 1.0 | 425.8532 | 82.52 % |

中点校验：$(520.7956 + 425.8532)/2 = 473.3244$，与 `alpha = 0.5` 的实测值逐位相同，验证了线性插值。

**这就是「校正后不能直接用原始 $f_x$ 换算深度」的定量理由**：$f'$ 与 $f_x$ 差 2.85 %（$\alpha = 0$）到 20.6 %（$\alpha = 1$），而深度正比于 $f$，误差会 1:1 传递。同时主点也变了（$c_y$ 从 235.54 到 243.05），任何依赖主点的计算都要用 $c'$。

### 7.6 视差—深度关系与 Q 矩阵

校正后，同一物点在两幅图上的行坐标相同，水平差即视差

$$
d(u_1) = u_1 - u_2
= \left(f'\frac{X}{Z} + c_{x1}'\right) - \left(f'\frac{X + T_x'}{Z} + c_{x2}'\right)
= -\frac{f' T_x'}{Z} + \left(c_{x1}' - c_{x2}'\right)
$$

记基线 $B = |T|$。由 §7.3 的符号约定 $T_x' = -B$，故 $-f'T_x'/Z = f'B/Z$。再记 $d_{\text{off}} = c_{x1}' - c_{x2}'$：

$$
\boxed{\;d = \frac{f'B}{Z} + d_{\text{off}}
\quad\Longleftrightarrow\quad
Z = \frac{f'B}{\,d - d_{\text{off}}\,}\;}
$$

$f'B$ 是本仓库代码里的 `focalBaseline`（单位 px·m），本数据集为 43.5506。

$Q$ 矩阵把上式写成线性形式（`reprojectImageTo3D` 用）：

$$
Q = \begin{bmatrix}
1&0&0&-c_{x1}'\\
0&1&0&-c_y'\\
0&0&0&f'\\
0&0&-\dfrac{1}{T_x'}&\dfrac{d_{\text{off}}}{T_x'}
\end{bmatrix}
$$

对 $(u, v, d, 1)^{\mathsf T}$ 做左乘，得齐次三维点，再除以第 4 个分量：

$$
Z = \frac{f'}{\left(d_{\text{off}} - d\right)/T_x'} = \frac{f'T_x'}{d_{\text{off}} - d} = \frac{f'B}{d - d_{\text{off}}} \;\;\checkmark
$$

本数据集实测 $Q(3,2) = 11.958 = -1/T_x'$、$Q(2,3) = 520.7956 = f'$、$Q(3,3) = 0$，与上式一致。

**裁剪对 $Q$ 的影响**：$Q$ 的前两行第 4 列是 $-c'$，而裁剪只是把像素原点平移 $(x_0, y_0)$，等价于 $c' \leftarrow c' - (x_0, y_0)$，所以

$$
Q(0,3) \mathrel{+}= x_0, \qquad Q(1,3) \mathrel{+}= y_0
$$

忘掉这一步时，$Z$ 仍然正确（第 3 行不受影响），但 $X = (u - c_{x1}')Z/f'$ 会整体偏移 $x_0 Z/f'$——点云外形看起来「歪了」，而所有深度值都正常，很难发现。`stereo_rectification` 里的 `checkQAgainstP()` 就是为此而写：把一个三维点经 $P_1, P_2$ 投影再用 $Q$ 反投，实测误差约 $10^{-16}$ m。

### 7.7 为什么校正后可以做一维搜索

有三个条件同时成立才能只沿行搜索：

1. **行对齐**：$v_1 = v_2$（几何保证，$F'$ 的第一行为零）。
2. **视差有界且符号一致**：$d = f'B/Z + d_{\text{off}}$ 随 $Z$ 减小而单调增大，$Z \in [Z_{\min}, Z_{\max}]$ 对应 $d \in [d_{\min}, d_{\max}]$。搜索窗口必须覆盖这个区间，否则近处/远处的点被强行匹配到窗口边界。
3. **两幅图来自同一虚拟相机**：$f'$ 与 $c_y'$ 相同，否则同样的水平位移对应不同的入射角。

实现上还有两条是「一维」的必要条件，但属于工程而非几何：

- 两幅图的尺寸必须完全一致（裁 ROI 时取交集，不能各裁各的）；
- 越界像素（$remap$ 填的 0）必须被标记为无效，否则黑边会与图像内容匹配出大片虚假视差。

---

## 8. 三角化与不确定度

### 8.1 三角化

给定 $P_1, P_2$ 与对应点 $\mathbf{x}_1, \mathbf{x}_2$，求 $\mathbf{X}$。由 $\mathbf{x}_i \times (P_i\mathbf{X}) = \mathbf{0}$，每个视图给出 2 个独立方程，堆成 $4\times4$：

$$
\begin{bmatrix}
u_2\,\mathbf{p}_2^{(3)} - \mathbf{p}_2^{(1)}\\
v_2\,\mathbf{p}_2^{(3)} - \mathbf{p}_2^{(2)}\\
u_1\,\mathbf{p}_1^{(3)} - \mathbf{p}_1^{(1)}\\
v_1\,\mathbf{p}_1^{(3)} - \mathbf{p}_1^{(2)}
\end{bmatrix}\mathbf{X} = \mathbf{0}
$$

$\mathbf{p}^{(k)}$ 是 $P$ 的第 $k$ 行。解取 $A$ 最小奇异值对应的右奇异向量（DLT），或对归一化坐标做类似运算。这一步在 `verify()` 里用来把校正后的角点还原成三维，从而测量靶标的平面性和格距。

**类型陷阱**：`triangulatePoints` 的输出深度跟随输入。传 `vector<Point2f>` 得到 `CV_32F` 的 $4\times N$ 矩阵，用 `at<double>` 读出来是垃圾。必须显式传 `CV_64F`。

### 8.2 视差误差的传播

由 $Z = f'B/d'$（$d' = d - d_{\text{off}}$）：

$$
\frac{\partial Z}{\partial d'} = -\frac{f'B}{d'^2} = -\frac{Z^2}{f'B}
\quad\Longrightarrow\quad
\boxed{\;\sigma_Z = \frac{Z^2}{f'B}\,\sigma_d\;}
$$

同时 $\sigma_Z/Z = \sigma_B/B$：基线误差是**相对**误差，与距离无关。

两条结论直接决定工程选择：

- **深度误差随 $Z^2$ 增长**。同一套硬件，远处精度掉得极快，这不是算法问题。
- **要提精度就得增大 $f'B$**（增大基线或焦距），代价是视差范围变大、遮挡变多、近处失配。

本数据集 $f'B = 43.55$ px·m，按 $\sigma_d = 0.2$ px（SGBM 亚像素的典型量级）估算：

| 深度 $Z$ | $\sigma_Z$ | 相对误差 |
| --- | --- | --- |
| 0.3 m | 0.41 mm | 0.14 % |
| 0.5 m | 1.15 mm | 0.23 % |
| 1.0 m | 4.59 mm | 0.46 % |
| 2.0 m | 18.4 mm | 0.92 % |
| 3.0 m | 41.3 mm | 1.38 % |

反过来，给定深度精度要求可以反解视差精度：$\sigma_d = Z^2 \sigma_Z / (f'B)$。

### 8.3 视差窗口的推导

由 $d = f'B/Z + d_{\text{off}}$，深度区间 $[Z_{\text{near}}, Z_{\text{far}}]$ 对应

$$
d_{\text{near}} = \frac{f'B}{Z_{\text{near}}} + d_{\text{off}}, \qquad
d_{\text{far}} = \frac{f'B}{Z_{\text{far}}} + d_{\text{off}}
$$

StereoBM/SGBM 要求 $\text{numDisparities}$ 是 16 的倍数，且 $\text{minDisparity} + \text{numDisparities} <$ 图像宽度。取

$$
\text{minDisparity} = 16\left\lfloor \frac{d_{\text{far}}}{16}\right\rfloor, \qquad
\text{numDisparities} = 16\left\lceil \frac{d_{\text{near}} + 1}{16}\right\rceil - \text{minDisparity}
$$

向下取整 $\text{minDisparity}$ 是为了把最远平面纳入窗口，向上取整 $\text{numDisparities}$ 是为了纳入最近平面。本数据集按 0.15–1.5 m 推导：$d \in [29.03,\,290.34]$ px → `minDisparity = 16`、`numDisparities = 288`（$16\times18$）。

---

## 9. 校验指标的数学含义

标定软件通常只报一个 RMS，但 RMS 只覆盖目标函数里出现过的量。下面每个指标各自测什么、测不到什么，都要分清。

### 9.1 重投影 RMS

$$
\text{RMS} = \sqrt{\frac{1}{2nm}\sum_{i}\sum_{j}\left\|\mathbf{x}_{ij} - \hat{\mathbf{x}}_{ij}\right\|^2}
$$

它衡量「模型对已观测像素的拟合程度」。测不到的东西：

| 测不到 | 原因 |
| --- | --- |
| 绝对尺度 | §6.4 的联合缩放不变性 |
| 校正后的行对齐 | RMS 在**原始**图像上计算，不经过 $R_1', P_1'$ |
| 靶标区域之外的精度 | 只在角点处有约束，外推靠模型 |
| 畸变模型是否合适 | 模型不足时残差被外参和主点吸收 |

数值上，「小」没有绝对标准，取决于角点定位精度。`cornerSubPix` 的典型精度是 0.1–0.5 px；RMS 低于 0.5 px 就算正常；明显低于 0.1 px 反而可疑，通常意味着模型在拟合角点噪声而不是真实几何。

### 9.2 极线误差

理论形式是点到极线的距离。极线 $\mathbf{l}_2 = F\mathbf{x}_1 = (l_u, l_v, l_w)^{\mathsf T}$，点 $\mathbf{x}_2 = (u_2, v_2, 1)^{\mathsf T}$ 到它的距离：

$$
\text{dist} = \frac{\left|\mathbf{x}_2^{\mathsf T}F\,\mathbf{x}_1\right|}{\sqrt{l_u^2 + l_v^2}}
$$

$\mathbf{l}_2 = (l_u, l_v, l_w)^{\mathsf T}$ 是齐次直线，对应方程 $l_u u + l_v v + l_w = 0$，其法向长度为 $\sqrt{l_u^2 + l_v^2}$。齐次坐标差一个尺度不改变直线，但会改变 $\mathbf{x}_2^{\mathsf T}F\mathbf{x}_1$ 的数值，所以必须除以法向长度才是欧氏距离。

**本仓库的实现测量的是另一个等价量**：把左右角点各经 `undistortPoints(K, D, R', P')` 投到校正后的像素坐标系，直接比较 $|v_1' - v_2'|$。这比用 $F$ 更直接——它测的是「实际要做匹配的那两幅图」的行差，而不是代数残差。两者在理想情况下一致。

实测（OpenCV 样例数据，702 个角点，13 个视图合并统计）：

| 指标 | 值 |
| --- | --- |
| 均值 | 0.1406 px |
| P95 | 0.3536 px |
| 最大 | 3.6434 px |

均值小于 0.5 px 时，水平搜索窗只需加 1–2 px 余量；均值超过 1 px 就不能做一维搜索了。

### 9.3 平面性与尺度

把校正后的角点三角化成三维点，逐视图拟合平面：设质心 $\bar{\mathbf{X}}$，对去中心化的点做 SVD，最小奇异向量即法向 $\mathbf{n}$，则

$$
\text{RMS}_{\text{planar}} = \sqrt{\frac{1}{m}\sum_{j}\left(\mathbf{n}^{\mathsf T}(\mathbf{X}_j - \bar{\mathbf{X}})\right)^2}
$$

**必须逐视图拟合**。13 个不同位姿的三维点混在一起拟合单一平面，得到的数字由靶标位姿的分散程度决定（本数据集为数十毫米量级），反映的是「靶标摆得有多开」，不是标定质量。

平面性测的是：镜头畸变残差、角点定位噪声、靶标本身的平整度。注意由于拟合已经扣掉了线性趋势，系统性的倾斜误差不会出现在这个数里。

格距（方格边长）取相邻格点的三维距离均值，与 `--square-size` 比较。**由 §6.4，这个比较在数学上是恒等的**：三角化坐标正比于 $B$，而 $B$ 正比于 `--square-size`，两者同比缩放，相对误差不变。实测三种方格尺寸的边长误差都是 +0.15 %，就是这个原因。

所以格距检验的实际价值是**检测内部不一致**，而不是验证尺度：

| 会让它变大的问题 | 机制 |
| --- | --- |
| 校正后裁剪错位 | 像素原点与 $P$ 不一致，三角化射线错位 |
| $Q$ 未按裁剪修正 | 只影响 $X, Y$，配合尺度检验可定位 |
| 视差图与 $P$ 的坐标系不匹配 | 同上 |
| 角点顺序在两路之间镜像 | 三角化出完全错误的三维点 |

绝对尺度只能靠外部手段：卡尺量方格，或用已知尺寸的物体做交叉验证。这是单目/双目几何的固有限制，不是实现缺陷。

---

## 10. 数值条件与退化

### 10.1 归一化不是可选项

DLT 与 $V\mathbf{b} = 0$ 都依赖 SVD 求最小奇异向量。若矩阵的奇异值跨度超出双精度有效位（约 $10^{16}$），最小奇异向量会被舍入误差污染。像素坐标 $10^2$–$10^3$ 与单应元素 $10^4$ 的混合尺度足以让条件数达到 $10^8$，因此 Hartley 归一化是必需步骤。

### 10.2 常见退化位姿

| 位姿模式 | 后果 | 表现 |
| --- | --- | --- |
| 全部正对镜头，仅平移 | $c_x, c_y, f$ 均不可观 | 主点漂移数十像素、焦距漂移数个百分点，RMS 却很小 |
| 位姿朝向单一（只绕一个轴倾斜） | $V$ 列近似共线 | 内参条件数极大，结果对噪声敏感 |
| 靶标只占画面中心 | 畸变无约束 | 边缘畸变系数发散 |
| 两相机纯旋转（无基线） | $T$ 无意义 | 视差恒为 0，深度不可恢复 |
| 靶标平面与基线平行 | 三角化条件差 | 单视图三角化精度低 |

### 10.3 误差的主源

按贡献排序（本数据集）：

1. **角点定位**（0.1–0.5 px）——决定 RMS 的下限，也决定 $R, T$ 的精度；
2. **畸变模型不足**——残差被外参吸收，表现为不同位姿的 $T$ 估计不一致；
3. **靶标平面度**——印刷靶标贴在非平面基底上时，等价于给世界坐标加了噪声；
4. **数值条件**——只在退化位姿下成为主项。

算法本身（LM 收敛）通常不是瓶颈：87 个参数、1404 个观测，在双精度下毫秒级收敛。

### 10.4 对称性

两台相机参数完全相同时（仿真环境的典型情况），$R = I$、$T = (-B, 0, 0)$，标定退化为「求 $B$」。此时任何几何自检都会给出极好的数字，**包括错误的数字**：如果仿真里两相机的 $c_y$ 被人为加了偏移，$R$ 会解出一个补偿性的旋转，RMS 依然很小，但极线不再水平。这类问题只能靠 §9.2 的极线误差（在校正后像素上测量）发现。

---

## 11. 与实现的对应

| 数学步骤 | 代码位置 |
| --- | --- |
| 角点亚像素定位（决定误差下限） | `detectCorners()`，`cv::cornerSubPix` |
| 角点顺序镜像纠正 | `reverseIfMirrored()` |
| 单目标定（§5 全部） | `calibrateMono()`，`cv::calibrateCamera` |
| 双目外参（§6） | `calibrateStereoPair()`，`cv::stereoCalibrate` |
| 立体校正（§7.3–7.5） | `cv::stereoRectify(..., STEREO_ZERO_DISPARITY, alpha, ...)` |
| 极线误差（§9.2） | `verify()` → `cv::undistortPoints` + 行差 $\lvert \Delta v \rvert$ |
| 三角化与平面性（§9.3） | `verify()` → `cv::triangulatePoints` + `planarityRms()` |
| $Q$ 与深度的自洽性（§7.6） | `stereo_rectification` 的 `checkQAgainstP()` |
| 视差窗口推导（§8.3） | `stereo_rectification` 的 `--depth-range` |

---

## 12. 参考

- Z. Zhang, *A Flexible New Technique for Camera Calibration*, IEEE TPAMI 22(11), 2000. —— §5 的原始文献，含 $B$ 的推导与闭式解。
- R. Hartley, A. Zisserman, *Multiple View Geometry in Computer Vision*, 2nd ed. —— §5.2（归一化 DLT）、§6.2（$E$/$F$ 的秩与自由度）、§7.2（校正的代数刻画）、§8.1（三角化）。
- J.-Y. Bouguet, *Camera Calibration Toolbox for Matlab*, 2004. —— §7.3 的校正构造。
- A. Fusiello, E. Trucco, A. Verri, *A Compact Algorithm for Rectification of Stereo Pairs*, Machine Vision and Applications 12, 2000. —— 另一种校正构造，与 Bouguet 等价但参数化方式不同。
- OpenCV 5 `opencv2/stereo.hpp` 中 `stereoRectify` 的文档 —— §7.4 中 $P_1, P_2, Q$ 的确切形式与 $\alpha$ 的定义。
