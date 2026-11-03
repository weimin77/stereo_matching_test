---
theme: default
title: 双目视觉成像技术路线调研报告
titleTemplate: '%s'
info: |
  ## 双目视觉成像技术路线调研报告
  立体匹配算法选型与 RK3562 部署可行性评估
author: weimin77
transition: slide-left
mdc: true
colorSchema: light
aspectRatio: '16/9'
themeConfig:
  primary: '#2563eb'
---

<style>
table { font-size: 0.72rem; }
.pipeline { display: flex; flex-wrap: wrap; align-items: center; justify-content: center; gap: 0.4rem; margin: 1.2rem 0; }
.pipe-box { padding: 0.5rem 1rem; border-radius: 0.5rem; background: #2563eb; color: #fff; font-weight: 600; font-size: 0.82rem; white-space: nowrap; }
.pipe-box.mature { background: #64748b; }
.pipe-box.focus { background: #f59e0b; color: #1f2937; }
.pipe-arrow { color: #94a3b8; font-weight: 700; }
.small { font-size: 0.85rem; }
</style>

---
layout: cover
---

# 双目视觉成像技术路线调研报告

立体匹配算法选型与 RK3562 部署可行性评估

<div class="abs-br m-6 text-sm opacity-60">
  目标平台：RK3562
</div>

---
layout: default
---

# 目录

<div class="grid grid-cols-2 gap-6 text-left">
<div>

**一、调研范围**

**二、双目立体匹配方案对比**
　· 传统方法
　· 深度学习方法
　· 主动散斑辅助

**三、算法精度与泛化性能对比**

</div>
<div>

**四、RK3562 部署可行性分析**

**五、技术选型建议**

**六、公开数据集调研**

**附录 A　开源算法与数据**

</div>
</div>

---
layout: section
---

# 第一章　双目视觉处理流程

---
layout: default
---

# 双目视觉处理流程

双目视觉由两台已标定的相机同时成像，利用同一物点在左右图中的**横向位置差**恢复距离。

<div class="pipeline">
  <span class="pipe-box mature">1 双目图像采集</span><span class="pipe-arrow">→</span>
  <span class="pipe-box mature">2 双目相机标定</span><span class="pipe-arrow">→</span>
  <span class="pipe-box mature">3 去畸变与极线校正</span><span class="pipe-arrow">→</span>
  <span class="pipe-box focus">4 立体匹配 ★调研重点</span><span class="pipe-arrow">→</span>
  <span class="pipe-box mature">5 视差图计算与优化</span><span class="pipe-arrow">→</span>
  <span class="pipe-box mature">6 深度计算</span><span class="pipe-arrow">→</span>
  <span class="pipe-box mature">7 三维点云重建</span>
</div>

<div class="text-sm opacity-80 mt-2">
灰底 = 已成熟环节（OpenCV 实现）；橙底 = 本报告调研对象。视差图经左右一致性检验与空洞填充后参与深度换算。
</div>

---
layout: image-right
image: /fig2.svg
---

# 双目成像与视差计算原理

经极线校正后，对应点位于同一水平扫描线，其水平坐标差称为**视差**。

<div class="text-center text-2xl font-bold my-6">
  Z = f · B / d
</div>

- `d`：视差（像素）
- `f`：焦距（像素）
- `B`：双目基线长度
- `Z`：深度

<v-click>

**关键结论**：只需沿水平方向搜索对应点（yL = yR）。

</v-click>

---
layout: default
---

# 为什么聚焦立体匹配？

- **标定、去畸变与极线校正**：成熟环节。棋盘格/圆点靶标一次标定，OpenCV `calibrateCamera`、`stereoCalibrate`、`initUndistortRectifyMap`、`remap` 完整覆盖，作为既定前置步骤，不再展开。

- **立体匹配**：为每个像素在左右图寻找同名点，是影响「视差质量 → 深度精度 → 点云完整性」的**核心环节**。

<v-click>

**难点**：弱纹理、反光、重复纹理、遮挡下算法表现差异明显；且通常是链路中计算量最集中的部分，直接影响能否在低成本 SoC 上运行。

</v-click>

---
layout: section
---

# 一、调研范围

---
layout: default
---

# 一、调研范围

覆盖双目立体匹配的主流技术路线：

<div class="grid grid-cols-3 gap-4 text-left mt-4">
<div class="p-3 border rounded-lg">
**传统方法**
StereoBM · StereoSGBM · libSGM · ADCensus · ELAS · PatchMatch Stereo
</div>
<div class="p-3 border rounded-lg">
**深度学习方法（按代价体组织）**
三维卷积 / 二维聚合 / 迭代优化 / 基础模型
</div>
<div class="p-3 border rounded-lg">
**主动方案**
散斑投射辅助双目匹配
</div>
</div>

<v-click>

对上述方案调研四项内容：

1. 公开数据集上的精度与跨域泛化性能
2. 开源许可与训练代码完整性（能否独立复现 + 自有数据微调）
3. 模型计算量、内存占用与算子构成
4. RK3562 上的部署可行性

</v-click>

---
layout: section
---

# 二、双目立体匹配方案对比

---
layout: default
---

# 2.1 传统方法 · 典型流程

<div class="pipeline">
  <span class="pipe-box">匹配代价计算</span><span class="pipe-arrow">→</span>
  <span class="pipe-box">代价聚合</span><span class="pipe-arrow">→</span>
  <span class="pipe-box">视差选择</span><span class="pipe-arrow">→</span>
  <span class="pipe-box">视差优化</span><span class="pipe-arrow">→</span>
  <span class="pipe-box mature">输出视差图</span>
</div>

<div class="grid grid-cols-2 gap-4 text-left mt-4">
<div>

**匹配代价**：SAD（简单、对曝光敏感）· Census（对光照稳健）· BT（亚像素采样，弱纹理更稳）

</div>
<div>

**代价聚合**：局部窗口（跨边界糊边）· 自适应窗口（保边）· SGM 多方向一维 DP（精度/速度均衡）

</div>
</div>

- **视差选择**：WTA（Winner-Takes-All）取代价最小位置。
- **视差优化**：左右一致性检查 · 亚像素拟合 · 斑点过滤与空洞填充。

<div class="text-sm opacity-70">
ELAS（稀疏支持点推断）与 PatchMatch（随机平面传播）不严格按此顺序实现。
</div>

---
layout: default
---

# 2.1 传统方法对比

| 方案 | 匹配代价 | 聚合/优化 | 内存 | 弱纹理 | RK3562 可用性 |
|---|---|---|---|---|---|
| StereoBM | SAD（固定窗口） | 局部窗口 | 极低 | 差 | 可用（CPU，OpenCV） |
| StereoSGBM | BT + Sobel 预滤波 | 5 方向 SGM（默认） | 低—中 | 一般 | 可用（CPU） |
| libSGM | Census | 8 方向 SGM，CUDA | 中 | 一般 | 仅参考，需改 CPU/NEON |
| ADCensus | Census + AD | 跨聚合 + 扫描优化 | 中高 | 较好 | 需自研实现 |
| ELAS | 稀疏支持点 + 先验 | 概率推断 | 低 | 差 | 需自研实现 |
| PatchMatch Stereo | 随机采样 + 倾斜平面 | 迭代传播 | 高 | 一般 | 不适用（内存） |

<div class="text-sm opacity-70">
内存为定性分级，以 640×480、视差 64 为参照。SGBM 默认单趟 5 方向，MODE_HH 为 8 方向。libSGM 为 CUDA 实现，移植 ARM 需重写。
</div>

---
layout: default
---

# StereoSGBM 关键实现细节

**代价构成**

- Birchfield–Tomasi（BT）采样代价（非原论文互信息代价）
- 3×3 横向 Sobel 预滤波（中心行权重 2），按 `preFilterCap` 截断
- BT 代价在原始灰度与滤波响应两通道同时计算（`calcPixelCostBT`）
- ⇒ 对局部梯度敏感，对噪声/曝光鲁棒性弱于 ADCensus

**聚合方向与运行模式**

| 模式 | 方向数 | 内存 | 说明 |
|---|---|---|---|
| MODE_SGBM（默认） | 5 | 约 1.4 MiB* | 单趟 |
| MODE_HH | 8 | 数十 MiB | 两趟 DP，不作 RK3562 候选 |
| MODE_SGBM_3WAY | 3 | 更低 | 分条带并行，精度略降 |
| MODE_HH4 | 4 | 同 HH 量级 | 耗时更低 |

<div class="text-sm opacity-70">
* 640×480、视差 64 下：方向代价缓冲约 0.64 MiB + 水平代价窗 0.49 MiB 等，合计约 1.4 MiB（不含输入图像/视差图/后处理）。
</div>

---
layout: default
---

# 2.2 深度学习 · 典型流程

<div class="pipeline">
  <span class="pipe-box">左右图像</span><span class="pipe-arrow">→</span>
  <span class="pipe-box">特征提取</span><span class="pipe-arrow">→</span>
  <span class="pipe-box">代价体构建</span><span class="pipe-arrow">→</span>
  <span class="pipe-box">代价聚合/优化</span><span class="pipe-arrow">→</span>
  <span class="pipe-box">视差回归</span><span class="pipe-arrow">→</span>
  <span class="pipe-box">视差优化</span><span class="pipe-arrow">→</span>
  <span class="pipe-box mature">输出视差图</span>
</div>

- **特征提取**：将原始图像映射为特征图。
- **代价体**：典型维度 H×W×D（D 为候选视差数）；逐通道相关则多出通道维 C，实际张量还带 batch 维。
- **代价聚合**：3D 卷积（空间 + 视差联合）或 2D 聚合（视差折入通道维）。
- **视差回归**：soft-argmin 加权（连续可微）；迭代类经 GRU 多轮更新。

---
layout: default
---

# 2.2 深度学习方法分类（按代价体组织）

| 类别 | 代表方法 | 代价体形式 | 聚合/迭代 | 训练代码 | RK3562 可行性 |
|---|---|---|---|---|---|
| 三维卷积 | PSMNet、GwcNet、GA-Net、ACVNet、MobileStereoNet-3D、BANet-3D | 4D（D×H×W） | 3D 卷积堆叠 | 完整 | 不可行（Conv3D 无支持） |
| 二维聚合 | MobileStereoNet-2D、AANet、HITNet、LightStereo、BANet-2D | 3D（视差作通道，1/4 分辨率） | 2D 卷积 / 分离聚合 | LightStereo 完整，BANet 需补 | 可评估 |
| 迭代优化 | RAFT-Stereo、CREStereo、IGEV、IGEV++ | 相关体 + 查表 | GRU 多次迭代 | 完整 | 风险高（查表类算子） |
| 基础模型 | FoundationStereo、Fast-FoundationStereo、MonSter、DEFOM-Stereo | 相关体 + 基础模型特征 | 迭代细化 | 前两者仅推理 | 不可行（规模与许可） |

<div class="text-sm opacity-70">
训练代码一栏依据仓库内是否存在可运行的训练入口与数据准备脚本判定，非依据论文描述。
</div>

---
layout: default
---

# 2.2 各类别要点

- **三维卷积**：代价体 4D，开销随视差范围线性增长；RKNN 不支持 Conv3D，无部署路径，仅作精度参照。
- **二维聚合**：视差维压缩为通道维。LightStereo 与 BANet-2D 都在 1/4 分辨率构建代价体，视差通道 = max_disp/4（默认 192 → 48），这是能上低成本芯片的结构基础。
  - LightStereo：通道提升 + 分离卷积；BANet：双边注意力拆分两支聚合。
- **迭代优化**：关键风险在**查表**（GridSample / gather 类算子），其支持程度决定能否转换到 RKNN。
- **基础模型**：ViT / 深度基础模型主干，泛化上界参照 + 伪标签来源；参数量超 RK3562 承载，训练代码基本未公开。

---
layout: default
---

# 2.3 主动散斑辅助

**原理**：弱纹理表面左右图缺少可匹配特征（灰度近恒定，BT/SAD 代价在所有视差接近）。向表面投射随机纹理，提升局部对比度与唯一性。不改变双目几何关系，仍靠视差算深度。

<v-click>

**对 SGBM**：散斑增加局部梯度，提高真实匹配与其他候选的代价区分度。
⇒ 收益受纹理密度、投射对比度、表面反射影响；密度过高会引入重复性伪匹配。

</v-click>

<v-click>

**对深度学习**：作用不确定——训练未见散斑时输入分布偏移，精度可能反而降低。

</v-click>

<v-click>

**局限**：投射距离与景深受限；反射率过高/过低时散斑对比度不足。

</v-click>

---
layout: section
---

# 三、算法精度与泛化性能对比

---
layout: default
---

# 3.1 同域测试精度

| 方法 | SceneFlow EPE (px) | 参数量 (M) | 计算量 | KITTI15 D1-all (%) |
|---|---|---|---|---|
| LightStereo-S | 0.73 | 3.44 | 22.71 GFLOPs | 2.30 |
| LightStereo-M | 0.62 | 7.64 | 36.36 GFLOPs | 2.04 |
| LightStereo-L | 0.59 | 24.29 | 91.85 GFLOPs | 1.93 |
| BANet-2D | 0.57 | — | 39 GFLOPs | 1.83 |
| BANet-3D | 0.51 | — | 85 GFLOPs | 1.77 |
| MobileStereoNet-2D | 1.11 | 2.23 | 128.84 GFLOPs | 2.83 |
| CoEx | 0.67 | 2.72 | 53.39 GFLOPs | 2.13 |
| Fast-ACVNet+ | 0.59 | 3.20 | 93.08 GFLOPs | 2.01 |
| HITNet | — | — | — | 1.98 |
| IGEV-Stereo | 0.47 | — | — | 1.59 |

<div class="text-sm opacity-70">
来源：各论文原始设定；KITTI15 D1-all 为在 KITTI 上微调后的官方报告值。BANet 未报告参数量（"—"非零值）。
</div>

---
layout: default
---

# 3.2 跨域零样本泛化

| 方法 | 训练数据 | Midd-H BP-1 | Midd-Q BP-1 | ETH3D BP-1 | KITTI15 D1-all | Booster-Q BP-2 | 耗时(ms) |
|---|---|---|---|---|---|---|---|
| FoundationStereo | 大规模混合 + 基础模型预训练 | 2.49 | 2.64 | 0.50 | 2.80 | 5.18 | 496 |
| Fast-FoundationStereo | 同上 + 140 万伪标签 | 4.80 | 4.51 | 1.22 | 3.25 | 6.61 | 49 |
| RAFT-Stereo | 多源合成 | 8.48 | 8.15 | 2.14 | 4.48 | 17.84 | 164 |
| IGEV（实时版） | 大规模混合 | 12.75 | 11.28 | 5.05 | 4.00 | 23.09 | 45 |
| LightStereo-L | 仅 SceneFlow | 37.49 | 30.08 | 45.46 | 12.08 | 待测 | 30 |
| LightStereo-L | 大规模混合 | 22.64 | 16.34 | 16.34 | 4.51 | 待测 | 30 |
| BANet-2D | 仅 SceneFlow | 43.78 | 37.33 | 44.89 | 13.88 | 待测 | 29 |

<div class="text-sm opacity-70">
BP-X 为误差 > X 像素的像素占比，D1-all 为误差同时超过 3px 与真值 5% 的占比，越低越好。耗时在单块 RTX 3090、Middlebury-Q 下测得。
</div>

---
layout: default
---

# 3.2 训练数据对泛化的影响

- LightStereo-L 仅 SceneFlow 训练：ETH3D BP-1 = **45.46%**
- LightStereo-L 大规模混合数据：ETH3D BP-1 = **16.34%**
- FoundationStereo（基础模型预训练特征）：ETH3D BP-1 = **0.50%**

<v-click>

**结论**：同一结构在不同训练条件下差异显著 ⇒ 除网络结构外，**训练数据覆盖范围**也是轻量模型选型的关键因素。

</v-click>

<v-click>

**注意**：同域与零样本跨域的评价条件不同；训练数据、模型版本（如 IGEV 与实时版、LightStereo-S 与 L）及评价协议存在差异，不能仅凭单一数据集排名判断泛化性能。

</v-click>

---
layout: default
---

# 3.3 场景相关失效因素

| 失效因素 | 成因 | 受影响方案 | 缓解手段 |
|---|---|---|---|
| 弱纹理 | 窗口灰度无变化，代价失去区分度 | SGBM、LightStereo、BANet、迭代类 | 散斑投射；降低 P2；训练集加无纹理样本 |
| 镜面反射 | 高光随视角变化，左右亮度不一致 | 全部被动方案 | 偏振片；散斑；双边聚合；反射数据增强 |
| 半透明 | 光线穿透产生多个候选深度 | 全部被动方案 | 难以根治；改变入射角或多视角融合 |
| 重复纹理 | 多个视差取得相同代价 | SGBM（uniquenessRatio）、深度学习类 | 增大窗口；左右一致性校验 |
| 遮挡 | 右图不存在对应点 | 全部方案 | 左右一致性检验 + 空洞填充 |
| 光照/曝光不一致 | 双相机曝光白平衡未同步 | 全部方案 | 硬件同步曝光；亮度扰动 |

---
layout: section
---

# 四、RK3562 部署可行性分析

---
layout: default
---

# 4.1 RK3562 平台能力

| 项目 | 参数 | 对双目匹配的影响 |
|---|---|---|
| NPU | 1 TOPS INT8，单核，折合 500 GMAC/s | 轻量网络有部署可能性，需板端测试 |
| CPU | 4×Cortex-A53 @ 2.0 GHz | SGBM 可纯 CPU 运行 |
| 内存 | LPDDR4X-3200，32 位，理论 12.8 GB/s | 关注容量、带宽与中间张量占用 |
| 量化配置 | w8a8 / w8a16（后者 RK3562 独占） | 激活保留 FP16 有利于相关与 Softmax 精度 |

<div class="text-sm opacity-70">
1 TOPS INT8 按 1 MAC = 2 运算折为 500 GMAC/s（标称峰值，实际有效算力通常更低）。DDR 可用带宽扣除系统占用后估计约 8 GB/s。
</div>

---
layout: default
---

# 4.2 代价体内存（LightStereo / BANet-2D）

代价体在 1/4 分辨率特征图上构建：`correlation_volume(features_left[0], features_right[0], max_disp // 4)`，max_disp = 192 时候选视差通道数 = 48。

| 输入分辨率 | 特征图（1/4） | 代价体通道 | 代价体 FP16 |
|---|---|---|---|
| 320×240 | 80×60 | 48 | 0.44 MiB |
| 384×288 | 96×72 | 48 | 0.63 MiB |
| 640×480 | 160×120 | 48 | 1.76 MiB |
| 960×540 | 240×135 | 48 | 2.97 MiB |

<div class="text-sm opacity-70">
按 (W/4)·(H/4)·48·2 字节计算。全分辨率 192 通道为 112.5 MiB，相差约 64 倍。代价体 ≠ 峰值内存（主干特征图、通道扩展、权重、框架工作区未计入，峰值通常为数倍）。
</div>

---
layout: default
---

# 4.2 候选模型计算量

| 模型 | 论文报告 (960×540) | 640×480 折算 | 384×288 折算 | MAC 区间 (640×480) |
|---|---|---|---|---|
| LightStereo-S | 22.71 GFLOPs | 13.5 | 4.8 | 6.7 – 13.5 G |
| BANet-2D | 39 GFLOPs | 23.1 | 8.3 | 11.6 – 23.1 G |
| LightStereo-L | 91.85 GFLOPs | 54.4 | 19.6 | 27.2 – 54.4 G |
| MobileStereoNet-2D | 128.84 GFLOPs | 76.3 | 27.5 | 38.2 – 76.3 G |

<div class="text-sm opacity-70">
折算按像素面积等比（假设计算量随分辨率线性缩放）。MAC 区间对应"1 MAC = 2 FLOPs"与"报告值即 MAC"两种口径（论文未说明）。折算未考虑固定开销，小分辨率下实际会更高。
</div>

---
layout: section
---

# 五、技术选型建议

---
layout: default
---

# 5.1 方案分组

| 分组 | 方案 | 定位与理由 |
|---|---|---|
| A 优先验证 | StereoSGBM + 主动散斑 | OpenCV 成熟 CPU 实现，无需训练，部署与整定简单 |
| A 优先验证 | LightStereo-S | 二维聚合不含 Conv3D，训练链完整，可复现全量训练 |
| B 备选 | BANet-2D | 结构与 LightStereo-S 同量级，反光子集较好，复现需补文档 |
| C PC 端参考 | RAFT-Stereo、IGEV-Stereo | 精度上界 + 生成伪标签 |
| C PC 端参考 | FoundationStereo、Fast-FoundationStereo | 零样本上界；研究权重限非商业 |
| D 暂不考虑 | PSMNet、GwcNet、GA-Net、ACVNet、MobileStereoNet-3D、BANet-3D | Conv3D 在 RKNN 无支持 |
| D 暂不考虑 | CREStereo、MonSter、DEFOM-Stereo | 环境过时 / 数据未发布 / 非商用许可 |

---
layout: default
---

# 5.2 重点方案分析

<div class="grid grid-cols-3 gap-4 text-left">
<div class="p-3 border rounded-lg">
**A. SGBM + 散斑**
- OpenCV 成熟 CPU 实现，无需训练
- 640×480 / 视差 64 ≈ 1.4 MiB
- 限制：弱纹理/反光需散斑；P1/P2、窗口、preFilterCap 整定影响大
</div>
<div class="p-3 border rounded-lg">
**B. LightStereo-S**
- 二维聚合，无 Conv3D / GridSample
- 代价体小，训练链完整可微调
- 3.44 M 参数，22.71 GFLOPs
- 限制：S 版无公开跨域数据，泛化待验证
</div>
<div class="p-3 border rounded-lg">
**C. BANet-2D**
- 二维聚合，计算量同量级
- 反光子集优于同类轻量方法
- 限制：缺训练文档；SceneFlow 单训 ETH3D 44.89%
</div>
</div>

---
layout: default
---

# 5.3 A/B/C 七维度对比

| 维度 | A. SGBM + 散斑 | B. LightStereo-S | C. BANet-2D |
|---|---|---|---|
| 深度精度 | 待测 | EPE 0.73；D1-all 2.30 | EPE 0.57；D1-all 1.83 |
| 泛化性能 | 与场景强相关，换场景需重整定 | 仅 L 版有公开数据，S 版待测 | 仅 SceneFlow 单训，ETH3D 44.89% |
| 弱纹理 | 需散斑辅助，否则大量空洞 | 待测 | 同左；另有双边聚合结构 |
| 计算资源 | 纯 CPU，缓冲约 1.4 MiB | 13.5 GFLOPs；3.44 M | 23.1 GFLOPs |
| RKNN 部署 | 不涉及 NPU | 需转换与算子适配，待板端验证 | 同左 + 补训练文档 |
| 开源许可 | Apache-2.0 | 未声明 | MIT |
| 训练微调 | 无需训练，参数整定 | 全量可复现 | 有脚本，缺文档 |

---
layout: default
---

# 5.4 许可与复现条件

| 方案 | 代码许可 | 权重许可 | 训练代码 | 复现等级 |
|---|---|---|---|---|
| StereoSGBM | Apache-2.0（OpenCV） | 不涉及 | 无需训练 | A |
| LightStereo | 未声明（无 LICENSE） | 需确认 | 完整（OpenStereo） | A |
| BANet | MIT | 随仓库 | 有脚本，缺文档 | B |
| RAFT-Stereo | MIT | 随仓库 | 完整 | A |
| IGEV | MIT | 随仓库 | 完整 | A |
| FoundationStereo | NVIDIA 研究许可（限非商业） | 同左 | 仅推理 | C |
| Fast-FoundationStereo | NVIDIA 研究许可 | 同左 | 仅推理 | C |
| MonSter | 自定义，限非商业 | 同左 | 有脚本，数据未发布 | B |

<div class="text-sm opacity-70">
复现等级：A 可独立完成预训练 + 微调；B 需补文档或数据；C 仅推理。代码许可不自动覆盖权重许可。
</div>

---
layout: section
---

# 六、双目立体匹配公开数据集调研

---
layout: default
---

# 6.1 静态场景数据集

相机自身运动不构成动态场景；仅当场景含车辆、行人、动物等独立运动目标才归为动态。

| 数据集 | 真实/合成 | 主要场景 | 特点 | 视差真值 | 推荐程度 |
|---|---|---|---|---|---|
| Middlebury v3 | 真实 | 室内 | 高分辨率，精细结构，弱纹理标注准 | 稠密 | 推荐 |
| ETH3D | 真实 | 室内外 | 远景 + 低纹理；激光真值仅部分像素 | 稀疏 | 推荐 |
| InStereo2K | 真实 | 室内 | 近距离，结构光真值 | 稠密 | 推荐 |
| Booster | 真实 | 室内外 | 透明/镜面，含材质标注 | 稠密 | 可选 |
| TartanAir v2 | 合成 | 多场景 | 弱光/雾天扰动，许可允许商用 | 稠密 | 可选 |

---
layout: default
---

# 6.2 动态场景数据集

| 数据集 | 真实/合成 | 主要场景 | 特点 | 视差真值 | 推荐程度 |
|---|---|---|---|---|---|
| KITTI 2015 | 真实 | 车载道路 | 独立运动车辆；天空/远景无标注 | 稀疏 | 推荐 |
| DrivingStereo | 真实 | 车载道路 | 晴、阴、雾、雨四类环境 | 稠密 | 推荐 |
| Spring | 合成 | 自然场景 | 亚像素真值，标注刚性/非刚性运动 | 稠密 | 可选 |
| Dynamic Replica | 合成 | 室内动态 | 行人动物，实例分割 + 遮挡标注 | 稠密 | 可选 |
| SceneFlow | 合成 | 合成物体 | 主流预训练集；仅限研究 | 稠密 | 可选 |

<div class="text-sm opacity-70">
KITTI 与 DrivingStereo 以自车运动为主；评估复杂运动宜配合 Spring 或 Dynamic Replica。SceneFlow 禁止商用。
</div>

---
layout: default
---

# 6.3 数据集选择建议

| 验证场景 | 推荐数据集 | 理由 |
|---|---|---|
| 高精度静态匹配 | Middlebury v3 | 高精度视差真值 |
| 真实场景泛化 | ETH3D | 远景 + 弱纹理，评测协议公开 |
| 室内近距离 | InStereo2K | 接近设备实际使用条件，可支撑微调 |
| 反光与透明表面 | Booster | 少数对非朗伯表面提供稠密真值 |
| 室外运动目标 | KITTI 2015 | 真实车载，含独立运动车辆 |
| 复杂环境条件 | DrivingStereo | 覆盖光照与天气变化 |
| 复杂动态运动 | Spring / Dynamic Replica | 真值与运动区域标注完备 |

---
layout: section
---

# 附录 A　开源算法与数据

---
layout: default
---

# 附录 A.1　算法 GitHub 仓库（节选）

| 算法 | 类别 | 仓库 |
|---|---|---|
| StereoBM / StereoSGBM | 传统 | opencv/opencv |
| libSGM | 传统 | fixstars/libSGM |
| ADCensus | 传统 | DLuensch/StereoVision-ADCensus |
| ELAS · PatchMatch Stereo | 传统 | jfalquez/ELAS · ethan-li-coding/PatchMatchStereo |
| PSMNet · GwcNet · GA-Net · ACVNet | 三维卷积 | 官方仓库 |
| MobileStereoNet · AANet · HITNet | 二维聚合 | 官方仓库 |
| LightStereo | 二维聚合 | XiandaGuo/OpenStereo（框架集成） |
| BANet | 二维聚合 | gangweiX/BANet |
| RAFT-Stereo · CREStereo · IGEV | 迭代优化 | princeton-vl/RAFT-Stereo 等 |
| FoundationStereo · Fast-FoundationStereo | 基础模型 | NVlabs/FoundationStereo 等 |

---
layout: default
---

# 附录 A.2　数据集入口

| 数据集 | 类型 | 入口 |
|---|---|---|
| Middlebury v3 | 静态 | vision.middlebury.edu/stereo/eval3 |
| ETH3D | 静态 | eth3d.net |
| InStereo2K | 静态 | github.com/YuhuaXu/StereoDataset |
| Booster | 静态 | cvlab-unibo.github.io/booster-web |
| TartanAir v2 | 静态 | tartanair.org |
| KITTI 2015 | 动态 | cvlibs.net/datasets/kitti |
| DrivingStereo | 动态 | drivingstereo-dataset.github.io |
| Spring | 动态 | spring-benchmark.org |
| Dynamic Replica | 动态 | dynamic-stereo.github.io |
| SceneFlow | 动态 | lmb.informatik.uni-freiburg.de |

---
layout: statement
---

# 结论

优先验证 **StereoSGBM + 主动散斑** 与 **LightStereo-S**；
以基础模型作为精度/泛化上界与伪标签来源，最终由 RK3562 板端实测确认。
