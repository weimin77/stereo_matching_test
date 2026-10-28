# Stage 07 — Left–Right Consistency Check（左右一致性检查）

对应 Hirschmüller (2008) `Stereo Processing by Semiglobal Matching and Mutual Information`，Section 2.3、公式 (15)，PDF 第 5 页，期刊页码 332。

**本阶段目标：**在 Stage06 的八方向 SGM + 亚像素基础上，分别计算 `D_L` 和 `D_R`，依据左→右→左的几何对应关系剔除不一致匹配。它只标记无效，不对视差插值。论文 Section 2.5.3 的遮挡/错误匹配分类与填洞属于后续步骤。

## 几何与符号：最容易出错的部分

此工程中 `D_L` 和 `D_R` 都是**非负的视差大小**（像素），不是带符号位移：

```text
xR = xL - D_L(xL,y)
xL = xR + D_R(xR,y)
一致条件： |D_L(xL,y) - D_R(xR,y)| <= max_difference
```

`left_right_consistency_check()` 对亚像素 `xR` 使用最近邻整数采样：`xr = round(xL - dL)`；采样前必须检查实际浮点 `xR` 是否在图像内。它是一个教学上的近似，后续可改成对右视差线性插值。容差默认 1.0 px，与原论文公式 (15) 一致。

**切勿只交换左右图像：** Stage02 的 `compute_cost_volume_sad_integral(base,match)` 总是搜索 `xmatch=xbase-d`，而右图为基准时需要 `xleft=xright+d`。这里用镜像恒等式解决：

```text
base'  = flip(original_right)
match' = flip(original_left)
按原代码求 base' 的视差
D_R    = flip(D_base')  // 恢复到原右图坐标系
```

原图 `xR -> xL=xR+d`，镜像后 `uL=(W-1-xR)-d=uR-d`，正好适用于之前的 SAD、SGM、亚像素模块。右视差需要重新执行一次完整匹配，耗时约为原来的两倍。这与 OpenCV StereoSGBM 的实际右图代价重利用和检查逻辑并不完全相同。

## 学习用文件

- `include/lr_consistency.hpp`：纯 C++，无 OpenCV 依赖；`flip_horizontal`、`compute_right_disparity_flipped`、`left_right_consistency_check`。
- `tests/test_lr_consistency.cpp`：已知数值的手工用例、镜像右视差正确性、完整 Stage02→05→06→07 合成图，以及非法输入。
- `src/main.cpp`：借助 OpenCV `imread` / `imwrite` 保存左右视差、LRC 前后结果和状态图、进行 GT 对比。

## Ubuntu 编译

```bash
sudo apt install build-essential cmake libopencv-dev
cd stereo_sgbm_learning/stage07_left_right_consistency
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

即使没有 OpenCV 开发包，也可以编译运行纯 C++ 核心单元测试；`handmade_sgm_lr` 演示程序只有找到 OpenCV 后才会编译。

## 命令行运行

```bash
./build/handmade_sgm_lr \
    /path/to/left.png /path/to/right.png ./out \
    48 80 5 200 800 1.0 \
    /path/to/disparity_gt_left_16.png \
    /path/to/valid_stereo_mask.png
```

参数：`minDisp numDisp blockSize P1 P2 lrThreshold [gt_disp16.png] [gt_valid_mask.png]`。示例值为教学初值，非普遍最优参数。

## 输出文件

三个前缀 `left_before_lrc`、`right_disparity`、`left_after_lrc`，分别对应左检查前、原右图坐标系的右视差、检查后的左视差；每组生成：

- `_disp16.png`：视差乘以 **256** 存到 uint16 PNG；不是 OpenCV `StereoSGBM` 的 *16 格式；数值 0 可以代表有效零视差，也可能是无效，**必须一起读 `_valid.png`**。
- `_valid.png`：255 有效，0 无效。
- `_vis.png`：伪彩色视差图，黑色无效。
- `_refined.png`：是否应用了非零亚像素修正。
- `.pfm`：浮点视差，单位 pixel，无效为 NaN，可用于后续计算深度。

`lrc_status.png`：按照以下灰度表示分类：

| 值 | 含义 |
| --- | --- |
| 0 | 左视差本来就无效 |
| 64 | 由左视差映射到右图越界 |
| 128 | 右图对应视差无效 |
| 192 | 左右视差差值大于阈值 |
| 255 | 检查通过 |

程序终端报告各分类计数、LRC 保留比例及（如果有 GT）检查前后 EPE、RMSE、Bad1、有效覆盖率。**过滤后的 EPE 降低不自动代表整体结果更好：同时需要考察覆盖率是否下降，以及是否错删真实结构。**

## 必须理解的三个函数

1. `compute_right_disparity_flipped()`：为什么 `xL=xR+d` 经双镜像后可以复用左图为基准的 SAD/SGM/亚像素处理；输出必须再翻回来。
2. `left_right_consistency_check()`：根据 `xR=xL-dL` 查找 `D_R(xR,y)`，比较 `abs(dL-dR)`，分类并将异常值设为 `valid=0, disparity=NaN`。
3. `report_lrc()`：检查前后的有效像素数不同；不能只看保留像素上的平均误差。

下一阶段：匹配错误过滤（uniquenessRatio 与 speckleFilter）及论文 Section 2.5 的遮挡/错误匹配区别，最后逐步对照 OpenCV `stereosgbm.cpp`。
