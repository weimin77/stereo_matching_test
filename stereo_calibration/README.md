# stereo_calibration

OpenCV 5 双目标定工具。pinhole 模型 + 平面棋盘格，输出内参、外参、立体校正参数，并对结果做两项独立验证。

## 1. 它做什么

```
图像对发现 → 角点检测(cornerSubPix) → 单目标定 → 双目标定 → 立体校正 → 验证
```

| 阶段 | OpenCV 5 API | 产物 |
| --- | --- | --- |
| 角点检测 | `findChessboardCorners` + `cornerSubPix` | 每幅图 9×6 内角点 |
| 单目标定 | `calibrateCamera` | `K1,D1` / `K2,D2` |
| 双目标定 | `stereoCalibrate`（默认 `CALIB_FIX_INTRINSIC`） | `R,T,E,F` |
| 立体校正 | `stereoRectify` + `initUndistortRectifyMap` | `R1,R2,P1,P2,Q` |
| 验证 | `undistortPoints` / `triangulatePoints` | 极线误差、平面性、尺度 |

### 为什么要单独验证

重投影误差小**不能**说明外参可用。它对两类错误完全不敏感：

1. **基线错误**：`T` 整体缩放一个系数，重投影误差分毫不变（因为重投影只依赖角度）。
2. **校正失败**：`R/T` 与畸变模型不自洽时，`stereoRectify` 之后的对应点仍可能不在同一行。

所以工具额外输出两项**与重投影误差无关**的指标：

- **极线误差**：把检测到的角点经 `undistortPoints(K, D, R1, P1)` 投到校正图像，统计左右对应点的 `|Δy|`。理想为 0，它直接决定稠密匹配能否用水平搜索。
- **三角化检验**：把校正后的角点 `triangulatePoints(P1, P2)` 还原成 3D，逐视图拟合平面并测量方格边长。
  - 平面性 RMS 反映标定质量与靶标平整度；
  - 方格边长与 `--square-size` 的一致性检验的是**内部自洽**（基线与焦距是否匹配），
    不是绝对尺度：`--square-size` 填错时三角化坐标同比缩放，边长误差不变。理由见
    [math-principles.md §6.4](./math-principles.md#64-尺度是不可观测的)。

## 2. 构建

```bash
cd stereo_calibration
cmake -S . -B build
cmake --build build -j
```

默认从 `$HOME/opt/opencv5` 找 OpenCV 5（与本仓库的 `test.cpp` 一致）；其他位置用
`-DOpenCV_DIR=<prefix>/lib/cmake/opencv5` 指定。

需要 OpenCV 5 的 `calib`、`stereo`、`geometry`、`objdetect` 四个模块。CMake 会在配置阶段检查，
缺哪个直接报出来，不会留到编译或链接才失败。

## 3. 运行

```bash
./build/stereo_calib \
  --images /home/min/projects/opencv-samples/samples/data \
  --reference /home/min/projects/opencv-samples/samples/data/left_intrinsics.yml \
  --output output
```

## 4. 参数

| 选项 | 默认 | 说明 |
| --- | --- | --- |
| `--images <dir>` | `.` | 图像目录 |
| `--left-prefix` / `--right-prefix` | `left` / `right` | 文件名前缀，数字后缀用于配对 |
| `--ext` | `.jpg` | 扩展名 |
| `--pattern <WxH>` | `9x6` | **内角点**数量（不是方格数） |
| `--square-size <m>` | `0.025` | 方格边长，单位米；决定 `T` 的量纲 |
| `--alpha <0..1>` | `0` | `stereoRectify` 的 balance：0 只保留有效像素，1 保留全部 |
| `--output <dir>` | `output` | 输出目录 |
| `--reference <yml>` | — | 与已知单目标定对比（读取 `camera_matrix`） |
| `--refine-intrinsics` | 关 | 联合优化内参；默认按 `CALIB_FIX_INTRINSIC` 冻结 |
| `--same-focal-length` | 关 | 强制两台相机 `fx == fy` |
| `--sb` | 关 | 改用 `findChessboardCornersSB` |
| `--no-debug-images` | 写图 | 不写校正预览图 |
| `--no-window` | 开窗 | 从不打开窗口 |
| `--quiet` | 详细 | 只打印阶段汇总 |

## 5. 输出

```
intrinsics.yml          M1 D1 M2 D2 + 各自的 RMS
stereo.yml              M1 D1 M2 D2 R T E F R1 R2 P1 P2 Q baseline_m rms
rectified_left.png      校正后的左图
rectified_right.png     校正后的右图
epipolar_check.png      左右拼接 + 水平参考线，对应点应落在同一条线上
calibration_report.md   Markdown 汇总，可直接贴进报告
```

`stereo.yml` 的键名与 OpenCV 官方 `stereo_calib` 样例一致，`cv::FileStorage` 可直接读取。

注意它**不能**直接当 `test.cpp` 的 `camera_file` 用：`applyCameraFile()` 只接受顶层的
`fx` 和 `baseline_m` 两个节点，缺任一个就抛异常，而这里 `fx` 在 `M1` 矩阵里。要在
`test.cpp` 里用本工具的标定，两者之间需要一个转换：

1. 若 `test.cpp` 的输入已经是校正后的图像，取 `stereo.yml` 的 `M1[0][0]` 作 `fx`、
   `baseline_m` 作 `baseline_m`，写成一个只含这两项的文件即可；
2. 若输入是原始图像，还需先用 `M1/D1/M2/D2/R1/R2/P1/P2` 做 `remap`，否则
   `test.cpp` 内部按无畸变、已校正的假设建立的视差-深度关系不成立。

## 6. 在 OpenCV 样例数据上的结果

数据：`opencv-samples/samples/data/left01..14` + `right01..14`（缺 10，实际 13 对，640×480）。

`--quiet` 输出：

```
=== 1. 图像对 ===
  找到 13 对图像

=== 2. 角点检测 ===
  图像尺寸 640 × 480，有效视图 13

=== 3. 单目标定 ===
  left  rms 0.4087 px  fx 536.073  fy 536.016  cx 342.370  cy 235.537  k1 -0.26509  k2 -0.04674
  right rms 0.4586 px  fx 542.355  fy 541.615  cx 328.324  cy 246.947  k1 -0.28054  k2 +0.10432

=== 4. 双目标定 ===
  rms 0.4478 px   baseline |T| = 0.083623 m
  roi1 640x480+0+0   roi2 640x479+0+0

=== 5. 验证 ===
  极线误差（702 角点）：均值 0.1406 px，P95 0.3536 px，最大 3.6434 px
  三角化：平面性 RMS 0.4173 mm（最差视图 1.6420 mm），
          方格边长 25.0384 mm（+0.154 %），平均深度 0.3228 m

  与参考标定对比（left）:
  fx  536.073  vs  535.916   (+0.0294 %)
  fy  536.016  vs  535.916   (+0.0188 %)
  cx  342.370  vs  342.283   (+0.0255 %)
  cy  235.537  vs  235.571   (-0.0144 %)

结论: 极线误差合格（均值 < 0.5 px）；三角化边长自洽（误差 < 1%，绝对尺度仍需外部确认）
```

`R` 接近单位阵（最大非对角元 4.1e-3），`|T|` 即基线长度，与 `stereo.yml` 里的
`baseline_m` 一致。

与数据集自带的 `left_intrinsics.yml` 对比：

| 参数 | 本工具 | 参考值 | 偏差 |
| --- | --- | --- | --- |
| fx | 536.073 | 535.916 | +0.029 % |
| fy | 536.016 | 535.916 | +0.019 % |
| cx | 342.370 | 342.283 | +0.026 % |
| cy | 235.537 | 235.571 | −0.014 % |
| 单目 RMS | 0.4087 px | 0.3926 px | +4 % |

四项内参的偏差都在 0.03 % 以内，说明实现与 OpenCV 官方样例一致。

### 选项对比（同一数据）

| 选项 | 有效视图 | 单目 RMS (L/R) | 基线 | 极线误差 均值 | 边长误差 |
| --- | --- | --- | --- | --- | --- |
| 默认（经典检测 + `cornerSubPix`） | 13 | 0.409 / 0.459 px | 0.083623 m | 0.141 px | +0.154 % |
| `--refine-intrinsics` | 13 | 同上 | 0.083453 m | 0.127 px | +0.139 % |
| `--alpha 1` | 13 | 同上 | 0.083623 m | 0.115 px | +0.154 % |
| `--sb` | **12** | 0.235 / 0.239 px | 0.082852 m | 0.102 px | −0.110 % |

两点值得注意：

- `--sb` 的 RMS 更低（0.235 vs 0.409 px），但 **fx 偏离参考值 −0.67 %**（532.3 vs 535.9），
  比经典检测差一个数量级。SB 检测的角点是自洽的，却与张正友标定的角点定义有系统偏差；
  在本数据上**经典检测 + `cornerSubPix` 更接近参考结果**。
- `--sb` 在 `05` 上检测失败（一副图像返回 0 角点），有效视图从 13 降到 12。低 RMS 有一部分
  来自少了一个难视图，不能直接横向比较。

## 7. 常见问题

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `fewer than 3 pairs yielded a complete board` | 棋盘格尺寸填错，或图像中靶标不完整 | 用 `--pattern WxH` 按**内角点**数填写；加 `--sb` 试 |
| 极线误差很大（> 1 px） | 内外参不匹配、镜头畸变模型不足、某几幅图角点偏差大 | 看“每视图重投影误差”，把明显偏大的视图从数据里剔除；必要时试用 `CALIB_RATIONAL_MODEL` 重编 |
| 边长误差 > 1 % | 基线与焦距不自洽（裁剪错位、`Q` 修正写反、坐标系不匹配） | 先确认极线误差正常，再查校正环节 |
| 边长误差 ≈ 0 但深度整体偏大/偏小 | `--square-size` 与真实方格不符 | 用卡尺量方格。**图像内的任何检验都发现不了这个**，见 [§6.4](./math-principles.md#64-尺度是不可观测的) |
| 基线明显不对 | 方格尺寸单位错（mm 当成 m） | `--square-size` 单位是**米** |
| `stereoCalibrate` 报 `d <= 2` 断言 | `perViewErrors` 用了 `vector<double>`；它要求 N×2 矩阵 | 已在本实现中处理，见 `StereoModel::perViewErrors` |

## 8. 实现要点

- **两步法**：先各自 `calibrateCamera`，再 `stereoCalibrate` 冻结内参。一次性优化 9+5+9+5+6
  个参数很难收敛；`--refine-intrinsics` 只是在此结果上做联合微调（`CALIB_USE_INTRINSIC_GUESS`）。
- **角点顺序一致性**：两路检测器可能对同一靶标给出相反的起始角（镜像），这会让每一对都违反
  极线约束，最终解出一个 180° 的 `R`。`reverseIfMirrored()` 用网格边方向检测并纠正。
- **`triangulatePoints` 的输出类型跟随输入**：传 `vector<Point2f>` 会得到 `CV_32F` 的 4×N
  矩阵，`at<double>` 读到的是垃圾。本实现显式传 `CV_64F`。
- **平面性必须逐视图统计**：13 个位姿的三维点数放在一起拟合单一平面没有意义。
