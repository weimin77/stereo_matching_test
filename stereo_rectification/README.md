# stereo_rectification

极线校正工具。读取 [stereo_calibration](../stereo_calibration) 产出的标定，把原始图像对转成
**可以直接送入立体匹配算法**的校正图像对，并给出匹配器需要的全部参数。

## 1. 它做什么

```
读标定 → stereoRectify → 裁有效区 → 建 map → 批量 remap → 验证 → 输出
```

| 阶段 | 接口 | 产物 |
| --- | --- | --- |
| 求校正变换 | `cv::stereoRectify` | `R1 R2 P1 P2 Q`，`roi1 roi2` |
| 建映射表 | `cv::initUndistortRectifyMap` | `CV_16SC2` 定点或 `CV_32FC1` 浮点 map |
| 重采样 | `cv::remap` | 校正后的左/右图 |
| 求有效区 | 同一 map 重映射全白图 | 掩膜 |
| 验证 | `findChessboardCorners` + `cornerSubPix`（可选） | 输出像素上的 `\|dy\|` |

### 为什么校正完还不能直接匹配

`stereoRectify` 只保证几何：同一物点在两幅校正图上的行坐标相同。它不管下面这些事，
而漏掉任意一条都会让匹配结果系统性出错。

**焦距变了。** `alpha < 1` 会缩小焦距来保住全部有效像素。OpenCV 样例数据上
`fx` 从 536.07 变成 520.80（−2.85 %）。继续用原始 `fx` 换算深度，整幅深度图就偏 2.9 %。

**四周有黑边。** `remap` 越界的像素是 0，匹配器分不清"黑边"和"真的黑物体"，
会在黑边与图像内容之间匹出大片虚假视差。本工具用同一套 map 重映射一张全白图得到掩膜，
再腐蚀 1 px 去掉插值污染的边界。

**`Q` 是未裁剪坐标系下的。** 裁掉 `(x0, y0)` 之后，`Q` 的主点项必须跟着平移，
否则 `reprojectImageTo3D` 的点云会整体侧移 `x0·Z/f'`——深度值还对，形状却歪了。

**视差窗口要按 `f'·B` 算。** 用原始 `fx` 推 `numDisparities` 会偏，而且 `alpha` 一改就全变。

## 2. 构建

```bash
cd stereo_rectification
cmake -S . -B build
cmake --build build -j
```

默认从 `$HOME/opt/opencv5` 找 OpenCV 5；其他位置用
`-DOpenCV_DIR=<prefix>/lib/cmake/opencv5` 指定。需要 `calib`、`stereo`、`imgproc`、
`imgcodecs`、`objdetect` 五个模块，缺哪个在配置阶段就报出来。

也可以在仓库根目录一起构建（`BUILD_STEREO_RECTIFICATION`，默认 `ON`）。

## 3. 运行

先标定，再校正：

```bash
# 1. 标定，得到 stereo.yml
../stereo_calibration/build/stereo_calib \
  --images /home/min/projects/opencv-samples/samples/data --output calib --no-window

# 2. 校正
./build/stereo_rectify \
  --calibration calib/stereo.yml \
  --images /home/min/projects/opencv-samples/samples/data \
  --depth-range 0.15,1.5 \
  --verify-pattern 9x6 \
  --output rectified
```

单对图像：

```bash
./build/stereo_rectify --calibration calib/stereo.yml \
  --left left01.jpg --right right01.jpg --output rectified
```

## 4. 参数

| 选项 | 默认 | 说明 |
| --- | --- | --- |
| `--calibration <yml>` | 必填 | `stereo.yml`；也可用任何含 `M1 D1 M2 D2` + `R T` 的文件 |
| `--images <dir>` | `.` | 批量模式：配对 `leftNN` / `rightNN` |
| `--left` `--right` | — | 单对模式，替代 `--images` |
| `--left-prefix` / `--right-prefix` | `left` / `right` | 文件名前缀 |
| `--ext` | `.jpg` | 输入扩展名 |
| `--out-ext` | `.png` | 输出扩展名 |
| `--output <dir>` | `output` | 输出目录 |
| `--alpha <0..1>` | `0` | `stereoRectify` 的 balance；0 = 全部像素有效，1 = 保留全部视野 |
| `--new-size <WxH>` | 同输入 | 校正后的图像尺寸 |
| `--no-crop` | 裁 | 保留完整校正画面，不裁共同有效区 |
| `--pad <n>` | `0` | 共同有效区左右各多留 n 列 |
| `--depth-range <near>,<far>` | — | 米；据此推导 `minDisparity` / `numDisparities` |
| `--min-disparity` / `--num-disparities` | — | 直接指定视差窗口，`numDisparities` 向上取 16 的倍数 |
| `--keep-rectification` | 关 | 复用标定里已存的 `R1/R2/P1/P2`（此时 `--alpha` 无效） |
| `--float-maps` | 关 | 用 `CV_32FC1` map。默认 `CV_16SC2` 把映射量化到 1/32 px；实测 OpenCV 5 多线程下 `CV_32FC1` 反而快约 2 倍（0.21 vs 0.44 ms @640×480），只有单线程小图时定点才更快 |
| `--cubic` | 关 | `INTER_CUBIC`；默认为 `INTER_LINEAR` |
| `--verify-pattern <WxH>` | — | 在校正**输出**上重检棋盘格并统计行差 |
| `--verify-sb` | 关 | 验证时用 `findChessboardCornersSB` |
| `--no-mask` | 写 | 不输出有效掩膜 |
| `--no-preview` | 写 | 不输出预览图 |
| `--quiet` | 详细 | 只打印阶段汇总 |

`--depth-range` 与 `--num-disparities` 互斥。

## 5. 输出

```
left/leftNN.png          校正后的左图
right/rightNN.png        校正后的右图，尺寸与左图一致
mask/leftNN.png          共同有效区，255 有效（重采样后再腐蚀 1 px）
camera.json              fx / baseline_m，可直接作为 test.cpp 的 camera_file
rectification.yml        R1 R2 P1 P2 Q（已按裁剪修正）+ 全部几何量
preview_left01.png       左右拼接 + 水平参考线 + 无效区紫色着色
rectification_report.md  Markdown 汇总
```

`camera.json`：

```json
{
    "fx": 520.7956488,
    "baseline_m": 0.08362318842,
    "fy": 520.7956488,
    "cx": 350.6118279,
    "cy": 243.0536995,
    "doffs": 0,
    "focal_baseline_px_m": 43.55059267,
    "image_width": 640,
    "image_height": 479,
    "min_disparity": 16,
    "num_disparities": 288
}
```

[test.cpp](../test.cpp) 的 `applyCameraFile()` 只读顶层 `fx` 和 `baseline_m`，其余字段被忽略，
所以这个文件可以直接用。注意 `fx` 是**校正后**的焦距，不是原始内参。

## 6. 在 OpenCV 样例数据上的结果

标定来自 `stereo_calibration`（13 对，640×480，9×6 角点，方格 25 mm）。

| 配置 | 输出尺寸 | 平均有效 | 检出视图 | \|dy\| 均值 | \|dy\| P95 | \|dy\| 最大 |
| --- | --- | --- | --- | --- | --- | --- |
| 默认（`alpha 0`） | 640×479 | 100.00 % | 13/13 | 0.1291 px | 0.3379 px | 1.6815 px |
| `--alpha 0.5` | 615×435 | 94.62 % | 13/13 | 0.1229 px | 0.3224 px | 1.7782 px |
| `--alpha 1` | 552×392 | 82.52 % | 5/13 | 0.0973 px | 0.3045 px | 0.6265 px |
| `--no-crop` | 640×480 | 99.99 % | 13/13 | 0.1291 px | 0.3379 px | 1.6815 px |

几何量（默认配置）：

```
fx' 520.7956  fy' 520.7956  cx1' 350.6118  cy1' 243.0537
doffs 0.0000 px   基线 0.083623 m   f'·B 43.5506 px·m
视差 29.03 … 290.34 px  ->  minDisparity 16, numDisparities 288
```

与 [stereo_calibration](../stereo_calibration) 报告的对齐：

- 那边用 `undistortPoints` 做纯数学投影，得到 `|dy|` 均值 0.1406 px / P95 0.3536 px；
- 这边在校正**输出像素**上重新检测角点，得到 0.1291 / 0.3379 px。

两者一致，说明 `remap` 引入的插值损失在这个数据集上小于角点检测本身的抖动。

几点值得注意：

- `alpha 0` 是唯一让有效像素达到 100 % 的配置，代价是焦距变长、视野变窄（输出 640×479）。
  这与直觉相反：`alpha` 越小视野越小，但黑边越少。做稠密匹配时优先选它。
- `alpha 1` 只有 5/13 个视图能检出棋盘。焦距缩到 425.85，视野变大，棋盘在画面中占比变小，
  还带黑边，检测器开始漏检。这提示 `alpha 1` 的校正结果对下游任务并不友好。
- `--verify-pattern` 的 `\|dy\|` 统计是**全局合并**的（所有视图所有角点一起排序），
  与标定报告的口径一致，不是"各视图均值的平均"。

## 7. 与 test.cpp 衔接

```bash
./build/algorithm_test sgbm_config.json
```

需要三步：

1. `sgbm_config.json` 的 `camera_file` 指向本工具产出的 `camera.json`；
2. 输入左右图换成 `rectified/left/` 和 `rectified/right/`；
3. 视差窗口用 `camera.json` 的 `min_disparity` / `num_disparities`，或用 `--depth-range` 让
   本工具算好。

**不要**把原始图像直接喂给 `test.cpp` 再让它读 `camera.json`。`test.cpp` 内部按"已校正、
无畸变"的假设建立视差-深度关系，未经 `remap` 的图像会让这个假设失效。

反过来说，如果图像来自 BlenderProc 仿真（同一 rotation、只沿 x 平移、无畸变），
校正就是恒等变换，`remap` 可以整个跳过，此时只需要 `fx` 和 `baseline_m`。

## 8. 常见问题

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `有效像素` 很低（< 60 %） | `alpha` 太小导致视野被裁得很窄，或标定质量差 | 提高 `--alpha`；看 `preview_*.png` 的紫色区域分布 |
| `--verify-pattern` 检出视图数少 | 棋盘在校正后画面中太小或落进黑边 | 提高 `--alpha`（视野内的黑边会减少），或降低 `--pad` |
| `\|dy\|` 均值 > 1 px | 标定外参有问题，或图像在采集时相机被碰过 | 回到 `stereo_calibration` 看极线误差与每视图 RMS |
| `calibration has no R/T` | 用了只含 `R1/P1` 的文件 | 加 `--keep-rectification`，或改用完整 `stereo.yml` |
| `--new-size` 配 `--keep-rectification` 报错 | 已存的 `P1/P2` 绑定标定时的图像尺寸 | 二选一 |
| 视差为负，匹配器搜不到 | `T` 的 x 分量为正，即第二台相机在 +x 侧 | 交换 `--left` / `--right` 重新标定；工具会打印警告 |
| 输出图像尺寸与 `test.cpp` 期望不符 | 裁剪改变了尺寸 | `--no-crop`，或同步修改配置里的图像尺寸 |

`--verify-pattern` 在 `alpha 1` 上只检出 5 对，属于正常的检测失败，不是程序错误。

## 9. 实现要点

- **map 先生成后切片。** `initUndistortRectifyMap` 按未裁剪尺寸建表，再用 `map(roi).clone()`
  取出子块。这样目标坐标已经带上裁剪偏移，`remap` 直接写到裁剪后的尺寸，不需要额外平移。
- **`Q` 的修正方向容易搞反。** `Q(0,3) = -cx1'`，裁剪后 `cx1'' = cx1' - x0`，
  所以 `Q(0,3)` 要 **加** `x0`。写反时深度值仍然正确，只有点云横向平移，很难靠肉眼发现。
  工具内置 `checkQAgainstP()`：把一个 3D 点经 `P1/P2` 投影、再用 `Q` 反投影，
  与原始坐标比对，实测误差 `~1e-16 m`。
- **`stereoRectify` 没有 `fov_scale`。** 那个参数只存在于 `cv::fisheye::stereoRectify`，
  pinhole 版本用 `alpha` 和 `newImageSize` 控制视野。
- **掩膜要腐蚀。** `remap` 在边界做双线性插值时会把常量边界色混进来，不腐蚀会在有效区边缘
  留一圈半黑的像素。
- **角点顺序会镜像。** 两个视角的棋盘检测可能从对角开始，直接按索引配对会得到上百像素的假
  `\|dy\|`。`matchRowErrors()` 同时试正序和逆序，取均值较小的那个。
- **`--keep-rectification` 不读 `roi`。** `stereo_calibration` 没有保存 `roi1`/`roi2`，
  所以这条路径不裁剪。需要裁剪就用默认路径重算。

## 10. 耗时

640×480 单张（OpenCV 5.0，16 线程，中位数）：

| 操作 | 耗时 |
| --- | --- |
| `stereoRectify`（一次性） | 0.20 ms |
| `initUndistortRectifyMap` ×2（一次性） | 1.8 ms |
| `remap` 单张彩色，`CV_32FC1` + `INTER_LINEAR` | 0.19 ms |
| `remap` 单张彩色，`CV_16SC2` + `INTER_LINEAR` | 0.42 ms |
| `remap` 单张彩色，`CV_16SC2` + `INTER_CUBIC` | 0.61 ms |
| 有效掩膜（一对图像） | 0.46 ms |
| 写 PNG（彩色） | 7.8 ms |
| 写 JPEG（彩色） | 1.1 ms |
| 写 BMP（彩色） | 0.8 ms |

几何校正本身是亚毫秒级；**端到端耗时由图像编码决定**。本工具处理一对图像
（`remap` ×2 + 掩膜 + 写 PNG ×2）实测 17.7 ms，其中 PNG 编码占 15.5 ms。
换成 `--out-ext .bmp` 后降到 3.8 ms。

作为参照，`test.cpp` 在同一对图像上的 SGBM 匹配是 43.4 ms/对（Release）。校正的
几何部分约占 1–2 %，但若把 PNG 落盘算进去就变成 40 %，所以批处理时应优先避免
中间文件，或改用无损但更快的容器。
