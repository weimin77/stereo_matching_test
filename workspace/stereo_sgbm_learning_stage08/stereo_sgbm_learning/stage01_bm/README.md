# 从零实现 StereoSGBM：阶段 1 / 2，手写 SAD-BM

目标：理解并亲手实现极线约束、代价计算、WTA。暂不实现 SGM 路径聚合。

## 安装与构建（Ubuntu）

```bash
sudo apt update
sudo apt install -y build-essential cmake libopencv-dev
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
```

## 用你已有的 BlenderProc 输出运行

```bash
./build/handmade_bm /ABS/PATH/output/left.png /ABS/PATH/output/right.png ./out 48 80 5 \
 /ABS/PATH/output/disparity_gt_left_16.png /ABS/PATH/output/valid_stereo_mask.png
```

参数：左图、右图、结果目录、最小视差、视差候选数、奇数窗口大小、可选GT 16位 PNG、可选GT掩码。

注意 BlenderProc 示例中 GT 为视差乘 256；并非 OpenCV StereoSGBM 内部的视差乘 16。

结果：`out/bm_disparity_16.png`, `out/bm_valid.png`, `out/bm_vis.png`，以及终端打印 EPE、RMSE、Bad1、Coverage。EPE 仅计算预测/GT 均有效的交集，需同时参考 Coverage。

## 无需 OpenCV 的逻辑单元测试

```bash
g++ -std=c++17 -O2 -Iinclude tests/test_bm.cpp -o /tmp/test_bm
/tmp/test_bm
```

该测试构造人为视差=7px的随机纹理，要求内部所有被测像素输出7。

## 算法说明

左图像素 (x,y) 假设视差 d 时，对应右图像素 (x-d,y)。

SAD cost = sum_{i,j in window} |L(y+i,x+j)-R(y+i,x-d+j)|。

d_best = argmin_d SAD(x,y,d)。

匹配窗口和视差范围越大，当前朴素实现越慢：O(W*H*D*blockSize^2)。下一阶段学习将其改为 O(W*H*D) 的滑动窗口，同时引入 Cost Volume，为 SGM 做准备。

## 自己思考的四个问题

1. 为什么像素必须是在经过极线校正的图片中，才适用 `(x-d,y)`?
2. `minDisp=48,numDisp=80` 的候选区间是哪些整数?
3. 为什么右边界遮挡、左边界搜索越界时会产生无效像素?
4. 为什么纹理均匀的食材会出现错误视差，即使 SAD 是正确计算的?

## 下一阶段

- 先实现按视差生成每张匹配代价图：`C(x,y,d)`。
- 使用二次滑动窗口或积分图计算 SAD。
- 实现单条 SGM 路径递推，再增加 4/8 路与亚像素优化。
- 最后按照 OpenCV 4.x stereosgbm.cpp 的 BT 成本、五路径和后处理调整。
