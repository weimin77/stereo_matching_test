# Stereo SGBM 从零学习工程：Stage 01–04

- stage01_bm: SAD 块匹配 + WTA。
- stage02_cost_volume: 全代价体 [H][W][D]，朴素和积分图 SAD。
- stage03_single_path: Hirschmueller (2008) 公式 (13)，左到右归一化动态规划。
- stage04_four_directions: 公式 (14)，四方向 SGM 代价聚合、WTA、单元测试、OpenCV 图像演示。

入门：阅读 `stage04_four_directions/README.md`；先运行该目录内的 CMake 和 ctest。

提醒：四方向 SGM 是学习中的中间版本，不等同 OpenCV 的 StereoSGBM。
