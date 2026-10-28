# 第八阶段 Stage 08 - 匹配唯一性与孤立区域过滤

从 `stage08_filters/README.md` 开始。保留了 Stage 01-07 源代码供对照。

核心：`stage08_filters/include/sgm_filters.hpp`。

测试：`stage08_filters/tests/test_sgm_filters.cpp`。

图像运行：`stage08_filters/src/main.cpp`（系统具有 OpenCV 开发库时构建）。

相比 Stage07，新增长代价曲线上的唯一性检查和视差图上的四连通域区域面积筛选。
