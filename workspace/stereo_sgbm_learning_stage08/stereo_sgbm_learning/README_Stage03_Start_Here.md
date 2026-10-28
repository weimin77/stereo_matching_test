# 从零实现 SGBM：Stage 01 → Stage 03

按顺序学习：

1. `stage01_bm`：SAD 块匹配 + WTA（基线算法）。
2. `stage02_cost_volume`：全图匹配代价体 C(x,y,d) + WTA。
3. `stage03_single_path`：SGM 单路径动态规划公式 (13)，输出左→右路径代价 L(x,y,d)。

本包的源代码保留了 Stage 01 和 Stage 02；Stage 03 通过 `#include "cost_volume.hpp"` 复用数据结构。

直接开始：

```bash
cd stereo_sgbm_learning/stage03_single_path
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

查看本阶段详细讲义：`stage03_single_path/README.md`。

注意：OpenCV 仅用于 demo 的图像读写，单元测试和核心 SGM 均不需要 OpenCV。
