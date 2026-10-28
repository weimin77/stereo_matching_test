# Stage05 · 八方向 SGM 学习工程

在 Stage01～04 原有代码之上新增 `stage05_eight_directions/`。

建议从以下文件依次阅读：

1. `stage05_eight_directions/README.md`：算法概念、对角线起点与运行方式。
2. `stage05_eight_directions/include/sgm_eight_directions.hpp`：核心实现。
3. `stage05_eight_directions/tests/test_sgm_eight_directions.cpp`：验证对角路径、与 Stage04 对齐及 WTA。
4. `stage05_eight_directions/src/main.cpp`：OpenCV 图像演示。

编译/测试：

```bash
cd stereo_sgbm_learning/stage05_eight_directions
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
ctest --test-dir build --output-on-failure
```
