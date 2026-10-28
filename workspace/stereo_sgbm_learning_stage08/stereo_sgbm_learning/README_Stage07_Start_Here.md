# Stage 07 — 开始这里

继续沿用 Stage01→06：

- 新增 `stage07_left_right_consistency`，内容为左右一致性检查，Hirschmüller 2008 Section 2.3 Eq.(15)。
- 重点文件：`stage07_left_right_consistency/include/lr_consistency.hpp`。
- 重点注意右图视差正向搜索方向与左图相反，工程通过翻转交换双图实现。
- 测试命令：`cmake -S stage07_left_right_consistency -B stage07_left_right_consistency/build`（若已进入 stage07 目录，则使用 `cmake -S . -B build`）；`cmake --build ...`；`ctest --test-dir ... --output-on-failure`。
- 完整说明：`stage07_left_right_consistency/README.md`。
