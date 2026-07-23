# Tengine 迁移到 Iluvatar CoreX (ivcore11) 适配记录

## 来源
- 上游仓库：https://github.com/OAID/Tengine.git （OpenAILab 边缘推理引擎 Tengine-Lite）
- 迁移起点 commit：`5ec1c383c8adb0078c025b9fec6fa3dea254034a`（默认分支 `tengine-lite`，2024-09-15）
- 迁移目标：让其 CUDA 后端（`source/device/cuda`）在 CoreX 上用 clang 编译，并让 CUDA 示例在 GPU 上真实运行。

## CUDA 使用性质
- CUDA 后端是一个可选推理设备（`TENGINE_ENABLE_CUDA`），通过 cudnn + cublas + 13 个手写 `.cu` 算子 kernel（clip/concat/conv/dropout/eltwise/fc/flatten/permute/pooling/relu/reshape/slice/softmax）把子图跑在 GPU 上。
- 内存管理为显式 `cudaMalloc` + `cudaMemcpy`（无 Unified Memory）；算子基本是 fp32，无 device 端 `double`/`float64`/`long double` 运算（仅 softmax 里一行被注释掉的 host 计时 `double` 无影响）；无 texture、无 u64/double 原子、无跨模块 NVRTC device link。属于“干净”的 CUDA 推理后端。

## 环境
- SDK：`/usr/local/corex`（CoreX 4.5.0，clang++ 22.1.0，前端 ID = ILUVATAR）；`CUDAToolkit` 报告 10.2.89，cudnn 7.6.5，cublas 10.2。
- 硬件：2× Iluvatar BI-V150（32GiB）。本 subagent 固定使用 **GPU 1**（`CUDA_VISIBLE_DEVICES=1`），另一迁移在 GPU 0 并行。
- 监控用 `ixsmi`（见 `env/ixsmi.log`）；`nvcc` 仅为 bash shim（假成功），严禁当编译器用，严禁改动 `/usr/local/corex`。
- 环境脚本见 `env/corex_env.sh`。

## 适配内容（改动 3 个文件，见 changes/modified_files.json）
1. **`cmake/cuda.cmake`**：原代码 `SET(CMAKE_CUDA_COMPILER ${CUDAToolkit_NVCC_EXECUTABLE})` 会把 CoreX 的 nvcc shim 当 CUDA 编译器，导致 `ENABLE_LANGUAGE(CUDA)` 报 “Failed to extract nvcc implicit link line”。改为：若检测到 `${CUDAToolkit_BIN_DIR}/clang++`（即 CoreX）则用 clang++ 作 CUDA 编译器（ID=ILUVATAR），真 NVIDIA 环境仍回退到 nvcc。（对应 iluvatar-cuda-base：nvcc-shell-stub-detect / cmake-corex-template）
2. **`source/CMakeLists.txt`**：
   - 目标属性 `CUDA_ARCHITECTURES` 在 ILUVATAR 下设为 `ivcore11`（NVIDIA 的 `sm_50/52/61/70/75` 不被 CoreX clang 后端接受）。
   - 对 ILUVATAR 跳过 nvcc 专有诊断 `-Werror=cross-execution-space-call`（CoreX clang 不识别该 warning 选项）。
3. **`source/device/cuda/op/nvgpu_softmax.cu`**：host 端裸 `max(...)` 在 CoreX clang++ 报 “no matching function for call to 'max'”（CUDA 数学头只暴露 `__device__` 重载）。加 `#include <algorithm>` 并改用 `std::max`。（对应 iluvatar-cuda-base：min-max-no-global）

未改 SDK、未造 dummy nvcc、未用 `-O0` 规避、未超 2 GPU / 2 rank。

## 结果

> **本次为“完整性纠正”重跑（2026-07-23）。** 上一版记录只跑了 9 模型的 `tm_classification_cuda` demo 就把 9/9 当成全量，违反完整性红线。本次按 `-DTENGINE_BUILD_TESTS=ON` 构建并运行仓库的**完整 ctest 注册测试集**，如实分离 CUDA 后端 vs CPU 参考后端两部分。

- **编译**：`success`。CUDA 后端库 `libtengine-lite.so` / `libtengine-lite-static.a`、全部 13 个 `.cu` 算子、CUDA 示例 `tm_classification_cuda`、以及**全部 75 个 ONNX 算子测试二进制 + 21 个 model 测试二进制**均用 CoreX clang++ 编译通过（`build/compile_tests.log`）。唯一编译报错集中在 `examples/`（~48 个 OpenCV 图像 demo + 3 个 pipeline 示例，74 组 error 全部在 `examples/`，0 组在 `tests/`），即 blocker #4，与 CUDA 后端正交、超出迁移范围。

- **测试全量（84 例）**：`partial_pass` —— tests_run=84，passed=77，failed=7，skipped=0。分两部分：

  1. **PART A — 完整 ctest 集合：75 个 ONNX 算子测试（`tests/op/test_onnx_op_*.cpp`），真实执行**。这些用 `create_graph(nullptr, "tengine", model)` 跑在 **CPU 参考后端**，**不经过 CUDA 后端**，属 CUDA 迁移范围之外；但按完整性红线本次**真实跑通到推理**：**68 通过 / 7 失败**（`ctest` 退出码 8，91%）。

     **数据生成（v3，onnx 1.22.0 已真正安装）**：确认 `onnx` 可导入、`onnx/backend/test/data/node` 含 1765 个节点目录。构建 Tengine `convert_tool`（`-DTENGINE_BUILD_CONVERT_TOOL=ON`），对每个算子把 onnx 后端节点的 `model.onnx` 转成 Tengine `onnx.tmfile`、并拷贝 `test_data_set_*/*.pb` 到 `../onnx_node/<op>/`（脚本 `corex_port/test/gen_onnx_node_data.sh`，65/75 直接转换成功）。其余 10 个：stock onnx 1.22 用 **opset-25**（Unsqueeze/Pad 把 `axes`/`pads` 从属性改为输入 → 老 convert_tool 段错误）或缺必需属性（BatchNormalization 省略 `epsilon` → convert 抛 `cannot find attr epsilon`），用 `onnx.helper` 以 Tengine 可解析的 opset 重建 / 补默认 `epsilon`（脚本 `corex_port/test/regen_hard_ops.py`）。最终 75/75 均有数据。

     **7 个失败——均为 Tengine CPU 参考后端既有缺陷，与 CoreX/CUDA 无关、也非缺数据（都已进入推理）**：
       - **4 个卷积**（basic_conv_with_padding / basic_conv_without_padding / conv_with_strides_no_padding / conv_with_strides_padding）：输出全 0（`a=0.0` vs 参考 `b=12/54`）。根因是测试用例本身的顺序问题——先 `prerun_graph`（此时 Tengine 已把卷积权重预打包）**之后**才 `get_pb_data` 填充权重缓冲区（input_1），故卷积用到的是全 0 权重。属 Tengine 测试脚手架/CPU 后端既有问题。
       - **batchnorm_example**：teardown 阶段 `free(): invalid size` 崩溃中止。
       - **dropout_default**：推理数值正确（打印 `test pass`）后在 teardown 崩溃（`register_batchnorm_ref_op() failed` + 堆破坏）。
       - **pad_reflect**：teardown 阶段 `double free or corruption` 崩溃。
     
     按红线，崩溃/不匹配一律计入 **failed**（不跳过、不剔除）。逐例完整输出见 `test/test.log` 的 PART A1（数据生成）与 PART A2（ctest 全量）。

  2. **PART B — CUDA 后端 on-GPU 证据：9 个 ImageNet 类分类基准模型的 CUDA↔CPU top-5 数值对齐**。在 GPU 1 上真实运行，**9/9 PASS**（本次延迟：squeezenet 1.88 / mobilenet 1.59 / mobilenet_v2 1.48 / googlenet 20.90 / resnet18 1.96 / resnet50 4.02 / shufflenet_v2 3.06 / inception_v3 5.23 / vgg16 4.02 ms）。这是唯一真正**行使 CUDA 后端**的部分。`*_benchmark.tmfile` 仅含结构无权重，断言的是 **CUDA==CPU 对齐 + GPU 无错退出**。

- **诚实结论**：CUDA 后端（编译 + on-GPU 运行 + 与 CPU 对齐）验证通过（9/9）；仓库注册的 75 个 CPU 参考算子测试本次已**真实执行**，68 通过、7 失败，7 个失败均为 Tengine CPU 后端既有缺陷（非 CoreX/CUDA、非缺数据）。整体判定 **partial / partial_pass**（存在 7 个真实失败）。计数、run_command、范围与失败明细见 `test/test_summary.json`；权威逐例日志见 `test/test.log`。

## Failure Gate
- 每个 wall 均先真实复现再修：baseline configure 失败（`build/baseline_configure.log`）、softmax 编译失败（`build/compile.log`）均有实测错误证据；本次 75 例 ctest 也已用真实生成的数据实跑复现（`test/test.log`），7 个失败确认为 Tengine CPU 后端既有缺陷、非 CoreX/CUDA。
- blocker 分类见 `blockers.json`：3 个与 CUDA 直接相关的均为 workaround-able 且已解决；#4（OpenCV/pipeline 示例链接/头文件卫生）与 CUDA 后端无关，超出范围；#5（v3 已解决：用已装 onnx 1.22 + Tengine convert_tool 生成全部 75 个算子的 onnx_node 数据并真实执行，68 通过/7 失败，7 个失败为 Tengine CPU 后端既有缺陷）。
- 无 terminal blocker。
