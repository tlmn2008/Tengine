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
- **编译**：`success`。CUDA 后端库 `libtengine-lite.so` / `libtengine-lite-static.a` 及全部 13 个 `.cu` 算子、CUDA 示例 `tm_classification_cuda` 均用 CoreX clang 编译通过。
- **测试（GPU 1 上真实运行）**：`all_pass`，9/9。用 CUDA 示例在 GPU 上跑 9 个 ImageNet 类分类基准模型（squeezenet_v1.1 / mobilenet / mobilenet_v2 / googlenet / resnet18 / resnet50 / shufflenet_v2 / inception_v3 / vgg16），并与 CPU 参考后端做 top-5 数值对齐（parity）——9 个模型 CUDA 与 CPU 输出完全一致，且 GPU 明显加速（如 vgg16：GPU 4.21ms vs CPU 485ms，约 115×）。
  - 说明：`*_benchmark.tmfile` 是仅含结构、无训练权重的测速模型，故 CPU 与 CUDA 的 top-5 都是恒定值（如 0.0）；测试断言的是 **CUDA==CPU 数值对齐 + 在 GPU 上无错退出**，而非具体类别标签。
  - 完整逐用例日志见 `test/test.log`，指标见 `test/test_summary.json`。

## Failure Gate
- 每个 wall 均先真实复现再修：baseline configure 失败（`build/baseline_configure.log`）、softmax 编译失败（`build/compile.log`）均有实测错误证据。
- blocker 分类见 `blockers.json`：3 个与 CUDA 直接相关的均为 workaround-able 且已解决；第 4 个（`cmake --build` 全量还会去编 ~48 个 OpenCV 图像 demo 示例 + 3 个 pipeline 示例，链接失败）经诊断为 OpenCV 示例链接/头文件卫生问题，**与 CUDA 后端无关**，已确认 OpenCV 4.6.0 及相关符号存在，属正交问题，迁移面按 CUDA 后端目标裁定，不纳入 CUDA 迁移范围。
- 无 terminal blocker。
