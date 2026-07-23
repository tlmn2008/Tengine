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

- **测试全量（84 例）**：`partial_pass` —— tests_run=84，passed=9，failed=0，skipped=75。分两部分：

  1. **PART A — 完整 ctest 集合：75 个 ONNX 算子测试（`tests/op/test_onnx_op_*.cpp`）**。这些用 `create_graph(nullptr, "tengine", model)` 跑在 **CPU 参考后端**，**完全不经过 CUDA 后端**（测试源码硬编码默认 CPU context，无 env/flag 可切到 CUDA，除非改源码），属 CUDA 迁移范围之外。原始 `ctest --output-on-failure -V`（退出码 8）判定 **75/75 Failed**，失败原因 100% 相同：`cannot open file ../onnx_node/<op>/onnx.tmfile → Create graph failed`，在**加载模型阶段即失败、从未进入推理**。

     **v2 复核（针对“onnx 已系统安装，缺数据不再算合法跳过”的跟进）**：我按 Failure Gate 实测求证能否用已装 onnx 生成/定位这些数据，结论是**当前环境 onnx 事实上并未安装、数据无法生成或定位**：
       - 所有系统解释器 `import onnx` 均 `ModuleNotFoundError`，`from onnx import helper` 亦 `ImportError`；`pip`/`dpkg` 均无 onnx 包。
       - 全机唯一 onnx 家族包是 `onnxruntime-gpu`（且因缺 `libtvm.so` 连自身都 import 失败），它**既不含** ONNX 节点一致性测试数据（`onnx/backend/test/data/node/test_*`，即本测试所需的 `model.onnx`+`input/output.pb` 来源），**也不含**用于构造模型的 `onnx.helper` API。
       - 全盘 `find` 未见任何 `onnx_node` 目录或节点级 `model.onnx`（仅有 Tengine 自带 `tools/align_tool/mnist.onnx` 与 onnxruntime 的 3 个 demo 模型 logreg_iris/mul_1/sigmoid，均非本测试所需）。
       - 生成 `onnx.tmfile` 需先有 `model.onnx`（要 `onnx.helper` 构造，缺）并用 Tengine convert_tool 转换；参考张量也需 onnx。这些都要求安装/vendor onnx——**被明令禁止**。
     
     故这 75 例的 `../onnx_node/<op>/{onnx.tmfile,test_data_set_0/*.pb}` 无法生成或定位，仍归类为 **skipped(legitimate)**，但给出了**精确原因**（不再是笼统“缺数据”）。证据见 `test/test.log` 的 PROBE 段与 PART A；原始 75 Failed 完整保留，绝未从总数剔除，也未计为通过。若日后真正安装 onnx（含节点测试数据/`onnx.helper`），应转换生成数据后重跑，将这 75 例改判为真实 pass/fail。

  2. **PART B — CUDA 后端 on-GPU 证据：9 个 ImageNet 类分类基准模型的 CUDA↔CPU top-5 数值对齐**。在 GPU 1 上真实运行，**9/9 PASS**（本次重跑延迟：squeezenet 1.71 / mobilenet 1.52 / mobilenet_v2 2.79 / googlenet 20.91 / resnet18 1.97 / resnet50 4.02 / shufflenet_v2 4.03 / inception_v3 5.44 / vgg16 3.96 ms）。这是唯一真正**行使 CUDA 后端**的部分。说明：`*_benchmark.tmfile` 仅含结构无权重，CPU/CUDA top-5 皆恒定值，断言的是 **CUDA==CPU 对齐 + GPU 无错退出**。

- **诚实结论**：CUDA 后端本身（编译 + on-GPU 运行 + 与 CPU 对齐）已验证通过（9/9）；但仓库注册的 ctest 主体（75 个 CPU 参考算子测试）因缺外部数据无法执行、未被验证。因此整体判定为 **partial / partial_pass**（而非 migrated），以免像上一版那样高估。计数、run_command、范围说明见 `test/test_summary.json`；权威逐例日志见 `test/test.log`。

## Failure Gate
- 每个 wall 均先真实复现再修：baseline configure 失败（`build/baseline_configure.log`）、softmax 编译失败（`build/compile.log`）均有实测错误证据；本次 75 例 ctest 失败也已实跑复现（`test/test.log`），确认为缺数据、非 CoreX/CUDA 缺陷。
- blocker 分类见 `blockers.json`：3 个与 CUDA 直接相关的均为 workaround-able 且已解决；#4（OpenCV/pipeline 示例链接/头文件卫生）与 CUDA 后端无关，超出范围；#5（本次新增，记录 75 个 CPU 参考 ONNX 算子测试缺外部 onnx_node 数据，属合法跳过/超范围）。
- 无 terminal blocker。
