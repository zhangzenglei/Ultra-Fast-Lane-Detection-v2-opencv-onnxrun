# UFLDv2 C++ / ONNX Runtime GPU

适配 Ubuntu 22.04，复用 hpc203 `onnxruntime/main.cpp` 的 CULane 预处理、anchor、双分支关键点解码和原图绘点。使用 C++17 和 ONNX Runtime CUDA EP；无 Python 推理依赖。2026-11-16 模型下载完成后，CULane Res18 已通过真实 CPU/CUDA 单图、批处理及 profiling 验证，见 [最新验证记录](CPP_GPU_VALIDATION_20261116.md)。文末保留 11 月 14 日初次适配时的历史验收记录。

## 本地依赖与构建

本次实际使用 GCC 11.4、CMake 3.22.1、OpenCV 4.12.0、已安装的 header-only `nlohmann/json.hpp`，以及以下现有目录：

```bash
export ORT_ROOT=/home/zenglei/opt/onnx-gpu/onnxruntime-linux-x64-gpu-1.15.1
export CUDNN_ROOT=/home/zenglei/opt/onnx-gpu/cudnn-linux-x86_64-8.7.0.84_cuda11-archive
rtk cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DORT_ROOT="$ORT_ROOT"
rtk cmake --build build -j4
rtk proxy ./scripts/run_cpp_gpu.sh --help
```

CMake 只启用 CXX，不调用 nvcc，不直接链接 CUDA Provider；ORT 在运行时加载 Provider。构建 RPATH 包含 ORT `lib`。`ORT_ROOT` 同时支持环境变量和 CMake cache（cache 优先）；更换 SDK 后需重新配置构建。没有安装、下载、覆盖 SDK 或修改系统配置。

`scripts/run_cpp_gpu.sh` 仅为子进程添加 ORT/cuDNN 搜索路径；可选 `CUDA_ROOT=/path/to/cuda` 添加该目录的 `lib64` 和 `targets/x86_64-linux/lib`，默认使用系统已有 CUDA runtime。支持 `UFLDV2_BINARY=/path/to/binary` 覆盖可执行文件。脚本不切换工作目录，模型、图片和输出的相对路径均相对于启动时的工作目录。

## 模型准备

模型只能使用项目提供的 UFLDv2 权重。上游 [README](README.md) 提供百度网盘：<https://pan.baidu.com/s/1b68-q_NX2PQPHZQn2h-x2A>，提取码 `jfwn`。本机 `model/weights/` 已有下载完成的四个模型，不需重复下载。

默认建议指定 `--model model/weights/ufldv2_culane_res18_320x1600.onnx`。`*.onnx.baiduyun.p.downloading` 若仍存在则不能作为模型使用，也不应仅改扩展名。不要修改权重内容。CULane Res34 尚未测试；当前程序不支持 TuSimple。

首版支持 CULane 静态 float32 模型：输入 `[1,3,320,1600]`，四个输出分别为 `[1,200,72,4]`、`[1,100,81,4]`、`[1,2,72,4]`、`[1,2,81,4]`。按唯一输出 shape 绑定各分支，允许导出器改名/重排输出。动态维度、TuSimple 和不匹配的输出会明确报错，不猜测模型配置。

## 运行命令（CULane Res18 已通过真实推理验证）

在项目根目录执行，按下载完成的真实文件位置调整 `--model`。

```bash
# 环境诊断；驱动不可用、缺库或缺模型时返回非零
rtk proxy ./scripts/check_gpu_env.sh model/weights/ufldv2_culane_res18_320x1600.onnx

# CPU 对照：仓库已有 CULane JPG
rtk proxy ./scripts/run_cpp_gpu.sh \
  --model model/weights/ufldv2_culane_res18_320x1600.onnx \
  --image onnxruntime/images/culane/00000.jpg \
  --output outputs/cpu/00000_result.jpg --output-dir outputs/cpu \
  --device cpu --save-points --benchmark --warmup 10 --iterations 100

# CUDA：同模型同图片，并保存独立 profiling
rtk proxy ./scripts/run_cpp_gpu.sh \
  --model model/weights/ufldv2_culane_res18_320x1600.onnx \
  --image onnxruntime/images/culane/00000.jpg \
  --output outputs/cuda/00000_result.jpg --output-dir outputs/cuda \
  --device cuda --gpu-id 0 --save-points --benchmark --warmup 10 --iterations 100

# 批处理：非递归扫描 JPG/JPEG/PNG，扩展名不区分大小写
rtk proxy ./scripts/run_cpp_gpu.sh \
  --model model/weights/ufldv2_culane_res18_320x1600.onnx \
  --image-dir onnxruntime/images/culane \
  --output-dir outputs/batch --device cuda --save-points
```

本地道路 JPG/PNG 可直接替换 `--image`，无需预先缩放。`--image` 和 `--image-dir` 必须二选一。单图默认输出 `outputs/<stem>_result.jpg`，JSON 与结果图同名。批量保留输入完整文件名作为输出 stem，例如 `road.jpg_result.jpg` 和 `road.png_result.jpg`，避免不同扩展名同 stem 的冲突。已有目标文件一律拒绝覆盖；重复运行请换输出目录/文件名。坏图记录失败并继续，其余图片处理后汇报 success/failed，出现失败返回非零；ORT 执行错误和 profiling 审计失败终止会话。

大量图片可集中写入一个输出文件夹。以下为本地 v6453 数据目录的实际运行命令：

```bash
rtk proxy ./scripts/run_cpp_gpu.sh \
  --model model/weights/ufldv2_culane_res18_320x1600.onnx \
  --image-dir /home/zenglei/Data/v6453/mcapWriter1_record_20260916-102329/sensor_camera_front_mono_image_encode \
  --output-dir outputs/v6453_front_20260916_102329 \
  --device cuda --save-points --skip-existing
```

`--skip-existing` 必须与 `--save-points` 一起使用，显式恢复中断的任务。只有结果图与 JSON 同时存在、图片非空，且 JSON 中的源图片、模型路径及设备匹配时才跳过；不完整或不匹配结果会报错，避免覆盖。最终汇报新成功、失败和跳过数量。默认不加此参数时仍拒绝已有输出。启动前的防覆盖检查已缓存规范化输入路径，避免上万张图片时重复遍历全部输入。

该目录 9,033 张 3840×2160 JPG 已于 2026-11-16 全部完成 CUDA 处理，失败 0，输出 9,033 张结果图和 9,033 个关键点 JSON。交互运行会话曾在完成 1,592 张后中断，独立进程使用上述恢复参数完成剩余 7,441 张，最终日志为 `Summary: success=7441 failed=0 skipped=1592`。所有结果文件与原始输入已逐一核对，坐标均在原图范围内；4004 张未检测到车道线，保存空列表，其余 5029 张有检测点。未检出不属于运行错误，也不能将处理成功视作准确性保证。完整清单统计和运行日志分别在输出目录的 `summary.json`、`run.log` 中。输出含两段成功 CUDA 会话的 profiling；原始图片没有改动。

## 手动使用 Res34 重新处理 v6453 图库

本节提供手动操作步骤。使用 `ufldv2_culane_res34_320x1600.onnx` 处理同一图库，将新结果保存到 `outputs/v6453_front_20260916_102329_res34/`；原 Res18 目录 `outputs/v6453_front_20260916_102329/` 保留。Res34 的实际兼容性、检测效果及耗时以以下命令的运行结果为准。

### 1. 进入项目，检查 GPU 和模型

在能访问 GPU 的本机 Bash 或 Zsh 终端执行：

```bash
cd /home/zenglei/Ultra-Fast-Lane-Detection-v2-opencv-onnxrun

rtk proxy nvidia-smi
rtk proxy ls -lh model/weights/ufldv2_culane_res34_320x1600.onnx
```

确认能看到 RTX 4050 和完整模型文件后继续。现有工程通过 `--model` 切换权重，无需为切换模型修改代码或重新安装 SDK。

### 2. 单图验证并测量推理性能

```bash
rtk proxy ./scripts/run_cpp_gpu.sh \
  --model model/weights/ufldv2_culane_res34_320x1600.onnx \
  --image /home/zenglei/Data/v6453/mcapWriter1_record_20260916-102329/sensor_camera_front_mono_image_encode/sensor_camera_front_mono_image_encode_1789525410500109.jpg \
  --output outputs/res34_check/result.jpg \
  --output-dir outputs/res34_check \
  --device cuda --gpu-id 0 \
  --save-points \
  --benchmark --warmup 10 --iterations 100
```

成功时应看到 `Verified EP: CUDAExecutionProvider`、`Benchmark Session::Run` 和 `Summary: success=1 failed=0 skipped=0`。查看 `outputs/res34_check/result.jpg` 与 `result.json`，profiling 在该目录的 `profiles/` 下。单图重复测试时请更换输出文件名；已有文件会被拒绝覆盖。若出现模型形状不匹配、CUDA 初始化失败等错误，先保留错误信息，解决后再进行全量处理。

### 3. 全量批处理，集中保存 Res34 结果与日志

```bash
rtk proxy mkdir -p outputs/v6453_front_20260916_102329_res34

set -o pipefail

rtk proxy stdbuf -oL -eL ./scripts/run_cpp_gpu.sh \
  --model model/weights/ufldv2_culane_res34_320x1600.onnx \
  --image-dir /home/zenglei/Data/v6453/mcapWriter1_record_20260916-102329/sensor_camera_front_mono_image_encode \
  --output-dir outputs/v6453_front_20260916_102329_res34 \
  --device cuda --gpu-id 0 \
  --save-points --skip-existing \
  2>&1 | rtk proxy tee -a outputs/v6453_front_20260916_102329_res34/run.log

echo "批处理退出码：$?"
```

保持终端打开，等待最终 `Summary`。这里不加 `--benchmark`，避免为每张图片执行额外的预热和重复推理。

- `stdbuf -oL -eL`：按行输出日志，方便观察实时进度。
- `2>&1`：将错误信息合并到标准输出。
- `tee -a`：同时显示并追加保存日志，恢复运行时保留之前的记录。
- `set -o pipefail`：管道中的推理程序失败时，即使最后的 `tee` 保存日志成功，整条管道也返回非零状态。仅影响当前 Shell 的退出状态判断，不改变推理或系统配置。
- 紧接管道执行的 `echo` 显示该管道退出码；0 表示正常完成，非零时检查日志。未检出车道线本身不等同于运行失败。

| 模型 | 输出目录 |
| --- | --- |
| 原 Res18 | `outputs/v6453_front_20260916_102329/` |
| 新 Res34 | `outputs/v6453_front_20260916_102329_res34/` |

### 4. 查看完成情况或恢复中断任务

```bash
rtk proxy tail -n 10 outputs/v6453_front_20260916_102329_res34/run.log
```

对于原有 9,033 张图片，首次完整运行的预期汇总为：

```text
Summary: success=9033 failed=0 skipped=0
```

以实际图库和程序输出为准。若中途中断，重新执行第 3 步的批量命令即可；`--skip-existing` 会验证并跳过已完成的匹配结果，不覆盖它们。恢复运行的汇总中，`success` 是本次新完成数，`skipped` 是先前完成数，`failed` 是本次失败数。

最终新目录包含结果 JPG、同名关键点 JSON、`profiles/` 和 `run.log`。其中单图 benchmark 日志可用于比较 Res18/Res34 的推理速度；检测效果需结合结果图检查，不能仅凭有检测点的图片数量判断准确率。

## 算法、坐标与耗时

- 保留上游流程：BGR 通道顺序（与本仓库 C++/Python 原实现一致，不额外转 RGB），resize 到 `1600 × int(320/0.6)`，取底部 320 行，除以 255 后按 `[0.485,0.456,0.406]` / `[0.229,0.224,0.225]` 归一化，按 NCHW 输入。
- 保留 row lanes 1/2、col lanes 0/3、有效点比例阈值、argmax 附近 ±1 网格 softmax 加权及原图 anchor 映射。修复上游所有负 logits 时 argmax 错误、网格末端邻域越界；使用减最大值后的标准 `exp` softmax 替代位近似，算法含义不变。
- JSON 字段：`source_image`、`image_width`、`image_height`、`model_path`、`device_requested`、`lanes`。每条非空车道是 `[[x,y], ...]`；无检测时 `lanes: []`。
- 坐标原点为原图左上角，x 向右、y 向下，整数像素。过滤非有限值和归一化后超出 `[0,1]` 的点；anchor 恰好落在右/下边界时截到 `width-1` / `height-1`。图片与 JSON 使用同一组点。未增加颜色/虚实线分类。
- 日志分别显示预处理、同步 `Session::Run()`、后处理/绘图和端到端毫秒数。端到端包含读图、保存图片/JSON，扣除 profiling 审计与额外 benchmark 循环，不含模型初始化。首次检测的 Run 包含 profiling 开销，不用于稳定性能结论。
- benchmark 在 profiling 关闭后使用已预处理输入 warmup 10 次、正式 100 次（可调整），显示均值、nearest-rank p50/p95；包括 CPU 输入/GPU 输出传输及同步，不包含读写图和预处理。CPU 固定 4 个 intra-op 线程。报告 CPU/GPU 性能时应同时注明这些条件。

## GPU 真实性验证与故障处理

CUDA 在 Session 创建前调用 `AppendExecutionProvider_CUDA`。CUDA/CPU 均对首张成功运行的图片开启 profiling，文件在 `--output-dir/profiles/`。首次 Run 后关闭 profiling，统计带 provider 的 CUDA/CPU kernel 事件数及累计节点时间；CUDA 必须存在 CUDA kernel 和 Conv/Gemm/MatMul 计算事件，否则明确报错，拒绝将 CPU 回退结果当作 CUDA 成功。少量 CPU 节点允许存在并如实报告。节点累计时间是审计数据，不能替代 benchmark 或证明所有计算都在 GPU。

只有模型 Run 成功且 profiling 审计通过后才输出 `Verified EP` 并保存图像/点。`nvidia-smi` 是辅助证据，CUDA EP API 注册成功也不是推理成功证据。

```bash
# 未配置 cuDNN 路径时，本机原始 ldd 报 libcudnn.so.8 not found
rtk proxy ldd /home/zenglei/opt/onnx-gpu/onnxruntime-linux-x64-gpu-1.15.1/lib/libonnxruntime_providers_cuda.so
# 此脚本在进程内加入已有 cuDNN 目录后再检查，不改系统
rtk proxy ./scripts/check_gpu_env.sh
```

本次加入现有 cuDNN `lib` 路径后 Provider `ldd` 无 `not found`，cuBLAS/cuBLASLt、cuFFT、cuRAND、CUDA runtime 均可解析，未发现明确缺失的 CUDA 11.8 动态库。不过 Toolkit 11.5 早于 ORT 1.15.x 官方测试 CUDA 11.6–11.8 范围，链接完整不能证明执行兼容。参见 [ORT CUDA EP 文档](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)。

普通沙箱运行进程中 `nvidia-smi` 失败、设备不可见，CUDA 返回 100（no CUDA-capable device is detected）。2026-11-16 在允许访问 GPU 的宿主环境中检查 `nvidia-smi` 正常，真实 CUDA 推理成功，确认这是沙箱访问限制。请在本机正常终端执行 CUDA 命令；若使用受限容器/沙箱，需允许 GPU 访问。没有执行驱动、CUDA 或 cuDNN 的系统修改。

若后续出现具体 CUDA 符号/库版本问题，先保留原 Toolkit 与 SDK，记录完整缺库/符号错误，再考虑在独立用户目录并行放置 CUDA 11.8 runtime、仅通过该脚本的进程搜索路径使用；需系统安装或管理员权限时先确认，不覆盖原 CUDA 11.5。

## 初次适配验收记录（历史：2026-11-14）

执行日期：2026-11-14（Asia/Shanghai）。初始 Git 状态只有未跟踪 `.codex/` 和 `model/`；原任务文档、下载中的权重、`opencv/`、`onnxruntime/main.py` 和样例图片均保留。

| 阶段 | 实际执行和结果 |
| --- | --- |
| A 环境/源码 | 已检查 Git、原始 C++/Python、SDK、cuDNN、编译器、OpenCV、原始及配置后 Provider ldd；环境检查脚本退出 1，原因是驱动不可访问和模型缺失 |
| B CMake | 使用本机 ORT 1.15.1 配置成功，Release 构建链接成功，最终无警告；可执行文件 ldd 无缺库、RUNPATH 含 SDK lib |
| C Linux/CUDA | 已编译窄字符 Linux 路径、RAII Session/名称 API、CUDA 注册及 CPU 分支；独立真实 SDK 探针成功调用 CUDA 注册 API，无模型执行 |
| D 脚本 | 已执行 shell 语法检查、进程局部库路径启动和诊断；未修改系统或 SDK |
| E 图片/JSON/批量 | 代码已实现；独立工具测试通过 JPG 解码、PNG 读写、空车道 JSON/路径转义、负 logits argmax/validity sum；真实检测及批处理仍待模型验证 |
| F GPU/性能 | 已执行 nvidia-smi、cudaGetDeviceCount 与无模型 CUDA 注册探针；真实 CPU/GPU Run、profiling 与 benchmark 均待验证，没有推理耗时或车道检测结果 |
| G 文档 | 本文含命令、依赖、算法、坐标、验证标准、故障和待办；构建产物、日志、权重加入忽略规则 |

任务书第 10 节逐项验收：

| 验收项 | 状态 |
| --- | --- |
| 仓库/依赖检查且未覆盖用户修改 | PASS（GPU 访问问题已如实记录） |
| 现有 ORT 1.15.1 SDK 构建成功 | PASS |
| Ubuntu 模型路径类型问题解决 | PASS（编译验证） |
| `--help`；单图 JPG/PNG | `--help` PASS；JPG/PNG 编解码 PASS；真实单图推理 NOT TESTED / 待验证 |
| 文件夹批处理和成功/失败汇总 | NOT TESTED / 待验证（已实现） |
| 车道叠加图片、原图对齐 | NOT TESTED / 待验证 |
| 关键点 JSON、坐标范围正确 | 序列化 PASS；真实关键点及范围 NOT TESTED / 待验证 |
| CUDA 注册与真实模型推理证据 | 注册 API PASS；模型推理/profiling NOT TESTED / 待验证 |
| GPU 初始化失败明确报错、不伪装成功 | 错误传播/拒绝 CPU 回退代码已实现；真实 Session GPU 故障路径 NOT TESTED / 待验证 |
| CPU 可独立推理 | NOT TESTED / 待验证 |
| CPU/GPU 推理耗时 | NOT TESTED / 待验证，无数值可报告 |
| 使用说明和可复制命令 | PASS |
| 修改文件及遗留问题汇总 | PASS |

实际通过的 CLI 用例共 11 项：help、无参数、未知参数、参数缺值、单图/目录互斥、无效设备、无效 GPU ID、无效迭代数、CUDA 缺模型、CPU 缺模型、拒绝下载中模型。检查的是预期退出码与错误文本。模型不存在时退出 1 并说明下载未完成；无参数显示 help 并退出 2。

本地审计文件在忽略目录 `outputs/validation/`：`build.log`、`environment.log`、`provider_ldd_unconfigured.log`、`cudnn_ldd.log`、`executable_ldd.log`、`cli_tests.json`、`ort_cuda_probe.cpp/.log`、`cuda_runtime.log`、`utility_checks.cpp/.log`。其中测试用 PNG 与空列表 JSON 仅用于编解码/序列化验证，绝非模型检测结果。

除上文构建/诊断/CLI 命令外，实际执行的检查还包括 `rtk proxy bash -n scripts/check_gpu_env.sh scripts/run_cpp_gpu.sh`、`rtk proxy ldd build/ufldv2_gpu`、`rtk proxy readelf -d build/ufldv2_gpu`、`rtk proxy git diff --check`。独立 CUDA 注册探针使用 `g++ -std=c++17`、现有 SDK 的 include/lib、`-lonnxruntime` 和 SDK RPATH 编译；通过 `UFLDV2_BINARY` 交给同一启动脚本执行。图像/JSON 工具检查源文件及日志保存在上述目录，使用相同 SDK、系统 OpenCV 库编译执行；没有构造或替换任何 ONNX 模型。另对 cuDNN 的真实版本拆分库逐个执行 `ldd`，结果见 `cudnn_ldd.log`。

修改/新增文件：`onnxruntime/main.cpp`、`CMakeLists.txt`、`scripts/check_gpu_env.sh`、`scripts/run_cpp_gpu.sh`、`.gitignore`、`README_CPP_GPU.md`。未提交 Git commit，未安装任何依赖。

后续待办：完成模型下载；确认推理进程 GPU 可见；执行上面的 CPU、CUDA、批处理命令，并追加本地真实 PNG 道路照片测试；检查叠加图与 JSON 对齐、范围及 CPU/GPU 坐标差异；留存 CUDA profiling 和 mean/p50/p95 日志，之后再判断性能与运行兼容性。
