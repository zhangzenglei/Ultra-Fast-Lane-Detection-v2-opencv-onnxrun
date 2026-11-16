# UFLDv2：Ubuntu 22.04 + C++ + ONNX Runtime GPU 开发任务书（供 VS Code Codex 执行）

> 版本：v1.0｜日期：2026-11-14  
> 任务类型：既有 GitHub 项目适配、构建、GPU 推理验证、道路图片测试  
> **目标仓库：** `/home/zenglei/Ultra-Fast-Lane-Detection-v2-opencv-onnxrun`  
> **任务入口：** `.codex/tasks/ufldv2_cpp_gpu_setup.md`  
> **执行原则：** 优先复用现有 C++/ONNX Runtime GPU SDK；最小必要修改；验证优先；保留上游预处理和后处理算法。

## 0. 给 Codex 的总指令（请先读）

你是当前仓库的 C++/CUDA/ONNX Runtime 工程实现助手。请**先检查真实源码和本机环境，再实施修改**，按第 6 节的阶段顺序推进，最终交付可实际编译运行的工程、必要脚本、使用说明和验证记录。不要只给出方案而不修改代码；除非必要的依赖/模型缺失或需要管理员权限，否则尽可能在本轮完成能执行的工作。

要求：

- 在仓库内改动，遵循任何现有的 `AGENTS.md`、代码风格和用户未提交更改；**先运行 `git status --short`，不得覆盖用户修改**。
- 保留 hpc203 项目 `onnxruntime/main.cpp` 中已实现的 UFLDv2 图像预处理、输出解析、车道线关键点还原和绘制逻辑，不要另写一套臆测的后处理。
- **必须走 C++ + ONNX Runtime CUDA Execution Provider**；不要改走 Python、PyTorch、OpenCV DNN 或重新训练模型。
- 未得到用户授权，不得执行 `sudo apt upgrade`、更换显卡驱动、全局替换 CUDA Toolkit/cuDNN、删除现有 SDK、修改 `~/.bashrc` 等影响其他工程的操作。可以创建仓库内构建目录和启动脚本；安装缺失的系统软件包须先征求授权。
- 任务所需的模型权重不随仓库提供，下载必须使用用户认可的官方/项目提供地址；**不能捏造模型文件或推理结果**。
- 如果 GPU 初始化失败，应输出清晰的错误，不得静默以 CPU 跑完并宣称 GPU 成功。
- 每个阶段记录实际命令、结果、问题和未完成项。最后按第 10 节验收。

## 1. 用户目标

在个人 Ubuntu 22.04 笔记本上使用 NVIDIA RTX 4050 Laptop GPU，为下载好的 hpc203 UFLDv2 ONNX 项目配置 **C++17 + OpenCV + ONNX Runtime GPU + CUDA** 推理工程。用户有多张本地道路实景 JPG/PNG，希望直接输入原始道路照片，得到：

1. 在原图上绘制车道线的可视化结果（如 `result.jpg`）。
2. 每条检测到的车道线的**原图像素坐标**（关键点列表，可导出 JSON 或 CSV，优先 JSON）。
3. 可选择单张图片或批量文件夹输入。
4. 能确认 ONNX Runtime 的主要推理计算确实运行在 GPU，并输出推理耗时。
5. 后续方便单独做 CPU/GPU 性能对比，不要求首版实现 TensorRT、CUDA Graph 或自定义 CUDA kernel。

先做到**推理**，不涉及模型训练、数据集标注、车道线颜色/虚实语义分类，也不涉及 BEV 或车辆坐标系标定。

## 2. 用户机器及已有软件（来自终端记录；执行时应再次核验）

| 项目 | 已知信息 | 注意事项 |
| --- | --- | --- |
| OS | Ubuntu 22.04 | 优先系统自带编译工具 |
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU，6 GB | GPU 编号默认 0 |
| NVIDIA Driver | 590.48.01 | `nvidia-smi` 显示的 CUDA 13.1 是**驱动支持版本** |
| CUDA Toolkit | `nvcc` 11.5（V11.5.119） | 不等于驱动显示的 CUDA 13.1 |
| cuDNN | 解压版 8.7.0.84 for CUDA 11 | 需要验证动态库搜索路径 |
| ONNX Runtime | C/C++ Linux GPU SDK 1.15.1 | **不是 Python wheel/虚拟环境** |
| Python `onnxruntime` | 未安装 | 本项目**不要求**安装 Python 包 |

SDK 的用户已知路径：

```text
/home/zenglei/opt/onnx-gpu/
├── onnxruntime-linux-x64-gpu-1.15.1/
├── onnxruntime-linux-x64-gpu-1.15.1.tgz
└── cudnn-linux-x86_64-8.7.0.84_cuda11-archive/
```

默认变量（脚本可设置，**允许被用户环境变量覆盖**）：

```bash
export ORT_ROOT=/home/zenglei/opt/onnx-gpu/onnxruntime-linux-x64-gpu-1.15.1
export CUDNN_ROOT=/home/zenglei/opt/onnx-gpu/cudnn-linux-x86_64-8.7.0.84_cuda11-archive
```

ONNX Runtime 1.15.x 官方 CUDA 11.8 预编译版本使用 cuDNN 8.x；官方记载测试 CUDA 11.6–11.8、cuDNN 8.2–8.9。**现有 Toolkit 11.5 比测试范围更旧**：先通过真实运行与依赖检查判断是否可用，必要时再建议并行补充 CUDA 11.8 运行库。不要因为驱动显示 CUDA 13.1 就错误改装 CUDA 13/ORT 新版。见第 11 节官方资料。

## 3. 真实上游仓库信息（已经核对，但以本地源码为准）

仓库：`https://github.com/hpc203/Ultra-Fast-Lane-Detection-v2-opencv-onnxrun`

- 关注 **`onnxruntime/main.cpp`**（C++），不是 `onnxruntime/main.py`。
- 原 C++ 文件包含预处理、ONNX Runtime `Session::Run`、后处理/绘图；无需重写算法。
- 原代码创建 Session 时未启用 CUDA；在 `main.cpp` 约第 90 行附近有**注释掉**的 CUDA EP 提示代码。
- 原代码曾将模型路径转为 `std::wstring`，这是 Windows 风格；Linux C++ 应采用 `std::string::c_str()`。
- 原 `main()` 模型路径写死为 `weights/ufldv2_culane_res18_320x1600.onnx`，图片写死为 `images/culane/00000.jpg`。
- 原仓库**没有现成的 CMake 配置**，需自行增加构建文件。
- 原 README 所述 OpenCV 4.7+ 主要针对 **OpenCV DNN 读取 ONNX**，本任务使用 ONNX Runtime，不应将其误作强制升级要求。
- 参考源码：`https://github.com/hpc203/Ultra-Fast-Lane-Detection-v2-opencv-onnxrun/blob/main/onnxruntime/main.cpp`

**编写任何 patch 前先打开本地源码，确认成员名、构造函数签名、OpenCV 用法和模型输入输出尺寸，不能照搬未验证的示例变量名。**

## 4. 交付范围和建议目录

以当前仓库为根目录，建议新增/修改如下（若现有结构更合理可调整，但需说明）：

```text
Ultra-Fast-Lane-Detection-v2-opencv-onnxrun/
├── .codex/
│   └── tasks/
│       └── ufldv2_cpp_gpu_setup.md        # 本需求文档，保留
├── onnxruntime/
│   ├── main.cpp                            # 复用并适配为 C++ GPU Demo
│   └── images/                             # 原有样例图片
├── CMakeLists.txt                         # 新增，直接构建 ONNX Runtime C++ Demo
├── scripts/
│   ├── check_gpu_env.sh                    # 新增，可重复执行的环境检查
│   └── run_cpp_gpu.sh                      # 新增，配置动态库并启动
├── weights/                               # 可放用户下载的 ONNX 文件，不提交大文件
├── outputs/                               # 推理结果，默认忽略版本控制
├── README_CPP_GPU.md                      # 新增：构建、运行、问题排查
└── build/                                 # CMake 构建目录
```

不要删除/覆盖上游 `opencv/`、`onnxruntime/main.py` 或其他已有资源。`.gitignore` 只做最小增补，保护原有规则；模型文件和本地照片不纳入提交。

## 5. 功能需求

### 5.1 CLI 参数（推荐接口）

新 C++ 可执行程序建议叫 `ufldv2_gpu`。提供 `--help`；支持以下参数（如与代码结构冲突，可保留等价接口，并在 README 说明）：

| 参数 | 说明 | 是否必需 |
| --- | --- | --- |
| `--model <path>` | UFLDv2 `.onnx` 模型路径 | 必需（可设合理默认） |
| `--image <path>` | 单张 JPG/PNG 道路照片 | 单图模式必需 |
| `--image-dir <dir>` | 批量扫描 JPG/JPEG/PNG | 批量模式必需，与 `--image` 互斥 |
| `--output <path>` | 单图绘制结果，如 `outputs/result.jpg` | 单图可选 |
| `--output-dir <dir>` | 批量输出目录 | 批量可选 |
| `--device cuda\|cpu` | 默认 `cuda`，供对比测试 | 可选 |
| `--gpu-id <n>` | GPU device ID，默认 0 | 可选 |
| `--save-points` | 同时输出每张图的车道线关键点 JSON | 可选 |
| `--benchmark` | 执行 warmup + 重复推理，并打印统计数据 | 可选 |

参考最终调用方式（**目标命令，而非上游已经支持的参数**）：

```bash
# 单图，GPU 推理
./build/ufldv2_gpu \
  --model weights/ufldv2_culane_res18_320x1600.onnx \
  --image /home/zenglei/road_images/road001.jpg \
  --output outputs/road001_result.jpg \
  --device cuda --gpu-id 0 --save-points

# 批量图片，GPU 推理
./build/ufldv2_gpu \
  --model weights/ufldv2_culane_res18_320x1600.onnx \
  --image-dir /home/zenglei/road_images \
  --output-dir outputs/batch \
  --device cuda --save-points

# CPU 对照
./build/ufldv2_gpu \
  --model weights/ufldv2_culane_res18_320x1600.onnx \
  --image /home/zenglei/road_images/road001.jpg \
  --output outputs/road001_cpu.jpg \
  --device cpu --benchmark
```

### 5.2 保留正确预处理/后处理

- 必须复用/核对原项目 **CULane ResNet18** ONNX 模型要求的 `crop_ratio`、裁剪区域、Resize、BGR/RGB、归一化方式和 NCHW 输入布局；**不能用任意 Resize 替代原逻辑**。
- 原图分辨率可不同于模型输入尺寸；关键点应正确映射回**原始输入图片像素坐标**。
- 优先从 `Session` 查询实际输入维度/类型和输出维度，验证与预处理一致；遇到动态维度需明确处理。
- 确保空图片、模型不存在、模型与后处理版本不匹配时输出可理解错误并返回非零退出码。
- 结果图片中车道线颜色明显，坐标位于原图有效范围内。若未检测到车道线，明确输出空列表，不能伪造检测点。
- 第一期不做车道线实/虚线或黄/白线分类；原模型未直接提供时不可臆测。

### 5.3 GPU 推理与回退策略

- CUDA 使用 ONNX Runtime 1.15.1 C++ API：

```cpp
OrtCUDAProviderOptions cuda_options{};
cuda_options.device_id = gpu_id;
sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
```

- 上面片段需要放在原有 Session 创建**之前**；按实际成员命名调整。
- Linux 模型路径改用 `model_path.c_str()`，不要在 Linux 继续向 `Ort::Session` 传递宽字符串。
- `--device cuda` 应主动注册 CUDA EP；GPU EP 初始化/执行失败时**不得默默回退后宣称 CUDA 成功**。能检测到回退则报错并给出诊断；如果个别运算节点合法落在 CPU，需通过 profiling 如实报告。
- `--device cpu` 显式使用 CPU，作为准确性和性能对比。
- 首版可采用 **CPU 读取/预处理 → CPU 输入 Tensor → ORT GPU 推理 → CPU 输出/后处理**，不强求零拷贝；后续性能优化再考虑 I/O Binding。
- 对检测时间分别统计：预处理、`Session::Run()`、后处理/绘图以及端到端耗时。GPU 统计要避免异步计时偏差；可以利用 `Run()` 的同步完成语义做基线测量。
- 开启 ORT profiling 或等价可审计方法，检查实际 `CUDAExecutionProvider` 节点；`nvidia-smi` 仅作为辅助证据。

### 5.4 保存关键点与图片

- 单图 `--save-points` 输出与图片同名的 `.json` 文件，例如 `road001_result.json`。
- JSON 至少包含：`source_image`、`image_width`、`image_height`、`model_path`、`device_requested`、`lanes`（每条车道线的 `[x, y]` 点序列）。
- 坐标是**输入原图**的像素坐标；文档必须说明坐标原点、方向和是否存在无效点过滤。
- 批量模式保留各图片的基本文件名；同名冲突要避免意外覆盖。某张图片失败时记录问题，并继续处理其他有效图片，退出时汇报成功/失败数量。

## 6. 分阶段执行步骤（按顺序落实）

### 阶段 A：环境和仓库检查（先做，无破坏性）

1. 运行：

```bash
cd /home/zenglei/Ultra-Fast-Lane-Detection-v2-opencv-onnxrun
git status --short
find onnxruntime -maxdepth 2 -type f | sort | head -80
nvidia-smi
nvcc --version
cmake --version
g++ --version
pkg-config --modversion opencv4 || true

export ORT_ROOT=/home/zenglei/opt/onnx-gpu/onnxruntime-linux-x64-gpu-1.15.1
export CUDNN_ROOT=/home/zenglei/opt/onnx-gpu/cudnn-linux-x86_64-8.7.0.84_cuda11-archive
find "$ORT_ROOT" -maxdepth 2 -type f -name '*onnxruntime*' | sort | head -50
find "$CUDNN_ROOT" -maxdepth 2 -name 'libcudnn.so*' -print
```

2. 核对 `ORT_ROOT/include/onnxruntime_cxx_api.h`、`ORT_ROOT/lib/libonnxruntime.so`、`libonnxruntime_providers_cuda.so` 是否存在；若分发包目录布局不同，以实际路径为准。
3. 用 `ldd` 检查 CUDA Provider 依赖，特别注意 `libcublas.so.11`、`libcublasLt.so.11`、`libcufft.so.10`、`libcudnn.so.8`、`libcudart.so.11.0` 等 `not found`，不可只看 `nvcc --version`。
4. 确认 ONNX 模型文件是否存在。若没有，说明下载来源、目标目录和手动下载步骤，**不要将缺失模型当作“已测试通过”**。
5. 检查上游源码原始类、默认路径、OpenCV 和 ORT API，记录需要的最小改动。

### 阶段 B：增加 CMake 构建

- 在仓库根目录新增 `CMakeLists.txt`，构建 `onnxruntime/main.cpp`，输出 `build/ufldv2_gpu`。
- `project(... LANGUAGES CXX)`、`CMAKE_CXX_STANDARD 17`；优先 `find_package(OpenCV REQUIRED)`。
- `ORT_ROOT` 作为可覆盖的 CMake cache 变量：`-DORT_ROOT=...`，不要将路径锁死到不可修改的 C++ 源码中。
- 包含 `${ORT_ROOT}/include`，链接 `${ORT_ROOT}/lib/libonnxruntime.so` 和 `${OpenCV_LIBS}`。
- ORT CUDA Provider 通常由 ORT 动态加载，不必盲目直接链接 `libonnxruntime_providers_cuda.so`。可设置 `BUILD_RPATH` 为 SDK 的 `lib` 目录，并提供显式 `LD_LIBRARY_PATH` 启动脚本以定位 CUDA/cuDNN 动态库。
- 不需要 `enable_language(CUDA)`，也不需要使用 `nvcc` 编译 C++ 源码（当前没有自定义 `.cu`）。
- 编译前确认依赖是否齐全；不允许在未经授权情况下修改全局系统配置。

### 阶段 C：Linux + CUDA 适配

- 保留原 `Ultra_Fast_Lane_Detection_v2` 相关推理逻辑。
- 把 `std::wstring` 模型路径改成 Linux 支持的 `const char*`，修复其他实际遇到的跨平台编译问题。
- 在 Session 创建前注册 CUDA EP，并添加 `--device cpu` 路线；异常消息包含模型路径、设备和底层 `Ort::Exception::what()`。
- CLI 解析清晰、易用；支持参数缺失时 `--help` 和非零退出。
- 单图与批量路径支持绝对路径、相对路径；Windows 代码应保留条件编译而不是破坏其他平台（如果仓库没有跨平台约束，以 Linux 优先）。

### 阶段 D：动态库/脚本配置

新增 `scripts/check_gpu_env.sh`，只做诊断不写系统配置：

- 显示 NVIDIA GPU、驱动、Toolkit、ORT SDK、cuDNN 路径。
- 对 Provider `.so` 运行 `ldd`，突出显示 `not found`。
- 验证 GPU 驱动、模型、OpenCV、构建工具的存在性。

新增 `scripts/run_cpp_gpu.sh`，**只为当前进程**配置 `ORT_ROOT`、`CUDNN_ROOT`、必要的 CUDA runtime 搜索路径、`LD_LIBRARY_PATH`，然后运行用户传入的 `ufldv2_gpu` 参数。不要修改用户 shell 启动文件。

### 阶段 E：推理、可视化和批处理

- 完成第 5 节的 CLI、模型输入预处理、数据正确性、后处理结果、保存图片和 JSON。
- 批量模式从用户提供目录检索常见图片扩展名；对单张失败给出错误，并最终汇总。
- 保证绘图使用**原图坐标系**；`imwrite` 返回失败时不能忽略。
- 如果测试图片或模型文件缺失，先利用仓库已有样例或完成构建/依赖测试，并明确报告未验证项目。

### 阶段 F：GPU 真实性与性能验证

- 至少分别跑一张图片的 `--device cpu` 和 `--device cuda`。
- 输出：推理 EP、输入分辨率、模型输入张量 shape、检测到的车道线条数、输出路径、前/后处理时间和 `Session::Run` 时间。
- benchmark 建议 warmup 10 次、正式 100 次（可配置）。统计均值、p50/p95；不要只测一次就下结论。
- GPU 模式开启 ONNX Runtime profiling，至少能说明主要计算节点使用 CUDA EP；若为 CPU 回退、模型/依赖缺失等，必须明确说明。
- 不预设 GPU 一定比 CPU 快；比较时考虑数据拷贝、模型较轻量和预/后处理开销。

### 阶段 G：文档及验收报告

- 完善 `README_CPP_GPU.md`：前置条件、目录布局、模型下载、build/run、单图、批量、CPU/GPU 对比、问题排查。
- 仅将合理的 `build/`、`outputs/`、权重文件等加入 `.gitignore`（保留已有用户配置）。
- 总结修改的文件、编译结果、执行过的命令、耗时、GPU 验证证据、遗留问题。没有证据的项目标注“未验证”，不能写“成功”。

## 7. 推荐构建与运行命令（Codex 可据此实现）

```bash
cd /home/zenglei/Ultra-Fast-Lane-Detection-v2-opencv-onnxrun

export ORT_ROOT=/home/zenglei/opt/onnx-gpu/onnxruntime-linux-x64-gpu-1.15.1
export CUDNN_ROOT=/home/zenglei/opt/onnx-gpu/cudnn-linux-x86_64-8.7.0.84_cuda11-archive
export LD_LIBRARY_PATH="$ORT_ROOT/lib:$CUDNN_ROOT/lib:${LD_LIBRARY_PATH:-}"

# 根据实际缺失情况决定是否额外加入 CUDA 11.x runtime lib 目录
ldd "$ORT_ROOT/lib/libonnxruntime_providers_cuda.so" | grep -E 'not found|cuda|cudnn|cublas|cufft|curand' || true

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DORT_ROOT="$ORT_ROOT"
cmake --build build -j"$(nproc)"

./scripts/run_cpp_gpu.sh \
  --model weights/ufldv2_culane_res18_320x1600.onnx \
  --image /home/zenglei/road_images/road001.jpg \
  --output outputs/result.jpg \
  --device cuda --save-points
```

注意：第 7 节命令是假定项目按本文要求改造后的**目标命令**；执行之前必须完成 CMake、脚本和 CLI 实现。用户的真实照片路径尚未确认，上述 `road001.jpg` 是示例。

## 8. 错误处理与排查优先级

| 故障 | 优先排查 | 禁止误判 |
| --- | --- | --- |
| 编译报 ORT 头文件不存在 | SDK include 路径 / CMake 配置 | 不要改成 Python |
| 找不到 `libonnxruntime.so` | RPATH、`LD_LIBRARY_PATH`、SDK 路径 | 不要直接复制到 `/usr/lib` |
| `libcudnn.so.8` / `libcublas.so.11` not found | cuDNN 和 CUDA runtime 的 `lib` 搜索路径 | 不要简单升级 NVIDIA 驱动 |
| CUDA EP 加载失败 | ORT 1.15.1 / CUDA 11.x / cuDNN 8.x 版本与依赖 | 不要把 CPU 成功当 GPU 成功 |
| 图片不存在/读图为空 | 输入路径、权限、扩展名 | 不要继续处理空 Mat |
| 模型输入 shape 或输出不一致 | ONNX 权重、预处理、后处理绑定 | 不要硬编结果点 |
| GPU 监控看到利用率 0% | 小模型推理过快；查 ORT Profiling | 不要仅凭瞬时 GPU Util 判定未运行 |
| GPU 比 CPU 慢 | 小 batch、数据拷贝、预/后处理、warmup | 不要直接断言 CUDA 配置错误 |

如现有 CUDA 11.5 运行库无法支持该 ORT GPU 包，**优先给出“并行安装 CUDA 11.8 runtime、不覆盖原 Toolkit”的详细方案，并在需要管理员权限或环境重大变更前取得用户同意**。

## 9. 安全边界与性能原则

- 任何 `rm -rf`、`git reset --hard`、重新克隆覆盖本地仓库等破坏操作禁止执行。
- 不自动修改现有原始模型文件；不把大模型、用户图片、诊断日志提交进 Git。
- 不因为 Git 仓库内没有 `AGENTS.md` 就忽略用户需求；**本文是本次开发任务主文档**。
- 如果要做跨文件重构，应先说明原因；避免为一个 Demo 引入复杂框架。
- 先保证功能正确，再优化性能；不要把未完成的 GPU 运行配置描述为“已完成”。

## 10. 验收清单（必须逐条汇报 PASS / FAIL / NOT TESTED）

- [ ] 本地仓库/依赖检查完成，且未覆盖用户未提交修改。
- [ ] 用现有 ONNX Runtime 1.15.1 C++ SDK 构建成功。
- [ ] Ubuntu 22.04 下模型路径类型问题解决。
- [ ] `--help` 可用；支持单图 JPG/PNG。
- [ ] 支持图片文件夹批处理并汇报成功/失败数。
- [ ] 输出车道线叠加图片，像素坐标与原图对齐。
- [ ] 可输出每条车道线关键点 JSON，坐标范围正确。
- [ ] CUDA EP 注册与模型推理成功（附运行日志/Profiling 证据）。
- [ ] GPU 初始化失败时明确报错，不伪装为 GPU 成功。
- [ ] `--device cpu` 可独立执行用于对比。
- [ ] 记录至少一次 CPU/GPU 推理耗时；缺少模型/输入时明确写 NOT TESTED。
- [ ] 编写 `README_CPP_GPU.md` 并给出可复制的命令。
- [ ] 汇总被修改/新增的文件、验证情况及后续待办。

## 11. 参考资料

- hpc203 项目仓库：https://github.com/hpc203/Ultra-Fast-Lane-Detection-v2-opencv-onnxrun
- hpc203 C++ 源码：https://github.com/hpc203/Ultra-Fast-Lane-Detection-v2-opencv-onnxrun/blob/main/onnxruntime/main.cpp
- 上游模型下载提示及网盘入口：见 hpc203 仓库 `README.md`（注意具体权重文件需要手动确认）。
- ONNX Runtime CUDA EP 官方兼容性：https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
- ONNX Runtime C++ API 1.15.1：https://github.com/microsoft/onnxruntime/blob/v1.15.1/include/onnxruntime/core/session/onnxruntime_cxx_api.h
- Codex `AGENTS.md` 自动加载规则：https://learn.chatgpt.com/docs/agent-configuration/agents-md

## 12. 在 VS Code Codex 里启动本任务（直接复制以下提示词）

请读取当前项目的 `.codex/tasks/ufldv2_cpp_gpu_setup.md`，严格按该文档执行 C++ + ONNX Runtime GPU 的 UFLDv2 适配。先检查本地源码、`git status` 和 CUDA/cuDNN/ORT 动态库，复用 `/home/zenglei/opt/onnx-gpu/` 中现有的 ORT 1.15.1 C++ GPU SDK 与 cuDNN 8.7；不要安装 Python ONNX Runtime，不要随意改系统 CUDA 或显卡驱动。依阶段完成 CMake、Linux 路径适配、CUDAExecutionProvider、单图/批量图片、可视化、关键点 JSON、GPU profiling、CPU/GPU benchmark、README，并在最后逐项报告实际编译/运行/未验证结果。遇到缺模型、缺库或需要 sudo 的事项先明确说明阻塞原因，不要伪造成功结果。
