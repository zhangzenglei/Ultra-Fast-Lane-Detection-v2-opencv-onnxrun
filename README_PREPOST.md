# 不训练的预处理 / 后处理对照实验

2026-11-17：使用已有 CULane Res18 ONNX 和 C++ / ORT GPU 1.15.1，不启动训练、不创建人工真值。新增的是可选实验参数；默认仍为原始 BGR、无增强、row 比例 0.5、col 比例 0.25。原有 `build/ufldv2_gpu` 没有重编译或替换，SHA-256 保持为 `7b902e1cc051088135ffa5f5886d775546faa5bcda5152eed27bc618da67de0c`。

## 两个问题的准确含义

### RGB / BGR

OpenCV `imread` 的通道顺序是 B、G、R。原程序直接把这三个通道按顺序放入 NCHW，并分别使用 `[0.485,0.456,0.406]` / `[0.229,0.224,0.225]` 归一化。本实验的 RGB 模式将通道读取顺序改成 R、G、B，再按同样顺序归一化；不会改变原图文件、绘图颜色或坐标。

官方训练通常使用 RGB，但现有第三方 ONNX 的实际要求仍应通过导出来源和对照验证确认，不能只凭扩展名认定。本次同图比较发现 RGB 在若干鱼骨路段明显增加有效 anchor，支持优先进一步验证 RGB。

纯白/灰像素的 R=G=B，交换顺序不改变其数值；但网络同时观察彩色背景、路面及车辆等上下文，整图的颜色顺序仍会改变预测。没有因为“鱼骨线是白色”就跳过颜色验证。

### 整条车道的有效 anchor 比例

row 分支为内侧槽位 1/2 设置 72 个采样位置。每个位置先根据两个 existence logits 的 argmax 判断存在/不存在，再累计存在数量。原逻辑只有 `valid_count > 72 × 0.5` 才解码并输出整条线，也就是至少 37 个；这里的 0.5 不是“单个点置信度阈值”。

| row 比例 | 整条线保留所需最少有效 anchor |
| --- | --- |
| 0.5 | 37 / 72 |
| 0.4 | 29 / 72 |
| 0.3 | 22 / 72 |

降低此比例只允许已有的短轨迹保留下来，不改变每个点的 existence 判定，不为不存在的位置补点，也不改善位置估计本身。col 分支槽位 0/3 仍使用 81 个采样位置、至少 21 个有效点，实验没有降低它的比例。

真实例子：frame index 2000，原图文件为 `sensor_camera_front_mono_image_encode_1789525618100108.jpg`。同一 BGR 输入下，两条内侧线分别已有 30 和 36 个有效 anchor；0.5 时整帧无输出，0.4 时保留两条线。这证明其中存在后处理过滤，而非完全没有网络响应。

另一例：frame 2500，BGR 下两条内侧线只有 1 和 8 个有效 anchor，降低到 0.4 仍无输出；换 RGB 后为 44 和 48 个，原 0.5 比例即可输出两条线。因此不能把所有漏检都归因于同一个阈值。

## 实际实验与结论边界

从 9,033 张源图片中选取 48 张：36 张鱼骨路段候选（包含附近过渡帧）和 12 张普通道路对照。此场景分组只用于选样，不是人工车道真值，也不是严格独立测试集。没有重新审核或修改既有 CVAT 标注。

执行 RGB/BGR × none/CLAHE × row 比例 0.5/0.4/0.3，共 **12 组、576 张结果图**。每组 C++ CUDA 运行成功，保存独立 ORT profiling、日志、结果图和 JSON。所有输出坐标均检查在原图范围内，crop_ratio 始终为 0.6，输入仍为 1×3×320×1600。

| 参数 | 候选图有输出帧 / 36 | 候选图总线数 | 普通对照有输出帧 / 12 | 普通对照总线数 |
| --- | --- | --- | --- | --- |
| BGR，无增强，0.5（默认） | 14 | 39 | 11 | 40 |
| BGR，无增强，0.4 | 15 | 43 | 11 | 40 |
| RGB，无增强，0.5 | 22 | 61 | 11 | 42 |
| RGB，无增强，0.4 | 24 | 64 | 11 | 42 |
| RGB，无增强，0.3 | 26 | 67 | 11 | 42 |
| RGB，CLAHE，0.4 | 26 | 74 | 11 | 38 |

这是**输出覆盖和数量，不是 Recall、Precision 或正确检出率**。RGB 配置在 frame 2500、3500、4500 等图恢复了纵向轨迹；frame 2000 说明 RGB 也不是逐帧必定优于 BGR。CLAHE 在 frame 6200 等图有帮助，但 frame 2500 等图出现更多抖动和可疑轨迹；普通对照的线数也会减少。frame 2100、6000、6250、6300、7200 等场景部分或全部设置仍无输出，未声称这些失败已解决。

对照图还可看到部分轨迹延伸到车头遮挡区域，新增点/新增线并不等于准确。尚未加入去畸变、道路分割、机盖遮罩、拟合/外推、时序融合或新的 crop；这些都需要单独实验，不能混入本次颜色和比例效果。

更新：用户检查 frame 1950 后指出 RGB 轨迹更抖、偏移更大。因此撤回仅依据输出覆盖数量就优先推荐 RGB 的结论，**目前保留原 BGR 作为部署基线**。RGB、0.4/0.3 和 CLAHE 都仅作为独立实验选项，需要核对输入约定及位置质量，不作为全局默认。没有人工真值时不能选“线数最多”的组就宣布最优。

## 查看结果

- `outputs/prepost_20261117/ablation/index.html`：完整统计和 48 张逐图入口；每张图展示全部 12 种配置和每槽位有效 anchor 数。
- `outputs/prepost_20261117/comparison.png` / `.svg`：五个场景的道路区域对照。彩色连线仅连接实际输出点，不做拟合、不外推，不是人工标注。
- `outputs/prepost_20261117/ablation/summary.json`、`summary.csv`、`frame_results.json`：可审计数据。
- 每组目录中的 `run.log`、`profiles/*.json`：CUDA 证据。
- `selection.json` 保存选中原图和旧 JSON 路径/哈希。48 份既有预测 JSON 核对未修改；其他原始结果和人工标注目录未写入。

默认结果回归：新的默认 BGR 与未修改生产程序复核 48 张图，车道数和每条点数一致；6 张存在最多 1 像素的整数坐标差，记录于 `outputs/prepost_20261117/baseline_recheck.json`。未修改生产程序与其旧存档重跑也有 2 张出现 1 像素差，不能声称全部结果位级相同。原生产二进制的哈希不变，实验不会替换现有部署。

## 构建与使用

```bash
rtk cmake -S . -B build/prepost -DCMAKE_BUILD_TYPE=Release \
  -DORT_ROOT=/home/zenglei/opt/onnx-gpu/onnxruntime-linux-x64-gpu-1.15.1
rtk cmake --build build/prepost -j4
```

新增参数只在新编译的实验程序中可用：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--input-color bgr|rgb` | bgr | NCHW 通道顺序 |
| `--enhance none|clahe` | none | resize 后在 YCrCb 亮度通道做 CLAHE，clipLimit 2.0、8×8 tiles |
| `--row-min-ratio R` | 0.5 | 有效 row anchor 数严格大于 R×72；R 取有限的 [0,1] |
| `--col-min-ratio R` | 0.25 | 同理作用于 col，实验保持默认 |

### 全量 RGB 单变量对照（由用户在终端启动）

`scripts/run_res18_rgb.sh` 固定使用同一个 CULane Res18、RGB、无 CLAHE、row=0.5、col=0.25，与原 BGR 流程相比只改变颜色顺序。默认输入是上述 9,033 张原图所在目录，结果保存到新的 `outputs/v6453_res18_rgb_r50/`。脚本使用 `build/prepost/ufldv2_gpu`，不会替换原 `build/ufldv2_gpu`，不会自动编译、安装或启动训练。

已检查现有 RGB 实现：归一化时先按 R/G/B 读取原图通道，再写入 NCHW；这与先 `cvtColor(BGR2RGB)` 再归一化等价。无需再额外转换一次，否则会交换回 BGR。

```bash
# 首次运行；本次准备阶段没有执行此推理命令
rtk proxy ./scripts/run_res18_rgb.sh

# 可选：先查看完整参数，不推理、不创建结果目录
rtk proxy ./scripts/run_res18_rgb.sh --dry-run

# 如中途停止，使用同样的参数恢复；只跳过配置匹配的完整结果
rtk proxy ./scripts/run_res18_rgb.sh --resume
```

结果目录包含逐图 JPG、关键点/参数/诊断 JSON、ORT profiling，以及独立的 `run_rgb_时间_进程号.log`。终端同步显示处理进度，日志不覆盖之前的运行日志。出现已有目录时，首次命令会拒绝运行；使用 `--resume` 或换新的 `--output-dir`。不要指向原 BGR 或人工标注目录。

也可以手动执行完全等价的命令：

```bash
rtk proxy env UFLDV2_BINARY="$PWD/build/prepost/ufldv2_gpu" \
  ./scripts/run_cpp_gpu.sh \
  --model model/weights/ufldv2_culane_res18_320x1600.onnx \
  --image-dir /home/zenglei/Data/v6453/mcapWriter1_record_20260916-102329/sensor_camera_front_mono_image_encode \
  --output-dir outputs/v6453_res18_rgb_r50 \
  --device cuda --save-points \
  --input-color rgb --enhance none --row-min-ratio 0.5 --col-min-ratio 0.25
```

单图时用 `--image <原图>` 和新的 `--output <结果.jpg>` 替代 `--image-dir`。不要覆盖原有结果或人工审核标签。实验 JSON 多出 `inference_settings`、`lane_diagnostics`，原 `lanes` 字段保持兼容。`--skip-existing` 会额外验证预/后处理参数；旧 JSON 没有配置时只按默认参数对待，不能用它跳过一个 RGB 实验。

`scripts/compare_prepost.py --help` 提供可重复的 12 组实验；它拒绝已有实验目录，使用原图软链接，不复制/修改输入图片。已执行的全部命令在 `outputs/prepost_20261117/process.json` 和每组 `summary.json` / `run.log` 中。

## 验证记录

- 独立 Release 编译成功，无警告；生产二进制未替换。
- C++ 无模型测试通过：真实 BGR/RGB NCHW 数值、底部裁剪、灰度不变性、源图不变、CLAHE 输出类型/尺寸、全部默认 row/col 整数边界、放宽比例边界、非法比例。
- 6 项 CLI/恢复检查通过：非法颜色、非法增强、NaN/越界比例、兼容旧默认结果、拒绝跨配置恢复。
- 12 组全部使用 CUDA，每组 profiling 均有 CUDA kernel 事件。
- 48 张原有 JSON 哈希未变化；576 份新 JSON 坐标范围检查通过。
- 没有人工 GT，因此未测真实召回率/误检率；没有把新增预测当真值或更新 CVAT。

改动：`onnxruntime/main.cpp` 增加可选参数及诊断；新增 `scripts/compare_prepost.py`、`tests/test_prepost_options.cpp` 和本文。CMake、现有运行/诊断脚本、CUDA/驱动/SDK、权重文件及人工标注不变。

## 当前 ONNX 输入约定的只读检查

检查程序：`scripts/inspect_onnx_input.py`，使用隔离验证环境中的 `onnx==1.17.0` 解析文件，不使用 Python ONNX Runtime、不运行模型、不修改权重。实际结果在 `outputs/prepost_20261117/onnx_input_inspection.json`：

- 文件：`model/weights/ufldv2_culane_res18_320x1600.onnx`，825194714 bytes。
- SHA-256：`edcee747f9d800247a200ab7676c03107bb75a4490c61d4f8c473100a9aabb4f`。
- producer：PyTorch 1.12.0，opset 11。
- 输入 `input`：FLOAT（float32），`[1,3,320,1600]`，通过首个 ONNX Conv 使用 NCHW。
- metadata、输入说明、模型/图说明均为空。输入直接进入 `Conv_0`；该路径没有显式的通道重排算子。

**这些只能确定类型/布局，不能从 Conv 权重或三个通道的形状直接判断 RGB/BGR，也不能排除转换曾把通道置换合入卷积权重的可能。** 同样不能单凭 ImageNet mean/std 推断颜色约定。

配套代码证据：[hpc203 的原始推理](https://github.com/hpc203/Ultra-Fast-Lane-Detection-v2-opencv-onnxrun/blob/main/onnxruntime/main.py) 使用 `cv2.imread` 后直接归一化、transpose，没有 BGR→RGB；它提供了作者按 BGR 调用的证据，不等同于已证明训练颜色约定。官方 [pt2onnx.py](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2/blob/master/deploy/pt2onnx.py) 用 dummy tensor 导出网络，没有在脚本中声明颜色顺序；也尚未证明它正是当前第三方文件的导出脚本。

进一步确定需要找到与该 SHA-256 对应的导出记录/训练预处理，或使用同一份 `.pth` 权重进行 PyTorch↔ONNX 的输入张量及四路原始输出数值对照，核实通道重排是否曾在导出时折叠到网络中。若只有一份没有元数据的 ONNX 和最终线条效果，只能选择经验上更合适的调用方式，不能宣称颜色语义已经确定。
