# UFLDv2 真实模型验证：2026-11-16

使用原有 Release 可执行文件和本机 ORT GPU C++ SDK 1.15.1、cuDNN 8.7、CUDA 11.5。未重编译、更换依赖或修改模型/系统配置。本次仅更新文档，检测产物写入忽略目录 `outputs/validation_20261116/`。

## 模型与 GPU 访问

已确认 `model/weights/` 四个 `.onnx` 下载完成。本次只验证 `ufldv2_culane_res18_320x1600.onnx`（825194714 bytes）；CULane Res34 未测试，TuSimple 不属于当前程序支持范围。

普通沙箱下 CUDA 初始化明确报错并退出 1：`CUDA failure 100: no CUDA-capable device is detected`，没有输出伪造的 CUDA 结果。随后在允许访问 GPU 的宿主环境执行 `nvidia-smi`，确认 RTX 4050 Laptop、驱动 590.48.01 正常，并在该环境完成真实 GPU 测试。问题来自沙箱设备访问限制，不需要为此升级 CUDA 或重装驱动。

## 实际命令

从项目根目录执行；GPU 命令需在能访问 GPU 的本机终端中运行。输出文件已存在时程序拒绝覆盖，重测请换输出路径。

```bash
rtk proxy ./scripts/run_cpp_gpu.sh \
  --model model/weights/ufldv2_culane_res18_320x1600.onnx \
  --image onnxruntime/images/culane/00000.jpg \
  --output outputs/validation_20261116/cpu/00000_result.jpg \
  --output-dir outputs/validation_20261116/cpu \
  --device cpu --save-points --benchmark --warmup 10 --iterations 100

rtk proxy ./scripts/run_cpp_gpu.sh \
  --model model/weights/ufldv2_culane_res18_320x1600.onnx \
  --image onnxruntime/images/culane/00000.jpg \
  --output outputs/validation_20261116/cuda/00000_result.jpg \
  --output-dir outputs/validation_20261116/cuda \
  --device cuda --save-points --benchmark --warmup 10 --iterations 100

rtk proxy ./scripts/run_cpp_gpu.sh \
  --model model/weights/ufldv2_culane_res18_320x1600.onnx \
  --image-dir onnxruntime/images/culane \
  --output-dir outputs/validation_20261116/batch_cpu --device cpu --save-points

rtk proxy ./scripts/run_cpp_gpu.sh \
  --model model/weights/ufldv2_culane_res18_320x1600.onnx \
  --image-dir onnxruntime/images/culane \
  --output-dir outputs/validation_20261116/batch_cuda --device cuda --save-points
```

## 实测结果

| 项目 | CPU | CUDA |
| --- | --- | --- |
| 单图退出状态 | 0，success=1 failed=0 | 0，success=1 failed=0 |
| 单图尺寸 / 车道数 | 1640×590 / 3 | 1640×590 / 3 |
| 每条车道点数 | 61、62、37 | 61、62、37 |
| benchmark mean | 132.258 ms | 11.1457 ms |
| benchmark p50 | 129.651 ms | 11.1073 ms |
| benchmark p95 | 156.531 ms | 11.6644 ms |
| 预处理（首图） | 8.66331 ms | 6.50155 ms |
| 首次 Run（含 profiling/冷启动） | 229.017 ms | 874.702 ms |
| 后处理及绘点（首图） | 3.58652 ms | 1.13143 ms |
| 首图端到端（不含 benchmark 额外循环） | 250.571 ms | 889.766 ms |
| 批量 5 张 | success=5 failed=0 | success=5 failed=0 |

benchmark 在关闭 profiling 后 warmup 10 次、测量 100 次；CPU intra-op 线程数为 4。均值对比 CUDA 约快 11.9 倍，仅指同步推理阶段，不能当作整条图片处理流程加速比。首次 CUDA Run 包含冷启动，因此明显较慢。GPU 批处理中后续图片 Run 约 10.6–11.3 ms，端到端约 19.9–20.8 ms。

单图 CPU/CUDA JSON 坐标完全一致，最大差为 0 像素。所有批量 JSON 坐标范围已用程序检查；`00470.jpg` 的 CPU/CUDA 均输出空车道列表，不构造点。已目视检查 `00000_result.jpg`：绿色关键点位于道路车道线附近；这不等同于有标注数据集的精度评测。批量 CPU/GPU 的逐图比较结果保存在 `summary.json`。

## CUDA 真实性证据

成功会话 profiling：

`outputs/validation_20261116/cuda/profiles/ort_cuda_1918292276815_2026-11-16_13-43-54.json`

- `CUDAExecutionProvider` kernel events：68。
- CUDA Conv：21；CUDA Gemm：2。
- `CPUExecutionProvider` kernel events：0。
- 日志明确输出 `Verified EP: CUDAExecutionProvider`。

上述文件已独立解析复核。相同目录另有普通沙箱初始化失败时的空 profiling 文件，它没有 CUDA 节点，不能用作成功证据。批量 CUDA 会话也记录 68 个 CUDA kernel、23 个 Conv/Gemm 事件、0 个 CPU kernel。

## 本次状态与后续

PASS：真实 CULane Res18 的 CPU/CUDA 单图、结果保存、关键点 JSON/范围、GPU profiling、CPU/GPU benchmark、5 张 JPG 批处理，以及 GPU 不可见时明确失败。

未验证：用户自己的道路照片、真实 PNG 推理、CULane Res34、批量坏图继续处理与同名冲突的实际场景。TuSimple 当前不支持。

所有产物与数值汇总：`outputs/validation_20261116/summary.json`。下一步可将运行命令中的 `--image` 换为本地道路照片，或用 `--image-dir` 指向本地照片目录；保留 `--save-points` 即可同时保存原图像素坐标。
