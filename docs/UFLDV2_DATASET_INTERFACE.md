# CVAT → CULane → UFLDv2 Res18：后续转换约定

状态：规划接口，尚未实现训练数据转换。当前入口是 `scripts/check_cvat_review.py`，只能审核人工修正后的 XML，不生成真值、缓存或训练进程。

未来微调还需要官方 Res18 的 PyTorch `.pth` checkpoint，按官方训练代码加载；当前部署用的 `.onnx` 不能直接当作官方 `finetune` 参数。该权重的选择/获取留到训练阶段，本次没有下载训练权重或安装 PyTorch。

## 核对的官方代码版本

UFLDv2：[`cfzd/Ultra-Fast-Lane-Detection-v2` @ c903880678454dfd9b55a63022368db05c00bc6d](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2/tree/c903880678454dfd9b55a63022368db05c00bc6d)。

| 官方文件 | 已核实的约束 |
| --- | --- |
| [`configs/culane_res18.py`](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2/blob/c903880678454dfd9b55a63022368db05c00bc6d/configs/culane_res18.py) | backbone 18、4 槽、输入 1600×320、crop_ratio 0.6、row/col head 为 72/81 anchors、200/100 cells |
| [`utils/common.py`](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2/blob/c903880678454dfd9b55a63022368db05c00bc6d/utils/common.py) | 实际 `get_train_loader()` 走 `TrainCollect`，读取 `list/train_gt.txt`；row anchor 为 0.42～1，col anchor 为 0～1 |
| [`data/dali_data.py`](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2/blob/c903880678454dfd9b55a63022368db05c00bc6d/data/dali_data.py) | 读取图片、seg mask、`culane_anno_cache.json`；CULane 宽高写死 1640/590；对图片、mask、点共同增强，resize 后底部 crop，插值/量化形成 row/col 监督 |
| [`scripts/cache_culane_ponits.py`](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2/blob/c903880678454dfd9b55a63022368db05c00bc6d/scripts/cache_culane_ponits.py) | 文件名原文就是 `ponits`；读取 `.lines.txt` 及 mask，缓存每图 `[4,35,2]`；y 样本为 250、260、…、590，缺失 x 初始化为 -99999，按 mask 像素 ID 确定槽位 |
| [`evaluation/eval_wrapper.py`](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2/blob/c903880678454dfd9b55a63022368db05c00bc6d/evaluation/eval_wrapper.py) | 多处将预测坐标乘 1640/590，不能只改配置文件就宣称原生 4K 已适配 |
| [`INSTALL.md`](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2/blob/c903880678454dfd9b55a63022368db05c00bc6d/INSTALL.md) | 官方 CULane 目录和 cache 步骤；训练另需 DALI 与 my_interp，不是本 C++ 推理 SDK 的一部分 |

仓库也包含 `data/dataloader.py` / `LaneClsDataset` 等代码，但不能据此误判当前主训练入口仍是该旧式 loader。上表从实际调用链核对。

## CULane 风格数据需要的文件

以完全审核后的帧为输入，未来转换器需要输出：

```text
dataset_root/
├── images/<sequence>/<frame>.jpg
├── images/<sequence>/<frame>.lines.txt
├── laneseg_label_w16/<sequence>/<frame>.png
├── list/train_gt.txt
├── list/val_gt.txt                          # 后续训练/评估入口显式接入
├── list/val.txt
├── list/test.txt                           # 独立测试，不参与调参
├── culane_anno_cache.json
├── manifest.json                           # 审核、原图与变换溯源
└── splits.csv                              # sequence/group → split
```

`.lines.txt` 每条存在的车道占一行，格式为 `x1 y1 x2 y2 ...`，不是任意 CSV。标准 `train_gt.txt` 每行是图片路径、mask 路径、从左到右 4 个 0/1 存在标记，见 [CULane 官方数据说明](https://xingangpan.github.io/projects/CULane.html)。mask 像素值 0 为背景，1～4 为槽位 0～3，不能保存成仅供显示的调色颜色或 0/255 二值图。

官方 DALI 路径即使 `use_aux=False` 仍读取 seg mask；只导出 `.lines.txt` 并不足以直接训练。cache 工具也不是把任意 CVAT 点直接打包即可，它依赖固定 y 采样、mask 中的车道 ID、正确的路径 key 和缺失点语义。`np.float` 在新 NumPy 中不可用等上游兼容问题需在未来独立训练环境处理，本阶段没有修改上游或系统依赖。

若继续复用原 cache 脚本，`.lines.txt` 的 y 必须按其标准行采样并写成整数字符串（如 `250`，不能是 `250.0`）；同一 y 存在多个 Polyline 交点时需明确处理，不能随意排序/外推。原工具会按中点附近的 mask ID 找槽位并使用 `mid_y-1` / `mid_x-1`，因此采样、坐标边界和 mask 绘制必须做联合验证。路径 key 为去掉开头 `/` 后的图片相对路径，与训练 list 保持一致。

标准 mask 的背景和车道存在标记只能从人工审核结果生成。`pending` / 未出现在人工导出里的图片必须拒绝；`excluded` 不进入任何监督集合；只有人工确认的 `approved_negative` 才能输出无车道负样本。

## 3840×2160 与 1640×590 的映射不能混用

原图宽高比约 1.778，标准 CULane 约 2.780，二者不同。当前 CVAT 阶段固定保存原图坐标，不做 resize/crop。后续转换前必须选择并记录一种策略，默认保持 `undecided` 并阻止产出训练集。

**策略 A：明确转换到标准 CULane 尺寸。** 整张图非等比缩放到 1640×590 时：

```text
sx = 1640 / 3840 ≈ 0.4270833
sy =  590 / 2160 ≈ 0.2731481
x' = sx × x
y' = sy × y
```

横纵比例不相同。图像、Polyline、mask 必须使用同一几何映射；mask 采用最近邻。实际实现需统一连续坐标/像素中心约定，并验证往返误差。该策略改变形状比例，应先检查样本效果再确定，不能悄悄宣称保持原图比例。

如果改为“裁剪后等比缩放”或 letterbox，必须显式记录裁剪区域 `(left, top, width, height)`、比例和 padding。映射统一表示为 `x'=sx×(x-left)+pad_x`、`y'=sy×(y-top)+pad_y`，处理图像裁边时还需裁切 Polyline 线段，不能只丢越界节点或压到边界。任何移除图片区域都会改变未标注/不可见语义，需审核政策。

**策略 B：保留原生 3840×2160 数据。** 这需要在独立的官方训练工程中修改 `TrainCollect` 的原始宽高、cache 采样、增强平移尺度、量化/缺失处理、评估坐标及 evaluator 参数，并验证全链路。只写 `train_width=1600`、`train_height=320` 或修改一处原图宽高不够。最终网络输入仍可为 1600×320，但原始标签坐标空间必须一致。本阶段不修改当前 C++ 工程。

两种策略均需额外核对官方训练输入通道顺序与当前 hpc203 C++ 的 BGR 预处理，不能未经对照就认定训练/部署完全一致；当前 C++ 行为保持原样。

## 四个训练槽位与实例不是一回事

当前推理保存 JSON 时会去掉空车道项，因此 `lanes[0]` 未必是模型槽位 0。CVAT 的 `prediction_id=lane_0` 仅表示该帧第一条保存的预测，不能直接成为 mask ID 1。

后续约定槽位从左到右为 0 左外、1 左内、2 右内、3 右外。应按当前行驶车道及几何关系人工确认 `ufld_slot`，每帧槽位唯一，缺失保留为空，不把其余车道向前压缩。模型 row 主分支处理 1/2，col 主分支处理 0/3，语义槽位不一致会破坏监督。无法唯一确定时保持 `unassigned`、不进入训练转换。

超过 4 条真实车道不能静默截断。先保留标注并排除该帧，或后续单独确定明确的 4 槽选线政策；若要改变输出数量，必须同时改变模型、损失、数据和部署，超出当前 Res18 四槽适配范围。

## 审核输入和预留配置

未来转换器的输入是人工导出的 XML、原始 task manifest、原图根目录、分组划分计划和显式几何变换配置。示例见 [ufldv2_dataset_plan.example.json](ufldv2_dataset_plan.example.json)，它只是规划文件，不是可执行转换器。

审核门禁现在即可运行：

```bash
rtk proxy python3 scripts/check_cvat_review.py \
  --annotations outputs/annotations/reviewed/v6453_0001/annotations.xml \
  --task-manifest outputs/annotations/v6453_prelabels/v6453_0001/manifest.json \
  --require-slots
```

全部待转换帧需通过检查；`excluded` 保留审计但不生成监督数据。门禁通过后仍需真正实现几何变换、采样/插值、mask、存在标记和 cache，并使用官方加载器读取真实样本验证；本次均未实施，不能称为训练就绪。

## 训练、验证和独立测试的划分

已在每帧 manifest 和清单中保留 `sequence_id=mcapWriter1_record_20260916-102329`。将来按采集行程/路段 group 划分 train/val/test，并固定种子、分配清单和数据版本；同一组不跨集合，不随机拆相邻帧。

当前只有一个连续采集组，不能自动生成可信的三份独立集合。优先增加其他行程/路段；如果必须按时段分组，需要显式定义片段、隔离间隙并检查场景重叠，仍应说明不是独立采集测试。验证集用于选择训练参数，独立测试集留到最终评估，不把它传给训练中的官方默认 test 流程。官方默认的九种测试分类列表和 1640/590 评估代码也须按自有数据显式接入，不能只改目录名。
