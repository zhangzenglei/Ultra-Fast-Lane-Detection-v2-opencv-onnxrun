# UFLDv2 自有数据预标注与人工修正

本阶段读取已有推理 JSON，生成 CVAT 1.1 图像标注。没有重新推理，没有修改 C++、CMake、ORT/CUDA 或原始预测 JSON，没有启动 PyTorch 训练。模型继续使用 CULane ResNet18。

## 已生成的交付物

项目根目录下：

```text
outputs/annotations/
├── v6453_prelabels/                         # 全量自动预标注，仅供人工修正
│   ├── labels.json                         # CVAT Raw 标签定义
│   ├── manifest.json                       # 原图/JSON/哈希/采集组对应表
│   ├── summary.json
│   ├── v6453_0001/                         # 每个任务最多 500 张
│   │   ├── annotations.xml                 # 原图像素坐标的独立 Polyline
│   │   ├── labels.json
│   │   ├── manifest.json
│   │   ├── images.txt                      # 原始图片绝对路径，每行一张
│   │   ├── images/                         # 指向原图的软链接，无图片复制
│   │   └── review_status.csv               # 任务初始待审核清单
│   └── ... v6453_0019/                     # 最后一批 33 张
├── cvat_smoke_sample/                      # 建议首次导入使用，5 张真实图片
│   ├── images.zip                         # 约 5.2 MB，原始 JPG 字节不变
│   └── import_smoke_0001/                  # XML、labels、manifest、图片链接
├── v6453_import_validation.json            # 全量官方解析器验证报告
├── cvat_smoke_import_validation.json
├── v6453_prelabels_pending_review_check.json
└── cvat_live_validation/                   # 真实 CVAT 2.83.1 导入/回导
    ├── live_report.json
    ├── roundtrip.zip
    └── roundtrip.xml                       # 仍为未审核预测，不是人工真值
```

已转换 **9,033 张图、16,602 条独立车道线、755,229 个点**，分成 19 个任务。4,004 张图没有自动检测到车道线，这些图片仍在任务中，且均标为 `pending`，不代表背景真值。

`outputs/` 已被原有 Git 规则忽略。自动预测、CVAT 预标注、人工修正后的标注应分别存放：

```text
outputs/v6453_front_20260916_102329/          # 原始推理 JSON/结果图，保留不动
outputs/annotations/v6453_prelabels/         # 本次生成的未审核预标注，保留不动
outputs/annotations/reviewed/v6453_0001/      # 您后续从 CVAT 导出的人工修正结果
```

不要把人工导出的文件写回前两个目录。`review_status.csv` 是初始清单，不会随 CVAT 自动同步；审核状态以人工导出的 XML 中 `frame_review` 标签为准。

## 可重复执行的批量转换命令

需要 Python 3.10+ 和 Pillow；本机已有，不依赖 Python ONNX Runtime、PyTorch 或 GPU。

```bash
rtk proxy python3 scripts/json_to_cvat.py \
  --json-dir outputs/v6453_front_20260916_102329 \
  --image-root /home/zenglei/Data/v6453/mcapWriter1_record_20260916-102329/sensor_camera_front_mono_image_encode \
  --output-dir outputs/annotations/v6453_prelabels \
  --batch-size 500 \
  --task-prefix v6453 \
  --sequence-id mcapWriter1_record_20260916-102329
```

上述命令已执行。重复转换请使用新的 `--output-dir`，程序拒绝覆盖已有目录。默认只读取 `*_result.json`，因此不会把 `process.json`、`summary.json` 当预测数据。其他命名可用 `--pattern` 明确选择。

规则：优先匹配 `source_image` 的真实路径；源路径迁移时允许唯一同名图片匹配并记录匹配方法；没有 `source_image` 时支持按原有 `图片.jpg_result.json` 或 `stem_result.json` 匹配。重名不猜测，多份预测对应同一原图报错。默认要求原图目录内每张图片都有预测 JSON；只有有意转换子集时才加 `--allow-missing-predictions`，遗漏图片会记录在报告中。

Pillow 读取实际原图尺寸并对照 JSON，拒绝不明确的 EXIF 旋转。每点必须为有限数值，满足 `0 <= x < width`、`0 <= y < height`；每条 Polyline 至少两个不同点。预测超过 4 条、短线、非法点和尺寸不一致均报错，不自动裁边、插点、丢弃车道线或猜测缩放。允许直接使用原始 3840×2160 坐标，没有裁 ROI 或重做预处理。

每个 `lanes[i]` 写为一个独立 `<polyline>`，保持原点序及坐标；`group_id=0` 的含义是“不分组”，不是所有车道属于一个实例。`prediction_id=lane_i` 只保留该帧内的实例对应，不表示训练槽位或跨帧跟踪 ID。

## 第一次在 CVAT 中导入

### 服务前提与本机检查结果

已检查 `~/cvat`：源码提交为 `271806df2ec87462dbb9fcdb76323a83182c8ba0`，描述为 `v2.74.0-300-g271806df2`。首次检查时服务未运行；后续用户环境中 CVAT 服务已启动。宿主环境实际访问 `http://localhost:8080/api/server/about` 返回 200，运行版本 **2.83.1**。

已经新建并保留 [验证任务 #1](http://localhost:8080/tasks/1)（`UFLDv2_import_validation_1c8fed2c`）：5 张图、10 条 Polyline、5 个待审核 Tag。通过真实服务的导入/导出 worker、数据库和 API 视图完成导入与回导比对，412 个点坐标和审核属性均保留。现有用户任务未修改。您可以直接打开该任务核对效果；全量 19 批尚未上传到 CVAT，以免自动复制全部原图。

该测试通过后端容器内的 DRF 本地管理测试客户端执行，复用现有管理员账户的测试上下文，不读取密码/令牌，也不改变账户。Chrome 控制连接有异常，未完成浏览器页面自动化；本次服务端实际导入/回导的成功不能写成浏览器 UI 操作验证。报告见 `outputs/annotations/cvat_live_validation/live_report.json`。如果服务运行在其他机器，应使用其真实地址；本地文件软链接不等于服务端共享盘。

### 先使用 5 张真实图片的小任务

1. 在 CVAT Tasks 页面创建一个新的图像任务，例如 `ufld_import_smoke`。不要选已有人工标注任务：CVAT 的上传标注操作会替换其现有标注。
2. Labels 切到 **Raw**，粘贴 `outputs/annotations/cvat_smoke_sample/import_smoke_0001/labels.json` 的完整数组。两个标签都要保留：`lane`（Polyline）和 `frame_review`（Tag），包括全部属性。若创建在 Project 下，先在项目中配置相同标签。
3. 上传 `outputs/annotations/cvat_smoke_sample/images.zip`。压缩包根目录就是 5 个原始 JPG 文件，不要上传推理结果图，也不要重命名图片。保留原图分辨率和全部帧，不设置筛帧、子采样或其他图像裁剪。
4. 待任务图片处理完成，在该任务的菜单选择 **Upload annotations**，导入 `import_smoke_0001/annotations.xml`。格式选 **CVAT 1.1**（该导入器接收 images 1.1 XML；有的界面显示相近的 CVAT images 名称）。不要选择 COCO、YOLO 或把 XML 当任务备份导入。
5. 打开任务核对 5 张图：每张应有 `frame_review.status=pending`；预测车道数分别覆盖 0、1、2、3、4。PolyLine 为独立可选对象；没有预测的那一张仍需检查并补标，不能直接通过。
6. 检查图像大小为 3840×2160，线条叠加在原始照片上；服务端已实际导出 **CVAT for images 1.1** 并核对文件名、尺寸和坐标。建议您在该验证任务中再目视确认显示效果，再上全量任务。

CVAT 原生 XML 支持 Polyline、Tag 和属性，格式与操作可查 [CVAT 官方格式文档](https://docs.cvat.ai/docs/dataset_management/formats/format-cvat/)、[创建任务](https://docs.cvat.ai/docs/workspace/tasks-page/)及[导入标注](https://docs.cvat.ai/docs/manual/advanced/import-datasets/)。

### 导入每批 500 张的任务

以 `v6453_0001` 为例：新建对应任务，使用该目录 `labels.json`；上传该目录 `images/` 中列出的全部原图，再上传同目录的 `annotations.xml`。文件选择器如不支持软链接，先给该任务打包：

```bash
rtk proxy python3 scripts/package_cvat_images.py \
  --task-dir outputs/annotations/v6453_prelabels/v6453_0001 \
  --output outputs/annotations/v6453_0001_images.zip
```

图片 ZIP 使用原始文件名和原始压缩字节，不重新编码；不是标注 ZIP。不要用系统 `zip -y` 打包软链接，那可能只存链接文本。本脚本直接从 manifest 中的原图路径读取真实图片。

全量图片约 7.2 GiB，本机当前剩余空间有限；不要一次打包所有批次或复制整个数据集。按任务打包、上传，或把 ZIP 放到有足够空间的磁盘。程序不会自动删除任何原图、预测或上传包。

## 人工修正与审核规则

在每张图中检查每条可见车道线，而不仅检查已有的绿色预测线：

1. 选择独立 `lane` Polyline，放大图像，拖动偏移节点使其沿车道线中心；必要时添加/删除节点，保持沿线连续的点序，避免线条来回折返。
2. 删除路沿、树影、车辆边缘等误检；一条实际车道线对应一个 Polyline，不把全部车道连成一条，也不要把虚线的每个短划分别作为车道实例。
3. 对漏检车道使用 Polyline 手工补标，标签仍为 `lane`。新对象的 `origin` 默认为 `manual`；原预测修正后的对象可保留 `origin=prediction` 和 `prediction_id` 作为来源记录。
4. 遮挡处只沿有证据的连续几何关系标注；无法确认的帧保持待审核或排除，不凭模型失败推断“没有车道”。原图中其他区域仍需人工检查；本次 XML 没有把 ROI 外区域定义为背景。
5. 本次任务只标车道几何，不推断黄/白、实/虚线语义。CVAT 中的 `occluded` 默认值只是导入字段，不能视为模型判断结果，按实际情况人工修正。
6. 全帧检查完，编辑该帧唯一的 **`frame_review` Tag**，填写 `reviewer` 并设置：

| status | 含义 |
| --- | --- |
| pending | 默认值；仍需检查，禁止用于监督训练 |
| approved_positive | 已人工修正并确认 1～4 条车道线 |
| approved_negative | 人工检查整张图后确认没有目标车道线；此时必须没有 lane 对象 |
| excluded | 不可可靠标注、模糊、歧义或超出当前模型约束；不进训练/验证/测试 |

**CVAT Job 的 Completed/Accepted 状态不能替代逐帧 `frame_review` 标签。** 特别是 4,004 张空预测，必须补标或逐帧明确确认为负样本；不能批量默认通过。

一帧实际有超过 4 条应标注车道时，先保留真实标注并标为 `excluded`，待后续明确 4 槽位选线政策；不要为满足 Res18 输出约束而静默删除真值。`ufld_slot` 初始都是 `unassigned`，预测数组下标不能替代语义槽位。后续做训练格式转换前，需要明确并填写唯一的槽位 0～3（左外、左内、右内、右外的约定见接口文档），不确定时先不赋值。

Polyline 编辑操作参见 [CVAT 官方 Polyline 标注说明](https://docs.cvat.ai/docs/manual/advanced/annotation-with-polylines/)。

## 保存人工修正结果并检查

1. 在 CVAT 保存修改，导出 **CVAT for images 1.1**，通常得到 ZIP。
2. 将导出的原始 ZIP 原样保留到新的 `outputs/annotations/reviewed/<任务名>/` 目录。解压出的 `annotations.xml` 也保存在这里；不要复制回自动预标注或推理结果目录。
3. 使用该任务原先的 manifest 核对图片覆盖、尺寸、独立车道、审核状态和坐标：

```bash
rtk proxy python3 scripts/check_cvat_review.py \
  --annotations outputs/annotations/reviewed/v6453_0001/annotations.xml \
  --task-manifest outputs/annotations/v6453_prelabels/v6453_0001/manifest.json \
  --report outputs/annotations/reviewed/v6453_0001/review_audit.json
```

未审核、丢帧、尺寸变化、非法点、正负样本与对象数量矛盾或缺少 reviewer，均返回非零。以后做训练格式转换前再加 `--require-slots` 检查 0～3 槽位唯一性。当前检查器不会导出 CULane，不会执行训练；检查 PASS 也不能代替实际的人工作业质量。

## 实际验证和边界

- 19 项 `unittest` 全部通过：实例/点序保留、空预测保持 pending、非法/NaN/布尔坐标、短线、实际尺寸不符、路径迁移、重名歧义、重复预测、缺 JSON、不同扩展名同 stem、超过四条、禁止覆盖、审核/槽位约束、重复训练槽位、超过四条人工车道排除但不截断、丢帧、原图 ZIP 字节一致性。
- 对全量 19 个任务，运行 `~/cvat` 原始代码中的 **`CvatExtractor`、`load_anno`、`match_dm_item`**。AST 选取原始定义执行，未修改其解析和匹配逻辑；数据库写入层使用内存任务适配器。Datumaro 1.12.0 提供真实图像及 Polyline 对象。此验证覆盖本次 Polyline/Tag 子集，不声称验证 Skeleton 或所有 CVAT 格式。
- 官方任务匹配函数在逆序任务帧编号下仍正确按文件名关联；原始 XML id 不被当作唯一关联依据。
- 验证全量 9,033 帧、16,602 条线、755,229 个点，逐个与源 JSON 对比，所有预测 JSON 的 SHA-256 不变。按每任务的预测车道数类别抽样实际解码原图，共 86 张图像解码检查；全量尺寸则已在转换时检查。
- 5 图导入样本覆盖 0～4 条车道，10 个独立 Polyline、412 个点。实际打包并检查 ZIP CRC、文件名及图片 SHA-256，全部通过。
- 把第一批未经审核的 500 帧送入审核检查器，得到 `valid=False`、500 个 pending 错误、退出 1；这是预期保护结果。
- **真实服务端导入/数据库/回导 PASS**：CVAT 2.83.1 任务 #1，5 张图、10 条线、412 个点、5 个 pending 标签，原图尺寸与坐标完全一致；属性 `origin=prediction`、`ufld_slot=unassigned` 保留。自动 Job Completed/审核真值没有被设置。导出的 roundtrip XML 仍是未审核预测，不是人工修正数据。
- 浏览器 UI 自动操作与用户目视审核：**NOT TESTED**。全量报告 `v6453_import_validation.json` 本身是离线官方核心验证，其 `live_cvat_server_import=NOT TESTED` 只描述该报告的范围；补做的真实服务验证另见 `cvat_live_validation/live_report.json`。

可重复运行测试：

```bash
rtk proxy env PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_annotation_pipeline.py' -v
```

官方解析器验证依赖放在 `build/annotation_validation/venv310/`，不影响系统或 C++ 环境。已安装的验证依赖为 `datumaro==1.12.0`；CVAT 当前固定依赖是另一个 fork，其源码安装因缺 Rust 未成功，本次没有为此安装系统工具链。使用已发布轮子提供 Polyline/Tag 所需运行对象，直接运行本机 CVAT 原始解析代码。重建验证环境时应明确指定 `/usr/bin/python3`（3.10），不要让 uv 默认选其他 Python：

```bash
rtk proxy uv --cache-dir build/annotation_validation/uv_cache venv \
  --python /usr/bin/python3 --system-site-packages build/annotation_validation/venv310
rtk proxy uv --cache-dir build/annotation_validation/uv_cache pip install \
  --python build/annotation_validation/venv310/bin/python --only-binary=:all: datumaro==1.12.0

# 报告路径应使用尚不存在的新文件名
rtk proxy build/annotation_validation/venv310/bin/python scripts/validate_cvat_import.py \
  --cvat-source /home/zenglei/cvat \
  --export-root outputs/annotations/v6453_prelabels \
  --report outputs/annotations/v6453_import_validation_repeat.json
```

重建 venv 时请使用新的目录名，或复用已存在的环境，不覆盖其他工程。上述依赖安装仅用于可选的导入验证；日常 JSON 转换及审核检查只需要现有 Python/Pillow。

服务端验证脚本为 `scripts/validate_cvat_live_backend.py`，只用于已有 CVAT 容器中的 `manage.py shell` 管理测试上下文。输入目录含 5 图 `images.zip`、`labels.json`、`annotations.xml`；`UFLD_VALIDATION_DIR` 指定该临时目录，`UFLD_VALIDATION_USERNAME` 指定已存在且被授权使用的本地账户。脚本只新建有唯一名称的验证任务，并保留任务作为交付，不修改其他任务或账户。设置 `UFLD_VALIDATION_RESUME=1` 时仅核对报告中自己创建的同名同所有者任务，不重复上传或替换其标注。不要将这个管理脚本作为互联网登录或一般用户认证流程。验证阶段的容器临时目录是 `/tmp/ufldv2_annotation_validation_20261116`，结果已复制回项目输出目录。

## 后续训练格式接口

见 [UFLDv2/CULane 转换接口约定](docs/UFLDV2_DATASET_INTERFACE.md) 和 [规划配置示例](docs/ufldv2_dataset_plan.example.json)。本次只提供接口与审核门禁；CVAT→CULane 实际转换、分割 mask/缓存生成、分组划分执行及 PyTorch 微调均未实施。
