# 标注流程验收记录

日期：2026-11-16。未重新运行模型或修改正常运行的 C++ 推理工程；未开始训练。

| 验收项 | 实际结果 |
| --- | --- |
| 真实 JSON 结构检查 | PASS；原图路径、宽高、独立 lanes 点序均复用 |
| 全量转换 | PASS；9,033 帧、16,602 条独立 Polyline、755,229 点、19 个任务 |
| 原图尺寸核对 | PASS；读取实际图片元数据，与 3840×2160 JSON 一致；不缩放坐标 |
| 不把空预测作为背景真值 | PASS；4,004 张空预测仍带 pending 标签；默认训练可用帧 0 |
| 原始 JSON 不变 | PASS；manifest 中所有 9,033 份 SHA-256 与原文件一致 |
| 官方导入核心验证 | PASS；执行本机 CVAT 原始 CvatExtractor、load_anno 和 match_dm_item，所有帧/线/点一致；逆序帧编号匹配通过 |
| 实际服务端导入和回导 | PASS；CVAT 2.83.1，新任务 #1，5 张真实图、10 条线、412 个点、5 个 Tag；导入 worker、数据库核对及导出 worker 均成功 |
| 回导审核状态 | PASS；仍全为 pending；再次审核检查返回 5 个错误、退出 1（预期拒绝），没有自动变成真值 |
| 边界/保护测试 | PASS；19 项 unittest |
| 当前 C++/CMake/启动诊断脚本 | PASS；git diff 为空，未修改 |
| 人工标注实际修正 | NOT TESTED；需要用户逐帧完成，提供了操作说明和独立保存目录 |
| 浏览器 UI 自动化 | NOT TESTED；Chrome 页面控制异常，使用后端管理测试客户端完成真实服务验证，未读取登录凭据 |
| 全量 19 个任务上传服务 | NOT EXECUTED；交付可导入 XML 与原图清单，真实服务仅创建 5 图验证任务，避免自动复制整批图片 |
| CULane 数据集和微调 | NOT IMPLEMENTED / NOT STARTED；仅完成官方代码研究、规划接口和审核门禁 |

真实验证任务：[http://localhost:8080/tasks/1](http://localhost:8080/tasks/1)，名称 `UFLDv2_import_validation_1c8fed2c`。只创建这一新任务，未修改其他用户任务或账户，未把预测标记为人工审核。

验证方式说明：服务端测试在已有 CVAT 容器内通过 DRF 管理测试客户端运行，使用现有本地管理员的授权测试上下文，调用真实 API 视图、实际数据库和后台 worker。它不是浏览器操作，也不验证一般用户的登录流程。导出的 ZIP 由 nginx X-Accel-Redirect 发送，验证程序读取该已授权响应指明的导出文件完成回导比对。

## 文件与证据

- 使用说明：`README_ANNOTATION.md`。
- 全量预标注：`outputs/annotations/v6453_prelabels/`。
- 小样本图片 ZIP：`outputs/annotations/cvat_smoke_sample/images.zip`，5187544 bytes，5 张原图字节和文件名一致。
- 全量核心验证：`outputs/annotations/v6453_import_validation.json`。
- 服务端验证：`outputs/annotations/cvat_live_validation/live_report.json`。
- 服务原始导出：`outputs/annotations/cvat_live_validation/roundtrip.zip`、`roundtrip.xml`；仍是未审核预测。
- 审核拒绝报告：`outputs/annotations/v6453_prelabels_pending_review_check.json`（500 个 pending），`outputs/annotations/cvat_live_validation/unreviewed_gate_check.json`（5 个 pending）。
- 未来数据转换接口：`docs/UFLDV2_DATASET_INTERFACE.md`、`docs/ufldv2_dataset_plan.example.json`。

全量核心验证使用本地 CVAT 源码 `271806df2ec87462dbb9fcdb76323a83182c8ba0` 和独立 Datumaro 1.12.0 环境。该源码版本与真实运行的 2.83.1 服务并非同一构建，因此服务端样本测试独立记录。官方 UFLDv2 研究固定在 `c903880678454dfd9b55a63022368db05c00bc6d`。

## 实际主要命令

```bash
rtk proxy python3 scripts/json_to_cvat.py \
  --json-dir outputs/v6453_front_20260916_102329 \
  --image-root /home/zenglei/Data/v6453/mcapWriter1_record_20260916-102329/sensor_camera_front_mono_image_encode \
  --output-dir outputs/annotations/v6453_prelabels \
  --batch-size 500 --task-prefix v6453 \
  --sequence-id mcapWriter1_record_20260916-102329

rtk proxy build/annotation_validation/venv310/bin/python scripts/validate_cvat_import.py \
  --cvat-source /home/zenglei/cvat \
  --export-root outputs/annotations/v6453_prelabels \
  --report outputs/annotations/v6453_import_validation.json

rtk proxy env PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s tests -p 'test_annotation_pipeline.py' -v
```

上述输出已存在，转换和报告重测请更换目标路径。服务端脚本 `scripts/validate_cvat_live_backend.py` 复制到容器临时目录后，通过 `manage.py shell` 执行；详见使用说明，避免对已存在的用户标注任务执行替换导入。
