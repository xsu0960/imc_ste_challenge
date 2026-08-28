# Release Package Guide

**Author:** Xin Su

该压缩包只包含源码、配置、测试、正式报告、报告插图和用于重建关键表格的审计 CSV。公开数据集、训练缓存、JSONL 全量日志和 checkpoint 均未打包。

发布目录同时提供 `imc_ste_documents_release.zip`。该归档只包含 README、正式报告、论文、实际引用的图表、可编辑流程图源文件和规范化结果表，不包含源码与测试。

## Recommended Verification

在已安装 CUDA 驱动的 Linux 环境中：

```bash
conda env create -f environment.yml
conda activate imc-ste
python scripts/verify_project.py --package-mode
```

该命令执行以下检查：

1. 导入全部运行依赖。
2. 在 CPU 上运行 dataset-free noisy/STE/online-profile smoke。
3. 运行 48 项单元测试。
4. 从包内审计 CSV 重建关键结果表和主结果图。
5. 校验正式报告、理论附录、图表和数据文件的 SHA-256。

若机器已预装兼容的 PyTorch/CUDA，也可使用轻量虚拟环境：

```bash
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --no-deps -e .
python scripts/verify_project.py --package-mode
```

## Full Training

完整训练和 checkpoint 评估命令位于 `README.md`。数据集不会随提交包分发；训练脚本会下载 CIFAR/TinyImageNet，VOC optional 需要按 README 准备公开数据。正式报告数值可直接由 `scripts/build_key_results.py` 重建，但从头训练仍需要相应数据、GPU 时间和随机种子。

## Scope

- 必选主结果使用统一 `noise_scale=1.0`。
- TinyImageNet 是经出题方确认的 ImageNet 方向替代任务。
- VOC 检测/分割属于 optional 机制验证，不冒充 COCO/Cityscapes 官方口径。
- 包内不包含 checkpoint，以控制体积并保持发布内容明确。
