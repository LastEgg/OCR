## 简介
这是一个基于PyTorch Lightning构建的文档矫正与文档增强的训练框架，用于复现各种文档矫正与增强的论文，以及一些属于自己的创新.  

## 主要特性
- 使用 PyTorch Lightning 构建，代码结构清晰
- 支持 Weights & Biases (wandb) 实验跟踪
- 灵活的配置系统，支持 YAML 配置和命令行参数
- 支持模型断点保存和恢复
- 支持混合精度训练
- 自动记录训练指标

## 项目结构
- `README.md`：项目的总体说明文档。
- `models/`：包含模型文件。
- `tools/`：包含数据集、LightningModule构建、Loss函数定义等
  - `datasets/`：数据集处理
  - `losses/`：自定义loss
  - `pl_tools/`：LightningModule构建
- `test_dir/`：用于存放测试数据与输出内容。
- `train_*.py`：用于启动训练
- `predict_*.py`：用于推理 custom data
- `export_*.py`：用于转换ONNX
- `configs/config.md`：[config.md](configs/config.md)
- `down_dataset.py`：用于下载M2E数据集

## 项目清单
- [ ] 新增export_*.py
- [ ] 整理ExpRate计算代码：test.py
- [ ] 实现LAST的 KV cache推理
- [ ] 实现LAST的 KV cache版本的训练（引入hugging face的transformer源码），comming soon

## 已有方法
| **方法名称** | **配置文件** | **数据集** | **备注** | **任务类型** |
| --- | --- | --- | --- | --- |
| LAST | config_last_m2edataset.yaml | Doc3D | <ul><li>进行中</li></ul> | OCR |



