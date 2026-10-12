# ai-image-classification

基于手写 ResNet 的图像分类项目(灵境竞赛组 · 后端 AI 方向 第一阶段考核)。

## 环境要求

| 项目 | 版本 |
| --- | --- |
| Python | 3.12(.python-version 已固定) |
| 包管理 | [uv](https://docs.astral.sh/uv/) |
| PyTorch | 2.11.0 + CUDA 12.8 |
| 显卡 | NVIDIA RTX 5060 Laptop(sm_120,8 GB 显存) |

PyTorch 固定使用 CUDA 12.8 的官方源(download.pytorch.org/whl/cu128),
配置写在 `pyproject.toml` 的 `[[tool.uv.index]]` 与 `[tool.uv.sources]` 里。
RTX 5060 属于 Blackwell 架构(sm_120),必须用 CUDA 12.8 及以上的构建版本,
否则会出现"能识别显卡但无法运行算子"的问题。

## 搭建步骤

```bash
# 1. 安装 uv(Windows PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. 按锁文件还原环境(会自动下载 Python 3.12 与全部依赖)
uv sync

# 3. 环境自检:验证 CUDA、显卡与各个库
uv run scripts/check_env.py
```

## 目录结构

```
.
├── pyproject.toml          # 依赖与 PyTorch CUDA 源配置
├── uv.lock                 # 锁定精确版本,保证环境可复现
├── .python-version         # 固定 Python 3.12
├── data/                   # 数据集(不入库)
├── docs/                   # 项目计划与学习笔记
├── outputs/                # 图表与统计结果(不入库)
├── scripts/
│   ├── check_env.py        # 环境自检脚本
│   └── download_data.py    # 数据集下载与校验
└── src/
    └── ai_image_classification/
```

## 数据集

采用德国交通标志识别基准 **GTSRB**(German Traffic Sign Recognition Benchmark),
共 43 类交通标志:

- 训练集 26,640 张,测试集 12,630 张,合计 39,270 张
- 图片为 PPM 格式,尺寸不统一(25×25 ~ 243×225),需统一缩放
- 类别不平衡:每类 150 ~ 1500 张,最多与最少相差 10 倍

```bash
uv run scripts/download_data.py
```

脚本会自动完成下载、MD5 校验、解压,并输出类别分布统计到
`outputs/class_distribution.csv`。数据来源:
https://benchmark.ini.rub.de/gtsrb_dataset.html
