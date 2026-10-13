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
├── docs/                   # 项目说明、实验结果、开发日志
│   └── figures/            # 入库的结果图,供文档引用
├── outputs/                # 图表与统计结果(不入库)
├── scripts/
│   ├── check_env.py        # 环境自检脚本
│   ├── download_data.py    # 数据集下载与校验
│   ├── analyze_data.py     # 类别分布与尺寸统计分析
│   ├── train.py            # 训练入口
│   ├── evaluate.py         # 测试集评估与多组对比
│   ├── compare_runs.py     # 训练曲线对比
│   ├── predict.py          # 推理演示
│   ├── gradcam.py          # 可解释性可视化
│   └── collect_figures.py  # 整理结果图到 docs/figures
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

## 主要结果

六个实验组在官方测试集(12630 张,只在最后评估一次)上的表现:

| 实验 | 测试准确率 | 最稀少 15 类平均 | 最差类别准确率 |
| --- | --- | --- | --- |
| 基线(18 层) | **98.71%** | 98.33% | 75.00% |
| 关闭数据增强 | 97.92% | 94.74% | 66.67% |
| 类别加权损失 | 98.26% | 99.04% | 75.00% |
| 去掉跳跃连接(18 层) | 98.65% | 99.41% | 79.17% |
| 带跳跃连接(34 层) | 97.74% | 98.41% | 75.83% |
| 去掉跳跃连接(34 层) | 97.92% | 92.13% | 50.00% |

![实验对比](docs/figures/test_comparison.png)

![训练曲线对比](docs/figures/curves_comparison.png)

![推理演示](docs/figures/predictions.png)

完整的实验数据、分析与复现方式见 [docs/实验结果.md](docs/实验结果.md),
项目说明与个人感悟见 [docs/项目说明.md](docs/项目说明.md)。
