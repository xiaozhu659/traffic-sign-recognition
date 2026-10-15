"""把关键结果图整理到 docs/figures/ 目录,方便在文档与仓库首页中展示。

为什么不直接把 outputs/ 提交上去
--------------------------------
outputs/ 里放的是每次运行的中间产物:六组实验各自的曲线、混淆矩阵、
每类准确率 CSV 等等,加起来体积不小,而且随时会被下一次训练覆盖。
仓库里只需要保留一组"定稿"的关键图,既能说明结果,又不会让仓库变脏。

所以分工是:
    outputs/        每次运行的工作目录,不入库
    docs/figures/   挑出来的定稿图,入库并在文档中引用

用法:
    uv run scripts/collect_figures.py
"""

import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "outputs"
TARGET_DIR = PROJECT_ROOT / "docs" / "figures"

# 源文件 -> 入库文件名
FIGURES = {
    "samples_grid.png": "samples.png",
    "class_gallery.png": "class_gallery.png",
    "fig_class_distribution.png": "class_distribution.png",
    "fig_image_size.png": "image_size.png",
    "fig_curves_comparison.png": "curves_comparison.png",
    "fig_comparison.png": "test_comparison.png",
    "fig_confusion_baseline.png": "confusion_baseline.png",
    "fig_confusion_skip34.png": "confusion_skip34.png",
    "fig_confusion_noskip34.png": "confusion_noskip34.png",
    "predictions.png": "predictions.png",
    "gradcam.png": "gradcam.png",
}


def main() -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    copied = 0
    missing = []

    for source_name, target_name in FIGURES.items():
        source = SOURCE_DIR / source_name
        if not source.exists():
            missing.append(source_name)
            continue
        shutil.copy2(source, TARGET_DIR / target_name)
        size_kb = source.stat().st_size / 1024
        print(f"  {source_name:32s} -> docs/figures/{target_name:26s} {size_kb:6.0f} KB")
        copied += 1

    print(f"\n已整理 {copied} 张图到 docs/figures/")
    if missing:
        print("以下文件不存在(对应的脚本还没运行过):")
        for name in missing:
            print(f"  {name}")


if __name__ == "__main__":
    main()
