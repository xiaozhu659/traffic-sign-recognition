"""把多组实验的训练过程画在一起,对比收敛速度与稳定性。

只看最终准确率会漏掉一个重要信息:残差连接的价值往往不体现在"最终精度更高",
而是"收敛更快、更稳"。所以要把每轮的验证准确率曲线叠在一张图上比。

用法:
    uv run scripts/compare_runs.py --tags baseline no-skip no-augment class-weights
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from ai_image_classification.data import FIGURE_DIR, setup_chinese_font  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "baseline": "基线(18 层,完整配置)",
    "no-skip": "去掉跳跃连接(18 层)",
    "skip34": "带跳跃连接(34 层)",
    "noskip34": "去掉跳跃连接(34 层)",
    "no-augment": "关闭数据增强",
    "class-weights": "类别加权损失",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="多组实验训练曲线对比")
    parser.add_argument("--tags", nargs="+",
                        default=["baseline", "no-skip", "no-augment", "class-weights"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_chinese_font()

    histories = {}
    for tag in args.tags:
        path = PROJECT_ROOT / "outputs" / "runs" / tag / "history.csv"
        if path.exists():
            histories[tag] = pd.read_csv(path)
        else:
            print(f"[跳过] {tag}:找不到 {path}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for tag, history in histories.items():
        name = LABELS.get(tag, tag)
        axes[0].plot(history["epoch"], history["val_acc"] * 100, marker="o",
                     markersize=3, label=name)
        axes[1].plot(history["epoch"], history["val_loss"], marker="o",
                     markersize=3, label=name)

    axes[0].axhline(99, color="gray", linestyle=":", linewidth=1)
    axes[0].set_xlabel("训练轮次")
    axes[0].set_ylabel("验证集准确率(%)")
    axes[0].set_title("验证集准确率收敛过程")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].set_yscale("log")
    axes[1].set_xlabel("训练轮次")
    axes[1].set_ylabel("验证集损失(对数坐标)")
    axes[1].set_title("验证集损失下降过程")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out_path = FIGURE_DIR / "fig_curves_comparison.png"
    fig.savefig(out_path, dpi=130)
    print(f"已保存:{out_path}\n")

    # 收敛速度汇总:第几轮第一次达到 99%
    print(f"{'运行':<15}{'第1轮':>8}{'第3轮':>8}{'第5轮':>8}{'达到99%所需轮次':>16}")
    for tag, history in histories.items():
        reached = history[history["val_acc"] >= 0.99]
        epoch_to_99 = int(reached.iloc[0]["epoch"]) if len(reached) else None
        print(f"{tag:<15}{history['val_acc'][0] * 100:>7.2f}%"
              f"{history['val_acc'][2] * 100:>7.2f}%"
              f"{history['val_acc'][4] * 100:>7.2f}%"
              f"{str(epoch_to_99):>16}")


if __name__ == "__main__":
    main()
