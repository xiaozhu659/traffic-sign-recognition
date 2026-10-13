"""在官方测试集上评估模型,并支持多组实验结果对比。

为什么不能只看验证集准确率
--------------------------
验证集是从训练集里切出来的,模型虽然没见过,但它和训练数据来自同一批拍摄条件。
官方测试集有 12630 张,拍摄场景更杂、更暗、更模糊,能更真实地反映模型水平。
另外验证集只有 2664 张,准确率很容易顶到 100% 而失去区分度,
这时候测试集上的细微差别才是真正有信息量的地方。
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from .data import FIGURE_DIR, setup_chinese_font
from .dataset import build_test_loader
from .labels import NUM_CLASSES
from .predict import load_model


@torch.no_grad()
def collect_predictions(model, loader, device) -> np.ndarray:
    """跑一遍测试集,返回混淆矩阵:confusion[真实类别, 预测类别]。"""
    model.eval()
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        predictions = model(images).argmax(dim=1).cpu().numpy()
        for t, p in zip(targets.numpy(), predictions):
            confusion[t, p] += 1
    return confusion


def per_class_accuracy(confusion: np.ndarray) -> np.ndarray:
    total = confusion.sum(axis=1)
    return np.divide(np.diag(confusion), total, out=np.zeros(NUM_CLASSES),
                     where=total > 0)


def plot_confusion(confusion: np.ndarray, tag: str, out_dir: Path) -> None:
    """画归一化混淆矩阵(按行归一,每行代表该类别的召回率)。"""
    total = confusion.sum(axis=1, keepdims=True)
    normalized = np.divide(confusion, total, out=np.zeros_like(confusion, dtype=float),
                           where=total > 0)

    fig, ax = plt.subplots(figsize=(11, 9.5))
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set_xlabel("预测类别编号")
    ax.set_ylabel("真实类别编号")
    ax.set_title(f"{tag} — 测试集混淆矩阵(按行归一,对角线越蓝越好)")
    fig.colorbar(image, ax=ax, fraction=0.045)
    fig.tight_layout()
    path = out_dir / f"fig_confusion_{tag}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  混淆矩阵:{path}")


def plot_comparison(results: list[dict], per_class: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    tags = [r["tag"] for r in results]
    palette = ["#5b9bd5", "#f0a04b", "#7fb069", "#c0392b", "#8e6bbf", "#d98cb3"]
    colors = palette[: len(tags)]

    # 左:整体测试准确率
    axes[0].bar(tags, [r["test_acc"] * 100 for r in results], color=colors)
    axes[0].set_ylabel("测试集准确率(%)")
    axes[0].set_title("整体准确率对比")
    axes[0].tick_params(axis="x", rotation=15)
    for i, r in enumerate(results):
        axes[0].text(i, r["test_acc"] * 100 + 0.05, f"{r['test_acc'] * 100:.2f}",
                     ha="center", fontsize=9)

    # 中:稀有类别的平均准确率。
    # 注意:这里的编号必须按样本数排序取出来,不能写死成 0-14——
    # 类别 0-14 里混着样本最多的限速牌,拿它们当"稀有类别"会得出完全错误的结论。
    rare_ids = list(per_class["samples"].sort_values().head(15).index)
    axes[1].bar(tags, [r["rare_acc"] * 100 for r in results], color=colors)
    axes[1].set_ylabel("准确率(%)")
    axes[1].set_title("样本最少的 15 个类别平均准确率")
    axes[1].tick_params(axis="x", rotation=15)
    for i, r in enumerate(results):
        axes[1].text(i, r["rare_acc"] * 100 + 0.05, f"{r['rare_acc'] * 100:.2f}",
                     ha="center", fontsize=9)

    # 右:每个类别的准确率曲线,看差距出在哪里
    for tag, color in zip(tags, colors):
        axes[2].plot(per_class.index, per_class[tag] * 100, marker="o", markersize=3,
                     label=tag, color=color)
    axes[2].set_xlabel("类别编号")
    axes[2].set_ylabel("准确率(%)")
    axes[2].set_title("逐类别准确率对比")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    path = out_dir / "fig_comparison.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"对比图:{path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="测试集评估与多组实验对比")
    parser.add_argument("--tags", nargs="+", default=["baseline"],
                        help="outputs/runs/<tag>/best.pt 里的运行名称,可传多个")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--plot-only", action="store_true",
                        help="不加载模型,直接用上次评估保存的 CSV 重画对比图")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_chinese_font()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = Path(__file__).resolve().parents[2]

    if args.plot_only:
        # 只重画图:模型评估很慢,改了画图逻辑没必要重跑一遍
        results = pd.read_csv(FIGURE_DIR / "comparison.csv").to_dict("records")
        per_class_df = pd.read_csv(FIGURE_DIR / "per_class_accuracy_all.csv", index_col=0)
        plot_comparison(results, per_class_df, FIGURE_DIR)
        return

    test_loader, annotations = build_test_loader(num_workers=args.num_workers)
    print(f"测试集共 {len(annotations)} 张,类别数 {NUM_CLASSES}\n")

    counts = annotations["class_id"].value_counts().sort_index().to_numpy()
    rare_ids = np.argsort(counts)[:15]  # 样本最少的 15 个类别

    results = []
    per_class_table = {}
    for tag in args.tags:
        checkpoint = project_root / "outputs" / "runs" / tag / "best.pt"
        if not checkpoint.exists():
            print(f"[跳过] {tag}:找不到 {checkpoint}")
            continue

        model, _ = load_model(checkpoint, device)
        confusion = collect_predictions(model, test_loader, device)
        accuracy = np.trace(confusion) / confusion.sum()
        acc_per_class = per_class_accuracy(confusion)
        rare_acc = acc_per_class[rare_ids].mean()

        results.append({
            "tag": tag,
            "test_acc": accuracy,
            "rare_acc": rare_acc,
            "min_class_acc": acc_per_class.min(),
        })
        per_class_table[tag] = acc_per_class

        print(f"{tag:15s} 测试准确率 {accuracy * 100:6.2f}%  "
              f"最稀少 15 类平均 {rare_acc * 100:6.2f}%  "
              f"最差类别 {acc_per_class.min() * 100:5.2f}%")
        plot_confusion(confusion, tag, FIGURE_DIR)

    if not results:
        return

    per_class_df = pd.DataFrame(per_class_table)
    per_class_df.index.name = "class_id"
    per_class_df["samples"] = counts
    per_class_df.to_csv(FIGURE_DIR / "per_class_accuracy_all.csv", encoding="utf-8-sig")

    pd.DataFrame(results).to_csv(FIGURE_DIR / "comparison.csv", index=False,
                                 encoding="utf-8-sig")
    plot_comparison(results, per_class_df, FIGURE_DIR)

    print("\n对比汇总:")
    print(pd.DataFrame(results).to_string(index=False,
                                          float_format=lambda x: f"{x * 100:.2f}%"))


if __name__ == "__main__":
    main()
