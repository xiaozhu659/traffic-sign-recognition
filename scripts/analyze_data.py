"""GTSRB 数据集统计分析:类别分布与图片尺寸。

用法:
    uv run scripts/analyze_data.py

输出:
    outputs/fig_class_distribution.png   类别分布图
    outputs/fig_image_size.png           图片尺寸分布图
    outputs/class_stats.csv              每类统计表
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ai_image_classification.data import (  # noqa: E402
    FIGURE_DIR,
    load_train_annotations,
    setup_chinese_font,
)

INPUT_SIZE = 64  # 计划统一缩放到的边长


def describe(name: str, series) -> None:
    print(f"{name}:均值 {series.mean():6.1f}  中位数 {series.median():6.1f}  "
          f"标准差 {series.std():5.1f}  最小 {series.min()}  最大 {series.max()}")


def plot_class_distribution(annotations, counts) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # 左图:按类别编号排列,直观看出哪些类样本少
    colors = ["#d9534f" if c < counts.mean() / 2 else "#5b9bd5" for c in counts]
    axes[0].bar(counts.index, counts.values, color=colors, width=0.8)
    axes[0].axhline(counts.mean(), color="black", linestyle="--", linewidth=1,
                    label=f"平均 {counts.mean():.0f} 张")
    axes[0].set_xlabel("类别编号")
    axes[0].set_ylabel("训练样本数")
    axes[0].set_title("各类别训练样本数(红色为不足均值一半的类别)")
    axes[0].legend()

    # 右图:从多到少排序,展示长尾
    sorted_counts = counts.sort_values(ascending=False)
    axes[1].bar(range(len(sorted_counts)), sorted_counts.values, color="#5b9bd5")
    axes[1].set_xlabel("类别排名(由多到少)")
    axes[1].set_ylabel("训练样本数")
    axes[1].set_title(f"类别样本数排序:最多 {counts.max()} 张,最少 {counts.min()} 张"
                      f"(相差 {counts.max() / counts.min():.0f} 倍)")

    fig.tight_layout()
    path = FIGURE_DIR / "fig_class_distribution.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"已保存:{path}")


def plot_image_size(annotations) -> None:
    width = annotations["Width"].to_numpy()
    height = annotations["Height"].to_numpy()
    short_side = np.minimum(width, height)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # 左图:宽高散点,点透明度低一点避免重叠成一团
    axes[0].scatter(width, height, s=4, alpha=0.15, color="#5b9bd5")
    axes[0].plot([INPUT_SIZE, INPUT_SIZE], [0, height.max()], color="#d9534f",
                 linestyle="--", label=f"统一缩放目标 {INPUT_SIZE}x{INPUT_SIZE}")
    axes[0].plot([0, width.max()], [INPUT_SIZE, INPUT_SIZE], color="#d9534f",
                 linestyle="--")
    axes[0].set_xlabel("图片宽度(像素)")
    axes[0].set_ylabel("图片高度(像素)")
    axes[0].set_title("宽高散点图:原始尺寸高度不统一")
    axes[0].legend()

    # 右图:短边分布,看有多少图要放大
    axes[1].hist(short_side, bins=40, color="#5b9bd5", edgecolor="white")
    axes[1].axvline(INPUT_SIZE, color="#d9534f", linestyle="--",
                    label=f"缩放目标 {INPUT_SIZE} 像素")
    axes[1].set_xlabel("图片短边长度(像素)")
    axes[1].set_ylabel("图片数量")
    axes[1].set_title("短边长度分布")
    axes[1].legend()

    fig.tight_layout()
    path = FIGURE_DIR / "fig_image_size.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"已保存:{path}")

    return short_side


def main() -> None:
    setup_chinese_font()
    FIGURE_DIR.mkdir(exist_ok=True)

    annotations = load_train_annotations()
    print(f"训练集图片总数:{len(annotations)}")
    print(f"类别数:{annotations['class_id'].nunique()}\n")

    counts = annotations["class_id"].value_counts().sort_index()
    describe("每类样本数", counts)
    imbalance = counts.max() / counts.min()
    print(f"不平衡比例(最多/最少):{imbalance:.1f} 倍")
    print(f"样本数不足均值一半的类别:{int((counts < counts.mean() / 2).sum())} 个\n")

    describe("宽度", annotations["Width"])
    describe("高度", annotations["Height"])

    short_side = np.minimum(annotations["Width"], annotations["Height"])
    need_upscale = (short_side < INPUT_SIZE).mean() * 100
    print(f"\n短边小于 {INPUT_SIZE} 像素、需要放大的图片占比:{need_upscale:.1f}%")
    print(f"用于缩放决策:选择 {INPUT_SIZE}x{INPUT_SIZE} 作为统一输入尺寸\n")

    plot_class_distribution(annotations, counts)
    plot_image_size(annotations)

    stats = annotations.groupby("class_id").agg(
        样本数=("Filename", "size"),
        平均宽度=("Width", "mean"),
        平均高度=("Height", "mean"),
    )
    stats["平均宽度"] = stats["平均宽度"].round(1)
    stats["平均高度"] = stats["平均高度"].round(1)
    csv_path = FIGURE_DIR / "class_stats.csv"
    stats.to_csv(csv_path, encoding="utf-8-sig")
    print(f"已保存:{csv_path}")


if __name__ == "__main__":
    main()
