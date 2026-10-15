"""把数据集画成图,直观查看数据长什么样。

用法:
    uv run scripts/preview_samples.py                # 抽样网格(默认 16 张)
    uv run scripts/preview_samples.py --num-samples 24
    uv run scripts/preview_samples.py --all-classes  # 43 个类别全览

输出:
    outputs/samples_grid.png    抽样网格
    outputs/class_gallery.png   43 类全览(加 --all-classes)

注意:抽样时**按类别编号均匀取**,不能直接取前 N 个类别。
类别编号是按标志类型成片排列的(0-9 大多是限速牌、10-19 是禁令标志、
20-31 是三角形警告标志、32-42 是解除限制与指示标志),
只取前 16 个会得到一张几乎全是红圈限速牌的图,完全体现不出数据的多样性。
"""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from ai_image_classification.data import (  # noqa: E402
    TRAIN_DIR,
    load_train_annotations,
    setup_chinese_font,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def one_sample_per_class(annotations: pd.DataFrame, class_ids) -> pd.DataFrame:
    """给定的每个类别各抽一张。"""
    picked = []
    for class_id in class_ids:
        rows = annotations[annotations["class_id"] == class_id]
        picked.append(rows.sample(n=1, random_state=0))
    return pd.concat(picked, ignore_index=True)


def draw_grid(picked: pd.DataFrame, columns: int, title: str, out_path: Path,
              figsize_scale: float = 2.3) -> None:
    rows = math.ceil(len(picked) / columns)
    fig, axes = plt.subplots(rows, columns,
                             figsize=(figsize_scale * columns, figsize_scale * rows + 0.4))
    axes = np.atleast_1d(axes).ravel()

    for ax, (_, row) in zip(axes, picked.iterrows()):
        path = TRAIN_DIR / row["image_path"]
        image = Image.open(path).convert("RGB")
        ax.imshow(image)
        ax.set_title(f"class {row['class_id']:05d}\n{image.width}x{image.height}",
                     fontsize=8)
        ax.axis("off")

    for ax in axes[len(picked):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"已保存:{out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="数据集可视化预览")
    parser.add_argument("--num-samples", type=int, default=16,
                        help="抽样网格的图片数量")
    parser.add_argument("--all-classes", action="store_true",
                        help="额外生成 43 个类别的全览图")
    args = parser.parse_args()

    setup_chinese_font()  # 标题里有中文,不设置字体的话会画成方框
    OUTPUT_DIR.mkdir(exist_ok=True)
    annotations = load_train_annotations()
    all_classes = sorted(annotations["class_id"].unique())
    print(f"数据集共 {len(annotations)} 张图、{len(all_classes)} 个类别")

    # 按类别编号均匀取,保证覆盖到各种类型的标志
    positions = np.linspace(0, len(all_classes) - 1, args.num_samples)
    class_ids = [all_classes[int(round(p))] for p in positions]
    picked = one_sample_per_class(annotations, class_ids)
    print(f"\n抽样 {len(picked)} 张(类别 " + ", ".join(f"{c:02d}" for c in class_ids) + "):")
    for _, row in picked.iterrows():
        print(f"  class {row['class_id']:05d}  {row['Filename']}  "
              f"{row['Width']}x{row['Height']}")

    draw_grid(picked, columns=4,
              title=f"GTSRB 抽样预览:{len(picked)} 张,覆盖全部 43 类中的 "
                    f"{len(class_ids)} 类",
              out_path=OUTPUT_DIR / "samples_grid.png")

    if args.all_classes:
        gallery = one_sample_per_class(annotations, all_classes)
        draw_grid(gallery, columns=7,
                  title="GTSRB 全部 43 个类别各一张",
                  out_path=OUTPUT_DIR / "class_gallery.png",
                  figsize_scale=1.75)


if __name__ == "__main__":
    main()
