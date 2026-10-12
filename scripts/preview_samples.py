"""从 GTSRB 中抽样若干图片拼成网格图,直观查看数据长什么样。

用法:
    uv run scripts/preview_samples.py

输出:
    outputs/samples_grid.png
"""

from pathlib import Path

import matplotlib
import pandas as pd
from PIL import Image

matplotlib.use("Agg")  # 无界面环境下绘图
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = PROJECT_ROOT / "data" / "gtsrb" / "GTSRB" / "Training"
OUT_PATH = PROJECT_ROOT / "outputs" / "samples_grid.png"

ROWS, COLS = 4, 4


def load_annotations() -> pd.DataFrame:
    frames = []
    for csv_path in sorted(TRAIN_DIR.rglob("GT-*.csv")):
        class_id = int(csv_path.stem.split("-")[1])
        df = pd.read_csv(csv_path, sep=";")
        df["class_id"] = class_id
        df["class_dir"] = csv_path.parent.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    annotations = load_annotations()

    # 每个大类抽一张,保证网格里出现的类别尽量分散
    picked = annotations.groupby("class_id", group_keys=False).sample(
        n=1, random_state=0
    ).head(ROWS * COLS)
    print(f"从 {annotations['class_id'].nunique()} 个类别中抽取 {len(picked)} 张:")

    fig, axes = plt.subplots(ROWS, COLS, figsize=(9, 9))
    for ax, (_, row) in zip(axes.ravel(), picked.iterrows()):
        path = TRAIN_DIR / row["class_dir"] / row["Filename"]
        img = Image.open(path).convert("RGB")
        ax.imshow(img)
        ax.set_title(f"class {row['class_id']:05d}\nimg {img.width}x{img.height}",
                     fontsize=9)
        ax.axis("off")
        print(f"  class {row['class_id']:05d}  {row['Filename']}  {img.width}x{img.height}")

    fig.suptitle("GTSRB training samples", fontsize=14)
    fig.tight_layout()
    OUT_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(OUT_PATH, dpi=130)
    print(f"\n网格图已保存:{OUT_PATH}")


if __name__ == "__main__":
    main()
