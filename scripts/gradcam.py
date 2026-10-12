"""生成 Grad-CAM 热力图,验证模型关注的区域是否合理。

用法:
    uv run scripts/gradcam.py --checkpoint outputs/runs/baseline/best.pt
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from ai_image_classification.data import (  # noqa: E402
    FIGURE_DIR,
    TEST_IMAGES_DIR,
    load_test_annotations,
    setup_chinese_font,
)
from ai_image_classification.dataset import CHANNEL_MEAN, CHANNEL_STD  # noqa: E402
from ai_image_classification.gradcam import GradCAM, denormalize  # noqa: E402
from ai_image_classification.labels import zh  # noqa: E402
from ai_image_classification.predict import load_model, prepare_image  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grad-CAM 可视化")
    parser.add_argument("--checkpoint", type=str, default="outputs/runs/baseline/best.pt")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_chinese_font()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = Path(__file__).resolve().parents[1] / checkpoint
    model, _ = load_model(checkpoint, device)

    cam_engine = GradCAM(model, model.layer4[-1])

    annotations = load_test_annotations()
    # 每张图都挑一个不同的类别,避免四张全是限速牌
    picked = annotations.groupby("class_id", group_keys=False).sample(
        n=1, random_state=args.seed
    ).sample(n=args.num_samples, random_state=args.seed)

    fig, axes = plt.subplots(2, args.num_samples, figsize=(3.6 * args.num_samples, 7.6))
    axes = np.atleast_2d(axes)

    for column, (_, row) in enumerate(picked.iterrows()):
        path = TEST_IMAGES_DIR / row["image_path"]
        tensor = prepare_image(path, device)
        class_id, cam = cam_engine(tensor)
        image = denormalize(tensor, CHANNEL_MEAN, CHANNEL_STD)

        axes[0, column].imshow(image)
        axes[0, column].set_title(f"真实:{zh(row['class_id'])}\n预测:{zh(class_id)}",
                                  fontsize=9)
        axes[0, column].axis("off")

        axes[1, column].imshow(image)
        axes[1, column].imshow(cam, cmap="jet", alpha=0.45)
        axes[1, column].set_title("Grad-CAM 关注区域", fontsize=9)
        axes[1, column].axis("off")

        print(f"{path.name:<22} 真实 {row['class_id']:2d} 预测 {class_id:2d}  {zh(class_id)}")

    fig.suptitle("模型判断依据可视化(Grad-CAM)", fontsize=15)
    fig.tight_layout()
    out_path = Path(args.out) if args.out else FIGURE_DIR / "gradcam.png"
    fig.savefig(out_path, dpi=130)
    print(f"已保存:{out_path}")


if __name__ == "__main__":
    main()
