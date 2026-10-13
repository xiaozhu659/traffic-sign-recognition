"""推理:对新的交通标志图片做分类,输出 top-3 结果并画图。

这个脚本同时是成果展示用的"演示程序":既可以对官方测试集里没见过的图片预测,
也可以对手机现拍的照片预测。
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

from .data import FIGURE_DIR, TEST_IMAGES_DIR, load_test_annotations, setup_chinese_font
from .dataset import CHANNEL_MEAN, CHANNEL_STD, INPUT_SIZE
from .labels import en, zh  # noqa: E402
from .models import build_resnet, infer_layers


def load_model(checkpoint: Path, device: torch.device, use_skip: bool | None = None):
    """加载检查点。

    网络结构和深度都从权重里自动推断,调用方不需要记住每个检查点是什么配置:
      * 去掉跳跃连接的模型没有 1x1 投影层,看有没有 downsample 就知道;
      * 深度看 layer1~layer4 各有几个残差块。
    否则按默认结构加载会报键名缺失或多余。
    """
    try:
        # 优先用安全模式加载(只允许张量和基本类型)
        state = torch.load(checkpoint, map_location=device, weights_only=True)
    except Exception:
        # 早期版本的检查点里混进了 numpy 标量,安全模式会拒绝,
        # 这里退回普通模式。因为权重都是自己训练出来的,可以信任。
        state = torch.load(checkpoint, map_location=device, weights_only=False)

    if use_skip is None:
        use_skip = any("downsample" in key for key in state["model"])

    layers = infer_layers(state["model"])
    model = build_resnet(layers=layers, num_classes=43, use_skip=use_skip).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, state


def prepare_image(path: Path, device: torch.device) -> torch.Tensor:
    """读入任意尺寸的图片,变成模型需要的张量。"""
    image = Image.open(path).convert("RGB").resize((INPUT_SIZE, INPUT_SIZE))
    array = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.array(CHANNEL_MEAN, dtype=np.float32)
    std = np.array(CHANNEL_STD, dtype=np.float32)
    array = (array - mean) / std
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)  # HWC -> NCHW
    return tensor.to(device)


@torch.no_grad()
def predict(model, tensor: torch.Tensor, topk: int = 3):
    """返回 (topk 类别, topk 概率)。"""
    logits = model(tensor)
    probabilities = F.softmax(logits, dim=1)[0]
    values, indices = probabilities.topk(topk)
    return indices.cpu().numpy(), values.cpu().numpy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="交通标志分类推理演示")
    parser.add_argument("--checkpoint", type=str,
                        default="outputs/runs/baseline/best.pt")
    parser.add_argument("--images", type=str, nargs="*", default=None,
                        help="待预测图片路径;不填则从官方测试集抽样")
    parser.add_argument("--num-samples", type=int, default=9,
                        help="从测试集抽样多少张(默认 9,即 3x3)")
    parser.add_argument("--out", type=str, default=None, help="输出图片路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_chinese_font()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = Path(__file__).resolve().parents[2] / checkpoint
    model, state = load_model(checkpoint, device)
    print(f"已加载权重:{checkpoint}(第 {state.get('epoch')} 轮,"
          f"验证准确率 {state.get('val_acc', 0) * 100:.2f}%)\n")

    cases = []  # (图片路径, 真实类别 或 None)
    if args.images:
        for path in args.images:
            cases.append((Path(path), None))
    else:
        annotations = load_test_annotations()
        sampled = annotations.sample(n=args.num_samples, random_state=1)
        for _, row in sampled.iterrows():
            cases.append((TEST_IMAGES_DIR / row["image_path"], int(row["class_id"])))

    columns = 3
    rows = (len(cases) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 4.6 * rows))
    axes = np.atleast_1d(axes).ravel()

    correct = 0
    labeled = 0
    for ax, (path, true_id) in zip(axes, cases):
        tensor = prepare_image(path, device)
        indices, probabilities = predict(model, tensor)

        predicted = int(indices[0])
        if true_id is not None:
            labeled += 1
            correct += int(predicted == true_id)

        image = Image.open(path).convert("RGB")
        ax.imshow(image)
        ax.axis("off")

        title = f"预测:{zh(predicted)} {probabilities[0] * 100:.1f}%"
        if true_id is not None:
            title += f"\n真实:{zh(true_id)}"
        color = "#1a7f37" if (true_id is None or predicted == true_id) else "#c0392b"
        ax.set_title(title, fontsize=10, color=color)

        detail = "  ".join(
            f"{zh(i)} {p * 100:.1f}%" for i, p in zip(indices, probabilities)
        )
        print(f"{path.name:<22} -> {detail}")
        if true_id is not None and predicted != true_id:
            print(f"     !! 真实标签 {true_id}({en(true_id)})")

    for ax in axes[len(cases):]:
        ax.axis("off")

    if labeled:
        print(f"\n抽样 {labeled} 张,预测正确 {correct} 张,准确率 {correct / labeled * 100:.1f}%")

    fig.suptitle("交通标志识别结果", fontsize=15)
    fig.tight_layout()
    out_path = Path(args.out) if args.out else FIGURE_DIR / "predictions.png"
    fig.savefig(out_path, dpi=130)
    print(f"结果图已保存:{out_path}")


if __name__ == "__main__":
    main()
