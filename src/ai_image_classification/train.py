"""训练与评估流程。

设计要点
--------
* 训练集再切出 10% 作为验证集(分层抽样),官方测试集只在最后评估时使用;
* 按验证集准确率保存最佳权重,而不是用最后一个 epoch 的权重;
* 用 CSV 记录每个 epoch 的指标,训练结束后自动画曲线图。
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from .data import FIGURE_DIR, setup_chinese_font  # noqa: E402
from .dataset import build_dataloaders, class_counts  # noqa: E402
from .models import build_resnet18, count_parameters  # noqa: E402


def train_one_epoch(model, loader, criterion, optimizer, device) -> tuple[float, float]:
    """跑完一轮训练,返回(平均损失, 准确率)。"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * targets.size(0)
        correct += (outputs.argmax(dim=1) == targets).sum().item()
        total += targets.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes: int = 43):
    """在验证/测试集上评估,返回(损失, 准确率, 每类准确率)。"""
    model.eval()
    total_loss = 0.0
    total = 0
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * targets.size(0)
        total += targets.size(0)

        predictions = outputs.argmax(dim=1)
        for t, p in zip(targets.cpu().numpy(), predictions.cpu().numpy()):
            confusion[t, p] += 1

    accuracy = np.trace(confusion) / confusion.sum()
    per_class_total = confusion.sum(axis=1)
    per_class_acc = np.divide(
        np.diag(confusion),
        per_class_total,
        out=np.zeros(num_classes),
        where=per_class_total > 0,
    )
    return total_loss / total, accuracy, per_class_acc


def make_class_weights(counts: np.ndarray, device) -> torch.Tensor:
    """按类别频率的倒数给每个类别加权,让稀有类别在损失里更"值钱"。"""
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(weights)  # 归一化,保持整体损失量级
    return torch.tensor(weights, dtype=torch.float32, device=device)


def plot_curves(history: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].plot(history["epoch"], history["train_loss"], label="训练损失")
    axes[0].plot(history["epoch"], history["val_loss"], label="验证损失")
    axes[0].set_xlabel("轮次")
    axes[0].set_ylabel("损失")
    axes[0].set_title(f"{title} — 损失曲线")
    axes[0].legend()

    axes[1].plot(history["epoch"], history["train_acc"] * 100, label="训练准确率")
    axes[1].plot(history["epoch"], history["val_acc"] * 100, label="验证准确率")
    axes[1].set_xlabel("轮次")
    axes[1].set_ylabel("准确率(%)")
    axes[1].set_title("准确率曲线")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练手写 ResNet 做交通标志分类")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None,
                        help="只取少量样本快速验证流程")
    parser.add_argument("--tag", type=str, default="baseline",
                        help="本次运行的名称,结果存到 outputs/runs/<tag>/")
    parser.add_argument("--no-skip", action="store_true",
                        help="去掉跳跃连接(消融实验用)")
    parser.add_argument("--class-weights", action="store_true",
                        help="使用类别加权损失(针对类别不平衡)")
    parser.add_argument("--no-augment", action="store_true",
                        help="关闭数据增强")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = FIGURE_DIR / "runs" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"设备:{device}")
    print(f"运行名称:{args.tag}  输出目录:{out_dir}")

    train_loader, val_loader, train_df, val_df = build_dataloaders(
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        augment=not args.no_augment,
        num_workers=args.num_workers,
        seed=args.seed,
        limit=args.limit,
    )
    print(f"训练样本 {len(train_df)} 张,验证样本 {len(val_df)} 张,"
          f"训练批次数 {len(train_loader)}")

    model = build_resnet18(num_classes=43, use_skip=not args.no_skip).to(device)
    print(f"模型参数量:{count_parameters(model) / 1e6:.2f} M"
          f"{'(已去掉跳跃连接)' if args.no_skip else ''}")

    if args.class_weights:
        weights = make_class_weights(class_counts(train_df), device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        print("已启用类别加权损失")
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    best_acc = 0.0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion,
                                                optimizer, device)
        val_loss, val_acc, per_class_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - epoch_start,
        })
        print(f"[{epoch:3d}/{args.epochs}] 训练损失 {train_loss:.4f} 准确率 {train_acc * 100:5.2f}%"
              f" | 验证损失 {val_loss:.4f} 准确率 {val_acc * 100:5.2f}%"
              f" | {history[-1]['seconds']:.1f}s")

        if val_acc > best_acc:
            best_acc = val_acc
            # 注意:这里显式转成 Python 原生类型再保存。
            # PyTorch 2.6 起 torch.load 默认使用 weights_only=True,只信任基本类型;
            # 如果直接塞 numpy 标量(比如 val_acc 是 np.float64),加载时会被拒绝。
            torch.save({
                "model": model.state_dict(),
                "epoch": int(epoch),
                "val_acc": float(val_acc),
                "use_skip": not args.no_skip,
            }, out_dir / "best.pt")

    history_df = pd.DataFrame(history)
    history_df.to_csv(out_dir / "history.csv", index=False, encoding="utf-8-sig")

    setup_chinese_font()
    plot_curves(history_df, out_dir / "curves.png", args.tag)

    # 记录本次运行的配置与最终结果,便于后续对比
    summary = {
        "tag": args.tag,
        "use_skip": not args.no_skip,
        "augment": not args.no_augment,
        "class_weights": args.class_weights,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "best_val_acc": best_acc,
        "final_val_acc": history[-1]["val_acc"],
        "total_seconds": time.time() - start,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # 每类准确率单独存下来,方便分析哪些类别拖后腿
    pd.DataFrame({
        "class_id": range(len(per_class_acc)),
        "accuracy": per_class_acc,
        "samples": class_counts(val_df),
    }).to_csv(out_dir / "per_class_accuracy.csv", index=False, encoding="utf-8-sig")

    print(f"\n最佳验证准确率:{best_acc * 100:.2f}%")
    print(f"总耗时:{summary['total_seconds'] / 60:.1f} 分钟")
    print(f"结果目录:{out_dir}")


if __name__ == "__main__":
    main()
