"""GTSRB 数据集封装:划分训练/验证集、定义预处理、构建 DataLoader。

PyTorch 的数据加载分三层理解:
    Dataset    负责"取第 i 个样本"(读图、变换、返回 张量+标签)
    Sampler    负责"按什么顺序取"(训练时要打乱)
    DataLoader 负责"批量取、多进程取、提前取"(组装成 batch,后台线程预取)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .data import TEST_IMAGES_DIR, TRAIN_DIR, load_test_annotations, load_train_annotations

INPUT_SIZE = 64

# 在 3000 张训练图上统计得到的通道均值与标准差(已缩放到 64x64、取值 0~1)
# 注意:GTSRB 的照片整体偏暗,均值约 0.33,明显低于 ImageNet 的 0.485,
# 因此不能沿用网上教程里的 ImageNet 归一化参数。
CHANNEL_MEAN = (0.3452, 0.3154, 0.3243)
CHANNEL_STD = (0.2768, 0.2656, 0.2728)


def build_transforms(augment: bool) -> tuple[transforms.Compose, transforms.Compose]:
    """返回 (训练用变换, 验证/测试用变换)。

    交通标志的增强有几个必须回避的坑:
      * 不能做水平翻转——"左转"翻过来就变成"右转",标签直接失效;
      * 不做大角度旋转——真实拍摄的倾斜幅度有限;
      * 适度调整亮度对比度——GTSRB 里有不少逆光、夜晚、过曝的样本。
    """
    normalize = transforms.Normalize(CHANNEL_MEAN, CHANNEL_STD)

    if augment:
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.7, 1.0), ratio=(0.85, 1.18)),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            normalize,
        ])

    eval_tf = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])
    return train_tf, eval_tf


def split_train_val(
    annotations: pd.DataFrame, val_ratio: float = 0.1, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按类别比例划分训练集与验证集(分层抽样)。

    直接随机切会让稀有的类别在验证集里只有个位数甚至零张,
    指标会剧烈跳动,所以这里对每个类别分别按比例抽。
    """
    shuffled = annotations.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val = shuffled.groupby("class_id", group_keys=False).sample(
        frac=val_ratio, random_state=seed
    )
    train = shuffled.drop(index=val.index)
    return train.reset_index(drop=True), val.reset_index(drop=True)


class GTSRBDataset(Dataset):
    """按标注表读取图片。

    初始化时就把 (图片路径, 标签) 展开成两个列表,
    避免每次 __getitem__ 都去查一次 DataFrame——后者在训练循环里是明显的开销。
    """

    def __init__(self, annotations: pd.DataFrame, root: Path, transform):
        self.paths = [root / p for p in annotations["image_path"]]
        self.labels = annotations["class_id"].astype(int).tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        image = Image.open(self.paths[index]).convert("RGB")
        return self.transform(image), self.labels[index]


def build_dataloaders(
    batch_size: int = 128,
    val_ratio: float = 0.1,
    augment: bool = True,
    num_workers: int = 4,
    seed: int = 0,
    limit: int | None = None,
) -> tuple[DataLoader, DataLoader, pd.DataFrame, pd.DataFrame]:
    """构建训练与验证的 DataLoader,同时返回划分好的标注表。"""
    annotations = load_train_annotations()
    train_df, val_df = split_train_val(annotations, val_ratio=val_ratio, seed=seed)

    if limit:
        # 只取一小部分样本,用于快速验证代码是否跑通
        train_df = train_df.groupby("class_id", group_keys=False).head(
            max(1, limit // train_df["class_id"].nunique())
        )
        val_df = val_df.groupby("class_id", group_keys=False).head(
            max(1, limit // 20 // val_df["class_id"].nunique())
        )

    train_tf, eval_tf = build_transforms(augment)
    train_set = GTSRBDataset(train_df, TRAIN_DIR, train_tf)
    val_set = GTSRBDataset(val_df, TRAIN_DIR, eval_tf)

    common = dict(num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, **common)
    return train_loader, val_loader, train_df, val_df


def build_test_loader(batch_size: int = 256, num_workers: int = 4) -> tuple[DataLoader, pd.DataFrame]:
    """构建官方测试集的 DataLoader(只在最终评估时使用)。"""
    annotations = load_test_annotations()
    _, eval_tf = build_transforms(augment=False)
    test_set = GTSRBDataset(annotations, TEST_IMAGES_DIR, eval_tf)
    loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return loader, annotations


def class_counts(annotations: pd.DataFrame) -> np.ndarray:
    """返回每个类别的样本数(长度为类别数的数组),用于类别加权。"""
    counts = annotations["class_id"].value_counts().sort_index()
    return counts.to_numpy()


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
