"""下载并校验 GTSRB 交通标志数据集,同时输出类别分布统计。

数据来源:德国交通标志识别基准(GTSRB)
https://benchmark.ini.rub.de/gtsrb_dataset.html

用法:
    uv run scripts/download_data.py

目录结构(与 torchvision.datasets.GTSRB 的约定一致):
    data/gtsrb/GTSRB/Training/<类别号>/*.ppm    训练集
    data/gtsrb/GTSRB/Final_Test/Images/*.ppm    测试集图片
    data/gtsrb/GT-final_test.csv                测试集标签
"""

import hashlib
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "gtsrb"

BASE_URL = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/"

# 文件名 -> 官方 MD5,用于校验下载是否完整
FILES = {
    "GTSRB-Training_fixed.zip": "513f3c79a4c5141765e10e952eaa2478",
    "GTSRB_Final_Test_Images.zip": "c7e4e6327067d32654124b0fe9e82185",
    "GTSRB_Final_Test_GT.zip": "fe31e9c9270bbcd7b84b7f21a9d9d9e5",
}


def md5_of(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path) -> None:
    """流式下载并打印进度。"""
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        with dest.open("wb") as fh:
            while chunk := resp.read(1 << 20):
                fh.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {dest.name}: {done / 1e6:6.1f} / {total / 1e6:.1f} MB"
                          f" ({done / total * 100:5.1f}%)", end="", flush=True)
    print()


def ensure_data() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    for name, expected_md5 in FILES.items():
        zip_path = DATA_ROOT / name
        if zip_path.exists() and md5_of(zip_path) == expected_md5:
            print(f"[跳过] {name} 已存在且校验通过")
        else:
            print(f"[下载] {name}")
            download(BASE_URL + name, zip_path)
            actual = md5_of(zip_path)
            if actual != expected_md5:
                zip_path.unlink()
                raise RuntimeError(f"{name} 校验失败:期望 {expected_md5},实际 {actual}")
            print(f"  校验通过 {actual}")

        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(DATA_ROOT)
        print(f"  已解压到 {DATA_ROOT}")


def report() -> None:
    from torchvision.datasets import GTSRB

    train = GTSRB(root=str(PROJECT_ROOT / "data"), split="train")
    test = GTSRB(root=str(PROJECT_ROOT / "data"), split="test")
    print(f"\n训练集样本数:{len(train)}")
    print(f"测试集样本数:{len(test)}")

    # 直接读标注文件统计分布,比逐张加载图片快得多
    frames = []
    for csv_path in sorted((DATA_ROOT / "GTSRB" / "Training").rglob("GT-*.csv")):
        class_id = int(csv_path.stem.split("-")[1])
        df = pd.read_csv(csv_path, sep=";")
        df["class_id"] = class_id
        frames.append(df)
    annotations = pd.concat(frames, ignore_index=True)

    print(f"标注总行数:{len(annotations)}")
    print(f"图片尺寸:{annotations['Width'].min()}x{annotations['Height'].min()}"
          f" ~ {annotations['Width'].max()}x{annotations['Height'].max()}")

    counts = annotations["class_id"].value_counts().sort_index()
    print(f"\n类别数:{counts.size}")
    print(f"每类样本数:最多 {counts.max()},最少 {counts.min()},平均 {counts.mean():.1f},"
          f"最多/最少 = {counts.max() / counts.min():.1f} 倍")

    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    counts.rename("count").to_csv(out_dir / "class_distribution.csv", encoding="utf-8-sig")
    print("类别分布已保存到 outputs/class_distribution.csv")


if __name__ == "__main__":
    ensure_data()
    report()
