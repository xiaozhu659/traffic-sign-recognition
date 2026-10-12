"""GTSRB 数据集的读取与统计工具。

把"数据集在哪里、标注怎么读"这类事情集中在一个模块里,
其他脚本直接 import,避免每个脚本各写一遍。
"""

from pathlib import Path

import pandas as pd

# __file__ 是 .../src/ai_image_classification/data.py,向上三级才是项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "gtsrb"
TRAIN_DIR = DATA_ROOT / "GTSRB" / "Training"
TEST_IMAGES_DIR = DATA_ROOT / "GTSRB" / "Final_Test" / "Images"
TEST_GT_CSV = DATA_ROOT / "GT-final_test.csv"

# matplotlib 输出的图片统一放这里
FIGURE_DIR = PROJECT_ROOT / "outputs"


def load_train_annotations() -> pd.DataFrame:
    """读取全部训练集标注,返回一张汇总表。

    返回的 DataFrame 每行是一张图片,包含:
        Filename / Width / Height / ROI 坐标 / ClassId / class_id / image_path
    """
    frames = []
    for csv_path in sorted(TRAIN_DIR.rglob("GT-*.csv")):
        class_id = int(csv_path.stem.split("-")[1])
        df = pd.read_csv(csv_path, sep=";")
        df["class_id"] = class_id
        # 拼接相对路径,方便后续拼出图片的绝对路径
        df["image_path"] = csv_path.parent.name + "/" + df["Filename"]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_test_annotations() -> pd.DataFrame:
    """读取测试集标注(同一格式,但只有一条汇总 CSV)。"""
    df = pd.read_csv(TEST_GT_CSV, sep=";")
    df["class_id"] = df["ClassId"]
    df["image_path"] = df["Filename"]
    return df


def setup_chinese_font() -> None:
    """让 matplotlib 能正常显示中文,否则会画出一堆方框。"""
    import matplotlib

    matplotlib.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False
