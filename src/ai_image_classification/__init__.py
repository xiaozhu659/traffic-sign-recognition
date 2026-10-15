"""交通标志识别:基于手写 ResNet 的图像分类项目。

模块分工:
    data      数据集路径与标注读取
    dataset   数据管道(增强策略、训练验证划分、DataLoader)
    models    手写残差块与 ResNet
    train     训练与评估流程
    evaluate  测试集评估与多组实验对比
    predict   推理演示
    gradcam   可解释性可视化
    labels    43 个类别的中英文名称

各功能的命令行入口在 scripts/ 目录下,例如:
    uv run scripts/train.py --tag baseline --epochs 20
    uv run scripts/evaluate.py --tags baseline no-skip --num-workers 0
"""


def main() -> None:
    """控制台入口:打印模块说明与可用的命令。"""
    print(__doc__)


if __name__ == "__main__":
    main()
