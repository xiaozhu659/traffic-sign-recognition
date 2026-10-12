"""环境自检脚本:确认 PyTorch、CUDA、GPU 与常用库都可用。

用法:
    uv run scripts/check_env.py
"""

import platform
import sys


def main() -> None:
    print("=" * 56)
    print("Python  :", sys.version.split()[0], f"({platform.architecture()[0]})")
    print("系统    :", platform.platform())
    print("解释器  :", sys.executable)

    import numpy as np
    import pandas as pd
    import matplotlib
    import PIL
    import torch
    import torchvision

    print("-" * 56)
    print("numpy       :", np.__version__)
    print("pandas      :", pd.__version__)
    print("matplotlib  :", matplotlib.__version__)
    print("pillow      :", PIL.__version__)
    print("torch       :", torch.__version__)
    print("torchvision :", torchvision.__version__)
    print("编译时 CUDA :", torch.version.cuda)

    print("-" * 56)
    available = torch.cuda.is_available()
    print("CUDA 可用   :", available)
    if not available:
        print("!! 没有检测到可用 GPU,请检查驱动和 PyTorch 版本")
        return

    print("GPU 数量    :", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i} 名称 : {props.name}")
        print(f"  计算能力  : sm_{props.major}{props.minor}")
        print(f"  显存      : {props.total_memory / 1024**3:.1f} GB")
    print("cuDNN       :", torch.backends.cudnn.version())

    # 真正跑一次前向 + 反向,确认不是"能看见却用不了"
    device = torch.device("cuda:0")
    x = torch.randn(8, 3, 64, 64, device=device)
    conv = torch.nn.Conv2d(3, 16, 3, padding=1).to(device)
    y = conv(x)
    loss = y.pow(2).mean()
    loss.backward()
    torch.cuda.synchronize()
    print("-" * 56)
    print("GPU 前向/反向测试通过,输出形状:", tuple(y.shape))
    print("显存占用    :", f"{torch.cuda.memory_allocated() / 1024**2:.1f} MB")
    print("=" * 56)


if __name__ == "__main__":
    main()
