"""手写 ResNet:自己实现残差块与网络主体,不调用 torchvision 的现成模型。

核心思想
--------
普通卷积网络堆到几十层后,训练误差反而变大——不是过拟合,而是梯度在反向传播
过程中被反复相乘,越传越小(梯度消失),浅层的参数几乎不更新。

ResNet 的做法是给主干加一条"直通车道":

        x ──────────────── identity(恒等映射)────────────┐
        │                                                 ↓
        └─→ conv → BN → ReLU → conv → BN ────────────→ (+) ─→ ReLU → 输出

于是这一块要学的不再是完整的映射 H(x),而是残差 F(x) = H(x) - x,
输出变成 y = F(x) + x。好处有三点:
  1. 反向传播时梯度可以沿着 identity 这条加法支路原样传回去,不会连乘衰减;
  2. 如果某一层不需要起作用,网络只要把 F(x) 学成 0 就是恒等映射,退化成"什么都不做",
     比学一个恒等映射容易得多;
  3. 加法要求形状完全一致,所以当通道数或尺寸变化时,需要用 1x1 卷积把 identity
     投影到同样的形状——这就是下面 downsample 分支的作用。
"""

import torch
from torch import nn


class BasicBlock(nn.Module):
    """基础残差块:两个 3x3 卷积 + 一条跳跃连接。"""

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_skip: bool = True,
    ) -> None:
        super().__init__()
        self.use_skip = use_skip

        # 第一个卷积负责改变尺寸(stride)和通道数,第二个卷积只做通道内的特征提取
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 维度匹配:只有当"输入形状"和"输出形状"不一致时才需要投影
        # - 通道数变化(如 64 -> 128):1x1 卷积把通道数对齐
        # - 空间尺寸变化(stride=2):1x1 卷积同时用同样的 stride 下采样
        # 如果两者都不变,identity 可以直接相加,不需要任何额外参数
        needs_projection = use_skip and (stride != 1 or in_channels != out_channels)
        if needs_projection:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_skip:
            if self.downsample is not None:
                identity = self.downsample(x)
            out = out + identity  # 关键的一行:残差相加

        return self.relu(out)


class ResNet(nn.Module):
    """由 4 个 stage 组成的 ResNet。

    layers=[2, 2, 2, 2] 即 ResNet-18;改成 [3, 4, 6, 3] 就是 ResNet-34。
    """

    def __init__(
        self,
        num_classes: int = 43,
        layers: tuple[int, int, int, int] = (2, 2, 2, 2),
        base_channels: int = 64,
        use_skip: bool = True,
    ) -> None:
        super().__init__()
        self.use_skip = use_skip
        self.in_channels = base_channels

        # 开头不用 ImageNet 版本的 7x7 卷积 + 最大池化:那种设计会把输入先降采样 4 倍,
        # 我们是 64x64 的小图,再降两次就没剩多少信息了,所以只用一个 3x3 卷积。
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        # 每隔一个 stage 用 stride=2 把特征图尺寸减半
        self.layer1 = self._make_layer(base_channels, layers[0], stride=1)
        self.layer2 = self._make_layer(base_channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(base_channels * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(base_channels * 8, layers[3], stride=2)

        self.pool = nn.AdaptiveAvgPool2d(1)  # 任意尺寸进来都变成 1x1
        self.fc = nn.Linear(base_channels * 8, num_classes)

        # 权重初始化:残差网络对初始化比较敏感,最后一层 BN 的 gamma 置 0,
        # 相当于每个残差块初始时接近恒等映射,训练更稳
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for s in strides:
            blocks.append(BasicBlock(self.in_channels, out_channels, stride=s,
                                     use_skip=self.use_skip))
            self.in_channels = out_channels
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# 常见深度的层数配置:第一个 stage 重复 2 次就是 ResNet-18,3/4/6/3 就是 ResNet-34
RESNET_DEPTHS = {
    18: (2, 2, 2, 2),
    34: (3, 4, 6, 3),
}


def build_resnet(
    layers: tuple[int, int, int, int] = (2, 2, 2, 2),
    num_classes: int = 43,
    use_skip: bool = True,
) -> ResNet:
    """按给定的层数配置构造网络。"""
    return ResNet(num_classes=num_classes, layers=layers, use_skip=use_skip)


def build_resnet18(num_classes: int = 43, use_skip: bool = True) -> ResNet:
    """构造 ResNet-18;use_skip=False 时得到去掉跳跃连接的对照组。"""
    return build_resnet(RESNET_DEPTHS[18], num_classes=num_classes, use_skip=use_skip)


def build_resnet34(num_classes: int = 43, use_skip: bool = True) -> ResNet:
    """构造 ResNet-34,用于验证"网络越深,残差连接越重要"这一判断。"""
    return build_resnet(RESNET_DEPTHS[34], num_classes=num_classes, use_skip=use_skip)


def infer_layers(state_dict) -> tuple[int, int, int, int]:
    """从权重里推断出网络深度配置。

    保存检查点时并不知道以后会拿它做什么,所以加载时最好自己判断结构:
    数一数 layer1~layer4 各有几个残差块,就能还原出 (2,2,2,2) 还是 (3,4,6,3)。
    """
    counts = []
    for stage in (1, 2, 3, 4):
        indices = {
            int(key.split(".")[1])
            for key in state_dict
            if key.startswith(f"layer{stage}.")
        }
        counts.append(len(indices))
    return tuple(counts)  # type: ignore[return-value]


def count_parameters(model: nn.Module) -> int:
    """统计可训练参数总量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
