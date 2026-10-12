"""Grad-CAM:看看模型到底是"看"图片的哪个部位做出判断的。

原理(三句话)
------------
1. 取某一层卷积的输出特征图(最后一组残差块最合适,既保留了空间位置信息,
   又有足够的语义层级);
2. 对目标类别的分数做反向传播,得到每个特征图上的梯度——梯度大小代表
   "这个位置稍微变一点,分数会变化多少",也就是这个位置有多重要;
3. 把梯度在空间上求平均当作权重,对特征图加权求和,再经过 ReLU 只保留正向贡献,
   就得到一张热力图。

它的价值在于排查模型是不是"蒙对的":如果热力图集中在标志牌本身,说明模型学到了
正确的东西;如果集中在背景、天空、水印上,那就说明模型走了捷径。
"""

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None

        # 前向钩子:把这一层的输出存下来
        target_layer.register_forward_hook(self._save_activation)
        # 反向钩子:把这一层的梯度存下来
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, tensor: torch.Tensor, class_id: int | None = None):
        """返回 (类别编号, 归一化到 0~1 的热力图, 形状 [H, W])。"""
        self.model.zero_grad(set_to_none=True)
        logits = self.model(tensor)

        if class_id is None:
            class_id = int(logits.argmax(dim=1).item())

        score = logits[0, class_id]
        score.backward()

        # 梯度形状 [1, C, h, w] -> 每个通道求平均,得到 [C]
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, h, w]
        cam = F.relu(cam)

        # 上采样回输入尺寸,并归一化到 0~1 方便画图
        cam = F.interpolate(cam, size=tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return class_id, cam


def denormalize(tensor: torch.Tensor, mean, std) -> np.ndarray:
    """把归一化后的张量还原成可以直接显示的图片数组。"""
    array = tensor[0].detach().cpu().permute(1, 2, 0).numpy()
    array = array * np.array(std, dtype=np.float32) + np.array(mean, dtype=np.float32)
    return np.clip(array, 0, 1)
