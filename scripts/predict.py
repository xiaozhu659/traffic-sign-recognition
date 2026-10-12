"""推理演示入口,实际逻辑在 ai_image_classification.predict 里。

用法示例:
    uv run scripts/predict.py                                  # 从测试集抽 9 张
    uv run scripts/predict.py --images D:\\照片\\sign1.jpg      # 预测自己的图片
"""

from ai_image_classification.predict import main

if __name__ == "__main__":
    main()
