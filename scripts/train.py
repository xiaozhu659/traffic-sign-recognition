"""训练入口,实际逻辑在 ai_image_classification.train 里。

用法示例:
    uv run scripts/train.py --tag baseline --epochs 20
    uv run scripts/train.py --tag no-skip --no-skip --epochs 20
"""

from ai_image_classification.train import main

if __name__ == "__main__":
    main()
