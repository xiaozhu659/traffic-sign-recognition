"""测试集评估入口,实际逻辑在 ai_image_classification.evaluate 里。

用法示例:
    uv run scripts/evaluate.py --tags baseline no-skip class-weights no-augment
"""

from ai_image_classification.evaluate import main

if __name__ == "__main__":
    main()
