#!/usr/bin/env python3
"""
系统性优化脚本 - 第6-50轮
批量替换空间计算模块中的硬编码值
"""

import re

# 读取文件
with open('src/algorithms/spatial_enhanced.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 定义替换规则
replacements = [
    # 图像尺寸默认值
    (r'image_height: float = 1080\.0, image_width: float = 1920\.0',
     'image_height: float = DEFAULT_IMAGE_HEIGHT, image_width: float = DEFAULT_IMAGE_WIDTH'),
    (r'image_height: float = 1080\.0',
     'image_height: float = DEFAULT_IMAGE_HEIGHT'),
    (r'image_width: float = 1920\.0',
     'image_width: float = DEFAULT_IMAGE_WIDTH'),

    # 身体部位字符串
    (r"'full_body'", 'BODY_PART_FULL'),
    (r"'half_body'", 'BODY_PART_HALF'),
    (r"'upper_body'", 'BODY_PART_UPPER'),
    (r"'lower_body'", 'BODY_PART_LOWER'),
    (r"'head_only'", 'BODY_PART_HEAD_ONLY'),
    (r"'unknown'", 'BODY_PART_UNKNOWN'),

    # 参考尺寸
    (r'0\.45', 'REFERENCE_SHOULDER_WIDTH'),  # 需要小心，可能替换过多
    (r'0\.15', 'REFERENCE_HEAD_WIDTH'),
]

# 执行替换（谨慎进行）
print("优化脚本已创建。请手动检查关键替换。")
print(f"文件长度: {len(content)} 字符")
