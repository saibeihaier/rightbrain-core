"""
pytest conftest - 全局 fixtures 和 mock 配置
"""
import sys
import os
from pathlib import Path

# 确保 src 在路径中
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / 'src'
sys.path.insert(0, str(SRC_DIR))

import pytest
import numpy as np


@pytest.fixture
def sample_red_circle():
    """生成红色圆形测试图像"""
    import cv2
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(img, (100, 100), 60, (0, 0, 255), -1)
    return img


@pytest.fixture
def sample_blue_rect():
    """生成蓝色长方形测试图像"""
    import cv2
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(img, (30, 30), (170, 100), (255, 0, 0), -1)
    return img


@pytest.fixture
def mock_marks_red_round():
    """红色圆形物体的特征标记"""
    return {
        '颜色': '红',
        '形状': '圆形',
        '大小': '中',
        '纹理': '光滑',
        '光照': '一般',
        '距离': '近',
        '边缘距离': '中',
        '感受': '中性',
        '情感联想': '',
        '愉悦度': 0.5,
        'has_face': False,
        'is_handheld': False,
    }


@pytest.fixture
def mock_marks_blue_rect():
    """蓝色矩形物体的特征标记"""
    return {
        '颜色': '蓝',
        '形状': '长方形',
        '大小': '中',
        '纹理': '光滑',
        '光照': '一般',
        '距离': '近',
        '边缘距离': '中',
        '感受': '中性',
        '情感联想': '',
        '愉悦度': 0.5,
        'has_face': False,
        'is_handheld': False,
    }


@pytest.fixture
def mock_experience_memory(tmp_path):
    """空的经验记忆实例（使用临时文件）"""
    from rightbrain.learning.memory import ExperienceMemory
    exp_file = tmp_path / "test_experiences.json"
    exp_file.write_text('[]')
    mem = ExperienceMemory()
    mem.json_path = str(exp_file)
    return mem
