"""
单元测试：经验记忆模块
"""
import pytest
import json
import time
from unittest.mock import patch, MagicMock


class TestExperienceMemory:
    """ExperienceMemory 核心功能测试"""

    def test_init_empty(self, mock_experience_memory):
        """空初始化测试"""
        from rightbrain.learning.memory import ExperienceMemory
        mem = ExperienceMemory()
        assert mem.experiences == []

    def test_add_experience(self, mock_experience_memory):
        """添加经验测试"""
        exp = {
            "condition": {"颜色": "红", "形状": "圆形", "大小": "中"},
            "name": "苹果",
            "action": "这是苹果，可以吃",
            "confidence": 0.8
        }
        result = mock_experience_memory.add_or_update(exp)
        assert result is True
        assert len(mock_experience_memory.experiences) == 1

    def test_match_exact(self, mock_experience_memory, mock_marks_red_round):
        """精确匹配测试"""
        exp = {
            "condition": {"颜色": "红", "形状": "圆形", "大小": "中"},
            "name": "苹果",
            "action": "这是苹果，可以吃",
            "confidence": 0.8
        }
        mock_experience_memory.add_or_update(exp)

        score, action, best_exp = mock_experience_memory.match(
            mock_marks_red_round, threshold=0.5
        )
        assert best_exp is not None
        assert score > 0.5
        assert best_exp["name"] == "苹果"

    def test_no_match(self, mock_experience_memory, mock_marks_blue_rect):
        """不匹配测试"""
        exp = {
            "condition": {"颜色": "红", "形状": "圆形"},
            "name": "苹果",
            "action": "这是苹果，可以吃",
            "confidence": 0.8
        }
        mock_experience_memory.add_or_update(exp)

        score, action, best_exp = mock_experience_memory.match(
            mock_marks_blue_rect, threshold=0.8
        )
        # 蓝色矩形 vs 红色圆形，高阈值下不匹配
        assert best_exp is None or score < 0.5

    def test_save_and_load(self, tmp_path):
        """加载/保存测试"""
        from rightbrain.learning.memory import ExperienceMemory
        
        mem = ExperienceMemory()
        mem.add_or_update({
            "condition": {"颜色": "黄", "形状": "长条形"},
            "name": "香蕉",
            "action": "可以剥皮吃",
            "confidence": 0.9
        })
        
        save_path = tmp_path / "test_save.json"
        mem.save(str(save_path))
        assert save_path.exists()
        
        # 重新加载
        mem2 = ExperienceMemory(str(save_path))
        assert len(mem2.experiences) == 1
        assert mem2.experiences[0]["name"] == "香蕉"

    def test_update_confidence(self, mock_experience_memory):
        """置信度更新测试"""
        condition = {"颜色": "红", "形状": "圆形"}
        mock_experience_memory.add_or_update({
            "condition": condition,
            "name": "测试",
            "action": "测试",
            "confidence": 0.8
        })
        
        mock_experience_memory.update_confidence(condition, success=False, delta=0.1)
        new_conf = mock_experience_memory.get_confidence(condition)
        assert new_conf == pytest.approx(0.7)  # 0.8 - 0.1

    def test_shape_similar(self):
        """形状相似度测试"""
        from rightbrain.learning.memory import _shape_similar
        assert _shape_similar('圆形', '圆形') is True
        assert _shape_similar('圆形', '椭圆形') is True
        assert _shape_similar('正方形', '长方形') is True
        assert _shape_similar('圆形', '三角形') is False

    def test_decay_unused(self, mock_experience_memory):
        """遗忘机制测试"""
        from rightbrain.learning.memory import FORGET_THRESHOLD_DAYS, FORGET_DECAY_RATE
        
        # 添加一个很久以前的经验
        exp = {
            "condition": {"颜色": "红", "形状": "圆形"},
            "name": "旧经验",
            "action": "旧的",
            "confidence": 0.8,
            "last_matched_time": time.time() - (FORGET_THRESHOLD_DAYS + 1) * 86400
        }
        mock_experience_memory.experiences.append(exp)
        
        count = mock_experience_memory.decay_unused_confidences(
            threshold_days=FORGET_THRESHOLD_DAYS,
            decay_rate=FORGET_DECAY_RATE
        )
        assert count == 1
        assert mock_experience_memory.experiences[0]["confidence"] == pytest.approx(0.7)  # 0.8 - 0.1
