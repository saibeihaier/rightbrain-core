"""
单元测试：决策引擎
"""
import pytest
from unittest.mock import patch, MagicMock


class TestDecision:
    """决策引擎测试"""

    def test_identify_matched(self, mock_experience_memory, mock_marks_red_round):
        """匹配成功时的识别"""
        from rightbrain.decision import identify
        
        mock_experience_memory.add_or_update({
            "condition": {"颜色": "红", "形状": "圆形", "大小": "中"},
            "name": "苹果",
            "action": "可以吃",
            "confidence": 0.9
        })
        
        action, score, exp, is_new = identify(
            mock_marks_red_round, mock_experience_memory,
            confidence_threshold=0.5
        )
        
        assert score > 0.5
        assert exp is not None
        assert is_new is False
        assert "苹果" in action

    def test_identify_unmatched(self, mock_experience_memory, mock_marks_blue_rect):
        """不匹配时的识别"""
        from rightbrain.decision import identify
        
        action, score, exp, is_new = identify(
            mock_marks_blue_rect, mock_experience_memory,
            confidence_threshold=0.9
        )
        
        assert score < 0.5
        assert exp is None

    def test_guess_new_object_with_associations(self):
        """左脑猜测包含联想推理（不依赖Ollama的静态测试）"""
        from rightbrain.decision import guess_new_object
        from rightbrain.learning.memory import ExperienceMemory
        from rightbrain.learning.associative_memory import get_associative_memory
        
        mem = ExperienceMemory()
        # 添加一个经验供联想使用
        mem.add_or_update({
            "condition": {"颜色": "红", "形状": "圆形", "大小": "中"},
            "name": "苹果",
            "action": "可以吃",
            "confidence": 0.9
        })
        
        assoc = get_associative_memory(mem)
        marks = {"颜色": "红", "形状": "圆形", "大小": "中", "纹理": "光滑", "光照": "一般"}
        result = guess_new_object(marks, assoc)
        
        # 联想推理应该在associations字段中产生结果
        assert 'associations' in result
        # 即使Ollama不可用，也应有完整结构
        assert 'guessed_name' in result
        assert 'question' in result
        assert 'exp' in result

    @pytest.mark.timeout(15)
    def test_guess_new_object_fallback_structured(self):
        """左脑猜测在无Ollama时返回结构化空结果"""
        from rightbrain.decision import guess_new_object
        from rightbrain.learning.memory import ExperienceMemory
        from rightbrain.learning.associative_memory import get_associative_memory
        
        mem = ExperienceMemory()
        assoc = get_associative_memory(mem)
        
        marks = {"颜色": "红", "形状": "圆形", "大小": "中", "纹理": "光滑", "光照": "一般"}
        result = guess_new_object(marks, assoc)
        
        assert result is not None
        assert 'guessed_name' in result
        assert 'question' in result
        assert 'associations' in result
        assert 'exp' in result

    def test_feature_hash_uniqueness(self):
        """特征哈希去重测试"""
        from rightbrain.decision import _generate_feature_hash
        
        h1 = _generate_feature_hash({"颜色": "红", "形状": "圆形", "大小": "中", "纹理": "光滑", "光照": "一般"})
        h2 = _generate_feature_hash({"颜色": "红", "形状": "圆形", "大小": "中", "纹理": "光滑", "光照": "一般"})
        h3 = _generate_feature_hash({"颜色": "蓝", "形状": "长方形", "大小": "中", "纹理": "光滑", "光照": "一般"})
        
        assert h1 == h2
        assert h1 != h3

    def test_learning_cache(self):
        """学习缓存生命周期测试"""
        from rightbrain.decision import (
            _mark_learning_start, _mark_learning_complete,
            _is_learning_in_progress, LEARNING_CACHE
        )
        
        marks = {"颜色": "红", "形状": "圆形", "大小": "中", "纹理": "光滑", "光照": "一般"}
        
        _mark_learning_start(marks)
        assert len(LEARNING_CACHE) == 1
        
        in_progress, _ = _is_learning_in_progress(marks)
        assert in_progress is True
        
        _mark_learning_complete(marks, "测试行动")
        assert len(LEARNING_CACHE) == 0
