"""
单元测试：对话管理器
"""
import pytest
from unittest.mock import patch, MagicMock


def _clean():
    """清除全局状态"""
    import rightbrain.dialogue as d
    d._pending_new = None
    d._conversation_history.clear()
    d._visual_context = "当前没有识别到明显物体。"


class TestDialogue:
    """对话管理器测试"""

    def test_visual_context(self):
        _clean()
        from rightbrain.dialogue import set_visual_context, get_visual_context
        set_visual_context("检测到苹果")
        assert get_visual_context() == "检测到苹果"

    def test_mark_new_object(self):
        _clean()
        from rightbrain.dialogue import mark_new_object, get_pending_new, mark_as_learned
        marks = {"颜色": "红", "形状": "圆形"}
        mark_new_object(marks)
        pending = get_pending_new()
        assert pending is not None
        assert pending['desc'] == "红圆形"
        assert pending['learned'] is False
        mark_as_learned()
        assert get_pending_new() is None

    def test_ask_what_is_this(self):
        _clean()
        from rightbrain.dialogue import mark_new_object, process_user_speech, set_visual_context
        set_visual_context("红色圆形")
        marks = {"颜色": "红", "形状": "圆形", "大小": "中"}
        mark_new_object(marks)
        reply = process_user_speech("这是什么")
        assert "不知道" in reply

    def test_teach_object(self):
        _clean()
        from rightbrain.dialogue import mark_new_object, process_user_speech
        marks = {"颜色": "红", "形状": "圆形", "大小": "中"}
        mark_new_object(marks)
        reply = process_user_speech("这是苹果")
        assert reply.startswith("__LEARN__:")
        assert "苹果" in reply

    def test_direct_name(self):
        _clean()
        from rightbrain.dialogue import mark_new_object, process_user_speech
        marks = {"颜色": "黄", "形状": "长条形", "大小": "中"}
        mark_new_object(marks)
        reply = process_user_speech("香蕉")
        assert reply.startswith("__LEARN__:")
        assert "香蕉" in reply

    def test_normal_conversation(self):
        _clean()
        from rightbrain.dialogue import process_user_speech, set_visual_context
        set_visual_context("当前主要物体：苹果")
        reply = process_user_speech("你好")
        assert reply is not None
        assert isinstance(reply, str)

    def test_history(self):
        _clean()
        from rightbrain.dialogue import add_to_history, get_history, clear_history
        clear_history()
        add_to_history("user", "你好")
        add_to_history("assistant", "你好")
        assert len(get_history()) == 2
        clear_history()
        assert len(get_history()) == 0

    def test_mark_without_pending(self):
        _clean()
        from rightbrain.dialogue import process_user_speech
        reply = process_user_speech("这是什么")
        assert reply is not None
        assert len(reply) > 0
