"""
单元测试：配置模块
"""
import pytest
import os


class TestConfig:
    """配置系统测试"""

    def test_default_values(self):
        """默认配置值"""
        from rightbrain.config import Config
        assert Config.DEBUG_MODE is True
        assert Config.CONFIDENCE_THRESHOLD == 0.6
        assert Config.OLLAMA_BASE == "http://127.0.0.1:11434"
        assert Config.YOLO_CLASS_MAPPING["apple"] == "苹果"
        assert Config.FEATURE_WEIGHTS["颜色"] == 0.35

    def test_env_override(self, monkeypatch):
        """环境变量覆盖"""
        monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.8")
        monkeypatch.setenv("DEBUG_MODE", "false")
        
        from rightbrain.config import Config
        Config.load_from_env()
        
        assert Config.CONFIDENCE_THRESHOLD == 0.8
        assert Config.DEBUG_MODE is False
        
        # 恢复
        Config.CONFIDENCE_THRESHOLD = 0.6
        Config.DEBUG_MODE = True

    def test_update_method(self):
        """Config.update() 方法"""
        from rightbrain.config import Config
        Config.update(CONFIDENCE_THRESHOLD=0.75)
        assert Config.CONFIDENCE_THRESHOLD == 0.75
        
        # 不影响其他值
        assert Config.CONFIDENCE_DELTA == 0.1
        
        # 恢复
        Config.CONFIDENCE_THRESHOLD = 0.6

    def test_yolo_mapping(self):
        """YOLO 类别映射完整性"""
        from rightbrain.config import Config
        
        # 常见的 YOLO 类别应该都在
        common_classes = ['cell phone', 'bottle', 'person', 'book', 'cup', 'laptop', 'chair', 'cat', 'dog']
        for cls in common_classes:
            assert cls in Config.YOLO_CLASS_MAPPING, f"Missing YOLO class: {cls}"

    def test_get_logger(self):
        """日志获取"""
        from rightbrain.config import get_logger, Config
        
        logger = get_logger("test_logger")
        assert logger is not None
        assert logger.name == "test_logger"
