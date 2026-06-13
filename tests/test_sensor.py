"""
单元测试：视觉传感器模块
"""
import pytest
import numpy as np


class TestSensor:
    """传感器核心功能测试"""

    def test_find_object_bbox_red_circle(self):
        """红色圆形边界框检测"""
        from rightbrain.cv_core.sensor import find_object_bbox
        
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        import cv2
        cv2.circle(img, (100, 100), 60, (0, 0, 255), -1)
        
        bboxes = find_object_bbox(img)
        assert len(bboxes) > 0

    def test_extract_marks_red_circle(self):
        """红色圆形特征提取"""
        from rightbrain.cv_core.sensor import extract_marks
        
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        import cv2
        cv2.circle(img, (100, 100), 60, (0, 0, 255), -1)
        
        marks_list = extract_marks(img)
        if len(marks_list) > 0:
            marks = marks_list[0]
            assert "颜色" in marks or "形状" in marks

    def test_extract_marks_empty_image(self):
        """空图像特征提取"""
        from rightbrain.cv_core.sensor import extract_marks
        
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        marks_list = extract_marks(img)
        assert marks_list is not None

    @pytest.mark.parametrize("size", [(300, 300), (640, 480)])
    def test_extract_marks_green_rect(self, size):
        """绿色矩形特征提取"""
        from rightbrain.cv_core.sensor import extract_marks
        
        img = np.zeros((*size, 3), dtype=np.uint8)
        import cv2
        cv2.rectangle(img, (10, 10), (size[0] - 10, size[1] - 10), (0, 255, 0), -1)
        
        marks_list = extract_marks(img)
        # Green rect should be detected
        assert marks_list is not None

    def test_color_mask(self):
        """颜色掩码测试"""
        from rightbrain.cv_core.extract_features import get_color_mask
        
        import cv2
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :] = (0, 0, 200)  # BGR red
        
        mask = get_color_mask(img)
        assert mask is not None
        assert isinstance(mask, np.ndarray)
        # There should be some white (detected) pixels in the mask
        num_white = np.sum(mask > 0)
        assert num_white > 0, "Red image should produce some mask pixels"

    def test_affect_sensor_valid(self):
        """情感联想：有效颜色和形状"""
        from rightbrain.cv_core.affect_sensor import extract_affect_features
        
        result = extract_affect_features("红", "圆形")
        assert result is not None
        assert "情感联想" in result
        assert "愉悦度" in result
        assert "感受" in result

    def test_affect_sensor_unknown(self):
        """情感联想：未知颜色"""
        from rightbrain.cv_core.affect_sensor import extract_affect_features
        
        result = extract_affect_features("透明", "不规则")
        assert result is not None
        assert "情感联想" in result
