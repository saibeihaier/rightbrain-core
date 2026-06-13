#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工具函数模块 - 提供通用工具函数
"""

import cv2
import numpy as np

def normalize_bbox(bbox):
    """
    统一边界框格式，支持字典和列表两种格式
    
    Args:
        bbox: 边界框，可以是字典（包含'bbox'键或四个坐标键）或列表/元组（四个整数）
    
    Returns:
        tuple: 标准化的边界框 (x1, y1, x2, y2)
    
    Examples:
        >>> normalize_bbox({'bbox': [10, 20, 30, 40]})
        (10, 20, 30, 40)
        
        >>> normalize_bbox({'xmin': 10, 'ymin': 20, 'xmax': 30, 'ymax': 40})
        (10, 20, 30, 40)
        
        >>> normalize_bbox([10, 20, 30, 40])
        (10, 20, 30, 40)
    """
    if isinstance(bbox, dict):
        # 处理字典格式
        if 'bbox' in bbox:
            return tuple(map(int, bbox['bbox']))
        elif 'xmin' in bbox and 'ymin' in bbox and 'xmax' in bbox and 'ymax' in bbox:
            return (
                int(bbox['xmin']),
                int(bbox['ymin']),
                int(bbox['xmax']),
                int(bbox['ymax'])
            )
        else:
            # 尝试获取所有数值类型的值
            values = [v for v in bbox.values() if isinstance(v, (int, float))]
            if len(values) >= 4:
                return tuple(map(int, values[:4]))
            else:
                raise ValueError(f"无法解析边界框字典: {bbox}")
    elif isinstance(bbox, (list, tuple)):
        # 处理列表/元组格式
        if len(bbox) >= 4:
            return tuple(map(int, bbox[:4]))
        else:
            raise ValueError(f"边界框需要至少4个坐标值: {bbox}")
    else:
        raise TypeError(f"不支持的边界框类型: {type(bbox)}")

def clip_bbox(bbox, image_shape):
    """
    将边界框裁剪到图像范围内
    
    Args:
        bbox: 边界框 (x1, y1, x2, y2)
        image_shape: 图像形状 (height, width, channels)
    
    Returns:
        tuple: 裁剪后的边界框
    """
    x1, y1, x2, y2 = bbox
    height, width = image_shape[:2]
    
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width - 1, x2)
    y2 = min(height - 1, y2)
    
    return (x1, y1, x2, y2)

def get_bbox_center(bbox):
    """
    获取边界框中心点坐标
    
    Args:
        bbox: 边界框 (x1, y1, x2, y2)
    
    Returns:
        tuple: 中心点坐标 (cx, cy)
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    return (cx, cy)

def get_bbox_size(bbox):
    """
    获取边界框尺寸
    
    Args:
        bbox: 边界框 (x1, y1, x2, y2)
    
    Returns:
        tuple: (宽度, 高度)
    """
    x1, y1, x2, y2 = bbox
    return (x2 - x1, y2 - y1)

def get_bbox_area(bbox):
    """
    获取边界框面积
    
    Args:
        bbox: 边界框 (x1, y1, x2, y2)
    
    Returns:
        int: 面积
    """
    width, height = get_bbox_size(bbox)
    return width * height

def draw_bbox(image, bbox, color=(0, 255, 0), thickness=2, label=None):
    """
    在图像上绘制边界框
    
    Args:
        image: OpenCV图像
        bbox: 边界框 (x1, y1, x2, y2)
        color: 颜色 (B, G, R)
        thickness: 线条粗细
        label: 标签文字
    
    Returns:
        image: 绘制后的图像
    """
    x1, y1, x2, y2 = bbox
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    
    if label:
        # 在边界框上方绘制标签
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        label_y = max(y1 - label_size[1] - 5, 0)
        cv2.rectangle(image, (x1, label_y), (x1 + label_size[0], y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return image

def is_point_inside_bbox(point, bbox):
    """
    判断点是否在边界框内
    
    Args:
        point: 点坐标 (x, y)
        bbox: 边界框 (x1, y1, x2, y2)
    
    Returns:
        bool: 是否在边界框内
    """
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2

def resize_image(image, max_dimension=1024):
    """
    等比例缩放图像，使最大维度不超过指定值
    
    Args:
        image: OpenCV图像
        max_dimension: 最大维度
    
    Returns:
        image: 缩放后的图像
        scale: 缩放比例
    """
    height, width = image.shape[:2]
    scale = 1.0
    
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        image = cv2.resize(image, None, fx=scale, fy=scale)
    
    return image, scale

def normalize_image(image):
    """
    归一化图像，转换为RGB格式并缩放到0-1范围
    
    Args:
        image: OpenCV图像（BGR格式）
    
    Returns:
        ndarray: 归一化后的图像
    """
    # 转换为RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    # 归一化到0-1
    return image_rgb / 255.0

def compute_iou(bbox1, bbox2):
    """
    计算两个边界框的交并比（IoU）
    
    Args:
        bbox1: 第一个边界框 (x1, y1, x2, y2)
        bbox2: 第二个边界框 (x1, y1, x2, y2)
    
    Returns:
        float: IoU值
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # 计算交集区域
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)
    
    # 检查是否有交集
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    
    # 计算面积
    area1 = get_bbox_area(bbox1)
    area2 = get_bbox_area(bbox2)
    inter_area = get_bbox_area((inter_x1, inter_y1, inter_x2, inter_y2))
    
    # 计算IoU
    return inter_area / (area1 + area2 - inter_area)
