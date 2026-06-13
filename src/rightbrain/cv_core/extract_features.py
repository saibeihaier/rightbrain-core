#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征提取模块 - 从图像中提取各种视觉特征

包含以下特征提取函数：
- 距离估算
- 尺寸估算  
- 纹理估算
- 光照估算
- 边缘距离计算
- 颜色掩码生成
"""

import cv2
import numpy as np

from rightbrain.utils.utils_functions import normalize_bbox


def get_edge_distance(image: np.ndarray, bbox) -> str:
    """
    计算物体中心到图像边缘的距离
    
    Args:
        image: OpenCV图像（BGR格式）
        bbox: 边界框，可以是字典或元组格式
    
    Returns:
        str: "近" 或 "远"
    """
    h, w = image.shape[:2]
    
    # 使用统一的边界框处理
    x1, y1, x2, y2 = normalize_bbox(bbox)
    
    center_x = (x1 + x2) / 2
    dist_left = center_x / w
    dist_right = (w - center_x) / w
    min_dist = min(dist_left, dist_right)
    return "近" if min_dist < 0.2 else "远"


def estimate_distance(image: np.ndarray, bbox) -> str:
    """
    根据物体高度估算距离
    
    Args:
        image: OpenCV图像（BGR格式）
        bbox: 边界框，可以是字典或元组格式
    
    Returns:
        str: "近"、"中" 或 "远"
    """
    x1, y1, x2, y2 = normalize_bbox(bbox)
    
    h_img = image.shape[0]
    obj_height_pixels = y2 - y1
    ratio = obj_height_pixels / h_img
    
    if ratio > 0.4:
        return "近"
    elif ratio > 0.15:
        return "中"
    else:
        return "远"


def estimate_size(image: np.ndarray, bbox) -> str:
    """
    根据物体面积估算尺寸
    
    Args:
        image: OpenCV图像（BGR格式）
        bbox: 边界框，可以是字典或元组格式
    
    Returns:
        str: "大"、"中" 或 "小"
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = normalize_bbox(bbox)
    
    obj_area = (x2 - x1) * (y2 - y1)
    img_area = h * w
    ratio = obj_area / img_area
    
    if ratio > 0.15:
        return "大"
    elif ratio > 0.03:
        return "中"
    else:
        return "小"


def estimate_texture(image: np.ndarray, bbox) -> str:
    """
    估算物体表面纹理
    
    Args:
        image: OpenCV图像（BGR格式）
        bbox: 边界框，可以是字典或元组格式
    
    Returns:
        str: "光滑"、"轻微纹理"、"粗糙" 或 "复杂纹理"
    """
    x1, y1, x2, y2 = normalize_bbox(bbox)
    
    # 裁剪到图像边界
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
    
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return "未知"
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    edge_density = np.mean(np.abs(laplacian))
    
    if edge_density < 5:
        return "光滑"
    elif edge_density < 20:
        return "轻微纹理"
    elif edge_density < 40:
        return "粗糙"
    else:
        return "复杂纹理"


def estimate_lighting(image: np.ndarray, bbox) -> str:
    """
    估算物体区域的光照情况
    
    Args:
        image: OpenCV图像（BGR格式）
        bbox: 边界框，可以是字典或元组格式
    
    Returns:
        str: "明亮"、"正常"、"较暗" 或 "黑暗"
    """
    x1, y1, x2, y2 = normalize_bbox(bbox)
    
    # 裁剪到图像边界
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
    
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return "未知"
    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    avg_brightness = np.mean(hsv[:, :, 2])
    
    if avg_brightness > 200:
        return "明亮"
    elif avg_brightness > 100:
        return "正常"
    elif avg_brightness > 50:
        return "较暗"
    else:
        return "黑暗"


def get_color_mask(image: np.ndarray) -> np.ndarray:
    """
    生成颜色检测掩码，用于检测常见颜色和肤色
    
    Args:
        image: OpenCV图像（BGR格式）
    
    Returns:
        np.ndarray: 颜色掩码图像
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 红色（两个范围）
    mask_red1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
    mask_red2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    
    # 蓝色
    mask_blue = cv2.inRange(hsv, np.array([100, 100, 100]), np.array([130, 255, 255]))
    
    # 黄色
    mask_yellow = cv2.inRange(hsv, np.array([25, 100, 100]), np.array([65, 255, 255]))
    
    # 绿色
    mask_green = cv2.inRange(hsv, np.array([50, 100, 100]), np.array([80, 255, 255]))
    
    # 橙色
    mask_orange = cv2.inRange(hsv, np.array([10, 100, 100]), np.array([30, 255, 255]))
    
    # 紫色
    mask_purple = cv2.inRange(hsv, np.array([130, 100, 100]), np.array([160, 255, 255]))
    
    # 青色
    mask_cyan = cv2.inRange(hsv, np.array([80, 100, 100]), np.array([100, 255, 255]))
    
    # 肤色
    mask_skin = cv2.inRange(hsv, np.array([5, 20, 80]), np.array([40, 150, 255]))
    
    # 合并所有颜色掩码
    mask = cv2.bitwise_or(mask_red, mask_blue)
    mask = cv2.bitwise_or(mask, mask_yellow)
    mask = cv2.bitwise_or(mask, mask_green)
    mask = cv2.bitwise_or(mask, mask_orange)
    mask = cv2.bitwise_or(mask, mask_purple)
    mask = cv2.bitwise_or(mask, mask_cyan)
    mask = cv2.bitwise_or(mask, mask_skin)
    
    return mask


def get_support_surface(image: np.ndarray, bbox) -> str:
    """
    判断物体是否有支撑表面（预留功能）
    
    Args:
        image: OpenCV图像（BGR格式）
        bbox: 边界框，可以是字典或元组格式
    
    Returns:
        str: "无"（预留）
    """
    return "无"


def extract_all_features(image: np.ndarray, bbox) -> dict:
    """
    提取所有特征并返回特征字典
    
    Args:
        image: OpenCV图像（BGR格式）
        bbox: 边界框，可以是字典或元组格式
    
    Returns:
        dict: 包含所有特征的字典
    """
    return {
        '距离': estimate_distance(image, bbox),
        '尺寸': estimate_size(image, bbox),
        '纹理': estimate_texture(image, bbox),
        '光照': estimate_lighting(image, bbox),
        '边缘距离': get_edge_distance(image, bbox),
        '支撑表面': get_support_surface(image, bbox),
    }
