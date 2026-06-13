"""
sensor2.py — 基于焦点和空间估算的感知器

核心思想：
- 焦点外的区域：只提取基础颜色块，不做精细识别
- 空间估算：基于图像坐标和物体尺寸做 2D 空间判断
- 支撑面稳定性：基于 x 轴位置和支撑物类型
- 标记提取：稀疏的、有区分度的标记

输出：一个标记字典，每个标记带置信度
"""
import cv2
import numpy as np
import time
from typing import Dict, Optional, List, Tuple

# 16 种基本色（人类能分辨的范围）
BASIC_COLORS = {
    '红': ([0, 100, 100], [10, 255, 255]),
    '红2': ([160, 100, 100], [180, 255, 255]),
    '橙': ([10, 100, 100], [25, 255, 255]),
    '黄': ([25, 60, 100], [40, 255, 255]),
    '绿': ([40, 60, 60], [80, 255, 255]),
    '青': ([80, 60, 60], [100, 255, 255]),
    '蓝': ([100, 80, 80], [130, 255, 255]),
    '紫': ([130, 60, 60], [160, 255, 255]),
    '白': ([0, 0, 200], [180, 30, 255]),
    '灰': ([0, 0, 80], [180, 30, 200]),
    '黑': ([0, 0, 0], [180, 255, 60]),
    '棕': ([10, 60, 60], [30, 150, 150]),
    '粉': ([150, 60, 100], [170, 150, 255]),
}

# 形状类型
SHAPE_TYPES = ['圆形', '椭圆形', '长方形', '长条形', '正方形', '三角形', '多边形', '不规则', '无']

# 空间位置区域
POSITION_ZONES = ['天空', '远方', '地面', '桌面', '近处', '手中', '未知']


def extract_marks(image: np.ndarray, focus_region: Optional[Tuple] = None,
                  full_analysis: bool = False) -> Dict:
    """
    从图像中提取稀疏标记。
    
    Args:
        image: BGR 图像 (H,W,3)
        focus_region: 焦点区域 (x1,y1,x2,y2)，None 表示全图
        full_analysis: 是否做完整分析（True=焦点内，False=焦点外像素块）
    
    Returns:
        dict: 标记字典，包含:
            - '颜色': 主色名称
            - '形状': 形状名称  
            - '大小': '大/中/小'
            - '纹理': '光滑/粗糙/中等'
            - '位置区域': 空间区域
            - '空间_x': x 轴相对位置 0-1
            - '空间_y': y 轴相对位置 0-1
            - '支撑面': '有/无'
            - '焦点内': bool
            - '_置信度': {标记名: 置信度}
    """
    h, w = image.shape[:2]
    marks = {
        '颜色': '未知',
        '形状': '无',
        '大小': '中',
        '纹理': '中等',
        '位置区域': '未知',
        '空间_x': 0.5,
        '空间_y': 0.5,
        '支撑面': '无',
        '距边缘': '远',
        '焦点内': focus_region is not None,
    }
    conf = {}
    
    if focus_region:
        x1, y1, x2, y2 = focus_region
        roi = image[y1:y2, x1:x2].copy()
        if roi.size == 0:
            return marks
    else:
        roi = image.copy()
        x1, y1, x2, y2 = 0, 0, w, h
    
    # 保存原始图像的 bbox 基准（用于空间计算）
    _base_w, _base_h = w, h
    
    roi_h, roi_w = roi.shape[:2]
    
    # === 颜色提取 ===
    color, color_conf = _extract_color(roi)
    marks['颜色'] = color
    conf['颜色'] = color_conf
    
    # === 空间估算（基于图像坐标）===
    cx = (x1 + x2) / 2 / w
    cy = (y1 + y2) / 2 / h
    marks['空间_x'] = round(cx, 2)
    marks['空间_y'] = round(cy, 2)
    
    # 位置区域判断（基于 ROI 中心在画面中的 y 比例）
    if cy < 0.35:
        marks['位置区域'] = '上方'
    elif cy > 0.65:
        marks['位置区域'] = '地面'
    else:
        marks['位置区域'] = '中间'
    conf['位置区域'] = 0.6
    
    # 距边缘距离
    edge_dist = min(x1, y1, w - x2, h - y2) / max(w, h)
    if edge_dist < 0.05:
        marks['距边缘'] = '近'
    elif edge_dist < 0.15:
        marks['距边缘'] = '中'
    else:
        marks['距边缘'] = '远'
    conf['距边缘'] = 0.7
    
    # === 焦点外：只做基础分析 ===
    if not full_analysis:
        marks['_置信度'] = conf
        return marks
    
    # === 焦点内：完整分析 ===
    
    # 形状识别
    shape, shape_conf = _extract_shape(roi)
    marks['形状'] = shape
    conf['形状'] = shape_conf
    
    # 大小估计
    area_ratio = (roi_w * roi_h) / (h * w)
    if area_ratio > 0.3:
        marks['大小'] = '大'
    elif area_ratio > 0.08:
        marks['大小'] = '中'
    else:
        marks['大小'] = '小'
    conf['大小'] = 0.7
    
    # 纹理
    texture, texture_conf = _extract_texture(roi)
    marks['纹理'] = texture
    conf['纹理'] = texture_conf
    
    # 支撑面检测（底部是否有支撑）
    bottom_strip = image[int(y2):min(y2 + 20, h), x1:x2]
    if bottom_strip.size > 0:
        gray_bottom = cv2.cvtColor(bottom_strip, cv2.COLOR_BGR2GRAY)
        if np.std(gray_bottom) > 20:
            marks['支撑面'] = '有'
        else:
            marks['支撑面'] = '无'
    conf['支撑面'] = 0.5
    
    marks['_置信度'] = conf
    return marks


def _extract_color(roi: np.ndarray) -> Tuple[str, float]:
    """提取主色。先用轮廓定位物体区域，排除纯黑背景。"""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)  # 排除纯黑区域
    # 如果排除后没有像素，放宽阈值
    if mask.sum() < 100:
        _, mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    scores = {}
    for name, (lower, upper) in BASIC_COLORS.items():
        color_mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        # 只统计非黑区域的像素
        valid_pixels = cv2.bitwise_and(color_mask, mask)
        if mask.sum() > 0:
            score = np.sum(valid_pixels > 0) / np.sum(mask > 0)
        else:
            score = 0
        if score > 0.05:
            simple_name = name.rstrip('2')
            scores[simple_name] = max(scores.get(simple_name, 0), score)
    
    if not scores:
        return '未知', 0.3
    
    best = max(scores, key=scores.get)
    best_score = scores[best]
    
    sorted_scores = sorted(scores.values(), reverse=True)
    if len(sorted_scores) >= 2 and sorted_scores[0] > sorted_scores[1] * 1.5:
        return best, min(0.9, best_score)
    else:
        return best, max(0.3, best_score * 0.7)


def _extract_shape(roi: np.ndarray) -> Tuple[str, float]:
    """提取形状"""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return '无', 0.0
    
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 50:
        return '无', 0.0
    
    peri = cv2.arcLength(c, True)
    if peri == 0:
        return '无', 0.0
    
    circularity = 4 * np.pi * area / (peri * peri)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    verts = len(approx)
    
    rect = cv2.boundingRect(c)
    aspect = rect[2] / max(rect[3], 1)
    
    if verts <= 2:
        return '不规则', 0.3
    elif verts == 3:
        return '三角形', 0.8
    elif verts == 4:
        if 0.85 <= aspect <= 1.15:
            return '正方形', 0.8
        elif aspect > 3.0:
            return '长条形', 0.7
        else:
            return '长方形', 0.8
    elif verts == 5:
        return '多边形', 0.6
    elif verts >= 6:
        if circularity > 0.75:
            return '圆形', min(0.9, circularity)
        elif circularity > 0.5:
            return '椭圆形', 0.7
        else:
            return '多边形', 0.5
    
    return '不规则', 0.3


def _extract_texture(roi: np.ndarray) -> Tuple[str, float]:
    """提取纹理特征"""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    if lap_var > 200:
        return '粗糙', 0.8
    elif lap_var > 50:
        return '中等', 0.6
    else:
        return '光滑', 0.8


def global_scan(image: np.ndarray, grid_size: int = 4) -> List[Dict]:
    """
    全局扫描：把图像分成 NxN 网格，每个网格提取基础标记。
    用于发现焦点候选区域。
    
    Returns: [{'region': (x1,y1,x2,y2), 'marks': {...}}, ...]
    """
    h, w = image.shape[:2]
    cells = []
    
    cell_h, cell_w = h // grid_size, w // grid_size
    
    for i in range(grid_size):
        for j in range(grid_size):
            x1 = j * cell_w
            y1 = i * cell_h
            x2 = (j + 1) * cell_w
            y2 = (i + 1) * cell_h
            
            cell_img = image[y1:y2, x1:x2]
            marks = extract_marks(cell_img, full_analysis=False)
            marks['网格位置'] = (i, j)
            
            cells.append({
                'region': (x1, y1, x2, y2),
                'marks': marks.copy(),
            })
    
    return cells


def find_focus_candidates(cells: List[Dict], memory_marks: List[str] = None) -> List[Tuple]:
    """
    根据全局扫描结果找出焦点候选区域。
    策略：选择标记变化大（高熵）或与经验标记匹配度低的区域。
    
    Returns: [(region, score), ...]
    """
    candidates = []
    
    for cell in cells:
        region = cell['region']
        marks = cell['marks']
        
        # 计算"新奇度"：颜色和位置组合的稀有度
        color = marks.get('颜色', '未知')
        position = marks.get('位置区域', '未知')
        
        # 天空位置出现非蓝色 = 高新奇度
        # 地面位置出现非灰色 = 高新奇度
        novelty = 0.5
        
        if marks.get('位置区域') == '上方' and color not in ['蓝', '白', '灰']:
            novelty = 0.8
        elif marks.get('位置区域') == '下方' and color not in ['灰', '绿', '棕', '黑']:
            novelty = 0.8
        elif marks.get('位置区域') == '中间' and color in ['红', '橙', '黄']:
            novelty = 0.7
        
        if novelty > 0.5:
            candidates.append((region, novelty))
    
    return sorted(candidates, key=lambda x: x[1], reverse=True)


def estimate_spatial_relation(marks_a: Dict, marks_b: Dict) -> Dict:
    """
    估算两个物体之间的空间关系。
    A 是焦点物体，B 是参照物。
    
    Returns: {
        '相对位置': '上面/下面/旁边/里面',
        '距离估测': '近/中/远',
        '稳定性': '稳/不稳/危险',
    }
    """
    ax = marks_a.get('空间_x', 0.5)
    ay = marks_a.get('空间_y', 0.5)
    bx = marks_b.get('空间_x', 0.5)
    by = marks_b.get('空间_y', 0.5)
    
    dx = abs(ax - bx)
    dy = abs(ay - by)
    
    if dy < 0.1 and dx < 0.1:
        dist = '近'
    elif dy < 0.2 and dx < 0.2:
        dist = '中'
    else:
        dist = '远'
    
    # 位置关系
    if ay < by - 0.15:
        pos = '上面'
    elif ay > by + 0.15:
        pos = '下面'
    elif dx < 0.1:
        pos = '里面'
    else:
        pos = '旁边'
    
    # 稳定性（简单启发式）
    support = marks_a.get('支撑面', '无')
    edge = marks_a.get('距边缘', '远')
    
    if support == '无' and edge == '近':
        stable = '危险'
    elif support == '无':
        stable = '不稳'
    else:
        stable = '稳'
    
    return {
        '相对位置': pos,
        '距离估测': dist,
        '稳定性': stable,
    }
