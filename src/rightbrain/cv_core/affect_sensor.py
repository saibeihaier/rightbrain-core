import random
import time
from typing import Dict, Tuple

LOG_ENABLED = True

def _log(message):
    if LOG_ENABLED:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [Affect] {message}")

AFFECT_MAP = {
    '甜': {'description': '甜', 'pleasant': 0.8, 'intensity': '中'},
    '酸': {'description': '酸', 'pleasant': 0.3, 'intensity': '中'},
    '苦': {'description': '苦', 'pleasant': 0.1, 'intensity': '强'},
    '辣': {'description': '辣', 'pleasant': 0.4, 'intensity': '强'},
    '咸': {'description': '咸', 'pleasant': 0.5, 'intensity': '中'},
    '鲜': {'description': '鲜', 'pleasant': 0.7, 'intensity': '中'},
    '涩': {'description': '涩', 'pleasant': 0.2, 'intensity': '弱'},
    '麻': {'description': '麻', 'pleasant': 0.35, 'intensity': '强'},
}

# 情感映射说明（用于日志）
AFFECT_DESCRIPTIONS = {
    '红': '热情、活力',
    '橙': '温暖、友好',
    '黄': '活泼、愉悦',
    '绿': '自然、清新',
    '蓝': '冷静、理性',
    '紫': '神秘、高贵',
    '白': '纯净、简洁',
    '黑': '深沉、严肃',
    '灰': '中性、沉稳',
    '亮灰': '柔和、轻盈',
    '中灰': '平衡、稳重',
    '深灰': '厚重、沉稳',
    '暗灰': '暗淡、压抑',
    '青': '清新、自然',
}

COLOR_AFFECT_MAPPING = {
    '红': ['甜', '酸'],
    '橙': ['甜', '酸'],
    '黄': ['甜', '酸'],
    '绿': ['酸', '涩'],
    '蓝': ['咸', '鲜'],
    '紫': ['甜', '酸'],
    '白': ['咸', '鲜'],
    '黑': ['苦', '咸'],
    '灰': ['咸', '涩'],
    '亮灰': ['咸', '鲜'],
    '中灰': ['咸', '涩'],
    '深灰': ['苦', '咸'],
    '暗灰': ['苦', '涩'],
    '青': ['酸', '鲜'],
}

SHAPE_AFFECT_MAPPING = {
    '圆形': ['甜', '鲜'],
    '椭圆形': ['甜', '酸'],
    '正方形': ['咸', '鲜'],
    '长方形': ['咸', '苦'],
    '三角形': ['酸', '辣'],
    '多边形': ['涩', '苦'],
    '无': ['咸'],
}

class AffectPreferences:
    def __init__(self):
        self.preferences: Dict[str, float] = {}
        self._init_default_preferences()
        _log("AffectPreferences 初始化完成")
    
    def _init_default_preferences(self):
        self.preferences = {
            '甜': 0.9,
            '酸': 0.75,
            '苦': 0.35,
            '辣': 0.6,
            '咸': 0.8,
            '鲜': 0.9,
            '涩': 0.4,
            '麻': 0.55,
        }
        _log(f"默认偏好配置: {self.preferences}")
    
    def set_preference(self, affect: str, preference: float):
        if not isinstance(affect, str):
            _log(f"错误: 情感类型必须是字符串，收到 {type(affect).__name__}")
            return False
        
        if affect not in AFFECT_MAP:
            _log(f"警告: 未知情感类型 '{affect}'")
            return False
        
        if not isinstance(preference, (int, float)):
            _log(f"错误: 偏好值必须是数字，收到 {type(preference).__name__}")
            return False
        
        if preference < 0:
            _log(f"警告: 偏好值 {preference} 为负数，已自动截断到 0.0")
            new_value = 0.0
        elif preference > 1:
            _log(f"警告: 偏好值 {preference} 超过 1.0，已自动截断到 1.0")
            new_value = 1.0
        else:
            new_value = preference
        
        old_value = self.preferences.get(affect, 0.5)
        self.preferences[affect] = new_value
        _log(f"偏好调整: {affect} = {old_value:.2f} → {new_value:.2f}")
        return True
    
    def get_preference(self, affect: str) -> float:
        if not isinstance(affect, str):
            _log(f"警告: get_preference 收到非字符串输入: {type(affect).__name__}")
            return 0.5
        return self.preferences.get(affect, 0.5)
    
    def get_overall_pleasantness(self, affects: list) -> float:
        if affects is None:
            _log("错误: 输入情感列表为 None，返回默认愉悦度 0.5")
            return 0.5
        
        if not isinstance(affects, list):
            _log(f"错误: 输入必须是列表，收到 {type(affects).__name__}")
            return 0.5
        
        if not affects:
            _log("输入情感列表为空，返回默认愉悦度 0.5")
            return 0.5
        
        valid_affects = []
        for affect in affects:
            if isinstance(affect, str) and affect in AFFECT_MAP:
                valid_affects.append(affect)
            else:
                _log(f"警告: 跳过无效情感 '{affect}'")
        
        if not valid_affects:
            _log("没有有效的情感输入，返回默认愉悦度 0.5")
            return 0.5
        
        _log(f"计算愉悦度，输入情感: {valid_affects}")
        total_score = 0.0
        for affect in valid_affects:
            affect_info = AFFECT_MAP.get(affect, {'pleasant': 0.5})
            preference = self.get_preference(affect)
            contribution = affect_info['pleasant'] * preference
            total_score += contribution
            _log(f"  {affect}: 基础愉悦度={affect_info['pleasant']:.2f}, 偏好={preference:.2f}, 贡献={contribution:.2f}")
        
        result = total_score / len(valid_affects)
        _log(f"最终愉悦度: {result:.2f}")
        return result

def infer_affect_from_visual(color: str, shape: str) -> Tuple[list, float]:
    _log(f"========== 情感推理开始 ==========")
    
    # 颜色语义说明
    color_desc = AFFECT_DESCRIPTIONS.get(color, '未知')
    _log(f"[输入] 颜色={color} ({color_desc}), 形状={shape}")
    
    # 颜色到情感的映射
    color_affects = COLOR_AFFECT_MAPPING.get(color, ['咸'])
    _log(f"[颜色映射] {color} -> 情感候选: {color_affects}")
    
    # 形状到情感的映射
    shape_affects = SHAPE_AFFECT_MAPPING.get(shape, ['咸'])
    _log(f"[形状映射] {shape} -> 情感候选: {shape_affects}")
    
    # 合并情感
    combined = list(set(color_affects + shape_affects))
    _log(f"[合并] 去重后情感: {combined}")
    
    # 如果情感过多，随机选择2个
    if len(combined) > 2:
        selected = random.sample(combined, 2)
        _log(f"[选择] 情感过多，随机选择: {selected}")
        combined = selected
    else:
        _log(f"[最终] 最终情感列表: {combined}")
    
    _log(f"========== 情感推理完成 ==========")
    return combined

def get_affect_description(affects: list) -> str:
    if not affects:
        return "未知情感"
    
    descriptions = [AFFECT_MAP[affect]['description'] for affect in affects]
    return '、'.join(descriptions)

def determine_pleasure_level(pleasantness: float) -> str:
    if pleasantness >= 0.7:
        return '愉悦'
    elif pleasantness >= 0.4:
        return '中性'
    else:
        return '厌恶'

def extract_affect_features(color: str, shape: str, preferences: AffectPreferences = None) -> dict:
    _log(f"========== 情感特征提取开始 ==========")
    _log(f"输入: color={color}, shape={shape}")
    
    if preferences is None:
        preferences = AffectPreferences()
        _log("使用默认偏好配置")
    
    affects = infer_affect_from_visual(color, shape)
    pleasantness = preferences.get_overall_pleasantness(affects)
    pleasure_level = determine_pleasure_level(pleasantness)
    
    result = {
        '情感联想': get_affect_description(affects),
        '情感类型': affects,
        '愉悦度': pleasantness,
        '感受': pleasure_level,
    }
    
    _log(f"[结果] 情感联想={result['情感联想']}, 愉悦度={pleasantness:.2f}, 感受={pleasure_level}")
    _log(f"========== 情感特征提取完成 ==========")
    
    return result

def apply_affect_to_action(action: str, feeling: str) -> str:
    """将情感应用到行动建议，仅修改语气，不改变行动内容"""
    if feeling == '愉悦':
        return f"【愉悦】{action}"
    elif feeling == '厌恶':
        return f"【厌恶】{action}"
    else:
        return action