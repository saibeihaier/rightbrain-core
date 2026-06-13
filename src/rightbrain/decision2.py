"""
decision2.py — 新原理的决策引擎

流程：
1. 右脑：sensor2 → 提取稀疏标记
2. 竞争：memory2.match() → WTA 胜出
3. 判断：
   - 高置信度胜出 → 直接行动
   - 低置信度 → 触发左脑
   - 胜出不明显 → 焦点重定向 + 再识别
4. 行动输出

注意：本模块仅支持 ExperienceMemory2 实例。
旧版 ExperienceMemory 实例请使用 decision.py。
"""
import time
import threading
from typing import Dict, Optional, Tuple, Any
from rightbrain.config import Config


def identify_v2(marks_dict: Dict, memory, context_marks: Dict = None,
                confidence_threshold: float = 0.5) -> Tuple[str, float, Any, bool]:
    """
    新原理的识别函数。
    
    Args:
        marks_dict: sensor2 提取的标记
        memory: ExperienceMemory2 实例（仅支持新版！）
        context_marks: 上下文标记（地名、时间等）
        confidence_threshold: 触发左脑的阈值
        
    Returns:
        (action_text, score, exp, is_new)
        和旧 identify() 完全兼容
    """
    # 类型检查：仅支持 ExperienceMemory2
    if not hasattr(memory, 'match') or not hasattr(memory, 'experiences'):
        raise TypeError(
            "decision2.identify_v2() 仅支持 ExperienceMemory2 实例。"
            "请使用 decision.identify() 处理旧版 ExperienceMemory 实例。"
        )
    
    # 调用新版match方法（返回 best_exp, win_score, runner_up_score）
    best_exp, win_score, runner_up = memory.match(marks_dict, context_marks)
    
    if best_exp is None:
        action = _build_unknown_action(marks_dict)
        return action, win_score, None, False
    
    # 构建行动文本
    action = best_exp.action
    if not action:
        action = best_exp.name
    
    # 是否需要左脑
    if best_exp.need_left and win_score < confidence_threshold:
        return action, win_score, best_exp, False
    
    return action, win_score, best_exp, False


def _build_unknown_action(marks_dict: Dict) -> str:
    """构建未知物体的描述"""
    color = marks_dict.get('颜色', '')
    shape = marks_dict.get('形状', '')
    position = marks_dict.get('位置区域', '')
    
    parts = []
    if color:
        parts.append(color)
    if shape:
        parts.append(shape)
    if position and position != '未知':
        parts.append(position)
    
    return ''.join(parts) if parts else '未知物体'


def check_safety(marks_dict: Dict, spatial_rel: Dict = None) -> Tuple[bool, str]:
    """
    安全检查：判断当前场景是否有危险。
    这是右脑"直觉"的一部分，不经过左脑。
    
    Returns: (is_dangerous, reason)
    """
    edge = marks_dict.get('距边缘', '远')
    support = marks_dict.get('支撑面', '无')
    position = marks_dict.get('位置区域', '')
    
    # 边缘危险（上方区域如天空/白云不触发）
    position = marks_dict.get('位置区域', '')
    if edge == '近' and position != '上方':
        return True, '物体在边缘，可能掉落'
    
    # 空间关系危险
    if spatial_rel:
        if spatial_rel.get('稳定性') == '危险':
            return True, '位置不稳定'
    
    return False, '安全'


def should_focus_refine(marks_dict: Dict, memory_result: Tuple) -> bool:
    """
    判断是否需要焦点重定向（精细再识别）。
    
    条件：标记的置信度过低，但该标记对区分当前候选很重要。
    """
    _, win_score, _ = memory_result
    
    # 中等置信度 + 可能有更多信息
    if 0.3 <= win_score < 0.5:
        return True
    
    return False
