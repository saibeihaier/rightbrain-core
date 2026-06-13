"""
memory2.py — 基于稀疏标记 + WTA竞争的识别引擎

与原来的 memory.py 不同：
1. 标记是稀疏的（只有有区分度的才存）
2. 匹配是竞争制（WTA），不是相似度排序
3. 经验条目的 condition 只存该场景特有的标记
4. 共同标记自动归入"模板"，不重复存储

核心函数：
- match(marks, context_marks=None) → WTA竞争
- learn(condition, action, distinguish_from=None) → 只存差异标记
- template_match(marks) → 匹配通用模板
"""
import json
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict


@dataclass
class Experience:
    """一条经验"""
    name: str = "未知"
    action: str = ""
    # 条件：只存该场景特异的标记（稀疏的）
    condition: Dict[str, str] = field(default_factory=dict)
    # 所属模板（如"皇家建筑"、"水果"），用于归类和联想
    template: str = ""
    # 置信度
    confidence: float = 0.5
    # 是否需要左脑参与
    need_left: bool = False
    # 位置区域偏好（用于空间先验）
    prefer_position: str = ""
    # 创建时间
    created_at: float = 0
    
    def to_dict(self):
        return {
            'name': self.name,
            'action': self.action,
            'condition': self.condition,
            'template': self.template,
            'confidence': self.confidence,
            'need_left': self.need_left,
            'prefer_position': self.prefer_position,
            'created_at': self.created_at or time.time(),
        }
    
    @classmethod
    def from_dict(cls, d):
        return cls(
            name=d.get('name', '未知'),
            action=d.get('action', ''),
            condition=d.get('condition', {}),
            template=d.get('template', ''),
            confidence=d.get('confidence', 0.5),
            need_left=d.get('need_left', False),
            prefer_position=d.get('prefer_position', ''),
            created_at=d.get('created_at', 0),
        )


class TemplateLibrary:
    """
    模板库：存储同类物体的共同标记。
    比如"皇家建筑"模板 = {颜色:黄, 形状:长方形, 有:龙椅}
    具体建筑只存 DIFF。
    """
    def __init__(self):
        self.templates: Dict[str, Dict] = {}
        self._init_defaults()
    
    def _init_defaults(self):
        self.templates['水果'] = {'大小': '中', '场景': '食物'}
        self.templates['皇家建筑'] = {'风格': '传统', '装饰': '金龙'}
        self.templates['道路'] = {'形状': '长条形', '位置区域': '地面'}
        self.templates['车辆'] = {'运动': '地面移动', '大小': '大'}
        self.templates['云'] = {'位置区域': '上方', '形状': '不规则', '纹理': '模糊'}
    
    def match(self, marks: Dict) -> Tuple[str, float]:
        """找最匹配的模板"""
        best = ('', 0.0)
        for name, template_marks in self.templates.items():
            matched = 0
            total = len(template_marks)
            for k, v in template_marks.items():
                if k in marks and marks[k] == v:
                    matched += 1
            score = matched / max(total, 1)
            if score > best[1]:
                best = (name, score)
        return best


class ExperienceMemory2:
    """
    基于稀疏标记+WTA竞争的识别引擎。
    
    匹配过程：
    1. 先匹配模板 → 缩小候选范围
    2. 在候选列表中做 WTA 竞争
    3. 胜出者的激活强度 > 第二名一定阈值 → 确定
    4. 否则 → 不确定，触发左脑
    """
    
    def __init__(self, json_path: str = None):
        self.experiences: List[Experience] = []
        self.templates = TemplateLibrary()
        from rightbrain.config import Config
        self.wta_threshold = Config.clamp_to_genome("wta_margin", Config.WTA_MARGIN)
        self.confidence_threshold = Config.clamp_to_genome("confidence_threshold", Config.CONFIDENCE_THRESHOLD)
        self.json_path = json_path
        
        if json_path:
            self.load(json_path)
    
    def load(self, path: str):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.experiences = [Experience.from_dict(d) for d in data]
            print(f"[Memory2] 加载 {len(self.experiences)} 条经验")
        except Exception as e:
            print(f"[Memory2] 加载失败: {e}")
    
    def save(self, path: str = None):
        p = path or self.json_path
        if p:
            data = [e.to_dict() for e in self.experiences]
            with open(p, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"[Memory2] 保存 {len(self.experiences)} 条经验")
    
    def add(self, name: str, condition: Dict, action: str,
            template: str = '', confidence: float = 0.6,
            need_left: bool = False, prefer_position: str = '',
            distinguish_from: List[str] = None):
        """
        添加经验。
        
        Args:
            condition: 只存有区分度的标记
            distinguish_from: 用于从哪些同类中区分（自动计算差异标记）
        """
        exp = Experience(
            name=name,
            condition=condition,
            action=action,
            template=template,
            confidence=confidence,
            need_left=need_left,
            prefer_position=prefer_position,
            created_at=time.time(),
        )
        
        # 去重：如果已有同名且同模板的经验，合并
        for i, e in enumerate(self.experiences):
            if e.name == name and e.template == template:
                self.experiences[i] = exp
                return
        
        self.experiences.append(exp)
    
    def match(self, marks: Dict, context_marks: Dict = None) -> Tuple[Optional[Experience], float, float]:
        """
        WTA 竞争匹配。
        
        Args:
            marks: 当前视觉标记
            context_marks: 上下文标记（如地名、时间）
            
        Returns:
            (best_exp, win_score, runner_up_score)
            - best_exp=None: 无法确定
            - win_score < confidence_threshold: 需要左脑
        """
        if not self.experiences:
            return None, 0.0, 0.0
        
        # 先匹配模板，缩小搜索范围
        template_name, template_score = self.templates.match(marks)
        
        # 激活所有经验
        activations = []
        
        for exp in self.experiences:
            # 基础激活：条件匹配
            condition_score = self._compute_condition_match(marks, exp.condition)
            
            # 模板加分
            template_bonus = 0.0
            if exp.template and exp.template == template_name:
                template_bonus = 0.15 * template_score
            
            # 位置先验加分
            position_bonus = 0.0
            if exp.prefer_position and marks.get('位置区域') == exp.prefer_position:
                position_bonus = 0.1
            
            # 上下文加分（地名等）
            context_bonus = 0.0
            if context_marks:
                for k, v in context_marks.items():
                    if k in exp.condition and exp.condition[k] == v:
                        context_bonus += 0.2
            
            # 置信度加权
            total = (condition_score * 0.8 + template_bonus + position_bonus + context_bonus)
            total = total * exp.confidence
            
            if total > 0:
                activations.append((total, exp))
        
        if not activations:
            return None, 0.0, 0.0
        
        # 排序
        activations.sort(key=lambda x: x[0], reverse=True)
        
        best_score, best_exp = activations[0]
        runner_up_score = activations[1][0] if len(activations) > 1 else 0.0
        
        # WTA 判定
        if best_score < self.confidence_threshold:
            return None, best_score, runner_up_score
        
        if len(activations) >= 2:
            win_margin = (best_score - runner_up_score) / max(best_score, 0.01)
            if win_margin < self.wta_threshold:
                # 胜出不够明显，不确定
                return None, best_score, runner_up_score
        
        return best_exp, best_score, runner_up_score
    
    def _compute_condition_match(self, marks: Dict, condition: Dict) -> float:
        """计算条件匹配度。支持相同、近似、空间关系。"""
        if not condition:
            return 0.0
        
        matched = 0.0
        total = float(len(condition))
        
        for key, expected_val in condition.items():
            actual_val = marks.get(key)
            if actual_val is None:
                continue
            
            # 精确匹配
            if actual_val == expected_val:
                matched += 1.0
                continue
            
            # === 近似匹配 ===
            if key == '形状':
                from rightbrain.learning.memory import _shape_similar
                if _shape_similar(actual_val, expected_val):
                    matched += 0.7
                    continue
                # 长方形态≈长条形互认
                long_types = ['长方形', '长条形', '矩形']
                if actual_val in long_types and expected_val in long_types:
                    matched += 0.9
                    continue
            
            if key == '颜色':
                warm = ['红', '橙', '黄', '粉']
                blueish = ['蓝', '青', '紫']
                greenish = ['绿']
                neutral = ['白', '灰', '黑', '棕']
                for group in [warm, blueish, greenish, neutral]:
                    if actual_val in group and expected_val in group:
                        matched += 0.6
                        break
            
            # 位置区域宽容度：上方≈中间（山可能在画面中部）
            if key == '位置区域':
                if expected_val == '上方' and actual_val == '中间':
                    matched += 0.6
                elif expected_val == '地面' and actual_val == '中间':
                    matched += 0.6
        
        return matched / max(total, 1.0)
    
    def remove(self, name: str, template: str = ''):
        """删除经验"""
        self.experiences = [
            e for e in self.experiences
            if not (e.name == name and (not template or e.template == template))
        ]
    
    def rename(self, old_name: str, new_name: str):
        """重命名"""
        for e in self.experiences:
            if e.name == old_name:
                e.name = new_name
