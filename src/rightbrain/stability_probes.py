"""
stability_probes.py – 稳定性探针
实时采样系统关键指标，不改变行为
"""
import time
import numpy as np
from collections import deque
from typing import Dict

class PerceptionProbe:
    def __init__(self, buffer_size=1000):
        self.confidence_history = deque(maxlen=buffer_size)
        self.shape_history = deque(maxlen=buffer_size)
        self.color_history = deque(maxlen=buffer_size)
        self.unknown_ratio_history = deque(maxlen=buffer_size)

    def record(self, marks: Dict):
        self.confidence_history.append(marks.get("深度置信度", 0.5))
        self.shape_history.append(marks.get("形状", "无"))
        self.color_history.append(marks.get("颜色", "未知"))
        unknown = 1 if marks.get("颜色") == "未知" or marks.get("形状") == "无" else 0
        self.unknown_ratio_history.append(unknown)

    def get_stats(self) -> Dict:
        if not self.confidence_history:
            return {"avg_confidence": 0.5, "unknown_ratio": 0, "color_entropy": 0, "shape_entropy": 0}
        avg_conf = float(np.mean(self.confidence_history))
        unknown_ratio = float(np.mean(self.unknown_ratio_history))
        
        colors = list(self.color_history)
        color_counts = {}
        for c in colors: color_counts[c] = color_counts.get(c, 0) + 1
        probs = np.array(list(color_counts.values())) / len(colors)
        color_entropy = float(-np.sum(probs * np.log2(probs + 1e-9))) if len(probs) > 0 else 0
        
        shapes = list(self.shape_history)
        shape_counts = {}
        for s in shapes: shape_counts[s] = shape_counts.get(s, 0) + 1
        probs = np.array(list(shape_counts.values())) / len(shapes)
        shape_entropy = float(-np.sum(probs * np.log2(probs + 1e-9))) if len(probs) > 0 else 0
        
        return {
            "avg_confidence": round(avg_conf, 4),
            "unknown_ratio": round(unknown_ratio, 4),
            "color_entropy": round(color_entropy, 4),
            "shape_entropy": round(shape_entropy, 4),
            "sample_count": len(self.confidence_history),
        }

class MemoryProbe:
    def __init__(self, memory_instance, buffer_size=100):
        self.memory = memory_instance
        self.history = deque(maxlen=buffer_size)
        self._snapshot()

    def _snapshot(self):
        exps = self.memory.experiences
        if not exps:
            return
        cond_count = {}
        for exp in exps:
            cond = tuple(sorted(exp.get("condition", {}).items()))
            cond_count[cond] = cond_count.get(cond, 0) + 1
        dup_ratio = sum(1 for c in cond_count.values() if c > 1) / len(cond_count) if cond_count else 0
        
        action_map = {}
        for exp in exps:
            cond = tuple(sorted(exp.get("condition", {}).items()))
            if cond not in action_map: action_map[cond] = set()
            action_map[cond].add(exp.get("action", ""))
        con_count = sum(1 for ac in action_map.values() if len(ac) > 1)
        con_ratio = con_count / len(action_map) if action_map else 0
        
        self.history.append({
            "timestamp": time.time(),
            "total_experiences": len(exps),
            "duplicate_ratio": dup_ratio,
            "contradiction_ratio": con_ratio,
        })

    def record(self):
        self._snapshot()

    def get_stats(self) -> Dict:
        if not self.history:
            return {"total_experiences": 0, "duplicate_ratio": 0, "contradiction_ratio": 0, "new_entries_rate": 0}
        latest = self.history[-1]
        new_rate = 0
        if len(self.history) > 1:
            prev = self.history[-2]
            new_rate = (latest["total_experiences"] - prev["total_experiences"]) / max(1, prev["total_experiences"])
        return {
            "total_experiences": latest["total_experiences"],
            "duplicate_ratio": round(latest["duplicate_ratio"], 4),
            "contradiction_ratio": round(latest["contradiction_ratio"], 4),
            "new_entries_rate": round(new_rate, 4),
        }

class DecisionProbe:
    def __init__(self, buffer_size=500):
        self.action_history = deque(maxlen=buffer_size)
        self.threshold_history = deque(maxlen=buffer_size)
        self.question_history = deque(maxlen=buffer_size)

    def record(self, action: str, threshold: float, is_question: bool = False):
        self.action_history.append(action)
        self.threshold_history.append(threshold)
        self.question_history.append(1 if is_question else 0)

    def get_stats(self) -> Dict:
        if not self.action_history:
            return {"action_entropy": 0, "repeat_action_rate": 0, "question_rate": 0, "threshold_volatility": 0}
        actions = list(self.action_history)
        action_counts = {}
        for a in actions: action_counts[a] = action_counts.get(a, 0) + 1
        probs = np.array(list(action_counts.values())) / len(actions)
        action_entropy = float(-np.sum(probs * np.log2(probs + 1e-9)))
        repeat_count = sum(1 for i in range(1, len(actions)) if actions[i] == actions[i-1])
        repeat_rate = repeat_count / (len(actions) - 1) if len(actions) > 1 else 0
        question_rate = float(np.mean(self.question_history))
        threshold_volatility = float(np.std(self.threshold_history)) if len(self.threshold_history) > 1 else 0
        return {
            "action_entropy": round(action_entropy, 4),
            "repeat_action_rate": round(repeat_rate, 4),
            "question_rate": round(question_rate, 4),
            "threshold_volatility": round(threshold_volatility, 4),
        }

# 全局实例
perception_probe = PerceptionProbe()
memory_probe = None
decision_probe = DecisionProbe()

def init_memory_probe(memory_instance):
    global memory_probe
    memory_probe = MemoryProbe(memory_instance)
