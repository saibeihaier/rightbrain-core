"""
RightBrain Stability Layer v1 — 统一调度层

调用四个独立探针模块：
  stability_probes.py    — 感知/记忆/决策探针
  memory_health.py       — 记忆健康评分
  decision_stabilizer.py — 决策稳定器
  genome_shadow.py       — 影子行为校验

额外提供：
  - 综合漂移分数
  - 反馈阻尼
  - 定时调度
"""
import time
import json
import os
import math
from typing import Dict, Optional

# 导入四个探针模块
from rightbrain.stability_probes import (
    perception_probe, memory_probe, decision_probe, init_memory_probe
)
from rightbrain.memory_health import MemoryHealthMonitor
from rightbrain.decision_stabilizer import decision_stabilizer
from rightbrain.genome_shadow import GenomeShadowChecker

DRIFT_WARNING_THRESHOLD = 0.3
MEMORY_HEALTH_LOW = 0.5

_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "logs")
os.makedirs(_LOG_DIR, exist_ok=True)
_STABILITY_LOG = os.path.join(_LOG_DIR, "stability.jsonl")


def _log_stability(event: str, data: dict):
    try:
        record = {"t": time.time(), "event": event, "data": data}
        with open(_STABILITY_LOG, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


class StabilityLayer:
    """
    稳定性层统一调度器。
    
    用法:
        st = StabilityLayer(memory_instance)
        st.probe_perception(marks)       # 每帧调用
        st.probe_decision(action, th)     # 每次决策调用
        clamped = st.stabilize_threshold(name, val)  # 阈值稳定
        damped = st.dampen_update(w, u, n)            # 反馈阻尼
        drift = st.get_drift_score()      # 定时检查
        health = st.get_memory_health()
    """
    
    def __init__(self, memory=None):
        # 探针
        self.perception = perception_probe
        self.decision_probe = decision_probe
        self._memory = memory
        
        # 记忆健康
        self.memory_health = None
        if memory:
            init_memory_probe(memory)
            self.memory_health = MemoryHealthMonitor(memory)
        
        # 影子校验
        self.shadow = GenomeShadowChecker(interval_seconds=300)
        self.shadow.start()
        
        # 反馈阻尼状态
        self._pattern_counts = {}
        
        self._last_log_time = 0
        print("[稳定性] ✅ Stability Layer v1 已加载（四探针模式）")
    
    # ----- 探针接口（转发） -----
    
    def probe_perception(self, marks: dict):
        self.perception.record(marks)
    
    def probe_decision(self, action: str, threshold: float, is_question: bool = False):
        self.decision_probe.record(action, threshold, is_question)
    
    def probe_memory(self):
        if memory_probe:
            memory_probe.record()
    
    # ----- 阈值稳定（转发） -----
    
    def stabilize_threshold(self, name: str, new_value: float,
                           min_val: float = 0.0, max_val: float = 1.0) -> float:
        """通过决策稳定器钳制阈值变化，再钳制到基因范围"""
        clamped = decision_stabilizer.clamp_threshold(new_value)
        clamped = max(min_val, min(max_val, clamped))
        return clamped
    
    # ----- 记忆健康 -----
    
    def get_memory_health(self) -> float:
        if not self.memory_health:
            return 1.0
        result = self.memory_health.compute_health()
        score = result.get("health_score", 1.0)
        if score < MEMORY_HEALTH_LOW:
            self._log_event("memory_health_low", result)
        return score
    
    def get_memory_health_detail(self) -> dict:
        if not self.memory_health:
            return {}
        return self.memory_health.compute_health()
    
    # ----- 综合漂移 -----
    
    def get_drift_score(self) -> float:
        p = self.perception.get_stats()
        m = memory_probe.get_stats() if memory_probe else {}
        d = self.decision_probe.get_stats()
        
        # 特征漂移：置信度下降 + 未知率上升
        fd = (1.0 - p.get("avg_confidence", 0.5)) * 0.5 + p.get("unknown_ratio", 0) * 0.5
        # 行为漂移：动作重复率 + 提问率
        bd = d.get("repeat_action_rate", 0) * 0.6 + d.get("question_rate", 0) * 0.4
        # 记忆漂移：冗余率 + 冲突率
        md = m.get("duplicate_ratio", 0) * 0.6 + m.get("contradiction_ratio", 0) * 0.4
        
        total = 0.4 * fd + 0.3 * bd + 0.3 * md
        if total > DRIFT_WARNING_THRESHOLD:
            self._log_event("drift_warning", {
                "total": round(total, 3), "feature": round(fd, 3),
                "behavior": round(bd, 3), "memory": round(md, 3),
            })
        return round(total, 4)
    
    # ----- 影子校验 -----
    
    def get_shadow_deviation(self) -> float:
        return self.shadow.get_deviation_score()
    
    # ----- 反馈阻尼 -----
    
    def dampen_update(self, current_weight: float, update_amount: float,
                     pattern_count: int, damping_base: float = 0.1) -> float:
        damping = damping_base * math.log(pattern_count + 1)
        return update_amount / (1.0 + damping)
    
    def count_pattern(self, key: str):
        """记录一个模式的出现次数（用于阻尼）"""
        self._pattern_counts[key] = self._pattern_counts.get(key, 0) + 1
    
    def get_pattern_count(self, key: str) -> int:
        return self._pattern_counts.get(key, 0)
    
    # ----- 摘要 -----
    
    def get_summary(self) -> str:
        drift = self.get_drift_score()
        health = self.get_memory_health()
        p = self.perception.get_stats()
        shadow = self.get_shadow_deviation()
        
        parts = [f"漂移:{drift:.2f}", f"记忆:{health:.2f}",
                 f"样本:{p.get('sample_count',0)}", f"影子:{shadow:.3f}"]
        if drift > DRIFT_WARNING_THRESHOLD: parts.append("⚠️漂移")
        if health < MEMORY_HEALTH_LOW: parts.append("⚠️记忆")
        return " | ".join(parts)
    
    def _log_event(self, event: str, data: dict):
        now = time.time()
        if now - self._last_log_time < 1:
            return
        self._last_log_time = now
        _log_stability(event, data)
        print(f"[稳定性] {event}")


_stability = None

def get_stability(memory=None):
    global _stability
    if _stability is None:
        _stability = StabilityLayer(memory)
    return _stability
