"""
memory_health.py – 记忆健康评分
"""
import time
import numpy as np
from typing import Dict

class MemoryHealthMonitor:
    def __init__(self, memory_instance):
        self.memory = memory_instance
        self.score_history = []

    def compute_health(self) -> Dict:
        exps = self.memory.experiences
        if not exps:
            return {"health_score": 0.5, "redundancy_rate": 0, "contradiction_rate": 0, "diversity_score": 0, "staleness_ratio": 0}
        
        cond_map = {}
        for exp in exps:
            cond = tuple(sorted(exp.get("condition", {}).items()))
            cond_map[cond] = cond_map.get(cond, 0) + 1
        redundancy_rate = sum(1 for c in cond_map.values() if c > 2) / len(cond_map) if cond_map else 0
        
        action_map = {}
        for exp in exps:
            cond = tuple(sorted(exp.get("condition", {}).items()))
            if cond not in action_map: action_map[cond] = set()
            action_map[cond].add(exp.get("action", ""))
        contradiction_rate = sum(1 for ac in action_map.values() if len(ac) > 1) / len(action_map) if action_map else 0
        
        all_actions = [exp.get("action", "") for exp in exps]
        action_counts = {}
        for a in all_actions: action_counts[a] = action_counts.get(a, 0) + 1
        probs = np.array(list(action_counts.values())) / len(all_actions)
        diversity_score = float(-np.sum(probs * np.log2(probs + 1e-9)))
        max_entropy = np.log2(len(action_counts)) if action_counts else 1
        diversity_score = diversity_score / max_entropy if max_entropy > 0 else 0
        
        now = time.time()
        stale_threshold = 7 * 24 * 3600
        stale_count = sum(1 for exp in exps if now - exp.get("last_matched_time", 0) > stale_threshold)
        staleness_ratio = stale_count / len(exps)
        
        health_score = 1.0 - (0.3 * redundancy_rate + 0.3 * contradiction_rate + 0.1 * staleness_ratio - 0.3 * diversity_score)
        health_score = max(0.0, min(1.0, health_score))
        
        return {
            "health_score": round(health_score, 4),
            "redundancy_rate": round(redundancy_rate, 4),
            "contradiction_rate": round(contradiction_rate, 4),
            "diversity_score": round(diversity_score, 4),
            "staleness_ratio": round(staleness_ratio, 4),
            "total_experiences": len(exps),
        }

    def get_health_trend(self, window=10) -> float:
        if len(self.score_history) < window:
            return 0.0
        recent = self.score_history[-window:]
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]
        return slope
