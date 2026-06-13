"""
boundary_enforcer.py – RightBrain 边界执行层 + 自适应稳定器

两层结构：
  ConfigBoundary      — 边界守护：钳制、漂移检测、一键恢复
  AdaptiveStabilizer  — 自适应稳定器：检测到漂移后自动回调

用法：
    from boundary_enforcer import ConfigBoundary, AdaptiveStabilizer
    boundary = ConfigBoundary(Config)
    stabilizer = AdaptiveStabilizer(boundary)
    # 漂移检测 + 自动回调
    drift = boundary.detect_drift()
    if drift["drift_detected"]:
        stabilizer.auto_correct(drift["details"])
"""
import copy
import time
import numpy as np
from collections import deque
from typing import Any, Dict, Generator, Tuple


class ConfigBoundary:
    """配置边界守护：所有对 Config 的修改都必须通过此层"""

    def __init__(self, config_instance):
        self.config = config_instance
        self._original_state = self._capture_state()
        self._clamp_rules = self._build_clamp_rules()
        self._history = deque(maxlen=200)

    def _capture_state(self) -> Dict:
        state = {}
        for key in dir(self.config):
            if key.isupper() and not key.startswith("_"):
                val = getattr(self.config, key)
                if isinstance(val, (int, float)):
                    state[key] = val
        return state

    def _build_clamp_rules(self) -> Dict[str, tuple]:
        rules = {}
        try:
            import json, os
            spec_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "genome", "genome_spec.json")
            with open(spec_path) as f:
                spec = json.load(f)
            constraints = spec.get("core_constraints", {})
            for key, value in constraints.items():
                if isinstance(value, dict) and "min" in value and "max" in value:
                    config_key = key.upper()
                    rules[config_key] = (float(value["min"]), float(value["max"]))
        except Exception as e:
            print(f"[Boundary] ⚠️ 无法加载 genome_spec.json: {e}")
        return rules

    def clamp(self, key: str, value: Any) -> Any:
        if key in self._clamp_rules:
            minv, maxv = self._clamp_rules[key]
            if isinstance(value, (int, float)):
                if value < minv:
                    print(f"[Boundary] ⚠️ {key}={value} 低于最小值{minv}，已截断")
                    return minv
                if value > maxv:
                    print(f"[Boundary] ⚠️ {key}={value} 超过最大值{maxv}，已截断")
                    return maxv
        return value

    def set_config(self, key: str, value: Any):
        clamped = self.clamp(key, value)
        setattr(self.config, key, clamped)
        self._record_snapshot()

    def _record_snapshot(self):
        snapshot = {}
        for key in self._clamp_rules.keys():
            snapshot[key] = getattr(self.config, key, None)
        self._history.append(snapshot)

    def detect_drift(self, min_samples=20) -> Dict:
        if len(self._history) < min_samples:
            return {"drift_detected": False, "reason": f"数据不足({len(self._history)}/{min_samples})"}
        drift_info = {}
        for key in self._clamp_rules.keys():
            values = [h.get(key) for h in self._history if h.get(key) is not None]
            if len(values) < 10:
                continue
            x = np.arange(len(values))
            slope = np.polyfit(x, values, 1)[0]
            if abs(slope) > 0.005:
                drift_info[key] = round(slope, 6)
        if drift_info:
            print(f"[Boundary] ⚠️ 配置趋势偏移: {drift_info}")
            return {"drift_detected": True, "details": drift_info}
        return {"drift_detected": False}

    def restore_original(self):
        for key, value in self._original_state.items():
            setattr(self.config, key, value)
        self._history.clear()
        print("[Boundary] ✅ 配置已恢复至系统初始值")


class AdaptiveStabilizer:
    """
    自适应稳定器 — 检测到漂移后自动回调。
    
    两层阈值：
      suggestion_threshold (0.01) — 超过此值只建议，不自动执行
      auto_correct_threshold (0.15) — 超过此值自动回调 0.05
    
    每次回调 0.05（小步、可逆），不会一次调太多。
    """

    def __init__(self, boundary: ConfigBoundary,
                 suggestion_threshold: float = 0.01,
                 auto_correct_threshold: float = 0.005,
                 correction_step: float = 0.05):
        self.boundary = boundary
        self.suggestion_threshold = suggestion_threshold
        self.auto_correct_threshold = auto_correct_threshold
        self.correction_step = correction_step
        self._known_good = {}  # 最近一次确认的稳定状态
        self._last_mark_time = 0
        self._correction_count = 0

    def mark_stable(self):
        """标记当前状态为'已知稳定'，漂移时优先回到此值"""
        for key in self.boundary._clamp_rules.keys():
            val = getattr(self.boundary.config, key, None)
            if val is not None:
                self._known_good[key] = val
        self._last_mark_time = time.time()

    def suggest_correction(self, drift_info: Dict) -> Generator[Tuple[str, float], None, None]:
        """根据漂移信息，生成配置调整建议（不自动执行）"""
        for key, slope in drift_info.items():
            if abs(slope) < self.suggestion_threshold:
                continue
            current = getattr(self.boundary.config, key, None)
            if current is None:
                continue
            # 持续向上漂移 → 向下回调；持续向下漂移 → 向上回调
            direction = -1.0 if slope > 0 else 1.0
            suggested = current + direction * self.correction_step
            # 钳制到基因范围
            clamped = self.boundary.clamp(key, suggested)
            if clamped != current:
                yield (key, clamped)

    def auto_correct(self, drift_info: Dict):
        """
        自动纠正严重漂移。
        有已知好状态时优先回到好状态。
        否则回调 step。
        """
        corrections = []
        for key, slope in drift_info.items():
            if float(abs(slope)) < self.auto_correct_threshold:
                continue
            current = getattr(self.boundary.config, key, None)
            if current is None:
                continue

            # 优先回到已知好状态
            if key in self._known_good:
                target = self._known_good[key]
                # 如果当前值和好状态不同，直接回调到好状态
                if current != target:
                    clamped = self.boundary.clamp(key, target)
                    if clamped != current:
                        self.boundary.set_config(key, clamped)
                        corrections.append((key, current, clamped, "→好状态"))
                        self._correction_count += 1
                        continue

            # 无好状态，硬回调 step
            direction = -1.0 if slope > 0 else 1.0
            new_val = current + direction * self.correction_step
            clamped = self.boundary.clamp(key, new_val)
            if clamped != current:
                self.boundary.set_config(key, clamped)
                corrections.append((key, current, clamped, "→step"))
                self._correction_count += 1

        if corrections:
            for key, old, new, method in corrections:
                print(f"[Adaptive] 🔄 {key}: {old:.3f} → {new:.3f} {method}")
        else:
            for key, suggested in self.suggest_correction(drift_info):
                print(f"[Adaptive] 💡 建议 {key}: → {suggested:.3f}")

    def get_stats(self) -> str:
        parts = [f"纠正{self._correction_count}次"]
        if self._known_good:
            parts.append(f"好状态锚点{len(self._known_good)}个")
        return " | ".join(parts)
