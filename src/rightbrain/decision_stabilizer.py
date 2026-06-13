"""
decision_stabilizer.py – 决策稳定器
防止决策参数慢性退化、动作反复横跳
"""
import numpy as np
from collections import deque, Counter

class DecisionStabilizer:
    def __init__(self, delta_max=0.05, smooth_alpha=0.3, action_window=10):
        self.delta_max = delta_max
        self.smooth_alpha = smooth_alpha
        self.action_window = action_window
        self.last_threshold = 0.6
        self.last_action = None
        self.action_history = deque(maxlen=action_window)
        self.smoothed_threshold = 0.6

    def clamp_threshold(self, new_threshold: float) -> float:
        delta = new_threshold - self.last_threshold
        if abs(delta) > self.delta_max:
            new_threshold = self.last_threshold + self.delta_max * (1 if delta > 0 else -1)
            print(f"[DecisionStabilizer] 阈值变化过大，已钳制: {self.last_threshold:.2f} -> {new_threshold:.2f}")
        self.last_threshold = new_threshold
        self.smoothed_threshold = self.smooth_alpha * new_threshold + (1 - self.smooth_alpha) * self.smoothed_threshold
        return self.smoothed_threshold

    def stabilize_action(self, action: str) -> str:
        self.action_history.append(action)
        if len(self.action_history) < self.action_window:
            return action
        changes = sum(1 for i in range(1, len(self.action_history)) if self.action_history[i] != self.action_history[i-1])
        change_rate = changes / (len(self.action_history) - 1)
        if change_rate > 0.6:
            most_common = Counter(self.action_history).most_common(1)[0][0]
            print(f"[DecisionStabilizer] 动作变化过快，稳定为: {most_common}")
            return most_common
        return action

    def record_threshold(self, threshold: float):
        self.last_threshold = threshold
        self.smoothed_threshold = self.smooth_alpha * threshold + (1 - self.smooth_alpha) * self.smoothed_threshold

decision_stabilizer = DecisionStabilizer()
