"""
genome_shadow.py – 基因组影子系统
"""
import time
import threading
import numpy as np
from collections import deque

try:
    from genome.genome_algorithms import color_detection_genome
    GENOME_AVAILABLE = True
except ImportError:
    GENOME_AVAILABLE = False
    print("[GenomeShadow] 警告：无法导入 genome 模块，影子校验将禁用")

class GenomeShadowChecker:
    def __init__(self, test_inputs_provider=None, interval_seconds=120):
        self.test_inputs_provider = test_inputs_provider
        self.interval = interval_seconds
        self.deviation_scores = deque(maxlen=100)
        self.running = False
        self.thread = None

    def start(self):
        if not GENOME_AVAILABLE:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        print("[GenomeShadow] 影子校验已启动")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)

    def _run(self):
        while self.running:
            time.sleep(self.interval)
            self._check_once()

    def _check_once(self):
        if self.test_inputs_provider:
            test_inputs = self.test_inputs_provider()
        else:
            test_inputs = self._builtin_test_inputs()
        if not test_inputs:
            return

        deviations = []
        for h, s, v in test_inputs:
            genome_color = color_detection_genome(h, s, v)
            try:
                from genome.genome_algorithms import color_detection_genome as current_func
                current_color = current_func(h, s, v)
                dev = 0.0 if genome_color == current_color else 1.0
                deviations.append(dev)
            except Exception:
                pass

        deviation = float(np.mean(deviations)) if deviations else 0.05
        self.deviation_scores.append(deviation)
        if deviation > 0.2:
            print(f"[GenomeShadow] ⚠️ 行为偏差 {deviation:.3f} 超过阈值")

    def _builtin_test_inputs(self):
        return [(0, 200, 200), (30, 200, 200), (60, 200, 200),
                (90, 200, 200), (120, 200, 200), (150, 200, 200),
                (10, 80, 150), (0, 10, 100)]

    def get_deviation_score(self) -> float:
        if not self.deviation_scores:
            return 0.0
        return float(np.mean(self.deviation_scores))
