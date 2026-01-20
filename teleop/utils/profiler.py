import time
from collections import defaultdict
from typing import Dict

class LoopProfiler:
    def __init__(self, report_period: float = 1.0):
        self.report_period = float(report_period)

        self._t_start = time.perf_counter()
        self._t_last_report = self._t_start
        self._n = 0

        self.sum_dt: Dict[str, float] = defaultdict(float)
        self.max_dt: Dict[str, float] = defaultdict(float)
        self.cnt_dt: Dict[str, int] = defaultdict(int)

    def add(self, name: str, dt: float):
        """구간 측정 누적"""
        self.sum_dt[name] += dt
        self.cnt_dt[name] += 1
        if dt > self.max_dt[name]:
            self.max_dt[name] = dt

    def tick_loop(self):
        """루프 한 번 돌았음을 기록"""
        self._n += 1

    def should_report(self) -> bool:
        return (time.perf_counter() - self._t_last_report) >= self.report_period

    def report_and_reset(self, top_k: int = 6):
        now = time.perf_counter()
        elapsed = now - self._t_last_report
        hz = self._n / max(elapsed, 1e-9)

        # 평균/최대 loop_total 계산(있다면)
        loop_avg_ms = None
        loop_max_ms = None
        if "loop_total" in self.sum_dt and self.cnt_dt["loop_total"] > 0:
            loop_avg_ms = 1000.0 * (self.sum_dt["loop_total"] / self.cnt_dt["loop_total"])
            loop_max_ms = 1000.0 * self.max_dt["loop_total"]

        # 항목 정렬(평균 기준 내림차순)
        items = []
        for k, s in self.sum_dt.items():
            c = self.cnt_dt[k]
            if c <= 0:
                continue
            avg_ms = 1000.0 * (s / c)
            mx_ms = 1000.0 * self.max_dt[k]
            items.append((avg_ms, mx_ms, k, c))
        items.sort(reverse=True)

        print(f"\n[PERF] window={elapsed:.2f}s  loops={self._n}  hz={hz:.1f}", end="")
        if loop_avg_ms is not None:
            print(f"  loop_avg={loop_avg_ms:.2f}ms  loop_max={loop_max_ms:.2f}ms")
        else:
            print()

        for avg_ms, mx_ms, k, c in items[:top_k]:
            print(f"  - {k:16s} avg={avg_ms:7.2f}ms  max={mx_ms:7.2f}ms  n={c}")

        # 리셋
        self._t_last_report = now
        self._n = 0
        self.sum_dt.clear()
        self.max_dt.clear()
        self.cnt_dt.clear()
