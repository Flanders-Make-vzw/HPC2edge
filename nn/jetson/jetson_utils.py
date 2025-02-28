import threading
import time
from jtop import jtop


class JetsonMonitor:
    """Class for monitoring Jetson device using jtop."""

    def __init__(self, interval=0.5):
        self.cpu_temps = []
        self.gpu_temps = []
        self.ram_usage = []
        self.gpu_usage = []
        self.power_usage = []
        self.cpu_usage = []

        self._jtop = jtop(interval=interval)
        self._jtop_thread = None
        self._run = False

    def _jtop_loop(self):
        self._jtop.start()
        while self._run and self._jtop.ok():
            self.cpu_temps.append(self._jtop.stats["Temp CPU"])
            self.gpu_temps.append(self._jtop.stats["Temp GPU"])
            self.ram_usage.append(self._jtop.stats["RAM"])
            self.gpu_usage.append(self._jtop.stats["GPU"])
            self.power_usage.append(self._jtop.stats["Power TOT"])
            self.cpu_usage.append(self._jtop.cpu["total"]["user"])
        self._jtop.close()

    def start(self):
        self._jtop_thread = threading.Thread(target=self._jtop_loop)
        self._run = True
        self._jtop_thread.start()

    def stop(self):
        if self._jtop_thread is not None:
            self._run = False
            self._jtop_thread.join()

    def get_stats(self):
        if len(self.cpu_temps) == 0:
            return
        return {
            "avg_cpu_temps": sum(self.cpu_temps) / len(self.cpu_temps),
            "min_cpu_temps": min(self.cpu_temps),
            "max_cpu_temps": max(self.cpu_temps),
            "avg_gpu_temps": sum(self.gpu_temps) / len(self.gpu_temps),
            "min_gpu_temps": min(self.gpu_temps),
            "max_gpu_temps": max(self.gpu_temps),
            "avg_ram_usage": sum(self.ram_usage) / len(self.ram_usage),
            "min_ram_usage": min(self.ram_usage),
            "max_ram_usage": max(self.ram_usage),
            "avg_gpu_usage": sum(self.gpu_usage) / len(self.gpu_usage),
            "min_gpu_usage": min(self.gpu_usage),
            "max_gpu_usage": max(self.gpu_usage),
            "avg_power_usage": sum(self.power_usage) / len(self.power_usage),
            "min_power_usage": min(self.power_usage),
            "max_power_usage": max(self.power_usage),
            "avg_cpu_usage": sum(self.cpu_usage) / len(self.cpu_usage),
            "min_cpu_usage": min(self.cpu_usage),
            "max_cpu_usage": max(self.cpu_usage),
        }
