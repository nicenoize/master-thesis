import time
import json
import os
from contextlib import contextmanager
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class PerformanceLogger:
    def __init__(self):
        self.logs = {}

    @contextmanager
    def measure_time(self, operation):
        start_time = time.time()
        yield
        end_time = time.time()
        self.logs.setdefault(operation, []).append(end_time - start_time)

    def generate_report(self, environment, use_local_models):
        report = {op: {"mean": sum(times) / len(times), "total": sum(times), "min": min(times), "max": max(times)} 
                  for op, times in self.logs.items()}
        
        output_dir = os.path.join("experiment_results", environment, "local" if use_local_models else "api", "performance")
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, "performance_report.json"), "w") as f:
            json.dump(report, f, indent=2)

        # Generate CSV report
        df = pd.DataFrame([(op, stat, value) 
                           for op, stats in report.items() 
                           for stat, value in stats.items()],
                          columns=["Operation", "Statistic", "Value"])
        df.to_csv(os.path.join(output_dir, "performance_report.csv"), index=False)

    def plot_performance(self, environment, use_local_models):
        output_dir = os.path.join("experiment_results", environment, "local" if use_local_models else "api", "performance")
        os.makedirs(output_dir, exist_ok=True)

        df = pd.DataFrame([(op, time) for op, times in self.logs.items() for time in times],
                          columns=["Operation", "Time"])

        plt.figure(figsize=(12, 6))
        sns.boxplot(x="Operation", y="Time", data=df)
        plt.title("Performance Distribution by Operation")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "performance_boxplot.png"))
        plt.close()

        plt.figure(figsize=(12, 6))
        sns.violinplot(x="Operation", y="Time", data=df)
        plt.title("Performance Distribution by Operation (Violin Plot)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "performance_violin.png"))
        plt.close()