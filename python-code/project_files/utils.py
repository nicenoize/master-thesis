# utils.py

import os
import sys
import json
import signal
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import config  # Import the entire config module
from classes import Conversation

conversation = Conversation()
logger = config.logger  # Use logger from config

def validate_file_path(file_path):
    if not file_path:
        return False, "File path is empty."
    if not os.path.exists(file_path):
        return False, f"File does not exist: {file_path}"
    if not os.path.isfile(file_path):
        return False, f"Path is not a file: {file_path}"
    return True, ""

def signal_handler(sig, frame):
    print("\nCtrl+C detected. Saving current state and exiting...")
    save_current_state()
    sys.exit(0)

def save_current_state():
    conversation.save_to_files()
    save_performance_logs()
    generate_performance_plots()
    if not config.EXPERIMENT_COMPLETED:
        with open(os.path.join(config.OUTPUT_DIR, "incomplete_experiment.txt"), "w") as f:
            f.write(
                f"Experiment interrupted.\nLast models used: GPT - {config.CURRENT_GPT_MODEL}, Whisper - {config.CURRENT_WHISPER_MODEL}"
            )

def num_tokens_from_string(string: str, model_name: str) -> int:
    import tiktoken

    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        # If model is not recognized, use a default encoding
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(string))


def save_performance_logs():
    os.makedirs(os.path.join(config.OUTPUT_DIR, "performance_logs"), exist_ok=True)
    with open(
        os.path.join(config.OUTPUT_DIR, "performance_logs", f"{config.CURRENT_ENVIRONMENT}_logs.json"), "w"
    ) as f:
        json.dump(config.PERFORMANCE_LOGS, f)

def load_performance_logs(environment):
    try:
        with open(os.path.join(config.OUTPUT_DIR, "performance_logs", f"{environment}_logs.json"), "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def generate_performance_plots():
    import numpy as np

    environments = config.ENVIRONMENTS
    gpt_models = ["gpt-4", "gpt-4-0613"]
    whisper_models = ["base", "small", "medium", "large"]

    # Load data for all environments
    all_data = {env: load_performance_logs(env) for env in environments}

    # Plotting functions
    def plot_boxplot(data, metric, title, filename):
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=data)
        plt.title(title)
        plt.ylabel("Time (seconds)")
        plt.xlabel("Configuration")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(config.OUTPUT_DIR, "plots", filename), dpi=300, bbox_inches="tight")
        plt.close()

    def plot_violin(data, metric, title, filename):
        plt.figure(figsize=(12, 6))
        sns.violinplot(data=data)
        plt.title(title)
        plt.ylabel("Time (seconds)")
        plt.xlabel("Configuration")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(config.OUTPUT_DIR, "plots", filename), dpi=300, bbox_inches="tight")
        plt.close()

    def plot_bar(data, metric, title, filename):
        means = [np.mean(d) for d in data]
        std_devs = [np.std(d) for d in data]

        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(data)), means, yerr=std_devs, capsize=5)
        plt.title(title)
        plt.ylabel("Mean Time (seconds)")
        plt.xlabel("Configuration")
        plt.xticks(range(len(data)), [f"Config {i+1}" for i in range(len(data))], rotation=45, ha="right")

        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        plt.savefig(os.path.join(config.OUTPUT_DIR, "plots", filename), dpi=300, bbox_inches="tight")
        plt.close()

    # Create plots directory
    os.makedirs(os.path.join(config.OUTPUT_DIR, "plots"), exist_ok=True)

    # Generate plots for each metric
    for metric in ["transcription", "translation", "analysis", "total_processing"]:
        for env in environments:
            if all_data[env]:
                data = []
                labels = []
                for gpt_model in gpt_models:
                    for whisper_model in whisper_models:
                        key = f"api_{gpt_model}_{whisper_model}"
                        if key in all_data[env][metric]:
                            data.append(all_data[env][metric][key])
                            labels.append(f"{env}\n{gpt_model}\n{whisper_model}")

                if data:
                    plot_boxplot(
                        data,
                        metric,
                        f"{metric.capitalize()} Time Comparison - {env}",
                        f"{env}_{metric}_boxplot.png",
                    )
                    plot_violin(
                        data,
                        metric,
                        f"{metric.capitalize()} Time Distribution - {env}",
                        f"{env}_{metric}_violin.png",
                    )
                    plot_bar(
                        data,
                        metric,
                        f"Mean {metric.capitalize()} Time Comparison - {env}",
                        f"{env}_{metric}_bar.png",
                    )

    # Statistical analysis
    def perform_anova(data, metric_name):
        if len(data) < 2:
            logger.warning(f"Not enough data to perform ANOVA for {metric_name}. Skipping ANOVA.")
            return
        f_statistic, p_value = stats.f_oneway(*data)
        with open(os.path.join(config.OUTPUT_DIR, "plots", "statistical_analysis.txt"), "a") as f:
            f.write(f"{metric.capitalize()} ANOVA Results:\n")
            f.write(f"F-statistic: {f_statistic}\n")
            f.write(f"p-value: {p_value}\n\n")

    for metric in ["transcription", "translation", "analysis", "total_processing"]:
        for env in environments:
            if all_data[env]:
                data = [
                    all_data[env][metric][key]
                    for key in all_data[env][metric]
                    if key.startswith("api_")
                ]
                if data:
                    perform_anova(data, f"{metric}_{env}")
