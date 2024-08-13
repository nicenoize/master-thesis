import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def combine_performance_data(output_dir):
    combined_data = {
        "transcription": [],
        "translation": [],
        "analysis": [],
        "total_processing": []
    }
    
    for filename in os.listdir(os.path.join(output_dir, "performance_logs")):
        if filename.endswith("_logs.json"):
            environment = filename.split("_logs.json")[0]
            with open(os.path.join(output_dir, "performance_logs", filename), 'r') as f:
                data = json.load(f)
                for metric in combined_data.keys():
                    for model, times in data[metric].items():
                        for time in times:
                            combined_data[metric].append({
                                "Environment": environment,
                                "Model": model,
                                "Time": time
                            })
    
    return combined_data

def create_comparative_graphs(combined_data, output_dir):
    for metric, data in combined_data.items():
        df = pd.DataFrame(data)
        
        plt.figure(figsize=(15, 10))
        sns.boxplot(x="Environment", y="Time", hue="Model", data=df)
        plt.title(f"{metric.capitalize()} Performance Comparison")
        plt.ylabel("Time (seconds)")
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric}_comparison.png"))
        plt.close()
        
        # Create a heatmap for average times
        pivot_df = df.pivot_table(values='Time', index='Environment', columns='Model', aggfunc='mean')
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="YlOrRd")
        plt.title(f"Average {metric.capitalize()} Time Heatmap")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric}_heatmap.png"))
        plt.close()

def main():
    output_dir = "output"  # Update this if your output directory is different
    combined_data = combine_performance_data(output_dir)
    create_comparative_graphs(combined_data, output_dir)
    print("Comparative graphs have been created and saved in the output directory.")

if __name__ == "__main__":
    main()