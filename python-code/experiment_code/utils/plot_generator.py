import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np

def generate_plots(results, environment, use_local_models):
    output_dir = os.path.join("experiment_results", environment, "local" if use_local_models else "api", "plots")
    os.makedirs(output_dir, exist_ok=True)

    # Prepare data
    data = []
    for model, result in results.items():
        whisper_model, gpt_model = model.split('_')
        data.append({
            'Whisper Model': whisper_model,
            'GPT Model': gpt_model,
            'Transcription Time': result['performance_logs']['transcription'],
            'Translation Time': result['performance_logs']['translation'],
            'Total Time': result['performance_logs']['total'],
            'Sentiment Score': np.mean([sent['score'] for sent in result.get('sentiment_analysis', [])]) if 'sentiment_analysis' in result else None,
            'Readability Score': result.get('text_analysis', {}).get('readability')
        })
    df = pd.DataFrame(data)

    # Generate plots
    generate_performance_plots(df, output_dir)
    generate_sentiment_plots(df, output_dir)
    generate_readability_plots(df, output_dir)
    generate_correlation_plot(df, output_dir)

def generate_performance_plots(df, output_dir):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Whisper Model', y='Transcription Time', data=df)
    plt.title("Transcription Time by Whisper Model")
    plt.savefig(os.path.join(output_dir, "transcription_time_boxplot.png"))
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.barplot(x='Whisper Model', y='Total Time', hue='GPT Model', data=df)
    plt.title("Total Processing Time by Model Combination")
    plt.savefig(os.path.join(output_dir, "total_time_barplot.png"))
    plt.close()

def generate_sentiment_plots(df, output_dir):
    plt.figure(figsize=(12, 6))
    sns.violinplot(x='Whisper Model', y='Sentiment Score', data=df)
    plt.title("Sentiment Score Distribution by Whisper Model")
    plt.savefig(os.path.join(output_dir, "sentiment_score_violin.png"))
    plt.close()

def generate_readability_plots(df, output_dir):
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='Readability Score', y='Sentiment Score', hue='Whisper Model', style='GPT Model', data=df)
    plt.title("Readability vs Sentiment Score")
    plt.savefig(os.path.join(output_dir, "readability_vs_sentiment.png"))
    plt.close()

def generate_correlation_plot(df, output_dir):
    correlation_matrix = df[['Transcription Time', 'Translation Time', 'Total Time', 'Sentiment Score', 'Readability Score']].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix of Performance Metrics")
    plt.savefig(os.path.join(output_dir, "correlation_matrix.png"))
    plt.close()