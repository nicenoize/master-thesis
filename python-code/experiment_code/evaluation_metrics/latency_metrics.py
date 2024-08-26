from typing import List, Dict
import numpy as np

def calculate_latency_stats(latencies: List[float]) -> Dict[str, float]:
    return {
        "mean": np.mean(latencies),
        "median": np.median(latencies),
        "p95": np.percentile(latencies, 95),
        "p99": np.percentile(latencies, 99),
        "min": np.min(latencies),
        "max": np.max(latencies)
    }

def measure_latency(api_results: Dict, local_results: Dict) -> Dict[str, Dict[str, Dict[str, float]]]:
    latency_results = {}
    
    for model_type in ["transcription", "translation", "sentiment"]:
        api_latencies = api_results[f"{model_type}_latencies"]
        local_latencies = local_results[f"{model_type}_latencies"]
        
        latency_results[model_type] = {
            "api": calculate_latency_stats(api_latencies),
            "local": calculate_latency_stats(local_latencies)
        }
    
    return latency_results