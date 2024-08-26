from typing import Dict

def calculate_api_cost(api_results: Dict, config) -> Dict[str, float]:
    cost_per_request = {
        "transcription": config.TRANSCRIPTION_API_COST,
        "translation": config.TRANSLATION_API_COST,
        "sentiment": config.SENTIMENT_API_COST
    }
    
    return {
        model_type: len(api_results[f"{model_type}_results"]) * cost_per_request[model_type]
        for model_type in ["transcription", "translation", "sentiment"]
    }

def calculate_local_cost(local_results: Dict, config) -> Dict[str, float]:
    # This is a simplified cost calculation. In reality, you'd need to factor in
    # electricity costs, hardware depreciation, maintenance, etc.
    total_time = sum(
        sum(local_results[f"{model_type}_latencies"])
        for model_type in ["transcription", "translation", "sentiment"]
    )
    
    # Assuming cost is primarily from GPU usage
    gpu_cost_per_hour = config.GPU_COST_PER_HOUR
    total_cost = (total_time / 3600) * gpu_cost_per_hour  # Convert seconds to hours
    
    # Distribute cost proportionally to each model type
    return {
        model_type: total_cost * (sum(local_results[f"{model_type}_latencies"]) / total_time)
        for model_type in ["transcription", "translation", "sentiment"]
    }

def calculate_cost_effectiveness(api_results: Dict, local_results: Dict, config) -> Dict[str, Dict[str, float]]:
    api_costs = calculate_api_cost(api_results, config)
    local_costs = calculate_local_cost(local_results, config)
    
    cost_effectiveness = {}
    for model_type in ["transcription", "translation", "sentiment"]:
        api_cost_per_request = api_costs[model_type] / len(api_results[f"{model_type}_results"])
        local_cost_per_request = local_costs[model_type] / len(local_results[f"{model_type}_results"])
        
        cost_effectiveness[model_type] = {
            "api_cost_per_request": api_cost_per_request,
            "local_cost_per_request": local_cost_per_request,
            "cost_ratio": api_cost_per_request / local_cost_per_request
        }
    
    return cost_effectiveness