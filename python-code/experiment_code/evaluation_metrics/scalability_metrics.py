from typing import Dict
from utils.concurrency_simulator import load_test

async def test_scalability(config, api_experiment, local_experiment) -> Dict[str, Dict[str, Dict[str, float]]]:
    scalability_results = {}
    
    for model_type in ["transcription", "translation", "sentiment"]:
        api_func = getattr(api_experiment, f"process_{model_type}")
        local_func = getattr(local_experiment, f"process_{model_type}")
        
        api_results = await load_test(api_func, config.LOAD_TEST_DURATION, config.MAX_CONCURRENT_REQUESTS)
        local_results = await load_test(local_func, config.LOAD_TEST_DURATION, config.MAX_CONCURRENT_REQUESTS)
        
        api_throughput = len(api_results) / config.LOAD_TEST_DURATION
        local_throughput = len(local_results) / config.LOAD_TEST_DURATION
        
        api_latencies = [result[1] for result in api_results]
        local_latencies = [result[1] for result in local_results]
        
        scalability_results[model_type] = {
            "api": {
                "throughput": api_throughput,
                "avg_latency": sum(api_latencies) / len(api_latencies)
            },
            "local": {
                "throughput": local_throughput,
                "avg_latency": sum(local_latencies) / len(local_latencies)
            }
        }
    
    return scalability_results