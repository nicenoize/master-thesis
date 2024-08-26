import asyncio
import time
from typing import Callable, List

async def simulate_request(func: Callable, *args):
    start_time = time.time()
    result = await func(*args)
    end_time = time.time()
    return result, end_time - start_time

async def simulate_concurrent_requests(func: Callable, num_requests: int, *args) -> List[tuple]:
    tasks = [simulate_request(func, *args) for _ in range(num_requests)]
    return await asyncio.gather(*tasks)

async def load_test(func: Callable, duration: int, max_concurrent: int, *args) -> List[tuple]:
    start_time = time.time()
    results = []
    
    while time.time() - start_time < duration:
        batch_size = min(max_concurrent, int((duration - (time.time() - start_time)) * max_concurrent / duration))
        batch_results = await simulate_concurrent_requests(func, batch_size, *args)
        results.extend(batch_results)
    
    return results

# Example usage:
# async def dummy_request():
#     await asyncio.sleep(0.1)
#     return "Result"
# 
# results = await load_test(dummy_request, duration=10, max_concurrent=100)