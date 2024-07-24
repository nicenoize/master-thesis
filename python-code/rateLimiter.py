import asyncio
import time
from collections import deque

class ImprovedRateLimiter:
    def __init__(self, rpm, burst_limit):
        self.rpm = rpm
        self.burst_limit = burst_limit
        self.interval = 60 / rpm
        self.tokens = burst_limit
        self.last_refill = time.time()
        self.queue = asyncio.Queue()
        self.requests = deque()

    async def add(self):
        await self.queue.put(None)

    async def acquire(self):
        await self.add()
        while True:
            await self.queue.get()
            self.refill()
            if self.tokens > 0:
                self.tokens -= 1
                self.requests.append(time.time())
                return
            else:
                await asyncio.sleep(self.interval)

    def refill(self):
        now = time.time()
        elapsed = now - self.last_refill
        new_tokens = int(elapsed / self.interval)
        if new_tokens > 0:
            self.tokens = min(self.tokens + new_tokens, self.burst_limit)
            self.last_refill = now

        while self.requests and now - self.requests[0] > 60:
            self.requests.popleft()

    async def wait_for_capacity(self):
        while len(self.requests) >= self.rpm:
            await asyncio.sleep(0.1)

# Usage
rate_limiter = ImprovedRateLimiter(rpm=500, burst_limit=500)
whisper_rate_limiter = ImprovedRateLimiter(rpm=50, burst_limit=50)

async def api_call_with_backoff(func, *args, **kwargs):
    max_retries = 10
    base_delay = 1
    for attempt in range(max_retries):
        await rate_limiter.wait_for_capacity()
        await rate_limiter.acquire()
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            if "429 Too Many Requests" in str(e):
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"Rate limit exceeded. Retrying in {delay:.2f} seconds.")
                await asyncio.sleep(delay)
            else:
                raise
    raise Exception("Max retries exceeded")

async def api_call_with_backoff_whisper(func, *args, **kwargs):
    max_retries = 10
    base_delay = 1
    for attempt in range(max_retries):
        await whisper_rate_limiter.wait_for_capacity()
        await whisper_rate_limiter.acquire()
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            if "429 Too Many Requests" in str(e):
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"Rate limit exceeded. Retrying in {delay:.2f} seconds.")
                await asyncio.sleep(delay)
            else:
                raise
    raise Exception("Max retries exceeded")