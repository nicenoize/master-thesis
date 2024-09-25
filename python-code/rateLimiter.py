import asyncio
import time
import random
import logging
from collections import deque

logger = logging.getLogger(__name__)

class AdvancedRateLimiter:
    def __init__(self, rpm, daily_limit, burst_limit):
        self.rpm = rpm
        self.daily_limit = daily_limit
        self.burst_limit = burst_limit
        self.interval = 60 / rpm
        self.tokens = burst_limit
        self.last_refill = time.time()
        self.queue = asyncio.Queue()
        self.requests = deque()
        self.lock = asyncio.Lock()
        self.daily_requests = 0
        self.daily_reset_time = time.time() + 86400  # 24 hours from now

    async def add(self):
        await self.queue.put(None)

    async def acquire(self):
        await self.add()
        async with self.lock:
            while True:
                await self.queue.get()
                self.refill()
                if self.tokens > 0 and self.daily_requests < self.daily_limit:
                    self.tokens -= 1
                    self.requests.append(time.time())
                    self.daily_requests += 1
                    return
                else:
                    delay = max(self.interval, (self.daily_reset_time - time.time()) / 2)
                    await asyncio.sleep(delay)

    def refill(self):
        now = time.time()
        if now >= self.daily_reset_time:
            self.daily_requests = 0
            self.daily_reset_time = now + 86400

        elapsed = now - self.last_refill
        new_tokens = int(elapsed / self.interval)
        if new_tokens > 0:
            self.tokens = min(self.tokens + new_tokens, self.burst_limit)
            self.last_refill = now

        while self.requests and now - self.requests[0] > 60:
            self.requests.popleft()

    async def wait_for_capacity(self):
        while len(self.requests) >= self.rpm or self.daily_requests >= self.daily_limit:
            await asyncio.sleep(0.1)

# Initialize rate limiters with appropriate limits
gpt_rate_limiter = AdvancedRateLimiter(rpm=500, daily_limit=30000, burst_limit=500)
whisper_rate_limiter = AdvancedRateLimiter(rpm=50, daily_limit=1000, burst_limit=50)

async def api_call_with_backoff(func, *args, **kwargs):
    max_retries = 10
    base_delay = 1
    for attempt in range(max_retries):
        await gpt_rate_limiter.wait_for_capacity()
        await gpt_rate_limiter.acquire()
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            if "429 Too Many Requests" in str(e):
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"GPT rate limit exceeded. Retrying in {delay:.2f} seconds.")
                await asyncio.sleep(delay)
            else:
                logger.error(f"GPT API call failed: {str(e)}")
                raise
    raise Exception("Max retries exceeded for GPT API")

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
                logger.warning(f"Whisper rate limit exceeded. Retrying in {delay:.2f} seconds.")
                await asyncio.sleep(delay)
            else:
                logger.error(f"Whisper API call failed: {str(e)}")
                raise
    raise Exception("Max retries exceeded for Whisper API")
