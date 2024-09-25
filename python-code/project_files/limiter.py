import asyncio
import time
import random
import logging
from collections import deque

logger = logging.getLogger(__name__)

class AdvancedRateLimiter:
    def __init__(self, max_calls, period, max_tokens=None):
        """
        max_calls: Maximum number of calls in the given period.
        period: Time period in seconds.
        max_tokens: Maximum burst tokens allowed.
        """
        self.max_calls = max_calls
        self.period = period
        self.max_tokens = max_tokens if max_tokens is not None else max_calls
        self.tokens = self.max_tokens
        self.lock = asyncio.Lock()
        self.last_check = time.time()
        self.queue = asyncio.Queue()

    async def acquire(self):
        await self.queue.put(None)
        async with self.lock:
            current_time = time.time()
            elapsed = current_time - self.last_check
            # Refill tokens
            refill_amount = (elapsed / self.period) * self.max_calls
            self.tokens = min(self.tokens + refill_amount, self.max_tokens)
            self.last_check = current_time

            if self.tokens >= 1:
                self.tokens -= 1
                self.queue.get_nowait()
                return
            else:
                # Calculate sleep time
                sleep_time = self.period / self.max_calls
                await asyncio.sleep(sleep_time)
                self.queue.get_nowait()
                await self.acquire()

    async def api_call_with_backoff(rate_limiter, func, *args, **kwargs):
        max_retries = 5
        base_delay = 1  # Initial delay in seconds

        for attempt in range(max_retries):
            await rate_limiter.acquire()
            try:
                return await func(*args, **kwargs)
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    # Extract 'Retry-After' header if available
                    retry_after = e.response.headers.get('Retry-After')
                    if retry_after:
                        delay = float(retry_after)
                    else:
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"Rate limit exceeded. Retrying in {delay:.2f} seconds.")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"API call failed: {str(e)}")
                    raise
        raise Exception("Max retries exceeded for API call")

